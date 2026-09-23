"""M10 子代理编排 —— 单个拉起（P3）+ 并行扇出与差异汇总（P5）。

单个子代理：**独立上下文 + 裁剪视图 + 结构化回传**。
第一个也是唯一的常驻用例是 M8.2 Memory Agent 的完整模式 —— 它约束最全：
专属工具子集、受限权限、定死的输出 schema、无状态。先把它跑通，其余都是简化版。

子代理和主 Loop 用的是**同一个 LoopRuntime**，差别只在四处：
  · 注册表是全局的一个子集视图（ScopedRegistry）—— 只看得到、只调得了被允许的工具
  · 权限闸门按 allowed 配，**L-external 永远不给**，也没有询问入口（子代理背后没有人）
  · 上下文独立：自己的滑窗、自己的系统提示词，不继承主对话历史，跑完即弃
  · 输出按 output_schema 校验，主 Agent 只收到结构化结论，不收过程

无状态是硬规则（ARCHITECTURE M8.2 三条硬规则之一）：状态全在 Store 里，
否则会出现「代理记得但库里没有」的幽灵状态，无法审计。stateless=False 只留给
将来明确需要多轮的任务型子代理，默认不用。

并行扇出（run_many）：同一个定义、多个任务并发跑，信号量限并发。
汇总时**保留每个候选的差异点**（各自独有的词），供人对比选择 —— 不替人挑。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from ...harness.context.assembler import ContextAssembler
from ...harness.context.window import ShortTermMemory, WindowPolicy
from ...harness.events.bus import EventBus, EventType
from ...harness.execution.loop import LoopRuntime, StopReason
from ...harness.model.budget import CostGuard
from ...harness.permission.gate import Decision, PermissionGate
from ...harness.tools.dispatcher import ToolDispatcher
from ...harness.tools.provider import PermissionLevel
from ...harness.tools.registry import ToolRegistry


class SubAgentDef(BaseModel):
    """子代理定义的最小契约（ARCHITECTURE M10）。"""

    name: str
    description: str = ""  # 决定主 Agent 何时调用它
    system_prompt: str
    tools: list[str] = Field(default_factory=list)  # 从全局注册表裁剪出的子集
    # 允许的副作用等级。默认只读；Memory Agent 是 READ + 仅限记忆库的 WRITE。
    # EXTERNAL 写了也会被剥掉 —— 子代理背后没有人，不可逆动作没人确认。
    allowed: list[PermissionLevel] = Field(default_factory=lambda: [PermissionLevel.READ])
    output_schema: dict[str, Any] = Field(default_factory=dict)  # 结论回传格式，必须结构化
    role: str = "subagent"  # 引用 models.yaml 的角色名，不是模型 id
    stateless: bool = True
    max_iterations: int = 8
    max_cost: float | None = None

    def policy(self) -> dict[PermissionLevel, Decision]:
        allowed = set(self.allowed) - {PermissionLevel.EXTERNAL}
        return {
            lvl: (Decision.ALLOW if lvl in allowed else Decision.DENY) for lvl in PermissionLevel
        }


class SubAgentResult(BaseModel):
    ok: bool
    data: Any = None  # 按 output_schema 解析出的结构；没有 schema 时是原文
    text: str = ""
    error: str | None = None
    iterations: int = 0
    cost: float | None = None
    stop_reason: str = ""
    duration_ms: int = 0


class FanOutSummary(BaseModel):
    """并行扇出的汇总：每个候选各自的结果 + 各自独有的差异点。"""

    total: int
    ok: int
    failed: int
    cost: float | None = None
    duration_ms: int = 0
    results: list[SubAgentResult] = Field(default_factory=list)
    diffs: list[list[str]] = Field(default_factory=list)

    def render(self) -> str:
        head = f"并行 {self.total} 个 · 成功 {self.ok} · 失败 {self.failed} · {self.duration_ms} ms"
        if self.cost is not None:
            head += f" · ¥{self.cost:.4f}"
        lines = [head]
        for i, (r, diff) in enumerate(zip(self.results, self.diffs, strict=True), 1):
            if not r.ok:
                lines.append(f"{i}. ✗ {r.error}")
                continue
            body = r.text.strip().replace("\n", " ")
            lines.append(f"{i}. {body[:60]}{'…' if len(body) > 60 else ''}")
            lines.append(f"   独有：{', '.join(diff) or '—'}")
        return "\n".join(lines)


# ---------------------------------------------------------------- 输出解析


def parse_json(text: str) -> Any | None:
    """从模型输出里剥出一个 JSON 对象或数组。容忍 ```json 围栏与前后解释文字。"""
    raw = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if m:
        raw = m.group(1).strip()
    starts = [i for i in (raw.find("{"), raw.find("[")) if i >= 0]
    if not starts:
        return None
    i = min(starts)
    closer = "}" if raw[i] == "{" else "]"
    j = raw.rfind(closer)
    if j <= i:
        return None
    try:
        return json.loads(raw[i : j + 1])
    except json.JSONDecodeError:
        return None


_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
}


def validate_output(data: Any, schema: dict[str, Any]) -> str:
    """轻量校验：顶层类型、必填字段、字段类型。返回错误说明，空串 = 通过。

    不上 jsonschema 库 —— 这里要的是"主 Agent 能不能安全地读这份结论"，
    三条检查够用；校验太严反而会把措辞略有出入的有效结论拒掉。
    """
    if not schema:
        return ""
    top = schema.get("type")
    if top in _TYPES and not isinstance(data, _TYPES[top]):
        return f"输出顶层应为 {top}，实际是 {type(data).__name__}"
    if isinstance(data, dict):
        missing = [k for k in schema.get("required", []) if k not in data]
        if missing:
            return f"缺少必填字段：{', '.join(missing)}"
        for key, spec in (schema.get("properties") or {}).items():
            want = spec.get("type") if isinstance(spec, dict) else None
            if key in data and want in _TYPES and not isinstance(data[key], _TYPES[want]):
                return f"字段 {key} 应为 {want}，实际是 {type(data[key]).__name__}"
    return ""


# ---------------------------------------------------------------- 差异点

_SPLIT = re.compile(r"[\s，。！？、；：,.!?;:\n\"'“”‘’（）()【】\[\]《》—\-…]+")


def fragments(text: str) -> list[str]:
    """按标点切段再按长度过滤。不分词、不调模型 —— 差异点只是给人对比时的路标。"""
    return [p for p in _SPLIT.split(text or "") if 2 <= len(p) <= 10]


def distinctive_terms(texts: list[str], top: int = 5) -> list[list[str]]:
    """每个候选**只有它有**的片段，按出现次数与长度排。空文本给空表。"""
    counters = [Counter(fragments(t)) for t in texts]
    out: list[list[str]] = []
    for i, c in enumerate(counters):
        others: set[str] = set()
        for j, other in enumerate(counters):
            if j != i:
                others |= set(other)
        unique = [(term, n) for term, n in c.items() if term not in others]
        unique.sort(key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
        out.append([t for t, _ in unique[:top]])
    return out


# ---------------------------------------------------------------- 运行器


class SubAgentRunner:
    """拉起一个子代理跑完一件事，回传结构化结论。"""

    def __init__(
        self,
        gateway: Any,
        registry: ToolRegistry,
        bus: EventBus,
        guard: CostGuard | None = None,
    ) -> None:
        self.gateway = gateway
        self.registry = registry
        self.bus = bus
        self.guard = guard
        self._memories: dict[str, ShortTermMemory] = {}  # 仅 stateless=False 的用

    def _system(self, defn: SubAgentDef) -> str:
        parts = [defn.system_prompt.strip()]
        if defn.output_schema:
            parts.append(
                "## 输出契约\n只输出一个 JSON，不要任何解释文字。结构：\n"
                + json.dumps(defn.output_schema, ensure_ascii=False)
            )
        return "\n\n".join(parts)

    def _memory(self, defn: SubAgentDef) -> ShortTermMemory:
        if defn.stateless:
            return ShortTermMemory(policy=WindowPolicy(window_turns=6, evict_at=8))
        if defn.name not in self._memories:
            self._memories[defn.name] = ShortTermMemory(
                policy=WindowPolicy(window_turns=6, evict_at=8)
            )
        return self._memories[defn.name]

    async def run(self, defn: SubAgentDef, task: str, context: str = "") -> SubAgentResult:
        started = time.perf_counter()
        scoped = self.registry.scoped(defn.tools)
        # 子代理背后没有人：没有询问入口，超预算或 L-external 一律拒
        gate = PermissionGate(self.bus, policy=defn.policy(), asker=None, guard=self.guard)
        scoped.gate = gate
        loop = LoopRuntime(
            gateway=self.gateway,
            registry=scoped,  # type: ignore[arg-type]
            dispatcher=ToolDispatcher(scoped, gate, self.bus),  # type: ignore[arg-type]
            assembler=ContextAssembler(self.bus, system_prompt=self._system(defn)),
            memory=self._memory(defn),
            bus=self.bus,
            role=defn.role,
            max_iterations=defn.max_iterations,
            max_cost=defn.max_cost,
            guard=self.guard,
        )
        await self.bus.emit(
            EventType.SUBAGENT_START, name=defn.name, role=defn.role, tools=list(defn.tools)
        )

        prompt = task if not context else f"{task}\n\n## 上下文\n{context}"
        try:
            r = await loop.run_turn(prompt)
        except Exception as e:  # noqa: BLE001 — 子代理失败不该让主链路跟着崩
            result = SubAgentResult(ok=False, error=f"{type(e).__name__}: {e}")
        else:
            result = self._collect(defn, r.text, r.iterations, r.cost, r.stop_reason)

        result.duration_ms = int((time.perf_counter() - started) * 1000)
        await self.bus.emit(
            EventType.SUBAGENT_END,
            name=defn.name,
            ok=result.ok,
            iterations=result.iterations,
            cost=result.cost,
            error=result.error,
            duration_ms=result.duration_ms,
        )
        return result

    async def run_many(
        self,
        defn: SubAgentDef,
        tasks: list[str],
        context: str = "",
        concurrency: int = 4,
    ) -> FanOutSummary:
        """并行扇出：同一个定义、多个任务并发跑，信号量限并发。

        结果按任务顺序回传，单个失败不影响其余（run() 内部已经兜住异常）。
        每个候选都是独立的无状态子代理 —— 互相看不见对方的输出，差异才是真差异。
        """
        started = time.perf_counter()
        sem = asyncio.Semaphore(max(1, int(concurrency)))

        async def one(task: str) -> SubAgentResult:
            async with sem:
                return await self.run(defn, task, context)

        await self.bus.emit(
            EventType.FANOUT_START, name=defn.name, tasks=len(tasks), concurrency=concurrency
        )
        results = list(await asyncio.gather(*(one(t) for t in tasks)))
        texts: list[str] = []
        for r in results:
            if not r.ok:
                texts.append("")
            elif isinstance(r.data, str) or r.data is None:
                texts.append(r.text)
            else:
                texts.append(json.dumps(r.data, ensure_ascii=False))
        costs = [r.cost for r in results if r.cost is not None]
        summary = FanOutSummary(
            total=len(results),
            ok=sum(1 for r in results if r.ok),
            failed=sum(1 for r in results if not r.ok),
            cost=sum(costs) if costs else None,
            duration_ms=int((time.perf_counter() - started) * 1000),
            results=results,
            diffs=distinctive_terms(texts),
        )
        await self.bus.emit(
            EventType.FANOUT_END,
            name=defn.name,
            total=summary.total,
            ok=summary.ok,
            failed=summary.failed,
            cost=summary.cost,
            duration_ms=summary.duration_ms,
        )
        return summary

    @staticmethod
    def _collect(
        defn: SubAgentDef, text: str, iterations: int, cost: float | None, stop: StopReason
    ) -> SubAgentResult:
        base = SubAgentResult(
            ok=True, text=text, iterations=iterations, cost=cost, stop_reason=stop.value
        )
        if stop is StopReason.ERROR:
            return base.model_copy(update={"ok": False, "error": text})
        if stop is StopReason.AWAITING_REVIEW:
            return base.model_copy(
                update={
                    "ok": False,
                    "error": "子代理不能请人审 —— 它背后没有人。别给它 request_review 这类工具。",
                }
            )
        if stop is StopReason.BUDGET_EXCEEDED:
            return base.model_copy(update={"ok": False, "error": "预算护栏触发：" + text})
        if not defn.output_schema:
            return base.model_copy(update={"data": text})
        data = parse_json(text)
        if data is None:
            return base.model_copy(update={"ok": False, "error": "子代理没有返回合法 JSON"})
        err = validate_output(data, defn.output_schema)
        if err:
            return base.model_copy(update={"ok": False, "data": data, "error": err})
        return base.model_copy(update={"data": data})

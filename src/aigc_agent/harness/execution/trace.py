"""M1 执行痕迹 —— 把 agentic 模式下的调用序列还原成 DAG。

这是「**图从输入变成输出**」（§4.1）真正兑现的地方。

没有预定义流程，但每次 function 调用消费和产出哪些资产是确定的，
按资产依赖连边，就得到一张和图模式等价的 DAG —— 只不过它是**跑完才有**的，
不是**跑之前画好**的。

有了它，agentic 模式补齐图独占的两样：
  · 进度可见 —— 实时渲染，人知道走到哪了
  · 单步定位/回退 —— 哪次调用产出了哪份资产、花了多少、之后哪些依赖它

实现方式和 M8 的打回落库一致：**订阅事件总线**。
Trace 不侵入 Loop，Loop 也不知道有 Trace 这回事。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..events.bus import Event, EventBus, EventType

# 资产 id 形如 as_ + 10 位 hex。用正则从任意入参里捞出来，
# 这样将来加新 function 不用注册「哪个参数是资产」。
ASSET_RE = re.compile(r"\bas_[0-9a-f]{10}\b")


def extract_assets(value: Any) -> list[str]:
    """从任意结构里递归捞出资产 id，保序去重。"""
    found: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, str):
            found.extend(ASSET_RE.findall(v))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(value)
    seen: set[str] = set()
    return [a for a in found if not (a in seen or seen.add(a))]


@dataclass
class TraceNode:
    id: str  # tool_call_id
    tool: str
    kind: str = "call"  # call | checkpoint
    turn: int = 0
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    ok: bool = True
    error: str | None = None
    duration_ms: int = 0
    # checkpoint 专用
    decision: str | None = None
    reason: str = ""
    stage: str = ""
    # 被后续回退作废 —— 保留在痕迹里但标记出来，复盘时看得见走过的弯路
    superseded: bool = False


@dataclass
class TraceEdge:
    src: str
    dst: str
    asset: str | None = None  # None 表示顺序边（无数据依赖）


class ExecutionTrace:
    """订阅事件总线，增量构建 DAG。"""

    def __init__(self) -> None:
        self.nodes: list[TraceNode] = []
        self._by_id: dict[str, TraceNode] = {}
        self._producer: dict[str, str] = {}  # asset -> 产出它的 call_id
        self.edges: list[TraceEdge] = []
        self._pending_args: dict[str, dict[str, Any]] = {}
        self._turn = 0

    def attach(self, bus: EventBus) -> None:
        bus.subscribe(self._on_event)

    # ---------- 事件处理 ----------

    def _on_event(self, ev: Event) -> None:
        d = ev.data
        if ev.type is EventType.LOOP_START:
            self._turn = d.get("turn", self._turn)

        elif ev.type is EventType.TOOL_CALL:
            self._pending_args[d.get("call_id", "")] = d.get("args") or {}

        elif ev.type in (EventType.TOOL_RESULT, EventType.TOOL_ERROR):
            self._add_call(d)

        elif ev.type is EventType.CHECKPOINT_DECIDED:
            self._decide(d)

    def _add_call(self, d: dict[str, Any]) -> None:
        cid = d.get("call_id") or f"call_{len(self.nodes)}"
        args = d.get("args") or self._pending_args.pop(cid, {})

        inputs = extract_assets(args)
        outputs = [d["asset_ref"]] if d.get("asset_ref") else []

        node = TraceNode(
            id=cid,
            tool=d.get("tool", "?"),
            kind="checkpoint" if d.get("suspend") else "call",
            turn=self._turn,
            inputs=inputs,
            outputs=outputs,
            ok=bool(d.get("ok", True)),
            error=d.get("error"),
            duration_ms=d.get("duration_ms", 0),
        )
        if node.kind == "checkpoint":
            node.stage = str(args.get("stage", ""))

        self.nodes.append(node)
        self._by_id[cid] = node

        # 数据依赖边：谁产出了我消费的资产，就连过来
        for a in inputs:
            src = self._producer.get(a)
            if src and src != cid:
                self.edges.append(TraceEdge(src=src, dst=cid, asset=a))

        for a in outputs:
            self._producer.setdefault(a, cid)

    def _decide(self, d: dict[str, Any]) -> None:
        """把决策贴到最近一个还没被决策的 checkpoint 上。"""
        for node in reversed(self.nodes):
            if node.kind == "checkpoint" and node.decision is None:
                node.decision = d.get("decision")
                node.reason = d.get("reason", "")
                return

    # ---------- 查询 ----------

    def node(self, call_id: str) -> TraceNode | None:
        return self._by_id.get(call_id)

    def producer_of(self, asset: str) -> TraceNode | None:
        cid = self._producer.get(asset)
        return self._by_id.get(cid) if cid else None

    def descendants(self, call_id: str) -> list[TraceNode]:
        """所有（直接或间接）依赖该调用产物的后续调用。"""
        out: list[TraceNode] = []
        frontier = {call_id}
        seen = {call_id}
        while frontier:
            nxt: set[str] = set()
            for e in self.edges:
                if e.src in frontier and e.dst not in seen:
                    seen.add(e.dst)
                    nxt.add(e.dst)
                    n = self._by_id.get(e.dst)
                    if n:
                        out.append(n)
            frontier = nxt
        return out

    def rollback_plan(self, asset: str) -> dict[str, Any]:
        """要回到某份资产的状态，需要作废哪些调用。

        agentic 模式下的「单步重跑」：不是回到图上某个节点，而是
        **选一份资产版本作为新的出发点**，它之后的一切标记作废。
        痕迹本身不删 —— 走过的弯路留着，复盘和记忆提炼都要用。
        """
        producer = self.producer_of(asset)
        if producer is None:
            return {"ok": False, "error": f"没有调用产出过 {asset}"}

        doomed = self.descendants(producer.id)
        return {
            "ok": True,
            "restart_from": producer.id,
            "keep_asset": asset,
            "supersede": [n.id for n in doomed],
            "lost_assets": [a for n in doomed for a in n.outputs],
            "cost_wasted_ms": sum(n.duration_ms for n in doomed),
        }

    def apply_rollback(self, asset: str) -> dict[str, Any]:
        plan = self.rollback_plan(asset)
        if plan["ok"]:
            for cid in plan["supersede"]:
                if cid in self._by_id:
                    self._by_id[cid].superseded = True
        return plan

    # ---------- 渲染 ----------

    def to_text(self) -> str:
        """CLI 用的紧凑视图。"""
        if not self.nodes:
            return "（还没有任何调用）"
        lines: list[str] = []
        for n in self.nodes:
            mark = "✗" if not n.ok else ("⏸" if n.kind == "checkpoint" else "·")
            if n.superseded:
                mark = "✘"
            line = f"{mark} {n.tool}"
            if n.inputs:
                line += f"  ←{','.join(n.inputs)}"
            if n.outputs:
                line += f"  →{','.join(n.outputs)}"
            if n.duration_ms:
                line += f"  [{n.duration_ms}ms]"
            if n.kind == "checkpoint":
                verdict = n.decision or "等待中"
                line += f"  人审({n.stage or '—'})：{verdict}"
                if n.reason:
                    line += f" — {n.reason}"
            if n.error:
                line += f"  错误：{n.error}"
            lines.append(line)
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """跑完自动得到的流程图 —— 这就是「图作为输出」。"""
        out = ["flowchart TD"]
        for n in self.nodes:
            label = n.tool
            if n.kind == "checkpoint":
                label = f"人审 {n.stage or ''}".strip()
                if n.decision:
                    label += f"<br/>{n.decision}"
                shape = f'{{{{"{label}"}}}}'  # 菱形
            elif not n.ok:
                shape = f'["{label} ✗"]'
            else:
                shape = f'["{label}"]'
            out.append(f"    {n.id}{shape}")
            if n.superseded:
                out.append(f"    class {n.id} dropped")

        for e in self.edges:
            arrow = f"-- {e.asset} -->" if e.asset else "-->"
            out.append(f"    {e.src} {arrow} {e.dst}")

        out.append("    classDef dropped stroke-dasharray: 4 4,opacity:0.45")
        return "\n".join(out)

    def summary(self) -> dict[str, Any]:
        return {
            "calls": len([n for n in self.nodes if n.kind == "call"]),
            "checkpoints": len([n for n in self.nodes if n.kind == "checkpoint"]),
            "failed": len([n for n in self.nodes if not n.ok]),
            "superseded": len([n for n in self.nodes if n.superseded]),
            "assets_produced": len(self._producer),
            "total_ms": sum(n.duration_ms for n in self.nodes),
        }

"""M1 Loop Runtime —— 自主循环。

组装请求 → 调模型 → 解析工具调用 → 并发执行 → 结果回填 → 判定继续/停机。

注意这是**主对话 Loop**。系统里还有第二个 Loop：Graph 的 agent 节点内部
那个（P1 加）。两者的区别见 ARCHITECTURE.md M1「系统里有两个 Loop」：
  · 主对话 Loop —— 跨会话，10 轮滑窗在这里，用户可见
  · 节点内 Loop —— 节点结束即销毁，不占用户对话轮次，用 max_iterations 护栏

2026-09-17 复盘后加的四样：
  · **预检**：每次调模型前按校准估算看装不装得下；装不下先剔最老的历史轮
    （整轮交给记忆提取），还不行就让装配层用应急档折叠当前轮。撞上模型
    上下文上限（401/400）压缩后重试一次，不再原样重发。
  · **错误分类**：额度用尽 / 超时 / 上下文超限 各给一句人能看懂的提示，
    不再只有一行异常名。
  · **金额护栏有出口**：撞上单任务/单日/单项目金额上限时，有询问器就问一次
    人追加多少；没询问器照旧停下，但停机文案里写清楚出口。
  · **一次迭代只挂一个人审**：模型同一次迭代里发多个 request_review，
    只有第一个能挂起；其余之前被当成功回填，模型以为都提交了 —— 现在按失败回填。
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..context.assembler import ContextAssembler
from ..context.window import ShortTermMemory, Turn
from ..events.bus import EventBus, EventType
from ..model.budget import CostGuard
from ..model.gateway import ModelGateway, ModelResponse, classify_model_error, estimate_tokens
from ..tools.dispatcher import ToolDispatcher
from ..tools.provider import ToolResult
from ..tools.registry import ToolRegistry

# 金额护栏询问器：收到超限原因，返回临时追加的金额（元）；None 或 <= 0 = 停下交给人。
BudgetAsker = Callable[[str], Awaitable[float | None]]

# finish_reason 的两类：被截断 / 被内容安全过滤（各家叫法不一）
_TRUNCATED = {"length", "max_tokens"}
_FILTERED = {"content_filter", "content-filter", "sensitive", "safety", "blocked"}

_EXTRA_REVIEW = (
    "[工具执行失败] 一次迭代里只能提一次人审，本次已有一条在等人决定；"
    "这条请求被忽略，等上一条决策回来后再提。"
)


class StopReason(StrEnum):
    NO_TOOL_CALLS = "no_tool_calls"  # 正常完成
    AWAITING_REVIEW = "awaiting_review"  # 模型主动请求人审，等 resume
    MAX_ITERATIONS = "max_iterations"
    BUDGET_EXCEEDED = "budget_exceeded"
    INTERRUPTED = "interrupted"
    ERROR = "error"
    # 模型服务商的内容安全过滤拦下了这次回复（finish_reason=content_filter）。
    # 之前当成正常完成，空回复就那么过去了（2026-09-23 审查）
    CONTENT_FILTER = "content_filter"


@dataclass
class LoopResult:
    text: str
    turn: Turn
    iterations: int
    stop_reason: StopReason
    cost: float | None = None
    # 停机是 error 时的类别（connection / timeout / quota / context_overflow / other）。
    # 网络类的可以在**同一轮**里接着跑（continue_turn），本轮做过的活不用重来。
    error_kind: str = ""
    # 本轮失败的工具调用（「工具名：原因」）。CLI 直接列给人看 —— 不管模型最后怎么说，
    # 人都能看到哪几步没成（真实日志里有工具报错后模型仍说「已完成」的）
    tool_failures: list[str] = field(default_factory=list)

    @property
    def resumable(self) -> bool:
        """这次失败能不能原地接着跑（不用人重新发话、不丢本轮进度）。"""
        return self.stop_reason is StopReason.ERROR and self.error_kind in ("connection", "timeout")


class LoopRuntime:
    def __init__(
        self,
        gateway: ModelGateway,
        registry: ToolRegistry,
        dispatcher: ToolDispatcher,
        assembler: ContextAssembler,
        memory: ShortTermMemory,
        bus: EventBus,
        role: str = "main_agent",
        max_iterations: int = 50,
        max_cost: float | None = None,
        guard: CostGuard | None = None,
        major_stages: tuple[str, ...] = ("剧本", "视频生成", "图片生成"),
        budget_asker: BudgetAsker | None = None,
        budget_hint: str = "",
    ) -> None:
        self.gateway = gateway
        self.registry = registry
        self.dispatcher = dispatcher
        self.assembler = assembler
        self.memory = memory
        self.bus = bus
        self.role = role
        self.max_iterations = max_iterations
        self.max_cost = max_cost
        # 任务级累计口径（含媒体调用），与 max_cost 的单轮文本口径并行。
        self.guard = guard
        # 挂起中的人审请求。非 None 时必须先 resume_turn() 才能继续。
        self.pending_review: dict[str, Any] | None = None
        # /auto 自动模式：小节点的 request_review 不挂起，按「已采纳」回填后继续跑；
        # 大节点（stage 命中 major_stages，如剧本/视频生成/图片生成）仍停下来问人一次。
        # 人随时能 Ctrl+C 打断。只影响人审 —— L-external 与预算超限照样问人。
        self.auto_review = False
        # 大节点环节关键词：/auto 下这些环节的人审不自动采纳，挂起等决策。
        self.major_stages = major_stages
        # 最近一次停机若是 error，是哪一类（connection / timeout / quota / …）。
        # 网络类的可以 continue_turn 原地续跑，不用人重发、不丢本轮进度。
        self.last_error_kind = ""
        # 金额护栏的询问器与停机提示（CLI 接终端，Web 接人审台）
        self.budget_asker = budget_asker
        self.budget_hint = budget_hint
        # 本轮（含人审前后、续跑）失败的工具调用，给 LoopResult.tool_failures
        self._failures: list[str] = []

    def _repair_interrupted_calls(self) -> int:
        """补全没有 tool 响应的 tool_calls。Ctrl+C 可能打断在「assistant 已回填、
        tool 结果还没回填」的中间态，不补的话下一轮组装的上下文 API 直接 400
        （"assistant message with 'tool_calls' must be followed by tool messages"）。
        补一句「已中断」让上下文重新合法。返回补了几条。"""
        answered: set[str] = set()
        dangling: list[tuple[Turn, str]] = []
        for t in self.memory.turns:
            for m in t.messages:
                if m.get("role") == "tool":
                    answered.add(m.get("tool_call_id", ""))
                elif m.get("role") == "assistant":
                    for c in m.get("tool_calls") or []:
                        dangling.append((t, c["id"]))
        n = 0
        take = getattr(self.dispatcher, "take_completed", None)
        for t, cid in dangling:
            if cid in answered:
                continue
            # 同一批里在中断前已经跑完的调用：回填真实结果（2026-09-23 审查：之前一律补
            # 「没有执行结果」，已付费生成的段被模型当成没做，又生成一遍）
            done = take(cid) if callable(take) else None
            if done is not None:
                content = done.to_message_content() + "\n（本轮被打断前这次调用已经完成）"
            else:
                content = (
                    "（这次调用被中断，没拿到执行结果。若是生图/生视频这类花钱的调用，"
                    "服务端可能已经在生成、已经计费 —— 重做之前先用 media_tasks / list_assets "
                    "核对，能取回就 media_recover 取回，别直接重新生成）"
                )
            t.messages.append({"role": "tool", "tool_call_id": cid, "content": content})
            n += 1
        return n

    # ---------- 装配与调用 ----------

    async def _assemble(
        self, turn: Turn, catalog: str, recalled: str = ""
    ) -> list[dict[str, Any]]:
        """装配 + 预检。

        估算（校准后）超预算的处理顺序：
          1. 剔最老的历史轮 —— 整轮、完整地交给记忆提取，只剩当前轮就不剔
          2. 还超：装配层切到应急档，把当前轮的旧迭代也折叠掉
        再不行就发出去让 API 报错，错误路径会再压缩一次重试。
        """
        asm = self.assembler
        messages = await asm.assemble(self.memory, turn, tool_catalog=catalog, recalled=recalled)
        while not asm.fits(messages) and len(self.memory.turns) > 1:
            evicted = self.memory.evict_oldest(1)
            if not evicted:
                break
            await self.bus.emit(
                EventType.WINDOW_EVICT,
                evicted_turns=[t.index for t in evicted],
                count=len(evicted),
                remaining=len(self.memory.turns),
                reason="token_budget",
            )
            if self.memory.on_evict is not None:
                self.memory.on_evict(evicted)
            messages = await asm.assemble(
                self.memory, turn, tool_catalog=catalog, recalled=recalled
            )
        if not asm.fits(messages) and not asm.shrink:
            asm.shrink = True
            await self.bus.emit(
                EventType.WARNING,
                message=f"上下文约 {asm.last_tokens:,} token 仍超预算，本轮切到应急压缩",
            )
            messages = await asm.assemble(
                self.memory, turn, tool_catalog=catalog, recalled=recalled
            )
        return messages

    async def _call_model(
        self,
        turn: Turn,
        tools: list[dict[str, Any]],
        catalog: str,
        recalled: str = "",
    ) -> tuple[ModelResponse | None, str]:
        """调一次模型。失败返回 (None, 给人看的错误文本)。

        撞上模型上下文上限时压缩后重试一次 —— 原样重发只会再撞一次。
        """
        for attempt in (1, 2):
            messages = await self._assemble(turn, catalog, recalled)
            try:
                resp = await self.gateway.chat(self.role, messages, tools=tools)
            except Exception as e:  # noqa: BLE001
                kind, hint = classify_model_error(e)
                if kind == "context_overflow" and attempt == 1:
                    self.assembler.shrink = True
                    await self.bus.emit(
                        EventType.WARNING,
                        message="请求超过模型上下文上限，已压缩当前轮后重试",
                    )
                    continue
                self.last_error_kind = kind
                await self.bus.emit(
                    EventType.LOOP_STOP_REASON, reason="error", detail=str(e), kind=kind
                )
                text = f"模型调用失败：{type(e).__name__}: {e}"
                if hint:
                    text += f"\n{hint}"
                return None, text
            self.assembler.observe(resp.usage.prompt_tokens)
            self.memory.observed_prompt_tokens = resp.usage.prompt_tokens
            return resp, ""
        return None, "模型调用失败"  # pragma: no cover

    # ---------- 主循环 ----------

    async def _iterate(
        self, turn: Turn, recalled: str = ""
    ) -> tuple[str, StopReason, float, int]:
        """在一个 turn 上跑迭代，直到停机。返回 (最终文本, 停机原因, 本段花费, 迭代数)。

        失败类别记在 self.last_error_kind，调用方据此判断能不能原地续跑。

        run_turn（新轮）和 _continue（人审结案/撞线续跑）共用这一段，
        人审前后仍算同一问一答。
        """
        tools = await self.registry.schemas_for_context()
        catalog = self.registry.catalog_digest()
        cost = 0.0
        i = 0
        self.last_error_kind = ""

        for i in range(1, self.max_iterations + 1):
            await self.bus.emit(EventType.ITERATION_START, turn=turn.index, iteration=i)

            resp, err = await self._call_model(turn, tools, catalog, recalled)
            if resp is None:
                return err, StopReason.ERROR, cost, i
            cost += resp.usage.cost or 0.0
            finish = (resp.finish_reason or "").strip().lower()

            # ---- 停机判定 0：被内容安全过滤拦下 ----
            # 在回填之前判：带着工具调用的半截回复不能进历史（没有工具结果就 400）
            if finish in _FILTERED:
                if resp.text:
                    turn.messages.append({"role": "assistant", "content": resp.text})
                await self.bus.emit(
                    EventType.LOOP_STOP_REASON, reason="content_filter", detail=finish
                )
                text = (resp.text + "\n\n" if resp.text else "") + (
                    "（这次回复被模型服务商的内容安全过滤拦下了。换个说法，"
                    "或者把涉及的内容调整后再试 —— 原样重发多半还会被拦）"
                )
                return text, StopReason.CONTENT_FILTER, cost, i

            # 助手消息回填（含工具调用）
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": resp.text or None}
            if resp.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                    }
                    for c in resp.tool_calls
                ]
            turn.messages.append(assistant_msg)

            # ---- 停机判定 1：没有工具调用 = 本轮完成 ----
            if not resp.wants_tools:
                text = resp.text
                if finish in _TRUNCATED:
                    text = (text or "") + "\n\n（回复到这里被截断了：超过模型单次输出上限）"
                return text, StopReason.NO_TOOL_CALLS, cost, i

            # ---- 执行工具（权限串行、执行并发）----
            results = await self.dispatcher.run(resp.tool_calls)
            suspended: tuple[str, ToolResult] | None = None
            for call, result in results:
                if result.suspend:
                    if suspended is None:
                        # 该工具要求停下来等人。先不回填它的结果，
                        # 等 resume_turn() 拿到人的决策再补上。
                        suspended = (call.id, result)
                        continue
                    # 同一次迭代里的第二个人审请求：不能同时挂两个，按失败回填。
                    # 之前被当成功回填，模型以为都提交了，实际没人会回答它。
                    turn.messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": _EXTRA_REVIEW}
                    )
                    continue
                content = result.to_message_content()
                if not result.ok:
                    self._failures.append(f"{call.name}：{(result.error or '')[:120]}")
                    if finish in _TRUNCATED and "不是合法 JSON" in (result.error or ""):
                        # 输出被截断 → 工具参数只写了一半。告诉模型真正的原因和出路，
                        # 否则它会原样重试到撞迭代上限（2026-09-23 审查）
                        content += (
                            "\n原因：这次输出超过了模型单次输出上限，参数被截断了。长内容别塞进"
                            "工具参数 —— 用 fs_write 分几块写到本地文件（第二块起 mode=append）"
                            "再 fs_import，或者传已经存好的资产 id"
                        )
                turn.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": content}
                )

            # ---- 停机判定 1.5：模型主动请求人审 ----
            # agentic 形态下 HITL 的落地方式：人审不是图上的固定节点，
            # 而是模型自己判断「这个该让人看一眼」时调用的一个 function。
            # /auto 开着时小节点不挂起：按采纳回填，循环继续，人随时 Ctrl+C 打断；
            # 大节点（剧本/视频生成/图片生成）照样挂起，等人拍板一次。
            if suspended is not None:
                call_id, result = suspended
                if self.auto_review and not self._is_major_review(result.suspend_payload):
                    await self._auto_adopt(turn, call_id, result)
                else:
                    self.pending_review = {
                        "call_id": call_id,
                        "turn_index": turn.index,
                        **result.suspend_payload,
                    }
                    await self.bus.emit(
                        EventType.CHECKPOINT_REACHED,
                        turn=turn.index,
                        call_id=call_id,
                        **result.suspend_payload,
                    )
                    return resp.text or result.content, StopReason.AWAITING_REVIEW, cost, i

            # ---- 停机判定 2：预算 ----
            # 两道口径：本轮文本花费（max_cost）与整个任务的累计（Cost Guard，
            # 含生图/生视频/TTS）。金额口径有询问器就问一次人追加多少，
            # 人点头就抬上限继续；否则**挂起交给人**，不静默降级。
            over, money = self._over_budget(cost)
            if over:
                if money and self.budget_asker is not None and self.guard is not None:
                    extra = await self.budget_asker(over)
                    if extra and extra > 0:
                        total = self.guard.allow_more_money(float(extra))
                        await self.bus.emit(
                            EventType.WARNING,
                            message=(
                                f"金额护栏：人已放行，临时追加 ¥{extra:.0f}"
                                f"（累计追加 ¥{total:.0f}）"
                            ),
                        )
                        continue
                await self.bus.emit(
                    EventType.LOOP_STOP_REASON,
                    reason="budget_exceeded",
                    cost=cost,
                    detail=over,
                )
                text = resp.text or f"预算护栏触发（{over}），暂停并交回给你决定是否继续。"
                if self.budget_hint:
                    text += "\n" + self.budget_hint
                return text, StopReason.BUDGET_EXCEEDED, cost, i

        await self.bus.emit(
            EventType.LOOP_STOP_REASON,
            reason="max_iterations",
            max_iterations=self.max_iterations,
        )
        return "已达最大工具调用轮次，停下来交回给你。", StopReason.MAX_ITERATIONS, cost, i

    async def run_turn(self, user_input: str, recalled: str = "") -> LoopResult:
        # 挂起中的人审必须先 resume_turn() 结案。直接开新轮会把上一轮
        # 缺 tool 响应的 assistant.tool_calls 发给模型，API 直接 400：
        # "assistant message with 'tool_calls' must be followed by tool messages"。
        if self.pending_review is not None:
            raise RuntimeError(
                "有挂起的人审未结案：先 resume_turn() 再开新轮，"
                "否则上下文里会留下没有 tool 响应的 tool_calls"
            )
        repaired = self._repair_interrupted_calls()
        if repaired:
            await self.bus.emit(
                EventType.WARNING,
                message=f"上一轮被中断：已补 {repaired} 条没有执行结果的调用",
            )
        turn = self.memory.new_turn()
        turn.messages.append({"role": "user", "content": user_input})
        self.assembler.shrink = False  # 应急压缩只管一轮
        self._failures = []

        await self.bus.emit(
            EventType.LOOP_START, turn=turn.index, role=self.role, input_preview=user_input[:200]
        )

        final_text, stop, turn_cost, i = await self._iterate(turn, recalled)
        turn.tokens = self.assembler.calibrated(estimate_tokens(turn.messages))

        # 挂起时这一轮还没走完，不做驱逐 —— 否则人审完回来上下文已经被剔了
        if stop is StopReason.AWAITING_REVIEW:
            await self.bus.emit(
                EventType.LOOP_END, turn=turn.index, iterations=i, stop_reason=stop.value
            )
            return LoopResult(
                text=final_text, turn=turn, iterations=i, stop_reason=stop, cost=turn_cost or None,
                tool_failures=list(self._failures),
            )

        # ---- 批量驱逐（涨到 evict_at 才一次性剔回 window_turns）----
        evicted = self.memory.maybe_evict()
        if evicted:
            await self.bus.emit(
                EventType.WINDOW_EVICT,
                evicted_turns=[t.index for t in evicted],
                count=len(evicted),
                remaining=len(self.memory.turns),
            )
            # P1：这里把 evicted 丢进 asyncio.Queue 交给 Memory Agent 提关键词。
            # 必须异步 —— 触发驱逐的那一轮恰好也是缓存击穿的那一轮，
            # 两个开销叠在同一轮上，用户会明显感觉卡顿。
            if self.memory.on_evict is not None:
                self.memory.on_evict(evicted)

        await self.bus.emit(
            EventType.LOOP_END,
            turn=turn.index,
            iterations=i,
            stop_reason=stop.value,
            cost=turn_cost or None,
            text_preview=(final_text or "")[:200],
        )
        return LoopResult(
            text=final_text,
            turn=turn,
            iterations=i,
            stop_reason=stop,
            cost=turn_cost or None,
            error_kind=self.last_error_kind,
            tool_failures=list(self._failures),
        )

    def _over_budget(self, turn_cost: float) -> tuple[str, bool]:
        """超了返回 (原因, 是否金额口径)，没超返回 ("", False)。"""
        if self.max_cost is not None and turn_cost >= self.max_cost:
            return f"本轮已花 ¥{turn_cost:.4f}，达到单轮上限 ¥{self.max_cost:.2f}", False
        if self.guard is not None:
            verdict = self.guard.check()
            if not verdict:
                return verdict.reason, verdict.money
        return "", False

    # 小节点（剧本中的某一集）的 stage 形如「第3集」「12集」「单集」——/auto 下不问人
    _MINOR_STAGE = re.compile(r"第?\d+\s*集|单集")

    def _is_major_review(self, payload: dict[str, Any]) -> bool:
        """这次人审是不是大节点：/auto 下大节点仍挂起问人一次，小节点自动采纳。

        request_review 显式给了 major 就听它的；没给才按 stage 关键词判：
        命中「第N集/单集」一律算小节点（即使带着「剧本」字样，如「剧本第3集」）；
        否则含任一 major_stages 关键词即大节点。
        """
        major = payload.get("major")
        if isinstance(major, bool):
            return major
        stage = str(payload.get("stage") or "")
        if self._MINOR_STAGE.search(stage):
            return False
        return any(k in stage for k in self.major_stages)

    async def _auto_adopt(self, turn: Turn, call_id: str, result: ToolResult) -> None:
        """/auto 模式的小节点人审结案：不挂起，按「已采纳」回填，循环继续。

        事件照发（CHECKPOINT_REACHED / CHECKPOINT_DECIDED，decided_by='auto'）——
        痕迹、回放、打回落库的消费方不需要区分这次是不是人点的。
        采纳不落避雷记忆（RejectionRecorder 本来就跳过 adopt）。
        """
        payload = result.suspend_payload
        turn.messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": "人的决策：已采纳（/auto 自动模式，人没有逐条确认）",
            }
        )
        await self.bus.emit(
            EventType.CHECKPOINT_REACHED, turn=turn.index, call_id=call_id, **payload
        )
        await self.bus.emit(
            EventType.CHECKPOINT_DECIDED,
            turn=turn.index,
            node=payload.get("stage", "review"),
            target_node=payload.get("target", payload.get("stage", "")),
            decision="adopt",
            reason="",
            decided_by="auto",
            candidates=payload.get("assets", []),
        )

    async def resume_turn(
        self, decision: str, reason: str = "", decided_by: str = "human"
    ) -> LoopResult:
        """人做完决策后继续被挂起的那一轮。

        把决策作为那次 request_review 调用的返回值填回去，模型据此接着干。
        对模型来说这就是一次普通的工具返回，不需要它理解「流程被挂起过」。
        """
        if self.pending_review is None:
            raise RuntimeError("当前没有挂起的人审请求")
        if decision != "adopt" and not reason.strip():
            raise ValueError("打回必须填理由 —— 不填的话下一版会犯一模一样的错")

        pending = self.pending_review
        self.pending_review = None
        turn = next(t for t in self.memory.turns if t.index == pending["turn_index"])

        await self.bus.emit(
            EventType.CHECKPOINT_DECIDED,
            turn=turn.index,
            node=pending.get("stage", "review"),
            target_node=pending.get("target", pending.get("stage", "")),
            decision=decision,
            reason=reason,
            decided_by=decided_by,
            candidates=pending.get("assets", []),
        )

        verdict = {"adopt": "已采纳", "revise": "打回重做", "reject": "方向不对，退回上一步"}
        body = f"人的决策：{verdict.get(decision, decision)}"
        if reason:
            # 采纳时的附言是要遵守的补充要求，不是要避开的理由 —— 标签写错，
            # 模型会把「控制在 60 集」当成要避开的东西，正好做反。
            label = "补充要求（后续必须遵守）" if decision == "adopt" else "理由（下一版必须避开）"
            body += f"\n{label}：" + reason

        turn.messages.append(
            {"role": "tool", "tool_call_id": pending["call_id"], "content": body}
        )
        # 决策回填之后再修补：挂起那一轮之前的轮次可能留过 Ctrl+C 的残缺。
        # 必须放在决策回填后，否则挂起的 call_id 会先被补一条「已中断」，
        # 同一个 tool_call_id 出现两条 tool 响应，API 照样 400。
        repaired = self._repair_interrupted_calls()
        if repaired:
            await self.bus.emit(
                EventType.WARNING,
                message=f"上一轮被中断：已补 {repaired} 条没有执行结果的调用",
            )
        return await self._continue(turn)

    async def continue_turn(self, turn: Turn) -> LoopResult:
        """max_iterations 撞线后在**同一轮**里接着跑（auto 模式自动续跑用）。

        max_iterations 是单段护栏不是任务终点：人审前后算同一问一答的约定
        在这里同样成立 —— 续跑不新开轮次，滑窗计数不变。
        """
        return await self._continue(turn)

    async def _continue(self, turn: Turn) -> LoopResult:
        """从已有的 turn 继续跑，不新建轮次 —— 人审前后仍算同一问一答。"""
        final_text, stop, cost, i = await self._iterate(turn)
        turn.tokens = self.assembler.calibrated(estimate_tokens(turn.messages))
        await self.bus.emit(
            EventType.LOOP_END, turn=turn.index, iterations=i, stop_reason=stop.value
        )
        return LoopResult(
            text=final_text,
            turn=turn,
            iterations=i,
            stop_reason=stop,
            cost=cost or None,
            error_kind=self.last_error_kind,
            tool_failures=list(self._failures),
        )

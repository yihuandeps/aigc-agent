"""M3 上下文装配 —— 决定每次请求里到底放什么、按什么顺序放。

排布顺序（稳定 → 易变，让 prompt cache 的前缀尽可能长）：

    系统提示词 → [外部声明区] → 能力目录 → system 位 pin
      → 短期记忆(10-15 轮) → 长期记忆召回 → pre_input 位 pin → 当前输入
                              └ 每轮变，故放末尾

两处顺序是有意为之：
  · 长期记忆召回放在短期记忆之后 —— 它每轮都可能变，放前面会让后面所有
    内容的缓存失效；放末尾则只影响它自己和当前输入。
  · pre_input 位的 pin 紧贴当前输入 —— 长上下文中段召回率明显低于首尾，
    硬约束（must_not）要放在注意力最强的位置。

另：OpenAI 兼容协议下工具定义走 tools= 参数，位置由服务端决定，我们控制
不了。所以 M20 的「外部声明区」防注入措施要落在**每个工具的 description
前缀**上，而不是靠消息排序。P2 接 MCP 时在 registry 里加。

2026-09-17 加的两样（见 compaction.py）：

  · **轮内压缩**：历史轮里超过阈值的工具参数/结果折叠成存根；当前轮只保留
    最近几次迭代的原文。正文都在资产库里，留在上下文里的只是重复品。
  · **估算校准**：估算器和真实分词器对不上是常态 —— 实测 Kimi 对「JSON 里的
    中文」低估四成，18 万的熔断实际要到 25 万才触发，直接撞上模型上限。
    每次拿到真实 usage 就校准一次，之后的预算判断都按校准值算。
"""

from __future__ import annotations

from typing import Any

from ..events.bus import EventBus, EventType
from ..model.gateway import estimate_tokens
from .compaction import HARD, CompactionPolicy, fold_turn_messages
from .window import ShortTermMemory, Turn

DEFAULT_SYSTEM_PROMPT = """你是一个助手。

- 需要外部信息或要执行动作时调用工具，不要凭空编造。
- 不确定就直说不确定。
- **如实汇报结果**：工具失败、被拦、被截断、没读全 —— 都在回复里直接说出来。

（领域相关的提示词由装配层注入，见 aigc_agent/domain/system_prompt.py）"""


class ContextAssembler:
    def __init__(
        self,
        bus: EventBus,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        capability_budget: int = 30_000,
        compaction: CompactionPolicy | None = None,
        token_budget: int = 0,
        calibration: float = 1.0,
    ) -> None:
        self.bus = bus
        self.system_prompt = system_prompt
        self.capability_budget = capability_budget
        self.compaction = compaction or CompactionPolicy()
        # 整个请求的 token 上限（校准后的估算）。0 = 不做预算判断。
        # 装配层给 Loop 提供 fits()，超了由 Loop 剔历史轮 —— 剔轮要走记忆提取，
        # 那是 Loop 的事，装配层不动 memory。
        self.token_budget = token_budget
        # 估算 → 真实 token 的校准系数。observe() 用真实 usage 持续修正。
        self.calibration = calibration
        self.last_estimate = 0  # 上次装配的原始估算
        self.last_tokens = 0  # 上次装配的校准估算
        self.last_folded = 0  # 上次装配折叠了几处
        # 应急档：撞上模型上限后由 Loop 置上，本轮内一直生效（新轮开始时清）
        self.shrink = False
        # 历史轮折不折的决定要记住 —— 校准系数会漂，阈值附近的轮次
        # 来回翻转会让前缀不稳定，缓存白白击穿
        self._fold_decisions: dict[int, bool] = {}
        # 随请求发出去的工具 schema 的估算（原始口径）。每次请求都带，是个加性常数：
        # 之前不计 —— 89 份 schema 约 1.5 万 token，首轮消息才估 3K，校准比值被顶到 3.0 上限，
        # 历史轮的折叠阈值实际降到约 700 token（2026-09-23 审查）
        self.tools_tokens = 0

    def set_tools(self, tools: list[dict[str, Any]] | None) -> None:
        import json

        self.tools_tokens = (
            estimate_tokens(json.dumps(tools, ensure_ascii=False)) if tools else 0
        )

    # ---------- 校准 ----------

    def calibrated(self, raw_estimate: int) -> int:
        return int(raw_estimate * self.calibration)

    def observe(self, actual_prompt_tokens: int) -> None:
        """拿真实 usage 校准估算。指数平滑，新观测权重大 —— 上下文构成会变
        （一批工具结果进来），校准要跟得上。"""
        if actual_prompt_tokens <= 0 or self.last_estimate <= 0:
            return
        ratio = max(0.5, min(3.0, actual_prompt_tokens / self.last_estimate))
        self.calibration = 0.4 * self.calibration + 0.6 * ratio

    def fits(self, messages: list[dict[str, Any]]) -> bool:
        if not self.token_budget:
            return True
        return self.tokens_of(messages) <= self.token_budget

    def tokens_of(self, messages: list[dict[str, Any]]) -> int:
        return self.calibrated(estimate_tokens(messages) + self.tools_tokens)

    def view_tokens(self, t: Turn) -> int:
        """这一轮作为历史轮时实际发出去的大小（折叠后的视图）。驱逐按它算。"""
        view, _ = self._history_view(t)
        return self.calibrated(estimate_tokens(view))

    # ---------- 视图 ----------

    def _history_view(self, t: Turn) -> tuple[list[dict[str, Any]], int]:
        """历史轮：超过阈值的整轮折叠（参数/结果换存根），小轮原样保留。"""
        policy = HARD if self.shrink else self.compaction
        if t.index not in self._fold_decisions:
            # t.tokens 存的已经是校准后的数（Loop 记的），不能再乘一次校准系数
            size = t.tokens or self.calibrated(estimate_tokens(t.messages))
            self._fold_decisions[t.index] = size > policy.history_turn_tokens
        if not self._fold_decisions[t.index] and not self.shrink:
            return list(t.messages), 0
        return fold_turn_messages(t.messages, keep_tail_iterations=0, policy=policy)

    def _current_view(self, t: Turn) -> tuple[list[dict[str, Any]], int]:
        """当前轮：涨过阈值后只保留最近几次迭代的原文，更早的折叠。"""
        policy = HARD if self.shrink else self.compaction
        size = self.calibrated(estimate_tokens(t.messages))
        if size <= policy.current_turn_tokens and not self.shrink:
            return list(t.messages), 0
        return fold_turn_messages(
            t.messages, keep_tail_iterations=policy.keep_recent_iterations, policy=policy
        )

    # ---------- 装配 ----------

    async def assemble(
        self,
        memory: ShortTermMemory,
        current_turn: Turn,
        tool_catalog: str = "",
        recalled: str = "",
    ) -> list[dict[str, Any]]:
        """组装本次请求的完整消息列表。

        current_turn 已经在 memory.turns 里（它是最后一个），
        这里把它连同历史一起铺开。
        """
        messages: list[dict[str, Any]] = []
        folded = 0

        # ---- 1. 系统区（最稳定，吃缓存）----
        system_parts = [self.system_prompt]

        if tool_catalog:
            system_parts.append("## 可用工具目录\n" + tool_catalog)

        for p in memory.pins_at("system"):
            system_parts.append(p.content)

        messages.append({"role": "system", "content": "\n\n---\n\n".join(system_parts)})

        # ---- 2. 短期记忆（历史轮，追加为主；大轮折叠成存根）----
        live = {t.index for t in memory.turns}
        self._fold_decisions = {k: v for k, v in self._fold_decisions.items() if k in live}
        history = [t for t in memory.turns if t.index != current_turn.index]
        for t in history:
            view, n = self._history_view(t)
            folded += n
            messages.extend(view)

        # ---- 3. 长期记忆召回（每轮可能变，故置于历史之后）----
        if recalled:
            messages.append(
                {"role": "system", "content": "## 相关记忆\n" + recalled}
            )

        # ---- 4. pre_input 位 pin（注意力最强的位置）----
        pre = memory.pins_at("pre_input")
        if pre:
            messages.append(
                {
                    "role": "system",
                    "content": "## 本次必须遵守\n" + "\n\n".join(p.content for p in pre),
                }
            )

        # ---- 5. 当前轮（涨大了折旧迭代）----
        view, n = self._current_view(current_turn)
        folded += n
        messages.extend(view)

        # 估算含工具 schema：校准系数是拿 usage.prompt_tokens（含 schema）对着它算的
        raw = estimate_tokens(messages) + self.tools_tokens
        self.last_estimate = raw
        self.last_tokens = self.calibrated(raw)
        self.last_folded = folded
        await self.bus.emit(
            EventType.CONTEXT_ASSEMBLED,
            messages=len(messages),
            turns_in_window=len(memory.turns),
            pins=len(memory.pins),
            est_tokens=raw,
            tokens=self.last_tokens,
            calibration=round(self.calibration, 2),
            folded=folded,
            shrink=self.shrink,
        )
        if folded:
            await self.bus.emit(
                EventType.CONTEXT_COMPACTED,
                folded=folded,
                tokens=self.last_tokens,
                shrink=self.shrink,
            )
        return messages

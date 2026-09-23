"""M3 窗口策略 —— 短期记忆的滑动与驱逐。

分工（见 ARCHITECTURE.md M8.0）：
  · M8 定策略 —— 窗口多大、一轮怎么算、什么该 pin、溢出后交给谁
  · M3 执行   —— 实际的消息裁剪、token 计数、组装请求
物理上只有一份消息队列，在这里。M8 不自己再存一份。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

# 给记忆提取的原文里，各角色消息最多留多少字（0 = 不截）。见 Turn.transcript
_TRANSCRIPT_CAP = {"tool": 600, "assistant": 3000}


class Turn(BaseModel):
    """一问一答 = 1 轮。

    该轮内助手的工具调用与结果都算在这一轮里，不单独计轮。
    所以 1 轮可能是 2 条消息，也可能是 60 条。
    """

    index: int
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tokens: int = 0  # 原文大小（校准后）
    # 作为历史轮时实际发出去的大小（折叠后、校准后）。驱逐按它算 —— 大轮在历史里是折叠的，
    # 按原文算会把 12 万的兜底提前触发（2026-09-23 审查）
    view_tokens: int = 0

    @property
    def weight(self) -> int:
        return self.view_tokens or self.tokens

    @property
    def transcript(self) -> str:
        """整轮的可读文本。给记忆提取用 —— 它要看到完整的一问一答，
        只给 user_text 会丢掉助手做了什么、人审说了什么。

        工具返回只留开头、助手的长回复截断（2026-09-23 审查：整本剧本、整份分镜原样喂给
        记忆提取，单次 4.1 万 token；记忆要的是人的偏好和决定，不是工具吐出来的正文）。
        用户的话和人审决策（短）原样保留。
        """
        out = []
        for m in self.messages:
            role = m.get("role", "")
            content = m.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            text = content.strip()
            cap = _TRANSCRIPT_CAP.get(role, 0)
            if cap and len(text) > cap:
                text = f"{text[:cap]}…（共 {len(text)} 字，后面略）"
            tag = {"user": "用户", "assistant": "助手", "tool": "工具", "system": "系统"}
            out.append(f"{tag.get(role, role)}：{text}")
        return "\n".join(out)

    @property
    def user_text(self) -> str:
        for m in self.messages:
            if m.get("role") == "user":
                return str(m.get("content") or "")
        return ""

    @property
    def assistant_text(self) -> str:
        for m in reversed(self.messages):
            if m.get("role") == "assistant" and m.get("content"):
                return str(m["content"])
        return ""


@dataclass
class WindowPolicy:
    """滑动窗口 + 批量驱逐。

    为什么是批量驱逐而不是逐轮驱逐：
      滑窗每滑动一次前缀就变，prompt cache 整个失效。一超 10 轮就每轮剔一轮
      等于每轮全量重算。让窗口涨到 15 再一次性剔回 10，每 5 轮只有 1 轮击穿
      缓存，命中率从 0% 提到 80%。代价是峰值多占约 60K —— 256K 完全吃得下。
    """

    window_turns: int = 10
    evict_at: int = 15
    # token 兜底（校准后的估算，见 assembler.observe）。正常不会触发，只是保险丝：
    # 防某一轮里连调几十次工具且结果没走资产引用。
    # 2026-09-17 从 180K 降到 120K：ARCHITECTURE 的软目标是 60–80K，
    # 之前的 180K 又叠上估算低估四成，实际要到 25 万才触发，直接撞模型上限。
    max_tokens: int = 120_000
    # 回差：token 兜底一旦触发，一次剔到这个水位以下，而不是刚好压到 max_tokens 下面 ——
    # 之前每轮都剔掉最老一轮，缓存每轮被打穿，外加每轮一次记忆提取
    low_tokens: int = 70_000

    def should_evict(self, turns: list[Turn]) -> bool:
        if len(turns) >= self.evict_at:
            return True
        return sum(t.weight for t in turns) > self.max_tokens

    def split(self, turns: list[Turn]) -> tuple[list[Turn], list[Turn]]:
        """返回 (保留, 驱逐)。驱逐的那批打包交给 Memory Agent 提关键词。"""
        if not self.should_evict(turns):
            return turns, []

        keep = turns[-self.window_turns :] if self.window_turns > 0 else []
        evicted = turns[: len(turns) - len(keep)]

        # token 兜底触发时继续从最老的剔，一次剔到低水位（低水位必须低于上限，否则按六成算）
        if sum(t.weight for t in turns) > self.max_tokens:
            low = (
                self.low_tokens
                if 0 < self.low_tokens < self.max_tokens
                else int(self.max_tokens * 0.6)
            )
            while keep and sum(t.weight for t in keep) > low and len(keep) > 1:
                evicted.append(keep.pop(0))

        return keep, evicted


@dataclass
class Pin:
    """常驻内容，不占窗口配额、不被驱逐。

    没有 pin 会出现：用户第 1 轮说「这个号不要用感叹号」，第一次批量驱逐
    时这句话被剔出窗口、压成关键词后极性还可能丢失，Agent 开始满屏感叹号。
    窗口从 5 放大到 10 只是把失效点推后，并没有消除它。

    position:
      · "system"    —— 并入系统提示词（稳定，吃缓存）
      · "pre_input" —— 紧贴当前用户输入之前（注意力最强的位置）
                       硬约束 must_not 放这里
    """

    key: str
    content: str
    position: str = "pre_input"


@dataclass
class ShortTermMemory:
    """短期记忆：滑动窗口内的原文对话。"""

    policy: WindowPolicy = field(default_factory=WindowPolicy)
    turns: list[Turn] = field(default_factory=list)
    pins: dict[str, Pin] = field(default_factory=dict)
    _next_index: int = 0

    # 驱逐回调：P1 接 Memory Agent 的压缩路径（异步入队，不阻塞主循环）
    on_evict: Callable[[list[Turn]], Any] | None = None
    # 上一次模型调用真实的 prompt token 数（usage 回传）。估算之外的真值，
    # 给预检和 /stat 用
    observed_prompt_tokens: int = 0

    def new_turn(self) -> Turn:
        t = Turn(index=self._next_index)
        self._next_index += 1
        self.turns.append(t)
        return t

    def pin(self, key: str, content: str, position: str = "pre_input") -> None:
        self.pins[key] = Pin(key=key, content=content, position=position)

    def unpin(self, key: str) -> None:
        self.pins.pop(key, None)

    def maybe_evict(self) -> list[Turn]:
        keep, evicted = self.policy.split(self.turns)
        if evicted:
            self.turns = keep
        return evicted

    def evict_oldest(self, n: int = 1) -> list[Turn]:
        """立刻剔掉最老的 n 轮（永远留着最后一轮，那是当前轮）。

        预检发现下一次请求装不下时用：剔整轮、完整地交给记忆提取，
        比在轮内乱截安全。
        """
        n = max(0, min(n, len(self.turns) - 1))
        if n == 0:
            return []
        evicted, self.turns = self.turns[:n], self.turns[n:]
        return evicted

    def pins_at(self, position: str) -> list[Pin]:
        return [p for p in self.pins.values() if p.position == position]

    @property
    def token_estimate(self) -> int:
        return sum(t.tokens for t in self.turns)

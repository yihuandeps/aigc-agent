"""按内容自动选音色和语速。

为什么值得单独做一步：同一个音色配所有内容，是"听着假"的一个隐性来源 ——
用新闻女声念一条生活种草，语气和内容对不上，听感就别扭，哪怕音色本身很自然。

为什么交给模型而不是写规则：选音色靠的是对文案语气的判断（这段是在讲事实、
讲故事，还是在吐槽），规则只能按关键词匹配，一遇到"用轻松口吻讲硬知识"
这种就抓瞎。模型看全文，判断更准。

但**模型的选择必须被校验**：它会编出不存在的音色名（实测 MiniMax 收到
不认识的 voice_id 不报错，静默换成默认音色 —— 比报错更难发现）。
所以这里只信白名单内的结果，其余一律退回配方默认值。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# 语速允许的范围。超出这个区间人耳会明显觉得不对劲，
# 模型有时会为了"更有节奏"给出 0.5 这种离谱值，必须夹住。
SPEED_MIN = 0.8
SPEED_MAX = 1.15


@dataclass(frozen=True)
class VoicePick:
    voice: str
    speed: float
    why: str = ""

    def line(self) -> str:
        return f"{self.voice}（语速 {self.speed:g}）" + (f" — {self.why}" if self.why else "")


def pick_prompt(script: str, voices: list[tuple[str, str]], topic: str = "") -> str:
    """voices 是 (音色 id, 说明) 列表，说明里写清楚适合什么内容。"""
    menu = "\n".join(f"- {name}：{note}" for name, note in voices if name)
    head = f"选题：{topic}\n\n" if topic else ""
    return (
        f"{head}这是一条短视频的口播文案：\n\n{script}\n\n"
        f"从下面的音色里挑一个最贴这段文案语气的：\n\n{menu}\n\n"
        "判断依据是**文案在做什么**，不是题材标签：\n"
        "  · 在陈述事实、给信息 → 偏播报型\n"
        "  · 在讲一件事的来龙去脉 → 偏叙述型，要有呼吸感\n"
        "  · 在表达观点或吐槽 → 偏有态度的\n"
        "  · 在跟人分享、种草 → 偏聊天感\n"
        "题材和语气经常不一致（科技选题也可以用聊天口吻讲），以**文案实际语气**为准。\n\n"
        f"再给一个语速，{SPEED_MIN} 到 {SPEED_MAX} 之间，1.0 是原速。\n"
        "信息密度高、句子长的放慢一点；短句多、节奏快的可以接近原速。\n\n"
        "只输出 JSON：\n"
        '{"voice": "音色 id 原样照抄", "speed": 0.95, "why": "一句话说明为什么"}'
    )


def parse_pick(
    text: str, allowed: list[str], fallback: str, default_speed: float = 1.0
) -> VoicePick:
    """解析并**校验**模型的选择。

    校验不是可选的：模型会编音色名，而 MiniMax 收到不认识的 id 不会报错，
    只会静默换成自己的默认音色 —— 你以为按内容选了音色，其实一直是同一个。
    """
    raw = _unwrap(text)
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return VoicePick(fallback, default_speed, "")
    if not isinstance(d, dict):
        return VoicePick(fallback, default_speed, "")

    voice = str(d.get("voice") or "").strip()
    if voice not in allowed:
        voice = _fuzzy(voice, allowed) or fallback

    try:
        speed = float(d.get("speed") or default_speed)
    except (TypeError, ValueError):
        speed = default_speed
    speed = round(min(SPEED_MAX, max(SPEED_MIN, speed)), 2)

    return VoicePick(voice, speed, str(d.get("why") or "").strip())


def _fuzzy(name: str, allowed: list[str]) -> str:
    """模型偶尔会把 id 抄漏一截或只写中文名，能对上就认。"""
    if not name:
        return ""
    low = name.lower()
    for a in allowed:
        if a.lower() == low:
            return a
    for a in allowed:
        if low in a.lower() or a.lower() in low:
            return a
    return ""


def _unwrap(text: str) -> str:
    raw = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if m:
        raw = m.group(1).strip()
    i, j = raw.find("{"), raw.rfind("}")
    return raw[i : j + 1] if i >= 0 and j > i else raw

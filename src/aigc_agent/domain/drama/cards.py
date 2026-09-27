"""剧本里的上屏字（字幕卡）—— 抽出来后期叠，不让生成模型画。

剧本写法：「【字幕：灵山】」（地点 / 时间 / 身份卡）、「【片尾字幕：剩余花瓣 9】」（片尾计数）。

2026-09-26 审查：「画面不许有字」（用户定的最高优先级规则）让这些字两头落空 ——
  · 拆分镜时被悄悄丢掉：《不渡》第 1 集剧本 6 张卡，分镜里只剩片尾那张；
  · 留下的被写进视频提示词让模型画（「黑白字迹无声浮现…【剩余花瓣 9】」）：模型不画，剧情
    计数器就没了；画了，字幕门判 ⛔、缺段不成片。
用户定的解法：分镜 / 提示词这一步把字幕卡抽成叠字清单，成片后用 overlay_text 叠上去。

这里只管认、剥、对（纯函数）；镜号定位在 format.card_lines，排时间线在 drama_shots，
叠字在 drama_render_shots 拼完成片之后。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 认哪些写法。「字幕」在剧本里指的是字卡（地名 / 时间 / 身份），不是台词字幕
_CARD = re.compile(
    r"【\s*(片尾字幕|片头字幕|字幕|字卡|标题卡|上屏字|屏幕字)\s*[：:]\s*([^】\n]{1,60}?)\s*】"
)
_KEY = re.compile(r"[0-9A-Za-z぀-ヿ一-鿿가-힯]")


@dataclass(frozen=True)
class Card:
    kind: str  # 字幕 / 片尾字幕 / …
    text: str

    @property
    def ending(self) -> bool:
        """片尾卡：叠到这一集结束。"""
        return self.kind.startswith("片尾")

    @property
    def marker(self) -> str:
        return f"【{self.kind}：{self.text}】"


def find_cards(text: str) -> list[Card]:
    return [Card(m.group(1), m.group(2).strip()) for m in _CARD.finditer(text or "")]


def card_key(text: str) -> str:
    """比对用：只留文字和数字（「剩余花瓣 9」和「剩余花瓣9」算同一张）。"""
    return "".join(_KEY.findall(text or ""))


def strip_cards(text: str, extra: list[str] | None = None) -> str:
    """去掉上屏字标记（喂给视频提示词的模型之前 / 存提示词之前）。

    extra：已知的卡片文字 —— 模型有时把「【片尾字幕：剩余花瓣 9】」改写成「【剩余花瓣 9】」，
    光认标记认不出来，按文字再剥一遍。"""
    out = _CARD.sub("", text or "")
    for t in extra or []:
        if t:
            out = re.sub(r"【\s*" + re.escape(t) + r"\s*】", "", out)
    # 剥掉后留下的「，。」「：。」这类
    out = re.sub(r"([，,：:])\s*([。．.！!？?])", r"\2", out)
    return re.sub(r"[ \t]{2,}", " ", out)


def missing_cards(script: str, board: str) -> list[Card]:
    """剧本里有、分镜里找不到的字幕卡（按文字比对）。"""
    have = {card_key(c.text) for c in find_cards(board)}
    seen: set[str] = set()
    out: list[Card] = []
    for c in find_cards(script):
        k = card_key(c.text)
        if k and k not in have and k not in seen:
            seen.add(k)
            out.append(c)
    return out


def card_loss(title: str, script: str, board: str) -> str:
    """分镜比剧本少了字幕卡的问题描述；没少返回空串。"""
    miss = missing_cards(script, board)
    if not miss:
        return ""
    shown = "、".join(c.marker for c in miss[:4])
    return (
        f"{title}：剧本里的上屏字分镜里少了 {len(miss)} 张（如 {shown}）—— 照抄整个【…】标记，"
        "写回它该出现的那个镜头行末尾（后期叠字要靠它，一张都不能丢）"
    )


CARD_STORYBOARD_RULE = (
    "· 剧本里的上屏字（【字幕：…】【片尾字幕：…】这类字卡）**照抄整个【…】标记**，写在它该出现的"
    "那个镜头行末尾，一张都不能丢；画面描述里不要写「字迹浮现」「字幕出现」这类让画面出字的话"
    " —— 上屏字由后期叠，生成的画面里不许有字。"
)
CARD_SHOTS_RULE = (
    "· **画面里不许出现任何文字**（地名、时间、计数、片名、招牌字都不行）：分镜里的上屏字已经"
    "单独抽出来由后期叠，description 里不要写这些字，也不要写「字迹浮现 / 字幕出现」这类画面。"
)

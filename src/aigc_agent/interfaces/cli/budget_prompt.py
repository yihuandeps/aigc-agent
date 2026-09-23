"""开工额度确认（用户 2026-09-13 定的规则：每次开工前确认上限 —— 金额 / 次数 / 视频秒数）。

2026-09-23 审查：这条规则在这份副本里从没落地 —— 启动流程不问、面板只列 /budget 用法，
视频又没有单价（金额上限看不见它），/auto 下次数超限还自动放行，结果视频花费没有任何刹车。
现在每次开工都把「这次开工」的额度列出来让人回车确认或改，超了一律停下来问人。
"""

from __future__ import annotations

import re
from typing import Any

# 认哪些说法（精确词优先，其次按包含的字判）
_EXACT: dict[str, str] = {
    **{w: "money" for w in ("金额", "钱", "元", "money", "¥", "预算")},
    **{w: "video_seconds" for w in ("视频秒", "视频秒数", "秒数", "秒", "秒钟", "seconds", "sec")},
    **{w: "video_calls" for w in ("视频段", "视频段数", "段数", "段", "视频", "video", "videos")},
    **{w: "image_calls" for w in ("图片", "张数", "图", "张", "image", "images", "img")},
}


def _key_for(word: str) -> str:
    w = word.strip().lower()
    if w in _EXACT:
        return _EXACT[w]
    if "秒" in w:
        return "video_seconds"
    if "段" in w or "视频" in w:
        return "video_calls"
    if "图" in w:
        return "image_calls"
    if any(c in w for c in ("金", "钱", "元", "¥")):
        return "money"
    return ""


def parse_budget(text: str) -> dict[str, float]:
    """「金额 300 视频秒 900 视频 80 图 100」→ {money: 300, video_seconds: 900, ...}。

    词和数字之间可以有空格、冒号、等号；多组用空格或逗号隔开。认不出的词忽略。
    """
    s = re.sub(r"[：:=，,;；]", " ", text or "")
    out: dict[str, float] = {}
    for m in re.finditer(r"([^\d\s.]+)\s*(\d+(?:\.\d+)?)", s):
        key = _key_for(m.group(1))
        if key:
            out[key] = float(m.group(2))
    return out


def render_budget(limits: dict[str, Any], priced: bool) -> str:
    """给人看的额度清单（CLI 面板里的正文）。"""

    def num(v: Any, unit: str, money: bool = False) -> str:
        if v is None:
            return "不限"
        return f"¥{float(v):.0f}" if money else f"{float(v):.0f} {unit}"

    lines = [
        f"金额   {num(limits.get('money'), '', money=True)}（本次开工，含文本模型）",
        f"视频   ≤ {num(limits.get('video_calls'), '段')} · "
        f"≤ {num(limits.get('video_seconds'), '秒')}",
        f"图片   ≤ {num(limits.get('image_calls'), '张')}",
    ]
    if not priced:
        lines.append(
            "[yellow]⚠ 媒体模型没配单价（config/media_models.yaml 的 price / price_per_second）："
            "金额上限只管文本，视频靠「段数 / 秒数」管[/]"
        )
    lines.append("[dim]超了会停下来问你（/auto 下也问）；单日 / 单项目的累计上限照常生效[/]")
    return "\n".join(lines)

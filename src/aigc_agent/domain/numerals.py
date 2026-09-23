"""集号里的中文数字（2026-09-23 审查：资产库和本地素材都不认「第十二集」）。

用户和模型两种写法都有：「第12集」「第十二集」「第一百零五集」。资产摘要、文件名、人审环节名
都要认得出来，否则「第十二集·合规修订版」不算第 12 集的剧本。
"""

from __future__ import annotations

import re

_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
CN_NUM = "零〇一二两三四五六七八九十百"
# 第 12 集 / 第十二集（最多三位数：一部短剧不会过一百集太多）
EPISODE_RE = re.compile(rf"第\s*(\d{{1,3}}|[{CN_NUM}]{{1,6}})\s*集")


def cn_to_int(text: str) -> int:
    """一 → 1，十二 → 12，二十 → 20，一百零五 → 105；阿拉伯数字原样。认不出返回 0。"""
    s = (text or "").strip()
    if not s:
        return 0
    if s.isdigit():
        return int(s)
    total = num = 0
    for ch in s:
        if ch in _DIGITS:
            num = _DIGITS[ch]
        elif ch == "十":
            total += (num or 1) * 10
            num = 0
        elif ch == "百":
            total += (num or 1) * 100
            num = 0
        else:
            return 0
    return total + num


def episode_from(text: str) -> int:
    """文字里的「第N集」→ N；没有返回 0。"""
    m = EPISODE_RE.search(text or "")
    return cn_to_int(m.group(1)) if m else 0

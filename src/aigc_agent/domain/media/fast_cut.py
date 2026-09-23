"""每个镜头不超过 3 秒（2026-09-20，用户定的硬性要求：不管一集多长，单镜必须 ≤3 秒）。

生成模型一次出 10–15 秒，所以一段视频必须是**多镜头快切**：段内每个镜头 ≤3 秒、镜头之间硬切。
两道保险，和禁字幕一个套路：
  1. **提示词**：短剧渲染的每段提示词最前面加一段硬约束 + 按 cuts 排好的时间线，末尾再复述。
  2. **镜头门**（drama.cut_gate）：生成后用 ffmpeg 的场景切换检测量出最长镜头，
     超过上限就用更硬的提示词重生成一次；还超就在结果里标出来交人复核。
     纯 ffmpeg，不花模型钱。
"""

from __future__ import annotations

import re

FAST_CUT_MARK = "【多镜头快切·硬性】"
FAST_CUT_RETRY_MARK = "【重新生成·镜头太长】"

_TAIL_CN = "再次强调：每个镜头不超过 {mc} 秒，到时间必须硬切到下一个镜头。"
_TAIL_EN = "Reminder: no single shot may last longer than {mc} seconds; hard cut to the next shot."


def timeline(cuts: list[float], seconds: float = 0.0) -> str:
    """cuts → '0–3s 镜头① / 3–5s 镜头② / …'。cuts 空则返回空串。"""
    if not cuts:
        return ""
    parts: list[str] = []
    t = 0.0
    for i, c in enumerate(cuts, 1):
        end = t + float(c)
        parts.append(f"{t:g}–{end:g}s 镜头{i}")
        t = end
    return " / ".join(parts)


def cut_rule_head(seconds: float, cuts: list[float], max_cut: float) -> str:
    """段首的硬约束：几个镜头、每个多长、镜头之间硬切；有 cuts 就带时间线。"""
    mc = f"{max_cut:g}"
    n = len(cuts) if cuts else max(1, int(-(-float(seconds) // max_cut))) if seconds else 0
    head = (
        f"{FAST_CUT_MARK}本段 {seconds:g} 秒是多镜头快切：至少 {n} 个镜头，"
        f"**每个镜头不超过 {mc} 秒**，镜头之间直接硬切（换机位 / 景别 / 角度），"
        "不用淡入淡出、不用一个镜头长时间推拉或停留；同一机位不得连续超过 "
        f"{mc} 秒。"
    )
    tl = timeline(cuts, seconds)
    if tl:
        head += f" 时间线：{tl}。"
    head += (
        f" HARD RULE — multi-shot fast cutting: at least {n} shots in this {seconds:g}s clip, "
        f"NO shot longer than {mc} seconds, hard cuts between shots (change camera angle / "
        "framing), no dissolves, no lingering single take."
    )
    return head


def fast_cut_prompt(core: str, seconds: float, cuts: list[float], max_cut: float) -> str:
    """把快切硬约束包在画面描述外层：开头一段 + 末尾复述。幂等。"""
    p = (core or "").strip()
    if FAST_CUT_MARK in p:
        return p
    mc = f"{max_cut:g}"
    tail = _TAIL_CN.format(mc=mc) + " " + _TAIL_EN.format(mc=mc)
    return f"{cut_rule_head(seconds, cuts, max_cut)}\n{p}\n{tail}"


def fast_cut_retry(prompt: str, longest: float, max_cut: float) -> str:
    """重生成用：在快切约束之后、正文之前插一句"上一版有镜头长达 X 秒"。幂等。"""
    p = (prompt or "").strip()
    if FAST_CUT_RETRY_MARK in p:
        return p
    note = (
        f"{FAST_CUT_RETRY_MARK}上一版有一个镜头长达 {longest:.1f} 秒，超过 {max_cut:g} 秒的上限，"
        "属于不合格。这次必须切得更碎：每个镜头 ≤ "
        f"{max_cut:g} 秒，到时间就硬切到另一个机位或景别。 "
        f"REGENERATION — the previous output had a shot lasting {longest:.1f}s, over the "
        f"{max_cut:g}s limit. Cut faster this time: every shot ≤ {max_cut:g}s, hard cut to a new "
        "angle when time is up."
    )
    i = p.find(FAST_CUT_MARK)
    if i >= 0:
        j = p.find("\n", i)
        if j < 0:
            return f"{p}\n{note}"
        return f"{p[:j]}\n{note}{p[j:]}"
    return f"{note}\n{p}"


# ---------------------------------------------------------------- 生成后校验（ffmpeg 场景切换）

_PTS = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")


def parse_showinfo(stderr: str) -> list[float]:
    """ffmpeg showinfo 输出里的 pts_time → 切换时刻列表（升序去重）。"""
    times = sorted({float(t) for t in _PTS.findall(stderr or "")})
    return times


def merge_close(times: list[float], gap: float = 0.3) -> list[float]:
    """挨得太近的切换点合并（甩镜 / 闪光会让连续几帧都触发）。"""
    out: list[float] = []
    for t in sorted(times):
        if out and t - out[-1] < gap:
            continue
        out.append(t)
    return out


def longest_shot(duration: float, cut_times: list[float], gap: float = 0.3) -> float:
    """最长的一个镜头有多少秒：切换点把 [0, duration] 分成若干段，取最长的。"""
    d = max(0.0, float(duration))
    if d <= 0:
        return 0.0
    bounds = [0.0] + [t for t in merge_close(cut_times, gap) if 0.0 < t < d] + [d]
    return max(b - a for a, b in zip(bounds, bounds[1:], strict=False))

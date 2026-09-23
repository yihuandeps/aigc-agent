"""字幕对齐 —— 时间轴用 ASR 的，文字用原稿的。

**为什么不能直接用 ASR 的文字**：配音是由我们自己的文案合成的，原文就在手上，
它才是 ground truth。ASR 再准也是二次识别，同音字必错在专业名词上 ——
实测一条讲氢能的片子，"氢"全程被听成"芯"（"卡住芯能的"），
而这是整条片子的核心词，还烧进了画面。

那为什么还要 ASR：**时间轴**。按字数估算的时间轴和真实语速对不上，
一旦有停顿、语气词、数字读法差异就会整体漂移。ASR 给的是真实断句时刻，
这个是拿不到替代品的。

所以做法是：ASR 出段落与时刻 → 把原稿按各段的比重切开 → 贴回去。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 中文断句标点。切分优先落在这些位置，避免把词切断。
_BREAKS = "。！？；，、：!?;,:"
# 句末标点。切分优先吸附到这里 —— 在句中切会把下一句的开头词拖进上一条字幕。
_STRONG = "。！？!?"


@dataclass
class Cue:
    index: int
    start: str
    end: str
    text: str


def parse_srt(srt: str) -> list[Cue]:
    """解析 srt。解析不了就返回空，由上层退回原字幕。"""
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", srt.strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        m = re.search(r"(\S+)\s*-->\s*(\S+)", lines[1] if len(lines) > 1 else "")
        if m is None:  # 有些 srt 没有序号行
            m = re.search(r"(\S+)\s*-->\s*(\S+)", lines[0])
            if m is None:
                continue
            text = " ".join(lines[1:])
            idx = len(cues) + 1
        else:
            try:
                idx = int(lines[0].strip())
            except ValueError:
                idx = len(cues) + 1
            text = " ".join(lines[2:])
        cues.append(Cue(idx, m.group(1), m.group(2), text.strip()))
    return cues


def render_srt(cues: list[Cue]) -> str:
    blocks = [f"{i}\n{c.start} --> {c.end}\n{c.text}" for i, c in enumerate(cues, 1)]
    return "\n\n".join(blocks) + "\n"


def align_script(script: str, srt: str) -> tuple[str, int]:
    """把原稿贴到 ASR 的时间轴上，返回 (新 srt, 改动了几条)。

    对不上就原样返回 —— 字幕有点错字，也好过时间轴错乱。
    """
    cues = parse_srt(srt)
    clean = _normalize(script)
    if not cues or not clean:
        return srt, 0

    # 按各段 ASR 文字的长度分配原稿。用长度而不是时长：同样 2 秒，
    # 念数字和念成语的字数差很多，ASR 的字数更贴近实际内容量。
    weights = [max(1, len(_normalize(c.text))) for c in cues]
    total_w = sum(weights)

    pieces: list[str] = []
    pos = 0
    for i in range(len(weights)):
        if i == len(weights) - 1:
            pieces.append(clean[pos:])
            break
        want = round(len(clean) * (sum(weights[: i + 1]) / total_w))
        cut = _snap(clean, pos, want)
        pieces.append(clean[pos:cut])
        pos = cut

    changed = 0
    out: list[Cue] = []
    for c, piece in zip(cues, pieces, strict=True):
        t = piece.strip().strip("".join(_BREAKS))
        if not t:  # 分配到空段就保留原文，别留空白字幕
            t = c.text
        if t != c.text:
            changed += 1
        out.append(Cue(c.index, c.start, c.end, t))

    return render_srt(out), changed


def _snap(s: str, low: int, want: int, window: int = 6) -> int:
    """把切点吸附到标点，免得把词从中间切开。

    两条规则都是踩坑踩出来的：

    **句末标点优先于句中标点**。否则"…体积大幅缩小。第二，管道输氢落地"
    会被切在"第二，"后面，于是上一条字幕变成"体积大幅缩小。第二"——
    下一句的开头词被拖进了上一条，屏幕上读起来是错的。

    **同样距离时取靠前的**。宁可让这一条短一点，也不要把下一句的头抢过来。
    """
    want = max(low + 1, min(want, len(s)))

    # 1) 往回找句末标点。只往回 —— 往前够句号会跨过中间的逗号，
    #    把下一条字幕的内容整段吞掉（实测第 6 条只剩"第二"两个字）。
    for off in range(window + 1):
        cand = want - off
        if low < cand <= len(s) and s[cand - 1] in _STRONG:
            return cand

    # 2) 退而求其次：最近的任意标点，同距离时取靠前的
    for off in range(window + 1):
        for cand in (want - off, want + off):
            if low < cand <= len(s) and s[cand - 1] in _BREAKS:
                return cand
    return want


def _normalize(s: str) -> str:
    """去掉换行和多余空白。中文字幕里的空格是噪音。"""
    return re.sub(r"\s+", "", s or "")

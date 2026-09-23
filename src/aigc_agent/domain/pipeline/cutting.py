"""剪辑节奏规划 —— 把长素材切成密集的短镜头。

**为什么不直接生成 3 秒的片段**：视频生成模型的出片长度是固定档位
（veo3.1 = 8s，sora = 12s），要不到 3 秒。而且真要 30 秒成片切 12 刀，
就得生成 12 段，成本和耗时都是 3 倍。

所以走真实剪辑的做法：**素材生成按模型原生长度，节奏在剪辑层做**。
4 段 8 秒素材足够剪出 12 刀，一分钱不多花。

同一段素材会被切成多刀、在成片里出现多次 —— 这不是偷懒，是蒙太奇的
常规手法：隔几刀回到同一主体的另一个瞬间，比一镜到底更有节奏。
排刀时保证**相邻两刀不来自同一段素材**，否则接起来看不出切点。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# 单刀时长的抖动范围。全部等长会像幻灯片，人眼能察觉到机械感。
_JITTER_LOW = 0.72
_JITTER_HIGH = 1.25

# 目标刀长取上限的这个比例，留出向上抖动的余地；
# 直接按上限均分的话，任何向上的抖动都会越界。
_HEADROOM = 0.8


@dataclass(frozen=True)
class Cut:
    """一刀：从第 clip 段素材的 start 秒起，取 dur 秒。"""

    clip: int
    start: float
    dur: float

    @property
    def end(self) -> float:
        return self.start + self.dur


def plan_cuts(
    clip_seconds: list[float],
    total: float,
    max_seconds: float = 3.0,
    min_seconds: float = 1.2,
    seed: int | None = None,
) -> list[Cut]:
    """排出一张剪辑表，总长正好等于 total。

    max_seconds 是硬约束 —— 返回的每一刀都不会超过它。
    """
    usable = [s for s in clip_seconds if s > 0]
    if not usable or total <= 0:
        return []
    if total <= max_seconds:
        return [Cut(0, 0.0, min(total, clip_seconds[0]))]

    lens = _lengths(total, max_seconds, min_seconds, seed)
    return _assign(lens, clip_seconds)


# ---------- 单刀时长 ----------


def _lengths(total: float, max_s: float, min_s: float, seed: int | None) -> list[float]:
    n = _cut_count(total, max_s, min_s)
    rng = random.Random(seed if seed is not None else 0)
    factors = [rng.uniform(_JITTER_LOW, _JITTER_HIGH) for _ in range(n)]

    scale = total / sum(factors)
    lens = [f * scale for f in factors]

    # 夹到 [min, max] 会让总长跑偏，把差额摊回还有余量的刀上。
    # 迭代几轮就能收敛；收不干净的零头最后并到某一刀里。
    for _ in range(8):
        lens = [min(max_s, max(min_s, x)) for x in lens]
        gap = total - sum(lens)
        if abs(gap) < 1e-6:
            break
        room = [(max_s - x) if gap > 0 else (x - min_s) for x in lens]
        pool = sum(room)
        if pool < 1e-9:
            break
        lens = [x + gap * (r / pool) for x, r in zip(lens, room, strict=True)]

    lens = [min(max_s, max(min_s, x)) for x in lens]
    gap = total - sum(lens)
    if abs(gap) > 1e-6:
        for i, x in enumerate(lens):
            if min_s <= x + gap <= max_s:
                lens[i] = x + gap
                break

    return [round(x, 3) for x in lens]


def _cut_count(total: float, max_s: float, min_s: float) -> int:
    """刀数。要同时满足 n*min <= total <= n*max，否则夹不出合法解。"""
    lo = math.ceil(total / max_s - 1e-9)
    hi = math.floor(total / min_s + 1e-9) if min_s > 0 else lo
    want = round(total / (max_s * _HEADROOM))
    if hi < lo:  # min 比 max 还紧，以 max 为准（硬约束优先）
        return max(1, lo)
    return max(lo, min(hi, max(1, want)))


# ---------- 把刀分配到素材 ----------


def _assign(lens: list[float], clip_seconds: list[float]) -> list[Cut]:
    """轮流取素材。

    多段素材时每段内部用游标顺着往后推：隔了几刀再回到这一段，画面里的
    主体已经动过了，接起来像"换了个角度看同一件事"，是想要的效果。

    只有一段素材时不能这么干 —— 顺着取出来的相邻两刀是连续帧，根本看不出
    切点。这时改成把素材切成互不重叠的槽位，隔着跳，保证每一刀都落在明显
    不同的瞬间（素材不够长时会重复用到同一个槽，这是素材量的物理限制）。
    """
    n_clips = len(clip_seconds)
    if n_clips == 1:
        return _assign_single(lens, clip_seconds[0])

    cursor = [0.0] * n_clips
    wraps = [0] * n_clips
    cuts: list[Cut] = []

    for i, dur in enumerate(lens):
        c = i % n_clips
        avail = clip_seconds[c]
        take = min(dur, avail)
        if cursor[c] + take > avail + 1e-6:
            # 到头了绕回。每绕一圈错开一点，否则会一字不差地重放同一个窗口。
            wraps[c] += 1
            cursor[c] = min(wraps[c] * 0.9, max(avail - take, 0.0))
        cuts.append(Cut(c, round(cursor[c], 3), round(take, 3)))
        cursor[c] += take

    return cuts


def _assign_single(lens: list[float], avail: float) -> list[Cut]:
    """单素材：切成互不重叠的槽位，隔一个取一个。"""
    # 槽宽按**最长的一刀**定死，不能按每刀现算 —— 现算的话长短刀会算出
    # 不同的槽格，相邻两刀可能落到同一个起点上，等于没切。
    longest = max(lens)
    slots = max(1, int(avail // longest))
    width = avail / slots
    # 槽宽 >= 最长刀，所以不同槽的窗口一定不重叠。
    step = 2 if slots >= 3 else 1

    cuts: list[Cut] = []
    for i, dur in enumerate(lens):
        take = min(dur, avail)
        slot = (i * step) % slots
        start = min(slot * width, max(avail - take, 0.0))
        cuts.append(Cut(0, round(start, 3), round(take, 3)))
    return cuts


def describe(cuts: list[Cut]) -> str:
    """给人看的一句话摘要。"""
    if not cuts:
        return "（无剪辑表）"
    total = sum(c.dur for c in cuts)
    longest = max(c.dur for c in cuts)
    return f"{len(cuts)} 刀 · 共 {total:.1f}s · 单刀 {min(c.dur for c in cuts):.1f}–{longest:.1f}s"

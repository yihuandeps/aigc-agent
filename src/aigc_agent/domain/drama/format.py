"""一集的规格（2026-09-18 用户定的）：**一集 4 分钟；开场 15 秒必须给高潮点。**
2026-09-20 用户追加硬性要求：**不管一集多长，每一个镜头（分镜）不超过 3 秒。**

短视频平台的观众没有耐心：前 15 秒不给高潮就划走，一个镜头停 5 秒就划走。
所以这份规格是**硬约束**，贯穿整条链，而不是写在某一段提示词里让模型自觉：

  写剧本   字数目标（4 分钟 ≈ 1280 字）+ 开场高潮点（标题下一行「> ⚡ 前15秒高潮点：…」）
           + 动作写成可切的短拍（一个动作 / 一个反应 / 一句短台词一拍）
  拆分镜   每个镜头行标时长 [近景/推入/平视/2s]，单镜 ≤3s；总时长 ≈ 240s → 每集 ≥68 个镜头行
           + 前 15 秒（≈ 前 5 镜）行首标「【高潮点】」
  提示词   一段 = 一次生成 10–15s，段内是多镜头快切：cuts 字段列出各镜头秒数、每个 ≤3s、
           加起来 = video_duration；总时长 ≈ 240s（±15%）+ 开场 15s 内的段标 "hook": true
  渲染     提示词最前面加多镜头快切硬约束 + 时间线；生成后 ffmpeg 场景切换检测量最长镜头，
           超 3 秒重生成一次（domain/media/fast_cut.py、drama.cut_gate）

每一步都有确定性检查（check_*），不合格由调用方让模型改一次再存。
数值来自 config/drama.yaml，改配置不改代码。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..media.naming import parse_scene
from .models import Episode, ShotPrompt, _is_slug

HOOK_MARK = "【高潮点】"  # 分镜脚本里高潮镜头的行首标记
HOOK_LINE = "⚡"  # 剧本标题下的高潮点说明行标记
DEFAULT_MAX_CUT = 3.0  # 单镜上限（秒），config/drama.yaml cut.max_seconds 没写时用它

_CONFIG_DIR = Path(__file__).resolve().parents[3].parent / "config"


@dataclass(frozen=True)
class EpisodeFormat:
    minutes: float = 4.0
    tolerance: float = 0.15
    chars_per_minute: int = 320
    hook_seconds: int = 15
    min_shot_seconds: int = 10
    max_shot_seconds: int = 15
    # 单个镜头（分镜行 / 段内切镜）的时长上下限。2026-09-20 用户定的硬性要求：≤3 秒
    max_cut_seconds: float = DEFAULT_MAX_CUT
    min_cut_seconds: float = 1.0

    # ---------- 派生 ----------

    @property
    def seconds(self) -> int:
        return int(round(self.minutes * 60))

    @property
    def script_chars(self) -> int:
        return int(self.minutes * self.chars_per_minute)

    @property
    def script_range(self) -> tuple[int, int]:
        """剧本字数允许区间：目标的 75%–130%。"""
        return int(self.script_chars * 0.75), int(self.script_chars * 1.3)

    @property
    def duration_range(self) -> tuple[int, int]:
        lo = int(round(self.seconds * (1 - self.tolerance)))
        hi = int(round(self.seconds * (1 + self.tolerance)))
        return lo, hi

    @property
    def shot_range(self) -> tuple[int, int]:
        """每集**段**数范围（一段 = 一次生成，10–15s）：总时长除以单段上限/下限。"""
        lo = math.ceil(self.seconds / max(1, self.max_shot_seconds))
        hi = math.ceil(self.seconds / max(1, self.min_shot_seconds))
        return lo, max(lo, hi)

    @property
    def cut_line_range(self) -> tuple[int, int]:
        """每集**镜头行**数允许区间：总时长（含偏差）除以单镜上限 / 下限。"""
        lo_t, hi_t = self.duration_range
        lo = math.ceil(lo_t / max(0.1, self.max_cut_seconds))
        hi = math.ceil(hi_t / max(0.1, self.min_cut_seconds))
        return lo, max(lo, hi)

    @property
    def cut_line_typical(self) -> tuple[int, int]:
        """给模型看的镜头行数参考值：全按 3 秒切 → 80 个；平均 2 秒 → 120 个。"""
        lo = math.ceil(self.seconds / max(0.1, self.max_cut_seconds))
        hi = math.ceil(self.seconds / max(0.1, min(self.max_cut_seconds, 2.0)))
        return lo, max(lo, hi)

    @property
    def hook_shots(self) -> int:
        """开场高潮点要落在前几个镜头里：15s / 单镜上限 3s → 前 5 镜。"""
        return max(1, math.ceil(self.hook_seconds / max(0.1, self.max_cut_seconds)))

    def cuts_per_segment(self, seconds: float) -> int:
        """一段 seconds 秒的视频至少要几个镜头：每镜 ≤3s → 10s 至少 4 个、15s 至少 5 个。"""
        return max(1, math.ceil(float(seconds) / max(0.1, self.max_cut_seconds) - 1e-9))

    @classmethod
    def load(cls, path: str | Path) -> EpisodeFormat:
        p = Path(path)
        if not p.exists():
            return cls()
        raw: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        ep = raw.get("episode") or {}
        hook = raw.get("hook") or {}
        shot = raw.get("shot") or {}
        cut = raw.get("cut") or {}
        return cls(
            minutes=float(ep.get("minutes") or 4.0),
            tolerance=float(ep.get("tolerance") or 0.15),
            chars_per_minute=int(ep.get("chars_per_minute") or 320),
            hook_seconds=int(hook.get("seconds") or 15),
            min_shot_seconds=int(shot.get("min_seconds") or 10),
            max_shot_seconds=int(shot.get("max_seconds") or 15),
            max_cut_seconds=float(cut.get("max_seconds") or DEFAULT_MAX_CUT),
            min_cut_seconds=float(cut.get("min_seconds") or 1.0),
        )

    def brief(self) -> str:
        lo, hi = self.shot_range
        return (
            f"一集 {self.minutes:g} 分钟（约 {self.script_chars} 字，{lo}–{hi} 段）；"
            f"单镜 ≤{self.max_cut_seconds:g} 秒；开场 {self.hook_seconds} 秒内给高潮点"
        )


DEFAULT_FORMAT = EpisodeFormat()


def global_max_cut(config_dir: str | Path | None = None) -> float:
    """全局单镜上限（秒）—— 短剧链和配方短视频链共用同一个数，改 config/drama.yaml 即可。"""
    return EpisodeFormat.load(Path(config_dir or _CONFIG_DIR) / "drama.yaml").max_cut_seconds


# ---------------------------------------------------------------- 规则文本（追加，不改原提示词）


def writing_rules(fmt: EpisodeFormat) -> str:
    lo, hi = fmt.script_range
    return (
        "【一集的规格（硬性）】\n"
        f"· 一集成片 {fmt.minutes:g} 分钟：剧本约 {fmt.script_chars} 字（含台词与动作，"
        f"允许 {lo}–{hi} 字）。少了会被要求扩写，多了会被截。\n"
        f"· 开场 {fmt.hook_seconds} 秒必须是高潮点：第一个场次的前 2–3 个镜头就是本集冲突的"
        "顶点、反转或危机现场 —— 冷开场直接进入，铺垫后置回叙，不交代背景。"
        f"在标题下面加一行「> {HOOK_LINE} 前{fmt.hook_seconds}秒高潮点：一句话说清是什么」。\n"
        "· 节奏按短视频平台来：每 30–40 秒一个小反转或信息增量，连续一分钟没有新信息就算失败；"
        "结尾断在信息缺口上。\n"
        f"· 成片每个镜头不超过 {fmt.max_cut_seconds:g} 秒：动作写成可切的短拍 —— 一个动作、"
        "一个反应、一句短台词就是一拍；台词按短句断开，不要一句话说十秒，不要大段静态对话。"
    )


def storyboard_rules(fmt: EpisodeFormat) -> str:
    lo, _ = fmt.cut_line_range
    t_lo, t_hi = fmt.cut_line_typical
    mc = fmt.max_cut_seconds
    return (
        "【时长与节奏（硬性）】\n"
        f"· **每个镜头不超过 {mc:g} 秒**（最短 {fmt.min_cut_seconds:g} 秒）：一个镜头 = 一个动作、"
        f"一个反应或一句短台词；超过 {mc:g} 秒的内容拆成多个镜头（换机位 / 景别 / 角度），"
        "长台词按语速切到多个镜头里。\n"
        "· 每个镜头行的方括号里**最后一项写时长**，如 [近景/推入/平视/2s]、[特写/固定/俯拍/1.5s]；"
        "没标时长或超过上限都算不合格。\n"
        f"· 每集分镜总时长 ≈ {fmt.seconds} 秒（±{int(fmt.tolerance * 100)}%），"
        f"所以每集至少 {lo} 个镜头行（一般 {t_lo}–{t_hi} 个）。不要一集只拆十几个大镜头。\n"
        f"· 开场 {fmt.hook_seconds} 秒内必须出现高潮点：把本集冲突顶点、反转或危机的画面放在"
        f"最前面的镜头里（前 {fmt.hook_seconds} 秒 ≈ 前 {fmt.hook_shots} 镜），"
        f"这些镜头的行首加标记「{HOOK_MARK}」（放在景别方括号之前），铺垫后置。\n"
        "· 每 30–40 秒要有一个反转或信息增量，不要连续三个镜头都是走位与环境。"
    )


def shots_rules(fmt: EpisodeFormat) -> str:
    lo, hi = fmt.duration_range
    mc = fmt.max_cut_seconds
    n10 = fmt.cuts_per_segment(fmt.min_shot_seconds)
    n15 = fmt.cuts_per_segment(fmt.max_shot_seconds)
    return (
        "【时长与节奏（硬性）】\n"
        f"· video_duration 每段 {fmt.min_shot_seconds}–{fmt.max_shot_seconds}s"
        f"（生成模型单段上限 {fmt.max_shot_seconds}s）；一集所有段加起来 ≈ {fmt.seconds}s"
        f"（允许 {lo}–{hi}s）。分镜脚本里的镜头全部覆盖，不要合并成几段长的。\n"
        f"· **每段是多镜头快切：段内每个镜头不超过 {mc:g} 秒**，所以 {fmt.min_shot_seconds}s 的段"
        f"至少 {n10} 个镜头、{fmt.max_shot_seconds}s 的段至少 {n15} 个。video_name 写清覆盖的"
        '分镜脚本镜头范围（如 "9-13"），并给字段 "cuts": [3, 2, 3, 2, 2] —— 各镜头的秒数、'
        f"按顺序、每个 ≤ {mc:g}、加起来 = video_duration。description 按 cuts 的顺序用"
        f"[切镜：…] 逐镜写，任何一个镜头都不能拖过 {mc:g} 秒。\n"
        f"· 开场 {fmt.hook_seconds} 秒内必须出现高潮点：第一段（必要时前两段）就是冲突顶点、"
        '反转或危机画面；在这些对象里加字段 "hook": true。开场没有高潮点视为不合格。'
    )


# ---------------------------------------------------------------- 确定性检查


def shot_lines(desc: str) -> list[str]:
    """分镜脚本正文里的镜头行（以 [ 开头、不是场景标头）。"""
    out: list[str] = []
    for line in (desc or "").splitlines():
        s = line.strip()
        if not s:
            continue
        head = s.split(HOOK_MARK, 1)[-1].strip() if s.startswith(HOOK_MARK) else s
        if head.startswith("[") and not _is_slug(head):
            out.append(s)
    return out


_FIRST_BRACKET = re.compile(r"^\[([^\]]*)\]")
_SECONDS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:s|S|秒)")


def shot_seconds(line: str) -> float | None:
    """镜头行的时长：第一个方括号里的「2s / 1.5秒」。没标返回 None。"""
    s = (line or "").strip()
    if s.startswith(HOOK_MARK):
        s = s.split(HOOK_MARK, 1)[-1].strip()
    m = _FIRST_BRACKET.match(s)
    if not m:
        return None
    found = _SECONDS.findall(m.group(1))
    if not found:
        return None
    try:
        return float(found[-1])
    except ValueError:
        return None


def _shot_label(line: str, limit: int = 24) -> str:
    s = line.strip()
    if s.startswith(HOOK_MARK):
        s = s.split(HOOK_MARK, 1)[-1].strip()
    return s[:limit] + ("…" if len(s) > limit else "")


def check_storyboard(ep: Episode, fmt: EpisodeFormat) -> list[str]:
    """一集分镜脚本的规格问题（空列表 = 合格）：
    每个镜头行标了时长且 ≤ 上限、总时长在区间内、镜头行数够、开场高潮点。"""
    problems: list[str] = []
    lines = shot_lines(ep.desc)
    secs = [shot_seconds(ln) for ln in lines]
    mc, mn = fmt.max_cut_seconds, fmt.min_cut_seconds

    missing = [ln for ln, s in zip(lines, secs, strict=False) if s is None]
    if missing:
        problems.append(
            f"{ep.title}：{len(missing)} 个镜头行没标时长（方括号最后一项写 2s 这种），"
            f"如「{_shot_label(missing[0])}」"
        )
    over = [(ln, s) for ln, s in zip(lines, secs, strict=False) if s is not None and s > mc]
    if over:
        longest = max(s for _, s in over)
        shown = "；".join(_shot_label(ln) for ln, _ in over[:3])
        problems.append(
            f"{ep.title}：{len(over)} 个镜头超过 {mc:g} 秒（最长 {longest:g}s），"
            f"必须拆成多个镜头：{shown}"
        )
    under = [s for s in secs if s is not None and s < mn]
    if under:
        problems.append(f"{ep.title}：{len(under)} 个镜头短于 {mn:g} 秒，看不清，合并或加长")

    lo_n, _ = fmt.cut_line_range
    known = [s for s in secs if s is not None]
    if known and len(known) >= 0.8 * max(1, len(lines)):
        total = sum(known)
        lo_t, hi_t = fmt.duration_range
        if total < lo_t or total > hi_t:
            problems.append(
                f"{ep.title}：镜头时长加起来 {total:g}s，应在 {lo_t}–{hi_t}s（一集 {fmt.seconds}s）"
            )
    elif len(lines) < lo_n:
        problems.append(
            f"{ep.title}：{len(lines)} 个镜头行，按单镜 ≤{mc:g}s、一集 {fmt.seconds}s "
            f"至少要 {lo_n} 个"
        )

    # 开场高潮点：有时长就按累计时间算前 15 秒，没有就按前 hook_shots 行
    opening: list[str] = []
    t = 0.0
    for ln, s in zip(lines, secs, strict=False):
        if s is None:
            if len(opening) >= fmt.hook_shots:
                break
        elif t >= fmt.hook_seconds:
            break
        opening.append(ln)
        t += s if s is not None else mc
    if not any(HOOK_MARK in ln or "高潮点" in ln for ln in opening):
        problems.append(
            f"{ep.title}：开场 {fmt.hook_seconds} 秒内没有高潮点"
            f"（前 {fmt.hook_seconds} 秒 ≈ 前 {fmt.hook_shots} 个镜头行，行首要有{HOOK_MARK}，"
            "且画面就是冲突顶点/反转/危机）"
        )
    return problems


def check_shots(shots: list[ShotPrompt], fmt: EpisodeFormat) -> list[str]:
    """视频提示词的规格问题：按集算总时长、单段时长、段内镜头（cuts）、开场高潮点。"""
    by_ep: dict[int, list[ShotPrompt]] = {}
    for s in shots:
        ep, _ = parse_scene(s.scene_index)
        by_ep.setdefault(ep, []).append(s)
    lo, hi = fmt.duration_range
    mc = fmt.max_cut_seconds
    problems: list[str] = []
    for ep, items in by_ep.items():
        label = f"第{ep}集" if ep else "未标集号"
        total = sum(s.seconds for s in items)
        if total < lo or total > hi:
            problems.append(f"{label}：总时长 {total}s，应在 {lo}–{hi}s（一集 {fmt.seconds}s）")
        lo_s, hi_s = fmt.min_shot_seconds, fmt.max_shot_seconds
        bad = [s for s in items if s.seconds < lo_s or s.seconds > hi_s]
        if bad:
            problems.append(
                f"{label}：{len(bad)} 段的 video_duration 不在 {fmt.min_shot_seconds}–"
                f"{fmt.max_shot_seconds}s（{', '.join(s.video_name for s in bad[:5])}）"
            )

        # 段内镜头：cuts 每个 ≤ 上限、加起来 = 段长；没给 cuts 的至少要 video_name 覆盖够镜头数
        no_cuts = [s for s in items if not s.cuts]
        if no_cuts:
            problems.append(
                f"{label}：{len(no_cuts)} 段没给 cuts（各镜头秒数，每个 ≤ {mc:g}s，"
                f"加起来 = video_duration）：{', '.join(s.video_name for s in no_cuts[:5])}"
            )
        too_long = [s for s in items if s.cuts and max(s.cuts) > mc + 1e-9]
        if too_long:
            worst = max(max(s.cuts) for s in too_long)
            problems.append(
                f"{label}：{len(too_long)} 段的 cuts 里有超过 {mc:g}s 的镜头（最长 {worst:g}s），"
                f"每个镜头都要 ≤ {mc:g}s：{', '.join(s.video_name for s in too_long[:5])}"
            )
        mismatch = [s for s in items if s.cuts and abs(sum(s.cuts) - s.seconds) > 1.0]
        if mismatch:
            shown = ", ".join(
                f"{s.video_name}({sum(s.cuts):g}s≠{s.seconds}s)" for s in mismatch[:5]
            )
            problems.append(
                f"{label}：{len(mismatch)} 段的 cuts 加起来不等于 video_duration：{shown}"
            )
        few = [
            s for s in no_cuts
            if s.shot_count and s.shot_count < fmt.cuts_per_segment(s.seconds)
        ]
        if few:
            shown = ", ".join(
                f"{s.video_name}({s.seconds}s 只有 {s.shot_count} 镜，至少 "
                f"{fmt.cuts_per_segment(s.seconds)} 镜)"
                for s in few[:5]
            )
            problems.append(f"{label}：{len(few)} 段覆盖的镜头太少，每镜 ≤{mc:g}s 切不够：{shown}")

        t = 0
        opening: list[ShotPrompt] = []
        for s in items:
            if t < fmt.hook_seconds:
                opening.append(s)
            t += s.seconds
        if not any(s.hook for s in opening):
            names = ", ".join(s.video_name for s in opening)
            problems.append(
                f"{label}：开场 {fmt.hook_seconds} 秒内（镜 {names}）没有标 \"hook\": true 的高潮点"
            )
    return problems


_HOOK_LINE_RE = re.compile(rf"^\s*>?\s*{HOOK_LINE}", re.M)


def check_script(text: str, fmt: EpisodeFormat) -> list[str]:
    """一集剧本的规格问题：字数区间、有没有写开场高潮点。"""
    problems: list[str] = []
    n = len(text or "")
    lo, hi = fmt.script_range
    if n < lo:
        problems.append(
            f"只有 {n} 字，一集 {fmt.minutes:g} 分钟需要约 {fmt.script_chars} 字（至少 {lo}）"
        )
    elif n > hi:
        problems.append(f"{n} 字，超出一集 {fmt.minutes:g} 分钟的上限 {hi} 字，成片会拖长")
    if not _HOOK_LINE_RE.search(text or ""):
        problems.append(
            f"没有写「> {HOOK_LINE} 前{fmt.hook_seconds}秒高潮点」这一行 —— 开场必须直接是高潮点"
        )
    return problems

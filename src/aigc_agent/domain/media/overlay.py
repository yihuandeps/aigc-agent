"""视频上屏文字（slogan / 卖点 / 片尾字卡 / 地点卡）—— 后期叠，不让生成模型画。

为什么后期叠：生成模型画中文几乎必错，改一个字就得重生成一段；用户定的最高优先级规则
也是「生成的画面不许有字」（2026-09-18）。广告的 slogan、英雄帧上的字卡、剧情里的地点卡
都在这一步用 ASS 字幕叠上去：本地 ffmpeg，不花钱，改字重跑几秒钟。

2026-09-26 审查：广告配方和指引一直写着「文字后期加」「英雄帧留给后期上字」，但系统里
能往**视频**上叠字的工具一个都没有（make_poster 只处理图片）—— 投放级广告成片没有 slogan。

这里只管排版（纯函数，好测）；烧进视频在 ffmpeg.burn_ass。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Windows 自带；libass 按名字找系统字体（2026-09-26 本机 ffmpeg 实测能渲中文）
FONT = "Microsoft YaHei"

# 位置 → (ASS 对齐码, 垂直边距占画面高的比例)。对齐码按小键盘：8 上 / 5 中 / 2 下
POSITIONS: dict[str, tuple[int, float]] = {
    "top": (8, 0.08),
    "center": (5, 0.0),
    "bottom": (2, 0.10),
    "lower_third": (2, 0.24),
}
# 字号 → 占画面高的比例（竖屏 1920 高时 l ≈ 119px）
SIZES: dict[str, float] = {"s": 0.032, "m": 0.045, "l": 0.062, "xl": 0.085}


@dataclass
class TextItem:
    text: str
    start: float = 0.0
    end: float = 0.0  # 0 = 到片尾
    position: str = "center"
    size: str = "l"
    box: bool = False  # 半透明底框（画面花的时候字更清楚）


def parse_items(
    items: list[dict[str, Any]] | None,
    end_card: dict[str, Any] | None,
    duration: float,
) -> tuple[list[TextItem], list[str]]:
    """工具参数 → 文字条目。返回 (条目, 纠正说明)。时间夹到片长以内。"""
    out: list[TextItem] = []
    notes: list[str] = []
    for n, raw in enumerate(items or [], 1):
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        start = _num(raw.get("start"), 0.0)
        end = _num(raw.get("end"), 0.0)
        pos = str(raw.get("position") or "center")
        size = str(raw.get("size") or "l")
        if pos not in POSITIONS:
            notes.append(f"第{n}条位置 {pos!r} 不认识，按居中")
            pos = "center"
        if size not in SIZES:
            notes.append(f"第{n}条字号 {size!r} 不认识，按 l")
            size = "l"
        out.append(TextItem(text, start, end, pos, size, bool(raw.get("box"))))
    if end_card:
        title = str(end_card.get("title") or "").strip()
        sub = str(end_card.get("subtitle") or "").strip()
        secs = _num(end_card.get("seconds"), 2.5) or 2.5
        start = max(0.0, duration - secs) if duration else 0.0
        pos = str(end_card.get("position") or "lower_third")
        if pos not in POSITIONS:
            pos = "lower_third"
        if title:
            out.append(TextItem(title, start, 0.0, pos, "xl"))
        if sub:
            out.append(TextItem(sub, start + 0.3, 0.0, "bottom", "m"))
    for it in out:
        if duration:
            it.start = min(max(0.0, it.start), max(0.0, duration - 0.2))
            if not it.end or it.end > duration:
                it.end = duration
        if it.end and it.end <= it.start:
            notes.append(f"「{it.text[:10]}」结束早于开始，改成到片尾")
            it.end = duration or it.start + 2.0
    return out, notes


def build_ass(items: list[TextItem], width: int, height: int) -> str:
    """文字条目 → ASS 文件内容。画布 = 视频实际尺寸，字号 / 边距按画面高的比例算。"""
    w = max(2, int(width or 1080))
    h = max(2, int(height or 1920))
    styles: list[str] = []
    events: list[str] = []
    for n, it in enumerate(items):
        align, margin = POSITIONS.get(it.position, POSITIONS["center"])
        fs = max(12, round(h * SIZES.get(it.size, SIZES["l"])))
        bold = -1 if it.size in ("l", "xl") else 0
        if it.box:
            # BorderStyle 3 = 不透明底框（颜色取 OutlineColour）
            border, outline, shadow, back = 3, max(6, fs // 5), 0, "&H00000000"
            out_colour = "&H66000000"
        else:
            border, outline, shadow, back = 1, max(2, fs // 22), 1, "&H80000000"
            out_colour = "&H00000000"
        side = round(w * 0.06)
        styles.append(
            f"Style: S{n},{FONT},{fs},&H00FFFFFF,&H000000FF,{out_colour},{back},"
            f"{bold},0,0,0,100,100,0,0,{border},{outline},{shadow},{align},"
            f"{side},{side},{round(h * margin)},1"
        )
        events.append(
            f"Dialogue: 0,{_ts(it.start)},{_ts(it.end)},S{n},,0,0,0,,"
            f"{{\\fad(250,250)}}{_escape(it.text)}"
        )
    return (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {w}\nPlayResY: {h}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        + "\n".join(styles)
        + "\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        + "\n".join(events)
        + "\n"
    )


def _ts(sec: float) -> str:
    """秒 → ASS 时间 H:MM:SS.cc。"""
    cs = max(0, round(float(sec) * 100))
    h, rem = divmod(cs, 360_000)
    m, rem = divmod(rem, 6_000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _escape(text: str) -> str:
    """正文里的反斜杠和花括号会被 ASS 当成控制码：换成全角；换行写成 \\N。"""
    t = text.replace("\\", "＼").replace("{", "｛").replace("}", "｝")
    return "\\N".join(line.strip() for line in t.replace("\r", "").split("\n"))


def _num(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

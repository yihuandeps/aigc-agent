"""海报 / 图文卡片渲染（Pillow，本地、不花钱、不出网）。

图文笔记要的是"一张有字的图"：封面写标题，内页写要点。生图模型写不好中文，
所以字是本地叠上去的 —— 底图可以是生图结果、素材库的图，或者不给底图用渐变。

字体从系统里找（Windows 微软雅黑 / macOS 苹方 / Linux Noto CJK），也可用环境变量
AIGC_POSTER_FONT 指定一个 ttf/ttc/otf。找不到中文字体就退回 Pillow 自带字体
（英文能看，中文会是方块），结果的 warnings 里会说 —— 不静默。
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

TEMPLATES: dict[str, str] = {
    "clean": "底图 + 下方白色信息条，深色标题；封面常用",
    "bold": "底图压暗、大字居中；强标题封面",
    "card": "底图虚化做背景、中间白卡片承载标题与要点；内页常用",
}

SIZES: dict[str, tuple[int, int]] = {
    "3:4": (1080, 1440),
    "1:1": (1080, 1080),
    "9:16": (1080, 1920),
    "4:3": (1440, 1080),
    "16:9": (1920, 1080),
}

FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
)

RGB = tuple[int, int, int]
DEFAULT_ACCENT: RGB = (255, 87, 51)
INK: tuple[int, int, int, int] = (31, 35, 40, 255)
DIM: tuple[int, int, int, int] = (107, 114, 128, 255)


# ---------------------------------------------------------------- 字体 / 排版


def find_font() -> Path | None:
    env = os.environ.get("AIGC_POSTER_FONT", "").strip()
    if env and Path(env).exists():
        return Path(env)
    for c in FONT_CANDIDATES:
        if Path(c).exists():
            return Path(c)
    return None


def load_font(path: Path | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if path is not None:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # 老 Pillow 的默认字体不认 size
        return ImageFont.load_default()


_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*\s*|.")


def wrap(draw: ImageDraw.ImageDraw, text: str, font: object, max_width: float) -> list[str]:
    """按像素宽换行。中文没空格：逐字累积；英文按单词；显式换行保留。"""
    lines: list[str] = []
    for para in text.replace("\r", "").split("\n"):
        cur = ""
        for token in _TOKEN.findall(para):
            trial = cur + token
            if not cur or draw.textlength(trial.rstrip(), font=font) <= max_width:
                cur = trial
            else:
                lines.append(cur.rstrip())
                cur = token.lstrip()
        lines.append(cur.rstrip())
    return lines


def parse_color(value: str, default: RGB) -> RGB:
    """#rrggbb 或 r,g,b；空用默认。"""
    v = (value or "").strip()
    if not v:
        return default
    if v.startswith("#") and len(v) == 7:
        try:
            return int(v[1:3], 16), int(v[3:5], 16), int(v[5:7], 16)
        except ValueError:
            pass
    parts = [p for p in re.split(r"[,\s]+", v) if p]
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        r, g, b = (min(255, int(p)) for p in parts)
        return r, g, b
    raise ValueError(f"颜色写成 #rrggbb 或 r,g,b：{value!r}")


def _mix(a: RGB, b: RGB, t: float) -> RGB:
    r, g, bl = (round(a[i] * (1 - t) + b[i] * t) for i in range(3))
    return r, g, bl


def _gradient(size: tuple[int, int], top: RGB, bottom: RGB) -> Image.Image:
    w, h = size
    strip = Image.new("RGB", (1, h))
    px = strip.load()
    for y in range(h):
        px[0, y] = _mix(top, bottom, y / max(1, h - 1))
    return strip.resize((w, h))


def _cover_fit(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """等比放大到盖满画布再居中裁切 —— 底图比例不对也不变形。"""
    w, h = size
    scale = max(w / img.width, h / img.height)
    new = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    resized = img.resize(new, Image.Resampling.LANCZOS)
    left = (resized.width - w) // 2
    top = (resized.height - h) // 2
    return resized.crop((left, top, left + w, top + h))


# ---------------------------------------------------------------- 规格 / 结果


@dataclass
class PosterSpec:
    title: str
    subtitle: str = ""
    template: str = "clean"
    aspect_ratio: str = "3:4"
    brand: str = ""
    accent: RGB = DEFAULT_ACCENT
    background: bytes | None = None
    aigc_label: str = "AI 生成"  # 角标；空则不打。平台都要求显式标识
    max_title_lines: int = 3


@dataclass
class PosterResult:
    png: bytes
    width: int
    height: int
    template: str
    font: str
    title_lines: int
    warnings: list[str] = field(default_factory=list)


def render(spec: PosterSpec) -> PosterResult:
    if spec.template not in TEMPLATES:
        raise ValueError(f"没有模板 {spec.template!r}，可选：{', '.join(TEMPLATES)}")
    if not spec.title.strip():
        raise ValueError("标题不能为空")
    size = SIZES.get(spec.aspect_ratio)
    if size is None:
        raise ValueError(f"比例只支持 {', '.join(SIZES)}，给的是 {spec.aspect_ratio!r}")

    warnings: list[str] = []
    font_path = find_font()
    if font_path is None:
        warnings.append(
            "没找到中文字体，用了 Pillow 自带字体（中文会是方块）；设 AIGC_POSTER_FONT 指定一个"
        )

    img = _base(spec, size)
    if spec.template == "clean":
        lines = _draw_clean(img, spec, font_path)
    elif spec.template == "bold":
        lines = _draw_bold(img, spec, font_path)
    else:
        lines = _draw_card(img, spec, font_path)
    _draw_label(img, spec, font_path)

    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG", optimize=True)
    return PosterResult(
        png=buf.getvalue(),
        width=size[0],
        height=size[1],
        template=spec.template,
        font=font_path.name if font_path else "default",
        title_lines=lines,
        warnings=warnings,
    )


# ---------------------------------------------------------------- 各模板


def _base(spec: PosterSpec, size: tuple[int, int]) -> Image.Image:
    if spec.background:
        try:
            src = Image.open(io.BytesIO(spec.background))
            src.load()
        except Exception as e:  # noqa: BLE001 — Pillow 抛的类型不止一种
            raise ValueError(f"底图不是能识别的图片：{type(e).__name__}") from e
        img = _cover_fit(src.convert("RGB"), size)
        if spec.template == "card":
            img = img.filter(ImageFilter.GaussianBlur(radius=max(4, size[0] // 90)))
        return img.convert("RGBA")
    light = _mix(spec.accent, (255, 255, 255), 0.82)
    dark = _mix(spec.accent, (20, 20, 30), 0.65)
    if spec.template == "bold":
        return _gradient(size, dark, _mix(dark, spec.accent, 0.35)).convert("RGBA")
    return _gradient(size, light, _mix(light, spec.accent, 0.35)).convert("RGBA")


def _fit(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: Path | None,
    max_width: float,
    start: int,
    minimum: int,
    max_lines: int,
) -> tuple[object, list[str], int]:
    """标题先按大字号排，行数超了就缩，缩到下限为止。"""
    size = start
    while True:
        font = load_font(font_path, size)
        lines = wrap(draw, text, font, max_width)
        if len(lines) <= max_lines or size <= minimum:
            return font, lines, size
        size = max(minimum, int(size * 0.88))


def _dim(img: Image.Image, rgba: tuple[int, int, int, int]) -> None:
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle((0, 0, img.width, img.height), fill=rgba)
    img.alpha_composite(overlay)


def _draw_clean(img: Image.Image, spec: PosterSpec, font_path: Path | None) -> int:
    w, h = img.size
    m = int(w * 0.07)
    draw = ImageDraw.Draw(img, "RGBA")
    inner = w - 2 * m - int(w * 0.03)
    font, lines, size = _fit(
        draw, spec.title, font_path, inner, w // 11, w // 24, spec.max_title_lines
    )
    line_h = int(size * 1.25)
    sub_font = load_font(font_path, max(18, size // 2))
    sub_lines = wrap(draw, spec.subtitle, sub_font, inner)[:3] if spec.subtitle.strip() else []
    sub_h = int(size // 2 * 1.45)
    brand_h = int(size * 0.9) if spec.brand else 0

    block = len(lines) * line_h + (int(size * 0.3) + len(sub_lines) * sub_h if sub_lines else 0)
    panel_h = m + block + brand_h + m // 2
    top = h - panel_h - m // 2
    draw.rounded_rectangle(
        (m // 2, top, w - m // 2, h - m // 2), radius=int(w * 0.03), fill=(255, 255, 255, 235)
    )
    y = top + m // 2 + int(size * 0.1)
    bar_w = max(6, int(w * 0.012))
    draw.rounded_rectangle(
        (m, y + int(size * 0.15), m + bar_w, y + len(lines) * line_h - int(size * 0.15)),
        radius=bar_w // 2,
        fill=(*spec.accent, 255),
    )
    x = m + int(w * 0.03)
    for ln in lines:
        draw.text((x, y), ln, font=font, fill=INK)
        y += line_h
    if sub_lines:
        y += int(size * 0.3)
        for ln in sub_lines:
            draw.text((x, y), ln, font=sub_font, fill=DIM)
            y += sub_h
    if spec.brand:
        bf = load_font(font_path, max(16, size // 3))
        draw.text((x, h - m - int(size * 0.55)), spec.brand, font=bf, fill=(*spec.accent, 255))
    return len(lines)


def _draw_bold(img: Image.Image, spec: PosterSpec, font_path: Path | None) -> int:
    w, h = img.size
    _dim(img, (10, 10, 20, 120))
    draw = ImageDraw.Draw(img, "RGBA")
    m = int(w * 0.09)
    inner = w - 2 * m
    font, lines, size = _fit(
        draw, spec.title, font_path, inner, w // 9, w // 20, spec.max_title_lines
    )
    line_h = int(size * 1.2)
    sub_font = load_font(font_path, max(18, size // 2))
    sub_lines = wrap(draw, spec.subtitle, sub_font, inner)[:3] if spec.subtitle.strip() else []
    sub_h = int(size // 2 * 1.45)
    block = len(lines) * line_h + (int(size * 0.5) + len(sub_lines) * sub_h if sub_lines else 0)
    y = (h - block) // 2
    for ln in lines:
        tw = draw.textlength(ln, font=font)
        draw.text(((w - tw) / 2, y), ln, font=font, fill=(255, 255, 255, 255))
        y += line_h
    ul = int(w * 0.12)
    draw.rounded_rectangle(
        ((w - ul) // 2, y, (w + ul) // 2, y + max(4, int(w * 0.008))),
        radius=3,
        fill=(*spec.accent, 255),
    )
    if sub_lines:
        y += int(size * 0.5)
        for ln in sub_lines:
            tw = draw.textlength(ln, font=sub_font)
            draw.text(((w - tw) / 2, y), ln, font=sub_font, fill=(230, 230, 235, 255))
            y += sub_h
    if spec.brand:
        bf = load_font(font_path, max(16, size // 3))
        tw = draw.textlength(spec.brand, font=bf)
        pos = ((w - tw) / 2, h - m - int(size * 0.4))
        draw.text(pos, spec.brand, font=bf, fill=(255, 255, 255, 200))
    return len(lines)


def _draw_card(img: Image.Image, spec: PosterSpec, font_path: Path | None) -> int:
    w, h = img.size
    _dim(img, (255, 255, 255, 60))
    draw = ImageDraw.Draw(img, "RGBA")
    cm = int(w * 0.08)
    pad = int(w * 0.07)
    card = (cm, int(h * 0.12), w - cm, int(h * 0.88))
    draw.rounded_rectangle(card, radius=int(w * 0.035), fill=(255, 255, 255, 242))
    inner = (card[2] - card[0]) - 2 * pad
    font, lines, size = _fit(draw, spec.title, font_path, inner, w // 13, w // 26, 2)
    line_h = int(size * 1.25)
    x, y = card[0] + pad, card[1] + pad
    for ln in lines:
        draw.text((x, y), ln, font=font, fill=INK)
        y += line_h
    y += int(size * 0.3)
    draw.rounded_rectangle(
        (x, y, x + int(w * 0.1), y + max(4, int(w * 0.006))), radius=3, fill=(*spec.accent, 255)
    )
    y += int(size * 0.6)
    if spec.subtitle.strip():
        body_size = max(20, int(size * 0.55))
        bf = load_font(font_path, body_size)
        body_h = int(body_size * 1.6)
        room = max(1, (card[3] - pad - int(size * 0.8) - y) // body_h)
        for ln in wrap(draw, spec.subtitle, bf, inner)[:room]:
            draw.text((x, y), ln, font=bf, fill=(55, 65, 81, 255))
            y += body_h
    if spec.brand:
        brf = load_font(font_path, max(16, size // 3))
        tw = draw.textlength(spec.brand, font=brf)
        draw.text(
            (card[2] - pad - tw, card[3] - pad - int(size * 0.45)),
            spec.brand,
            font=brf,
            fill=(*spec.accent, 255),
        )
    return len(lines)


def _draw_label(img: Image.Image, spec: PosterSpec, font_path: Path | None) -> None:
    """右上角 AIGC 角标 —— 平台都要求画面显式标识，别让人上传前再补。"""
    label = spec.aigc_label.strip()
    if not label:
        return
    w, _ = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    fs = max(14, w // 45)
    f = load_font(font_path, fs)
    tw = draw.textlength(label, font=f)
    ph = int(fs * 1.9)
    pw = int(tw + w * 0.03)
    x1, y0 = w - int(w * 0.04), int(w * 0.04)
    draw.rounded_rectangle((x1 - pw, y0, x1, y0 + ph), radius=ph // 2, fill=(0, 0, 0, 110))
    pos = (x1 - pw + int(w * 0.015), y0 + (ph - fs) // 2 - 2)
    draw.text(pos, label, font=f, fill=(255, 255, 255, 230))

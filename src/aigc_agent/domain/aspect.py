"""视频画幅（2026-09-25 用户要的：能出 16:9 横屏，比例按项目个性化选）。

一处定义、各环节共用：短剧渲染、短视频出片、gen_video 的默认值、素材站搜索方向、图片转镜头的
尺寸。项目的画幅记在会话快照里（CLI /ratio 设），没设时短剧默认竖屏 9:16、短视频按配方。
"""

from __future__ import annotations

import re
from typing import Any

from ..harness.model.media import MediaKind

DEFAULT_ASPECT = "9:16"
# 视频模型目录（config/media_models.yaml）里出现的几种；具体某个模型支持哪些看它的 aspect_ratios
VIDEO_ASPECTS = ("9:16", "16:9", "1:1")

_LABELS = {"9:16": "竖屏 9:16", "16:9": "横屏 16:9", "1:1": "方形 1:1"}
_WORDS = {
    "竖屏": "9:16", "竖版": "9:16", "竖": "9:16", "vertical": "9:16", "portrait": "9:16",
    "横屏": "16:9", "横版": "16:9", "横": "16:9", "宽屏": "16:9", "horizontal": "16:9",
    "landscape": "16:9", "方形": "1:1", "正方形": "1:1", "方": "1:1", "square": "1:1",
}
_RATIO = re.compile(r"^\s*(\d{1,2})\s*[:：xX×/]\s*(\d{1,2})\s*$")


def parse_aspect(text: str) -> str:
    """「16:9」「16：9」「16x9」「横屏」→ 「16:9」；认不出或不在支持范围里返回空串。"""
    t = (text or "").strip().lower()
    if not t:
        return ""
    if t in _WORDS:
        return _WORDS[t]
    m = _RATIO.match(t)
    if not m:
        return ""
    ratio = f"{int(m.group(1))}:{int(m.group(2))}"
    return ratio if ratio in VIDEO_ASPECTS else ""


def aspect_label(ratio: str) -> str:
    """给人看的名字：「横屏 16:9」。"""
    return _LABELS.get(ratio, ratio or _LABELS[DEFAULT_ASPECT])


def orientation_of(ratio: str) -> str:
    """素材站搜索用的方向：portrait / landscape / square。"""
    return {"16:9": "landscape", "1:1": "square"}.get(ratio, "portrait")


def still_size(ratio: str) -> tuple[int, int]:
    """图片做成静止镜头的尺寸（720p 档）。"""
    return {"16:9": (1280, 720), "1:1": (720, 720)}.get(ratio, (720, 1280))


def supported_aspects(catalog: Any, model: str) -> list[str]:
    """视频模型目录里这个模型支持的画幅；目录没列就返回空（= 不知道，不拦）。"""
    get = getattr(catalog, "get", None)
    if not callable(get) or not model:
        return []
    try:
        spec = get(MediaKind.VIDEO, model)
    except Exception:  # noqa: BLE001 —— 目录查不到不该挡住生成
        return []
    return [str(x) for x in (getattr(spec, "aspect_ratios", None) or [])]


def unsupported_note(catalog: Any, model: str, ratio: str) -> str:
    """这个视频模型不支持这个画幅时的拦截说明；支持或不知道返回空串。"""
    listed = supported_aspects(catalog, model)
    if ratio and listed and ratio not in listed:
        return (
            f"视频模型 {model} 不支持画幅 {ratio}（支持 {' / '.join(listed)}），没有发起生成。"
            "换一个画幅（用户用 /ratio 改），或先征得用户同意换视频模型"
        )
    return ""

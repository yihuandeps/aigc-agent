"""视频画面里不许出现字幕/文字（2026-09-18，用户定的最高优先级约束）。

seedance 这类原生出声的视频模型，遇到提示词里有台词，时不时会把台词渲成画面底部的
字幕条，有时还带水印或标题。成片的字幕由我们自己的链路后期压（能控字体、位置、
双语），模型烧进画面的字幕既不可控也去不掉，所以必须在源头禁掉。

两道保险：
  1. **提示词**：所有 gen_video 调用统一在提示词最前面加一段最高优先级禁令，末尾再复述
     （allow_text=true 才不加）。
  2. **字幕门**（短剧渲染）：每段视频生成后抽几帧给视觉模型看，发现字幕/文字就用更硬的
     提示词重生成一次；还有就在结果里标出来交人复核。
"""

from __future__ import annotations

import json
import re
from typing import Any

NO_TEXT_HEAD = (
    "【最高优先级硬约束】画面中严禁出现任何字幕、台词字幕条、文字、标题、水印、Logo、贴纸或字符；"
    "所有台词只以声音呈现，绝不能以文字形式出现在画面里。此条优先级高于下文所有内容。 "
    "HIGHEST-PRIORITY RULE: absolutely NO subtitles, captions, on-screen text, titles, "
    "watermarks, logos or lettering anywhere in the frame; dialogue is audio only and must never "
    "be rendered as text. This rule overrides everything below."
)

NO_TEXT_TAIL = (
    "再次强调：画面中不得出现任何字幕或文字。 "
    "No subtitles, no captions, no on-screen text of any kind."
)

NO_TEXT_RETRY = (
    "【重新生成】上一版画面里出现了字幕/文字，属于不合格。这次画面中绝对不能有任何字幕、字幕条、"
    "文字或水印；台词只用声音表达。 "
    "REGENERATION — the previous output contained subtitles or on-screen text, which is forbidden. "
    "This time the frame must contain NO subtitles, captions, text or watermark of any kind; "
    "dialogue is audio only."
)


def no_text_prompt(prompt: str) -> str:
    """把禁字幕约束包在提示词最外层：开头一段（最高优先级）+ 末尾复述。幂等。"""
    p = (prompt or "").strip()
    if p.startswith(NO_TEXT_HEAD):
        return p
    return f"{NO_TEXT_HEAD}\n{p}\n{NO_TEXT_TAIL}"


def no_text_retry(prompt: str) -> str:
    """重生成用：在禁令之后、正文之前插一句"上一版出了字幕"。幂等。"""
    p = (prompt or "").strip()
    if NO_TEXT_RETRY in p:
        return p
    if p.startswith(NO_TEXT_HEAD):
        return f"{NO_TEXT_HEAD}\n{NO_TEXT_RETRY}\n{p[len(NO_TEXT_HEAD):].lstrip()}"
    return f"{NO_TEXT_RETRY}\n{p}"


_CHECK_PROMPT = (
    "这是同一段 AI 生成视频按时间均匀抽出的几帧画面。请判断画面里有没有**叠加在画面上**的"
    "字幕、台词字幕条、文字、标题、水印、Logo 或字符（画面底部/顶部的对白字幕、角落的水印、"
    "居中的标题都算；场景里本来就有的招牌、书页、屏幕上的文字不算）。\n"
    '只输出 JSON：{"text_found": false, "where": "哪一帧、什么位置、大概什么内容"}'
)


def subtitle_check_messages(frame_urls: list[str]) -> list[dict[str, Any]]:
    """给视觉模型的字幕检查消息。frame_urls 是抽帧的 data URL（或 https 链接）。"""
    content: list[dict[str, Any]] = [{"type": "text", "text": _CHECK_PROMPT}]
    content += [
        {"type": "image_url", "image_url": {"url": u, "detail": "low"}} for u in frame_urls
    ]
    return [{"role": "user", "content": content}]


def parse_subtitle_verdict(text: str) -> tuple[bool | None, str]:
    """解析检查结果：(是否发现文字, 说明)。解析不出来返回 (None, 原因)。"""
    raw = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if m:
        raw = m.group(1).strip()
    i, j = raw.find("{"), raw.rfind("}")
    if i < 0 or j <= i:
        return None, "字幕检查输出不是 JSON"
    try:
        data = json.loads(raw[i : j + 1])
    except json.JSONDecodeError:
        return None, "字幕检查输出不是合法 JSON"
    found = data.get("text_found")
    if not isinstance(found, bool):
        return None, "字幕检查没给出 text_found"
    return found, str(data.get("where") or "")

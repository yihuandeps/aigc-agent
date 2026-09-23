"""剧本 / 参考视频 → 分镜。

提示词骨架沿用原项目，但有三处**必须改**，照搬会坏：

1. **单镜时长**。原项目写死 10–15 秒，那是按即梦的规格来的。这边 veo3.1
   单次最长 8 秒、sora 12 秒，写 15 秒模型会按 15 秒的信息量去编排画面，
   生成出来必然是截断的。所以时长由调用方给。
2. **输出契约**。原项目让模型直接吐 JSON 数组、再用正则剥 markdown。
   这里保留数组格式（下游要按序消费），但解析放宽：模型偶尔会包一层
   {"shots": [...]}，也得认。
3. **旁白**。原项目的 narration 是"对应这个镜头的原文"。做短视频时旁白要
   能直接拿去配音，所以要求可念，不能是场记式的描述。

角色一致性的做法没动：护照逐镜注入，这是整个移植的核心价值。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..realism import prompt_rules
from .passport import Character, render_passport

# 画面提示词用英文逗号短语（ComfyUI / 即梦风格），旁白用中文 —— 沿用原项目。
# 这个组合是有道理的：生成模型对英文标签的响应比中文长句稳定得多，
# 而旁白是要念出来的，必须中文。
_PROMPT_STYLE = "English, comma-separated phrases"


@dataclass
class Shot:
    number: int
    prompt: str  # 英文画面提示词
    narration: str = ""  # 中文旁白
    duration: str = ""

    def line(self) -> str:
        head = f"第{self.number}镜" + (f"（{self.duration}）" if self.duration else "")
        body = f"  画面：{self.prompt}"
        return f"{head}\n{body}" + (f"\n  旁白：{self.narration}" if self.narration else "")


def system_prompt(
    character: Character | None = None,
    seconds_each: int = 8,
    shot_count: int = 0,
) -> str:
    """分镜师的系统提示词。

    seconds_each 一定要和实际要调的生成模型对齐 —— 模型只能出 8 秒，
    却让分镜师按 15 秒编排，画面会被硬截断。
    """
    count_line = (
        f"{shot_count} 个镜头" if shot_count > 0 else "镜头数由剧本内容决定，不要硬凑"
    )
    passport = render_passport(character, "natural")
    if passport:
        role = (
            "已指定目标角色，**每一镜的画面提示词里都必须带上这个角色的外貌特征**，"
            "这是保证跨镜头人物一致性的唯一手段，不能只在第一镜写、后面用"
            '"她"或"the same woman"指代 —— 生成模型每次调用是独立的，'
            "指代对它没有意义。\n\n"
            f"目标角色：\n{passport}\n"
        )
    else:
        role = (
            "没有指定角色。如果剧本里出现反复登场的人物，你要自己为 TA 固定一套"
            "外貌描述，并在每一镜里原样复用，否则同一个人在不同镜头里会长得不一样。\n"
        )

    return (
        "你是一位专业的动画导演和分镜师。把用户给的剧本或参考画面拆解成连续的分镜。\n\n"
        f"{role}\n"
        "每个镜头输出四个字段：\n"
        "  shotNumber  镜头序号，从 1 开始\n"
        f"  duration    时长，固定写 \"{seconds_each}s\"\n"
        f"  prompt      画面提示词，**{_PROMPT_STYLE}**。必须包含具体的运镜"
        "（push in / pull back / pan / tracking shot / crane up 等）"
        "和光影指令（cinematic lighting / rim light / golden hour 等）。"
        "写具体可拍的画面，不要抽象概念。\n"
        "  narration   中文旁白，**要能直接拿去配音**，是说出来的话，"
        "不是场记式的画面描述。没有旁白就留空字符串。\n\n"
        f"镜头数：{count_line}。\n"
        f"单镜时长 {seconds_each} 秒 —— 这是生成模型的硬上限，"
        "每个镜头的画面信息量要能在这个时长里讲完。\n\n"
        f"{prompt_rules()}\n\n"
        "**只输出一个 JSON 数组，不要 markdown 代码块，不要任何解释文字。**\n"
        '[{"shotNumber":1,"duration":"'
        + f"{seconds_each}s"
        + '","prompt":"...","narration":"..."}]'
    )


def script_messages(
    script: str,
    character: Character | None = None,
    seconds_each: int = 8,
    shot_count: int = 0,
    reference_image: str = "",
) -> list[dict[str, Any]]:
    """剧本 → 分镜的消息体。reference_image 是 data URL，有就走视觉。"""
    content: list[dict[str, Any]] = [
        {"type": "text", "text": f"把下面的剧本拆解为分镜，直接返回 JSON 数组：\n\n{script}"}
    ]
    if reference_image:
        content += [
            {
                "type": "text",
                "text": "这是目标角色的参考图，把 TA 的外貌特征充分融进每一镜的提示词：",
            },
            {"type": "image_url", "image_url": {"url": reference_image, "detail": "low"}},
        ]
    return [
        {"role": "system", "content": system_prompt(character, seconds_each, shot_count)},
        {"role": "user", "content": content if reference_image else content[0]["text"]},
    ]


def frames_messages(
    frames: list[str],
    character: Character | None = None,
    seconds_each: int = 8,
    shot_count: int = 0,
    reference_image: str = "",
) -> list[dict[str, Any]]:
    """参考视频的关键帧 → 分镜（换主角）。

    frames 是按时间顺序的 data URL。用 detail=low —— 帧数一多，
    high 会让 token 直接爆掉，而这里要的是构图和运镜，不是细节。
    """
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "下面是一段参考视频按时间顺序抽出的关键帧。"
                "**精准模仿**它的构图、运镜、动作节奏和情节推进，拆成连续分镜；"
                "但画面里的主角**必须换成**我指定的目标角色。"
                "直接返回 JSON 数组。"
            ),
        }
    ]
    if reference_image:
        content += [
            {"type": "text", "text": "目标角色参考图（提示词里的人换成 TA）："},
            {"type": "image_url", "image_url": {"url": reference_image, "detail": "low"}},
            {"type": "text", "text": "以下是参考视频的关键帧："},
        ]
    for f in frames:
        content.append({"type": "image_url", "image_url": {"url": f, "detail": "low"}})

    return [
        {"role": "system", "content": system_prompt(character, seconds_each, shot_count)},
        {"role": "user", "content": content},
    ]


# ---------- 解析 ----------


def parse_shots(text: str) -> tuple[list[Shot], str]:
    """从模型输出里抽出分镜表，返回 (镜头, 警告)。

    模型的包装方式有好几种：裸数组、```json 围栏、外面再套一层
    {"shots": [...]}。都得认 —— 每种格式漂移都让整条流程失败是不可接受的。
    """
    raw = _unwrap(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return [], f"模型输出不是合法 JSON：{e}"

    if isinstance(data, dict):
        data = data.get("shots") or data.get("storyboard") or []
    if not isinstance(data, list):
        return [], "模型输出不是分镜数组"

    shots: list[Shot] = []
    for i, item in enumerate(data, 1):
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            continue
        shots.append(
            Shot(
                number=int(item.get("shotNumber") or item.get("number") or i),
                prompt=prompt,
                narration=str(item.get("narration") or "").strip(),
                duration=str(item.get("duration") or "").strip(),
            )
        )

    if not shots:
        return [], "解析出来一个镜头都没有"
    return shots, ""


def _unwrap(text: str) -> str:
    raw = text.strip()
    m = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if m:
        raw = m.group(1).strip()
    # 数组优先：这里的契约是 JSON 数组，先找 [ ]，找不到再退回 { }
    i, j = raw.find("["), raw.rfind("]")
    if i >= 0 and j > i:
        return raw[i : j + 1]
    i, j = raw.find("{"), raw.rfind("}")
    return raw[i : j + 1] if i >= 0 and j > i else raw

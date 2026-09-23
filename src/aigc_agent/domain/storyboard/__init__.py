"""分镜与角色护照 —— 从 ai-character-passport 移植。

原项目是一个 Next.js 网页应用，真正有价值的是三块逻辑：
  · 角色护照：把人物外观固化成结构化字段，逐镜注入提示词，保证跨镜头一致性
  · 剧本拆解：把剧本转成带运镜与光影的英文提示词 + 中文旁白
  · 三种提示词输出格式：自然语言 / 标签串 / JSON

UI、zustand、IndexedDB 全部丢掉 —— 这边有 AssetStore。
浏览器端的 canvas 抽帧也丢掉 —— 这边有 ffmpeg，更准也更快。
"""

from .breakdown import Shot, frames_messages, parse_shots, script_messages, system_prompt
from .passport import Character, render_passport

__all__ = [
    "Character",
    "Shot",
    "frames_messages",
    "parse_shots",
    "render_passport",
    "script_messages",
    "system_prompt",
]

"""角色护照 —— 让同一个人在不同镜头里长得一样。

这是整个移植里最值钱的一块。视频生成模型每次调用是独立的，同一句
"一位工程师"生成四次会得到四张不同的脸。护照的做法是把外观拆成结构化
字段，每一镜的提示词里都原样带上，把随机性压到最小。

字段划分沿用原项目（name / basePrompt / appearanceTags / styleLighting），
因为这个划分本身是有讲究的：
  base_prompt     人物是谁 —— 一句话，给模型一个整体印象
  appearance      长什么样 —— 标签串，是一致性的主力，越具体越好
  style_lighting  怎么拍 —— 风格光影，决定"看起来是不是同一部片子"
把它们揉成一段话会让模型自由发挥的空间变大，一致性就散了。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

# 三种输出格式，对应下游不同的消费方
FORMATS = ("natural", "tags", "json")


@dataclass
class Character:
    name: str
    base_prompt: str = ""
    appearance: str = ""
    style_lighting: str = ""
    reference_image: str = ""  # 资产 id；有参考图时喂给视觉模型
    notes: str = ""

    @property
    def filled(self) -> bool:
        """至少要有名字和一项外观描述，否则注入进去等于没注入。"""
        return bool(self.name and (self.base_prompt or self.appearance))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Character:
        known = {k: d.get(k, "") for k in cls.__dataclass_fields__}
        return cls(**known)

    def render(self, fmt: str = "natural") -> str:
        return render_passport(self, fmt)


def render_passport(c: Character | None, fmt: str = "natural") -> str:
    """把护照渲染成提示词片段。

    natural  一段自然语言，给对话式模型（也是注入分镜提示词时用的）
    tags     逗号分隔标签串，给 ComfyUI / 即梦这类吃标签的
    json     结构化，给需要程序化消费的下游
    """
    if c is None or not c.name and not c.base_prompt:
        return ""
    if fmt == "tags":
        parts = [c.name, c.base_prompt, c.appearance, c.style_lighting]
        tags: list[str] = []
        for p in parts:
            if not p:
                continue
            tags += [t.strip() for t in _split_tags(p) if t.strip()]
        return ", ".join(dict.fromkeys(tags))  # 去重但保序
    if fmt == "json":
        return json.dumps(
            {
                "character": {
                    "name": c.name or None,
                    "base_prompt": c.base_prompt or None,
                    "appearance": [t.strip() for t in _split_tags(c.appearance) if t.strip()],
                    "style_and_lighting": c.style_lighting or None,
                }
            },
            ensure_ascii=False,
            indent=2,
        )

    bits = [f"This is {c.name}." if c.name else ""]
    if c.base_prompt:
        bits.append(c.base_prompt.rstrip("。.") + ".")
    if c.appearance:
        bits.append(f"Appearance: {c.appearance}.")
    if c.style_lighting:
        bits.append(f"Style and lighting: {c.style_lighting}.")
    return " ".join(b for b in bits if b).strip()


def _split_tags(s: str) -> list[str]:
    out = [s]
    for sep in (",", "，", ";", "；", "\n"):
        out = [piece for chunk in out for piece in chunk.split(sep)]
    return out


@dataclass
class Roster:
    """一个剧本里的全部角色。多角色时逐个注入，模型才分得清谁是谁。"""

    characters: list[Character] = field(default_factory=list)

    def get(self, name: str) -> Character | None:
        low = name.strip().lower()
        return next((c for c in self.characters if c.name.strip().lower() == low), None)

    def render(self, fmt: str = "natural") -> str:
        parts = [c.render(fmt) for c in self.characters if c.filled]
        return "\n".join(p for p in parts if p)

"""角色音色锁定（2026-09-18）—— 多集视频里同一角色的声音不能漂。

seedance 是逐段生成的：每段视频各自按提示词里的台词"发明"一个声音，几十段下来
同一角色必然漂 —— 这一段低沉、下一段清亮，第 8 集听起来已经不是第 1 集那个人。
两道锁：

  1. **音色卡**：第②步资产库给每个角色一行音色描述（性别 | 年龄听感 | 音高质地 |
     语速节奏 | 口音习惯），渲染时逐段把**说话角色**的音色卡锁进提示词开头。
     文字描述只能把声音框在一个范围里，锁不死，所以还要第二道。
  2. **音色锚点**：每个角色第一段（尽量是**只有他一个人说话**的那段）成功的视频当作
     声音基准，之后所有他开口的片段都把这段视频作为参考传给模型（seedance 2.0 支持
     @视频N / @音频N 参考音色：从参考里取音色、表演风格），提示词写明
     「(角色) 的声音以 @视频N 中的男声为准，只取声音不取画面」。
     锚点跨集持久（存成资产），整部剧都对着同一段声音，而不是一段接一段地传话漂移。

这里只放**不依赖 AssetStore 与网络**的纯逻辑：谁在说话、选哪段当锚点、
提示词怎么写、锚点表怎么增删。读写资产和调模型在 functions/drama.py。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .models import AssetLibrary, ShotPrompt

# 台词：中文引号 / 直引号 / 日式引号；紧跟在说话者的动作之后（第③步规则要求台词内嵌）
_LINE = re.compile(r"[“\"「]([^”\"」]{1,300})[”\"」]")
# 引用标记：(角色-服装-[1]) / (场景) / (L-Cut Voice-over)
_TOKEN = re.compile(r"\(([^()]{2,80}?)\)")
# 切镜/画外音标记不是角色
_MARKERS = ("切镜", "L-Cut", "VO", "Front", "Profile", "Back")

ANCHORS_CREATOR = "tool:drama_voice_anchors"


def speakers_of(shot: ShotPrompt, lib: AssetLibrary) -> dict[str, int]:
    """镜头里谁说了几句：{角色名: 台词句数}，按首次开口顺序。

    规则：一句台词归**它前面最近的那个角色引用**（第③步要求台词紧嵌在说话者动作后，
    L-Cut 画外音也是 (角色ID)(L-Cut Voice-over)说："…" 的格式，同样成立）。
    引号前一个角色都没出现的台词（旁白）不计。
    """
    text = shot.description or ""
    events: list[tuple[int, str, str]] = []  # (位置, 类型, 值)
    for m in _TOKEN.finditer(text):
        t = m.group(1).strip()
        if t.startswith(_MARKERS):
            continue
        ch = lib.character_of(t)
        if ch is not None:
            events.append((m.start(), "who", ch.name))
    for m in _LINE.finditer(text):
        events.append((m.start(), "line", m.group(1)))
    events.sort(key=lambda e: e[0])

    out: dict[str, int] = {}
    current = ""
    for _, kind, val in events:
        if kind == "who":
            current = val
        elif current:
            out[current] = out.get(current, 0) + 1
    return out


def has_lines(shot: ShotPrompt) -> bool:
    return bool(_LINE.search(shot.description or ""))


# ---------------------------------------------------------------- 锚点表


@dataclass
class Anchor:
    """一个角色的声音基准：哪段视频、来自哪个镜头、是不是独白段、人有没有钉死。"""

    character: str
    asset: str  # 视频片段资产 id
    scene: str = ""  # 如 [第1集-1场]
    clip: str = ""  # 镜头范围，如 1-4
    lines: int = 0
    solo: bool = False  # 那段里只有他一个人说话（干净的声音样本）
    pinned: bool = False  # 人工指定的，自动逻辑不覆盖
    rolled_from: str = ""  # 原锚点链接过期后接替的，记一下原来是谁

    def as_dict(self) -> dict[str, Any]:
        return {
            "character": self.character,
            "asset": self.asset,
            "scene": self.scene,
            "clip": self.clip,
            "lines": self.lines,
            "solo": self.solo,
            "pinned": self.pinned,
            "rolled_from": self.rolled_from,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Anchor:
        return cls(
            character=str(d.get("character") or ""),
            asset=str(d.get("asset") or ""),
            scene=str(d.get("scene") or ""),
            clip=str(d.get("clip") or ""),
            lines=int(d.get("lines") or 0),
            solo=bool(d.get("solo")),
            pinned=bool(d.get("pinned")),
            rolled_from=str(d.get("rolled_from") or ""),
        )


def parse_anchors(text: str) -> dict[str, Anchor]:
    import json

    try:
        data = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    out: dict[str, Anchor] = {}
    rows = data.values() if isinstance(data, dict) else data
    for row in rows or []:
        if isinstance(row, dict) and row.get("character") and row.get("asset"):
            a = Anchor.from_dict(row)
            out[a.character] = a
    return out


def dump_anchors(anchors: dict[str, Anchor]) -> str:
    import json

    return json.dumps(
        {k: v.as_dict() for k, v in anchors.items()}, ensure_ascii=False, indent=2
    )


# ---------------------------------------------------------------- 选锚点镜头


def choose_anchor_shot(
    speakers_by_shot: list[dict[str, int]], character: str
) -> int | None:
    """给一个角色挑当声音基准的镜头下标。

    优先**独白段**（只有他说话 —— 参考里没有别人的声音，模型不会取错人），
    独白里挑台词最多的；没有独白就挑他台词最多的；都一样取最早的。
    """
    best: tuple[int, int, int] | None = None  # (solo, lines, -index) 越大越好
    for i, sp in enumerate(speakers_by_shot):
        n = sp.get(character, 0)
        if n <= 0:
            continue
        key = (1 if len(sp) == 1 else 0, n, -i)
        if best is None or key > best:
            best = key
    return -best[2] if best else None


@dataclass
class AnchorPlan:
    """这一批渲染里的锚点安排。"""

    births: dict[str, int] = field(default_factory=dict)  # 角色 → 作为其锚点的镜头下标
    ready: dict[str, Anchor] = field(default_factory=dict)  # 已有且可用的锚点
    stale: list[str] = field(default_factory=list)  # 链接过期、本次重定的角色
    unvoiced: list[str] = field(default_factory=list)  # 有台词却没有音色卡的角色
    spoken: list[str] = field(default_factory=list)  # 这一批里开口的全部角色

    @property
    def anchor_shots(self) -> set[int]:
        return set(self.births.values())


def plan_anchors(
    speakers_by_shot: list[dict[str, int]],
    lib: AssetLibrary,
    existing: dict[str, Anchor],
    usable: set[str],
) -> AnchorPlan:
    """决定哪些角色沿用旧锚点、哪些角色本次要新定锚点。

    usable：existing 里链接仍有效的角色名；不在里面的按过期处理（本次重定）。
    """
    plan = AnchorPlan()
    plan.spoken = list(dict.fromkeys(name for sp in speakers_by_shot for name in sp))
    for name in plan.spoken:
        ch = lib.character_of(name)
        if ch is not None and not ch.voice:
            plan.unvoiced.append(name)
        old = existing.get(name)
        if old and name in usable:
            plan.ready[name] = old
            continue
        if old:
            plan.stale.append(name)
        idx = choose_anchor_shot(speakers_by_shot, name)
        if idx is not None:
            plan.births[name] = idx
    return plan


# ---------------------------------------------------------------- 提示词


def voice_block(
    speakers: list[str],
    lib: AssetLibrary,
    anchor_tokens: dict[str, int],
    carry_tokens: list[tuple[str, int]] | None = None,
) -> str:
    """锁在镜头提示词开头的音色段。

    anchor_tokens：角色 → 本段 @视频N 的序号（有锚点参考的角色）。
    carry_tokens：[(前序场次, @视频N 序号)]，告诉模型 {第1集-1场} 就是哪个参考视频。
    没有说话角色时返回空串（纯环境镜头不挂）。
    """
    if not speakers:
        return ""
    parts: list[str] = []
    for name in speakers:
        ch = lib.character_of(name)
        card = (ch.voice if ch else "") or "音色与该角色此前片段保持一致"
        line = f"({name})：{card}"
        n = anchor_tokens.get(name)
        if n:
            line += f"；声音必须与 @视频{n} 中该角色的声音完全一致（只取声音，不取画面与动作）"
        parts.append(line)
    text = "【音色锁定】" + "。".join(parts) + "。全片同一角色音色一致，台词按各自音色发声。"
    if carry_tokens:
        text += "".join(
            f"前序片段 {{{scene}}} 即 @视频{n}，画面从其结尾状态延续。" for scene, n in carry_tokens
        )
    return text


def anchors_table(anchors: dict[str, Anchor]) -> str:
    """给人看的锚点表。"""
    if not anchors:
        return "（还没有音色锚点：渲过带台词的镜头后会自动定）"
    lines = []
    for a in anchors.values():
        flags = []
        if a.solo:
            flags.append("独白段")
        if a.pinned:
            flags.append("人工指定")
        if a.rolled_from:
            flags.append(f"接替 {a.rolled_from}")
        tag = f"（{'、'.join(flags)}）" if flags else ""
        lines.append(f"  {a.character} ← {a.scene} 镜{a.clip} {a.asset} {a.lines} 句{tag}")
    return "\n".join(lines)

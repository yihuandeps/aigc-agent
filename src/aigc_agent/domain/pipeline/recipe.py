"""配方驱动的短视频生产 —— 流程是数据，不是代码。

为什么不写成硬编码的函数：这条流程要反复调效果（镜头数、画面风格、
口播语气、模型档位），每次调都改 Python 是不可接受的迭代速度。
配方是 yaml，改完立刻生效。

流程本身是确定的（简报 → 生成 → 配音 → 字幕 → 合成），
所以走「确定性骨架 + 节点内让模型发挥」——见 ARCHITECTURE.md §4.1 里
关于什么时候该固化成流程的判断：**这个流程你会跑第二次吗？会，就固化。**

配方只描述风格和参数；流程在 domain/functions/short_video.py（对话和 `agent video make`
走的是同一组工具）。2026-09-23 审查后删掉了老链专用的文案 / 选题提示词和解析
（script_prompt / pick_prompt / parse_plan / parse_pick / override），和新链的是重复实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from ..drama.format import global_max_cut

RECIPES_DIR = Path(__file__).resolve().parents[3].parent / "config" / "recipes"


@dataclass
class Recipe:
    name: str
    description: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    shots: dict[str, Any] = field(default_factory=dict)
    voiceover: dict[str, Any] = field(default_factory=dict)
    subtitle: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    review: dict[str, Any] = field(default_factory=dict)
    cut: dict[str, Any] = field(default_factory=dict)
    grounding: dict[str, Any] = field(default_factory=dict)
    # 风格信息（2026-09-18 抖音分支：配方 = 风格）：
    # label / when / footage / min_seconds / max_seconds
    style: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    # ---------- 风格 ----------

    @property
    def key(self) -> str:
        return self.path.stem if self.path else self.name

    @property
    def style_label(self) -> str:
        return str(self.style.get("label") or self.name)

    @property
    def style_desc(self) -> str:
        return str(self.style.get("when") or self.description)

    @property
    def duration_bounds(self) -> tuple[int, int]:
        """这种风格允许的成片时长范围（秒）。"""
        lo = int(self.style.get("min_seconds") or 15)
        hi = int(self.style.get("max_seconds") or max(lo, 60))
        return lo, max(lo, hi)

    # ---------- 派生 ----------

    @property
    def shot_count(self) -> int:
        return int(self.shots.get("count", 4))

    @property
    def seconds_each(self) -> int:
        return int(self.shots.get("seconds_each", 8))

    @property
    def source_seconds(self) -> int:
        """素材池总长 = 生成几段 × 每段多长。不等于成片长度。"""
        return self.shot_count * self.seconds_each

    @property
    def total_seconds(self) -> int:
        """**成片**时长。

        开了快切之后这两个数就分开了：素材照旧按模型原生长度生成（8s 一段），
        剪辑层把它切成密集短镜头，成片长度由 output.duration 说了算。
        不分开的话，为了 12 刀就得生成 12 段，成本和耗时都是三倍。
        """
        if self.cut_enabled and self.output.get("duration"):
            return int(self.output["duration"])
        return self.source_seconds

    # ---------- 剪辑节奏 ----------

    @property
    def cut_enabled(self) -> bool:
        return bool(self.cut.get("enabled"))

    @property
    def cut_max(self) -> float:
        """单个镜头最长几秒。0 = 不快切。

        不会超过全局单镜上限（config/drama.yaml cut.max_seconds，用户 2026-09-20 定的
        每镜 ≤3 秒）：配方里写 4 也按 3 切；**配方关了快切也按全局上限切**（2026-09-23 审查：
        之前 cut.enabled=false 返回 0，合成时整段拼接，一镜 8 秒）。"""
        cap = global_max_cut(RECIPES_DIR.parent)
        if not self.cut_enabled:
            return cap
        own = float(self.cut.get("max_seconds", 0))
        return min(own, cap) if own > 0 else cap

    @property
    def cut_min(self) -> float:
        return float(self.cut.get("min_seconds", 1.2))

    def filename(self, topic: str, tier: str) -> str:
        tpl = str(self.output.get("filename") or "{topic}_{date}.mp4")
        name = tpl.format(topic=_slug(topic), date=date.today().strftime("%Y%m%d"), tier=tier)
        return name if name.endswith(".mp4") else name + ".mp4"

    # ---------- 提示词 ----------

    @property
    def grounded(self) -> bool:
        return bool(self.grounding.get("enabled"))

    def shot_prompt(self, shot: str, level: str = "") -> str:
        """单镜提交给生成模型的完整 prompt。

        真实感段落放最后：它是**全局约束**（皮肤/布光/光学瑕疵），
        放前面会被当成画面主体描述，模型容易照着生成"一堆雀斑的特写"。
        realism 写 auto/true 时用全局档位（media_models.yaml drama.realism_level），
        写 skin 只加皮肤那段（不改布光 / 光学：手机随拍类配方用），
        写成文字就原样用 —— 旧配方里手写的那段仍然有效。
        """
        # 局部导入：避免 pipeline ↔ realism 的加载顺序问题
        from ..realism import skin_suffix, video_suffix

        parts = [shot, str(self.shots.get("style") or "")]
        realism = self.shots.get("realism")
        mode = "" if isinstance(realism, bool) else str(realism or "").strip().lower()
        if isinstance(realism, bool) or mode in ("auto", "true"):
            if realism:
                parts.append(video_suffix(level))
        elif mode in ("skin", "person"):
            parts.append(skin_suffix(level))
        elif realism:
            parts.append(" ".join(str(realism).split()))
        return "，".join(p for p in parts if p.strip())


def load_recipe(name_or_path: str, recipes_dir: Path | None = None) -> Recipe:
    d = recipes_dir or RECIPES_DIR
    p = Path(name_or_path)
    if not p.exists():
        p = d / (name_or_path if name_or_path.endswith(".yaml") else f"{name_or_path}.yaml")
    if not p.exists():
        avail = ", ".join(f.stem for f in d.glob("*.yaml")) if d.exists() else "（无）"
        raise FileNotFoundError(f"找不到配方 {name_or_path!r}。可用：{avail}")

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return Recipe(
        name=raw.get("name", p.stem),
        description=raw.get("description", ""),
        output=raw.get("output") or {},
        shots=raw.get("shots") or {},
        voiceover=raw.get("voiceover") or {},
        subtitle=raw.get("subtitle") or {},
        models=raw.get("models") or {},
        review=raw.get("review") or {},
        cut=raw.get("cut") or {},
        grounding=raw.get("grounding") or {},
        style=raw.get("style") or {},
        path=p,
    )


def list_recipes(recipes_dir: Path | None = None) -> list[Recipe]:
    d = recipes_dir or RECIPES_DIR
    if not d.exists():
        return []
    out = []
    for f in sorted(d.glob("*.yaml")):
        try:
            out.append(load_recipe(str(f)))
        except Exception:  # noqa: BLE001 — 单个配方写坏不该拖垮列表
            continue
    return out


def _slug(name: str, limit: int = 40) -> str:
    cleaned = "".join(c if (c.isalnum() or c in "._-（）()") else "_" for c in name)
    return cleaned.strip("_")[:limit] or "video"

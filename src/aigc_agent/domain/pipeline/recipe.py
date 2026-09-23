"""配方驱动的短视频生产 —— 流程是数据，不是代码。

为什么不写成硬编码的函数：这条流程要反复调效果（镜头数、画面风格、
口播语气、模型档位），每次调都改 Python 是不可接受的迭代速度。
配方是 yaml，改完立刻生效。

流程本身是确定的（写文案 → 定分镜 → 生成 → 配音 → 字幕 → 合成），
所以走「确定性骨架 + 节点内让模型发挥」——见 ARCHITECTURE.md §4.1 里
关于什么时候该固化成流程的判断：**这个流程你会跑第二次吗？会，就固化。**
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from ..drama.format import global_max_cut

RECIPES_DIR = Path(__file__).resolve().parents[3].parent / "config" / "recipes"

# 中文口播语速估算：约 4.5 字/秒
CHARS_PER_SECOND = 4.5


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

    @property
    def script_chars(self) -> int:
        return int(self.total_seconds * CHARS_PER_SECOND)

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

    # ---------- 覆盖 ----------

    def override(self, **kw: Any) -> Recipe:
        """命令行临时覆盖。不写回文件——一次性调整不该污染配方。"""
        r = Recipe(**{k: v for k, v in self.__dict__.items()})
        # --max-cut 要先处理：它决定了 --duration 该改成片长度还是改素材段数
        if kw.get("max_cut"):
            r.cut = {**self.cut, "enabled": True, "max_seconds": float(kw["max_cut"])}
        if kw.get("duration"):
            total = int(kw["duration"])
            if r.cut_enabled:
                # 快切模式下成片长度和素材量无关，改 output 而不是加素材
                r.output = {**self.output, "duration": total}
            else:
                r.shots = {**self.shots, "count": max(1, round(total / self.seconds_each))}
        if kw.get("tier"):
            r.models = {**self.models, "video_tier": kw["tier"]}
        if kw.get("voice"):
            r.voiceover = {**self.voiceover, "voice": kw["voice"]}
        if kw.get("no_voiceover"):
            r.voiceover = {**self.voiceover, "enabled": False}
        if kw.get("no_subtitle"):
            r.subtitle = {**self.subtitle, "enabled": False}
        return r

    # ---------- 提示词 ----------

    @property
    def grounded(self) -> bool:
        return bool(self.grounding.get("enabled"))

    def pick_prompt(self, topic: str, hot: str) -> str:
        """给模型看真实热榜，让它挑选题。

        这一步是上一版视频效果差的根因所在：没有它，「热点」就只能靠
        模型按常识编，内容根基是虚的。
        """
        nl = "\n"
        hint = str(self.grounding.get("pick_hint") or "")
        return (
            f"方向：{topic}{nl}{nl}"
            f"以下是刚拉到的**真实热榜**（带热度数值）：{nl}{nl}{hot}{nl}{nl}"
            f"{hint}{nl}{nl}"
            f"只输出 JSON：{nl}"
            '{"topic": "你选定的具体选题", "why": "一句话说明为什么选它"}'
        )

    def script_prompt(self, topic: str, facts: str = "") -> str:
        """facts = 刚拉到的真实热榜原文。

        带上它不只是给背景：口播里的数字必须有出处，否则模型会顺手编一个
        听起来很顺的（「比阿波罗登月计算机强一百万倍」就是这么来的）。
        """
        hint = str(self.voiceover.get("script_hint") or "").format(chars=self.script_chars)
        ground = (
            f"""以下是刚拉到的**真实热榜数据**，口播里出现的数字只能来自这里：

{facts}

拿不准的数字就不要写，用定性描述代替。**不要编造具体数值。**

"""
            if facts
            else ""
        )
        return (
            f"{ground}主题：{topic}\n\n"
            f"写一条 {self.total_seconds} 秒短视频的口播文案。\n{hint}\n\n"
            f"然后给出 {self.shot_count} 个分镜画面描述，每镜 {self.seconds_each} 秒。\n"
            f"画面风格：{self.shots.get('style', '')}\n"
            f"{self.shots.get('prompt_hint', '')}\n\n"
            "**只输出一个 JSON，不要任何解释文字**，格式：\n"
            '{"script": "口播全文", "shots": ["第1镜画面描述", "第2镜...", ...]}'
        )

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


def parse_pick(text: str) -> tuple[str, str]:
    """解析选题结果，返回 (选题, 理由)。解析失败返回空，由上层回退。"""
    raw = _unwrap_json(text)
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return "", ""
    return str(d.get("topic") or "").strip(), str(d.get("why") or "").strip()


def _unwrap_json(text: str) -> str:
    """模型常把 JSON 包在 ```json 块里或前后带解释，都要能剥出来。"""
    raw = text.strip()
    m = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if m:
        return m.group(1).strip()
    i, j = raw.find("{"), raw.rfind("}")
    return raw[i : j + 1] if i >= 0 and j > i else raw


def parse_plan(text: str, expect_shots: int) -> tuple[str, list[str], str]:
    """从模型输出里抽出 {script, shots}。

    模型常把 JSON 包在 ```json 块里，或前后带解释文字 —— 都要能容忍，
    否则每次格式漂移都会让整条流程失败。返回 (script, shots, 警告)。
    """
    raw = _unwrap_json(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return "", [], f"模型输出不是合法 JSON：{e}"

    script = str(data.get("script") or "").strip()
    shots = [str(s).strip() for s in (data.get("shots") or []) if str(s).strip()]
    if not script:
        return "", [], "输出里没有 script"
    if not shots:
        return script, [], "输出里没有 shots"

    warn = ""
    if len(shots) != expect_shots:
        warn = f"要 {expect_shots} 个分镜，模型给了 {len(shots)} 个，按实际数量走"
    return script, shots, warn


def _slug(name: str, limit: int = 40) -> str:
    cleaned = "".join(c if (c.isalnum() or c in "._-（）()") else "_" for c in name)
    return cleaned.strip("_")[:limit] or "video"

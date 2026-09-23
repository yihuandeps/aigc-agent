"""M17 Analytics & Feedback —— 数据回流。

这是让 Agent 越用越准的唯一闭环：发布后的数据 → 复盘 → 反哺账号层记忆。

MVP 形态（ARCHITECTURE M17 原话）：**先做人工录入，但接口留出来。**
  · MetricsStore   JSONL 追加写，一条一次采集，不覆盖历史
  · MetricsSource  协议：将来接平台 API 时实现 fetch()，录入路径不变
  · Reviewer       复盘：各项指标相对账号中位数的倍率，排出强/弱作品
  · Feedback       提炼：显著优于 / 劣于中位数的作品写成 **DATA 来源**的账号层记忆

为什么 DATA 来源可以进账号层而 INFERRED 不行：前者是被真实数据验证过的，
后者是模型的臆测。这是 M8.2 「晋升」那条硬规则的另一面。
提炼出来的记忆是 preference 类：强作品 positive（进 Brief 的建议区），
弱作品 neutral 并写明「慎重复制」—— 数据说的是"这类曾经不灵"，不是禁令。
"""

from __future__ import annotations

import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ...capabilities.memory.store import (
    Category,
    Keyword,
    Layer,
    Memory,
    MemoryStore,
    Polarity,
    Source,
)


class Metrics(BaseModel):
    id: str = Field(default_factory=lambda: "mt_" + uuid.uuid4().hex[:10])
    package_id: str  # 待发布包的资产 id
    platform: str = ""
    title: str = ""
    tags: list[str] = Field(default_factory=list)
    url: str = ""
    views: int = 0
    completion_rate: float | None = None  # 0-1
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    follows: int = 0
    conversions: int = 0
    source: str = "manual"  # manual | api:<name>
    note: str = ""
    collected_at: float = Field(default_factory=time.time)

    @property
    def engagement(self) -> float:
        """互动率 = (赞 + 评 + 转 + 藏) / 播放。播放为 0 时为 0。"""
        if self.views <= 0:
            return 0.0
        return (self.likes + self.comments + self.shares + self.saves) / self.views


class MetricsSource(Protocol):
    """平台 API 的接口位。实现 fetch()，其余（存储、复盘、提炼）不用改。"""

    name: str

    async def fetch(self, package_id: str, url: str) -> Metrics | None: ...


class ManualSource:
    """人工录入：不会主动去拉。"""

    name = "manual"

    async def fetch(self, package_id: str, url: str) -> Metrics | None:  # noqa: ARG002
        return None


class MetricsStore:
    """JSONL 追加写。一个包可以有多次采集（发布后 1 天、7 天…），复盘取每包最新一条。"""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root
        self._items: list[Metrics] = []
        if root:
            root.mkdir(parents=True, exist_ok=True)
            self._load()

    @property
    def path(self) -> Path | None:
        return (self.root / "metrics.jsonl") if self.root else None

    def _load(self) -> None:
        p = self.path
        if p is None or not p.exists():
            return
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                self._items.append(Metrics.model_validate_json(line))
            except Exception:  # noqa: BLE001 — 单条坏数据不该让整个库起不来
                continue

    def add(self, m: Metrics) -> Metrics:
        self._items.append(m)
        p = self.path
        if p is not None:
            with p.open("a", encoding="utf-8") as f:
                f.write(m.model_dump_json() + "\n")
        return m

    def all(self, platform: str = "") -> list[Metrics]:
        items = [m for m in self._items if not platform or m.platform == platform]
        return sorted(items, key=lambda m: m.collected_at)

    def latest_per_package(self, platform: str = "") -> list[Metrics]:
        latest: dict[str, Metrics] = {}
        for m in self.all(platform):
            latest[m.package_id] = m  # 已按时间排序，后者覆盖前者
        return list(latest.values())

    def __len__(self) -> int:
        return len(self._items)


# ---------------------------------------------------------------- 复盘


class PieceReview(BaseModel):
    package_id: str
    title: str
    platform: str
    views: int
    views_ratio: float  # 相对中位数
    completion_rate: float | None
    completion_ratio: float | None
    engagement: float
    engagement_ratio: float
    verdict: str  # strong | weak | normal
    tags: list[str] = Field(default_factory=list)


class Review(BaseModel):
    platform: str = ""
    samples: int = 0
    median_views: float = 0.0
    median_completion: float | None = None
    median_engagement: float = 0.0
    pieces: list[PieceReview] = Field(default_factory=list)
    generated_at: float = Field(default_factory=time.time)

    @property
    def strong(self) -> list[PieceReview]:
        return [p for p in self.pieces if p.verdict == "strong"]

    @property
    def weak(self) -> list[PieceReview]:
        return [p for p in self.pieces if p.verdict == "weak"]

    def render(self) -> str:
        if self.samples == 0:
            return "还没有任何数据，先用 record_metrics 录入。"
        head = (
            f"复盘（{self.platform or '全部平台'}）· {self.samples} 个作品 · "
            f"中位播放 {self.median_views:.0f} · 中位互动率 {self.median_engagement:.2%}"
            + (
                f" · 中位完播 {self.median_completion:.0%}"
                if self.median_completion is not None
                else ""
            )
        )
        if self.samples < 3:
            head += "\n样本不足 3 个，倍率仅供参考，不做提炼。"
        lines = [head, ""]
        for p in sorted(self.pieces, key=lambda x: x.views_ratio, reverse=True):
            mark = {"strong": "▲", "weak": "▼", "normal": "·"}[p.verdict]
            comp = f" · 完播 {p.completion_rate:.0%}" if p.completion_rate is not None else ""
            lines.append(
                f"{mark} {p.title[:24]}（{p.package_id}）播放 {p.views}（{p.views_ratio:.1f}×）"
                f" · 互动 {p.engagement:.2%}（{p.engagement_ratio:.1f}×）{comp}"
            )
        return "\n".join(lines)


class Reviewer:
    def __init__(self, store: MetricsStore, strong: float = 1.5, weak: float = 0.5) -> None:
        self.store = store
        self.strong = strong
        self.weak = weak

    def summarize(self, platform: str = "") -> Review:
        items = self.store.latest_per_package(platform)
        if not items:
            return Review(platform=platform)
        med_views = statistics.median([m.views for m in items]) or 0.0
        comps = [m.completion_rate for m in items if m.completion_rate is not None]
        med_comp = statistics.median(comps) if comps else None
        med_eng = statistics.median([m.engagement for m in items]) or 0.0

        pieces: list[PieceReview] = []
        for m in items:
            vr = (m.views / med_views) if med_views else 0.0
            er = (m.engagement / med_eng) if med_eng else 0.0
            cr = (
                (m.completion_rate / med_comp)
                if (med_comp and m.completion_rate is not None)
                else None
            )
            verdict = "normal"
            if len(items) >= 3:
                # 播放定强弱；完播（若有）只做否决：播放很高但完播明显低于中位数不算强，
                # 播放很低但完播明显高于中位数也不算弱 —— 那是分发问题不是内容问题
                if vr >= self.strong and (cr is None or cr >= 0.95):
                    verdict = "strong"
                elif vr <= self.weak and (cr is None or cr <= 1.05):
                    verdict = "weak"
            pieces.append(
                PieceReview(
                    package_id=m.package_id,
                    title=m.title or m.package_id,
                    platform=m.platform,
                    views=m.views,
                    views_ratio=round(vr, 2),
                    completion_rate=m.completion_rate,
                    completion_ratio=round(cr, 2) if cr is not None else None,
                    engagement=round(m.engagement, 4),
                    engagement_ratio=round(er, 2),
                    verdict=verdict,
                    tags=list(m.tags),
                )
            )
        return Review(
            platform=platform,
            samples=len(items),
            median_views=float(med_views),
            median_completion=float(med_comp) if med_comp is not None else None,
            median_engagement=float(med_eng),
            pieces=pieces,
        )


# ---------------------------------------------------------------- 提炼进记忆


class Feedback:
    """复盘 → 账号层记忆（DATA 来源）。同一个包只提炼一次（按 origin_ref 去重）。"""

    # 样本门槛：之前 3 个样本就提炼进账号层 —— 一条偶然爆款就能改写账号偏好（2026-09-23 审查）
    def __init__(self, memories: MemoryStore, min_samples: int = 8) -> None:
        self.memories = memories
        self.min_samples = min_samples

    def distill(self, review: Review) -> list[Memory]:
        if review.samples < self.min_samples:
            return []
        existing = {m.origin_ref for m in self.memories.all()}
        out: list[Memory] = []
        for p in review.strong + review.weak:
            ref = f"metrics:{p.package_id}"
            if ref in existing:
                continue
            comp = f"，完播 {p.completion_rate:.0%}" if p.completion_rate is not None else ""
            if p.verdict == "strong":
                content = (
                    f"「{p.title}」在{p.platform or '平台'}表现优于账号中位数"
                    f"（播放 {p.views_ratio:.1f} 倍{comp}），这类选题/写法值得复用"
                )
                polarity = Polarity.POSITIVE
            else:
                content = (
                    f"「{p.title}」在{p.platform or '平台'}表现不及账号中位数一半"
                    f"（播放 {p.views_ratio:.1f} 倍{comp}），这类做法慎重复制"
                )
                polarity = Polarity.NEUTRAL
            terms = _terms(p)
            mem = Memory(
                layer=Layer.ACCOUNT,
                content=content,
                keywords=[
                    Keyword(term=t, polarity=polarity, category=Category.PREFERENCE,
                            origin_quote=f"播放 {p.views}")
                    for t in terms
                ],
                origin_ref=ref,
                source=Source.DATA,  # 数据验证过的才能进账号层
                # 样本越多越可信；单条作品的强弱只是一条建议（PREFERENCE），不是禁令
                confidence=min(0.8, 0.3 + 0.04 * review.samples),
                weight=1.5 if p.verdict == "strong" else 1.0,
            )
            self.memories.put(mem)
            out.append(mem)
        return out


def _terms(p: PieceReview) -> list[str]:
    """召回索引：话题标签 + 标题里的片段。"""
    terms = [t for t in p.tags if 2 <= len(t) <= 12]
    title = p.title.strip()
    if title:
        terms.append(title[:12])
    if not terms:
        terms.append(p.package_id)
    return list(dict.fromkeys(terms))


def review_to_params(review: Review) -> dict[str, Any]:
    return json.loads(review.model_dump_json())

"""M17 数据回流验收 + P4 完整闭环。

P4 验收原话：「完整闭环；数据回流能提炼进账号层记忆」。
  存稿 → 机审 → 待发布包 → 人上传记回链接 → 录数据 → 复盘 → DATA 来源的账号层记忆 → 进 Brief
"""

from __future__ import annotations

from pathlib import Path

from aigc_agent.capabilities.memory.brief import build_brief
from aigc_agent.capabilities.memory.store import Layer, MemoryStore, Polarity, Source
from aigc_agent.capabilities.retrieval import MemorySource, RetrievalHub
from aigc_agent.domain.analytics import Feedback, ManualSource, Metrics, MetricsStore, Reviewer
from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.compliance import ComplianceChecker, ComplianceRules
from aigc_agent.domain.distribution import Packager, PlatformCatalog
from aigc_agent.domain.functions.analytics import AnalyticsFunctions
from aigc_agent.domain.functions.compliance import ComplianceFunctions
from aigc_agent.domain.functions.content import ContentFunctions
from aigc_agent.domain.functions.distribution import DistributionFunctions
from aigc_agent.domain.functions.retrieval import AssetSource, RetrievalFunctions
from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
CATALOG = PlatformCatalog.load(ROOT / "config" / "platforms.yaml")
RULES = ComplianceRules.load(ROOT / "config" / "compliance.yaml")


def _metric(
    pid: str, views: int, title: str = "", completion: float | None = None, **kw
) -> Metrics:
    return Metrics(
        package_id=pid, platform="douyin", title=title or pid, views=views,
        completion_rate=completion, **kw,
    )


# ---------------------------------------------------------------- 存储与复盘


def test_指标追加写并按包取最新(tmp_path: Path):
    store = MetricsStore(tmp_path / "analytics")
    store.add(_metric("p1", 100, likes=5))
    store.add(_metric("p1", 300, likes=20))
    store.add(_metric("p2", 50))
    assert len(store) == 3
    latest = {m.package_id: m for m in store.latest_per_package()}
    assert latest["p1"].views == 300 and latest["p2"].views == 50
    assert len(MetricsStore(tmp_path / "analytics")) == 3, "落盘后能读回"
    assert abs(latest["p1"].engagement - 20 / 300) < 1e-9


def test_复盘倍率与强弱判定():
    store = MetricsStore()
    for pid, v in (("a", 100), ("b", 100), ("c", 400), ("d", 20)):
        store.add(_metric(pid, v, title=f"作品{pid}"))
    review = Reviewer(store).summarize("douyin")
    assert review.samples == 4 and review.median_views == 100
    by = {p.package_id: p for p in review.pieces}
    assert by["c"].verdict == "strong" and by["c"].views_ratio == 4.0
    assert by["d"].verdict == "weak" and by["d"].views_ratio == 0.2
    assert by["a"].verdict == "normal"
    text = review.render()
    assert "▲ 作品c" in text and "▼ 作品d" in text


def test_完播率也要过线才算强():
    store = MetricsStore()
    store.add(_metric("a", 100, completion=0.5))
    store.add(_metric("b", 100, completion=0.5))
    store.add(_metric("c", 400, completion=0.2))  # 播放高但完播差
    review = Reviewer(store).summarize()
    assert {p.package_id: p.verdict for p in review.pieces}["c"] == "normal"


def test_样本不足不判定不提炼():
    store = MetricsStore()
    store.add(_metric("a", 100))
    store.add(_metric("b", 1000))
    review = Reviewer(store).summarize()
    assert all(p.verdict == "normal" for p in review.pieces)
    assert "样本不足" in review.render()
    assert Feedback(MemoryStore()).distill(review) == []
    assert "还没有任何数据" in Reviewer(MetricsStore()).summarize().render()


def test_提炼进账号层记忆且去重():
    store = MetricsStore()
    # 样本门槛是 8（少了一条偶然爆款就改写账号偏好）
    for pid, v in (("a", 100), ("b", 100), ("c", 400), ("d", 20),
                   ("e", 100), ("f", 100), ("g", 100), ("h", 100)):
        store.add(_metric(pid, v, title=f"露营{pid}", tags=["露营", "装备"]))
    memories = MemoryStore()
    fb = Feedback(memories)
    new = fb.distill(Reviewer(store).summarize())
    assert len(new) == 2
    strong = next(m for m in new if "优于" in m.content)
    weak = next(m for m in new if "不及" in m.content)
    assert all(m.layer is Layer.ACCOUNT and m.source is Source.DATA for m in new)
    assert strong.keywords[0].polarity is Polarity.POSITIVE
    assert weak.keywords[0].polarity is Polarity.NEUTRAL and "慎重复制" in weak.content
    assert {k.term for k in strong.keywords} >= {"露营", "装备"}
    assert fb.distill(Reviewer(store).summarize()) == [], "同一个包只提炼一次"

    brief = build_brief(memories)
    assert any("优于" in x for x in brief.should) and any("慎重复制" in x for x in brief.should)
    assert not brief.must and not brief.must_not, "数据结论是建议，不是禁令"


async def test_接口位_人工源不主动拉():
    assert await ManualSource().fetch("p1", "https://x") is None


# ---------------------------------------------------------------- 完整闭环


CLEAN = "本内容由 AI 生成。{}装备怎么选：先看睡袋温标，再看帐篷防水。"


async def test_完整闭环_存稿到账号层记忆(tmp_path: Path):
    store = AssetStore(tmp_path / "assets")
    memories = MemoryStore()
    metrics = MetricsStore(tmp_path / "analytics")
    checker = ComplianceChecker(RULES, memories)
    packager = Packager(tmp_path / "releases", CATALOG, store)
    bus = EventBus()
    reg = ToolRegistry(bus)
    reg.register(ContentFunctions(store))
    reg.register(ComplianceFunctions(checker, store, bus))
    reg.register(DistributionFunctions(packager, checker, store, bus))
    reg.register(
        # 这条测闭环的机制，样本门槛放低到 3（默认 8，见 test_提炼进账号层记忆且去重）
    AnalyticsFunctions(
        metrics, Reviewer(metrics), Feedback(memories, min_samples=3), packager, store, bus
    )
    )
    reg.register(RetrievalFunctions(RetrievalHub([AssetSource(store), MemorySource(memories)])))
    await reg.refresh()

    pkgs: list[str] = []
    for topic in ("露营", "钓鱼", "骑行"):
        r = await reg.invoke("save_draft", {"content": CLEAN.format(topic), "kind": "copy"})
        draft = r.asset_ref
        r = await reg.invoke("check_compliance", {"asset_id": draft, "platform": "xiaohongshu"})
        assert r.ok and "通过" in r.content
        r = await reg.invoke(
            "build_release_package",
            {
                "content_asset_id": draft,
                "platform": "xiaohongshu",
                "title": f"{topic}装备怎么选",
                "tags": [topic, "户外"],
            },
        )
        assert r.ok, r.error
        pkgs.append(r.asset_ref)
        r = await reg.invoke("mark_published", {"package_asset_id": r.asset_ref, "url": f"https://xhs/{topic}"})
        assert r.ok

    for pid, views in zip(pkgs, (100, 100, 800), strict=True):
        r = await reg.invoke(
            "record_metrics",
            {"package_asset_id": pid, "views": views, "likes": views // 10, "completion_rate": 0.6},
        )
        assert r.ok, r.error
    assert "提炼进账号层记忆" in r.content and "骑行" in r.content

    account = memories.all(layer=Layer.ACCOUNT)
    assert len(account) == 1 and account[0].source is Source.DATA
    assert "骑行" in account[0].content and "8.0 倍" in account[0].content
    assert metrics.all()[-1].url == "https://xhs/骑行", "没传 url 就用发布时记回的链接"

    brief = build_brief(memories)
    assert any("骑行" in x for x in brief.should)

    r = await reg.invoke("search_library", {"query": "骑行", "kind": "memory"})
    assert r.ok and "[memory]" in r.content and "版权:data" in r.content

    r = await reg.invoke("review_performance", {"platform": "xiaohongshu"})
    assert r.ok and store.get(r.asset_ref).type is AssetType.REPORT and "▲" in r.content
    r = await reg.invoke("list_metrics", {})
    assert r.ok and r.content.count("- mt_") == 3

    kinds = {e.type for e in bus.history}
    assert {
        EventType.COMPLIANCE_CHECKED, EventType.PACKAGE_BUILT,
        EventType.PACKAGE_PUBLISHED, EventType.METRICS_RECORDED,
    } <= kinds


async def test_录数据前必须是待发布包(tmp_path: Path):
    store = AssetStore()
    metrics = MetricsStore()
    packager = Packager(tmp_path / "r", CATALOG, store)
    bus = EventBus()
    reg = ToolRegistry(bus)
    fns = AnalyticsFunctions(
        metrics, Reviewer(metrics), Feedback(MemoryStore()), packager, store, bus
    )
    reg.register(fns)
    await reg.refresh()
    plain = store.create("不是包", creator="model:k3")
    r = await reg.invoke("record_metrics", {"package_asset_id": plain.id, "views": 10})
    assert not r.ok and "不是待发布包" in r.error
    r = await reg.invoke("record_metrics", {"package_asset_id": "as_0000000000", "views": 10})
    assert not r.ok

"""缺口 B（2026-09-23 审查）：钱的护栏看不见视频。

审查结论：媒体目录一个单价都没填 → 金额口径看不见视频；视频次数上限按「一集 6 段」定成 14，
一集 4 分钟 16–28 段，每集渲到第 15 段必停（真实会话 36 分钟弹 40 次，人被训练成一路点「是」）；
/auto 下次数超限自动放行；开工前不确认额度（用户规则没落地）；挂起问人、被参考图门拦下这些
一分钱没花的调用也占着额度。
"""

from __future__ import annotations

from pathlib import Path

from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.functions.media import MediaFunctions
from aigc_agent.domain.generators.catalog import MediaCatalog
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.model.budget import CostGuard
from aigc_agent.harness.model.ledger import CostLedger
from aigc_agent.harness.model.media import FakeMediaProvider, MediaGateway, MediaKind
from aigc_agent.harness.permission.gate import PermissionGate
from aigc_agent.harness.tools.registry import ToolRegistry
from aigc_agent.interfaces.cli.budget_prompt import parse_budget, render_budget

CATALOG = Path(__file__).resolve().parents[1] / "config" / "media_models.yaml"


# ---------------------------------------------------------------- 秒数口径 + 退还


def test_视频秒数口径_不依赖单价也拦得住(tmp_path: Path):
    g = CostGuard(seconds_limit=60, daily_seconds_limit=100,
                  ledger=CostLedger(tmp_path / "l.jsonl"))
    assert g.check("video", units=4, seconds=60)
    g.record_call("video", n=4, seconds=60)
    v = g.check("video", units=1, seconds=15)
    assert not v and v.dimension == "seconds" and v.level == "task" and "60 秒" in v.reason
    g.allow_more("video", n=1, seconds=15)
    assert g.check("video", units=1, seconds=15)
    # 日级：台账跨会话累计
    g2 = CostGuard(daily_seconds_limit=100, ledger=CostLedger(tmp_path / "l.jsonl"))
    v2 = g2.check("video", units=1, seconds=50)
    assert not v2 and v2.level == "day"


def test_没真花钱的退回额度_台账也跟着冲回(tmp_path: Path):
    led = CostLedger(tmp_path / "l.jsonl")
    g = CostGuard(call_limits={"video": 3}, seconds_limit=45, ledger=led)
    g.record_call("video", n=3, seconds=45, money=3.0)
    assert not g.check("video", units=1, seconds=5)
    g.refund("video", n=2, seconds=30, money=2.0)
    assert g.usage.calls["video"] == 1 and g.usage.seconds == 15 and abs(g.usage.money - 1) < 1e-9
    assert led.calls("video", day=led.summary()["today"]) == 1  # 台账里记了负数冲回
    assert g.check("video", units=1, seconds=15)


def test_开工额度_set与limits往返():
    g = CostGuard()
    g.set_limits(money=300, video_calls=80, video_seconds=900, image_calls=100)
    assert g.limits() == {"money": 300.0, "video_calls": 80, "video_seconds": 900.0,
                          "image_calls": 100}
    g.set_limits(video_seconds=1200)  # 给了哪项改哪项
    assert g.limits()["video_calls"] == 80 and g.limits()["video_seconds"] == 1200


# ---------------------------------------------------------------- 闸门按预估事前拦


async def _media(guard: CostGuard, provider=None) -> tuple[ToolRegistry, MediaFunctions]:
    catalog = MediaCatalog.load(CATALOG)
    bus = EventBus()
    gw = MediaGateway({catalog.provider: provider or FakeMediaProvider(
        urls=["https://x/a.mp4"])}, bus, poll_interval=0.01, max_poll_interval=0.02)
    fns = MediaFunctions(gw, catalog, AssetStore())
    fns.video_lock = next(m.id for m in catalog.video if m.id.startswith("seedance-2.0"))
    fns.image_lock = next(iter(catalog.image)).id
    reg = ToolRegistry(bus)
    reg.register(fns)
    await reg.refresh()
    reg.gate = PermissionGate(bus, guard=guard)
    return reg, fns


async def test_批量视频按总秒数事前拦_一段都不发():
    guard = CostGuard(seconds_limit=40)
    reg, _ = await _media(guard)
    jobs = [{"prompt": f"镜头{i}", "duration": 15, "allow_no_refs": True} for i in range(3)]
    meta = reg.meta_for_call("gen_videos", {"jobs": jobs})
    assert meta is not None and meta.estimate["units"] == 3 and meta.estimate["seconds"] == 45
    r = await reg.invoke("gen_videos", {"jobs": jobs, "allow_no_refs": True})
    assert not r.ok and "45 秒" in (r.error or "") and "40 秒" in (r.error or "")
    assert guard.usage.seconds == 0 and not guard.usage.calls.get("video")


async def test_被参考图门拦下_不占额度():
    guard = CostGuard(call_limits={"video": 5})
    reg, fns = await _media(guard)
    fns.ref_guard = lambda prompt, summary: "已拦截：短剧镜头没带参考图"
    r = await reg.invoke("gen_video", {"prompt": "第1集 镜3 陆离开门", "duration": 10})
    assert not r.ok and "已拦截" in (r.error or "")
    assert guard.usage.calls.get("video", 0) == 0 and guard.usage.seconds == 0


async def test_换模型挂起问人_不占额度():
    guard = CostGuard(call_limits={"video": 5})
    reg, fns = await _media(guard)
    other = next(m.id for m in fns.catalog.video if m.id != fns.video_lock)
    r = await reg.invoke("gen_video", {"prompt": "空镜", "model": other, "allow_no_refs": True})
    assert r.suspend and guard.usage.calls.get("video", 0) == 0


async def test_生图按张数计():
    guard = CostGuard(call_limits={"image": 3})
    reg, _ = await _media(guard, FakeMediaProvider(urls=["https://x/a.png", "https://x/b.png"]))
    assert (await reg.invoke("gen_image", {"prompt": "x", "n": 2})).ok
    assert guard.usage.calls["image"] == 2
    r = await reg.invoke("gen_image", {"prompt": "y", "n": 2})
    assert not r.ok and "这次要 2 个" in (r.error or "")


def test_目录按秒计价():
    catalog = MediaCatalog.load(CATALOG)
    m = next(x for x in catalog.video if x.id.startswith("seedance-2.0"))
    assert catalog.price_of(MediaKind.VIDEO, m.id, {"duration": 10}) is None  # 没填单价
    m.price_per_second = 0.1
    m.resolution_factor = {"480p": 0.5}
    assert catalog.price_of(MediaKind.VIDEO, m.id, {"duration": 10}) == 1.0
    assert catalog.price_of(MediaKind.VIDEO, m.id, {"duration": 10, "resolution": "480p"}) == 0.5
    assert catalog.priced


# ---------------------------------------------------------------- 开工确认的输入解析


def test_开工额度的输入解析():
    assert parse_budget("金额 300 视频秒 900 视频 80 图 100") == {
        "money": 300, "video_seconds": 900, "video_calls": 80, "image_calls": 100,
    }
    assert parse_budget("金额：300，秒=1200") == {"money": 300, "video_seconds": 1200}
    assert parse_budget("视频段数 40") == {"video_calls": 40}
    assert parse_budget("好的") == {}
    text = render_budget({"money": 200, "video_calls": 60, "video_seconds": 600,
                          "image_calls": 80}, priced=False)
    assert "¥200" in text and "600 秒" in text and "没配单价" in text


# ---------------------------------------------------------------- 快照：额度、挂起人审


async def test_额度与挂起的人审跟着会话快照走(tmp_path: Path):
    from aigc_agent.app import Agent

    ws = tmp_path / "ws"
    a = Agent.create(session_id="s-budget", workspace=ws)
    try:
        a.apply_budget({"money": 321, "video_calls": 33, "video_seconds": 444, "image_calls": 55})
        turn = a.memory.new_turn()
        turn.messages.append({"role": "user", "content": "渲第 1 集"})
        turn.messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "request_review", "arguments": "{}"}}]})
        a.loop.pending_review = {"call_id": "c1", "turn_index": turn.index,
                                 "question": "这版行吗", "stage": "剧本", "assets": []}
        a.session_store.save(a.memory, pending_review=a.loop.pending_review)
    finally:
        await a.aclose()

    b = Agent.create(session_id="s-budget", workspace=ws)
    try:
        assert b.budget_defaults()["video_seconds"] == 444  # 上次确认的当默认值
        assert await b.restore_session() == 1
        assert b.loop.pending_review is not None and b.loop.pending_review["call_id"] == "c1"
    finally:
        await b.aclose()


# ---------------------------------------------------------------- /auto 按集流水


def test_流水线按这一集要花的段数和秒数查额度():
    import json

    from aigc_agent.domain.output import OutputPrefs
    from aigc_agent.domain.pipeline.episode_pipeline import EpisodePipeline

    store = AssetStore()
    shots = [{"scene_index": f"[第1集-{i}场]", "video_name": f"{i}", "video_duration": "15s",
              "description": "x"} for i in range(1, 5)]
    sid = store.create(json.dumps(shots, ensure_ascii=False), summary="提示词").id
    pipe = EpisodePipeline(registry=None, assets=store, bus=EventBus(),
                           output_prefs=OutputPrefs(Path(".")),
                           guard=CostGuard(seconds_limit=50))
    assert pipe._render_need(1, sid) == (4, 60.0)

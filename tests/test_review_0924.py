"""2026-09-24 复审：四路只读审查对 09-23 那批修复的复核，发现的问题逐条回归。

  · 短剧：参考图包重新托管后按包 id 比对 → 已付费的段全部不复用、整集重付；limit 只渲了
    前几段照样 complete=True；redo / accept 按工具说明的写法对不上、静默无效；渲了一半的集
    因为旧规格戳被重出提示词、全集重渲；only=characters 重生成了脸旧脸服装留在包里；redo
    的旧段仍 active 占着主文件名；「邻家女孩气质」被当成未成年
  · 钱：MediaGateway.close 被误缩进成死代码；expired / canceled 当 running；台账增量读漏行；
    没提交出去的调用照样计次；取回的任务照样计次
  · 短视频：复用不看参考图 / 档位；文件名里的「第2镜」被当成新镜头；参考图只是提醒过期就整次拒
  · 内核：用户约束里的数字被当成错误码两天作废；简报预筛把 50 条都算命中；workspace 根下的
    台账和 rpa 登录态不算 Agent 的状态；产物目录相对路径派生出另一个项目
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from aigc_agent.capabilities.memory.brief import transient_ttl
from aigc_agent.capabilities.memory.store import Category, Keyword, Memory, MemoryStore, Polarity
from aigc_agent.domain.assets.store import AssetStatus, AssetStore, AssetType
from aigc_agent.domain.distribution import _is_generated
from aigc_agent.domain.drama.refpack import load_pack, pack_identity
from aigc_agent.domain.functions.drama import DramaFunctions, _label_hit
from aigc_agent.domain.functions.files import FileFunctions, FsPolicy
from aigc_agent.domain.functions.short_video import ShortVideoFunctions
from aigc_agent.domain.output import OutputPrefs
from aigc_agent.domain.pipeline.short_video import parse_material_reply
from aigc_agent.domain.realism import is_minor
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.model.ledger import CostLedger, LedgerEntry
from aigc_agent.harness.model.media import MediaGateway, TaskStatus, normalize_status
from aigc_agent.harness.tools.provider import ToolResult
from tests.test_drama_stage_fixes import Registry as AssetRegistry
from tests.test_drama_stage_fixes import _lib, _pack
from tests.test_episode_pipeline import FakeRegistry, _build, _prepopulate, _settle
from tests.test_gate_chain import _fns as _gate_fns
from tests.test_gate_chain import _seed
from tests.test_short_video import RECIPES, Gateway, _catalog, _plan
from tests.test_short_video import Registry as SvRegistry

# ---------------------------------------------------------------- 短剧：片段复用


def test_redo_accept_认工具说明教的写法():
    key = ("[第1集-2场]", "5-9")
    forms = ("第1集-2场 镜5-9", "镜5-9", "5-9", "第1集-2场", "[第1集-2场] 5-9",
             "【第1集-2场】镜 5-9")
    for w in forms:
        assert _label_hit(key, [w]), w
    assert not _label_hit(key, ["5"]) and not _label_hit(key, ["9"]), "不按子串瞎命中"
    assert _label_hit(key, "镜5-9"), "模型传成字符串也按整条认，不按字符迭代"
    assert not _label_hit(("[第1集-1场]", "1-4"), "5-9")


def test_参考图包签名_换链接不变_换脸才变():
    a = {"安妮": {"asset": "as_1", "url": "https://a/1.png"}, "景": {"asset": "as_2", "url": "u"}}
    b = {"安妮": {"asset": "as_1", "url": "https://b/1.png?x"}, "景": {"asset": "as_2", "url": "v"}}
    assert pack_identity(a) == pack_identity(b), "重新托管只换链接，签名不变"
    c = {**a, "安妮": {"asset": "as_9", "url": "https://a/9.png"}}
    assert pack_identity(a) != pack_identity(c)
    assert pack_identity({}) == "" and load_pack("not json") == {}


async def test_参考图包重新托管后_已付费的段照样复用_换脸才重渲():
    store = AssetStore()
    fns, reg, _ = _gate_fns(store, {})
    shots_id, pack_id = _seed(store)
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.ok and len(reg.calls) == 2

    # 托管 / 刷新链接：新建一个包资产，图（资产 id）没变
    old = json.loads(store.content(pack_id))
    pack2 = store.create(
        json.dumps({k: {**v, "url": v["url"] + "?rehosted"} for k, v in old.items()}),
        summary="参考图包·重新托管", creator="tool:drama_render_assets", parents=[pack_id],
    )
    reg.calls.clear()
    r2 = await fns._fn_drama_render_shots(shots_id, rendered_id=pack2.id)
    assert r2.ok and not reg.calls, "图没换，两段都该复用，不该重付"

    # 真换了脸（新图片资产）：不复用
    img2 = store.create("", type_=AssetType.IMAGE, summary="角色·安妮·新脸", creator="model:x")
    img2.uri = "https://img/anne2.png"
    store.put(img2)
    pack3 = store.create(
        json.dumps({"安妮": {"asset": img2.id, "url": img2.uri, "kind": "角色"}}),
        summary="参考图包·换脸", creator="tool:drama_render_assets", parents=[pack_id],
    )
    r3 = await fns._fn_drama_render_shots(shots_id, rendered_id=pack3.id)
    assert r3.ok and len(reg.calls) == 2, "换了脸就不复用旧段"


def test_老标签只记包id_也按内容签名比():
    store = AssetStore()
    fns, _, _ = _gate_fns(store, {})
    shots_id, pack_id = _seed(store)
    clip = store.create(
        "", type_=AssetType.VIDEO, summary="[第1集-1场] 1-4", creator="model:x",
        gen_params={"tags": {"shots_id": shots_id, "scene": "[第1集-1场]", "name": "1-4",
                             "pack": pack_id, "accepted": True}},
    )
    clip.uri = "https://fake/c.mp4"
    store.put(clip)
    old = json.loads(store.content(pack_id))
    pack2 = store.create(
        json.dumps({k: {**v, "url": v["url"] + "?x"} for k, v in old.items()}),
        summary="包2", creator="tool:drama_render_assets", parents=[pack_id],
    )
    sig = pack_identity(load_pack(store.content(pack2.id)))
    assert fns._previous_clips(shots_id, 1, pack2.id, pack_sig=sig) == {
        ("[第1集-1场]", "1-4"): clip.id
    }


async def test_limit只渲前几段_不算渲完_去掉limit后只补其余():
    store = AssetStore()
    fns, reg, _ = _gate_fns(store, {})
    shots_id, pack_id = _seed(store)
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, limit=1, episode=1)
    assert r.ok and len(reg.calls) == 1 and not reg.composed, "limit 时不拼成片"
    assert r.meta["complete"] is False and "只渲了前 1/2 段" in r.content
    idx = store.get(r.asset_ref)
    assert idx.gen_params["complete"] is False and idx.gen_params["limit"] == 1

    reg.calls.clear()
    r2 = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, episode=1)
    assert r2.ok and [c["summary"] for c in reg.calls] == ["[第1集-2场] 5-8"], "只补没渲的那段"
    assert r2.meta["complete"] is True and reg.composed


async def test_redo点名的旧段挪进废弃_不再active():
    store = AssetStore()
    fns, reg, _ = _gate_fns(store, {})
    shots_id, pack_id = _seed(store)
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.ok
    clip = next(
        a for a in store.find(type_=AssetType.VIDEO) if a.gen_params["tags"]["name"] == "5-8"
    )
    reg.calls.clear()
    r2 = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, redo=["镜5-8"])
    assert r2.ok and [c["summary"] for c in reg.calls] == ["[第1集-2场] 5-8"]
    old = store.get(clip.id)
    assert old.status is AssetStatus.REJECTED and "redo" in old.gen_params["tags"]["rejected"]


async def test_只重渲主形象_旧脸服装不留在新包里():
    store = AssetStore()
    lib_id = _lib(store)
    fns = DramaFunctions(None, store, registry=AssetRegistry(store))
    first = await fns._fn_drama_render_assets(lib_id)
    p1 = _pack(store, first.asset_ref)
    r = await fns._fn_drama_render_assets(lib_id, only="characters", reuse=False)
    p2 = _pack(store, r.asset_ref)
    assert p2["小满"]["asset"] != p1["小满"]["asset"]
    assert "小满-旧连帽衫-[1-3]" not in p2, "按旧脸生成的服装图不能留在新包里"
    assert "旧脸" in r.content


def test_未成年判定_不把气质形容当年龄():
    adults = ("邻家女孩气质的女记者", "少女感十足的 26 岁白领", "两个孩子的妈，38岁",
              "少年气的男演员", "女 | 30岁 | 记者，带着孩子的单亲妈妈")
    for t in adults:
        assert not is_minor(t), t
    for t in ("16岁流浪少女", "少女，扎双马尾", "青少年", "萌宝", "小学生", "a teenage boy",
              "孩子，圆脸大眼"):
        assert is_minor(t), t


# ---------------------------------------------------------------- 按集流水


async def test_渲了一半的集_旧规格也不重出提示词(tmp_path):
    store, bus = AssetStore(), EventBus()
    old = _prepopulate(store, "S1")
    store.create("[]", summary="片段·第1集", creator="tool:drama_render_shots",
                 gen_params={"episode": 1, "complete": False, "failed": 1})
    fake = FakeRegistry(store)
    fake.spec = "S2"
    pipe = _build(store, bus, fake, tmp_path, total=1, spec="S2")
    try:
        pipe.kick()
        await _settle(pipe)
        assert fake.count("drama_shots") == 0, "渲过一部分的集不重出（重出会换 shots_id）"
        renders = fake.args_of("drama_render_shots")
        assert len(renders) == 1 and renders[0]["shots_id"] == old
    finally:
        await pipe.aclose()


async def test_参考图包重新托管_流水线不当输入变了(tmp_path):
    store, bus = AssetStore(), EventBus()
    pipe = _build(store, bus, FakeRegistry(store), tmp_path, total=1, spec="S2")
    try:
        p1 = store.create(json.dumps({"安妮": {"asset": "as_1", "url": "u1"}}), summary="包1",
                          creator="tool:drama_render_assets")
        v1 = pipe._scan()
        p2 = store.create(json.dumps({"安妮": {"asset": "as_1", "url": "u2"}}), summary="包2·托管",
                          creator="tool:drama_render_assets", parents=[p1.id])
        v2 = pipe._scan()
        assert v1["refs"] == p1.id and v2["refs"] == p2.id
        assert v1["refs_sig"] == v2["refs_sig"] != ""
    finally:
        await pipe.aclose()


# ---------------------------------------------------------------- 钱


def test_媒体网关有close_终态词表():
    assert callable(getattr(MediaGateway, "close", None)), "close 曾被缩进成模块级函数里的死代码"
    for s in ("expired", "canceled", "cancelled", "rejected", "timeout"):
        assert normalize_status(s) is TaskStatus.FAILED, s
    assert normalize_status("processing") is TaskStatus.RUNNING


def test_台账增量读不漏别的进程插进来的行(tmp_path):
    day = "2026-09-24"
    path = tmp_path / "l.jsonl"
    a, b = CostLedger(path), CostLedger(path)
    a.append(LedgerEntry(kind="video", calls=1, seconds=10, day=day))
    b.append(LedgerEntry(kind="video", calls=1, seconds=10, day=day))
    # 模拟：a 在 append 里 _load() 读完之后、写自己这行之前，b 插了一行
    orig = a._load
    state = {"done": False}

    def load_then_other_writes(until: int | None = None) -> None:
        orig(until)
        if until is None and not state["done"]:
            state["done"] = True
            b.append(LedgerEntry(kind="video", calls=1, seconds=7, day=day))

    a._load = load_then_other_writes  # type: ignore[method-assign]
    a.append(LedgerEntry(kind="video", calls=1, seconds=5, day=day))
    a._load = orig  # type: ignore[method-assign]
    a._load()
    assert a.seconds(day=day) == 32 and a.calls("video", day=day) == 4, "b 插的那行不能被跳过"
    assert CostLedger(path).seconds(day=day) == 32


async def test_没提交出去的调用退额度(tmp_path):
    from tests.test_media_leftovers import _media

    fns, _ = _media(tmp_path)
    r = await fns.invoke("gen_video", {"prompt": "x", "nonsense": 1})
    assert not r.ok and r.meta.get("charged") is False, "参数错没提交，闸门记的账要退"
    r2 = await fns.invoke("gen_videos", {"jobs": []})
    assert not r2.ok and r2.meta.get("charged") is False


def test_同一个任务自动取回有上限(tmp_path):
    from aigc_agent.harness.model.task_ledger import MediaTaskLedger

    led = MediaTaskLedger(tmp_path / "t.jsonl", max_auto_recoveries=2)
    led.submitted("t1", kind="video", model="m", provider="p", fingerprint="fp", prompt="x")
    led.update("t1", "timeout", error="没等到")
    assert led.recoverable("fp") is not None
    led.recovered("t1")
    led.update("t1", "timeout", error="又没等到")
    assert led.recoverable("fp") is not None, "取回一次还没到上限"
    led.recovered("t1")
    assert led.recoverable("fp") is None, "取回两次都没等到：不再自动取回，改为正常提交"
    assert led.get("t1").undelivered, "人仍可用 media_recover 点名取回"


# ---------------------------------------------------------------- 短视频


def test_短视频复用只认同样的生成条件():
    store = AssetStore()
    fns = ShortVideoFunctions(Gateway(_plan()), store, registry=SvRegistry(store),
                              catalog=_catalog(), recipes_dir=RECIPES)
    brief = store.create("{}", summary="简报", creator="tool:short_video_brief")
    clip = store.create("", type_=AssetType.VIDEO, summary="第1镜", creator="model:x")
    clip.uri = "https://f/1.mp4"
    store.put(clip)
    same: dict[str, Any] = {"model": "", "tier": "fast", "refs": [], "shot_refs": {}}
    store.create(json.dumps({"clips": {"1": clip.id}, "gen": same}), summary="出片记录",
                 parents=[brief.id], creator="tool:short_video_produce")
    assert fns._previous_clips(brief.id, same) == {1: clip.id}
    with_ref = {**same, "refs": ["as_prod"]}
    assert fns._previous_clips(brief.id, with_ref) == {}, "这次带了产品图，旧的无产品片段不复用"
    assert fns._previous_clips(brief.id, {**same, "tier": "quality"}) == {}
    # 老记录没记条件：这次也没带参考图才算同样
    store.create(json.dumps({"clips": {"2": clip.id}}), summary="老出片记录",
                 parents=[brief.id], creator="tool:short_video_produce")
    assert fns._previous_clips(brief.id, same).get(2) == clip.id
    assert 2 not in fns._previous_clips(brief.id, {**same, "refs": ["as_prod"]})


def test_文件名里的第N镜不是新镜头():
    got = parse_material_reply(r"第3镜 E:\素材\第2镜.mp4", [2, 3])
    assert got == {3: ("file", r"E:\素材\第2镜.mp4")}
    assert parse_material_reply(r"第3镜：E:\a.mp4", [3]) == {3: ("file", r"E:\a.mp4")}
    assert parse_material_reply("第1镜 生成，第2镜 联网找", [1, 2]) == {
        1: ("generate", ""), 2: ("online", "")
    }


async def test_参考图有链接只是提醒可能过期_照用():
    store = AssetStore()
    a = store.create("", type_=AssetType.IMAGE, summary="产品图", creator="human")
    a.uri = "https://x/p.png"
    store.put(a)

    async def ensure(store_: Any, asset: Any, ttl: Any) -> tuple[str, str]:
        return "https://x/p.png", "链接可能已过期，没有本地副本"

    hosting = SimpleNamespace(enabled=True, ensure_asset=ensure)
    fns = ShortVideoFunctions(None, store, registry=None, catalog=None, hosting=hosting)
    urls, err = await fns._ref_urls([a.id])
    assert urls == ["https://x/p.png"] and not err


async def test_字幕门重生成失败_如实说不误报():
    store = AssetStore()
    plan = _plan(shots=[{"desc": "晶圆厂机械臂", "seconds": 10, "source": "generate"}])
    reg = SvRegistry(store)
    fns = ShortVideoFunctions(Gateway(plan), store, registry=reg, catalog=_catalog(),
                              recipes_dir=RECIPES)
    b = await fns.invoke("short_video_brief", {"keyword": "k", "style": "tech-short"})
    assert b.ok, b.error

    async def burned(aid: str) -> tuple[bool | None, str]:
        return True, "左下角"

    fns._burned_text = burned  # type: ignore[method-assign]
    calls = {"n": 0}
    orig = reg.invoke

    async def inv(name: str, args: dict[str, Any]) -> ToolResult:
        if name == "gen_video":
            calls["n"] += 1
            if calls["n"] >= 2:
                return ToolResult(ok=False, error="服务端 500")
        return await orig(name, args)

    reg.invoke = inv  # type: ignore[method-assign]
    fns.approve(b.asset_ref)  # 人在确认单上点了头（2026-09-26：confirm=true 只认人的确认）
    r = await fns.invoke("short_video_produce", {
        "brief_id": b.asset_ref, "confirm": True, "no_voiceover": True, "no_subtitle": True,
    })
    text = (r.content or "") + (r.error or "")
    assert "重生成失败（服务端 500）" in text, text
    assert "重生成后仍有" not in text


def test_别人的原片不标AIGC():
    store = AssetStore()
    mine = store.create("", type_=AssetType.VIDEO, summary="生成的", creator="model:seedance")
    theirs = store.create("", type_=AssetType.VIDEO, summary="抓的", creator="tool:fetch_douyin")
    assert _is_generated(mine) and not _is_generated(theirs)


# ---------------------------------------------------------------- 内核 / 记忆 / 文件


def test_用户约束里的数字不是错误码():
    for t in ("预算上限 400 元", "每集控制在 500 字以内", "视频总长 420 秒", "生成超时要重试"):
        assert transient_ttl(t) is None, t
    for t in ("seedance 402 余额不足", "报错 429", "HTTP 500", "返回 503 错误", "接口 504 error"):
        assert transient_ttl(t) is not None, t


def test_简报预筛不计命中(tmp_path):
    s = MemoryStore(tmp_path)
    m = s.put(Memory(content="开头不要硬广", project_id="p1",
                     keywords=[Keyword(term="硬广", polarity=Polarity.NEGATIVE,
                                       category=Category.CONSTRAINT)]))
    s.recall(["硬广"], project_id="p1", touch=False)
    assert s.get(m.id).hit_count == 0, "预筛不算命中"
    s.recall(["硬广"], project_id="p1")
    assert s.get(m.id).hit_count == 1


def test_workspace根下的台账和rpa登录态也是Agent的状态(tmp_path):
    ws, out = tmp_path / "ws", tmp_path / "out"
    fns = FileFunctions(AssetStore(), ws, OutputPrefs(out), FsPolicy(), project_root=tmp_path / "p")
    assert fns.protected(ws / "media_tasks.jsonl")
    assert fns.protected(ws / "blobs" / "x.bin")
    assert fns.protected(ws / "rpa" / "profile" / "Default" / "Local State")
    assert not fns.protected(out / "第1集.mp4"), "产物目录永远不算"
    assert fns.denied(ws / "rpa" / "profile" / "Default" / "Local State"), "浏览器 profile 硬拒绝"


def test_产物目录先resolve(tmp_path, monkeypatch):
    from aigc_agent.interfaces.cli.main import _ensure_dir

    monkeypatch.chdir(tmp_path)
    p = _ensure_dir("蜘蛛精")
    assert p.is_absolute() and p == (tmp_path / "蜘蛛精").resolve()


# ================================================================ 第二批（同日续修）


def test_redo写第N集_认整集的段():
    key = ("[第1集-2场]", "5-9")
    assert _label_hit(key, ["第1集"])
    assert not _label_hit(key, ["第11集"]) and not _label_hit(("[第11集-2场]", "5-9"), ["第1集"])


def test_未成年判定_英文按整词_还是个孩子算():
    assert is_minor("她还是个孩子") and is_minor("a teen boy") and is_minor("two kids")
    for t in ("a childhood friend", "the kidnapper", "eighteen and restless"):
        assert not is_minor(t), t


def test_字数时长不是错误码():
    assert transient_ttl("每次返回 500 字") is None
    assert transient_ttl("接口返回 500") is not None


def test_素材回复_不带分隔符的第N镜照样认_文件名里的不算():
    assert parse_material_reply("第1镜生成第2镜联网找", [1, 2]) == {
        1: ("generate", ""), 2: ("online", "")
    }
    assert parse_material_reply("第3镜 素材第2镜.mp4", [2, 3]) == {3: ("file", "素材第2镜.mp4")}


def test_剪辑表_素材比计划短_差额摊回别的刀():
    from aigc_agent.domain.pipeline.cutting import plan_cuts

    for seed in (None, 1, 7):
        cuts = plan_cuts([8.0, 2.0, 8.0], 30.0, 3.0, 1.2, seed=seed, ordered=True)
        assert abs(sum(c.dur for c in cuts) - 30.0) < 0.02, seed
        assert all(c.dur <= 3.0 + 1e-6 for c in cuts)
        assert all(c.start >= -1e-6 and c.end <= [8.0, 2.0, 8.0][c.clip] + 1e-6 for c in cuts)
    # 素材总量不够：只能短一点，但每刀不越界
    cuts = plan_cuts([2.0, 2.0], 12.0, 3.0, 1.2)
    assert all(c.end <= 2.0 + 1e-6 and c.dur <= 3.0 for c in cuts)


def test_图片按真实格式落盘(tmp_path):
    from aigc_agent.domain.functions.media import _fix_image_ext

    p = tmp_path / "x.png"
    p.write_bytes(bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"0" * 16)
    q = _fix_image_ext(p)
    assert q.name == "x.jpg" and q.exists() and not p.exists()
    png = tmp_path / "y.png"
    png.write_bytes(bytes([0x89]) + b"PNG" + b"0" * 16)
    assert _fix_image_ext(png) == png


async def test_没本地副本_镜头门要留痕():
    store = AssetStore()
    clip = store.create("", type_=AssetType.VIDEO, summary="段", creator="model:x")
    clip.uri = "https://fake/c.mp4"
    store.put(clip)
    fns = DramaFunctions(None, store, registry=None)
    longest, why = await fns._check_cuts(clip.id, 0.3)
    assert longest is None and "没有本地副本" in why


def test_引用门拦短视频时指路ref_images():
    store = AssetStore()
    _lib(store)
    fns = DramaFunctions(None, store, registry=None)
    why = fns.reference_guard("小满在地铁里跑", "产品广告·第1镜")
    assert why.startswith("⛔") and "short_video_produce" in why and "ref_images" in why


async def test_抖音点热榜_结果说清不是按关键词搜的(monkeypatch, tmp_path):
    from contextlib import asynccontextmanager

    from aigc_agent.domain.functions import rpa as rpa_mod
    from aigc_agent.domain.rpa.interact import Captured

    @asynccontextmanager
    async def session(_profile):
        yield SimpleNamespace(context=None, goto=_noop)

    async def _noop(*a, **k):
        return None

    async def capture(s, site, cards, count, pace):
        caps = [Captured(platform="douyin", url="https://v/1", title="AI 芯片又涨价了"),
                Captured(platform="douyin", url="https://v/2", title="明星婚讯")]
        return caps, ""

    monkeypatch.setattr(rpa_mod, "have_playwright", lambda: True)
    monkeypatch.setattr(rpa_mod, "BrowserSession", session)
    monkeypatch.setattr(rpa_mod, "grant_clipboard", _noop)
    monkeypatch.setattr(rpa_mod, "browse_and_capture", capture)
    store = AssetStore()
    fns = rpa_mod.RpaFunctions(store, tmp_path)
    r = await fns._fn_browse_and_copy("douyin", keyword="芯片", count=2)
    assert r.ok and "全站热榜" in r.content and "只有第 1 条" in r.content
    assert "全站热榜" in store.get(r.asset_ref).summary


async def test_快切没配音时保留片段原声(tmp_path):
    from aigc_agent.domain.media import ffmpeg

    if not ffmpeg.have_ffmpeg():
        import pytest

        pytest.skip("需要 ffmpeg")
    loud, mute = tmp_path / "loud.mp4", tmp_path / "mute.mp4"
    code, err = await ffmpeg.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=240x426:d=3:r=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(loud),
    ])
    assert code == 0, err
    code, err = await ffmpeg.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=240x426:d=3:r=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mute),
    ])
    assert code == 0, err
    out = tmp_path / "out.mp4"
    cuts = [(0, 0.0, 1.0), (1, 0.5, 1.0), (0, 1.5, 1.2)]
    ok, err = await ffmpeg.concat_cuts([loud, mute], cuts, out, keep_audio=True)
    assert ok, err
    info = await ffmpeg.probe(out)
    assert info.has_audio, "片段原声被丢了"
    assert abs(info.duration - 3.2) < 0.2, info.duration
    ok, err = await ffmpeg.concat_cuts([loud, mute], cuts, tmp_path / "v.mp4")
    assert ok and not (await ffmpeg.probe(tmp_path / "v.mp4")).has_audio, "默认仍只拼画面"


def test_video_make接了询问器(monkeypatch):
    from aigc_agent.interfaces.cli import video_cmd

    seen: dict[str, Any] = {}

    class _Stop(Exception):
        pass

    def create(*a: Any, **k: Any) -> Any:
        seen.update(k)
        raise _Stop

    monkeypatch.setattr(video_cmd.Agent, "create", staticmethod(create))
    import asyncio

    try:
        asyncio.run(video_cmd._make("科技", "tech-short", "", 0, "", "", False, False, True, True))
    except _Stop:
        pass
    assert seen.get("asker") is video_cmd._ask, "超额度要能问人，不能直接拒掉那几镜"


# ================================================================ 09-25：短剧模型在服务端用不了


def _price_error() -> Any:
    import httpx2 as httpx
    from openai import APIStatusError

    req = httpx.Request("POST", "https://api.apimart.ai/v1/chat/completions")
    msg = (
        "Error code: 400 - {'error': {'message': '模型 gemini-3.1-pro-preview "
        "的价格尚未由管理员配置，暂时无法使用，请联系站点管理员开启该模型；"
        "Model gemini-3.1-pro-preview has not been "
        "priced by the administrator yet.'}}"
    )
    return APIStatusError(msg, response=httpx.Response(400, request=req), body=None)


def test_认出模型在服务端用不了_别的错误不误判():
    import httpx2 as httpx
    from openai import APIStatusError

    from aigc_agent.harness.model.gateway import classify_model_error

    kind, hint = classify_model_error(_price_error())
    assert kind == "model_unavailable" and "换" in hint
    req = httpx.Request("POST", "https://x")
    bad_param = APIStatusError(
        "Error code: 400 - invalid temperature",
        response=httpx.Response(400, request=req),
        body=None,
    )
    assert classify_model_error(bad_param)[0] == "other", "参数错不是模型用不了"
    assert classify_model_error(FileNotFoundError("E:/x does not exist"))[0] == "other"


async def test_网关把模型用不了换成说清怎么办的错误_不重试():
    import pytest

    from aigc_agent.harness.model.gateway import ModelUnavailableError, classify_model_error
    from tests.test_net_resilience import Flaky, _gateway

    gw, _ = _gateway(EventBus(), max_attempts=3)
    flaky = Flaky(fails=9, exc=_price_error)
    gw._call_once = flaky  # type: ignore[method-assign]
    with pytest.raises(ModelUnavailableError) as ei:
        await gw.chat("main_agent", [{"role": "user", "content": "x"}])
    e = ei.value
    assert flaky.calls == 1, "400 不重试"
    assert (e.role, e.provider, e.model) == ("main_agent", "p", "m")
    text = f"{type(e).__name__}: {e}"
    for kw in ("用不了", "不要去读写", "agent models --role main_agent", "价格尚未由管理员配置"):
        assert kw in text, kw
    assert classify_model_error(e)[0] == "model_unavailable"


async def test_短剧工具把这条说明原样报给主模型():
    from aigc_agent.domain.system_prompt import SYSTEM_PROMPT
    from aigc_agent.harness.model.gateway import ModelUnavailableError

    fns = DramaFunctions(None, AssetStore(), registry=None)

    async def boom(**_: Any) -> Any:
        raise ModelUnavailableError(
            "drama", "gemini", "gemini-3.1-pro-preview", "价格尚未由管理员配置"
        )

    fns._fn_drama_assets = boom  # type: ignore[method-assign]
    r = await fns.invoke("drama_assets", {})
    assert not r.ok and r.error.startswith("ModelUnavailableError: 角色「drama」")
    assert "provider 「gemini」" in r.error and "不要去读写" in r.error
    assert "ModelUnavailableError" in SYSTEM_PROMPT, "系统提示词要告诉主模型遇到它怎么做"


async def test_fs_search_也收单个文件(tmp_path):
    ws, out = tmp_path / "ws", tmp_path / "out"
    cfg = out / "cfg"
    cfg.mkdir(parents=True)
    (cfg / "models.yaml").write_text("roles:\n  drama: gemini\n", encoding="utf-8")
    (cfg / "other.yaml").write_text("drama: x\n", encoding="utf-8")
    fns = FileFunctions(AssetStore(), ws, OutputPrefs(out), FsPolicy(), project_root=tmp_path / "p")
    r = await fns._fn_fs_search(str(cfg / "models.yaml"), "drama")
    assert r.ok, r.error
    assert "models.yaml:2:" in r.content and "other.yaml" not in r.content


# ================================================================ 09-25：集长按项目放宽（/length）


def test_集长放宽_规格跟着变_规格戳不变():
    from dataclasses import replace

    from aigc_agent.domain.drama.format import DEFAULT_FORMAT

    f8 = replace(DEFAULT_FORMAT, minutes=8)
    assert f8.seconds == 480 and f8.duration_range == (408, 552)
    assert f8.shot_range == (32, 48), "一集 8 分钟：每集约 32–48 段视频"
    assert f8.stamp == DEFAULT_FORMAT.stamp, "规格戳不含分钟数：改集长不会把已有提示词当旧规格重做"


def test_会话快照记住集长_整体重写也不丢(tmp_path):
    from aigc_agent.capabilities.memory.session import SessionSnapshot
    from aigc_agent.harness.context.window import ShortTermMemory

    s = SessionSnapshot(tmp_path, "p")
    s.set_episode_minutes(8)
    assert SessionSnapshot(tmp_path, "p").episode_minutes == 8
    s.save(ShortTermMemory())
    assert SessionSnapshot(tmp_path, "p").episode_minutes == 8, "每轮结束的整体重写不能把它丢了"
    s.set_episode_minutes(0)
    assert SessionSnapshot(tmp_path, "p").episode_minutes == 0


async def test_集长套到写剧本和拆分镜上_重启沿用(tmp_path):
    from aigc_agent.app import Agent

    ws = tmp_path / "ws"
    a = Agent.create(session_id="s-len", workspace=ws)
    try:
        base = a.base_episode_fmt.minutes
        assert a.drama_fns.fmt.minutes == base and a.episode_fns.fmt.minutes == base
        a.set_episode_minutes(8)
        assert a.drama_fns.fmt.minutes == 8 and a.episode_fns.fmt.minutes == 8
        line = a.episode_spec_line()
        assert "408–552 秒" in line and "/length" in line and "不要为了让检查通过" in line
    finally:
        await a.aclose()

    b = Agent.create(session_id="s-len", workspace=ws)
    try:
        assert b.drama_fns.fmt.minutes == 8, "重启后沿用这个项目的集长"
        b.set_episode_minutes(None)
        assert b.drama_fns.fmt.minutes == b.base_episode_fmt.minutes
        assert b.session_store.episode_minutes == 0
    finally:
        await b.aclose()


def test_length命令在命令表里_说明写清区间和段数():
    from dataclasses import replace

    from aigc_agent.domain.drama.format import DEFAULT_FORMAT
    from aigc_agent.domain.system_prompt import SYSTEM_PROMPT
    from aigc_agent.interfaces.cli.main import _length_brief
    from aigc_agent.interfaces.cli.promptbox import known_command

    assert known_command("/length 8") and known_command("/集长 8") and known_command("/length")
    agent = SimpleNamespace(
        episode_fmt=replace(DEFAULT_FORMAT, minutes=8), base_episode_fmt=DEFAULT_FORMAT
    )
    text = _length_brief(agent)  # type: ignore[arg-type]
    assert "一集 8 分钟" in text and "408–552 秒" in text and "32–48 段" in text
    assert "本项目设置" in text
    assert "/length" in SYSTEM_PROMPT, "系统提示词要告诉主模型：对不上时让用户用 /length 放宽"


async def test_流式中途断开_自动重发一次_反复断就停():
    import httpcore2
    import httpx2 as httpx
    import pytest

    from tests.test_net_resilience import Flaky, _gateway

    msg = "peer closed connection without sending complete message body (incomplete chunked read)"
    makers = (lambda: httpx.RemoteProtocolError(msg), lambda: httpcore2.RemoteProtocolError(msg))
    for make in makers:
        gw, provider = _gateway(EventBus(), max_attempts=3)
        once = Flaky(fails=1, exc=make)
        gw._call_once = once  # type: ignore[method-assign]
        resp = await gw._with_retry(provider, {}, use_stream=False)
        assert resp.text == "ok" and once.calls == 2, "断一次：自动重发成功"

        always = Flaky(fails=9, exc=make)
        gw._call_once = always  # type: ignore[method-assign]
        with pytest.raises((httpx.RemoteProtocolError, httpcore2.RemoteProtocolError)):
            await gw._with_retry(provider, {}, use_stream=False)
        assert always.calls == 2, "反复断：只重发一次，不无限花钱"


# ================================================================ 09-25：提示词不丢镜头、按场分批


def _long_storyboard(scenes: int = 3, per: int = 50, secs: float = 2.0) -> str:
    from aigc_agent.domain.drama.format import HOOK_MARK

    lines: list[str] = []
    n = 0
    for _ in range(scenes):
        lines.append("[夜] [内] [地铁车厢]")
        for _ in range(per):
            n += 1
            mark = HOOK_MARK if n == 1 else ""
            lines.append(f"{mark}[近景/手持/{secs:g}s] 陆离第{n}个动作：“台词{n}”")
    return "\n".join(lines)


class _ChunkGateway:
    """按提示词里的〔N〕镜号和【第N集-M场】场次写视频提示词：每 5 个镜头一段（10 秒）。
    drop 里的镜号故意不写（drop_always=False 时补写那一轮会写上）。"""

    def __init__(self, drop: set[int] | None = None, drop_always: bool = False) -> None:
        self.calls: list[list[dict[str, Any]]] = []
        self.drop = set(drop or ())
        self.drop_always = drop_always

    async def chat(self, role: str, messages: list[dict[str, Any]], **_: Any) -> Any:
        import re

        self.calls.append(messages)
        user = messages[1]["content"]
        fill = "【补写】" in user
        label = "第1集-1场"
        picked: list[tuple[str, int]] = []
        for line in user.splitlines():
            m = re.match(r"^【(第\d+集-\d+场)】", line)
            if m:
                label = m.group(1)
                continue
            m = re.match(r"^〔(\d+)〕", line)
            if m:
                n = int(m.group(1))
                if n in self.drop and (self.drop_always or not fill):
                    continue
                picked.append((label, n))
        rows: list[dict[str, Any]] = []
        i = 0
        while i < len(picked):
            lab = picked[i][0]
            seg = [n for lb, n in picked[i : i + 5] if lb == lab]
            i += len(seg)
            rows.append({
                "scene_index": f"[{lab}]", "video_name": f"{seg[0]}-{seg[-1]}",
                "video_duration": f"{2 * len(seg)}s", "cuts": [2] * len(seg),
                # 台词照抄进描述（真实的提示词要求台词 100% 还原）
                "description": "(地铁车厢) (陆离-风衣-[全集]) 镜头 "
                + " ".join(f"“台词{n}”" for n in seg),
                "hook": not rows and not fill and "不是开场" not in user,
            })
        return SimpleNamespace(text=json.dumps(rows, ensure_ascii=False), finish_reason="stop")


def _shots_setup(gw: Any) -> tuple[Any, AssetStore, str, str]:
    from aigc_agent.domain.drama.format import EpisodeFormat
    from tests.test_episode_format import LIB

    store = AssetStore()
    sb = store.create(
        json.dumps([{"episodeIndex": 1, "episodeTitle": "第1集",
                     "episodeDesc": _long_storyboard()}], ensure_ascii=False),
        summary="分镜", creator="tool:drama_storyboard",
    )
    lib = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库",
                       creator="tool:drama_assets")
    fns = DramaFunctions(gw, store, registry=None, catalog=None, fmt=EpisodeFormat())
    return fns, store, sb.id, lib.id


async def test_长分镜按场分批_每个镜头都写到_时长跟着分镜走():
    gw = _ChunkGateway()
    fns, store, sb, lib = _shots_setup(gw)
    r = await fns._fn_drama_shots(sb, lib)
    assert r.ok, r.error
    assert len(gw.calls) == 3, "150 镜按场分 3 批，一批一次调用，不用改写"
    saved = json.loads(store.content(r.asset_ref))
    assert len(saved) == 30 and sum(int(s["video_duration"].rstrip("s")) for s in saved) == 300
    assert "分 3 批" in r.content and "时长按分镜 300s" in r.content
    assert "✓ 规格" in r.content, "按分镜的 300 秒检查，不再拿「一集 4 分钟」去压"
    assert store.get(r.asset_ref).gen_params["chunks"] == 3
    firsts = [m[1]["content"] for m in gw.calls]
    assert sum("不是开场" in u for u in firsts) == 2, "第二批起不要求开场高潮点"
    assert all("〔" in u and "【第1集-" in u for u in firsts), "镜号和场次写明，不让模型自己数"


async def test_漏了镜头_自动补写一次():
    gw = _ChunkGateway(drop={96, 97, 98, 99, 100})
    fns, store, sb, lib = _shots_setup(gw)
    r = await fns._fn_drama_shots(sb, lib)
    assert r.ok, r.error
    fills = [m[1]["content"] for m in gw.calls if "【补写】" in m[1]["content"]]
    assert len(fills) == 1 and "96–100" in fills[0]
    saved = json.loads(store.content(r.asset_ref))
    names = [s["video_name"] for s in saved]
    assert "96-100" in names and names.index("96-100") == names.index("91-95") + 1, "按镜号插回原位"


async def test_补写后仍漏_不保存_说清缺哪几镜():
    gw = _ChunkGateway(drop={96, 97, 98, 99, 100}, drop_always=True)
    fns, store, sb, lib = _shots_setup(gw)
    r = await fns._fn_drama_shots(sb, lib)
    assert not r.ok and "96–100" in r.error and "没有保存" in r.error
    assert not store.find(creator="tool:drama_shots"), "缺一截的提示词不能落库（会渲出半集）"


def test_压缩检测_与覆盖检测():
    from aigc_agent.domain.drama.format import compression_problems, coverage_gaps
    from aigc_agent.domain.drama.models import ShotPrompt

    secs = {n: 2.0 for n in range(1, 21)}
    squeezed = [
        ShotPrompt("[第1集-1场]", "1-10", "12s", "x"),
        ShotPrompt("[第1集-2场]", "11-15", "10s", "x"),
    ]
    probs = compression_problems(squeezed, secs)
    assert len(probs) == 1 and "1-10：分镜 20s → 这段 12s" in probs[0]
    assert coverage_gaps(squeezed, set(range(1, 21))) == [(16, 20)]


def test_服装全名纠正():
    from aigc_agent.domain.drama.models import fix_costume_refs

    names = {"孙悟空-黄直裰虎皮围裙-[1-18]", "唐僧-缁衣-[1]", "唐僧-布袍-[2-20]"}
    text, fixed = fix_costume_refs("(孙悟空-黄直裰虎皮围裙-[1]) 抡棒", names, 2)
    assert text == "(孙悟空-黄直裰虎皮围裙-[1-18]) 抡棒" and len(fixed) == 1
    same, none = fix_costume_refs("(唐僧-缁衣-[1]) 合掌", names, 1)
    assert none == [] and same == "(唐僧-缁衣-[1]) 合掌", "库里就有的不动"
    odd, none2 = fix_costume_refs("(八戒-僧衣-[3]) 吃饭", names, 3)
    assert none2 == [] and odd == "(八戒-僧衣-[3]) 吃饭", "认不准的不动，留给引用检查报"


def test_分镜比规格长_建议用length放宽():
    from aigc_agent.domain.drama.format import EpisodeFormat
    from aigc_agent.domain.drama.models import Episode
    from aigc_agent.domain.functions.drama import _length_hint

    long_ep = Episode(index=2, title="第2集", desc=_long_storyboard(scenes=4, per=50, secs=2.5))
    hint = _length_hint([long_ep], EpisodeFormat())
    assert "/length 9" in hint and "第2集 约 8.3 分钟" in hint and "不要删台词" in hint
    ok_ep = Episode(index=1, title="第1集", desc=_long_storyboard(scenes=2, per=48, secs=2.5))
    assert _length_hint([ok_ep], EpisodeFormat()) == ""


async def test_渲染前核对覆盖_漏镜头的提示词不花钱():
    from tests.test_episode_format import LIB

    store = AssetStore()
    sb = store.create(
        json.dumps([{"episodeIndex": 1, "episodeTitle": "第1集",
                     "episodeDesc": _long_storyboard(scenes=1, per=20)}], ensure_ascii=False),
        summary="分镜", creator="tool:drama_storyboard",
    )
    lib = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库",
                       creator="tool:drama_assets")
    half = [{"scene_index": "[第1集-1场]", "video_name": f"{a}-{a + 4}", "video_duration": "10s",
             "cuts": [2] * 5, "description": "(地铁车厢) (陆离-风衣-[全集]) 镜头"}
            for a in (1, 6)]
    shots = store.create(json.dumps(half, ensure_ascii=False), summary="提示词",
                         creator="tool:drama_shots", parents=[sb.id, lib.id])
    called: list[Any] = []

    async def invoke(name: str, args: dict[str, Any]) -> Any:
        called.append(name)
        raise AssertionError("不该发起任何生成")

    fns = DramaFunctions(None, store, registry=SimpleNamespace(invoke=invoke))
    r = await fns._fn_drama_render_shots(shots.id)
    assert not r.ok and "漏了分镜第 11–20 镜" in r.error and "drama_shots" in r.error
    assert r.meta.get("charged") is False and not called


async def test_渲染前核对_镜头都在但被压缩的也拦():
    from tests.test_episode_format import LIB

    store = AssetStore()
    sb = store.create(
        json.dumps([{"episodeIndex": 1, "episodeTitle": "第1集",
                     "episodeDesc": _long_storyboard(scenes=1, per=20)}], ensure_ascii=False),
        summary="分镜", creator="tool:drama_storyboard",
    )
    lib = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库",
                       creator="tool:drama_assets")
    squeezed = [{"scene_index": "[第1集-1场]", "video_name": f"{a}-{a + 9}",
                 "video_duration": "12s", "cuts": [3, 3, 3, 3],
                 "description": "(地铁车厢) (陆离-风衣-[全集]) 镜头"}
                for a in (1, 11)]
    shots = store.create(json.dumps(squeezed, ensure_ascii=False), summary="提示词",
                         creator="tool:drama_shots", parents=[sb.id, lib.id])
    fns = DramaFunctions(None, store, registry=None)
    r = await fns._fn_drama_render_shots(shots.id)
    assert not r.ok and "压缩" in r.error and "分镜 40s → 提示词 24s" in r.error


async def test_分镜比规格短_提示词也跟着分镜_不拉长加戏():
    from dataclasses import replace

    from aigc_agent.domain.drama.format import EpisodeFormat
    from tests.test_episode_format import LIB

    store = AssetStore()
    # 按 4 分钟拆的分镜（120 镜 × 2s = 240s），项目后来改成一集 8 分钟
    sb = store.create(
        json.dumps([{"episodeIndex": 1, "episodeTitle": "第1集",
                     "episodeDesc": _long_storyboard(scenes=2, per=60)}], ensure_ascii=False),
        summary="分镜", creator="tool:drama_storyboard",
    )
    lib = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库",
                       creator="tool:drama_assets")
    gw = _ChunkGateway()
    fns = DramaFunctions(gw, store, registry=None, catalog=None,
                         fmt=replace(EpisodeFormat(), minutes=8))
    r = await fns._fn_drama_shots(sb.id, lib.id)
    assert r.ok, r.error
    saved = json.loads(store.content(r.asset_ref))
    assert sum(int(s["video_duration"].rstrip("s")) for s in saved) == 240, "不按 8 分钟拉成两倍"
    assert "分镜比规格短" in r.content and "重拆" in r.content
    assert len(gw.calls) == 2 and "✓ 规格" in r.content


# ================================================================ 09-25：视频画幅按项目选（/ratio）


def test_画幅认得各种写法():
    from aigc_agent.domain.aspect import aspect_label, orientation_of, parse_aspect, still_size

    for t in ("16:9", "16：9", "16x9", "16 : 9", "横屏", "landscape"):
        assert parse_aspect(t) == "16:9", t
    assert parse_aspect("竖屏") == "9:16" and parse_aspect("1:1") == "1:1"
    assert parse_aspect("方形") == "1:1"
    assert parse_aspect("4:3") == "" and parse_aspect("") == "" and parse_aspect("超宽") == ""
    assert orientation_of("16:9") == "landscape" and orientation_of("9:16") == "portrait"
    assert still_size("16:9") == (1280, 720) and still_size("9:16") == (720, 1280)
    assert aspect_label("16:9") == "横屏 16:9"


def test_会话快照记住画幅(tmp_path):
    from aigc_agent.capabilities.memory.session import SessionSnapshot
    from aigc_agent.harness.context.window import ShortTermMemory

    s = SessionSnapshot(tmp_path, "p")
    s.set_aspect_ratio("16:9")
    assert SessionSnapshot(tmp_path, "p").aspect_ratio == "16:9"
    s.save(ShortTermMemory())
    assert SessionSnapshot(tmp_path, "p").aspect_ratio == "16:9", "每轮结束的整体重写不能把它丢了"


async def test_项目画幅套到各环节_重启沿用(tmp_path):
    from aigc_agent.app import ASPECT_PIN, Agent

    ws = tmp_path / "ws"
    a = Agent.create(session_id="s-ratio", workspace=ws)
    try:
        assert a.aspect_ratio == "9:16" and not a.aspect_custom
        assert ASPECT_PIN not in a.memory.pins
        a.set_aspect_ratio("横屏")
        assert a.drama_fns.aspect_ratio == "16:9" and a.short_video_fns.aspect_ratio == "16:9"
        assert a.media_fns.default_aspect == "16:9"
        assert a.material_fns.default_orientation == "landscape"
        assert "横屏 16:9" in a.episode_spec_line()
        assert "横屏 16:9" in a.memory.pins[ASPECT_PIN].content
    finally:
        await a.aclose()

    b = Agent.create(session_id="s-ratio", workspace=ws)
    try:
        assert b.aspect_ratio == "16:9" and b.drama_fns.aspect_ratio == "16:9", "重启沿用"
        b.set_aspect_ratio(None)
        assert b.drama_fns.aspect_ratio == "9:16" and b.short_video_fns.aspect_ratio == ""
        assert b.media_fns.default_aspect == "" and ASPECT_PIN not in b.memory.pins
    finally:
        await b.aclose()


async def test_短剧按项目画幅渲_换画幅不复用旧片段():
    store = AssetStore()
    fns, reg, _ = _gate_fns(store, {})
    shots_id, pack_id = _seed(store)
    composed: list[dict[str, Any]] = []
    orig = reg.invoke

    async def spy(name: str, args: dict[str, Any]) -> Any:
        if name == "compose_video":
            composed.append(dict(args))
        return await orig(name, args)

    reg.invoke = spy  # type: ignore[method-assign]
    fns.aspect_ratio = "16:9"
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.ok, r.error
    assert {c["aspect_ratio"] for c in reg.calls} == {"16:9"}
    assert all(c["tags"]["aspect"] == "16:9" for c in reg.calls)
    assert composed and composed[-1]["aspect_ratio"] == "16:9", "成片画布跟着画幅"

    reg.calls.clear()
    r2 = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r2.ok and not reg.calls, "同画幅：已付费的段全部复用"
    r3 = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, aspect_ratio="9:16")
    assert r3.ok and len(reg.calls) == 2, "换了画幅：旧片段不拼进新比例的成片"
    assert {c["aspect_ratio"] for c in reg.calls} == {"9:16"}


async def test_锁定的视频模型不支持的画幅_花钱前拦下():
    store = AssetStore()
    fns, reg, _ = _gate_fns(store, {})
    shots_id, pack_id = _seed(store)
    fns.catalog.get = lambda kind, model: SimpleNamespace(aspect_ratios=["16:9", "9:16"])
    fns.video_lock_source = lambda: "seedance-2.0"
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, aspect_ratio="1:1")
    assert not r.ok and "不支持画幅 1:1" in r.error and r.meta["charged"] is False
    assert not reg.calls
    r2 = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, aspect_ratio="4:3")
    assert not r2.ok and "认不出画幅" in r2.error and not reg.calls


async def test_gen_video没传比例用项目画幅_模型不支持的拦下(tmp_path):
    from tests.test_media_leftovers import _media

    fns, provider = _media(tmp_path)
    fns.default_aspect = "16:9"
    args = {"prompt": "海边日落", "model": "seedance-2.0", "duration": 5}
    r = await fns.invoke("gen_video", args)
    assert r.ok, r.error
    assert provider.submitted[-1].get("aspect_ratio") == "16:9"
    n = len(provider.submitted)
    r2 = await fns.invoke("gen_video", {"prompt": "海边日落", "aspect_ratio": "1:1", "duration": 5})
    assert not r2.ok and "不支持画幅 1:1" in r2.error and r2.meta.get("charged") is False
    assert len(provider.submitted) == n, "不支持的画幅不提交"


async def test_短视频画幅_这次指定的优先_其次项目_最后配方():
    from tests.test_short_video import Gateway, Registry, _catalog, _plan

    store = AssetStore()
    reg = Registry(store)
    fns = ShortVideoFunctions(Gateway(_plan()), store, registry=reg, catalog=_catalog(),
                              recipes_dir=RECIPES)
    b = await fns.invoke("short_video_brief", {"keyword": "AI 芯片", "style": "tech-short"})
    assert b.ok, b.error
    base = {"brief_id": b.asset_ref, "confirm": True, "no_voiceover": True, "no_subtitle": True}

    # 每次都要重新生成：每次都得人在确认单上点头（2026-09-26：confirm=true 只认人的确认）
    fns.approve(b.asset_ref)
    r = await fns.invoke("short_video_produce", base)
    assert r.ok, r.error
    assert {a["aspect_ratio"] for a in reg.of("gen_video")} == {"9:16"}, "没设：按配方（竖屏）"

    reg.calls.clear()
    fns.aspect_ratio = "16:9"
    fns.approve(b.asset_ref)
    r = await fns.invoke("short_video_produce", base)
    assert r.ok, r.error
    assert reg.of("gen_video"), "换了画幅：上次的竖屏片段不复用"
    assert {a["aspect_ratio"] for a in reg.of("gen_video")} == {"16:9"}
    assert reg.of("compose_video")[-1]["aspect_ratio"] == "16:9"

    reg.calls.clear()
    fns.approve(b.asset_ref)
    r = await fns.invoke("short_video_produce", {**base, "aspect_ratio": "1:1"})
    assert r.ok, r.error
    assert {a["aspect_ratio"] for a in reg.of("gen_video")} == {"1:1"}, "这次指定的优先"


async def test_素材站搜索方向跟着项目画幅(tmp_path):
    from aigc_agent.domain.functions.materials import MaterialFunctions

    fns = MaterialFunctions(AssetStore(), tmp_path)
    seen: list[str] = []

    async def fake_search(query: str, kind: str, n: int, orientation: str) -> Any:
        seen.append(orientation)
        return [], ""

    fns._key = lambda s: "k"  # type: ignore[method-assign]
    fns._search_pexels = fake_search  # type: ignore[method-assign]
    fns._search_pixabay = fake_search  # type: ignore[method-assign]
    await fns._fn_stock_media_search("海边")
    fns.default_orientation = "landscape"
    await fns._fn_stock_media_search("海边")
    await fns._fn_stock_media_search("海边", orientation="square")
    assert seen == ["portrait", "portrait", "landscape", "landscape", "square", "square"]


def test_ratio命令在命令表里_说明写清支持哪些():
    from aigc_agent.domain.generators.catalog import MediaCatalog
    from aigc_agent.domain.system_prompt import SYSTEM_PROMPT
    from aigc_agent.interfaces.cli.main import _ratio_brief
    from aigc_agent.interfaces.cli.promptbox import known_command
    from tests.test_media_leftovers import CATALOG_PATH

    assert known_command("/ratio 16:9") and known_command("/画幅 横屏") and known_command("/比例")
    agent = SimpleNamespace(
        aspect_ratio="16:9", aspect_custom=True,
        media_fns=SimpleNamespace(video_lock="seedance-2.0"),
        catalog=MediaCatalog.load(CATALOG_PATH),
    )
    text = _ratio_brief(agent)  # type: ignore[arg-type]
    assert "横屏 16:9（本项目设置）" in text and "seedance-2.0 支持 16:9 / 9:16" in text
    assert "/ratio" in SYSTEM_PROMPT


# ================================================================ 09-25：过滤自动重试、项目卡


class _FilterOnce:
    """第一次回内容过滤，之后回 good。"""

    def __init__(self, good: str, times: int = 1) -> None:
        self.good = good
        self.times = times
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(self, role: str, messages: list[dict[str, Any]], **_: Any) -> Any:
        self.calls.append(messages)
        if len(self.calls) <= self.times:
            return SimpleNamespace(text="", finish_reason="content_filter")
        return SimpleNamespace(text=self.good, finish_reason="stop")


async def test_分镜被内容过滤拦一次_自动用克制措辞重试():
    from aigc_agent.domain.drama.format import EpisodeFormat
    from tests.test_episode_format import _desc
    from tests.test_wardrobe import SCRIPT

    good = json.dumps([{"episodeIndex": 1, "episodeTitle": "第1集", "episodeDesc": _desc(96)}],
                      ensure_ascii=False)
    gw = _FilterOnce(good)
    fns = DramaFunctions(gw, AssetStore(), registry=None, catalog=None, fmt=EpisodeFormat())
    r = await fns._fn_drama_storyboard(SCRIPT, ethnicity="asian", language="zh")
    assert r.ok, r.error
    assert len(gw.calls) == 2 and "【措辞】" in gw.calls[1][-1]["content"]
    assert "内容安全过滤拦了一次" in r.content

    gw2 = _FilterOnce(good, times=2)
    fns2 = DramaFunctions(gw2, AssetStore(), registry=None, catalog=None, fmt=EpisodeFormat())
    r2 = await fns2._fn_drama_storyboard(SCRIPT, ethnicity="asian", language="zh")
    assert not r2.ok and "已经自动用克制的措辞重试过一次" in r2.error


async def test_提示词批次被内容过滤拦一次_自动重试():
    class _Filtered(_ChunkGateway):
        async def chat(self, role: str, messages: list[dict[str, Any]], **kw: Any) -> Any:
            if not self.calls:
                self.calls.append(messages)
                return SimpleNamespace(text="", finish_reason="content_filter")
            return await super().chat(role, messages, **kw)

    gw = _Filtered()
    fns, store, sb, lib = _shots_setup(gw)
    r = await fns._fn_drama_shots(sb, lib)
    assert r.ok, r.error
    assert "内容安全过滤拦了 1 次" in r.content
    assert any("【措辞】" in m[-1]["content"] for m in gw.calls)


def test_项目卡标出漏镜头的提示词_最新一份完整就不标(tmp_path):
    from aigc_agent.domain.drama.card import build_project_card
    from tests.test_episode_format import LIB

    (tmp_path / ".drama-state.json").write_text(
        json.dumps({"dramaTitle": "不渡", "totalEpisodes": 20}, ensure_ascii=False),
        encoding="utf-8",
    )
    store = AssetStore()
    sb = store.create(
        json.dumps([{"episodeIndex": 1, "episodeTitle": "第1集",
                     "episodeDesc": _long_storyboard(scenes=1, per=20)}], ensure_ascii=False),
        summary="分镜", creator="tool:drama_storyboard",
    )
    lib = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库",
                       creator="tool:drama_assets")

    def shots(spans: list[int]) -> str:
        rows = [{"scene_index": "[第1集-1场]", "video_name": f"{a}-{a + 4}",
                 "video_duration": "10s", "cuts": [2] * 5, "description": "(地铁车厢) 镜头"}
                for a in spans]
        return json.dumps(rows, ensure_ascii=False)

    store.create(shots([1, 6]), summary="提示词", creator="tool:drama_shots",
                 parents=[sb.id, lib.id])
    card = build_project_card(store, tmp_path)
    assert "⚠ 第 1 集的视频提示词漏了分镜镜头或被压缩" in card, card

    store.create(shots([1, 6, 11, 16]), summary="提示词·重做", creator="tool:drama_shots",
                 parents=[sb.id, lib.id])
    assert "视频提示词漏了" not in build_project_card(store, tmp_path), "最新一份完整就不标"


# ================================================================ 09-25：参考图包只用这套资产库的


def test_参考图包血缘_一路追到资产库_别的剧的包不挑():
    from aigc_agent.domain.drama.refpack import pack_library, pick_pack

    store = AssetStore()
    lib_a = store.create("{}", summary="资产库A", creator="tool:drama_assets")
    lib_b = store.create("{}", summary="资产库B", creator="tool:drama_assets")
    body = '{"唐僧": {"asset": "as_x", "url": "https://x/1.png"}}'
    p1 = store.create(body, summary="包A", creator="tool:drama_render_assets", parents=[lib_a.id])
    p2 = store.create(body, summary="包A·托管", creator="tool:drama_render_assets",
                      parents=[p1.id, lib_a.id])
    p3 = store.create(body, summary="包A·再托管", creator="tool:drama_render_assets",
                      parents=[p2.id, p1.id])
    assert pack_library(store, p3.id) == lib_a.id, "资产库不在直接父级也要追到"
    legacy = store.create(body, summary="没血缘的老包", creator="tool:drama_render_assets")
    packs = sorted([p1, p2, p3, legacy], key=lambda a: -a.seq)
    assert pick_pack(store, packs, lib_a.id) == p3.id
    assert pick_pack(store, packs, lib_b.id) == legacy.id, "别的资产库的包不要，只退到血缘不明的"
    assert pick_pack(store, [p1], lib_b.id) == ""


async def test_渲染时传了别的剧的参考图包_花钱前拦下():
    store = AssetStore()
    fns, reg, _ = _gate_fns(store, {})
    shots_id, pack_id = _seed(store)
    other_lib = store.create(
        json.dumps({"characters": [{"name": "安妮", "prompt": "x"}]}, ensure_ascii=False),
        summary="别的剧的资产库", creator="tool:drama_assets",
    )
    other_pack = store.create(store.content(pack_id), summary="别的剧的参考图",
                              creator="tool:drama_render_assets", parents=[other_lib.id])
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=other_pack.id)
    assert not r.ok and "属于另一套资产库" in r.error and r.meta["charged"] is False
    assert not reg.calls, "同名角色（安妮）能对上也不能用别的剧的脸"
    r2 = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r2.ok, r2.error


async def test_按集流水只用这套资产库的参考图包(tmp_path):
    store, bus = AssetStore(), EventBus()
    pipe = _build(store, bus, FakeRegistry(store), tmp_path, total=1, enabled=False)
    try:
        old_lib = store.create("{}", summary="旧剧资产库", creator="tool:drama_assets")
        store.create("{}", summary="旧剧参考图", creator="tool:drama_render_assets",
                     parents=[old_lib.id])
        lib = store.create("{}", summary="资产库", creator="tool:drama_assets")
        assert pipe._scan()["refs"] == "", "旧剧的参考图包不能拿来渲新剧"
        pack = store.create("{}", summary="参考图", creator="tool:drama_render_assets",
                            parents=[lib.id])
        assert pipe._scan()["refs"] == pack.id
    finally:
        await pipe.aclose()


# ======================================================== 09-25：分镜的台词念不念得完、有没有丢


def _talky(n: int, secs: float, line: str = "这句台词一共有十二个字啊") -> str:
    """n 个镜头、每镜 secs 秒、每镜一句台词（默认 12 个字，按每秒 4.5 字要念 2.7 秒）。"""
    from aigc_agent.domain.drama.format import HOOK_MARK

    rows = ["[夜] [内] [地铁车厢]"]
    for i in range(1, n + 1):
        mark = HOOK_MARK if i == 1 else ""
        rows.append(f"{mark}[近景/固定/平视/{secs:g}s] 陆离看着她，第{i}句：“{line}”")
    return "\n".join(rows)


_SAID = [f"这是第{c}句要说的台词内容" for c in "一二三四五六七八九十"]
_TALK_SCRIPT = (
    "# 第1集《花》\n\n### 1-1 夜 内 地铁车厢\n\n△ 陆离擦着灯罩，门外有人敲了三下。\n\n"
    + "\n\n".join(f"陆离：（低声）{s}" for s in _SAID)
    + "\n\n苏晏：嗯。\n\n△ 灯灭了。\n\n> 🎣 本集钩子：她是谁\n"
    + "△ 车厢里只剩下风声，陆离把灯罩放回原处，没有回头。\n" * 8
)


def test_分镜把台词压快了_总时长达标也要拦():
    from aigc_agent.domain.drama.format import (
        EpisodeFormat,
        check_storyboard,
        dialogue_rushed,
        speech_seconds,
    )
    from aigc_agent.domain.drama.models import Episode
    from aigc_agent.domain.functions.drama import _length_hint

    assert speech_seconds("（笑）一二三四五六七八九") == 2.0, "括号里的表演提示不念"
    assert speech_seconds("Hello there, my friend") == 4 / 3
    fmt = EpisodeFormat()  # 一集 4 分钟：204–276s
    # 旧规格下「修」出来的样子：160 镜 × 1.5s = 240s 正好达标，光念台词就要 427s
    rushed = Episode(index=1, title="第1集", desc=_talky(160, 1.5))
    probs = check_storyboard(rushed, fmt)
    assert dialogue_rushed(rushed)
    assert any("台词被压快了" in p and "不许删台词" in p and "/length" in p for p in probs), probs
    assert not any("应在 204–276s" in p for p in probs), "不能再要求往 4 分钟里压"
    hint = _length_hint([rushed], fmt)
    assert "光念台词就要约 7.1 分钟" in hint and "/length 10" in hint, hint

    # 照实拆的：480s 比规格长，但光台词就超出规格了 —— 不算分镜的毛病，交给用户 /length
    honest = Episode(index=1, title="第1集", desc=_talky(160, 3))
    assert not dialogue_rushed(honest)
    assert not any("加起来" in p or "台词" in p for p in check_storyboard(honest, fmt))
    assert "约 8.0 分钟" in _length_hint([honest], fmt)
    # 注水到台词的 2.5 倍以上还是要压，而且提醒别压说台词的镜头
    padded = Episode(index=1, title="第1集", desc=_talky(160, 3) + "\n" + "\n".join(
        f"[远景/固定/平视/3s] 空镜{i}" for i in range(260)))
    assert any("说台词的镜头别压" in p for p in check_storyboard(padded, fmt))


def test_台词少的分镜不受影响():
    from aigc_agent.domain.drama.format import EpisodeFormat, check_storyboard, dialogue_rushed
    from aigc_agent.domain.drama.models import Episode
    from tests.test_episode_format import _desc

    ok = Episode(index=1, title="第1集", desc=_desc(96))
    assert not dialogue_rushed(ok) and check_storyboard(ok, EpisodeFormat()) == []


def test_分镜比剧本少了台词_要补回来_拆开的长台词认得出_译成英文的不查():
    from aigc_agent.domain.drama.format import (
        EpisodeFormat,
        missing_dialogue,
        script_dialogue,
        split_script,
        storyboard_problems,
    )
    from aigc_agent.domain.drama.models import Episode

    assert script_dialogue(_TALK_SCRIPT) == [f"（低声）{s}" for s in _SAID], "「嗯」、钩子行不算"
    assert set(split_script("# 第一集\n甲：你好你好啊\n# 第二集\n乙：再见再见啦")) == {1, 2}

    def board(said: list[str]) -> str:
        rows = ["[夜] [内] [地铁车厢]"]
        rows += [f"[近景/固定/平视/3s] 陆离低声说：“{s}”" for s in said]
        return "\n".join(rows)

    fmt = EpisodeFormat()
    dropped = Episode(index=1, title="第1集", desc=board(_SAID[:6]))
    total, miss = missing_dialogue(_TALK_SCRIPT, dropped.desc)
    assert total == 10 and len(miss) == 4
    probs = storyboard_problems([dropped], _TALK_SCRIPT, fmt)
    assert any("分镜里找不到 4 句" in p and "逐字保留" in p for p in probs), probs

    # 第一句拆到两个镜头里（断在前 6 个字中间）也认得出
    split = board(["这是第一", "句要说的台词内容", *_SAID[1:]])
    assert missing_dialogue(_TALK_SCRIPT, split)[1] == []
    # 分镜按 language=en 译成英文：不同文字不查
    english = board([f"This is line {i} that must be said" for i in range(10)])
    assert missing_dialogue(_TALK_SCRIPT, english) == (0, [])
    # 少一两句（可能只是改了措辞）不算
    two = Episode(index=1, title="第1集", desc=board(_SAID[:8]))
    assert not any("找不到" in p for p in storyboard_problems([two], _TALK_SCRIPT, fmt))


class _Seq:
    """按顺序回几份分镜（最后一份一直重复）。"""

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(self, role: str, messages: list[dict[str, Any]], **_: Any) -> Any:
        self.calls.append(messages)
        text = self.texts[min(len(self.calls), len(self.texts)) - 1]
        return SimpleNamespace(text=text, finish_reason="stop")


def _board_json(desc: str, index: int = 1) -> str:
    return json.dumps([{"episodeIndex": index, "episodeTitle": f"第{index}集",
                        "episodeDesc": desc}], ensure_ascii=False)


async def test_分镜工具_压快台词的版本不被采用_改对了照实报集长():
    from aigc_agent.domain.drama.format import EpisodeFormat
    from tests.test_wardrobe import SCRIPT

    store = AssetStore()
    gw = _Seq(_board_json(_talky(160, 1.5)), _board_json(_talky(160, 3)))
    fns = DramaFunctions(gw, store, registry=None, catalog=None, fmt=EpisodeFormat())
    r = await fns._fn_drama_storyboard(SCRIPT, ethnicity="asian", language="zh")
    assert r.ok, r.error
    assert len(gw.calls) == 2 and "台词被压快了" in gw.calls[1][-1]["content"]
    assert "[近景/固定/平视/3s]" in store.content(r.asset_ref), "改对的那版要被采用"
    assert "📏" in r.content and "/length" in r.content
    assert "规格检查未过" not in r.content and "—— 达标" not in r.content

    # 模型坚持压快：两版都压快，保留下来但照实报「规格检查未过」
    gw2 = _Seq(_board_json(_talky(160, 1.5)))
    fns2 = DramaFunctions(gw2, AssetStore(), registry=None, catalog=None, fmt=EpisodeFormat())
    r2 = await fns2._fn_drama_storyboard(SCRIPT, ethnicity="asian", language="zh")
    assert r2.ok and "规格检查未过" in r2.content and "台词被压快了" in r2.content


async def test_分镜工具_删了台词让模型补回来():
    from aigc_agent.domain.drama.format import EpisodeFormat

    def board(said: list[str]) -> str:
        from aigc_agent.domain.drama.format import HOOK_MARK

        rows = ["[夜] [内] [地铁车厢]"]
        rows += [f"{HOOK_MARK if i == 0 else ''}[近景/固定/平视/2.5s] 陆离：“{s}”"
                 for i, s in enumerate(said)]
        rows += [f"[远景/固定/平视/2.5s] 空镜{i}" for i in range(86)]
        return "\n".join(rows)

    gw = _Seq(_board_json(board(_SAID[:5])), _board_json(board(_SAID)))
    fns = DramaFunctions(gw, AssetStore(), registry=None, catalog=None, fmt=EpisodeFormat())
    r = await fns._fn_drama_storyboard(_TALK_SCRIPT, ethnicity="asian", language="zh")
    assert r.ok, r.error
    assert len(gw.calls) == 2 and "分镜里找不到 5 句" in gw.calls[1][-1]["content"]
    assert "规格检查未过" not in r.content, r.content


async def test_提示词工具_分镜压快了要提醒不能算完成():
    from aigc_agent.domain.drama.format import EpisodeFormat
    from tests.test_episode_format import LIB

    store = AssetStore()
    sb = store.create(_board_json(_talky(120, 2)), summary="分镜", creator="tool:drama_storyboard")
    lib = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库",
                       creator="tool:drama_assets")
    fns = DramaFunctions(_ChunkGateway(), store, registry=None, catalog=None,
                         fmt=EpisodeFormat())
    r = await fns._fn_drama_shots(sb.id, lib.id)
    assert r.ok, r.error
    assert "第1集 的分镜把台词压快了" in r.content and "/length" in r.content


def test_项目卡标出压快台词和少了台词的分镜(tmp_path):
    from aigc_agent.domain.drama.card import build_project_card

    (tmp_path / ".drama-state.json").write_text(
        json.dumps({"dramaTitle": "不渡", "totalEpisodes": 2}, ensure_ascii=False),
        encoding="utf-8",
    )
    store = AssetStore()
    store.create("# 第1集《花》\n" + "△ 陆离擦着灯罩，门外有人敲了三下。他没抬头。\n" * 16,
                 type_=AssetType.SCRIPT, summary="第1集 花",
                 creator="tool:drama_write", gen_params={"episode": 1})
    store.create(_TALK_SCRIPT.replace("第1集", "第2集"), type_=AssetType.SCRIPT,
                 summary="第2集 花", creator="tool:drama_write", gen_params={"episode": 2})
    ep2 = "\n".join(["[夜] [内] [地铁车厢]"] + [
        f"[近景/固定/平视/3s] 陆离：“{s}”" for s in _SAID[:5]] + [
        f"[远景/固定/平视/3s] 空镜{i}" for i in range(5)])
    old = json.dumps([
        {"episodeIndex": 1, "episodeTitle": "第1集", "episodeDesc": _talky(160, 1.5)},
        {"episodeIndex": 2, "episodeTitle": "第2集", "episodeDesc": ep2},
        {"episodeIndex": 9, "episodeTitle": "第9集", "episodeDesc": _talky(160, 1.5)},
    ], ensure_ascii=False)
    store.create(old, summary="分镜", creator="tool:drama_storyboard")
    card = build_project_card(store, tmp_path)
    assert "⚠ 第 1 集的分镜把台词压快了" in card, card
    assert "第9集" not in card and "第 1, 9 集" not in card, "不是这部剧的集号（没迁移的旧剧）不标"
    assert "⚠ 分镜比剧本少了台词（第2集少 5 句）" in card, card

    store.create(_board_json(_talky(160, 3)), summary="分镜·重拆", creator="tool:drama_storyboard")
    card2 = build_project_card(store, tmp_path)
    assert "把台词压快了" not in card2, "重拆过的集按最新的分镜算"
    assert "第2集少 5 句" in card2


def test_系统提示词_项目卡打了警告的集不能报完成():
    from aigc_agent.domain.system_prompt import SYSTEM_PROMPT

    assert "打了 ⚠ 的集没有完成" in SYSTEM_PROMPT and "不要打 ✅" in SYSTEM_PROMPT


# ======================================================== 09-26：资产库 JSON 坏了、一部剧一份资产库


def test_JSON_漏逗号尾逗号本地修好_截断的报出错位置():
    from aigc_agent.domain.drama.parse import _load

    assert _load('{"a": [{"x": 1}\n  {"x": 2}]}')[0] == {"a": [{"x": 1}, {"x": 2}]}
    assert _load('{"a": "x"\n  "b": "y"}')[0] == {"a": "x", "b": "y"}, (
        "行尾漏逗号：之前被当成内嵌引号转义掉，报「字符串没收尾」"
    )
    assert _load('{"a": [1, 2,], "b": {"c": 1,},}')[0] == {"a": [1, 2], "b": {"c": 1}}
    assert _load('{"a": "他说"走"然后"\n "b": "y"}')[0] == {"a": '他说"走"然后', "b": "y"}
    data, err = _load('{"a": [{"x": 1}, {"x": 2')
    assert data is None and "出错处附近" in err and "▲" in err, "截断的不瞎修，报出错位置"


def _scripts(store: AssetStore, eps: tuple[int, ...]) -> list[str]:
    from tests.test_wardrobe import SCRIPT

    return [
        store.create(SCRIPT, type_=AssetType.SCRIPT, summary=f"第{n}集", creator="tool:drama_write",
                     gen_params={"episode": n}).id
        for n in eps
    ]


async def test_资产库输出坏了_自动重发一次_记下用了哪几集的剧本():
    from aigc_agent.domain.drama.format import EpisodeFormat
    from tests.test_episode_format import LIB

    store = AssetStore()
    ids = _scripts(store, (1, 2, 3))
    good = json.dumps(LIB, ensure_ascii=False)
    gw = _Seq(good[: len(good) // 2], good)  # 第一次输出坏了（本地修不好）
    fns = DramaFunctions(gw, store, registry=None, catalog=None, fmt=EpisodeFormat())
    r = await fns._fn_drama_assets(script_ids=ids[:2], ethnicity="asian", language="zh")
    assert r.ok, r.error
    assert len(gw.calls) == 2 and "【格式】" in gw.calls[1][-1]["content"]
    a = store.get(r.asset_ref)
    assert a.gen_params["covers"] == [1, 2] and a.parent_ids == ids[:2]
    assert "只用了第 1–2 集的剧本，项目里还有第 3 集" in r.content
    assert "一部剧只要一份资产库" in r.content

    gw2 = _Seq('{"characters": [')
    fns2 = DramaFunctions(gw2, AssetStore(), registry=None, catalog=None, fmt=EpisodeFormat())
    r2 = await fns2._fn_drama_assets(script="甲：你好。\n" * 60, ethnicity="asian", language="zh")
    assert not r2.ok and len(gw2.calls) == 2
    assert "已自动重发一次" in r2.error and "不要拆成几段" in r2.error and "手工合并" in r2.error


async def test_分镜输出坏了_自动重发一次():
    from aigc_agent.domain.drama.format import EpisodeFormat
    from tests.test_episode_format import _desc
    from tests.test_wardrobe import SCRIPT

    gw = _Seq("这一次模型没输出 JSON", _board_json(_desc(96)))
    fns = DramaFunctions(gw, AssetStore(), registry=None, catalog=None, fmt=EpisodeFormat())
    r = await fns._fn_drama_storyboard(SCRIPT, ethnicity="asian", language="zh")
    assert r.ok, r.error
    assert len(gw.calls) == 2 and "【格式】" in gw.calls[1][-1]["content"]


async def test_提示词批次输出坏了_自动重发一次():
    class _BadOnce(_ChunkGateway):
        async def chat(self, role: str, messages: list[dict[str, Any]], **kw: Any) -> Any:
            if not self.calls:
                self.calls.append(messages)
                return SimpleNamespace(text='[{"scene_index": ', finish_reason="stop")
            return await super().chat(role, messages, **kw)

    gw = _BadOnce()
    fns, _, sb, lib = _shots_setup(gw)
    r = await fns._fn_drama_shots(sb, lib)
    assert r.ok, r.error
    assert any("【格式】" in m[-1]["content"] for m in gw.calls)


async def test_资产库只用了别的集的剧本_不出提示词():
    from aigc_agent.domain.drama.format import EpisodeFormat
    from tests.test_episode_format import LIB

    store = AssetStore()
    sb = store.create(_board_json(_long_storyboard(scenes=1, per=20)), summary="分镜",
                      creator="tool:drama_storyboard")
    lib = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库",
                       creator="tool:drama_assets", gen_params={"covers": list(range(11, 21))})
    gw = _ChunkGateway()
    fns = DramaFunctions(gw, store, registry=None, catalog=None, fmt=EpisodeFormat())
    r = await fns._fn_drama_shots(sb.id, lib.id)
    assert not r.ok and r.meta["charged"] is False
    assert "只用第 11–20 集的剧本生成" in r.error and "一部剧只要一份资产库" in r.error
    assert not gw.calls


def test_项目卡_资产库只覆盖半部剧要标出来(tmp_path):
    from aigc_agent.domain.drama.card import build_project_card
    from tests.test_episode_format import LIB

    store = AssetStore()
    _scripts(store, (1, 2, 3, 4))
    lib = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库",
                       creator="tool:drama_assets", gen_params={"covers": [3, 4]})
    card = build_project_card(store, tmp_path)
    assert f"资产库 {lib.id}（第 3–4 集）" in card
    assert f"⚠ 最新的资产库 {lib.id} 只用了第 3–4 集，第 1–2 集不在里面" in card, card

    # 老的库没记用了哪几集：按服装标的集数推
    old = {"characters": [{"baseRoleName": "陆离", "roleTotalDesc": "男 | 35岁",
                           "roleCostumeList": [{"costumeName": "陆离-风衣-[1-2]",
                                                "costumeDesc": "深色风衣", "episodes": "1-2"}]}]}
    lib2 = store.create(json.dumps(old, ensure_ascii=False), summary="资产库·旧",
                        creator="tool:drama_assets")
    card2 = build_project_card(store, tmp_path)
    assert f"⚠ 最新的资产库 {lib2.id} 的服装只排到第 1–2 集，第 3–4 集不在里面" in card2, card2


def test_系统提示词_一部剧一份资产库():
    from aigc_agent.domain.system_prompt import SYSTEM_PROMPT

    assert "一部剧只要一份资产库" in SYSTEM_PROMPT and "手工合并" in SYSTEM_PROMPT


# ======================================================== 09-26：分镜重拆 / 资产库重做后的旧提示词


def _two_scene_board() -> str:
    rows = ["[夜] [内] [酒店走廊]"] + [f"[近景/固定/平视/3s] 安妮第{i}步" for i in range(1, 5)]
    rows += ["[日] [内] [酒店大堂]"] + [f"[近景/固定/平视/3s] 安妮第{i}步" for i in range(5, 9)]
    return _board_json("\n".join(rows))


async def test_分镜重拆后_旧提示词花钱前拦下():
    from tests.test_gate_chain import SHOTS

    store = AssetStore()
    fns, reg, _ = _gate_fns(store, {})
    img = store.create("", type_=AssetType.IMAGE, summary="角色·安妮", creator="model:x")
    img.uri = "https://img/anne.png"
    store.put(img)
    lib = store.create(json.dumps({"characters": [{"name": "安妮", "prompt": "x"}]}),
                       summary="资产库", creator="tool:drama_assets")
    pack = store.create(json.dumps({"安妮": {"asset": img.id, "url": img.uri, "kind": "角色"}}),
                        summary="参考图包", creator="tool:drama_render_assets", parents=[lib.id])
    sb = store.create(_two_scene_board(), summary="分镜", creator="tool:drama_storyboard")
    shots = store.create(json.dumps(SHOTS, ensure_ascii=False), summary="提示词",
                         parents=[sb.id, lib.id])
    r = await fns._fn_drama_render_shots(shots.id, rendered_id=pack.id)
    assert r.ok, r.error
    calls = len(reg.calls)

    store.create(_two_scene_board(), summary="分镜·重拆", creator="tool:drama_storyboard")
    r2 = await fns._fn_drama_render_shots(shots.id, rendered_id=pack.id, reuse=False)
    assert not r2.ok and "分镜已经重拆过" in r2.error and r2.meta["charged"] is False
    assert len(reg.calls) == calls, "按旧分镜出的提示词不许花钱"


def _outdated_setup(store: AssetStore, new_board: bool, new_lib: bool) -> str:
    """第 1 集：有提示词、有参考图、还没渲；之后分镜重拆了 / 资产库重做了。"""
    store.create("第1集 剧本", type_=AssetType.SCRIPT, summary="第1集", creator="model",
                 gen_params={"episode": 1})
    sb_json = json.dumps([{"episodeIndex": 1, "episodeTitle": "第1集",
                           "episodeDesc": "[第1集-1场] 镜头"}], ensure_ascii=False)
    sb = store.create(sb_json, type_=AssetType.STORYBOARD, summary="分镜",
                      creator="tool:drama_storyboard")
    lib = store.create("{}", summary="资产库", creator="tool:drama_assets")
    store.create("{}", summary="参考图", creator="tool:drama_render_assets")
    old = store.create("[]", summary="提示词·第1集", creator="tool:drama_shots",
                       gen_params={"episode": 1, "spec": "S2"}, parents=[sb.id, lib.id])
    if new_board:
        store.create(sb_json, type_=AssetType.STORYBOARD, summary="分镜·重拆",
                     creator="tool:drama_storyboard")
    if new_lib:
        store.create("{}", summary="资产库·重做", creator="tool:drama_assets")
    return old.id


async def test_按集流水_分镜重拆或资产库重做后_提示词先重出再渲(tmp_path):
    for new_board, new_lib, why in ((True, False, "分镜"), (False, True, "资产库")):
        store, bus = AssetStore(), EventBus()
        old = _outdated_setup(store, new_board, new_lib)
        fake = FakeRegistry(store)
        fake.spec = "S2"
        pipe = _build(store, bus, fake, tmp_path, total=1, spec="S2")
        try:
            pipe.kick()
            await _settle(pipe)
            assert fake.count("drama_shots") == 1, why
            renders = fake.args_of("drama_render_shots")
            assert len(renders) == 1 and renders[0]["shots_id"] != old, "不许拿旧的去花钱"
            assert any(f"按旧的{why}出的" in n for n in pipe.notes), pipe.notes
        finally:
            await pipe.aclose()


async def test_按集流水_渲过一部分的集_按旧分镜的提示词不重出也不渲(tmp_path):
    store, bus = AssetStore(), EventBus()
    _outdated_setup(store, new_board=True, new_lib=False)
    store.create("[]", summary="片段·第1集", creator="tool:drama_render_shots",
                 gen_params={"episode": 1, "complete": False})
    fake = FakeRegistry(store)
    fake.spec = "S2"
    pipe = _build(store, bus, fake, tmp_path, total=1, spec="S2")
    try:
        pipe.kick()
        await _settle(pipe)
        assert fake.count("drama_shots") == 0 and fake.count("drama_render_shots") == 0
        assert any("已经渲过一部分" in n for n in pipe.notes), pipe.notes
    finally:
        await pipe.aclose()


def test_项目卡_按旧分镜出的提示词要标出来(tmp_path):
    from aigc_agent.domain.drama.card import build_project_card
    from tests.test_episode_format import LIB

    (tmp_path / ".drama-state.json").write_text(
        json.dumps({"dramaTitle": "不渡", "totalEpisodes": 20}, ensure_ascii=False),
        encoding="utf-8",
    )
    store = AssetStore()
    board = _board_json(_long_storyboard(scenes=1, per=20))
    sb = store.create(board, summary="分镜", creator="tool:drama_storyboard")
    lib = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库",
                       creator="tool:drama_assets")
    rows = [{"scene_index": "[第1集-1场]", "video_name": f"{a}-{a + 4}", "video_duration": "10s",
             "cuts": [2] * 5, "description": "(地铁车厢) 镜头"} for a in (1, 6, 11, 16)]
    store.create(json.dumps(rows, ensure_ascii=False), summary="提示词",
                 creator="tool:drama_shots", parents=[sb.id, lib.id])
    assert "按旧的分镜或资产库" not in build_project_card(store, tmp_path)

    store.create(board, summary="分镜·重拆", creator="tool:drama_storyboard")
    card = build_project_card(store, tmp_path)
    assert "⚠ 第 1 集的视频提示词是按旧的分镜或资产库出的" in card, card
    assert "漏了分镜镜头" not in card, "和自己的旧分镜对得上，不算漏镜头"

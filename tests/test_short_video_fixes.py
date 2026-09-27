"""2026-09-23 审查的抖音 / 广告 / 设计链问题。

  · 快切 i % n 轮转：5 个镜头排成 1-2-3-4-5-1（没以英雄帧收尾），8 个镜头时第 7、8 镜付了钱不出现
  · 口播出镜：出镜素材被轮转切碎、原声被丢、没有字幕
  · 广告的产品图没有通道传给生成（身份锁不存在）
  · 出片前不报成本
  · 素材回复「第1镜 生成，第2镜 联网找」只认出第 1 镜
  · 关了口播的配方照样按口播字数拉长时长；配方的写法要点没进简报
  · resolve_materials 改版后，复用认不出上次的镜头
  · 抖音 RPA 不认关键词，全站热榜被当成「相关热点」
  · 别人的原片被判成「自己生成的」，机审查不出版权问题
"""

from __future__ import annotations

import json
from typing import Any

from aigc_agent.domain.assets.store import AssetStore, AssetType, rights_of
from aigc_agent.domain.functions.rpa import _relevant
from aigc_agent.domain.pipeline.cutting import _apportion, plan_aroll_cuts, plan_cuts
from aigc_agent.domain.pipeline.short_video import brief_prompt, parse_brief, parse_material_reply
from tests.test_short_video import Registry, _fns, _plan

# ---------------------------------------------------------------- 排刀


def test_按简报顺序排刀_每镜都出现_以最后一镜收尾():
    cuts = plan_cuts([8.0] * 5, 15, 3.0, 1.0, ordered=True)
    order = [c.clip for c in cuts]
    assert set(order) == {0, 1, 2, 3, 4} and order == sorted(order), order
    assert order[-1] == 4, "英雄帧收尾"
    assert all(c.dur <= 3.0 + 1e-6 for c in cuts)
    assert abs(sum(c.dur for c in cuts) - 15) < 0.05

    eight = plan_cuts([8.0] * 8, 15, 3.0, 1.0, ordered=True)
    assert {c.clip for c in eight} == set(range(8)), "8 个镜头都要出现（刀数跟着加）"

    # 同一镜的几刀在素材里错开取，不是连续帧
    one = [c for c in plan_cuts([9.0, 9.0], 12, 3.0, 1.0, ordered=True) if c.clip == 0]
    assert len({c.start for c in one}) == len(one)


def test_按分量分刀():
    assert _apportion(6, [1, 1, 1]) == [2, 2, 2]
    assert _apportion(5, [3, 1, 1]) == [3, 1, 1]
    assert _apportion(2, [1, 1, 1]) == [1, 1, 0]


def test_口播出镜_出镜人开头收尾_BROLL穿插_口型对得上():
    cuts = plan_aroll_cuts(20.0, [8.0, 8.0], max_seconds=3.0, min_seconds=1.5)
    assert cuts[0].clip == 0 and cuts[-1].clip == 0, "第一刀和最后一刀都是出镜人"
    assert any(c.clip > 0 for c in cuts), "有 B-roll 穿插"
    t = 0.0
    for c in cuts:
        if c.clip == 0:
            assert abs(c.start - t) < 1e-3, "出镜的刀按原时间取，口型对得上"
        t += c.dur
    assert abs(t - 20.0) < 1e-3 and all(c.dur <= 3.0 + 1e-6 for c in cuts)


# ---------------------------------------------------------------- 简报与回复


def test_回复按第N镜切段_逗号不吞():
    got = parse_material_reply("第1镜 生成，第2镜 联网找，第3镜 E:\\素材\\x.mp4，第4镜 生成",
                               [1, 2, 3, 4])
    assert got == {1: ("generate", ""), 2: ("online", ""), 3: ("file", "E:\\素材\\x.mp4"),
                   4: ("generate", "")}


def test_关了口播不按字数拉长_配方要点进简报():
    plan = json.dumps(_plan(duration_seconds=15, script="字" * 200))
    b, warns = parse_brief(plan, "k", "product-ad", 8, 20, voiceover=False)
    assert b is not None and b.duration == 15 and not any("调到" in w for w in warns)
    p = brief_prompt("k", "产品广告", "d", "", 8, 20, prompt_hint="先写英雄产品",
                     script_hint="总字数不要超过 {chars} 字", voiceover=False, shot_seconds=3)
    assert "先写英雄产品" in p and "总字数不要超过 90 字" in p
    assert "不配 TTS 口播" in p and '"seconds": 3' in p and "全站热榜" in p


# ---------------------------------------------------------------- 出片


async def _brief(fns: Any) -> str:
    r = await fns.invoke("short_video_brief", {"keyword": "k", "style": "tech-short"})
    return r.asset_ref


async def test_镜头多于刀数_多的不生成():
    store = AssetStore()
    reg = Registry(store)
    shots = [{"desc": f"镜头{i}", "seconds": 1, "source": "generate"} for i in range(1, 31)]
    fns = _fns(store, reg, _plan(shots=shots, duration_seconds=20))
    bid = await _brief(fns)
    fns.approve(bid)  # 人在确认单上点了头（2026-09-26：confirm=true 只认人的确认）
    r = await fns.invoke("short_video_produce", {"brief_id": bid, "confirm": True})
    assert r.ok, r.error
    assert len(reg.of("gen_video")) < 30 and "进不了成片，没有生成" in r.content


async def test_产品参考图逐镜带上当身份锁():
    store = AssetStore()
    reg = Registry(store)
    fns = _fns(store, reg)
    bid = await _brief(fns)
    fns.approve(bid)
    r = await fns.invoke("short_video_produce", {
        "brief_id": bid, "confirm": True,
        "ref_images": ["https://cdn/product.png"],
        "shot_refs": {"3": ["https://cdn/hero.png"]},
    })
    assert r.ok, r.error
    by = {g["summary"].split("·")[-1]: g for g in reg.of("gen_video")}
    assert by["第1镜"]["image"] == ["https://cdn/product.png"]
    assert by["第3镜"]["image"] == ["https://cdn/hero.png"]
    assert by["第1镜"]["prompt"].startswith("【参考锁定】")

    local = store.create("", type_=AssetType.IMAGE, summary="产品图")
    bad = await fns.invoke("short_video_produce",
                           {"brief_id": bid, "confirm": True, "ref_images": [local.id],
                            "reuse": False})
    assert not bad.ok and "公网链接" in (bad.error or "") and bad.meta.get("charged") is False


async def test_口播出镜_不配TTS_字幕从出镜原声转写(tmp_path):
    store = AssetStore()
    reg = Registry(store)
    fns = _fns(store, reg, _plan())
    fns.recipes_dir = None  # 用仓库里的配方（talking-head：footage=real、voiceover 关）
    made = await fns.invoke("short_video_brief", {"keyword": "k", "style": "talking-head"})
    bid = made.asset_ref
    talk = tmp_path / "talk.mp4"
    talk.write_bytes(b"mp4")
    a = store.create("", type_=AssetType.VIDEO, summary="出镜", creator="human:import",
                     gen_params={"local": str(talk)})
    fns.approve(bid)
    r = await fns.invoke("short_video_produce", {"brief_id": bid, "aroll": a.id, "confirm": True})
    assert r.ok, r.error
    assert not reg.of("tts"), "出镜人自己在说，不配 TTS"
    assert reg.of("transcribe")[0]["asset_id"] == a.id, "字幕从出镜原声转写"
    comp = reg.of("compose_video")[0]
    assert comp["aroll_id"] == a.id and comp["aspect_ratio"] == "9:16"
    assert "字幕：按出镜人的原声转写" in r.content


async def test_实拍配方没给出镜素材_用TTS兜底():
    store = AssetStore()
    reg = Registry(store)
    fns = _fns(store, reg, _plan())
    fns.recipes_dir = None
    made = await fns.invoke("short_video_brief", {"keyword": "k", "style": "talking-head"})
    bid = made.asset_ref
    fns.approve(bid)
    r = await fns.invoke("short_video_produce", {"brief_id": bid, "confirm": True})
    assert r.ok, r.error
    assert reg.of("tts") and "TTS 口播兜底" in r.content


async def test_改版后的简报也复用上次的镜头(tmp_path):
    store = AssetStore()
    reg = Registry(store)
    fns = _fns(store, reg)
    bid = await _brief(fns)
    fns.approve(bid)
    assert (await fns.invoke("short_video_produce", {"brief_id": bid, "confirm": True})).ok
    n = len(reg.of("gen_video"))
    new = (await fns.invoke("resolve_materials", {"brief_id": bid, "reply": "第2镜 生成"}))
    reg.calls.clear()
    r = await fns.invoke("short_video_produce", {"brief_id": new.asset_ref})
    assert r.ok and not r.suspend, "只剩第 2 镜要生成时才需要确认；其余全部复用"
    assert len(reg.of("gen_video")) == 0 or len(reg.of("gen_video")) < n


# ---------------------------------------------------------------- 热点与版权


def test_热榜条目和关键词沾不沾边():
    assert _relevant("AI 芯片禁令升级", "AI芯片")
    assert _relevant("国产芯片新突破", "芯片产能")
    assert not _relevant("某明星官宣结婚", "芯片")


def test_版权状态认授权信息():
    store = AssetStore()
    theirs = store.create("", type_=AssetType.VIDEO, creator="tool:fetch_douyin")
    stock = store.create("", type_=AssetType.VIDEO, creator="tool:fetch_stock_media",
                         gen_params={"license": "Pexels License（可商用）"})
    url = store.create("", type_=AssetType.VIDEO, creator="tool:fetch_stock_media",
                       gen_params={"license": "unknown（来源不明）", "source_url": "https://x"})
    mine = store.create("", type_=AssetType.IMAGE, creator="model:gpt-image-2",
                        gen_params={"source_url": "https://cdn/x.png"})
    assert rights_of(theirs) == "unknown", "别人的原片不是自己生成的"
    assert rights_of(stock) == "licensed" and rights_of(url) == "unknown"
    assert rights_of(mine) == "generated"

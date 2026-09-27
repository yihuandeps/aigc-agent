"""2026-09-26：「要人同意」的闸门，不能由模型自己填个参数就放行。

审查发现五处都是模型自己就能填的参数，人从头到尾不知情：
  · 出片确认     short_video_produce(confirm=true) 第一次就带上，确认单根本不出现、直接花钱
  · 带病打包     build_release_package(override_reason=…) 模型写一句理由，block 级问题进了待发布包
  · 放开禁字     gen_video(allow_text=true) 绕过用户定的最高优先级规则「画面不许有字」
  · 自报授权     fetch_media_url(license=…) 版权状态从 unknown 变成 licensed；fs_import 重新登记
                 一遍下载回来的别人的原片，直接洗成「人给的」
  · 大节点降级   request_review(major=false) 让 /auto 自动采纳「视频生成」这种大节点
只有「换模型」那道门做对了：放行记录来自人的决定。这里钉住：要么只认人的决定（总线上人采纳的
事件 / 终端 approve），要么当场按 L-external 问人（/auto 也问、默认不放行，模型没法替人答）。

同批：短视频缺镜头不合成（和短剧「缺段不成片」同一条规则）；配方的 review / grounding 字段
接上；广告成片的上屏文字用 overlay_text 后期叠（之前没有能往视频上叠字的工具）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from aigc_agent.domain.assets.store import AssetStore, AssetType, rights_of
from aigc_agent.domain.functions.distribution import DistributionFunctions
from aigc_agent.domain.functions.files import FileFunctions, FsPolicy
from aigc_agent.domain.functions.materials import MaterialFunctions
from aigc_agent.domain.functions.media import MediaFunctions
from aigc_agent.domain.functions.short_video import CONFIRM_STAGE, REVIEW_STAGE, SCRIPT_STAGE
from aigc_agent.domain.functions.video_edit import VideoEditFunctions
from aigc_agent.domain.generators.catalog import MediaCatalog
from aigc_agent.domain.media import ffmpeg, overlay
from aigc_agent.domain.system_prompt import MINOR_STAGE
from aigc_agent.harness.events.bus import Event, EventBus, EventType
from aigc_agent.harness.model.budget import CostGuard
from aigc_agent.harness.model.media import FakeMediaProvider, MediaGateway
from aigc_agent.harness.permission.gate import PermissionGate
from aigc_agent.harness.tools.provider import PermissionLevel, ToolResult
from aigc_agent.harness.tools.registry import ToolRegistry
from tests.test_short_video import Registry, _fns, _plan

CATALOG = Path(__file__).resolve().parents[1] / "config" / "media_models.yaml"
TITLE = "AI 芯片为什么不够用"


def _decided(bid: str, decision: str = "adopt", by: str = "human") -> Event:
    return Event(type=EventType.CHECKPOINT_DECIDED, data={
        "node": CONFIRM_STAGE, "decision": decision, "decided_by": by, "candidates": [bid],
    })


async def _brief(fns: Any, style: str = "tech-short") -> str:
    r = await fns.invoke("short_video_brief", {"keyword": "k", "style": style})
    assert r.ok, r.error
    return r.asset_ref


# ---------------------------------------------------------------- 出片确认只认人的决定


async def test_没经过人的confirm不算数_自动采纳和打回也不算_人采纳才生成():
    store = AssetStore()
    reg = Registry(store)
    fns = _fns(store, reg)
    bid = await _brief(fns)

    r = await fns.invoke("short_video_produce", {"brief_id": bid, "confirm": True})
    assert r.suspend and r.suspend_payload["stage"] == CONFIRM_STAGE
    assert r.suspend_payload["major"] is True and "不算数" in r.content
    assert not reg.of("gen_video"), "模型第一次就带 confirm=true，也得先出确认单"

    fns.on_event(_decided(bid, by="auto"))  # /auto 自动采纳的不算人点头
    r = await fns.invoke("short_video_produce", {"brief_id": bid, "confirm": True})
    assert r.suspend and not reg.of("gen_video")

    fns.on_event(_decided(bid, decision="revise"))  # 打回不算
    r = await fns.invoke("short_video_produce", {"brief_id": bid, "confirm": True})
    assert r.suspend and not reg.of("gen_video")

    fns.on_event(_decided(bid))  # 人采纳
    r = await fns.invoke("short_video_produce", {"brief_id": bid, "confirm": True})
    assert r.ok and not r.suspend and len(reg.of("gen_video")) == 3

    # 用一次就收回：同一张单子不能反复刷
    assert bid not in fns._confirmed  # noqa: SLF001


async def test_确认过的单子_换了档位要重新确认():
    store = AssetStore()
    reg = Registry(store)
    fns = _fns(store, reg)
    bid = await _brief(fns)
    ask = await fns.invoke("short_video_produce", {"brief_id": bid})
    assert ask.suspend and "第 1、2、3 镜" in ask.suspend_payload["question"]
    fns.on_event(_decided(bid))
    r = await fns.invoke("short_video_produce", {"brief_id": bid, "confirm": True,
                                                 "tier": "quality"})
    assert r.suspend and not reg.of("gen_video"), "人确认的是 fast 档，换 quality 要重新确认"


async def test_这一轮结束没用掉的确认收回():
    store = AssetStore()
    reg = Registry(store)
    fns = _fns(store, reg)
    bid = await _brief(fns)
    await fns.invoke("short_video_produce", {"brief_id": bid})
    fns.on_event(_decided(bid))
    fns.on_event(Event(type=EventType.LOOP_END, data={"stop_reason": "awaiting_review"}))
    assert bid in fns._confirmed, "挂起等人审的那种 LOOP_END 不收"  # noqa: SLF001
    fns.on_event(Event(type=EventType.LOOP_END, data={"stop_reason": "no_tool_calls"}))
    r = await fns.invoke("short_video_produce", {"brief_id": bid, "confirm": True})
    assert r.suspend and not reg.of("gen_video")


# ---------------------------------------------------------------- 缺镜头不成片


async def test_缺镜头不合成_重跑只补缺的():
    store = AssetStore()
    reg = Registry(store, fail_once={f"{TITLE}·第3镜": "内容安全拒绝"})
    fns = _fns(store, reg)
    bid = await _brief(fns)
    fns.approve(bid)
    r = await fns.invoke("short_video_produce", {"brief_id": bid, "confirm": True})
    assert r.ok and "未成片" in r.content and "第 3 镜" in r.content
    assert r.meta.get("complete") is False and r.meta.get("missing") == 1
    assert not reg.of("compose_video") and not reg.of("tts"), "缺镜头就不合成、也不花配音的钱"
    assert "skip_shots" in r.content and "gen_video" in r.content

    reg.calls.clear()
    fns.approve(bid)
    r = await fns.invoke("short_video_produce", {"brief_id": bid, "confirm": True})
    assert [g["summary"] for g in reg.of("gen_video")] == [f"{TITLE}·第3镜"], "只补缺的"
    assert r.ok and len(reg.of("compose_video")[0]["clips"]) == 3


async def test_用户说缺的不要了_skip_shots要当场问人_然后缺着出片():
    store = AssetStore()
    reg = Registry(store, fail_once={f"{TITLE}·第3镜": "内容安全拒绝"})
    fns = _fns(store, reg)
    bid = await _brief(fns)
    fns.approve(bid)
    assert "未成片" in (await fns.invoke(
        "short_video_produce", {"brief_id": bid, "confirm": True})).content

    level, why = fns.permission_for("short_video_produce", {"brief_id": bid, "skip_shots": [3]})
    assert level is PermissionLevel.EXTERNAL and "第 3 镜" in why
    assert fns.permission_for("short_video_produce", {"brief_id": bid}) is None

    reg.calls.clear()
    r = await fns.invoke("short_video_produce", {"brief_id": bid, "skip_shots": [3]})
    assert r.ok and not r.suspend and not reg.of("gen_video"), "第 1、2 镜复用，第 3 镜不要了"
    assert len(reg.of("compose_video")[0]["clips"]) == 2
    assert "按用户要求去掉第 3 镜" in r.content


# ---------------------------------------------------------------- 配方的 review / grounding 接上


async def test_配方要文案审核_简报出来先停_不算大节点():
    store = AssetStore()
    fns = _fns(store, Registry(store))
    r = await fns.invoke("short_video_brief", {"keyword": "k", "style": "tech-short"})
    assert r.suspend and r.asset_ref
    assert r.suspend_payload["stage"] == SCRIPT_STAGE and r.suspend_payload["major"] is False
    assert r.suspend_payload["assets"] == [r.asset_ref]


async def test_产品广告成片出来先给人审_auto也停():
    store = AssetStore()
    reg = Registry(store)
    shots = [{"desc": f"镜头{i}", "seconds": 4, "source": "generate"} for i in (1, 2, 3)]
    fns = _fns(store, reg, _plan(shots=shots, duration_seconds=12))
    bid = await _brief(fns, style="product-ad")
    fns.approve(bid)
    r = await fns.invoke("short_video_produce", {"brief_id": bid, "confirm": True})
    assert r.suspend and r.suspend_payload["stage"] == REVIEW_STAGE
    assert r.suspend_payload["major"] is True and r.suspend_payload["assets"] == [r.asset_ref]
    assert reg.of("compose_video") and "overlay_text" in r.content


class _HotRegistry(Registry):
    async def invoke(self, name: str, args: dict[str, Any]) -> ToolResult:
        if name == "douyin_hot_list":
            self.calls.append((name, dict(args)))
            text = " 1. 氢能储运瓶颈加速打通  热度 1194.7w  [科技]"
            a = self.store.create(text, summary="抖音热榜", creator="tool:douyin_hot_list")
            return ToolResult(content=text, asset_ref=a.id)
        return await super().invoke(name, args)


async def test_配方要接热榜_对话里没抓也自动拉一次_用户不要就不拉():
    store = AssetStore()
    reg = _HotRegistry(store)
    fns = _fns(store, reg)
    r = await fns.invoke("short_video_brief", {"keyword": "氢能", "style": "tech-short"})
    hot = reg.of("douyin_hot_list")
    assert hot and hot[0]["category"] == "科技"
    assert "氢能储运瓶颈" in fns.gateway.calls[-1][1], "热榜原文要进写简报的提示词"
    assert "已自动拉了一次抖音热榜" in r.content
    assert store.get(r.asset_ref).parent_ids, "简报的来源要记下热榜资产"

    reg.calls.clear()
    await fns.invoke("short_video_brief", {"keyword": "氢能", "style": "tech-short",
                                           "grounding": False})
    assert not reg.of("douyin_hot_list")
    await fns.invoke("short_video_brief", {"keyword": "饮料", "style": "product-ad"})
    assert not reg.of("douyin_hot_list"), "产品广告的配方不接热榜"


# ---------------------------------------------------------------- 按参数当场问人


async def _media_reg(asker: Any = None, guard: CostGuard | None = None):
    catalog = MediaCatalog.load(CATALOG)
    bus = EventBus()
    gw = MediaGateway({catalog.provider: FakeMediaProvider(urls=["https://x/a.mp4"])}, bus,
                      poll_interval=0.01, max_poll_interval=0.02)
    fns = MediaFunctions(gw, catalog, AssetStore())
    fns.video_lock = next(m.id for m in catalog.video if m.id.startswith("seedance-2.0"))
    reg = ToolRegistry(bus)
    reg.register(fns)
    await reg.refresh()
    reg.gate = PermissionGate(bus, asker=asker, guard=guard)
    return reg, fns


async def test_放开画面禁字要当场问人_单项里写的也算():
    reg, _ = await _media_reg()
    m = reg.meta_for_call("gen_video", {"prompt": "x", "allow_text": True})
    assert m is not None and m.permission is PermissionLevel.EXTERNAL and "最高优先级" in m.summary
    m = reg.meta_for_call("gen_videos", {"jobs": [{"prompt": "a"}, {"prompt": "b",
                                                                   "allow_text": True}]})
    assert m is not None and m.permission is PermissionLevel.EXTERNAL
    m = reg.meta_for_call("gen_video", {"prompt": "x"})
    assert m is not None and m.permission is PermissionLevel.COMPUTE


async def test_人拒绝就不生成_人同意也照样计次():
    answers = iter([False, True])

    async def asker(meta: Any, args: Any) -> bool:
        return next(answers)

    guard = CostGuard(call_limits={"video": 5})
    reg, _ = await _media_reg(asker, guard)
    args = {"prompt": "空镜，海边日落", "allow_text": True, "allow_no_refs": True, "duration": 5}
    r = await reg.invoke("gen_video", args)
    assert not r.ok and "用户拒绝" in (r.error or "")
    assert guard.usage.calls.get("video", 0) == 0
    r = await reg.invoke("gen_video", args)
    assert r.ok, r.error
    # 提到 L-external 之前会连预算护栏一起跳过（只认 L-compute）
    assert guard.usage.calls.get("video", 0) == 1, "人点了头也要计次计秒"


def test_带病打包_填了放行理由要当场问人():
    fns = DistributionFunctions(None, None, AssetStore())  # type: ignore[arg-type]
    level, why = fns.permission_for(
        "build_release_package", {"content_asset_id": "as_x", "platform": "douyin",
                                  "override_reason": "用户说先发了再改"})
    assert level is PermissionLevel.EXTERNAL and "带病发布" in why and "先发了再改" in why
    assert fns.permission_for("build_release_package", {"override_reason": " "}) is None


def test_下载素材自报授权要当场问人(tmp_path: Path):
    fns = MaterialFunctions(AssetStore(), tmp_path)
    level, why = fns.permission_for("fetch_media_url", {"url": "https://x/a.mp4",
                                                        "license": "品牌方官方物料"})
    assert level is PermissionLevel.EXTERNAL and "品牌方官方物料" in why
    assert fns.permission_for("fetch_media_url", {"url": "https://x/a.mp4"}) is None
    assert fns.permission_for("fetch_media_url", {"license": "unknown"}) is None


async def test_重新登记下载回来的文件_不洗成人给的(tmp_path: Path):
    store = AssetStore(tmp_path / "assets")
    ws, out, ext = tmp_path / "ws", tmp_path / "out", tmp_path / "ext"
    for d in (ws / "douyin", out, ext):
        d.mkdir(parents=True, exist_ok=True)
    fns = FileFunctions(store, ws, out, FsPolicy(roots=[ext, ws]), project_root=tmp_path / "proj")

    theirs = ext / "别人的原片.mp4"
    theirs.write_bytes(b"mp4")
    got = store.create("", type_=AssetType.VIDEO, summary="抓的", creator="tool:fetch_douyin",
                       gen_params={"local": str(theirs)})
    r = await fns.invoke("fs_import", {"path": str(theirs)})
    assert r.ok, r.error
    again = store.get(r.asset_ref)
    assert again.creator == "tool:fetch_douyin" and rights_of(again) == "unknown"
    assert again.gen_params["imported_from"] == got.id

    loose = ws / "douyin" / "没登记过的.mp4"
    loose.write_bytes(b"mp4")
    r = await fns.invoke("fs_import", {"path": str(loose)})
    assert r.ok, r.error
    assert rights_of(store.get(r.asset_ref)) == "unknown", "Agent 自己下载目录里的，来源不明"

    mine = ext / "我自己拍的.mp4"
    mine.write_bytes(b"mp4")
    r = await fns.invoke("fs_import", {"path": str(mine)})
    assert rights_of(store.get(r.asset_ref)) == "human"


def test_小节点要写明第几集_集数不算():
    assert MINOR_STAGE.search("剧本第3集") and MINOR_STAGE.search("第十二集")
    assert MINOR_STAGE.search("单集")
    assert not MINOR_STAGE.search("图片生成（全剧12集）")
    assert not MINOR_STAGE.search("12集")


# ---------------------------------------------------------------- 上屏文字后期叠


def test_上屏文字排版_时间夹到片长_花括号不当控制码():
    items, notes = overlay.parse_items(
        [
            {"text": "一口入魂", "start": 0.5, "end": 3, "position": "center", "size": "xl"},
            {"text": "  "},
            {"text": "卖点{1}\\第二行", "position": "左边", "end": 99},
        ],
        {"title": "清爽一夏", "subtitle": "今夏新品"},
        15.0,
    )
    assert [i.text for i in items] == ["一口入魂", "卖点{1}\\第二行", "清爽一夏", "今夏新品"]
    assert any("位置" in n for n in notes) and items[1].position == "center"
    assert items[1].end == 15.0, "超过片长的夹回片尾"
    card = items[2]
    assert card.start == 12.5 and card.end == 15.0 and card.size == "xl"
    ass = overlay.build_ass(items, 1080, 1920)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass and overlay.FONT in ass
    assert "0:00:12.50" in ass and "卖点｛1｝＼第二行" in ass
    assert ass.count("Dialogue:") == 4


async def test_成片叠字_本机ffmpeg实测(tmp_path: Path):
    if not ffmpeg.have_ffmpeg():
        pytest.skip("需要 ffmpeg")
    src = tmp_path / "src.mp4"
    code, err = await ffmpeg.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=0x224466:s=240x426:d=3:r=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src),
    ])
    assert code == 0, err
    store = AssetStore()
    a = store.create("", type_=AssetType.VIDEO, summary="成片", creator="tool:compose_video",
                     gen_params={"local": str(src)})
    a.uri = str(src)
    store.put(a)
    fns = VideoEditFunctions(store, tmp_path / "ws")
    r = await fns.invoke("overlay_text", {
        "video": a.id, "items": [{"text": "限时 7 折", "position": "top", "size": "m"}],
        "end_card": {"title": "清爽一夏", "seconds": 1.5}, "out_dir": str(tmp_path / "out"),
    })
    assert r.ok, r.error
    made = store.get(r.asset_ref)
    final = Path(made.gen_params["local"])
    assert await asyncio.to_thread(final.exists)
    assert final.name == "src_字.mp4" and made.parent_ids == [a.id]
    info = await ffmpeg.probe(final)
    assert abs(info.duration - 3.0) < 0.3 and info.has_audio, "音轨原样保留"
    assert json.dumps(made.gen_params["overlay"], ensure_ascii=False).count("清爽一夏") == 1
    assert await asyncio.to_thread(src.exists), "原片不动"

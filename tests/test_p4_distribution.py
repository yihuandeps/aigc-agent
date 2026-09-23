"""M16 分发验收 —— 待发布包。

发布方式定了：只产待发布包，由人上传。验：
  · 包目录三份文件齐全，manifest 带 AIGC 隐式标识与血缘
  · 格式适配：话题超上限保留前 N 个；标题/正文超长、类型不支持**列为问题不截断**
  · 机审有 block 拒绝打包，除非人给放行理由并记进 manifest
  · 图模式：copy 图的 finalize 节点用 $draft 引用槽位，产出真包
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.compliance import ComplianceChecker, ComplianceRules
from aigc_agent.domain.distribution import Packager, PlatformCatalog
from aigc_agent.domain.functions.content import ContentFunctions
from aigc_agent.domain.functions.distribution import DistributionFunctions
from aigc_agent.domain.pipeline.executors import ToolNodeExecutor
from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.execution.graph.models import Decision, GraphDef, NodeType
from aigc_agent.harness.execution.graph.runtime import GraphRuntime
from aigc_agent.harness.tools.registry import ToolRegistry

from .test_p1_graph import FakeAgentExecutor  # noqa: TID252

ROOT = Path(__file__).resolve().parents[1]
CATALOG = PlatformCatalog.load(ROOT / "config" / "platforms.yaml")
RULES = ComplianceRules.load(ROOT / "config" / "compliance.yaml")
GRAPH = ROOT / "src/aigc_agent/domain/pipeline/graphs/copy.yaml"
CLEAN = "# 露营装备怎么选\n\n本内容由 AI 生成。新手先看睡袋温标，再看帐篷防水。"


def _env(tmp_path: Path):
    store = AssetStore(tmp_path / "assets")
    packager = Packager(tmp_path / "releases", CATALOG, store)
    return store, packager, ComplianceChecker(RULES)


def test_平台规格可加载():
    assert {"douyin", "xiaohongshu", "wechat", "zhihu", "generic"} <= set(CATALOG.specs)
    assert CATALOG.get("wechat").kinds == ["text"] and CATALOG.get("wechat").tags_max == 0
    assert "抖音" in CATALOG.render()


def test_打包产出目录与三份文件(tmp_path: Path):
    store, packager, checker = _env(tmp_path)
    content = store.create(CLEAN, creator="model:k3", summary="露营")
    report = checker.check(CLEAN, platform="wechat", generated=True)
    assert report.passed

    asset, m = packager.build(content.id, "wechat", tags=["露营", "#装备"], report=report)
    folder = Path(asset.uri)
    assert (folder / "content.md").read_text(encoding="utf-8") == CLEAN, "正文原样，不改字"
    upload = (folder / "upload.md").read_text(encoding="utf-8")
    assert "待发布包 · 微信公众号" in upload and "AIGC 标识" in upload
    assert "本内容由 AI 生成" in upload
    saved = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert saved["package_id"] == m.package_id

    assert m.title == "露营装备怎么选", "标题取正文第一行并去掉 #"
    assert m.tags == [] and any("不支持话题标签" in i for i in m.issues)
    assert m.kind == "text" and m.status == "ready"
    assert m.aigc["generated"] and m.aigc["generators"] == ["model:k3"]
    assert m.aigc["implicit"]["content_asset"] == content.id
    assert m.lineage == [content.id]
    assert asset.type is AssetType.PACKAGE and asset.parent_ids == [content.id]
    assert store.content(asset.id) == upload


def test_标题超长与正文超长列为问题不截断(tmp_path: Path):
    store, packager, _ = _env(tmp_path)
    content = store.create(CLEAN, creator="model:k3")
    long_title = "标" * 25  # 小红书上限 20
    with pytest.raises(ValueError, match="标题 25 字"):
        packager.build(content.id, "xiaohongshu", title=long_title)
    asset, m = packager.build(
        content.id, "xiaohongshu", title=long_title, override_reason="运营确认平台已放宽"
    )
    assert m.title == long_title, "不截断"
    assert any(i.startswith("[block] 标题") for i in m.issues)
    assert m.override_reason == "运营确认平台已放宽"


def test_话题超上限只保留前N并注明(tmp_path: Path):
    store, packager, _ = _env(tmp_path)
    content = store.create(CLEAN, creator="model:k3")
    video = store.create("", type_=AssetType.VIDEO, summary="成片", creator="tool:compose_video")
    video.uri = "https://example.com/v.mp4"
    store.put(video)
    tags = [f"tag{i}" for i in range(8)] + ["tag0"]  # 重复的去掉
    _, m = packager.build(content.id, "douyin", tags=tags, media_asset_ids=[video.id])
    assert m.kind == "video"
    assert m.tags == ["tag0", "tag1", "tag2", "tag3", "tag4"]
    assert any("只保留前 5 个" in i for i in m.issues)
    assert m.files[video.id] == "https://example.com/v.mp4", "外链只记 URL"


def test_平台不支持的类型(tmp_path: Path):
    store, packager, _ = _env(tmp_path)
    content = store.create(CLEAN, creator="model:k3")
    video = store.create("", type_=AssetType.VIDEO, summary="成片", creator="tool:x")
    with pytest.raises(ValueError, match="不支持 video"):
        packager.build(content.id, "wechat", media_asset_ids=[video.id])
    with pytest.raises(ValueError, match="未知平台"):
        packager.build(content.id, "weibo")


def test_机审block拒绝打包除非人放行(tmp_path: Path):
    store, packager, checker = _env(tmp_path)
    bad = "最好的露营装备"  # 极限词 + 无 AI 标识
    content = store.create(bad, creator="model:k3")
    report = checker.check(bad, generated=True)
    assert not report.passed
    with pytest.raises(PermissionError, match="block"):
        packager.build(content.id, "generic", report=report)
    asset, m = packager.build(content.id, "generic", report=report, override_reason="法务已确认")
    assert m.override_reason == "法务已确认" and m.compliance["passed"] is False
    assert "带病发布" in store.content(asset.id)


def test_本地媒体复制进包(tmp_path: Path):
    store, packager, _ = _env(tmp_path)
    content = store.create(CLEAN, creator="model:k3")
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 64)
    video = store.create("", type_=AssetType.VIDEO, summary="片段", creator="model:veo")
    video.uri = str(src)
    store.put(video)
    asset, m = packager.build(content.id, "douyin", media_asset_ids=[video.id])
    copied = Path(asset.uri) / "media" / f"{video.id}.mp4"
    assert copied.exists() and copied.stat().st_size == 64
    assert m.files[video.id] == f"media/{video.id}.mp4"
    assert set(m.aigc["generators"]) == {"model:k3", "model:veo"}
    assert asset.parent_ids == [content.id, video.id]


def test_mark_published写回manifest(tmp_path: Path):
    store, packager, _ = _env(tmp_path)
    content = store.create(CLEAN, creator="model:k3")
    asset, _ = packager.build(content.id, "wechat")
    m = packager.mark_published(asset.id, "https://mp.weixin.qq.com/s/abc")
    assert m.status == "published" and m.published_url.endswith("/abc") and m.published_at > 0
    on_disk = json.loads((Path(asset.uri) / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "published"
    assert store.get(asset.id).gen_params["manifest"]["published_url"].endswith("/abc")
    (a, mm), = packager.packages()
    assert a.id == asset.id and mm.status == "published"


# ---------------------------------------------------------------- function + 图


async def test_function入口与事件(tmp_path: Path):
    store, packager, checker = _env(tmp_path)
    bus = EventBus()
    reg = ToolRegistry(bus)
    reg.register(ContentFunctions(store))
    reg.register(DistributionFunctions(packager, checker, store, bus))
    await reg.refresh()

    r = await reg.invoke("save_draft", {"content": CLEAN, "kind": "copy"})
    draft = r.asset_ref
    r = await reg.invoke("build_release_package", {"content_asset_id": draft, "platform": "wechat"})
    assert r.ok, r.error
    pkg = store.get(r.asset_ref)
    assert pkg.type is AssetType.PACKAGE
    assert len(pkg.parent_ids) == 2, "正文 + 机审报告"
    assert store.get(pkg.parent_ids[1]).type is AssetType.REPORT
    assert any(e.type is EventType.PACKAGE_BUILT for e in bus.history)

    r = await reg.invoke("mark_published", {"package_asset_id": pkg.id, "url": "https://x/1"})
    assert r.ok and any(e.type is EventType.PACKAGE_PUBLISHED for e in bus.history)
    r = await reg.invoke("list_packages", {})
    assert pkg.id in r.content and "published" in r.content
    r = await reg.invoke("list_platforms", {})
    assert "抖音" in r.content

    bad = await reg.invoke("save_draft", {"content": "最好的产品", "kind": "copy"})
    r = await reg.invoke(
        "build_release_package", {"content_asset_id": bad.asset_ref, "platform": "wechat"}
    )
    assert not r.ok and "block" in r.error


async def test_copy图的finalize节点产出真包(tmp_path: Path):
    graph = GraphDef.load(GRAPH)
    fin = graph.node("finalize")
    assert fin.tool == "build_release_package" and fin.tool_args["content_asset_id"] == "$draft"

    store, packager, checker = _env(tmp_path)
    bus = EventBus()
    reg = ToolRegistry(bus)
    reg.register(DistributionFunctions(packager, checker, store, bus))
    await reg.refresh()
    rt = GraphRuntime(
        bus, {NodeType.AGENT: FakeAgentExecutor(store), NodeType.TOOL: ToolNodeExecutor(reg, store)}
    )
    state = rt.new_state(graph)
    state.slots["topic"] = store.create("露营", creator="human:cli").id
    state = await rt.run(graph, state)
    assert state.status == "awaiting_review"
    state = await rt.resume(graph, state, Decision.ADOPT)
    assert state.status == "done"

    final = store.get(state.slots["final"])
    assert final.type is AssetType.PACKAGE, "工具自己落的资产直接接到槽位上"
    m = packager.manifest_of(final)
    assert m.content_asset == state.slots["draft"] and m.platform == "wechat"
    assert m.override_reason, "假草稿没有 AI 标识 → 机审 block → 靠图里的放行理由打包"
    assert m.compliance["passed"] is False

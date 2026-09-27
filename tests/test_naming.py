"""产物文件命名（2026-09-18）：生成时带序号落盘，已生成的能按同一套规则补改。

用户要的是后期剪辑能按文件名排出顺序：`as_7f79f5c08d.mp4` 排不了，
`第01集-03_2场_镜9-18.mp4` 可以。
"""

from __future__ import annotations

import json
from pathlib import Path

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions.drama import DramaFunctions
from aigc_agent.domain.functions.media import MediaFunctions
from aigc_agent.domain.generators.catalog import MediaCatalog
from aigc_agent.domain.media.naming import (
    apply_renames,
    clip_name,
    pad_numbers,
    parse_scene,
    plan_renames,
    recipe_shot_name,
    reference_name,
    safe_name,
    unique_path,
    write_manifest,
)
from aigc_agent.domain.output import OutputPrefs
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.model.media import FakeMediaProvider, MediaGateway
from aigc_agent.harness.tools.provider import ToolResult
from aigc_agent.harness.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "config" / "media_models.yaml"


# ---------------------------------------------------------------- 规则


def test_文件名去掉非法字符并补零():
    assert safe_name('a/b:c*d?"<>|  e') == "a-b-c-d-e"
    assert safe_name("陆离-深色风衣-[8/1]") == "陆离-深色风衣-[8-1]"
    assert safe_name("  ..  ") == "untitled"
    assert pad_numbers("第1集第3镜·第12场") == "第01集第03镜·第12场"
    assert parse_scene("[第1集-1场]") == (1, "1场")
    assert parse_scene("开场") == (0, "开场")
    assert clip_name("[第1集-3场]", "9-18", 3) == "第01集-03_3场_镜9-18"
    assert clip_name("片头", "1-2", 1) == "片段-01_片头_镜1-2"
    assert reference_name("服装", 2, "陆离-深色风衣-[8/1]") == "参考图-服装-02_陆离-深色风衣-[8-1]"
    assert recipe_shot_name("AI 芯片", 3) == "AI-芯片_第03镜"


def test_同名不覆盖加版本号(tmp_path):
    assert unique_path(tmp_path, "a.mp4") == tmp_path / "a.mp4"
    (tmp_path / "a.mp4").write_bytes(b"x")
    assert unique_path(tmp_path, "a.mp4") == tmp_path / "a-v2.mp4"
    (tmp_path / "a-v2.mp4").write_bytes(b"x")
    assert unique_path(tmp_path, "a.mp4") == tmp_path / "a-v3.mp4"


# ---------------------------------------------------------------- 生成时落盘


async def _fake_download(url: str, dest: Path) -> bool:
    dest.write_bytes(b"media")  # noqa: ASYNC240 — 测试替身，几个字节
    return True


async def _media(tmp_path: Path):
    store = AssetStore()
    bus = EventBus()
    catalog = MediaCatalog.load(CATALOG_PATH)
    gw = MediaGateway(
        {"apimart": FakeMediaProvider(urls=["https://x/a.mp4"])},
        bus, poll_interval=0.01, max_poll_interval=0.02,
    )
    prefs = OutputPrefs(tmp_path, downloader=_fake_download)
    reg = ToolRegistry(bus)
    reg.register(MediaFunctions(gw, catalog, store, prefs=prefs))
    await reg.refresh()
    return reg, store


async def test_gen_video按local_name落盘_重渲不覆盖(tmp_path):
    reg, store = await _media(tmp_path)
    r = await reg.invoke(
        "gen_video", {"prompt": "x", "prefer": "fast", "local_name": "第01集-01_1场_镜1-4"}
    )
    assert r.ok, r.error
    assert (tmp_path / "videos" / "第01集-01_1场_镜1-4.mp4").exists()
    assert store.get(r.asset_ref).gen_params["local"].endswith("第01集-01_1场_镜1-4.mp4")
    assert "第01集-01_1场_镜1-4.mp4" in r.content

    r2 = await reg.invoke(
        "gen_video", {"prompt": "x", "prefer": "fast", "local_name": "第01集-01_1场_镜1-4"}
    )
    assert (tmp_path / "videos" / "第01集-01_1场_镜1-4-v2.mp4").exists(), "重渲加版本号，旧版还在"
    assert store.get(r2.asset_ref).gen_params["local"].endswith("-v2.mp4")

    r3 = await reg.invoke("gen_video", {"prompt": "x", "prefer": "fast"})
    assert (tmp_path / "videos" / f"{r3.asset_ref}.mp4").exists(), "没给名字仍用资产 id"


async def test_gen_image多张按序号编(tmp_path):
    store = AssetStore()
    bus = EventBus()
    catalog = MediaCatalog.load(CATALOG_PATH)
    gw = MediaGateway(
        {"apimart": FakeMediaProvider(urls=["https://x/a.png", "https://x/b.png"])},
        bus, poll_interval=0.01, max_poll_interval=0.02,
    )
    reg = ToolRegistry(bus)
    reg.register(MediaFunctions(gw, catalog, store, prefs=OutputPrefs(tmp_path, _fake_download)))
    await reg.refresh()
    r = await reg.invoke(
        "gen_image", {"prompt": "x", "prefer": "fast", "n": 2, "local_name": "海报/候选"}
    )
    assert r.ok, r.error
    names = sorted(p.name for p in (tmp_path / "images").iterdir())
    assert names == ["海报-候选-1.png", "海报-候选-2.png"]


# ---------------------------------------------------------------- 短剧渲染传名字


class _Reg:
    def __init__(self, store: AssetStore) -> None:
        self.store = store
        self.calls: list[tuple[str, dict]] = []

    async def invoke(self, name, args):
        self.calls.append((name, args))
        if name == "compose_video":
            return ToolResult(ok=True, content="拼好了", asset_ref=self.store.create("m").id)
        a = self.store.create("", summary=args.get("summary", name), creator="fake")
        a.uri = f"https://fake/{a.id}"
        self.store.put(a)
        return ToolResult(ok=True, content="ok", asset_ref=a.id)

    def of(self, tool: str) -> list[dict]:
        return [a for n, a in self.calls if n == tool]


LIB = {
    "characters": [
        {"baseRoleName": "陆离", "roleTotalDesc": "守门人",
         "roleCostumeList": [{"costumeName": "陆离-深色风衣-[8/1]", "costumeDesc": "风衣"}]},
        {"baseRoleName": "小满", "roleTotalDesc": "女孩",
         "roleCostumeList": [{"costumeName": "小满-旧连帽衫-[前10集]", "costumeDesc": "卫衣"}]},
    ],
    "scenes": [{"name": "地铁车厢", "description": "末班车"}],
    "props": [{"name": "暗色骨簪", "description": "骨簪"}],
}
def _shot(scene: str, name: str, desc: str) -> dict:
    return {"scene_index": scene, "video_name": name, "video_duration": "14s", "description": desc}


SHOTS = [
    _shot("[第1集-1场]", "1-4", "a (陆离)"),
    _shot("[第1集-2场]", "5-8", "b (陆离)"),
    _shot("[第2集-1场]", "1-3", "c (小满)"),
]


async def test_参考图按类别序号命名():
    store = AssetStore()
    lib_id = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库").id
    reg = _Reg(store)
    r = await DramaFunctions(None, store, registry=reg)._fn_drama_render_assets(lib_id)
    assert r.ok, r.error
    names = {a["summary"]: a["local_name"] for a in reg.of("gen_image")}
    assert names["角色·陆离"] == "参考图-角色-01_陆离"
    assert names["角色·小满"] == "参考图-角色-02_小满"
    assert names["服装·陆离-深色风衣-[8/1]"] == "参考图-服装-01_陆离-深色风衣-[8-1]"
    assert names["服装·小满-旧连帽衫-[前10集]"] == "参考图-服装-02_小满-旧连帽衫-[前10集]"
    assert names["场景·地铁车厢"] == "参考图-场景-01_地铁车厢"
    assert names["道具·暗色骨簪"] == "参考图-道具-01_暗色骨簪"


async def test_分镜视频按集内序号命名_过滤不影响编号():
    store = AssetStore()
    shots_id = store.create(json.dumps(SHOTS, ensure_ascii=False), summary="提示词").id
    reg = _Reg(store)
    fns = DramaFunctions(None, store, registry=reg)
    r = await fns._fn_drama_render_shots(shots_id)
    assert r.ok, r.error
    assert [a["local_name"] for a in reg.of("gen_video")] == [
        "第01集-01_1场_镜1-4", "第01集-02_2场_镜5-8", "第02集-01_1场_镜1-3",
    ]
    assert "→ 第01集-02_2场_镜5-8.mp4" in r.content

    # reuse=False：默认会复用上一轮成功的片段，这里要看的是重新生成时的命名
    reg2 = _Reg(store)
    r2 = await DramaFunctions(None, store, registry=reg2)._fn_drama_render_shots(
        shots_id, episode=1, limit=1, reuse=False
    )
    assert r2.ok
    # 只渲第 1 集且 limit=1：编号仍按完整列表，不会因为过滤变成 01
    assert [a["local_name"] for a in reg2.of("gen_video")] == ["第01集-01_1场_镜1-4"]
    assert not reg2.of("compose_video"), "limit 只渲了一部分：不算渲完、不拼成片（2026-09-24）"

    # 整集渲完才拼，成片按集命名
    reg3 = _Reg(store)
    r3 = await DramaFunctions(None, store, registry=reg3)._fn_drama_render_shots(
        shots_id, episode=1, reuse=False
    )
    assert r3.ok and reg3.of("compose_video")[0]["filename"] == "第01集.mp4"


# ---------------------------------------------------------------- 已生成文件补改


def _media_asset(store, root, sub, summary, ext, creator="model:x", type_=None):
    a = store.create(
        "", type_=type_ or (AssetType.VIDEO if ext == ".mp4" else AssetType.IMAGE),
        summary=summary, creator=creator,
    )
    p = root / sub / f"{a.id}{ext}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    a.gen_params["local"] = str(p)
    store.put(a)
    return a


def test_已生成文件按包的顺序补改名(tmp_path):
    store = AssetStore()
    lib = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库",
                       creator="tool:drama_assets")
    shots = store.create(json.dumps(SHOTS, ensure_ascii=False), summary="提示词",
                         creator="tool:drama_shots", parents=["sb", lib.id])
    # 视频：渲染包只有第 2 段和第 3 段（第 1 段没渲），编号仍按提示词全列表
    v2 = _media_asset(store, tmp_path, "videos", "[第1集-2场] 5-8", ".mp4", "model:seedance")
    v3 = _media_asset(store, tmp_path, "videos", "[第2集-1场] 1-3", ".mp4", "model:seedance")
    store.create(json.dumps([
        {"scene": "[第1集-2场]", "name": "5-8", "asset": v2.id},
        {"scene": "[第2集-1场]", "name": "1-3", "asset": v3.id},
    ]), summary="分镜视频", creator="tool:drama_render_shots", parents=[shots.id])
    # 参考图：老格式的包（没有 kind），类别从资产库认
    img_c = _media_asset(store, tmp_path, "images", "角色·小满", ".png", "model:gpt")
    img_s = _media_asset(store, tmp_path, "images", "场景·地铁车厢", ".png", "model:gpt")
    store.create(json.dumps({
        "小满": {"asset": img_c.id, "url": "u1"},
        "地铁车厢": {"asset": img_s.id, "url": "u2"},
    }), summary="参考图", creator="tool:drama_render_assets", parents=[lib.id])
    # 零散的：配方短视频素材段、用户自己改过名的
    loose = _media_asset(store, tmp_path, "videos", "AI芯片·第3镜", ".mp4", "model:veo")
    renamed = _media_asset(store, tmp_path, "videos", "[第9集-1场] 1-2", ".mp4", "model:seedance")
    custom = tmp_path / "videos" / "我自己起的名.mp4"
    Path(renamed.gen_params["local"]).rename(custom)
    renamed.gen_params["local"] = str(custom)
    store.put(renamed)
    # 摘要是自动截的（以 … 结尾）：不改，名字不可靠
    _media_asset(store, tmp_path, "images", "原图模式，无美颜无磨皮…", ".png", "model:gpt")

    plan = plan_renames(store, tmp_path)
    got = {r.asset_id: r.new.name for r in plan}
    assert got[v2.id] == "第01集-02_2场_镜5-8.mp4"
    assert got[v3.id] == "第02集-01_1场_镜1-3.mp4"
    assert got[img_c.id] == "参考图-角色-02_小满.png"
    assert got[img_s.id] == "参考图-场景-01_地铁车厢.png"
    assert got[loose.id] == "AI芯片·第03镜.mp4"
    assert renamed.id not in got, "用户自己改过的名字不碰"
    assert len(plan) == 5

    # 冲突：目标名已被占用 → 加 -v2；被别的程序占用的文件跳过、其余照改
    (tmp_path / "videos" / "第02集-01_1场_镜1-3.mp4").write_bytes(b"old")
    locked = next(r for r in plan if r.asset_id == loose.id)
    handle = open(locked.old, "rb")  # noqa: SIM115 — 模拟播放器占着文件
    try:
        report = apply_renames(store, plan)
    finally:
        handle.close()
    if report.failed:  # Windows 上会被占用；其他平台改名不受打开句柄影响
        assert [r.asset_id for r, _ in report.failed] == [loose.id]
        assert "占用" in report.render_failed()
        assert len(report.done) == 4
        assert len(apply_renames(store, plan_renames(store, tmp_path)).done) == 1
    else:
        assert len(report.done) == 5
    done = report.done
    assert (tmp_path / "videos" / "第01集-02_2场_镜5-8.mp4").exists()
    assert (tmp_path / "videos" / "第02集-01_1场_镜1-3-v2.mp4").exists()
    assert not (tmp_path / "videos" / f"{v2.id}.mp4").exists()
    assert store.get(v2.id).gen_params["local"].endswith("第01集-02_2场_镜5-8.mp4")
    assert store.get(v3.id).gen_params["local"].endswith("-v2.mp4")

    assert done and plan_renames(store, tmp_path) == [], "改过一遍再跑是幂等的"

    manifest = write_manifest(store, tmp_path)
    assert manifest is not None
    text = manifest.read_text(encoding="utf-8")
    assert "videos/第01集-02_2场_镜5-8.mp4" in text and v2.id in text
    assert "我自己起的名.mp4" in text


def test_没有本地文件的资产不进计划(tmp_path):
    store = AssetStore()
    a = store.create("", type_=AssetType.VIDEO, summary="[第1集-1场] 1-4", creator="model:x")
    a.gen_params["local"] = str(tmp_path / "videos" / "gone.mp4")
    store.put(a)
    assert plan_renames(store, tmp_path) == []
    assert write_manifest(store, tmp_path) is None


def test_默认位置的文件没记local也能找到(tmp_path):
    store = AssetStore()
    a = store.create("", type_=AssetType.VIDEO, summary="[第1集-1场] 1-4", creator="model:x")
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / f"{a.id}.mp4").write_bytes(b"x")
    plan = plan_renames(store, tmp_path)
    # 不在任何渲染包里，只能按摘要命名（集号补零）
    assert len(plan) == 1 and plan[0].new.name == "[第01集-1场]-1-4.mp4"

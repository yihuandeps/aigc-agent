"""资产库定稿后冻结、只增量补新集；服装图不像主形象不进包（2026-09-26 用户定的第 6 条）。

- 渲过参考图的资产库 = 定稿：再调 drama_assets 只补新集里新出现的角色 / 服装 / 场景 / 道具，
  已有条目原样不动（模型重复输出、改写过的一律丢掉）；新集都在库里就不重做、不花钱
- 增量版渲参考图：名字和描述都没变的图沿用上一版，只生成新增的（之前换一版资产库就是
  174 张全部重生成、全员换脸）
- 整份重做（rebuild）要当场问人；重做后只有描述变了的条目重生成
- 同一条增量链上的资产库算同一套：旧提示词不算过期、参考图包照常能用；整份重做的不算
- 服装图和主形象不像（重生成过还是不像）：不进参考图包，渲视频时退回主形象；补生成过了再进包
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.drama.format import EpisodeFormat
from aigc_agent.domain.drama.identity import IdentityVerdict
from aigc_agent.domain.drama.parse import parse_assets
from aigc_agent.domain.drama.refpack import library_chain, pack_library, same_library
from aigc_agent.domain.functions.drama import DramaFunctions, _resolve_ref
from aigc_agent.domain.output import OutputPrefs
from aigc_agent.domain.pipeline.episode_pipeline import EpisodePipeline
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.tools.provider import PermissionLevel
from aigc_agent.harness.tools.registry import ToolRegistry
from tests.test_drama_stage_fixes import Registry as ImageRegistry
from tests.test_drama_stage_fixes import _lib as _stage_lib
from tests.test_drama_stage_fixes import _pack
from tests.test_episode_pipeline import FakeRegistry
from tests.test_gate_chain import SHOTS
from tests.test_gate_chain import _fns as _gate_fns
from tests.test_review_0924 import _scripts, _Seq

BASE = {
    "characters": [
        {
            "baseRoleName": "陆离",
            "roleTotalDesc": "男 | 35岁 | 守门人",
            "voice": "低沉",
            "roleCostumeList": [
                {"costumeName": "陆离-深色风衣-[1-2]", "costumeDesc": "深色风衣",
                 "scenes": ["地铁车厢"], "episodes": "1-2"},
            ],
        },
        {
            "baseRoleName": "小满",
            "roleTotalDesc": "女 | 30岁 | 记者",
            "voice": "清亮",
            "roleCostumeList": [
                {"costumeName": "小满-旧连帽衫-[全集]", "costumeDesc": "旧卫衣",
                 "episodes": "全集"},
            ],
        },
    ],
    "scenes": [{"name": "地铁车厢", "description": "末班车"}],
    "props": [{"name": "暗色骨簪", "description": "骨簪"}],
}

# 第 3 集的增量：一个新角色（带服装）、老角色的一套新服装、一个新场景；模型还把已有的
# 服装 / 角色 / 场景改写了一遍重复输出 —— 这些都不许进库
ADD = {
    "characters": [
        {
            "baseRoleName": "老鬼",
            "roleTotalDesc": "男 | 60岁 | 修表匠",
            "voice": "沙哑",
            "roleCostumeList": [
                {"costumeName": "老鬼-劳保服-[3]", "costumeDesc": "蓝色劳保服", "episodes": "3"},
            ],
        },
        {
            "baseRoleName": "陆离",
            "roleTotalDesc": "",
            "roleCostumeList": [
                {"costumeName": "陆离-白大褂-[3]", "costumeDesc": "白大褂", "episodes": "3"},
                {"costumeName": "陆离-深色风衣-[1-2]", "costumeDesc": "改写过的风衣"},
            ],
        },
        {"baseRoleName": "小满", "roleTotalDesc": "女 | 30岁 | 记者，又被改写了一遍"},
    ],
    "scenes": [
        {"name": "修表铺", "description": "老街尽头"},
        {"name": "地铁车厢", "description": "改写过的车厢"},
    ],
    "props": [],
}


def _j(d: Any) -> str:
    return json.dumps(d, ensure_ascii=False)


async def _finalized(store: AssetStore, gw: Any) -> dict[str, Any]:
    """第 1–2 集出资产库、渲好参考图（= 定稿）。项目里第 3 集的剧本也已经写好。"""
    ids = _scripts(store, (1, 2, 3))
    fns = DramaFunctions(gw, store, registry=None, catalog=None, fmt=EpisodeFormat())
    r = await fns._fn_drama_assets(script_ids=ids[:2], ethnicity="asian", language="zh")
    assert r.ok, r.error
    reg = ImageRegistry(store)
    painter = DramaFunctions(None, store, registry=reg)
    p = await painter._fn_drama_render_assets(r.asset_ref)
    assert p.ok, p.error
    return {"fns": fns, "ids": ids, "v1": r.asset_ref, "pack1": p.asset_ref,
            "reg": reg, "painter": painter}


def _names(lib: Any) -> dict[str, list[str]]:
    return {
        "角色": [c.name for c in lib.characters],
        "服装": [x.name for c in lib.characters for x in c.costumes],
        "场景": [s.name for s in lib.scenes],
        "道具": [p.name for p in lib.props],
    }


# ---------------------------------------------------------------- 定稿后只增量


async def test_定稿后续写新集_只增量补新出现的_已有条目原样冻结():
    store = AssetStore()
    gw = _Seq(_j(BASE), _j(ADD))
    s = await _finalized(store, gw)

    r = await s["fns"]._fn_drama_assets(script_ids=s["ids"], ethnicity="asian", language="zh")
    assert r.ok, r.error
    assert len(gw.calls) == 2
    system, user = gw.calls[1][0]["content"], gw.calls[1][-1]["content"]
    assert "增量补充" in system and "不要改写描述" in system
    assert "【已定稿的资产库" in user and "陆离" in user and "【新增的剧本】" in user

    v2 = store.get(r.asset_ref)
    assert v2.gen_params["mode"] == "incremental" and v2.gen_params["base"] == s["v1"]
    assert v2.gen_params["covers"] == [1, 2, 3] and v2.parent_ids[0] == s["v1"]
    old, _ = parse_assets(store.content(s["v1"]))
    new, _ = parse_assets(store.content(r.asset_ref))
    assert _names(new) == {
        "角色": ["陆离", "小满", "老鬼"],
        "服装": ["陆离-深色风衣-[1-2]", "陆离-白大褂-[3]", "小满-旧连帽衫-[全集]",
                 "老鬼-劳保服-[3]"],
        "场景": ["地铁车厢", "修表铺"],
        "道具": ["暗色骨簪"],
    }
    before = {c.name: c for c in old.characters}
    for c in new.characters:
        if c.name in before:
            assert c.body == before[c.name].body, f"{c.name} 的形象描述被改写了"
    coat = next(x for c in new.characters for x in c.costumes if x.name == "陆离-深色风衣-[1-2]")
    assert coat.body == "深色风衣", "模型重复输出的旧服装丢掉，不覆盖定稿的描述"
    assert next(x for x in new.scenes if x.name == "地铁车厢").desc == "末班车"
    assert "已经定稿" in r.content and "第 3 集" in r.content and "老鬼" in r.content
    assert "只生成新增的 4 张" in r.content


async def test_定稿库已经覆盖的集_再调不重做不花钱():
    store = AssetStore()
    gw = _Seq(_j(BASE))
    s = await _finalized(store, gw)
    r = await s["fns"]._fn_drama_assets(script_ids=s["ids"][:2], ethnicity="asian",
                                        language="zh")
    assert r.ok and r.asset_ref == s["v1"]
    assert r.meta.get("charged") is False
    assert len(gw.calls) == 1, "这几集都在定稿的库里：不调模型"
    assert "不用重做" in r.content and "rebuild=true" in r.content
    assert len(store.find(creator="tool:drama_assets")) == 1


async def test_没渲过参考图的库不算定稿_照常整份生成():
    store = AssetStore()
    ids = _scripts(store, (1, 2))
    gw = _Seq(_j(BASE))
    fns = DramaFunctions(gw, store, registry=None, catalog=None, fmt=EpisodeFormat())
    r1 = await fns._fn_drama_assets(script_ids=ids, ethnicity="asian", language="zh")
    r2 = await fns._fn_drama_assets(script_ids=ids, ethnicity="asian", language="zh")
    assert r1.ok and r2.ok and r1.asset_ref != r2.asset_ref
    assert store.get(r2.asset_ref).gen_params["mode"] == "full"
    assert len(gw.calls) == 2


# ---------------------------------------------------------------- 参考图沿用


async def test_增量版渲参考图_已有的图沿用_只生成新增的():
    store = AssetStore()
    gw = _Seq(_j(BASE), _j(ADD))
    s = await _finalized(store, gw)
    r = await s["fns"]._fn_drama_assets(script_ids=s["ids"], ethnicity="asian", language="zh")
    assert r.ok, r.error
    reg, pack1 = s["reg"], _pack(store, s["pack1"])
    n0 = len(reg.images())

    p = await s["painter"]._fn_drama_render_assets(r.asset_ref)
    assert p.ok, p.error
    made = sorted(a["summary"] for a in reg.images()[n0:])
    assert made == ["场景·修表铺", "服装·老鬼-劳保服-[3]", "服装·陆离-白大褂-[3]", "角色·老鬼"]
    pack2 = _pack(store, p.asset_ref)
    for name, entry in pack1.items():
        assert pack2[name]["asset"] == entry["asset"], f"{name} 没变，沿用旧图"
    assert {"老鬼", "老鬼-劳保服-[3]", "陆离-白大褂-[3]", "修表铺"} <= set(pack2)
    assert pack2["陆离-白大褂-[3]"]["portrait"] == pack1["陆离"]["asset"], "新服装按沿用的脸生成"
    assert pack_library(store, p.asset_ref) == r.asset_ref, "新包归增量版的库"
    assert "复用 6" in p.content and "本次新生成 4" in p.content


async def test_整份重做要当场问人_描述变了的图才重生成():
    store = AssetStore()
    gw = _Seq(_j(BASE), _j(ADD))
    s = await _finalized(store, gw)
    fns = s["fns"]
    assert fns.permission_for("drama_assets", {"rebuild": False}) is None
    level, why = fns.permission_for("drama_assets", {"rebuild": True})
    assert level is PermissionLevel.EXTERNAL and "整份重做" in why
    tools = ToolRegistry(EventBus())
    tools.register(fns)
    await tools.refresh()
    meta = tools.meta_for_call("drama_assets", {"ethnicity": "asian", "language": "zh",
                                                "rebuild": True})
    assert meta is not None and meta.permission is PermissionLevel.EXTERNAL
    meta = tools.meta_for_call("drama_assets", {"ethnicity": "asian", "language": "zh"})
    assert meta is not None and meta.permission is not PermissionLevel.EXTERNAL

    v2 = await fns._fn_drama_assets(script_ids=s["ids"], ethnicity="asian", language="zh")
    assert v2.ok, v2.error
    await s["painter"]._fn_drama_render_assets(v2.asset_ref)
    # 人同意重做：模型把整份重写了一遍，只有小满的形象描述变了
    redo = json.loads(store.content(v2.asset_ref))
    for c in redo["characters"]:
        if c["baseRoleName"] == "小满":
            c["roleTotalDesc"] = "女 | 30岁 | 记者，短发"
    gw.texts.append(_j(redo))
    v3 = await fns._fn_drama_assets(script_ids=s["ids"], ethnicity="asian", language="zh",
                                    rebuild=True)
    assert v3.ok, v3.error
    gp = store.get(v3.asset_ref).gen_params
    assert gp["mode"] == "rebuild" and gp["base"] == v2.asset_ref

    reg = s["reg"]
    n0 = len(reg.images())
    p = await s["painter"]._fn_drama_render_assets(v3.asset_ref)
    assert p.ok, p.error
    made = sorted(a["summary"] for a in reg.images()[n0:])
    assert made == ["服装·小满-旧连帽衫-[全集]", "角色·小满"], (
        "只重生成描述变了的脸和按它生成的服装"
    )
    assert "1 个条目的描述和上一版不同，旧图没沿用（小满）" in p.content



async def test_重做完还没渲图又整份生成_仍沿用渲过的旧图():
    """整份生成也记上一版（base）。之前只有 rebuild 记：重做完还没渲参考图、又整份生成一次，
    base 断了，渲图时一张旧图都沿用不上（2026-09-27）。"""
    store = AssetStore()
    gw = _Seq(_j(BASE), _j(ADD))
    s = await _finalized(store, gw)
    fns = s["fns"]
    v2 = await fns._fn_drama_assets(script_ids=s["ids"], ethnicity="asian", language="zh")
    assert v2.ok, v2.error
    await s["painter"]._fn_drama_render_assets(v2.asset_ref)
    same = json.loads(store.content(v2.asset_ref))
    gw.texts.append(_j(same))
    v3 = await fns._fn_drama_assets(script_ids=s["ids"], ethnicity="asian", language="zh",
                                    rebuild=True)
    assert v3.ok, v3.error
    gw.texts.append(_j(same))
    v4 = await fns._fn_drama_assets(script_ids=s["ids"], ethnicity="asian", language="zh")
    assert v4.ok, v4.error
    gp = store.get(v4.asset_ref).gen_params
    assert gp["mode"] == "full" and gp["base"] == v3.asset_ref
    assert not same_library(store, v4.asset_ref, v3.asset_ref), "整份生成仍不算同一套库"

    reg = s["reg"]
    n0 = len(reg.images())
    p = await s["painter"]._fn_drama_render_assets(v4.asset_ref)
    assert p.ok, p.error
    assert reg.images()[n0:] == [], "名字和描述都没变：沿 base 找到渲过的包，一张都不重生成"

# ---------------------------------------------------------------- 同一套库


def _lib_asset(store: AssetStore, mode: str = "full", base: str = "") -> str:
    gp: dict[str, Any] = {"mode": mode}
    if base:
        gp["base"] = base
    return store.create(_j({"characters": [{"name": "安妮", "prompt": "x"}]}), summary="资产库",
                        creator="tool:drama_assets", gen_params=gp,
                        parents=[base] if base else None).id


def test_增量链算同一套库_整份重做不算():
    store = AssetStore()
    v1 = _lib_asset(store)
    v2 = _lib_asset(store, "incremental", v1)
    v3 = _lib_asset(store, "incremental", v2)
    v4 = _lib_asset(store, "rebuild", v3)
    assert library_chain(store, v3) == [v3, v2, v1]
    assert library_chain(store, v4) == [v4], "整份重做：链到这里断"
    assert same_library(store, v1, v3) and same_library(store, v3, v1)
    assert not same_library(store, v3, v4) and not same_library(store, v1, v4)
    assert not same_library(store, "", v1)


def test_流水线_增量补过的资产库不让旧提示词过期_整份重做的才过期(tmp_path: Path):
    store = AssetStore()
    sb = store.create(
        _j([{"episodeIndex": 1, "episodeTitle": "第1集", "episodeDesc": "[第1集-1场] 镜头"}]),
        type_=AssetType.STORYBOARD, summary="分镜", creator="tool:drama_storyboard",
    )
    v1 = _lib_asset(store)
    shots = store.create("[]", summary="提示词·第1集", creator="tool:drama_shots",
                         gen_params={"episode": 1}, parents=[sb.id, v1])
    pipe = EpisodePipeline(FakeRegistry(store), store, EventBus(), OutputPrefs(tmp_path))
    assert pipe._scan()["outdated_shots"] == {}  # noqa: SLF001

    v2 = _lib_asset(store, "incremental", v1)
    view = pipe._scan()  # noqa: SLF001
    assert view["assets_lib"] == v2
    assert view["outdated_shots"] == {}, "定稿后只补了新集：已有条目没动，旧提示词照常能渲"

    _lib_asset(store, "rebuild", v2)
    assert pipe._scan()["outdated_shots"] == {shots.id: "资产库"}  # noqa: SLF001


async def test_提示词和参考图包在同一条增量链上_照常渲_整份重做的包拦下():
    store = AssetStore()
    fns, reg, _ = _gate_fns(store, {})
    img = store.create("", type_=AssetType.IMAGE, summary="角色·安妮", creator="model:x")
    img.uri = "https://img/anne.png"
    store.put(img)
    entry = _j({"安妮": {"asset": img.id, "url": img.uri, "kind": "角色"}})
    v1 = _lib_asset(store)
    pack1 = store.create(entry, summary="参考图包", creator="tool:drama_render_assets",
                         parents=[v1])
    v2 = _lib_asset(store, "incremental", v1)
    pack2 = store.create(entry, summary="参考图包·增量", creator="tool:drama_render_assets",
                         parents=[v2, pack1.id])
    shots = store.create(_j(SHOTS), summary="提示词", parents=["as_sb", v1])
    r = await fns._fn_drama_render_shots(shots.id, rendered_id=pack2.id)
    assert r.ok, r.error
    assert len(reg.calls) == 2

    v3 = _lib_asset(store, "rebuild", v2)
    pack3 = store.create(entry, summary="参考图包·重做", creator="tool:drama_render_assets",
                         parents=[v3])
    reg.calls.clear()
    r2 = await fns._fn_drama_render_shots(shots.id, rendered_id=pack3.id)
    assert not r2.ok and "属于另一套资产库" in (r2.error or "")
    assert r2.meta.get("charged") is False and not reg.calls


# ---------------------------------------------------------------- 服装图不像主形象


async def test_服装图和主形象不像_不进参考图包_渲视频退回主形象_补生成后进包():
    store = AssetStore()
    lib_id = _stage_lib(store)
    reg = ImageRegistry(store)
    catalog = SimpleNamespace(
        drama={"identity_gate": "true", "identity_retries": "1", "face_audit": "false"},
        max_concurrency=lambda k: 0,
    )
    fns = DramaFunctions(SimpleNamespace(chat=None), store, registry=reg, catalog=catalog)
    unlike = {"小满"}

    async def ident(asset_id: str, refs: Any, is_video: bool = False, pass_score: int = 7):
        who = store.get(asset_id).summary.removeprefix("服装·").split("-", 1)[0]
        ok = who not in unlike
        return IdentityVerdict(passed=ok, score=9 if ok else 3,
                               issues=[] if ok else ["脸型不对"])

    fns._check_identity = ident  # type: ignore[method-assign]
    r = await fns._fn_drama_render_assets(lib_id)
    assert r.ok, r.error
    pack = _pack(store, r.asset_ref)
    assert "小满" in pack and "小满-旧连帽衫-[1-3]" not in pack, "不像的服装图不进包"
    assert "陆离-深色风衣-[1]" in pack
    assert len(reg.images("服装·小满")) == 2, "不像先按差异重生成一次，还是不像才不要"
    assert "和主形象不像（3/10），没进参考图包" in r.content
    ref = _resolve_ref("小满-旧连帽衫-[1-3]", pack)
    assert ref is not None and ref.key == "小满" and ref.url == pack["小满"]["url"], (
        "渲视频时这套服装退回主形象（脸）"
    )

    unlike.clear()
    r2 = await fns._fn_drama_render_assets(lib_id, only="costumes")
    assert r2.ok, r2.error
    pack2 = _pack(store, r2.asset_ref)
    assert "小满-旧连帽衫-[1-3]" in pack2
    assert pack2["小满-旧连帽衫-[1-3]"]["portrait"] == pack["小满"]["asset"]
    assert len(reg.images("服装·小满")) == 3, "只补上次没进包的那套"
    assert len(reg.images("服装·陆离")) == 1, "进了包的服装不重生成"

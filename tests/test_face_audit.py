"""角色面容审查（2026-09-22 用户提的问题）：同一个角色只留一张脸。

用户的现场：主角「阿蛛」的主形象被一家模型的内容护栏拒了，换三家模型各出一张，
产物目录里躺着三张脸。谁当准，后面 12 集全跟着走 —— 不先裁掉，渲视频必然打架。

用户定的两条：**Agent 自动挑、歧义才问**；**渲完参考图后自动跑**。
清理一律可逆：文件移进 workspace/trash，资产只打 superseded_by 记号，从不真删。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.drama.faces import (
    FaceCandidate,
    audit_lines,
    conflict_question,
    decide,
    matches_character,
    pick_baseline,
)
from aigc_agent.domain.functions.drama import FACE_CONFLICT_STAGE, DramaFunctions
from aigc_agent.harness.tools.provider import ToolResult


def _exists(path: str) -> bool:
    """os.path.exists 而不是 Path.exists —— 后者在 async 用例里会被 lint 判成阻塞调用。"""
    return os.path.exists(path)

LIB = {
    "characters": [
        {"baseRoleName": "阿蛛", "roleTotalDesc": "女 | 8岁 [强制白底证件照]",
         "roleCostumeList": [{"costumeName": "阿蛛-童装-[全集]", "costumeDesc": "旧卫衣"}]},
        {"baseRoleName": "朱锦", "roleTotalDesc": "女 | 33岁 [强制白底证件照]",
         "roleCostumeList": []},
    ],
    "scenes": [{"name": "老街", "description": "x"}],
    "props": [],
}


# ---------------------------------------------------------------- 纯逻辑


def _c(aid: str, source: str = "asset", seq: int = 0, who: str = "阿蛛") -> FaceCandidate:
    return FaceCandidate(character=who, asset_id=aid, source=source, seq=seq, local=f"/x/{aid}.png")


def test_主形象比服装图有资格当基准():
    """服装图是照着主形象生的，哪怕更新也不能反过来当准绳。"""
    portrait = FaceCandidate(character="阿蛛", asset_id="p", source="pack", kind="角色", seq=1)
    costume = FaceCandidate(character="阿蛛", asset_id="c", source="pack", kind="服装", seq=99)
    base, _ = pick_baseline([costume, portrait])
    assert base.asset_id == "p"


def test_基准优先级_用户钉的最大():
    cands = [_c("a3", "asset", seq=30), _c("a1", "pack", seq=10), _c("a2", "user", seq=1)]
    base, why = pick_baseline(cands)
    assert base.asset_id == "a2" and "钉过" in why
    base, why = pick_baseline([_c("a3", "asset", seq=30), _c("a1", "pack", seq=10)])
    assert base.asset_id == "a1" and "参考图包" in why
    base, why = pick_baseline([_c("a3", "asset", seq=30), _c("a4", "asset", seq=99)])
    assert base.asset_id == "a4", "同为散落产物时取最新的"
    assert pick_baseline([])[0] is None


def test_基准有权威时不问人_直接裁():
    """用户钉过或参考图包在用 —— 他已经表过态，剩下的一律算漂移。"""
    base = _c("keep", "user", seq=1)
    others = [_c("x1", seq=2), _c("x2", seq=3)]
    a = decide("阿蛛", [base, *others], {"x1": (False, 3, ["脸换了"]), "x2": (False, 4, [])},
               base, "你钉过的")
    assert not a.ambiguous and [c.asset_id for c in a.drift] == ["x1", "x2"]
    assert a.issues["x1"] == ["脸换了"] and a.scores["x2"] == 4


def test_没人表过态且各自成群_挂起问人():
    base = _c("x1", "asset", seq=3)
    cands = [base, _c("x2", seq=2), _c("x3", seq=1)]
    a = decide("阿蛛", cands, {"x2": (False, 3, []), "x3": (False, 2, [])}, base, "最新的")
    assert a.ambiguous and "定不了谁是准的" in a.why


def test_散落产物但占多数_多数派说了算():
    base = _c("x1", "asset", seq=3)
    cands = [base, _c("x2", seq=2), _c("x3", seq=1)]
    a = decide("阿蛛", cands, {"x2": (True, 9, []), "x3": (False, 2, [])}, base, "最新的")
    assert not a.ambiguous and len(a.same) == 1 and len(a.drift) == 1


def test_全都一致时没有冲突():
    base = _c("x1", "asset", seq=3)
    a = decide("阿蛛", [base, _c("x2")], {"x2": (True, 9, [])}, base, "最新的")
    assert not a.conflicted and not a.ambiguous


def test_更长的同名角色不会被算进来():
    names = ["朱锦", "朱锦娘"]
    assert matches_character("角色·朱锦", "朱锦", names)
    assert not matches_character("角色·朱锦娘", "朱锦", names), "朱锦娘不该算成朱锦"
    assert matches_character("角色·朱锦娘", "朱锦娘", names)
    assert not matches_character("角色·陆离", "朱锦", names)


def test_报告里冲突展开_无冲突一行带过():
    base = _c("keep", "user")
    a = decide("阿蛛", [base, _c("bad")], {"bad": (False, 3, ["下颌变了"])}, base, "你钉过的")
    b = decide("朱锦", [_c("ok", who="朱锦")], {}, _c("ok", who="朱锦"), "最新的")
    text = "\n".join(audit_lines([a, b]))
    assert "✂ 阿蛛" in text and "弃用：bad" in text and "下颌变了" in text
    assert "✓ 朱锦" in text
    q = conflict_question([decide("阿蛛", [_c("x1", seq=2), _c("x2", seq=1)],
                                  {"x2": (False, 2, [])}, _c("x1", seq=2), "最新的")])
    assert "阿蛛" in q and "资产 id" in q


# ---------------------------------------------------------------- 工具


class _Reg:
    async def invoke(self, name: str, args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content="ok")


def _img(store: AssetStore, summary: str, tmp: Path, name: str) -> str:
    p = tmp / "out" / f"{name}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    a = store.create("", type_=AssetType.IMAGE, summary=summary, creator="model:x")
    a.uri = f"https://img/{name}.png"
    a.gen_params["local"] = str(p)
    store.put(a)
    return a.id


def _fns(store: AssetStore, tmp: Path, verdicts: dict[str, dict[str, Any]]) -> DramaFunctions:
    """verdicts: 被比对的资产 id → 视觉模型的判定 JSON。"""
    order: list[str] = []

    async def chat(role: str, messages: list[dict[str, Any]], **_: Any) -> Any:
        return SimpleNamespace(text=json.dumps(verdicts.get(order.pop(0), {"score": 9})))

    fns = DramaFunctions(SimpleNamespace(chat=chat), store, registry=_Reg(), catalog=None)
    fns.catalog = SimpleNamespace(drama={"identity_pass_score": "7"}, max_concurrency=lambda k: 0)
    fns.files = SimpleNamespace(trash=tmp / "trash")

    real = fns._check_identity

    async def check(target: str, refs: Any, is_video: bool, pass_score: int) -> Any:
        order.append(target)
        return await real(target, refs, is_video, pass_score)

    fns._check_identity = check  # type: ignore[method-assign]
    fns._image_payload = lambda aid, url: f"data:{aid}"  # type: ignore[method-assign]
    return fns


async def test_保留基准_其余移进回收目录并标记(tmp_path: Path):
    store = AssetStore(tmp_path / "assets")
    lib = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库",
                       creator="tool:drama_assets")
    keep = _img(store, "角色·阿蛛", tmp_path, "keep")
    bad1 = _img(store, "阿蛛·模型试探·qwen", tmp_path, "bad1")
    bad2 = _img(store, "阿蛛·模型试探·flux", tmp_path, "bad2")
    pack = store.create(
        json.dumps({"阿蛛": {"asset": keep, "url": "https://img/keep.png", "kind": "角色"}}),
        type_=AssetType.STORYBOARD, summary="资产参考图·1张",
        parents=[lib.id], creator="tool:drama_render_assets")
    assert pack

    fns = _fns(store, tmp_path, {bad1: {"score": 3, "issues": ["脸换了"]},
                                 bad2: {"score": 2, "issues": ["不是同一人"]}})
    r = await fns.invoke("drama_audit_faces", {"assets_id": lib.id})
    assert r.ok and not r.suspend, r.error
    assert "✂ 阿蛛" in r.content and "3 张里有 2 张是另一张脸" in r.content
    assert "脸换了" in r.content

    # 弃用的：文件移进 trash、资产标 superseded_by、本地路径跟着改到新位置
    for aid in (bad1, bad2):
        a = store.get(aid)
        assert a.gen_params["superseded_by"] == keep
        moved = str(a.gen_params["local"])
        assert _exists(moved) and "trash" in moved and "阿蛛" in moved
    assert _exists(store.get(keep).gen_params["local"]), "保留的那张不能动"


async def test_歧义时挂起问人_一个文件都不动(tmp_path: Path):
    store = AssetStore(tmp_path / "assets")
    lib = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库",
                       creator="tool:drama_assets")
    a1 = _img(store, "阿蛛 a", tmp_path, "a1")
    a2 = _img(store, "阿蛛 b", tmp_path, "a2")
    a3 = _img(store, "阿蛛 c", tmp_path, "a3")
    fns = _fns(store, tmp_path, {a1: {"score": 2}, a2: {"score": 2}, a3: {"score": 2}})
    r = await fns.invoke("drama_audit_faces", {"assets_id": lib.id})
    assert r.suspend and r.suspend_payload["stage"] == FACE_CONFLICT_STAGE
    assert r.suspend_payload["major"] is True
    assert "定不了保留哪张" in r.suspend_payload["question"]
    for aid in (a1, a2, a3):
        assert "superseded_by" not in store.get(aid).gen_params
        assert _exists(store.get(aid).gen_params["local"])


async def test_只出报告不动文件(tmp_path: Path):
    store = AssetStore(tmp_path / "assets")
    lib = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库",
                       creator="tool:drama_assets")
    keep = _img(store, "角色·阿蛛", tmp_path, "keep")
    bad = _img(store, "阿蛛·另一版", tmp_path, "bad")
    store.create(json.dumps({"阿蛛": {"asset": keep, "url": "u", "kind": "角色"}}),
                 type_=AssetType.STORYBOARD, summary="包", parents=[lib.id],
                 creator="tool:drama_render_assets")
    fns = _fns(store, tmp_path, {bad: {"score": 3}})
    r = await fns.invoke("drama_audit_faces", {"assets_id": lib.id, "apply": False})
    assert r.ok and "只出报告没动文件" in r.content
    assert "superseded_by" not in store.get(bad).gen_params


async def test_参考图包指向被弃用的就换成保留的(tmp_path: Path):
    """包里在用的那张被判成另一张脸时，要换掉 —— 不然渲视频还是引用到错的。"""
    store = AssetStore(tmp_path / "assets")
    lib = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库",
                       creator="tool:drama_assets")
    pinned = _img(store, "角色·阿蛛·用户素材", tmp_path, "pinned")
    inpack = _img(store, "角色·阿蛛", tmp_path, "inpack")
    store.create(
        json.dumps({
            "阿蛛": {"asset": pinned, "url": "https://img/pinned.png", "kind": "角色",
                     "source": "user"},
            "阿蛛-童装-[全集]": {"asset": inpack, "url": "https://img/inpack.png", "kind": "服装"},
        }),
        type_=AssetType.STORYBOARD, summary="包", parents=[lib.id],
        creator="tool:drama_render_assets")
    fns = _fns(store, tmp_path, {inpack: {"score": 2, "issues": ["不是同一张脸"]}})
    r = await fns.invoke("drama_audit_faces", {"assets_id": lib.id})
    assert r.ok and r.asset_ref, r.error
    newpack = json.loads(store.content(r.asset_ref))
    assert newpack["阿蛛-童装-[全集]"]["asset"] == pinned, "冲突项换成了保留的那张"
    assert newpack["阿蛛"]["asset"] == pinned


async def test_渲完参考图自动查_可以关掉(tmp_path: Path):
    store = AssetStore(tmp_path / "assets")
    lib = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库",
                       creator="tool:drama_assets")
    fns = _fns(store, tmp_path, {})
    fns.catalog.drama["face_audit"] = "false"
    assert fns._face_cfg()[0] is False
    fns.catalog.drama.pop("face_audit")
    assert fns._face_cfg()[0] is True, "默认开着"
    assert lib


async def test_默认不查服装图_deep才查(tmp_path: Path):
    """服装图生成时已经过了一道一致性门，默认再查一遍就是重复花视觉模型的钱。"""
    store = AssetStore(tmp_path / "assets")
    lib = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库",
                       creator="tool:drama_assets")
    keep = _img(store, "角色·阿蛛", tmp_path, "keep")
    cos = _img(store, "服装·阿蛛-童装-[全集]", tmp_path, "cos")
    store.create(
        json.dumps({
            "阿蛛": {"asset": keep, "url": "u1", "kind": "角色"},
            "阿蛛-童装-[全集]": {"asset": cos, "url": "u2", "kind": "服装"},
        }),
        type_=AssetType.STORYBOARD, summary="包", parents=[lib.id],
        creator="tool:drama_render_assets")

    fns = _fns(store, tmp_path, {cos: {"score": 2, "issues": ["脸变了"]}})
    r = await fns.invoke("drama_audit_faces", {"assets_id": lib.id})
    assert r.ok and "✓ 阿蛛" in r.content, "默认只看主形象，一张候选不发视觉调用"
    assert "superseded_by" not in store.get(cos).gen_params

    fns2 = _fns(store, tmp_path, {cos: {"score": 2, "issues": ["脸变了"]}})
    r2 = await fns2.invoke("drama_audit_faces", {"assets_id": lib.id, "deep": True})
    assert r2.ok and "✂ 阿蛛" in r2.content, "deep 才把服装图拉进来复核"
    assert store.get(cos).gen_params["superseded_by"] == keep

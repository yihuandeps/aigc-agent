"""缺口 D（2026-09-23 审查）：质检门「做不了就放行、到上限标记放行」。

审查结论：
  · 三道门串着各跑各的，后面的门重生成之后不回查前面的门 —— 为了消字幕重生成的那版
    可能变了脸，照样放行
  · 每道门拿原始提示词重来，前一道门的修正丢了
  · 字幕查不成（没本地副本 / 抽不出帧 / 判读不出结果）只记一句就放行
  · 到上限只在结果里提一句，照样拼进成片；被换掉的废片还占着主文件名
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.drama.format import EpisodeFormat
from aigc_agent.domain.drama.identity import IdentityVerdict
from aigc_agent.domain.functions.drama import DramaFunctions
from aigc_agent.domain.media.fast_cut import FAST_CUT_RETRY_MARK
from aigc_agent.domain.media.no_text import NO_TEXT_RETRY
from aigc_agent.harness.tools.provider import ToolResult

SHOTS = [
    {"scene_index": "[第1集-1场]", "video_name": "1-4", "video_duration": "12s",
     "cuts": [3, 3, 3, 3], "description": "开场 (安妮)"},
    {"scene_index": "[第1集-2场]", "video_name": "5-8", "video_duration": "12s",
     "cuts": [3, 3, 3, 3], "description": "收尾 (安妮)"},
]


class Registry:
    def __init__(self, store: AssetStore) -> None:
        self.store = store
        self.calls: list[dict] = []
        self.composed: list[list[str]] = []

    async def invoke(self, name: str, args: dict) -> ToolResult:
        if name == "compose_video":
            self.composed.append(list(args["clips"]))
            a = self.store.create("mp4", summary="成片", creator="fake")
            return ToolResult(ok=True, content="拼好了", asset_ref=a.id)
        self.calls.append(dict(args))
        a = self.store.create("", type_=AssetType.VIDEO, summary=args["summary"],
                              creator="model:x", gen_params={"tags": dict(args.get("tags") or {})})
        a.uri = f"https://fake/{a.id}.mp4"
        self.store.put(a)
        return ToolResult(ok=True, content="ok", asset_ref=a.id)


def _fns(store: AssetStore, script: dict[str, list[dict[str, Any]]], **cfg: str):
    """script：summary → 每一版的检查结果 {"sub": bool, "cut": float, "id": int}（按版本顺序）。"""
    reg = Registry(store)
    fns = DramaFunctions(SimpleNamespace(chat=None), store, registry=reg, catalog=None,
                         fmt=EpisodeFormat())
    fns.catalog = SimpleNamespace(
        drama={"subtitle_gate": "true", "cut_gate": "true", "identity_gate": "true",
               "identity_pass_score": "7", **cfg},
        max_concurrency=lambda k: 0,
    )
    version: dict[str, int] = {}
    checked: list[tuple[str, str]] = []

    def now(asset_id: str) -> dict[str, Any]:
        s = store.get(asset_id).summary
        seq = script.get(s) or [{}]
        # 同一资产被三道门各问一次：按资产认版本
        key = f"{s}#{asset_id}"
        if key not in version:
            version[key] = sum(1 for k in version if k.startswith(s + "#"))
        return seq[min(version[key], len(seq) - 1)]

    async def cuts(asset_id: str, thr: float):
        checked.append((store.get(asset_id).summary, "cut"))
        return float(now(asset_id).get("cut", 2.5)), ""

    async def subs(asset_id: str):
        checked.append((store.get(asset_id).summary, "sub"))
        v = now(asset_id).get("sub", False)
        return (False, v) if isinstance(v, str) else (bool(v), "")

    async def ident(asset_id: str, refs, is_video: bool, pass_score: int):
        checked.append((store.get(asset_id).summary, "id"))
        score = int(now(asset_id).get("id", 9))
        return IdentityVerdict(passed=score >= pass_score, score=score,
                               issues=[] if score >= pass_score else ["换了个人"])

    fns._check_cuts = cuts  # type: ignore[method-assign]
    fns._check_subtitles = subs  # type: ignore[method-assign]
    fns._check_identity = ident  # type: ignore[method-assign]
    return fns, reg, checked


def _seed(store: AssetStore) -> tuple[str, str]:
    """一套分镜提示词 + 参考图包（引用 (安妮) 带一张参考图，才会跑一致性门）。"""
    img = store.create("", type_=AssetType.IMAGE, summary="角色·安妮", creator="model:x")
    img.uri = "https://img/anne.png"
    store.put(img)
    lib = store.create(json.dumps({"characters": [{"name": "安妮", "prompt": "x"}]}),
                       summary="资产库", creator="tool:drama_assets")
    pack = store.create(json.dumps({"安妮": {"asset": img.id, "url": img.uri, "kind": "角色"}}),
                        summary="参考图包", creator="tool:drama_render_assets",
                        parents=[lib.id])
    shots = store.create(json.dumps(SHOTS, ensure_ascii=False), summary="提示词",
                         parents=["as_sb", lib.id])
    return shots.id, pack.id


async def _render(fns: DramaFunctions, store: AssetStore, **kw: Any) -> ToolResult:
    shots_id, pack_id = _seed(store)
    return await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, **kw)


async def test_每一版都全量重查_修正合并进同一次重生成():
    store = AssetStore()
    # 第 1 版：有字幕 + 镜头太长；第 2 版：字幕和镜头好了但变脸；第 3 版：全过
    fns, reg, checked = _fns(store, {"[第1集-1场] 1-4": [
        {"sub": True, "cut": 5.0}, {"id": 3}, {}]}, gate_max_regen="3", identity_retries="2")
    r = await _render(fns, store)
    assert r.ok, r.error
    gens = [c for c in reg.calls if c["summary"] == "[第1集-1场] 1-4"]
    assert len(gens) == 3
    # 第 2 版一次重生成里同时带上字幕与快切两道修正
    assert NO_TEXT_RETRY in gens[1]["prompt"] and FAST_CUT_RETRY_MARK in gens[1]["prompt"]
    # 第 3 版在前面的修正之上再加一致性修正（前一道门的修正不丢）
    assert NO_TEXT_RETRY in gens[2]["prompt"] and "换了个人" in gens[2]["prompt"]
    # 第 2 版三道门都查了一遍（之前后面的门重生成后不回查前面的门）
    v2 = [g for s, g in checked if s == "[第1集-1场] 1-4"][3:6]
    assert sorted(v2) == ["cut", "id", "sub"]
    assert r.meta["complete"] is True and reg.composed, "全过才拼成片"


async def test_到上限还有字幕_不进成片_accept才放行():
    store = AssetStore()
    fns, reg, _ = _fns(store, {"[第1集-2场] 5-8": [{"sub": True}, {"sub": True}]})
    shots_id, pack_id = _seed(store)
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.ok and not reg.composed, "带字幕的段不拼成片"
    assert "⛔ 仍有字幕/文字" in r.content and "未成片：1 段没过质检门" in r.content
    assert r.meta == {"complete": False, "failed": 0, "blocked": 1}

    # 不点名放行就重跑：⛔ 的段不复用，只重渲它（第 1 段过了门，复用）
    reg.calls.clear()
    r1 = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert [c["summary"] for c in reg.calls] == ["[第1集-2场] 5-8"] * 2 and not reg.composed
    assert r1.meta["blocked"] == 1

    # 用户看过说可以 → accept 放行：不重生成，直接复用那一版并拼成片
    reg.calls.clear()
    r2 = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, accept=["5-8"])
    assert r2.ok and reg.composed and not reg.calls
    assert r2.meta["complete"] is True


async def test_镜头超时默认只标记_照样成片():
    store = AssetStore()
    fns, reg, _ = _fns(store, {"[第1集-1场] 1-4": [{"cut": 4.0}, {"cut": 3.9}]})
    r = await _render(fns, store)
    assert r.ok and reg.composed and "⚠ 仍有超过 3 秒的镜头" in r.content
    assert "⛔" not in r.content


async def test_被换掉的版本挪进废弃_主文件名留给采用的那版(tmp_path: Path):
    store = AssetStore()
    fns = DramaFunctions(None, store, registry=None, catalog=None)
    videos = tmp_path / "videos"
    await asyncio.to_thread(videos.mkdir)

    def clip(name: str) -> str:
        p = videos / name
        p.write_bytes(b"mp4")
        a = store.create("", type_=AssetType.VIDEO, summary=name, creator="model:x",
                         gen_params={"local": str(p)})
        return a.id

    first = await asyncio.to_thread(clip, "第01集-01_1场_镜1-4.mp4")
    second = await asyncio.to_thread(clip, "第01集-01_1场_镜1-4-v2.mp4")
    fns._retire_version(first, "subtitle")
    fns._reclaim_name(second, first)
    old = store.get(first)
    assert "废弃" in old.gen_params["local"] and old.gen_params["tags"]["rejected"] == "subtitle"
    assert store.get(second).gen_params["local"].endswith("第01集-01_1场_镜1-4.mp4")
    assert (videos / "第01集-01_1场_镜1-4.mp4").exists()
    assert not (videos / "第01集-01_1场_镜1-4-v2.mp4").exists()


async def test_流水线不把没渲完整的集当成完成(tmp_path: Path):
    from aigc_agent.domain.output import OutputPrefs
    from aigc_agent.domain.pipeline.episode_pipeline import EpisodePipeline
    from aigc_agent.harness.events.bus import EventBus

    store = AssetStore()
    store.create("[]", summary="分镜视频·第1集", creator="tool:drama_render_shots",
                 gen_params={"episode": 1, "complete": False})
    store.create("[]", summary="分镜视频·第2集", creator="tool:drama_render_shots",
                 gen_params={"episode": 2, "complete": True})
    pipe = EpisodePipeline(registry=None, assets=store, bus=EventBus(),
                           output_prefs=OutputPrefs(tmp_path))
    assert set(pipe._scan()["rendered"]) == {2}

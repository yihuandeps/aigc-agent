"""渲染复用与质检的第二轮（2026-09-26 用户定的第 5、11 条 + 纯 bug J1 / J4 / J5）。

- 按段指纹复用：提示词重出了、内容和参考图没变的段直接复用（之前换了提示词 id 就整集重付）
- 前序段重渲了，承接它尾帧的段跟着重渲；报价里说清「以前过了质检、这次要重付」的段
- 字幕「查不成」判 ⛔ 的段：重跑先补下载、重查，没字就复用不重付
- 字幕连续查不成：熔断，后面的段先不渲
- 没过质检的段不当前序参考：承接它的段先不渲（之前错脸 / 带字的尾帧往后传）
- 质检重生成时超时的任务：重跑按段指纹从任务台账取回（提示词改过，请求指纹对不上）
- redo 换掉的旧版本比新版好：accept 按片段 id 要回来
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aigc_agent.domain.assets.store import AssetStatus, AssetStore, AssetType
from aigc_agent.harness.model.budget import CostGuard
from aigc_agent.harness.model.task_ledger import MediaTaskLedger
from aigc_agent.harness.tools.provider import ToolResult
from tests.test_gate_chain import SHOTS, _seed
from tests.test_gate_chain import Registry as GateRegistry
from tests.test_gate_chain import _fns as _gate_fns


def _shots(store: AssetStore, rows: list[dict[str, Any]], lib_id: str) -> str:
    a = store.create(json.dumps(rows, ensure_ascii=False), summary="提示词·重出",
                     parents=["as_sb", lib_id])
    return a.id


def _lib_of(store: AssetStore, pack_id: str) -> str:
    return store.get(pack_id).parent_ids[0]


def _gens(reg: Any, name: str) -> int:
    return sum(1 for c in reg.calls if c.get("summary", "").endswith(name))


class _Quotes:
    """会报价的闸门：记下报价单。"""

    def __init__(self) -> None:
        self.quotes: list[str] = []
        self.guard = CostGuard(call_limits={"video": 100})

    async def confirm_batch(self, tool: str, text: str, args: Any) -> bool:
        self.quotes.append(text)
        return True


class _Reg(GateRegistry):
    """test_gate_chain 的假注册表，外加补下载（fetch_asset_file）和取回任务（media_recover）。"""

    def __init__(self, store: AssetStore, tmp: Path, ledger: MediaTaskLedger | None = None):
        super().__init__(store)
        self.tmp = tmp
        self.ledger = ledger
        self.fetched: list[str] = []
        self.recovered: list[str] = []
        self.timeout_once: set[str] = set()  # 这些段的下一次生成「轮询超时」（任务已提交）

    async def invoke(self, name: str, args: dict) -> ToolResult:
        if name == "fetch_asset_file":
            for aid in args["asset_ids"]:
                p = self.tmp / f"{aid}.mp4"
                p.write_bytes(b"mp4")
                a = self.store.get(aid)
                a.gen_params["local"] = str(p)
                self.store.put(a)
                self.fetched.append(aid)
            return ToolResult(ok=True, content="下好了")
        if name == "media_recover":
            self.recovered.append(args["task_id"])
            rec = self.ledger.get(args["task_id"]) if self.ledger else None
            tags = dict(((rec.params if rec else {}) or {}).get("tags") or {})
            a = self.store.create("", type_=AssetType.VIDEO, summary=args["summary"],
                                  creator="model:x", gen_params={"tags": tags})
            a.uri = f"https://fake/{a.id}.mp4"
            self.store.put(a)
            return ToolResult(ok=True, content="取回了", asset_ref=a.id)
        summary = str(args.get("summary") or "")
        if name != "compose_video" and summary in self.timeout_once:
            self.timeout_once.discard(summary)
            self.calls.append(dict(args))
            if self.ledger is not None:
                self.ledger.submitted(
                    f"task-{len(self.calls)}", kind="video", model="m", provider="p",
                    fingerprint=f"req-{len(self.calls)}", prompt=args["prompt"],
                    params={"tags": dict(args.get("tags") or {})},
                )
            return ToolResult(ok=False, error="轮询超时：任务已提交，没等到结果")
        return await super().invoke(name, args)


def _setup(tmp_path: Path, script: dict | None = None, ledger: bool = False, **cfg: str):
    store = AssetStore()
    fns, _, _ = _gate_fns(store, script or {}, **cfg)
    led = MediaTaskLedger(tmp_path / "tasks.jsonl") if ledger else None
    reg = _Reg(store, tmp_path, led)
    fns.registry = reg
    fns.task_ledger = led
    return store, fns, reg


# ---------------------------------------------------------------- 按段指纹复用


async def test_提示词重出_内容没变的段按指纹复用_改了的才重渲(tmp_path):
    store, fns, reg = _setup(tmp_path)
    shots_id, pack_id = _seed(store)
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, episode=1)
    assert r.ok and r.meta["complete"] and len(reg.calls) == 2

    rows = json.loads(json.dumps(SHOTS))
    rows[1]["description"] = "收尾，改了一句 (安妮)"
    new_id = _shots(store, rows, _lib_of(store, pack_id))
    reg.calls.clear()
    r2 = await fns._fn_drama_render_shots(new_id, rendered_id=pack_id, episode=1)
    assert r2.ok and r2.meta["complete"]
    assert [c["summary"] for c in reg.calls] == ["[第1集-2场] 5-8"], (
        "换了提示词 id，第 1 段内容没变照样复用；只重渲改了的那段"
    )


async def test_前序段重渲_承接它的段跟着重渲_报价说出要重付的已通过段(tmp_path):
    store, fns, reg = _setup(tmp_path)
    gate = _Quotes()
    reg.gate = gate
    shots_id, pack_id = _seed(store)
    rows = json.loads(json.dumps(SHOTS))
    rows[1]["description"] = "{第1集-1场} 接上一段，收尾 (安妮)"
    lib = _lib_of(store, pack_id)
    first = _shots(store, rows, lib)
    r = await fns._fn_drama_render_shots(first, rendered_id=pack_id, episode=1)
    assert r.ok and r.meta["complete"] and len(reg.calls) == 2
    assert reg.calls[1].get("video_urls"), "第 2 段带着第 1 段当前序参考"

    rows[0]["description"] = "开场，换了个动作 (安妮)"  # 只改第 1 段
    second = _shots(store, rows, lib)
    reg.calls.clear()
    r2 = await fns._fn_drama_render_shots(second, rendered_id=pack_id, episode=1)
    assert r2.ok and r2.meta["complete"]
    assert sorted(c["summary"] for c in reg.calls) == ["[第1集-1场] 1-4", "[第1集-2场] 5-8"], (
        "第 2 段自己没改，但它承接的第 1 段重渲了：尾帧变了，跟着重渲"
    )
    quote = gate.quotes[-1]
    assert "要新渲 2 段" in quote
    assert "其中 2 段以前渲过、过了质检" in quote and "要重渲" in quote


# ---------------------------------------------------------------- 字幕查不成：补查 + 熔断


async def test_字幕没查成的段_重跑先补下载重查_没字就复用不重付(tmp_path):
    store, fns, reg = _setup(tmp_path, {"[第1集-2场] 5-8": [{"sub": "视觉接口 502"}]})
    shots_id, pack_id = _seed(store)
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, episode=1)
    assert r.ok and r.meta["blocked"] == 1 and not reg.composed
    assert "字幕没查成" in r.content

    async def fine(asset_id: str):  # 接口恢复了：这次查得成，画面没字
        return False, ""

    fns._check_subtitles = fine  # type: ignore[method-assign]
    reg.calls.clear()
    r2 = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, episode=1)
    assert r2.ok and r2.meta["complete"] and reg.composed
    assert not [c for c in reg.calls if c.get("summary")], "查得成、没字：不重新生成"
    assert len(reg.fetched) == 1, "没本地副本的先补下载再查"
    clip = next(a for a in store.find(type_=AssetType.VIDEO)
                if a.summary == "[第1集-2场] 5-8")
    assert clip.gen_params["tags"]["accepted"] is True
    assert any("重查字幕" in n for n in clip.gen_params["tags"]["notes"])


async def test_字幕连续查不成_熔断_后面的段先不渲(tmp_path):
    rows = [
        {"scene_index": f"[第1集-{k}场]", "video_name": f"{k}", "video_duration": "12s",
         "cuts": [3, 3, 3, 3], "description": f"第 {k} 段 (安妮)"}
        for k in range(1, 5)
    ]
    script = {f"[第1集-{k}场] {k}": [{"sub": "视觉接口 502"}] for k in range(1, 5)}
    store, fns, reg = _setup(tmp_path, script, unchecked_break="2")
    fns.catalog.max_concurrency = lambda k: 1  # 一段一段来，熔断才拦得住后面的
    _, pack_id = _seed(store)
    shots_id = _shots(store, rows, _lib_of(store, pack_id))
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, episode=1)
    gens = [c for c in reg.calls if c.get("summary")]
    assert len(gens) == 2, "连续 2 段没查成就停，后 2 段没花钱"
    assert r.ok and r.meta["complete"] is False
    assert "连续 2 段没查成" in r.content


# ---------------------------------------------------------------- J1：⛔ 的段不往后传


async def test_没过质检的段不当前序参考_承接它的段先不渲(tmp_path):
    store, fns, reg = _setup(tmp_path, {"[第1集-1场] 1-4": [{"sub": True}, {"sub": True}]})
    _, pack_id = _seed(store)
    rows = json.loads(json.dumps(SHOTS))
    rows[1]["description"] = "{第1集-1场} 接上一段 (安妮)"
    shots_id = _shots(store, rows, _lib_of(store, pack_id))
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, episode=1)
    assert r.meta["complete"] is False
    gens = [c["summary"] for c in reg.calls if c.get("summary")]
    assert gens == ["[第1集-1场] 1-4"] * 2, "第 1 段重生成后仍带字：第 2 段不拿它的尾帧去渲"
    assert "要接的前序段" in r.content


# ---------------------------------------------------------------- J4：按段指纹取回


async def test_质检重生成时超时的任务_重跑按段指纹取回_不重付(tmp_path):
    store, fns, reg = _setup(tmp_path, ledger=True)
    shots_id, pack_id = _seed(store)
    reg.timeout_once.add("[第1集-2场] 5-8")
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, episode=1)
    assert r.meta["complete"] is False and r.meta["failed"] == 1

    reg.calls.clear()
    r2 = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, episode=1)
    assert r2.ok and r2.meta["complete"], r2.content
    assert reg.recovered == ["task-2"], "按段指纹找到上次没等到的任务，取回"
    assert not [c for c in reg.calls if c.get("summary")], "取回的过了质检：不重新生成"
    assert "取回了上次没等到的任务" in r2.content


# ---------------------------------------------------------------- J5：按片段 id 要回旧版


async def test_redo换掉的旧版本_accept按片段id要回来(tmp_path):
    store, fns, reg = _setup(tmp_path)
    shots_id, pack_id = _seed(store)
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, episode=1)
    assert r.ok
    old = next(a.id for a in store.find(type_=AssetType.VIDEO)
               if a.summary == "[第1集-2场] 5-8")

    r2 = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, episode=1,
                                          redo=["5-8"])
    assert r2.ok and store.get(old).status is AssetStatus.REJECTED
    newer = reg.composed[-1][1]
    assert newer != old

    reg.calls.clear()
    r3 = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, episode=1,
                                          accept=[old])
    assert r3.ok and r3.meta["complete"]
    assert not [c for c in reg.calls if c.get("summary")], "要回旧版：不重新生成"
    assert reg.composed[-1][1] == old, "成片用回旧版"
    assert store.get(old).status is AssetStatus.ACTIVE

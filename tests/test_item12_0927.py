"""待拍板第 12 条的四项（2026-09-27 用户：「请按你的推荐来」）。

1. 自带剧本（/length auto 跟剧本走）的开场：用剧本自己的开场；平的才加 ≤5 秒无台词画面闪前
   （【闪前】）；不许把后面的台词复制到开头；开场没高潮点只提醒；项目卡列出加了闪前 / 复制了台词的集
2. 人物一致「没查成」不算过：⛔ 不进成片、不重生成；重跑先补查，过了复用不重付；和字幕的没查成
   一起算熔断；配置里没开（没网关 / 没视觉角色）的不算没查成
3. 参考位不够要省场景 / 道具的段：花钱之前写进整批报价
4. 镜头超 3 秒：默认只标 ⚠；/cut 拦 之后超了重生成、仍超 ⛔；渲染结果和第 1 集停点列出超了的段
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aigc_agent.capabilities.memory.session import SessionSnapshot
from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.drama.card import build_project_card
from aigc_agent.domain.drama.format import (
    FLASH_MARK,
    EpisodeFormat,
    check_shots,
    check_storyboard,
    copied_opening,
    has_flash,
    opening_note,
    shots_rules,
    storyboard_problems,
    storyboard_rules,
)
from aigc_agent.domain.drama.identity import IdentityVerdict
from aigc_agent.domain.drama.models import Episode, ShotPrompt
from aigc_agent.domain.functions.drama import DramaFunctions
from aigc_agent.domain.output import OutputPrefs
from aigc_agent.domain.pipeline.episode_pipeline import EpisodePipeline
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.model.budget import CostGuard
from aigc_agent.harness.tools.provider import ToolResult
from tests.test_episode_pipeline import FakeRegistry
from tests.test_gate_chain import _fns as _gate_fns
from tests.test_gate_chain import _seed
from tests.test_render_reuse_0926 import _setup as _render_setup

FOLLOW = replace(EpisodeFormat(), follow_script=True)
PLAIN = EpisodeFormat()

SCENE = "[夜] [内] [酒店走廊]"
FLAT = (  # 剧本开场是铺垫：走进来、看看、坐下
    f"{SCENE}\n"
    "[中景/推入/平视/3s] 陆离推门走进走廊，环顾四周\n"
    "[近景/固定/平视/3s] 陆离低头看表\n"
    "[近景/固定/平视/3s] 小满跟进来，说：“你来得真早。”\n"
    "[中景/固定/平视/3s] 两人并排站着\n"
    "[特写/固定/平视/3s] 陆离说：“把箱子给我，数到三就跑。”\n"
    "[中景/固定/平视/3s] 小满愣住\n"
)


def _ep(desc: str, index: int = 1) -> Episode:
    return Episode(index, f"第{index}集", desc)


# ================================================================ 1. 自带剧本的开场


def test_跟剧本走_规则_开场用剧本的_只许无台词闪前_不许复制台词():
    rules = storyboard_rules(FOLLOW)
    assert "开场照剧本自己的开场拆" in rules and FLASH_MARK in rules
    assert "不许把后面的台词复制到开头" in rules and "无台词" in rules
    assert "开场 15 秒内必须出现高潮点" in storyboard_rules(PLAIN), "Agent 自己写的剧本照旧"
    assert "分镜开场没标就不标" in shots_rules(FOLLOW)
    assert "开场没有高潮点视为不合格" in shots_rules(PLAIN)


def test_跟剧本走_开场没高潮点只提醒_不算问题():
    ep = _ep(FLAT)
    assert not any("高潮点" in p for p in check_storyboard(ep, FOLLOW))
    assert "开场 15 秒没有高潮点" in opening_note(ep, FOLLOW)
    assert any("开场 15 秒内没有高潮点" in p for p in check_storyboard(ep, PLAIN)), (
        "不跟剧本走的照旧要求"
    )
    shots = [ShotPrompt(scene_index="[第1集-1场]", video_name="1-5", duration="15s",
                        description="x", cuts=[3, 3, 3, 3, 3])]
    assert not any("hook" in p for p in check_shots(shots, FOLLOW))
    assert any("hook" in p for p in check_shots(shots, PLAIN)), "不跟剧本走的照旧要求"


def test_闪前_只许画面_不超过5秒():
    ok = (
        f"{SCENE}\n【高潮点】{FLASH_MARK}[特写/固定/平视/2s] 刀光一闪，箱子落地\n"
        + FLAT.split("\n", 1)[1]
    )
    assert not any("闪前" in p for p in check_storyboard(_ep(ok), FOLLOW))
    assert has_flash(ok) and opening_note(_ep(ok), FOLLOW) == ""
    talk = ok.replace("刀光一闪，箱子落地", "陆离喊：“数到三就跑！”")
    assert any("闪前镜头里有台词" in p for p in check_storyboard(_ep(talk), FOLLOW))
    long = (
        f"{SCENE}\n【高潮点】{FLASH_MARK}[特写/固定/平视/3s] 刀光\n"
        f"{FLASH_MARK}[特写/固定/平视/3s] 箱子落地\n" + FLAT.split("\n", 1)[1]
    )
    assert any("开场闪前 6 秒" in p for p in check_storyboard(_ep(long), FOLLOW))


def test_开场复制了后面的台词_不许_剧本里本来就说两遍的不算():
    script = "第1集\n小满：你来得真早。\n陆离：把箱子给我，数到三就跑。\n"
    copied = (
        f"{SCENE}\n【高潮点】[特写/固定/平视/3s] 陆离说：“把箱子给我，数到三就跑。”\n"
        + FLAT.split("\n", 1)[1]
    )
    why = copied_opening("第1集", script, copied, FOLLOW)
    assert "开场把后面的台词复制了一遍" in why and "把箱子给我" in why
    assert any("复制了一遍" in p for p in storyboard_problems([_ep(copied)], script, FOLLOW))
    assert not any("复制了一遍" in p for p in storyboard_problems([_ep(copied)], script, PLAIN))
    twice = script + "陆离：把箱子给我，数到三就跑。\n"
    assert copied_opening("第1集", twice, copied, FOLLOW) == "", "剧本里本来就说两遍"


def test_项目卡_跟剧本走的项目列出复制了台词和加了闪前的集():
    store = AssetStore()
    script = "第1集\n小满：你来得真早。\n陆离：把箱子给我，数到三就跑。\n"
    boards = {
        1: f"{SCENE}\n【高潮点】[特写/固定/平视/3s] 陆离说：“把箱子给我，数到三就跑。”\n"
        + FLAT.split("\n", 1)[1],
        2: f"{SCENE}\n【高潮点】{FLASH_MARK}[特写/固定/平视/2s] 刀光一闪\n"
        + FLAT.split("\n", 1)[1],
    }
    for n, desc in boards.items():
        s = store.create(script.replace("第1集", f"第{n}集"), type_=AssetType.SCRIPT,
                         summary=f"第{n}集", creator="human:import", gen_params={"episode": n})
        store.create(
            json.dumps([{"episodeIndex": n, "episodeTitle": f"第{n}集", "episodeDesc": desc}],
                       ensure_ascii=False),
            type_=AssetType.STORYBOARD, summary="分镜", creator="tool:drama_storyboard",
            parents=[s.id],
        )
    card = build_project_card(store, None, FOLLOW)
    assert "第 1 集的分镜开场把后面的台词复制了一遍" in card
    assert "第 2 集的开场加了无台词画面闪前" in card
    assert "复制了一遍" not in build_project_card(store, None, PLAIN), "不跟剧本走的项目不查"


# ================================================================ 2. 人物一致「没查成」


async def test_人物一致没查成_不进成片_不重生成_重跑先补查_过了复用不重付(tmp_path):
    store, fns, reg = _render_setup(tmp_path)
    shots_id, pack_id = _seed(store)
    asked: list[str] = []

    async def flaky(asset_id: str, refs: Any, is_video: bool, pass_score: int) -> Any:
        name = store.get(asset_id).summary
        asked.append(name)
        if name == "[第1集-2场] 5-8" and asked.count(name) == 1:
            return IdentityVerdict(note="一致性校验调用失败：ReadTimeout", unchecked=True)
        return IdentityVerdict(passed=True, score=9)

    fns._check_identity = flaky  # type: ignore[method-assign]
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, episode=1)
    assert r.meta["blocked"] == 1 and not reg.composed
    assert "人物一致没查成" in r.content
    assert len([c for c in reg.calls if c.get("summary")]) == 2, "没查成不是画面问题：不重生成"

    reg.calls.clear()
    r2 = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, episode=1)
    assert r2.ok and r2.meta["complete"] and reg.composed
    assert not [c for c in reg.calls if c.get("summary")], "补查过了：复用，不重付"
    clip = next(a for a in store.find(type_=AssetType.VIDEO) if a.summary == "[第1集-2场] 5-8")
    assert any("重查人物一致" in n for n in clip.gen_params["tags"]["notes"])


async def test_配置里没开一致性检查_不算没查成():
    store = AssetStore()
    fns, reg, _ = _gate_fns(store, {})

    async def off(asset_id: str, refs: Any, is_video: bool, pass_score: int) -> Any:
        return IdentityVerdict(note="没有文本网关，跳过一致性校验")

    fns._check_identity = off  # type: ignore[method-assign]
    shots_id, pack_id = _seed(store)
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.ok and r.meta["complete"] is True and reg.composed


async def test_人物一致连续没查成也熔断(tmp_path):
    rows = [
        {"scene_index": f"[第1集-{k}场]", "video_name": f"{k}", "video_duration": "12s",
         "cuts": [3, 3, 3, 3], "description": f"第 {k} 段 (安妮)"}
        for k in range(1, 5)
    ]
    store, fns, reg = _render_setup(tmp_path, unchecked_break="2")
    fns.catalog.max_concurrency = lambda k: 1

    async def down(asset_id: str, refs: Any, is_video: bool, pass_score: int) -> Any:
        return IdentityVerdict(note="一致性校验调用失败：ConnectError", unchecked=True)

    fns._check_identity = down  # type: ignore[method-assign]
    _, pack_id = _seed(store)
    lib = store.get(pack_id).parent_ids[0]
    shots = store.create(json.dumps(rows, ensure_ascii=False), summary="提示词",
                         parents=["as_sb", lib])
    r = await fns._fn_drama_render_shots(shots.id, rendered_id=pack_id, episode=1)
    assert len([c for c in reg.calls if c.get("summary")]) == 2
    assert "连续 2 段没查成" in r.content and "人物一致" in r.content


# ================================================================ 3. 参考位不够：报价时先说


class _QuoteReg:
    """造视频片段 + 会报价的闸门（记下报价单）。"""

    def __init__(self, store: AssetStore) -> None:
        self.store = store
        self.calls: list[dict[str, Any]] = []
        self.quotes: list[str] = []
        outer = self

        class _Gate:
            guard = CostGuard(call_limits={"video": 100})

            async def confirm_batch(self, tool: str, text: str, args: Any) -> bool:
                outer.quotes.append(text)
                return True

        self.gate = _Gate()

    async def invoke(self, name: str, args: dict[str, Any]) -> ToolResult:
        if name == "compose_video":
            a = self.store.create("mp4", summary="成片", creator="fake")
            return ToolResult(ok=True, content="拼好了", asset_ref=a.id)
        self.calls.append(dict(args))
        a = self.store.create("", type_=AssetType.VIDEO, summary=args["summary"],
                              creator="model:x", gen_params={"tags": dict(args.get("tags") or {})})
        a.uri = f"https://fake/{a.id}.mp4"
        self.store.put(a)
        return ToolResult(ok=True, content="ok", asset_ref=a.id)


def _three_people(store: AssetStore) -> tuple[str, str]:
    people = ["安妮", "老王", "小李"]
    lib = {
        "characters": [
            {"baseRoleName": n, "roleTotalDesc": "成年人",
             "roleCostumeList": [{"costumeName": f"{n}-便装-[全集]", "costumeDesc": "便装"}]}
            for n in people
        ],
        "scenes": [{"name": "酒店走廊", "description": "走廊"}],
        "props": [{"name": "行李箱", "description": "银色"}],
    }
    lib_a = store.create(json.dumps(lib, ensure_ascii=False), summary="资产库",
                         creator="tool:drama_assets")
    pack: dict[str, dict[str, str]] = {}
    for key, kind in [(n, "角色") for n in people] + [
        (f"{n}-便装-[全集]", "服装") for n in people
    ] + [("酒店走廊", "场景"), ("行李箱", "道具")]:
        img = store.create("", type_=AssetType.IMAGE, summary=key, creator="model:x")
        img.uri = f"https://img/{img.id}.png"
        store.put(img)
        pack[key] = {"asset": img.id, "url": img.uri, "kind": kind}
    pack_a = store.create(json.dumps(pack, ensure_ascii=False), summary="参考图包",
                          creator="tool:drama_render_assets", parents=[lib_a.id])
    shots = [{
        "scene_index": "[第1集-1场]", "video_name": "1-4", "video_duration": "12s",
        "cuts": [3, 3, 3, 3],
        "description": "三人同框 (安妮-便装-[全集]) (老王-便装-[全集]) (小李-便装-[全集]) "
                       "站在 (酒店走廊)，脚边 (行李箱)",
    }]
    shots_a = store.create(json.dumps(shots, ensure_ascii=False), summary="提示词",
                           parents=["as_sb", lib_a.id])
    return shots_a.id, pack_a.id


async def test_参考位不够要省场景道具_花钱之前写进报价():
    store = AssetStore()
    reg = _QuoteReg(store)
    fns = DramaFunctions(None, store, registry=reg)
    fns.catalog = SimpleNamespace(
        drama={"subtitle_gate": "false", "cut_gate": "false", "identity_gate": "false",
               "ref_probe": "false"},
        max_concurrency=lambda k: 0,
        get=lambda kind, model: SimpleNamespace(max_refs=7, max_duration=0, aspect_ratios=[]),
    )
    shots_id, pack_id = _three_people(store)
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, episode=1)
    assert r.ok, r.error
    quote = reg.quotes[0]
    assert "参考位不够：1 段要省掉场景 / 道具参考图" in quote
    assert "[第1集-1场] 1-4：道具「行李箱」" in quote, "3 张服装 + 3 张脸 + 场景 = 7，道具让位"
    assert "⚠ 参考位不够（上限 7）：1 段省掉了场景 / 道具参考图" in r.content
    assert len(reg.calls[0]["image"]) == 7


# ================================================================ 4. 镜头超 3 秒


async def test_镜头超3秒_默认只标照样进成片_结果里列出来给判断办法():
    store = AssetStore()
    fns, reg, _ = _gate_fns(store, {"[第1集-1场] 1-4": [{"cut": 5.0}, {"cut": 4.2}]})
    shots_id, pack_id = _seed(store)
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.ok and r.meta["complete"] is True and reg.composed, "默认只标 ⚠"
    assert "1 段重生成后仍有超过 3 秒的镜头" in r.content
    assert "[第1集-1场] 1-4（最长 4.2s）" in r.content
    assert "/cut 拦" in r.content and "5 段里至少 4 段确实超了" in r.content


async def test_cut拦之后_超了重生成_仍超不进成片():
    store = AssetStore()
    fns, reg, _ = _gate_fns(store, {"[第1集-1场] 1-4": [{"cut": 5.0}, {"cut": 4.2}]})
    fns.cut_block = True
    assert "cut" in fns._gate_cfg()["block"]  # noqa: SLF001
    shots_id, pack_id = _seed(store)
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.meta["complete"] is False and r.meta["blocked"] == 1 and not reg.composed
    assert "/cut 拦" not in r.content, "已经拦了，不再提示"


def test_cut设置按项目存_第二个窗口也抄过去(tmp_path: Path):
    snap = SessionSnapshot(tmp_path, "proj")
    assert snap.cut_block is False
    snap.set_cut_block(True)
    assert SessionSnapshot(tmp_path, "proj").cut_block is True
    other = SessionSnapshot(tmp_path, "proj~2")
    other.inherit_settings(snap)
    assert SessionSnapshot(tmp_path, "proj~2").cut_block is True


def test_第1集停点列出镜头超3秒的段(tmp_path: Path):
    store = AssetStore()
    for name, note in (("1-4", "⚠ 仍有超过 3 秒的镜头（最长 4.2s）"), ("5-8", "人物一致 9/10")):
        a = store.create("", type_=AssetType.VIDEO, summary=f"[第1集-1场] {name}",
                         creator="model:x",
                         gen_params={"tags": {"episode": 1, "scene": "[第1集-1场]", "name": name,
                                              "notes": [note]}})
        store.put(a)
    pipe = EpisodePipeline(FakeRegistry(store), store, EventBus(), OutputPrefs(tmp_path))
    text = pipe._cut_report(1)  # noqa: SLF001
    assert "镜头超 3 秒的有 1 段（[第1集-1场] 1-4（最长 4.2s））" in text and "/cut 拦" in text
    assert pipe._cut_report(2) == ""  # noqa: SLF001

"""2026-09-23 审查收尾的零散项。

  · 记忆提取把整本剧本、整份分镜原样喂进去（单次 4.1 万 token）
  · 手机随拍类配方被真实感段的「侧光 / 闪光灯直闪」覆盖掉「自然光、随手拍」
  · 抽帧查字幕只有短剧链有：短视频镜头画上字幕条、标题照样进成片
  · 配方里的 tts_model 写了没人读
  · 回收站从不清理
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.assets.trash import purge_trash, trash_folders
from aigc_agent.domain.media.no_text import NO_TEXT_RETRY
from aigc_agent.domain.pipeline.recipe import load_recipe
from aigc_agent.harness.context.window import Turn
from tests.test_short_video import Registry, _fns


def test_记忆提取不喂整本工具返回():
    t = Turn(index=0, messages=[
        {"role": "user", "content": "第3集开头别写成硬广"},
        {"role": "tool", "tool_call_id": "c1", "content": "剧本正文" * 2000},
        {"role": "tool", "tool_call_id": "c2", "content": "人的决策：打回重做\n理由：太硬广"},
        {"role": "assistant", "content": "好的"},
    ])
    s = t.transcript
    assert "用户：第3集开头别写成硬广" in s and "助手：好的" in s
    assert "理由：太硬广" in s, "人审决策（短）要原样留着"
    assert "共 8000 字" in s and len(s) < 1200


def test_随拍配方只加皮肤真实感_不改布光():
    p = load_recipe("ugc-vlog").shot_prompt("手冲咖啡溢出来", "subtle")
    assert "毛孔" in p, "真人出镜，皮肤真实感还要"
    assert "侧光" not in p and "闪光灯" not in p, "和「自然光、iPhone 随手拍」冲突"
    assert "侧光" in load_recipe("talking-head").shot_prompt("x", "subtle"), "auto 照旧全套"


async def _brief(fns: Any) -> str:
    r = await fns.invoke("short_video_brief", {"keyword": "k", "style": "tech-short"})
    assert r.ok, r.error
    return r.asset_ref


async def test_短视频镜头画面有字_带禁令重生成():
    store = AssetStore()
    reg = Registry(store)
    fns = _fns(store, reg)
    verdicts = iter([(True, "底部字幕条")])

    async def fake(asset_id: str) -> tuple[bool | None, str]:
        return next(verdicts, (False, ""))

    fns._burned_text = fake  # type: ignore[method-assign]
    bid = await _brief(fns)
    fns.approve(bid)  # 人在确认单上点了头（2026-09-26：confirm=true 只认人的确认）
    r = await fns.invoke("short_video_produce", {"brief_id": bid, "confirm": True})
    assert r.ok, r.error
    prompts = [a["prompt"] for a in reg.of("gen_video")]
    assert sum(NO_TEXT_RETRY in p for p in prompts) == 1, "有字的那一镜带着「上一版出了字」重生成"
    assert "画面里有字" not in r.content


async def test_短视频镜头重生成后还有字_不进成片():
    store = AssetStore()
    reg = Registry(store)
    fns = _fns(store, reg)

    async def fake(asset_id: str) -> tuple[bool | None, str]:
        return store.get(asset_id).summary.endswith("第1镜"), "左上角标题字"

    fns._burned_text = fake  # type: ignore[method-assign]
    bid = await _brief(fns)
    fns.approve(bid)  # 人在确认单上点了头（2026-09-26：confirm=true 只认人的确认）
    r = await fns.invoke("short_video_produce", {"brief_id": bid, "confirm": True})
    assert r.ok, r.error
    assert "第1镜：画面里有字（左上角标题字）" in r.content
    # 有字的镜头不进成片，缺镜头就不合成（2026-09-26：和短剧「缺段不成片」同一条规则）
    assert "未成片" in r.content and r.meta.get("complete") is False
    assert not reg.of("compose_video")


async def test_短视频查不了字幕要说出来():
    store = AssetStore()
    reg = Registry(store)
    fns = _fns(store, reg)  # 假注册表生成的片段没有本地副本 → 查不了
    bid = await _brief(fns)
    fns.approve(bid)  # 人在确认单上点了头（2026-09-26：confirm=true 只认人的确认）
    r = await fns.invoke("short_video_produce", {"brief_id": bid, "confirm": True})
    assert r.ok, r.error
    assert "没做成字幕检查" in r.content


async def test_配方里的tts_model接上():
    store = AssetStore()
    reg = Registry(store)
    fns = _fns(store, reg)
    orig = fns._recipe

    def with_tts(name: str) -> Any:
        recipe, err = orig(name)
        if recipe is not None:
            recipe.models["tts_model"] = "speech-2.8-hd"
        return recipe, err

    fns._recipe = with_tts  # type: ignore[method-assign]
    bid = await _brief(fns)
    fns.approve(bid)  # 人在确认单上点了头（2026-09-26：confirm=true 只认人的确认）
    r = await fns.invoke("short_video_produce", {"brief_id": bid, "confirm": True})
    assert r.ok, r.error
    tts = reg.of("tts")
    assert tts and tts[0]["model"] == "speech-2.8-hd"


def test_回收站按天数清理_迁移备份不清(tmp_path: Path):
    trash = tmp_path / "trash"
    old = trash / "20200101-000000"
    mig = trash / "migrate-20200101-000000"
    new = trash / time.strftime("%Y%m%d-%H%M%S")
    for d, name in ((old, "a.txt"), (mig, "manifest.json"), (new, "b.txt")):
        d.mkdir(parents=True)
        (d / name).write_text("x", encoding="utf-8")
    entries = trash_folders(tmp_path)
    assert {e.name for e in entries} == {old.name, mig.name, new.name}
    assert entries[0].age_days > 365 and entries[-1].age_days < 1
    gone = purge_trash(tmp_path, 30)
    assert gone == [old], "迁移备份撤销要用，最近的也不动"
    assert mig.exists() and new.exists() and not old.exists()

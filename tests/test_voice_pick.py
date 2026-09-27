"""按内容自动选音色验收。

要解决的问题：同一个音色配所有内容，是"听着假"的一个隐性来源 ——
用新闻女声念一条生活种草，语气和内容对不上，哪怕音色本身很自然也别扭。

这里最关键的一条是**校验**。MiniMax 收到不认识的 voice_id **不会报错**，
而是静默换成自己的默认音色（实测过）。所以模型编一个音色名出来，
表现是"一切正常但音色从没变过" —— 比报错难发现得多。
"""

from __future__ import annotations

import json

import pytest

from aigc_agent.domain.pipeline.voice_pick import (
    SPEED_MAX,
    SPEED_MIN,
    parse_pick,
    pick_prompt,
)

VOICES = [
    ("Chinese (Mandarin)_News_Anchor", "新闻女声，干练清晰，资讯播报"),
    ("Chinese (Mandarin)_Radio_Host", "电台男主播，娓娓道来有呼吸感，叙事"),
    ("Chinese (Mandarin)_Warm_Bestie", "温暖闺蜜，像朋友聊天，种草测评"),
]
ALLOWED = [v for v, _ in VOICES]
FALLBACK = ALLOWED[0]


def _resp(**kw: object) -> str:
    return json.dumps(kw, ensure_ascii=False)


# ---------- 提示词 ----------


def test_音色说明要进提示词():
    """只给 id 模型挑不准，得让它看到"适合什么内容"。"""
    p = pick_prompt("随便一段文案", VOICES)
    for _, note in VOICES:
        assert note in p


def test_按语气判断而不是按题材():
    p = pick_prompt("文案", VOICES, topic="科技")
    assert "语气" in p
    assert "题材" in p, "没说清楚题材和语气可能不一致，模型会按选题标签硬选"


def test_语速范围写进提示词():
    p = pick_prompt("文案", VOICES)
    assert str(SPEED_MIN) in p and str(SPEED_MAX) in p


# ---------- 校验：核心 ----------


def test_编造的音色被挡回默认():
    """这条是整个模块的意义所在。

    MiniMax 收到不认识的 voice_id 不报错，静默换成默认音色 ——
    不校验的话，表现是"自动选音色一直在工作"，其实从没生效过。
    """
    pick = parse_pick(_resp(voice="甜妹音色v2", speed=1.0), ALLOWED, FALLBACK)
    assert pick.voice in ALLOWED


def test_选对了就用它():
    pick = parse_pick(
        _resp(voice="Chinese (Mandarin)_Radio_Host", speed=0.95, why="叙事口吻"),
        ALLOWED,
        FALLBACK,
    )
    assert pick.voice == "Chinese (Mandarin)_Radio_Host"
    assert pick.speed == 0.95
    assert pick.why == "叙事口吻"


def test_抄漏一截也能对上():
    """模型有时只写 Radio_Host 或带多余空格。"""
    assert parse_pick(_resp(voice="Radio_Host"), ALLOWED, FALLBACK).voice == (
        "Chinese (Mandarin)_Radio_Host"
    )


@pytest.mark.parametrize("bad", [0.3, 2.5, -1, 0])
def test_离谱语速被夹住(bad):
    """模型为了"更有节奏"给过 0.5 这种值，人耳一听就不对。"""
    pick = parse_pick(_resp(voice=ALLOWED[0], speed=bad), ALLOWED, FALLBACK)
    assert SPEED_MIN <= pick.speed <= SPEED_MAX


def test_语速不是数字时退回默认():
    pick = parse_pick(_resp(voice=ALLOWED[0], speed="快一点"), ALLOWED, FALLBACK, 0.9)
    assert pick.speed == 0.9


# ---------- 退路 ----------


@pytest.mark.parametrize(
    "text",
    ["模型在瞎说", "", "[1,2,3]", "```json\n不是合法JSON\n```"],
)
def test_解析失败退回默认而不是抛(text):
    pick = parse_pick(text, ALLOWED, FALLBACK, 0.95)
    assert pick.voice == FALLBACK
    assert pick.speed == 0.95


def test_认markdown围栏():
    text = "好的：\n```json\n" + _resp(voice=ALLOWED[1], speed=1.0) + "\n```"
    assert parse_pick(text, ALLOWED, FALLBACK).voice == ALLOWED[1]


def test_没给音色时退回默认():
    assert parse_pick(_resp(speed=1.0), ALLOWED, FALLBACK).voice == FALLBACK


# ---------- 配方 ----------


def test_配方开了自动选音色():
    from aigc_agent.app import PROJECT_ROOT
    from aigc_agent.domain.pipeline.recipe import load_recipe

    r = load_recipe(str(PROJECT_ROOT / "config" / "recipes" / "tech-short.yaml"))
    assert str(r.voiceover.get("voice")).lower() == "auto"


def test_候选池够大才谈得上自动选():
    """只有两三个音色，"自动选"没有意义。"""
    from aigc_agent.app import PROJECT_ROOT
    from aigc_agent.domain.generators.catalog import MediaCatalog

    c = MediaCatalog.load(PROJECT_ROOT / "config" / "media_models.yaml")
    assert len(c.speech_voices) >= 10
    assert all(v.note for v in c.speech_voices), "有音色没写用途说明，模型挑不准"


# ---------- 接线：选出来的音色必须真的用上 ----------

import asyncio  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any  # noqa: E402

from aigc_agent.domain.assets.store import AssetStore  # noqa: E402
from aigc_agent.domain.functions.short_video import ShortVideoFunctions  # noqa: E402
from aigc_agent.interfaces.cli import video_cmd  # noqa: E402
from tests.test_short_video import RECIPES, Gateway, Registry, _plan  # noqa: E402


class _Gateway(Gateway):
    """选音色那步选 Radio_Host、语速 0.9；简报按 test_short_video 的计划走。"""

    async def chat(self, role: str, messages: list[dict[str, Any]], **kw: Any) -> Any:
        if role == "voice_select":
            self.calls.append((role, messages[-1]["content"]))
            return SimpleNamespace(text=_resp(voice=ALLOWED[1], speed=0.9, why="叙事"))
        return await super().chat(role, messages, **kw)


class _Agent:
    """真的简报 / 出片工具，假的生成、配音、合成；记下 tts 拿到了什么。

    `agent video make` 2026-09-23 起走和对话同一组工具（short_video_brief →
    short_video_produce），选音色在 produce 里做，所以这里要把真工具接上。
    """

    def __init__(self) -> None:
        self.assets = AssetStore()
        self.inner = Registry(self.assets)  # gen_video / tts / transcribe / compose_video
        self.gateway = _Gateway(_plan())
        self.catalog = SimpleNamespace(
            drama={"realism_level": "subtle", "video_retries": "1"},
            max_concurrency=lambda kind: 0,
            speech_voices=[SimpleNamespace(name=n, note=t) for n, t in VOICES],
            speech_default_voice=FALLBACK,
        )
        self.fns = ShortVideoFunctions(
            self.gateway, self.assets, registry=self, catalog=self.catalog, recipes_dir=RECIPES
        )
        self.short_video_fns = self.fns  # 终端确认后命令行调它的 approve()
        self.registry = self

    async def setup(self, mcp: bool = True) -> None: ...
    async def aclose(self) -> None: ...

    async def invoke(self, name: str, args: dict[str, Any]) -> Any:
        if name.startswith("short_video_"):
            return await self.fns.invoke(name, args)
        return await self.inner.invoke(name, args)


def test_选出来的音色真的传给了tts(monkeypatch):
    """这是个**接线** bug 的回归测试。

    真出过一次：自动选音色跑通了、控制台也打印了"音色：沉稳高管"，
    但返回值没接到 tts 调用上，实际发出去的还是配方里的字面量 "auto"，
    于是配音直接失败、成片无声。
    日志看起来一切正常 —— 只有成片打开才发现没声音。
    """
    a = _Agent()
    monkeypatch.setattr(video_cmd.Agent, "create", staticmethod(lambda *x, **k: a))
    asyncio.run(
        video_cmd._make("科技", "tech-short", "", 0, "", "", False, False, True, False, True, 0.0)
    )

    tts = a.inner.of("tts")
    assert tts, "压根没调 tts"
    assert tts[0]["voice"] != "auto", "把配方里的 auto 字面量直接发出去了"
    assert tts[0]["voice"] in ALLOWED
    assert tts[0]["speed"] == 0.9, "选出来的语速也没传过去"

"""短视频「接真实热点」验收（`agent video make`）。

这条链路出过一次**静默跳过**的事故：代码没落盘，`0/5 拉热榜` 整步没执行，
流程照常跑完出片，文案里的数字全是模型编的（「比阿波罗登月计算机强一百万倍」），
肉眼看输出完全正常。所以这里盯的不是"函数能不能跑"，而是：

  · 配方声明了接热点，跑起来就**必须**真去拉热榜；
  · 拉到的热榜原文**必须**进到写简报（口播 + 分镜）的提示词里，否则数字仍然没有出处。

2026-09-23 审查后 `agent video make` 不再自己写文案，走和对话同一组工具
（short_video_brief → short_video_produce），这里连同出片前确认一起验。
"""

from __future__ import annotations

import asyncio
from typing import Any

from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.functions.short_video import ShortVideoFunctions
from aigc_agent.domain.pipeline.recipe import load_recipe
from aigc_agent.harness.tools.provider import ToolResult
from aigc_agent.interfaces.cli import video_cmd
from tests.test_short_video import RECIPES, Gateway, Registry, _catalog, _plan

HOT = """ 1. 氢能储运瓶颈加速打通  热度 1194.7w  [科技]
 2. 奥尔特曼称OpenAI或放缓AI研发  热度 933.1w  [科技]"""


class _FakeAgent:
    """真的简报 / 出片工具，假的热榜、生成、配音和合成；记下每一次工具调用。"""

    def __init__(self, hot_ok: bool = True) -> None:
        self.assets = AssetStore()
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.hot_ok = hot_ok
        self.inner = Registry(self.assets)  # gen_video / tts / transcribe / compose_video
        self.gateway = Gateway(_plan())
        self.fns = ShortVideoFunctions(
            self.gateway, self.assets, registry=self, catalog=_catalog(), recipes_dir=RECIPES
        )
        self.short_video_fns = self.fns  # 终端确认后命令行调它的 approve()
        self.registry = self

    async def setup(self, mcp: bool = True) -> None: ...
    async def aclose(self) -> None: ...

    async def invoke(self, name: str, args: dict[str, Any]) -> ToolResult:
        self.calls.append((name, dict(args)))
        if name == "douyin_hot_list":
            if not self.hot_ok:
                return ToolResult(ok=False, error="热榜返回空")
            a = self.assets.create(HOT, summary="抖音热榜", creator="tool:douyin_hot_list")
            return ToolResult(content=HOT, asset_ref=a.id)
        if name.startswith("short_video_"):
            return await self.fns.invoke(name, args)
        return await self.inner.invoke(name, args)

    def names(self) -> list[str]:
        return [n for n, _ in self.calls if not n.startswith(("gen_", "tts", "transcribe"))]


def _run(monkeypatch, agent: _FakeAgent, *, topic: str = "科技", yes: bool = True,
         dry: bool = True, no_ground: bool = False) -> None:
    monkeypatch.setattr(video_cmd.Agent, "create", staticmethod(lambda *a, **k: agent))
    asyncio.run(
        video_cmd._make(topic, "tech-short", "", 0, "", "", False, False, yes, dry, no_ground)
    )


# ---------- 配方 ----------


def test_配方声明了接热点():
    g = load_recipe("tech-short").grounding
    assert g.get("enabled") is True
    assert g.get("category"), "要按分类筛：接口的 keyword 只搜热词标题，搜『科技』匹配不到"


# ---------- 端到端：这步必须真的发生 ----------


def test_声明接热点就一定会先拉热榜_热榜原文进简报提示词(monkeypatch):
    a = _FakeAgent()
    _run(monkeypatch, a)
    assert a.names() == ["douyin_hot_list", "short_video_brief"], "开了 grounding 却没去拉热榜"
    assert a.calls[0][1].get("category") == "科技"
    hot_id = a.calls[1][1]["sources"][0]
    assert a.assets.get(hot_id).creator == "tool:douyin_hot_list"
    _, prompt = a.gateway.calls[0]
    assert "氢能储运瓶颈加速打通" in prompt, "热榜原文没进写简报的提示词，数字就还是没出处"


def test_热榜挂了要降级而不是崩(monkeypatch):
    """热榜是加分项。它挂了仍要出简报 —— 但不能假装有数据。"""
    a = _FakeAgent(hot_ok=False)
    _run(monkeypatch, a)
    assert a.names() == ["douyin_hot_list", "short_video_brief"]
    assert a.calls[1][1]["sources"] == []
    _, prompt = a.gateway.calls[0]
    assert "没有抓到任何热点数据" in prompt


def test_no_ground_能关掉(monkeypatch):
    a = _FakeAgent()
    _run(monkeypatch, a, no_ground=True)
    assert "douyin_hot_list" not in a.names(), "--no-ground 之后不该再拉热榜"


def test_出片走同一个工具_先报成本再生成(monkeypatch):
    a = _FakeAgent()
    _run(monkeypatch, a, dry=False, yes=True)
    produce = [args for n, args in a.calls if n == "short_video_produce"]
    assert len(produce) == 2, "先不带 confirm 拿成本确认，再带 confirm 生成"
    assert not produce[0].get("confirm") and produce[1]["confirm"] is True
    assert a.inner.of("gen_video"), "确认之后才生成"
    assert a.inner.of("compose_video")


def test_dry_run只出简报(monkeypatch):
    a = _FakeAgent()
    _run(monkeypatch, a, dry=True)
    assert "short_video_produce" not in a.names() and not a.inner.of("gen_video")

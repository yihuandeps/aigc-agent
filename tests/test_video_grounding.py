"""短视频「接真实热点」验收。

这条链路出过一次**静默跳过**的事故：代码没落盘，`0/5 拉热榜` 整步没执行，
流程照常跑完出片，文案里的数字全是模型编的（「比阿波罗登月计算机强一百万倍」），
肉眼看输出完全正常。所以这里盯的不是"函数能不能跑"，而是：

  · 配方声明了接热点，跑起来就**必须**真去拉热榜；
  · 拉到的热榜原文**必须**进到写文案的提示词里，否则数字仍然没有出处。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from aigc_agent.domain.pipeline.recipe import load_recipe, parse_pick
from aigc_agent.interfaces.cli import video_cmd

HOT = """ 1. 氢能储运瓶颈加速打通  热度 1194.7w  [科技]
 2. 奥尔特曼称OpenAI或放缓AI研发  热度 933.1w  [科技]"""


class _FakeAgent:
    """记下每一次工具调用和每一条提示词，供断言检查。"""

    def __init__(self, hot_ok: bool = True) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.prompts: list[str] = []
        self.hot_ok = hot_ok
        self.registry = SimpleNamespace(invoke=self._invoke)
        self.gateway = SimpleNamespace(chat=self._chat)

    async def setup(self, mcp: bool = True) -> None: ...
    async def aclose(self) -> None: ...

    async def _invoke(self, name: str, args: dict):
        self.calls.append((name, args))
        if name == "douyin_hot_list":
            if not self.hot_ok:
                return SimpleNamespace(ok=False, content="", error="热榜返回空")
            return SimpleNamespace(ok=True, content=HOT, error="")
        return SimpleNamespace(ok=False, content="", error="本测试不该调到它")

    async def _chat(self, role: str, messages: list[dict]):
        self.prompts.append(messages[-1]["content"])
        if "只输出 JSON" in messages[-1]["content"]:  # 选题那一步
            return SimpleNamespace(text='{"topic": "氢怎么运", "why": "有画面"}')
        return SimpleNamespace(
            text='{"script": "口播", "shots": ["一", "二", "三", "四"]}'
        )


def _run(monkeypatch, agent: _FakeAgent, **kw) -> None:
    monkeypatch.setattr(video_cmd.Agent, "create", staticmethod(lambda *a, **k: agent))
    asyncio.run(
        video_cmd._make(
            kw.pop("topic", "科技"), "tech-short", "", 0, "", "",
            False, False, True, True, kw.pop("no_ground", False),
        )
    )


# ---------- 配方 ----------


def test_配方声明了接热点():
    g = load_recipe("tech-short").grounding
    assert g.get("enabled") is True
    assert g.get("category"), "要按分类筛：接口的 keyword 只搜热词标题，搜『科技』匹配不到"


def test_override_不会把_grounding_弄丢():
    r = load_recipe("tech-short")
    assert r.override(duration=45, tier="quality").grounded, "命令行覆盖不该把接热点关掉"


# ---------- 提示词 ----------


def test_热榜原文进了写文案的提示词():
    p = load_recipe("tech-short").script_prompt("氢怎么运", HOT)
    assert "氢能储运瓶颈加速打通" in p, "热榜原文没带进去，数字就还是没出处"
    assert "不要编造具体数值" in p


def test_没热榜时不假装有数据():
    p = load_recipe("tech-short").script_prompt("氢怎么运")
    assert "真实热榜数据" not in p, "没拿到热榜却说『以下是真实热榜』，等于教模型编"


@pytest.mark.parametrize(
    "text",
    [
        '{"topic": "甲", "why": "乙"}',
        '```json\n{"topic": "甲", "why": "乙"}\n```',
        '好的，我选：\n{"topic": "甲", "why": "乙"}\n以上。',
    ],
)
def test_选题解析容忍模型的各种包装(text):
    assert parse_pick(text) == ("甲", "乙")


def test_选题解析失败返回空而不是抛():
    assert parse_pick("我觉得选甲比较好") == ("", "")


# ---------- 端到端：这步必须真的发生 ----------


def test_声明接热点就一定会先拉热榜(monkeypatch):
    a = _FakeAgent()
    _run(monkeypatch, a)
    assert [n for n, _ in a.calls] == ["douyin_hot_list"], "配方开了 grounding 却没去拉热榜"
    assert a.calls[0][1].get("category") == "科技"


def test_写文案用的是热榜选出来的题而不是用户给的宽泛方向(monkeypatch):
    a = _FakeAgent()
    _run(monkeypatch, a, topic="科技")
    script_prompt = a.prompts[-1]
    assert "氢怎么运" in script_prompt, "选出来的题没用上"
    assert "氢能储运瓶颈加速打通" in script_prompt, "热榜原文没进提示词"


def test_热榜挂了要降级而不是崩(monkeypatch):
    """热榜是加分项。它挂了仍要出片 —— 但提示词里不能再出现『真实热榜数据』。"""
    a = _FakeAgent(hot_ok=False)
    _run(monkeypatch, a)
    assert "真实热榜数据" not in a.prompts[-1]


def test_no_ground_能关掉(monkeypatch):
    a = _FakeAgent()
    _run(monkeypatch, a, no_ground=True)
    assert a.calls == [], "--no-ground 之后不该再拉热榜"

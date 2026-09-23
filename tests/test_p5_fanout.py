"""M10 并行扇出验收。

P5 验收原话：「并行提速」。盯四件事：
  1. 真并发：5 个候选的耗时接近 1 个，不是 5 个之和；并发上限生效
  2. 结果按任务顺序回传，单个失败不影响其余
  3. 汇总保留每个候选**独有**的差异点，供人对比，不替人挑
  4. fan_out_candidates 每版落资产、挂血缘
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from aigc_agent.capabilities.subagents import (
    SubAgentDef,
    SubAgentRunner,
    distinctive_terms,
    fragments,
)
from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions.fanout import CANDIDATE_DEF, MAX_ANGLES, FanOutFunctions
from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.model.gateway import ModelResponse, Usage
from aigc_agent.harness.tools.provider import PermissionLevel
from aigc_agent.harness.tools.registry import ToolRegistry


class _SlowGw:
    """每次调用睡 delay 秒；按任务里的角度回不同文本；可指定某个角度失败。"""

    def __init__(self, delay: float = 0.05, fail_angle: str = "") -> None:
        self.delay = delay
        self.fail_angle = fail_angle
        self.active = 0
        self.max_active = 0
        self.calls = 0

    async def chat(self, role, messages, tools=None, **kw):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            user = next(m["content"] for m in reversed(messages) if m.get("role") == "user")
            m = re.search(r"本版角度：(.+)", user)
            angle = m.group(1).strip() if m else "默认"
            if angle == self.fail_angle:
                raise RuntimeError("模拟这一路挂了")
            return ModelResponse(
                text=f"这是{angle}版本。共同的结尾句。独有词{angle}特色。",
                usage=Usage(3, 3, cost=0.01),
            )
        finally:
            self.active -= 1


DEF = SubAgentDef(name="writer", system_prompt="写一版", role="subagent", max_iterations=2)


async def _runner(gw: _SlowGw) -> tuple[SubAgentRunner, EventBus]:
    bus = EventBus()
    reg = ToolRegistry(bus)
    await reg.refresh()
    return SubAgentRunner(gw, reg, bus), bus


def _tasks(*angles: str) -> list[str]:
    return [f"写文案\n\n本版角度：{a}" for a in angles]


# ---------------------------------------------------------------- 并发


async def test_并行比串行快():
    gw = _SlowGw(delay=0.05)
    runner, _ = await _runner(gw)
    started = time.perf_counter()
    s = await runner.run_many(DEF, _tasks("甲", "乙", "丙", "丁", "戊"), concurrency=5)
    elapsed = time.perf_counter() - started
    assert s.total == 5 and s.ok == 5 and s.failed == 0
    assert elapsed < 0.05 * 5 * 0.7, f"串行要 0.25s，并行只该接近 0.05s，实际 {elapsed:.3f}s"
    assert gw.max_active == 5


async def test_并发上限生效():
    gw = _SlowGw(delay=0.02)
    runner, _ = await _runner(gw)
    await runner.run_many(DEF, _tasks("a", "b", "c", "d", "e", "f"), concurrency=2)
    assert gw.max_active <= 2


async def test_结果按顺序_单个失败不影响其余():
    gw = _SlowGw(delay=0.01, fail_angle="乙")
    runner, _ = await _runner(gw)
    s = await runner.run_many(DEF, _tasks("甲", "乙", "丙"), concurrency=3)
    assert [r.ok for r in s.results] == [True, False, True]
    assert "甲" in s.results[0].text and "丙" in s.results[2].text
    assert s.failed == 1 and "挂了" in (s.results[1].error or "")
    assert s.cost is not None and abs(s.cost - 0.02) < 1e-9, "只加成功那两次的成本"


async def test_扇出事件():
    runner, bus = await _runner(_SlowGw(delay=0.001))
    await runner.run_many(DEF, _tasks("x", "y"), concurrency=2)
    start = [e for e in bus.history if e.type is EventType.FANOUT_START]
    end = [e for e in bus.history if e.type is EventType.FANOUT_END]
    assert start and start[0].data["tasks"] == 2 and start[0].data["concurrency"] == 2
    assert end and end[0].data["ok"] == 2 and end[0].data["total"] == 2
    assert len([e for e in bus.history if e.type is EventType.SUBAGENT_END]) == 2


# ---------------------------------------------------------------- 差异点


def test_差异点只列各自独有的():
    texts = [
        "反差开场，先说结论。共同的结尾句。",
        "数据开场，先给数字。共同的结尾句。",
        "故事开场，先讲人物。共同的结尾句。",
    ]
    diffs = distinctive_terms(texts, top=3)
    assert "反差开场" in diffs[0] and "数据开场" in diffs[1] and "故事开场" in diffs[2]
    assert all("共同的结尾句" not in d for d in diffs), "大家都有的不算差异"
    assert distinctive_terms(["", ""]) == [[], []]


def test_切段不分词不调模型():
    parts = fragments("反差开场，先说结论！共同的结尾句。")
    assert parts == ["反差开场", "先说结论", "共同的结尾句"]
    assert fragments("") == []


async def test_汇总渲染():
    runner, _ = await _runner(_SlowGw(delay=0.001, fail_angle="乙"))
    s = await runner.run_many(DEF, _tasks("甲", "乙"), concurrency=2)
    text = s.render()
    assert text.startswith("并行 2 个 · 成功 1 · 失败 1")
    assert "独有：" in text and "✗" in text


# ---------------------------------------------------------------- function


async def _fanout_registry(gw: _SlowGw) -> tuple[ToolRegistry, AssetStore, EventBus]:
    bus = EventBus()
    reg = ToolRegistry(bus)
    store = AssetStore()
    reg.register(FanOutFunctions(SubAgentRunner(gw, reg, bus), store))
    await reg.refresh()
    return reg, store, bus


async def test_每版落资产并挂血缘():
    reg, store, _ = await _fanout_registry(_SlowGw(delay=0.001))
    outline = store.create("大纲", type_=AssetType.OUTLINE, summary="大纲")
    r = await reg.invoke(
        "fan_out_candidates",
        {
            "task": "写一段露营装备种草文案",
            "angles": ["反差开场", "数据开场", "故事开场"],
            "kind": "copy",
            "parent_id": outline.id,
        },
    )
    assert r.ok, r.error
    assert "并行产出 3/3 个候选" in r.content and "独有：" in r.content
    assert "request_review" in r.content, "汇总要提示下一步交给人选"
    candidates = [a for a in store.all() if a.gen_params.get("fanout")]
    assert len(candidates) == 3
    assert all(a.parent_ids == [outline.id] and a.type is AssetType.TEXT for a in candidates)
    assert {a.gen_params["angle"] for a in candidates} == {"反差开场", "数据开场", "故事开场"}
    assert r.asset_ref == candidates[0].id
    assert all(a.gen_cost == 0.01 for a in candidates)


async def test_一路失败其余照样落资产():
    reg, store, _ = await _fanout_registry(_SlowGw(delay=0.001, fail_angle="数据开场"))
    r = await reg.invoke(
        "fan_out_candidates", {"task": "写", "angles": ["反差开场", "数据开场"]}
    )
    assert r.ok and "并行产出 1/2 个候选" in r.content and "✗ 候选2" in r.content
    assert len(store) == 1


async def test_角度为空或过多被拒():
    reg, _, _ = await _fanout_registry(_SlowGw(delay=0.001))
    r = await reg.invoke("fan_out_candidates", {"task": "写", "angles": []})
    assert not r.ok and "angles" in r.error
    r = await reg.invoke(
        "fan_out_candidates", {"task": "写", "angles": [f"角度{i}" for i in range(MAX_ANGLES + 1)]}
    )
    assert not r.ok and "最多" in r.error
    r = await reg.invoke(
        "fan_out_candidates", {"task": "写", "angles": ["a"], "parent_id": "as_0000000000"}
    )
    assert not r.ok, "血缘挂不上要在动手前就报"


def test_候选写手不带工具且只读():
    assert CANDIDATE_DEF.tools == [] and CANDIDATE_DEF.allowed == [PermissionLevel.READ]
    assert CANDIDATE_DEF.stateless


async def test_function本身不计媒体次数():
    reg, _, _ = await _fanout_registry(_SlowGw(delay=0.001))
    meta = reg.meta("fan_out_candidates")
    assert meta is not None and meta.permission is PermissionLevel.COMPUTE
    assert meta.cost_kind == "", "文本子代理的花费从 COST 事件记，不在闸门按媒体计次"


async def test_子代理之间互相看不见():
    gw = _SlowGw(delay=0.001)
    runner, _ = await _runner(gw)
    seen: list[list[dict[str, Any]]] = []
    orig = gw.chat

    async def spy(role, messages, tools=None, **kw):
        seen.append(messages)
        return await orig(role, messages, tools=tools, **kw)

    gw.chat = spy  # type: ignore[method-assign]
    await runner.run_many(DEF, _tasks("甲", "乙"), concurrency=2)
    for msgs in seen:
        users = [m["content"] for m in msgs if m.get("role") == "user"]
        assert len(users) == 1, "每个候选只看到自己的任务"

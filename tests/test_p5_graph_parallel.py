"""图运行时的 parallel / join 节点（P5，M10 节点级扇出）。

NodeType 里早就有 PARALLEL / JOIN，运行时一直没实现。盯：
  1. 分支真并发，汇聚后继续
  2. 打回能回到某条分支里的节点，只重跑那条分支
  3. 分支里有 human 节点、分支不汇聚 → 挂起交给人，不静默
"""

from __future__ import annotations

import asyncio
import time

import pytest

from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.execution.graph.models import Decision, GraphDef, NodeType
from aigc_agent.harness.execution.graph.runtime import GraphRuntime

from .test_p1_graph import FakeAgentExecutor, FakeToolExecutor  # noqa: TID252


def _graph(**overrides) -> GraphDef:
    spec = {
        "id": "par",
        "entry": "plan",
        "nodes": [
            {"id": "plan", "type": "agent", "out_slots": ["plan"]},
            {"id": "fan", "type": "parallel"},
            {"id": "draft", "type": "agent", "in_slots": ["plan"], "out_slots": ["draft"]},
            {"id": "images", "type": "agent", "in_slots": ["plan"], "out_slots": ["images"]},
            {"id": "merge", "type": "join"},
            {"id": "review", "type": "human", "in_slots": ["draft", "images"]},
            {
                "id": "pack",
                "type": "tool",
                "tool": "now",
                "in_slots": ["draft"],
                "out_slots": ["final"],
            },
            {"id": "end", "type": "terminal"},
        ],
        "edges": [
            {"from": "plan", "to": "fan"},
            {"from": "fan", "to": "draft"},
            {"from": "fan", "to": "images"},
            {"from": "draft", "to": "merge"},
            {"from": "images", "to": "merge"},
            {"from": "merge", "to": "review"},
            {"from": "review", "to": "pack", "type": "conditional", "when": "adopt"},
            {"from": "review", "to": "draft", "type": "fallback", "when": "revise"},
            {"from": "pack", "to": "end"},
        ],
    }
    spec.update(overrides)
    g = GraphDef.model_validate(spec)
    problems = g.validate_graph()
    assert not problems, problems
    return g


class _SlowAgent(FakeAgentExecutor):
    def __init__(self, store: AssetStore, delay: float = 0.05) -> None:
        super().__init__(store)
        self.delay = delay
        self.windows: dict[str, tuple[float, float]] = {}

    async def execute(self, node, state, inputs):
        t0 = time.perf_counter()
        await asyncio.sleep(self.delay)
        out = await super().execute(node, state, inputs)
        self.windows[node.id] = (t0, time.perf_counter())
        return out


def _build(graph: GraphDef, delay: float = 0.05):
    bus = EventBus()
    store = AssetStore()
    agent = _SlowAgent(store, delay)
    rt = GraphRuntime(bus, {NodeType.AGENT: agent, NodeType.TOOL: FakeToolExecutor(store)})
    state = rt.new_state(graph)
    state.slots["topic"] = store.create("露营").id
    return bus, store, agent, rt, state


async def test_分支并发执行并汇聚():
    graph = _graph()
    bus, store, agent, rt, state = _build(graph)
    started = time.perf_counter()
    state = await rt.run(graph, state)
    elapsed = time.perf_counter() - started

    assert state.status == "awaiting_review" and state.current == "review"
    assert {"plan", "draft", "images"} <= set(state.slots)
    assert state.checkpoints[-1].candidates == [state.slots["draft"], state.slots["images"]]
    # plan 串行 0.05 + 两个分支并发 0.05 ≈ 0.10，串行会是 0.15
    assert elapsed < 0.05 * 3 * 0.9, f"分支没并发：{elapsed:.3f}s"
    d0, d1 = agent.windows["draft"]
    i0, i1 = agent.windows["images"]
    assert d0 < i1 and i0 < d1, "两个分支的执行窗口要重叠"
    assert state.nodes_run == 5  # plan + draft + images + fan + merge
    assert state.node_status["fan"].value == "done" and state.node_status["merge"].value == "done"

    ps = [e for e in bus.history if e.type is EventType.PARALLEL_START]
    pe = [e for e in bus.history if e.type is EventType.PARALLEL_END]
    assert ps and ps[0].data["branches"] == ["draft", "images"]
    assert pe and pe[0].data["join"] == "merge" and pe[0].data["branches"] == 2


async def test_采纳后走完_打回只重跑那条分支():
    graph = _graph()
    bus, store, agent, rt, state = _build(graph, delay=0.001)
    state = await rt.run(graph, state)
    first_draft, first_images = state.slots["draft"], state.slots["images"]

    state = await rt.resume(graph, state, Decision.REVISE, reason="正文太平")
    assert state.status == "awaiting_review"
    assert agent.runs.count("draft") == 2, "打回的那条分支重跑"
    assert agent.runs.count("images") == 1, "兄弟分支不重跑"
    assert state.slots["draft"] != first_draft and state.slots["images"] == first_images
    assert len(state.checkpoints) == 2

    state = await rt.resume(graph, state, Decision.ADOPT)
    assert state.status == "done" and "final" in state.slots


async def test_分支里有human节点会挂起交给人():
    graph = _graph(
        nodes=[
            {"id": "plan", "type": "agent", "out_slots": ["plan"]},
            {"id": "fan", "type": "parallel"},
            {"id": "draft", "type": "agent", "in_slots": ["plan"], "out_slots": ["draft"]},
            {"id": "ask", "type": "human", "in_slots": ["plan"]},
            {"id": "merge", "type": "join"},
            {"id": "end", "type": "terminal"},
        ],
        edges=[
            {"from": "plan", "to": "fan"},
            {"from": "fan", "to": "draft"},
            {"from": "fan", "to": "ask"},
            {"from": "draft", "to": "merge"},
            {"from": "ask", "to": "merge"},
            {"from": "merge", "to": "end"},
        ],
    )
    bus, store, agent, rt, state = _build(graph, delay=0.001)
    state = await rt.run(graph, state)
    assert state.status == "halted"
    halt = [e for e in bus.history if e.type is EventType.GRAPH_HALT][-1]
    assert halt.data["reason"].startswith("parallel_failed") and "human" in halt.data["reason"]


async def test_分支不汇聚到同一join会挂起():
    graph = _graph(
        nodes=[
            {"id": "plan", "type": "agent", "out_slots": ["plan"]},
            {"id": "fan", "type": "parallel"},
            {"id": "a", "type": "agent", "in_slots": ["plan"], "out_slots": ["a"]},
            {"id": "b", "type": "agent", "in_slots": ["plan"], "out_slots": ["b"]},
            {"id": "j1", "type": "join"},
            {"id": "j2", "type": "join"},
            {"id": "end", "type": "terminal"},
        ],
        edges=[
            {"from": "plan", "to": "fan"},
            {"from": "fan", "to": "a"},
            {"from": "fan", "to": "b"},
            {"from": "a", "to": "j1"},
            {"from": "b", "to": "j2"},
            {"from": "j1", "to": "end"},
            {"from": "j2", "to": "end"},
        ],
    )
    bus, store, agent, rt, state = _build(graph, delay=0.001)
    state = await rt.run(graph, state)
    assert state.status == "halted"
    halt = [e for e in bus.history if e.type is EventType.GRAPH_HALT][-1]
    assert "同一个 join" in halt.data["reason"]


def test_校验parallel至少两条出边():
    spec = {
        "id": "bad",
        "entry": "fan",
        "nodes": [
            {"id": "fan", "type": "parallel"},
            {"id": "a", "type": "agent", "out_slots": ["a"]},
            {"id": "end", "type": "terminal"},
        ],
        "edges": [{"from": "fan", "to": "a"}, {"from": "a", "to": "end"}],
    }
    problems = GraphDef.model_validate(spec).validate_graph()
    assert any("至少要两条出边" in p for p in problems)
    with pytest.raises(AssertionError):
        _graph(edges=[{"from": "plan", "to": "fan"}, {"from": "fan", "to": "draft"}])

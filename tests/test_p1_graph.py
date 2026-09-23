"""P1 验收测试 —— Graph Runtime。

**验收标准不是「能不能生成一篇文案」，而是「打回能不能正确回退并重跑」。**
前者一次模型调用就够了，后者才是这套架构存在的理由。

模型调用打桩，图执行 / 快照 / 回退边 / 护栏 / 资产血缘全部跑真实代码。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.execution.graph.models import (
    Decision,
    EdgeType,
    GraphDef,
    NodeStatus,
    NodeType,
)
from aigc_agent.harness.execution.graph.runtime import GraphRuntime

GRAPH_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/aigc_agent/domain/pipeline/graphs/copy.yaml"
)


class FakeAgentExecutor:
    """按节点记录调用次数，产出可区分的资产。"""

    def __init__(self, store: AssetStore) -> None:
        self.store = store
        self.runs: list[str] = []

    async def execute(self, node, state, inputs):
        self.runs.append(node.id)
        n = self.runs.count(node.id)
        asset = self.store.create(
            content=f"{node.id} 的第 {n} 版内容",
            type_=AssetType.OUTLINE if node.id == "outline" else AssetType.TEXT,
            parents=list(inputs.values()),
            creator="model:fake",
        )
        return {node.out_slots[0]: asset.id} if node.out_slots else {}


class FakeToolExecutor:
    def __init__(self, store: AssetStore) -> None:
        self.store = store

    async def execute(self, node, state, inputs):
        asset = self.store.create(
            content=f"{node.id} 完成", parents=list(inputs.values()), creator=f"tool:{node.tool}"
        )
        return {node.out_slots[0]: asset.id} if node.out_slots else {}


def _build():
    graph = GraphDef.load(GRAPH_PATH)
    bus = EventBus(session_id="p1")
    store = AssetStore()
    agent = FakeAgentExecutor(store)
    rt = GraphRuntime(
        bus,
        {
            NodeType.AGENT: agent,
            NodeType.TOOL: FakeToolExecutor(store),
        },
    )
    return graph, rt, rt.new_state(graph), store, agent, bus


# ---------------------------------------------------------------- 图定义


def test_图定义能加载且静态校验通过():
    graph = GraphDef.load(GRAPH_PATH)
    assert graph.id == "copy"
    assert graph.entry == "outline"
    assert graph.validate_graph() == []

    # 两条回退边是这张图的核心
    fallbacks = [e for e in graph.edges if e.type is EdgeType.FALLBACK]
    assert {(e.from_, e.to, e.when) for e in fallbacks} == {
        ("review", "draft", "revise"),
        ("review", "outline", "reject"),
    }


def test_图定义写错会在加载时报而不是跑一半才炸(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "id: bad\nentry: a\nnodes:\n"
        "  - {id: a, type: agent}\n"
        "edges:\n  - {from: a, to: 不存在的节点}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as e:
        GraphDef.load(bad)
    assert "不存在的节点" in str(e.value)


# ---------------------------------------------------------------- 正常路径


async def test_跑到human节点会挂起而不是继续():
    graph, rt, state, store, agent, _ = _build()
    state = await rt.run(graph, state)

    assert state.status == "awaiting_review"
    assert state.current == "review"
    assert agent.runs == ["outline", "draft"]
    assert set(state.slots) == {"outline", "draft"}
    # finalize 绝不能在人点头之前跑
    assert state.node_status.get("finalize") is None


async def test_采纳后走到终点():
    graph, rt, state, store, agent, _ = _build()
    state = await rt.run(graph, state)
    state = await rt.resume(graph, state, Decision.ADOPT)

    assert state.status == "done"
    assert state.node_status["end"] is NodeStatus.DONE
    assert "final" in state.slots
    assert state.checkpoints[-1].decision is Decision.ADOPT


# ---------------------------------------------------------------- 回退边（P1 核心）


async def test_revise打回_退回draft重跑且大纲保留():
    graph, rt, state, store, agent, _ = _build()
    state = await rt.run(graph, state)
    outline_v1 = state.slots["outline"]
    draft_v1 = state.slots["draft"]

    state = await rt.resume(graph, state, Decision.REVISE, reason="开头太硬广了")

    # 又停在人审，说明确实回去重跑了一遍 draft
    assert state.status == "awaiting_review"
    assert agent.runs == ["outline", "draft", "draft"]

    # 大纲原样保留，正文是新的
    assert state.slots["outline"] == outline_v1
    assert state.slots["draft"] != draft_v1
    assert "第 2 版" in store.content(state.slots["draft"])


async def test_reject打回_退回outline且大纲也作废():
    graph, rt, state, store, agent, _ = _build()
    state = await rt.run(graph, state)
    outline_v1 = state.slots["outline"]

    state = await rt.resume(graph, state, Decision.REJECT, reason="选题方向不对")

    assert agent.runs == ["outline", "draft", "outline", "draft"]
    # 大纲重做了 —— reject 比 revise 退得更远
    assert state.slots["outline"] != outline_v1
    assert "outline 的第 2 版" in store.content(state.slots["outline"])


async def test_回退会恢复快照而不是叠加状态():
    """核心机制：退回某节点 = 恢复它执行之前的槽位状态。"""
    graph, rt, state, _, _, _ = _build()
    state = await rt.run(graph, state)
    assert len(state.snapshots) == 2  # outline, draft

    await rt.resume(graph, state, Decision.REJECT, reason="重来")
    # 退回 outline 前 → 快照截断到 0 → 重跑 outline+draft → 又是 2 个
    assert len(state.snapshots) == 2
    assert [s.node_id for s in state.snapshots] == ["outline", "draft"]


async def test_打回不填理由直接拒绝():
    """半自动模式下人会反复打回，不记原因 Agent 第二次会重犯。"""
    graph, rt, state, _, _, _ = _build()
    state = await rt.run(graph, state)

    with pytest.raises(ValueError) as e:
        await rt.resume(graph, state, Decision.REVISE, reason="   ")
    assert "必须填理由" in str(e.value)

    # 采纳则不需要理由
    state = await rt.resume(graph, state, Decision.ADOPT)
    assert state.status == "done"


async def test_打回理由被记录下来():
    graph, rt, state, _, _, _ = _build()
    state = await rt.run(graph, state)
    state = await rt.resume(graph, state, Decision.REVISE, reason="开头太硬广了")
    state = await rt.resume(graph, state, Decision.ADOPT)

    reasons = [c.reason for c in state.checkpoints if c.reason]
    assert reasons == ["开头太硬广了"]
    # P3 接 Memory Agent 后，这些理由要进 Brief 的 must_not 区
    assert state.checkpoints[0].decided_by == "human"


async def test_反复打回多轮仍然正确():
    graph, rt, state, store, agent, _ = _build()
    state = await rt.run(graph, state)
    for i in range(3):
        state = await rt.resume(graph, state, Decision.REVISE, reason=f"第{i}次不满意")
        assert state.status == "awaiting_review"

    assert agent.runs.count("draft") == 4  # 首次 + 3 次重写
    assert agent.runs.count("outline") == 1  # revise 不动大纲
    state = await rt.resume(graph, state, Decision.ADOPT)
    assert state.status == "done"


# ---------------------------------------------------------------- 护栏


async def test_节点数护栏会挂起交给人():
    graph, rt, state, _, _, _ = _build()
    graph.max_total_nodes = 3
    state = await rt.run(graph, state)
    for _ in range(5):
        if state.status != "awaiting_review":
            break
        state = await rt.resume(graph, state, Decision.REVISE, reason="再改")

    assert state.status == "halted"  # 不静默继续，也不静默降级


async def test_状态不对时resume会报错():
    graph, rt, state, _, _, _ = _build()
    with pytest.raises(RuntimeError) as e:
        await rt.resume(graph, state, Decision.ADOPT)
    assert "不在等待人审" in str(e.value)


# ---------------------------------------------------------------- 资产血缘


async def test_资产血缘可回溯():
    graph, rt, state, store, _, _ = _build()
    state = await rt.run(graph, state)
    state = await rt.resume(graph, state, Decision.ADOPT)

    chain = store.lineage(state.slots["final"])
    assert [a.type for a in chain] == [AssetType.OUTLINE, AssetType.TEXT, AssetType.TEXT]
    # P4 起 finalize 是真打包（build_release_package），P1 时是 now 占位
    assert chain[-1].creator == "tool:build_release_package"


def test_revise产出新版本而不是覆盖():
    store = AssetStore()
    v1 = store.create("第一版", summary="v1", creator="model:fake")
    v2 = store.revise(v1.id, "第二版", creator="model:fake")

    assert v2.version == 2
    assert v2.parent_ids == [v1.id]
    assert store.content(v1.id) == "第一版"  # 原版还在，可对比可回退
    assert len(store.lineage(v2.id)) == 2


def test_资产进上下文只带引用和摘要():
    """M3 原则：上下文放引用不放内容。"""
    store = AssetStore()
    a = store.create("正" * 5000, summary="一篇很长的稿子")
    brief = a.brief()
    assert a.id in brief and "一篇很长的稿子" in brief
    assert len(brief) < 100  # 不管内容多长，进上下文的都是这么点


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

"""P1 验收测试 —— Router 分派 + Memory Store 落库。

Router 的验收点：**规则能判的绝不调模型。**
Memory 的验收点：**打回理由必须落库**——这类数据补不回来。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aigc_agent.capabilities.memory.recorder import RejectionRecorder
from aigc_agent.capabilities.memory.store import (
    Category,
    Layer,
    MemoryStore,
    Polarity,
    Source,
)
from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.pipeline.registry import GraphRegistry
from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.execution.graph.models import Decision, GraphDef, NodeType
from aigc_agent.harness.execution.graph.runtime import GraphRuntime
from aigc_agent.harness.execution.router import ExecutionRouter, Route

from .test_p1_graph import FakeAgentExecutor, FakeToolExecutor  # noqa: TID252

GRAPH_PATH = Path(__file__).resolve().parents[1] / "src/aigc_agent/domain/pipeline/graphs/copy.yaml"


# ---------------------------------------------------------------- 图注册表


def test_注册表扫描目录并暴露触发词():
    r = GraphRegistry()
    r.load_all()
    assert r.errors == {}
    assert r.has("copy")
    assert "文案" in r.triggers()["copy"]


def test_单个图写错不影响其余图可用(tmp_path: Path):
    (tmp_path / "good.yaml").write_text(
        "id: good\nentry: a\nnodes:\n  - {id: a, type: terminal}\n", encoding="utf-8"
    )
    (tmp_path / "bad.yaml").write_text(
        "id: bad\nentry: 不存在\nnodes:\n  - {id: a, type: terminal}\n", encoding="utf-8"
    )
    r = GraphRegistry(tmp_path)
    r.load_all()
    assert r.has("good")
    assert "bad" in r.errors


# ---------------------------------------------------------------- Router


async def _router(model_fallback=None):
    reg = GraphRegistry()
    reg.load_all()
    return ExecutionRouter(EventBus(), catalog=reg, model_fallback=model_fallback)


async def test_触发词命中走graph():
    r = await _router()
    d = await r.route("帮我写一篇公众号长文，主题是露营")
    assert d.route is Route.GRAPH
    assert d.graph_id == "copy"
    assert d.by == "rule"


async def test_探索性提问走loop():
    r = await _router()
    for q in ["为什么这条视频火了", "帮我看看这个竞品", "什么是完播率"]:
        d = await r.route(q)
        assert d.route is Route.LOOP, q


async def test_探索信号优先于触发词():
    """「分析一下这篇文案」是问答，不是要生产一篇文案。"""
    r = await _router()
    d = await r.route("分析一下这篇文案为什么数据不好")
    assert d.route is Route.LOOP


async def test_有实例等人审时优先回到它():
    r = await _router()
    d = await r.route("随便说点什么", pending_run=("copy", "run123"))
    assert d.route is Route.GRAPH_RESUME
    assert (d.graph_id, d.run_id) == ("copy", "run123")


async def test_规则能判时不调用模型兜底():
    called = []

    async def fallback(text, triggers):
        called.append(text)
        return "copy"

    r = await _router(model_fallback=fallback)
    await r.route("帮我写一篇公众号推文")  # 触发词命中
    await r.route("为什么这条数据不好")  # 探索信号命中
    assert called == [], "规则已经能判了，不该再花一次模型调用"


async def test_规则判不了才问模型():
    async def fallback(text, triggers):
        return "copy"

    r = await _router(model_fallback=fallback)
    d = await r.route("露营装备种草")  # 无触发词、无探索信号
    assert d.route is Route.GRAPH and d.by == "model"


async def test_模型返回未知图时不采信():
    async def fallback(text, triggers):
        return "根本不存在的图"

    r = await _router(model_fallback=fallback)
    d = await r.route("露营装备种草")
    assert d.route is Route.LOOP  # 兜底，不是盲信模型


async def test_无匹配时不会自动编排新图():
    """动态生成图必须人确认后才执行，不能因为一句没匹配的话就悄悄建流水线。"""
    r = await _router()
    d = await r.route("嗯")
    assert d.route is Route.LOOP
    assert d.route is not Route.PROPOSE_GRAPH


async def test_路由决策会发事件():
    reg = GraphRegistry()
    reg.load_all()
    bus = EventBus()
    r = ExecutionRouter(bus, catalog=reg)
    await r.route("写一篇推文")
    ev = [e for e in bus.history if e.type is EventType.ROUTE]
    assert len(ev) == 1 and ev[0].data["graph"] == "copy"


# ---------------------------------------------------------------- Memory Store


def test_打回理由落库带正确的来源与极性():
    store = MemoryStore()
    m = store.record_rejection(
        reason="开头太硬广了", node_id="review", target_node="draft", run_id="r1", project_id="p1"
    )
    assert m.layer is Layer.PROJECT
    assert m.source is Source.HUMAN  # 人明确说的，最高可信
    assert m.confidence == 1.0
    assert m.weight > 1.0  # 打回理由权重高于一般记忆
    assert "开头太硬广了" in m.content

    kw = m.keywords[0]
    assert kw.polarity is Polarity.NEGATIVE
    assert kw.category is Category.CONSTRAINT
    assert kw.origin_quote == "开头太硬广了"  # ← 原话，比孤立词管用
    assert m.origin_ref == "run:r1#review->draft"


def test_关键词召回按权重排序():
    store = MemoryStore()
    store.record_rejection(reason="开头太硬广", node_id="review", target_node="draft", run_id="r1")
    store.record_rejection(
        reason="结尾没有引导", node_id="review", target_node="draft", run_id="r1"
    )
    store.record_rejection(
        reason="大纲结构乱", node_id="review", target_node="outline", run_id="r1"
    )

    hits = store.recall(["draft"])
    assert len(hits) == 2
    assert all(m.origin_ref.endswith("draft") for m in hits)
    assert all(m.hit_count == 1 for m in hits)


def test_brief把打回理由放进避雷区():
    store = MemoryStore()
    store.record_rejection(
        reason="开头太硬广了", node_id="review", target_node="draft", run_id="r1"
    )
    brief = store.render_brief(store.recall(["draft"]))
    assert "避雷" in brief and "不要重犯" in brief
    assert "开头太硬广了" in brief


def test_落盘后能重新加载(tmp_path: Path):
    s1 = MemoryStore(tmp_path)
    s1.record_rejection(reason="语气太说教", node_id="review", target_node="draft", run_id="r1")
    assert len(s1) == 1

    s2 = MemoryStore(tmp_path)  # 新进程重新起
    assert len(s2) == 1
    assert s2.recall(["draft"])[0].content.endswith("语气太说教")


def test_坏数据不影响整个库起来(tmp_path: Path):
    (tmp_path / "mem_broken.json").write_text("{不是json", encoding="utf-8")
    s = MemoryStore(tmp_path)
    s.record_rejection(reason="ok", node_id="review", target_node="draft", run_id="r1")
    assert len(s) == 1


# ---------------------------------------------------------------- 端到端串联


async def test_图打回自动落库_采纳不落库():
    """P1 的关键闭环：GraphRuntime(L0) 发事件 → Recorder(L1) 写库。

    L0 完全不知道有记忆这回事，靠事件总线解耦。
    """
    graph = GraphDef.load(GRAPH_PATH)
    bus = EventBus()
    assets = AssetStore()
    memories = MemoryStore()
    RejectionRecorder(memories, project_id="proj1").attach(bus)

    rt = GraphRuntime(
        bus,
        {NodeType.AGENT: FakeAgentExecutor(assets), NodeType.TOOL: FakeToolExecutor(assets)},
    )
    state = rt.new_state(graph)

    state = await rt.run(graph, state)
    state = await rt.resume(graph, state, Decision.REVISE, reason="开头太硬广了")
    state = await rt.resume(graph, state, Decision.REJECT, reason="选题方向就不对")
    state = await rt.resume(graph, state, Decision.ADOPT)

    assert state.status == "done"

    recorded = memories.all(layer=Layer.PROJECT, project_id="proj1")
    assert len(recorded) == 2, "两次打回都要落库，采纳不落"
    contents = " ".join(m.content for m in recorded)
    assert "开头太硬广了" in contents
    assert "选题方向就不对" in contents
    assert all(m.source is Source.HUMAN for m in recorded)


async def test_落库的理由能被召回并进brief():
    """下一轮生成时，这些理由要能拿回来 —— 否则 Agent 会重犯。"""
    graph = GraphDef.load(GRAPH_PATH)
    bus = EventBus()
    memories = MemoryStore()
    RejectionRecorder(memories).attach(bus)
    assets = AssetStore()

    rt = GraphRuntime(
        bus,
        {NodeType.AGENT: FakeAgentExecutor(assets), NodeType.TOOL: FakeToolExecutor(assets)},
    )
    state = await rt.run(graph, rt.new_state(graph))
    await rt.resume(graph, state, Decision.REVISE, reason="别用感叹号")

    brief = memories.render_brief(memories.recall(["draft"]))
    assert "别用感叹号" in brief


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

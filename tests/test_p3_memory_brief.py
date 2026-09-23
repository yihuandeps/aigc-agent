"""M8.2 完整模式验收 —— Memory Brief。

P3 验收原话：「节点启动前能拿到 Memory Brief；打回理由不再重犯」。

盯五件事：
  1. 四区分类靠已有字段就能判，不调模型
  2. 推测（inferred）不进硬约束、不自动晋升
  3. 避雷不做相关性过滤 —— 漏一条就是重犯一次
  4. 矛盾暴露给人，不裁决
  5. 完整模式走子代理，模型删了避雷不采信；坏输出退回规则版
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from aigc_agent.capabilities.memory.agent import CONSOLIDATE_DEF, MemoryAgent
from aigc_agent.capabilities.memory.brief import INFERRED_TAG, MemoryBrief, build_brief
from aigc_agent.capabilities.memory.store import (
    Category,
    Keyword,
    Layer,
    Memory,
    MemoryStore,
    Polarity,
    Source,
)
from aigc_agent.capabilities.subagents import SubAgentRunner
from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.pipeline.executors import AgentNodeExecutor
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.execution.graph.models import GraphNode, GraphState, NodeType
from aigc_agent.harness.model.gateway import ModelResponse, Usage
from aigc_agent.harness.permission.gate import PermissionGate
from aigc_agent.harness.tools.builtin import builtin
from aigc_agent.harness.tools.dispatcher import ToolDispatcher
from aigc_agent.harness.tools.registry import ToolRegistry


def _mem(
    content: str,
    terms: list[str],
    *,
    layer: Layer = Layer.PROJECT,
    source: Source = Source.INFERRED,
    polarity: Polarity = Polarity.NEUTRAL,
    category: Category = Category.CONSTRAINT,
    project: str = "p1",
    quote: str = "",
    weight: float = 1.0,
) -> Memory:
    return Memory(
        layer=layer,
        content=content,
        keywords=[
            Keyword(term=t, polarity=polarity, category=category, origin_quote=quote)
            for t in terms
        ],
        source=source,
        project_id=project,
        weight=weight,
    )


def _store() -> MemoryStore:
    s = MemoryStore()
    s.put(_mem("这个号不用感叹号", ["感叹号"], layer=Layer.ACCOUNT, source=Source.HUMAN,
               polarity=Polarity.NEGATIVE, quote="别用感叹号", project=""))
    s.put(_mem("每条开头必须带产品卖点", ["卖点"], layer=Layer.ACCOUNT, source=Source.HUMAN,
               polarity=Polarity.POSITIVE, project=""))
    s.record_rejection("开头太硬广了", node_id="review", run_id="r1",
                       target_node="draft", project_id="p1")
    s.put(_mem("时长控制在 30 秒内", ["时长"], polarity=Polarity.POSITIVE))
    s.put(_mem("喜欢口语化的开头", ["口语化"], polarity=Polarity.POSITIVE,
               category=Category.PREFERENCE))
    s.put(_mem("参考上次爆款 as_1234567890", ["爆款"], category=Category.TOPIC))
    s.put(_mem("别的项目的打回", ["其他"], polarity=Polarity.NEGATIVE, project="p2"))
    return s


# ---------------------------------------------------------------- 规则版


def test_四区分类():
    b = build_brief(_store(), "p1")
    assert "每条开头必须带产品卖点" in b.must
    assert any("感叹号" in x for x in b.must_not)
    assert any("开头太硬广了" in x for x in b.must_not), "打回理由必须在避雷区"
    assert any(x.startswith("时长控制") for x in b.should)
    assert any("口语化" in x for x in b.should)
    assert b.refs == ["as_1234567890"]
    everything = " ".join(b.must + b.must_not + b.should + b.conflicts)
    assert "别的项目" not in everything, "其它项目的记忆不该混进来"
    assert b.sources["must_not"], "每条可回溯到记忆 id"


def test_推测不进硬约束且带标记():
    b = build_brief(_store(), "p1")
    assert not any("时长" in x for x in b.must)
    hit = next(x for x in b.should if "时长" in x)
    assert hit.endswith(INFERRED_TAG)


def test_避雷不做相关性过滤_建议才过滤():
    b = build_brief(_store(), "p1", topic="露营装备")
    assert any("开头太硬广了" in x for x in b.must_not), "主题无关也必须带上打回理由"
    assert any("感叹号" in x for x in b.must_not)
    assert not any("口语化" in x for x in b.should), "偏好按主题过滤"


def test_矛盾暴露不裁决():
    s = _store()
    s.put(_mem("这条要用感叹号制造情绪", ["感叹号"], source=Source.HUMAN,
               polarity=Polarity.POSITIVE, quote="这条多用感叹号"))
    b = build_brief(s, "p1")
    assert b.conflicts and "感叹号" in b.conflicts[0]
    assert "要求" in b.conflicts[0] and "禁止" in b.conflicts[0]
    # 两边都还在，没有被谁覆盖谁
    assert any("感叹号" in x for x in b.must_not)


def test_render与pin_text():
    b = build_brief(_store(), "p1")
    text = b.render()
    assert "### 避雷" in text and "### 必须遵守" in text and "### 参考资产" in text
    pin = b.pin_text()
    assert pin.startswith("## 记忆简报") and "禁止/避雷" in pin and "必须：" in pin
    assert "口语化" not in pin, "pin 区只放硬约束，建议不进"
    assert MemoryBrief().empty and MemoryBrief().render() == "" and MemoryBrief().pin_text() == ""


def test_空库出空简报():
    assert build_brief(MemoryStore(), "p1").empty


def test_近似重复只留一条():
    """按批提取常把同一句话记成措辞略有出入的两条，简报里不该各占一行。"""
    s = MemoryStore()
    human = Source.HUMAN  # 避雷只收人说的（推测的只进建议，见下一条）
    s.put(_mem("科技短视频开头不要写成硬广式的震惊腔调，用户明确讨厌这种风格", ["硬广"],
               polarity=Polarity.NEGATIVE, source=human))
    s.put(_mem("科技短视频开头不要写成硬广式震惊腔调", ["硬广"], polarity=Polarity.NEGATIVE,
               source=human))
    s.put(_mem("结尾不要喊口号", ["口号"], polarity=Polarity.NEGATIVE, source=human))
    b = build_brief(s, "p1")
    assert len(b.must_not) == 2
    assert any("口号" in x for x in b.must_not)


def test_推测出来的避雷只进建议():
    """2026-09-23 审查：「需逐镜 gen_video 传 image_urls（推测）」进了 must_not，每轮 pin 着，
    和「镜头只能走 drama_render_shots」正面冲突。"""
    s = MemoryStore()
    s.put(_mem("不要用 drama_render_shots，要逐镜 gen_video", ["渲染"],
               polarity=Polarity.NEGATIVE))
    s.put(_mem("开头不要太平", ["开头"], polarity=Polarity.NEGATIVE, source=Source.HUMAN))
    b = build_brief(s, "p1")
    assert b.must_not == ["开头不要太平"]
    assert any(x.startswith("避免：") and "（推测）" in x for x in b.should)
    assert "drama_render_shots" not in b.pin_text(), "推测的不 pin"


def test_标点与修饰不同仍算同一条():
    s = MemoryStore()
    s.put(_mem("科技短视频文案开头不要写成硬广式的“震惊！”腔调，用户明确讨厌这种风格", ["硬广"],
               polarity=Polarity.NEGATIVE, source=Source.HUMAN))
    s.put(_mem('科技短视频开头不要写成硬广式"震惊！"腔调', ["硬广"], polarity=Polarity.NEGATIVE,
               source=Source.HUMAN))
    assert len(build_brief(s, "p1").must_not) == 1


def test_只差数字的不能合并():
    """「30 秒」和「60 秒」是真矛盾，合并掉就把人该看的信号吞了。"""
    s = MemoryStore()
    s.put(_mem("时长控制在 30 秒以内", ["时长"], source=Source.HUMAN, polarity=Polarity.POSITIVE))
    s.put(_mem("时长控制在 60 秒以内", ["时长"], source=Source.HUMAN, polarity=Polarity.NEGATIVE))
    b = build_brief(s, "p1")
    assert b.conflicts and "时长" in b.conflicts[0]
    assert any("30" in x for x in b.must) and any("60" in x for x in b.must_not)


def test_同一句话被打成两种极性不算矛盾():
    s = MemoryStore()
    s.put(_mem("时长控制在 30 秒以内，超了完播率会掉", ["完播率"], polarity=Polarity.POSITIVE))
    s.put(_mem("时长控制在 30 秒以内，超了完播率会掉。", ["完播率"], polarity=Polarity.NEGATIVE))
    assert build_brief(s, "p1").conflicts == []


def test_晋升只能由人或数据触发():
    s = _store()
    target = next(m for m in s.all() if "时长" in m.content)
    with pytest.raises(ValueError):
        s.promote(target.id, by="model")
    assert build_brief(s, "p1").must == ["每条开头必须带产品卖点"]
    s.promote(target.id, by="human")
    assert target.layer is Layer.ACCOUNT and target.source is Source.HUMAN
    assert "时长控制在 30 秒内" in build_brief(s, "p1").must


def test_作废后不再进简报():
    s = _store()
    rej = next(m for m in s.all() if "打回" in m.content)
    s.forget(rej.id)
    assert not any("开头太硬广了" in x for x in build_brief(s, "p1").must_not)
    with pytest.raises(KeyError):
        s.get("mem_nope")


# ---------------------------------------------------------------- 完整模式（子代理）


class _Gw:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    async def chat(self, role, messages, tools=None, **kw):
        self.calls.append({"role": role, "messages": messages})
        return ModelResponse(text=self.text, usage=Usage(2, 2))


async def _runner(gw: _Gw) -> SubAgentRunner:
    bus = EventBus()
    reg = ToolRegistry(bus)
    await reg.refresh()
    return SubAgentRunner(gw, reg, bus)


async def test_没有运行器时consolidate退回规则版():
    agent = MemoryAgent(None, _store(), project_id="p1")
    base = agent.brief()
    assert not base.empty
    assert await agent.consolidate() == base


async def test_consolidate走子代理_避雷只增不减():
    gw = _Gw(
        '{"must": ["开头带卖点"], "must_not": [], "should": ["口语化"], '
        '"conflicts": ["待定"]}'
    )
    agent = MemoryAgent(None, _store(), project_id="p1", runner=await _runner(gw))
    base = agent.brief()
    b = await agent.consolidate()
    assert gw.calls[0]["role"] == CONSOLIDATE_DEF.role == "memory_consolidate"
    assert "记忆整理员" in gw.calls[0]["messages"][0]["content"]
    assert b.must == ["开头带卖点"], "措辞合并采信"
    assert b.must_not == base.must_not, "模型删掉了避雷，不采信，退回规则版的避雷"
    assert b.should == ["口语化"] and b.conflicts == ["待定"]
    assert b.refs == base.refs


async def test_consolidate输出坏了退回规则版():
    agent = MemoryAgent(None, _store(), project_id="p1", runner=await _runner(_Gw("垃圾")))
    assert await agent.consolidate() == agent.brief()


def test_整理子代理是无状态只读的():
    assert CONSOLIDATE_DEF.stateless
    assert CONSOLIDATE_DEF.tools == []
    assert "must_not" in CONSOLIDATE_DEF.output_schema["required"]


# ---------------------------------------------------------------- 接线


def test_主循环每轮前刷新简报并pin():
    from aigc_agent.app import Agent

    assert "prepare_turn" in inspect.getsource(Agent.chat)
    src = inspect.getsource(Agent.prepare_turn)
    assert "brief(" in src and "BRIEF_PIN" in src and "pre_input" in src


class _NodeGw:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def chat(self, role, messages, tools=None, **kw):
        self.messages = messages
        return ModelResponse(text="第二版正文", usage=Usage(2, 2))


async def test_节点启动前拿到brief并pin进节点内loop():
    store = AssetStore()
    bus = EventBus()
    reg = ToolRegistry(bus)
    reg.register(builtin)
    await reg.refresh()
    gw = _NodeGw()
    seen: list[str] = []

    async def brief_fn(node, state):
        seen.append(node.id)
        return build_brief(_store(), "p1", stage=node.stage)

    ex = AgentNodeExecutor(
        gw, reg, ToolDispatcher(reg, PermissionGate(bus), bus), bus, store,  # type: ignore[arg-type]
        brief_fn=brief_fn,
    )
    node = GraphNode(id="draft", type=NodeType.AGENT, stage="正文", out_slots=["draft"])
    state = GraphState(graph_id="copy", run_id="r1")
    out = await ex.execute(node, state, {})

    assert seen == ["draft"]
    assert store.content(out["draft"]) == "第二版正文"
    user = gw.messages[-1]
    assert user["role"] == "user" and "## 记忆简报" in user["content"]
    assert "开头太硬广了" in user["content"], "打回理由进了节点契约"
    pinned = gw.messages[-2]
    assert pinned["role"] == "system" and "禁止/避雷" in pinned["content"], "must_not 贴着当前输入"


async def test_brief拿不到不阻塞节点():
    store = AssetStore()
    bus = EventBus()
    reg = ToolRegistry(bus)
    await reg.refresh()

    async def broken(node, state):
        raise RuntimeError("记忆库炸了")

    ex = AgentNodeExecutor(
        _NodeGw(), reg, ToolDispatcher(reg, PermissionGate(bus), bus), bus, store,  # type: ignore[arg-type]
        brief_fn=broken,
    )
    node = GraphNode(id="draft", type=NodeType.AGENT, out_slots=["draft"])
    out = await ex.execute(node, GraphState(graph_id="g", run_id="r"), {})
    assert "draft" in out

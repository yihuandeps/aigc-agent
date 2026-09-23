"""agentic 模式的「单步重跑」验收。

P2 验收标准原话：「agentic 模式下也能看到进度图并单步重跑」。
进度图（/trace、/mermaid）早就有；「单步重跑」之前只有 trace.rollback_plan()
算出该作废什么，**没有任何入口真的执行它**。这里验的是 rollback_to()：

  1. 痕迹里下游调用标作废（只标记不删除）
  2. 资产不删，血缘还在
  3. 一段说明 pin 进 pre_input 位 —— 下一轮模型看得到，且只看一次
"""

from __future__ import annotations

from aigc_agent.app import ROLLBACK_PIN, rollback_to
from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.functions.content import ContentFunctions
from aigc_agent.harness.context.assembler import ContextAssembler
from aigc_agent.harness.context.window import ShortTermMemory
from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.execution.loop import LoopRuntime
from aigc_agent.harness.execution.trace import ExecutionTrace
from aigc_agent.harness.model.gateway import ModelResponse, ToolCall, Usage
from aigc_agent.harness.permission.gate import PermissionGate
from aigc_agent.harness.tools.dispatcher import ToolDispatcher
from aigc_agent.harness.tools.registry import ToolRegistry


def _tc(cid: str, name: str, args: str) -> ToolCall:
    return ToolCall(id=cid, name=name, arguments=args)


class _Gw:
    """存 v1 → 基于 v1 改出 v2 → 基于 v2 改出 v3 → 结束。"""

    def __init__(self, ids: list[str]) -> None:
        self.n = 0
        self.ids = ids

    async def chat(self, role, messages, tools=None, **kw):
        self.n += 1
        if self.n == 1:
            return ModelResponse(
                tool_calls=[_tc("c1", "save_draft", '{"content":"v1","kind":"copy"}')],
                usage=Usage(1, 1),
            )
        if self.n in (2, 3):
            parent = self.ids[-1]
            return ModelResponse(
                tool_calls=[
                    _tc(
                        f"c{self.n}",
                        "save_draft",
                        f'{{"content":"v{self.n}","kind":"copy","parent_id":"{parent}"}}',
                    )
                ],
                usage=Usage(1, 1),
            )
        return ModelResponse(text="三版都存了", usage=Usage(1, 1))


async def _run():
    ids: list[str] = []
    bus = EventBus()
    assets = AssetStore()
    trace = ExecutionTrace()
    trace.attach(bus)
    registry = ToolRegistry(bus)
    registry.register(ContentFunctions(assets))
    await registry.refresh()
    memory = ShortTermMemory()
    assembler = ContextAssembler(bus)
    loop = LoopRuntime(
        gateway=_Gw(ids),  # type: ignore[arg-type]
        registry=registry,
        dispatcher=ToolDispatcher(registry, PermissionGate(bus), bus),
        assembler=assembler,
        memory=memory,
        bus=bus,
    )
    orig = loop.dispatcher.run

    async def spy(calls):
        out = await orig(calls)
        for _, res in out:
            if res.asset_ref:
                ids.append(res.asset_ref)
        return out

    loop.dispatcher.run = spy  # type: ignore[method-assign]
    await loop.run_turn("写三版")
    assert len(ids) == 3
    return ids, bus, assets, trace, memory, assembler, loop


async def test_回退到中间版本_下游作废但不删():
    ids, bus, assets, trace, memory, _, _ = await _run()
    v1, v2, v3 = ids

    plan = await rollback_to(v2, trace=trace, memory=memory, assets=assets, bus=bus)
    assert plan["ok"]
    assert plan["supersede"] == ["c3"], "只有产出 v3 的那次调用作废"
    assert plan["lost_assets"] == [v3]
    assert trace.node("c3").superseded and not trace.node("c2").superseded

    # 资产一个都不删，血缘完整
    assert assets.get(v3).parent_ids == [v2]
    assert [a.id for a in assets.lineage(v3)] == [v1, v2, v3]

    events = [e for e in bus.history if e.type is EventType.TRACE_ROLLBACK]
    assert events and events[-1].data["asset"] == v2 and events[-1].data["lost_assets"] == [v3]


async def test_回退说明pin进下一轮上下文且只看一次():
    ids, bus, assets, trace, memory, assembler, loop = await _run()
    v1, v2, v3 = ids
    plan = await rollback_to(v2, trace=trace, memory=memory, assets=assets, bus=bus)

    note = plan["note"]
    assert v2 in note and v3 in note and f"parent_id={v2}" in note
    pins = memory.pins_at("pre_input")
    assert [p.key for p in pins] == [ROLLBACK_PIN]

    # 装配时它贴在当前输入之前（注意力最强的位置）
    turn = memory.new_turn()
    turn.messages.append({"role": "user", "content": "继续"})
    messages = await assembler.assemble(memory, turn)
    assert messages[-1]["content"] == "继续"
    assert messages[-2]["role"] == "system" and "已回退到" in messages[-2]["content"]

    # 看过一次就该撤掉 —— Agent.chat() 在每轮结束时 unpin
    memory.unpin(ROLLBACK_PIN)
    assert memory.pins_at("pre_input") == []


async def test_回退到最新版本没有东西可作废():
    ids, bus, assets, trace, memory, _, _ = await _run()
    plan = await rollback_to(ids[-1], trace=trace, memory=memory, assets=assets, bus=bus)
    assert plan["ok"] and plan["supersede"] == [] and plan["lost_assets"] == []
    assert "已作废" not in plan["note"]


async def test_回退不存在的资产报错且不留pin():
    _, bus, assets, trace, memory, _, _ = await _run()
    plan = await rollback_to("as_ffffffffff", trace=trace, memory=memory, assets=assets, bus=bus)
    assert not plan["ok"] and "没有调用产出过" in plan["error"]
    assert memory.pins == {}

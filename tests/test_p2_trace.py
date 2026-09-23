"""执行痕迹 → DAG 验收。

补齐 agentic 模式相对图模式缺的两样：进度可见 + 单步定位/回退。
判据：**跑完之后，能不能还原出一张和图模式等价的 DAG。**
"""

from __future__ import annotations

import pytest

from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.functions.content import ContentFunctions
from aigc_agent.harness.context.assembler import ContextAssembler
from aigc_agent.harness.context.window import ShortTermMemory
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.execution.loop import LoopRuntime, StopReason
from aigc_agent.harness.execution.trace import ExecutionTrace, extract_assets
from aigc_agent.harness.model.gateway import ModelResponse, ToolCall, Usage
from aigc_agent.harness.permission.gate import PermissionGate
from aigc_agent.harness.tools.dispatcher import ToolDispatcher
from aigc_agent.harness.tools.registry import ToolRegistry


def _tc(cid, name, args="{}"):
    return ToolCall(id=cid, name=name, arguments=args)


# ---------------------------------------------------------------- 资产提取


def test_从任意入参里捞出资产id():
    assert extract_assets({"parent_id": "as_0123456789"}) == ["as_0123456789"]
    assert extract_assets({"asset_ids": ["as_aaaaaaaaaa", "as_bbbbbbbbbb"]}) == [
        "as_aaaaaaaaaa",
        "as_bbbbbbbbbb",
    ]
    # 嵌套 + 去重保序
    assert extract_assets(
        {"a": {"b": ["as_cccccccccc"]}, "c": "见 as_cccccccccc 和 as_dddddddddd"}
    ) == ["as_cccccccccc", "as_dddddddddd"]
    # 不误伤
    assert extract_assets({"text": "as_短的 asxx as_ZZZZZZZZZZ"}) == []


# ---------------------------------------------------------------- 端到端


class _Gw:
    """三步：存稿 → 请人审 → 基于原稿改写。"""

    def __init__(self, holder: dict[str, str]) -> None:
        self.n = 0
        self.holder = holder

    async def chat(self, role, messages, tools=None, **kw):
        self.n += 1
        if self.n == 1:
            return ModelResponse(
                tool_calls=[
                    _tc("c1", "save_draft", '{"content":"初稿正文","kind":"copy","summary":"初稿"}')
                ],
                usage=Usage(5, 5),
            )
        if self.n == 2:
            aid = self.holder["id"]
            return ModelResponse(
                tool_calls=[
                    _tc(
                        "c2",
                        "request_review",
                        f'{{"asset_ids":["{aid}"],"question":"行吗","stage":"正文"}}',
                    )
                ],
                usage=Usage(5, 5),
            )
        if self.n == 3:
            return ModelResponse(
                tool_calls=[
                    _tc(
                        "c3",
                        "save_draft",
                        f'{{"content":"第二版","kind":"copy","parent_id":"{self.holder["id"]}"}}',
                    )
                ],
                usage=Usage(5, 5),
            )
        return ModelResponse(text="改好了", usage=Usage(5, 5))


async def _run():
    holder: dict[str, str] = {}
    bus = EventBus()
    assets = AssetStore()
    trace = ExecutionTrace()
    trace.attach(bus)

    registry = ToolRegistry(bus)
    registry.register(ContentFunctions(assets))
    await registry.refresh()

    loop = LoopRuntime(
        gateway=_Gw(holder),  # type: ignore[arg-type]
        registry=registry,
        dispatcher=ToolDispatcher(registry, PermissionGate(bus), bus),
        assembler=ContextAssembler(bus),
        memory=ShortTermMemory(),
        bus=bus,
    )

    orig = loop.dispatcher.run

    async def spy(calls):
        out = await orig(calls)
        for _, res in out:
            if res.asset_ref and "id" not in holder:
                holder["id"] = res.asset_ref
        return out

    loop.dispatcher.run = spy  # type: ignore[method-assign]

    r1 = await loop.run_turn("写一段正文")
    return loop, trace, assets, holder, r1


async def test_痕迹还原出调用序列():
    loop, trace, assets, holder, r1 = await _run()
    assert r1.stop_reason is StopReason.AWAITING_REVIEW

    tools = [n.tool for n in trace.nodes]
    assert tools == ["save_draft", "request_review"]

    # request_review 被识别为 checkpoint（因为它 suspend 了）
    assert trace.nodes[0].kind == "call"
    assert trace.nodes[1].kind == "checkpoint"
    assert trace.nodes[1].stage == "正文"


async def test_按资产依赖自动连边():
    """这是 DAG 的核心：谁产出、谁消费，边自己长出来。"""
    loop, trace, assets, holder, _ = await _run()
    aid = holder["id"]

    # save_draft 产出 aid，request_review 消费 aid → 应有一条带资产标签的边
    assert len(trace.edges) == 1
    e = trace.edges[0]
    assert (e.src, e.dst, e.asset) == ("c1", "c2", aid)

    assert trace.producer_of(aid).tool == "save_draft"


async def test_人审决策贴回痕迹():
    loop, trace, assets, holder, _ = await _run()
    await loop.resume_turn("revise", reason="开头太硬广了")

    cp = next(n for n in trace.nodes if n.kind == "checkpoint")
    assert cp.decision == "revise"
    assert cp.reason == "开头太硬广了"

    # 改写那次调用也进了痕迹，并挂在原稿下面
    tools = [n.tool for n in trace.nodes]
    assert tools == ["save_draft", "request_review", "save_draft"]
    assert any(e.src == "c1" and e.dst == "c3" for e in trace.edges)


async def test_下游依赖可追溯():
    loop, trace, assets, holder, _ = await _run()
    await loop.resume_turn("revise", reason="重写")

    downstream = {n.id for n in trace.descendants("c1")}
    assert downstream == {"c2", "c3"}


# ---------------------------------------------------------------- 回退


async def test_回退计划算出该作废哪些调用():
    """agentic 的单步重跑：选一份资产版本当新起点，之后的一切作废。"""
    loop, trace, assets, holder, _ = await _run()
    await loop.resume_turn("revise", reason="重写")

    plan = trace.rollback_plan(holder["id"])
    assert plan["ok"]
    assert plan["restart_from"] == "c1"
    assert set(plan["supersede"]) == {"c2", "c3"}
    assert plan["lost_assets"]  # c3 产出的第二版会丢


async def test_回退只标记不删除():
    """走过的弯路留在痕迹里 —— 复盘和记忆提炼都要用。"""
    loop, trace, assets, holder, _ = await _run()
    await loop.resume_turn("revise", reason="重写")

    before = len(trace.nodes)
    trace.apply_rollback(holder["id"])

    assert len(trace.nodes) == before, "节点不该被删掉"
    assert trace.node("c3").superseded is True
    assert trace.node("c1").superseded is False
    assert trace.summary()["superseded"] == 2


async def test_回退不存在的资产会报错():
    _, trace, _, _, _ = await _run()
    plan = trace.rollback_plan("as_9999999999")
    assert not plan["ok"] and "没有调用产出过" in plan["error"]


# ---------------------------------------------------------------- 渲染


async def test_文本视图可读():
    loop, trace, assets, holder, _ = await _run()
    await loop.resume_turn("revise", reason="开头太硬广了")
    text = trace.to_text()

    assert "save_draft" in text
    assert "⏸" in text  # checkpoint 标记
    assert "人审(正文)：revise" in text
    assert "开头太硬广了" in text
    assert holder["id"] in text  # 资产流向看得见


async def test_mermaid输出即为自动生成的流程图():
    """跑完才有的图 —— 这就是「图作为输出」。"""
    loop, trace, assets, holder, _ = await _run()
    await loop.resume_turn("revise", reason="重写")
    trace.apply_rollback(holder["id"])

    m = trace.to_mermaid()
    assert m.startswith("flowchart TD")
    assert "c1[\"save_draft\"]" in m
    assert "人审" in m and "revise" in m
    assert f"c1 -- {holder['id']} --> c2" in m
    assert "class c3 dropped" in m  # 作废的用虚线标出
    assert "classDef dropped" in m


async def test_摘要统计():
    loop, trace, assets, holder, _ = await _run()
    await loop.resume_turn("adopt")
    s = trace.summary()

    assert s["calls"] == 2  # 两次 save_draft
    assert s["checkpoints"] == 1
    assert s["failed"] == 0
    assert s["assets_produced"] == 2


async def test_失败的调用也进痕迹():
    """走过的弯路要看得见，不能只记成功的。"""
    bus = EventBus()
    assets = AssetStore()
    trace = ExecutionTrace()
    trace.attach(bus)
    registry = ToolRegistry(bus)
    registry.register(ContentFunctions(assets))
    await registry.refresh()

    class Gw:
        def __init__(self):
            self.n = 0

        async def chat(self, role, messages, tools=None, **kw):
            self.n += 1
            if self.n == 1:
                return ModelResponse(
                    tool_calls=[_tc("x1", "read_asset", '{"asset_id":"as_0000000000"}')],
                    usage=Usage(1, 1),
                )
            return ModelResponse(text="那份不存在", usage=Usage(1, 1))

    loop = LoopRuntime(
        gateway=Gw(),  # type: ignore[arg-type]
        registry=registry,
        dispatcher=ToolDispatcher(registry, PermissionGate(bus), bus),
        assembler=ContextAssembler(bus),
        memory=ShortTermMemory(),
        bus=bus,
    )
    await loop.run_turn("读一下")

    assert len(trace.nodes) == 1
    assert trace.nodes[0].ok is False
    assert "✗" in trace.to_text()
    assert trace.summary()["failed"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

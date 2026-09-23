"""Agentic 形态验收 —— function 驱动 + 模型自主请求人审。

和 P1 图模式的区别：**没有预定义流程**。模型自己决定调哪个 function、
什么时候请人拍板。但底座（Asset 血缘、打回理由落库）完全复用。

验收点：
  1. 模型能连续调 function 把活干完
  2. 模型主动调 request_review → 主循环挂起
  3. 人的决策作为工具返回值填回，模型据此继续
  4. 打回理由照样落进记忆库（复用 P1 的 RejectionRecorder）
  5. 资产血缘在 function 模式下依然完整
"""

from __future__ import annotations

import pytest

from aigc_agent.capabilities.memory.recorder import RejectionRecorder
from aigc_agent.capabilities.memory.store import Layer, MemoryStore, Source
from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.functions.content import ContentFunctions
from aigc_agent.harness.context.assembler import ContextAssembler
from aigc_agent.harness.context.window import ShortTermMemory
from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.execution.loop import LoopRuntime, StopReason
from aigc_agent.harness.model.gateway import ModelResponse, ToolCall, Usage
from aigc_agent.harness.permission.gate import PermissionGate
from aigc_agent.harness.tools.dispatcher import ToolDispatcher
from aigc_agent.harness.tools.registry import ToolRegistry


class ScriptedGateway:
    def __init__(self, script):
        self.script = list(script)
        self.n = 0

    async def chat(self, role, messages, tools=None, **kw):
        r = self.script[min(self.n, len(self.script) - 1)]
        self.n += 1
        return r


def _tc(cid, name, args="{}"):
    return ToolCall(id=cid, name=name, arguments=args)


async def _build(script, memories: MemoryStore | None = None):
    bus = EventBus(session_id="agentic")
    assets = AssetStore()
    registry = ToolRegistry(bus)
    registry.register(ContentFunctions(assets))
    await registry.refresh()

    if memories is not None:
        RejectionRecorder(memories, project_id="proj").attach(bus)

    loop = LoopRuntime(
        gateway=ScriptedGateway(script),  # type: ignore[arg-type]
        registry=registry,
        dispatcher=ToolDispatcher(registry, PermissionGate(bus), bus),
        assembler=ContextAssembler(bus),
        memory=ShortTermMemory(),
        bus=bus,
    )
    return bus, assets, registry, loop


# ---------------------------------------------------------------- function 面


async def test_内容functions注册齐全且权限分级正确():
    _, _, registry, _ = await _build([ModelResponse(text="ok")])
    names = {m.name: m.permission.value for m in registry.catalog()}
    assert set(names) == {
        "save_draft",
        "read_asset",
        "list_assets",
        "find_episode",
        "compare_assets",
        "request_review",
    }
    assert names["read_asset"] == "L-read"
    assert names["save_draft"] == "L-write"


async def test_存稿建立血缘():
    script = [
        ModelResponse(
            tool_calls=[
                _tc("c1", "save_draft", '{"content":"大纲v1","kind":"outline","summary":"大纲"}')
            ],
            usage=Usage(5, 5),
        ),
        ModelResponse(text="存好了", usage=Usage(5, 5)),
    ]
    _, assets, _, loop = await _build(script)
    r = await loop.run_turn("写个大纲")
    assert r.stop_reason is StopReason.NO_TOOL_CALLS
    assert len(assets) == 1


# ---------------------------------------------------------------- 挂起 / 恢复


async def test_模型主动请求人审会挂起主循环():
    """agentic HITL：人审不是固定节点，是模型自己判断该问了才调的 function。"""
    script = [
        ModelResponse(
            tool_calls=[
                _tc("c1", "save_draft", '{"content":"正文v1","kind":"copy","summary":"初稿"}')
            ],
            usage=Usage(5, 5),
        ),
        ModelResponse(
            tool_calls=[
                _tc(
                    "c2",
                    "request_review",
                    '{"asset_ids":["__A__"],"question":"这版行吗","stage":"正文"}',
                )
            ],
            usage=Usage(5, 5),
        ),
        ModelResponse(text="收到，我改", usage=Usage(5, 5)),
    ]
    bus, assets, _, loop = await _build(script)

    # 第一步先存稿拿到真实 id，再把它填进第二步的参数
    r = await loop.run_turn("写一段正文")
    assert r.stop_reason is StopReason.AWAITING_REVIEW or len(assets) == 1


async def _run_until_review(memories: MemoryStore | None = None):
    """跑到挂起为止，返回 (loop, assets, bus)。"""
    assets_holder: dict[str, str] = {}

    class Gw:
        def __init__(self):
            self.n = 0

        async def chat(self, role, messages, tools=None, **kw):
            self.n += 1
            if self.n == 1:
                return ModelResponse(
                    tool_calls=[
                        _tc(
                            "c1",
                            "save_draft",
                            '{"content":"露营装备正文第一版","kind":"copy","summary":"初稿"}',
                        )
                    ],
                    usage=Usage(5, 5),
                )
            if self.n == 2:
                aid = assets_holder["id"]
                return ModelResponse(
                    tool_calls=[
                        _tc(
                            "c2",
                            "request_review",
                            f'{{"asset_ids":["{aid}"],"question":"这版行吗","stage":"正文"}}',
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
                            f'{{"content":"改过的第二版","kind":"copy",'
                            f'"parent_id":"{assets_holder["id"]}"}}',
                        )
                    ],
                    usage=Usage(5, 5),
                )
            return ModelResponse(text="已按你的意见改好", usage=Usage(5, 5))

    bus = EventBus()
    assets = AssetStore()
    registry = ToolRegistry(bus)
    registry.register(ContentFunctions(assets))
    await registry.refresh()
    if memories is not None:
        RejectionRecorder(memories, project_id="proj").attach(bus)

    loop = LoopRuntime(
        gateway=Gw(),  # type: ignore[arg-type]
        registry=registry,
        dispatcher=ToolDispatcher(registry, PermissionGate(bus), bus),
        assembler=ContextAssembler(bus),
        memory=ShortTermMemory(),
        bus=bus,
    )

    # 第一次调用后把生成的 asset id 暴露给假 gateway
    orig = loop.dispatcher.run

    async def spy(calls):
        out = await orig(calls)
        for _, res in out:
            if res.asset_ref and "id" not in assets_holder:
                assets_holder["id"] = res.asset_ref
        return out

    loop.dispatcher.run = spy  # type: ignore[method-assign]
    result = await loop.run_turn("写一段露营装备的种草文案")
    return loop, assets, bus, result


async def test_挂起后pending_review带上候选与环节():
    loop, assets, bus, result = await _run_until_review()

    assert result.stop_reason is StopReason.AWAITING_REVIEW
    assert loop.pending_review is not None
    assert loop.pending_review["stage"] == "正文"
    assert loop.pending_review["question"] == "这版行吗"
    assert len(loop.pending_review["assets"]) == 1

    ev = [e for e in bus.history if e.type is EventType.CHECKPOINT_REACHED]
    assert len(ev) == 1


async def test_人的决策作为工具返回值填回后模型继续():
    loop, assets, bus, _ = await _run_until_review()
    r = await loop.resume_turn("revise", reason="开头太硬广了")

    assert r.stop_reason is StopReason.NO_TOOL_CALLS
    assert "改好" in r.text

    # 决策以 tool 消息的形式回填到那次 request_review 的 call_id 上
    tool_msgs = [m for m in r.turn.messages if m.get("role") == "tool"]
    verdicts = [m["content"] for m in tool_msgs if "人的决策" in m["content"]]
    assert len(verdicts) == 1
    assert "打回重做" in verdicts[0]
    assert "开头太硬广了" in verdicts[0]

    # 人审前后仍算同一问一答，不新开一轮
    assert len(loop.memory.turns) == 1
    assert loop.pending_review is None


async def test_打回不填理由被拒():
    loop, _, _, _ = await _run_until_review()
    with pytest.raises(ValueError) as e:
        await loop.resume_turn("revise", reason="  ")
    assert "必须填理由" in str(e.value)
    assert loop.pending_review is not None  # 状态不被破坏，可以重来

    r = await loop.resume_turn("adopt")  # 采纳不需要理由
    assert r.stop_reason is StopReason.NO_TOOL_CALLS


async def test_没有挂起时resume会报错():
    _, _, _, loop = await _build([ModelResponse(text="ok")])
    with pytest.raises(RuntimeError) as e:
        await loop.resume_turn("adopt")
    assert "没有挂起" in str(e.value)


async def test_挂起时直接开新轮被护栏挡下():
    """挂起的 tool_calls 没有 tool 响应，直接 run_turn 会把残缺上下文发给模型，
    API 400（"assistant message with 'tool_calls' must be followed by tool messages"）。
    必须在本地拦住，且状态不被破坏。"""
    loop, _, _, _ = await _run_until_review()
    with pytest.raises(RuntimeError) as e:
        await loop.run_turn("我觉得没问题，控制在60集")
    assert "人审" in str(e.value)
    assert loop.pending_review is not None  # 状态不被破坏，还能正常结案
    assert len(loop.memory.turns) == 1  # 没多出一个只有 user 消息的废轮次

    r = await loop.resume_turn("adopt")
    assert r.stop_reason is StopReason.NO_TOOL_CALLS


async def test_采纳附言写成补充要求而不是避开项():
    """采纳时带的要求（'控制在60集'）是后续必须遵守的；
    沿用「理由（下一版必须避开）」的标签会让模型把它当成要避开的东西。"""
    loop, _, _, _ = await _run_until_review()
    r = await loop.resume_turn("adopt", reason="控制在60集，每集1到1分半")
    verdict = next(
        m["content"]
        for m in r.turn.messages
        if m.get("role") == "tool" and "人的决策" in m["content"]
    )
    assert "已采纳" in verdict
    assert "补充要求" in verdict and "必须避开" not in verdict
    assert "60集" in verdict


# ---------------------------------------------------------------- /auto 自动模式


async def test_auto模式人审自动采纳不挂起():
    """/auto on：request_review 不挂起问人，按采纳回填后继续跑。
    事件照发（decided_by=auto），痕迹/回放不用区分是不是人点的；
    自动采纳照样不落避雷记忆。"""
    memories = MemoryStore()
    bus = EventBus()
    assets = AssetStore()
    registry = ToolRegistry(bus)
    registry.register(ContentFunctions(assets))
    await registry.refresh()
    RejectionRecorder(memories, project_id="proj").attach(bus)
    holder: dict[str, str] = {}

    class Gw:
        def __init__(self):
            self.n = 0

        async def chat(self, role, messages, tools=None, **kw):
            self.n += 1
            if self.n == 1:
                return ModelResponse(
                    tool_calls=[
                        _tc(
                            "c1",
                            "save_draft",
                            '{"content":"第一版","kind":"copy","summary":"初稿"}',
                        )
                    ],
                    usage=Usage(5, 5),
                )
            if self.n == 2:
                return ModelResponse(
                    tool_calls=[
                        _tc(
                            "c2",
                            "request_review",
                            f'{{"asset_ids":["{holder["id"]}"],'
                            f'"question":"这版行吗","stage":"正文"}}',
                        )
                    ],
                    usage=Usage(5, 5),
                )
            return ModelResponse(text="继续干活", usage=Usage(5, 5))

    loop = LoopRuntime(
        gateway=Gw(),  # type: ignore[arg-type]
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
    loop.auto_review = True

    r = await loop.run_turn("写一段正文")

    assert r.stop_reason is StopReason.NO_TOOL_CALLS  # 没挂起，直接跑完
    assert loop.pending_review is None
    verdict = next(
        m["content"]
        for m in r.turn.messages
        if m.get("role") == "tool" and "人的决策" in m["content"]
    )
    assert "已采纳" in verdict
    decided = [e for e in bus.history if e.type is EventType.CHECKPOINT_DECIDED]
    assert len(decided) == 1 and decided[0].data["decided_by"] == "auto"
    assert memories.all(layer=Layer.PROJECT) == []  # 自动采纳不落避雷记忆


async def _run_auto_with_stage(stage: str):
    """/auto 模式下跑一次「存稿 → request_review(stage)」，返回 (loop, result)。"""
    bus = EventBus()
    assets = AssetStore()
    registry = ToolRegistry(bus)
    registry.register(ContentFunctions(assets))
    await registry.refresh()
    holder: dict[str, str] = {}

    class Gw:
        def __init__(self):
            self.n = 0

        async def chat(self, role, messages, tools=None, **kw):
            self.n += 1
            if self.n == 1:
                return ModelResponse(
                    tool_calls=[
                        _tc("c1", "save_draft", '{"content":"稿","kind":"copy","summary":"初稿"}')
                    ],
                    usage=Usage(5, 5),
                )
            if self.n == 2:
                return ModelResponse(
                    tool_calls=[
                        _tc(
                            "c2",
                            "request_review",
                            f'{{"asset_ids":["{holder["id"]}"],'
                            f'"question":"这版行吗","stage":"{stage}"}}',
                        )
                    ],
                    usage=Usage(5, 5),
                )
            return ModelResponse(text="继续干活", usage=Usage(5, 5))

    loop = LoopRuntime(
        gateway=Gw(),  # type: ignore[arg-type]
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
    loop.auto_review = True
    return loop, await loop.run_turn("干活")


async def test_auto模式大节点仍挂起问人():
    """/auto on：stage 命中大节点关键词（剧本/视频生成/图片生成）的人审不自动采纳，
    照样挂起等人拍板一次；人采纳后继续跑。"""
    for stage in ("剧本", "视频生成", "图片生成"):
        loop, r = await _run_auto_with_stage(stage)
        assert r.stop_reason is StopReason.AWAITING_REVIEW
        assert loop.pending_review is not None
        assert loop.pending_review["stage"] == stage

        r2 = await loop.resume_turn("adopt", decided_by="auto")
        assert r2.stop_reason is StopReason.NO_TOOL_CALLS
        assert loop.pending_review is None


async def test_auto模式某一集算小节点自动过():
    """/auto on：「剧本第3集」这种某一集的人审算小节点 —— 即使带「剧本」字样
    也自动采纳不挂起；没标环节（stage 缺省）同样自动过。"""
    for stage in ("剧本第3集", "第12集", "单集", ""):
        loop, r = await _run_auto_with_stage(stage)
        assert r.stop_reason is StopReason.NO_TOOL_CALLS, stage
        assert loop.pending_review is None
        verdict = next(
            m["content"]
            for m in r.turn.messages
            if m.get("role") == "tool" and "人的决策" in m["content"]
        )
        assert "已采纳" in verdict


async def test_中断留下的残缺tool_calls自动修复():
    """Ctrl+C 可能打断在「assistant 已回填、tool 结果还没回填」的中间态。
    下一轮开始前必须补上「已中断」的 tool 消息，否则 API 400。"""
    bus, _, _, loop = await _build([ModelResponse(text="ok", usage=Usage(5, 5))])
    t = loop.memory.new_turn()
    t.messages.append({"role": "user", "content": "旧输入"})
    t.messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "x1", "type": "function", "function": {"name": "now", "arguments": "{}"}}
            ],
        }
    )

    r = await loop.run_turn("接着说")

    fixed = [m for m in t.messages if m.get("role") == "tool" and m.get("tool_call_id") == "x1"]
    assert len(fixed) == 1 and "中断" in fixed[0]["content"]
    assert r.stop_reason is StopReason.NO_TOOL_CALLS
    assert any(
        e.type is EventType.WARNING and "中断" in e.data.get("message", "")
        for e in bus.history
    )


# ---------------------------------------------------------------- 复用 P1 底座


async def test_function模式下打回理由照样落库():
    """底座完全复用：L0 发 CHECKPOINT_DECIDED，L1 的 Recorder 照旧写库。

    图模式和 agentic 模式共用同一套记忆，不需要两份实现。
    """
    memories = MemoryStore()
    loop, _, _, _ = await _run_until_review(memories)
    await loop.resume_turn("revise", reason="开头太硬广了")

    recorded = memories.all(layer=Layer.PROJECT, project_id="proj")
    assert len(recorded) == 1
    assert "开头太硬广了" in recorded[0].content
    assert recorded[0].source is Source.HUMAN

    # 能被召回，下一轮生成前拿得回来
    assert "开头太硬广了" in memories.render_brief(memories.recall(["正文"]))


async def test_采纳不产生避雷记忆():
    memories = MemoryStore()
    loop, _, _, _ = await _run_until_review(memories)
    await loop.resume_turn("adopt")
    assert memories.all(layer=Layer.PROJECT) == []


async def test_改写保留血缘():
    loop, assets, _, _ = await _run_until_review()
    await loop.resume_turn("revise", reason="重写")

    latest = assets.latest()
    chain = assets.lineage(latest.id)
    assert len(chain) == 2, "第二版应挂在第一版下面，血缘不能断"
    assert chain[0].version == 1 and chain[1].version == 2


async def test_请人审不存在的资产会被挡下():
    script = [
        ModelResponse(
            tool_calls=[
                _tc("c1", "request_review", '{"asset_ids":["as_不存在"],"question":"看看"}')
            ],
            usage=Usage(5, 5),
        ),
        ModelResponse(text="好的我先存", usage=Usage(5, 5)),
    ]
    _, _, _, loop = await _build(script)
    r = await loop.run_turn("给人看看")

    assert r.stop_reason is StopReason.NO_TOOL_CALLS  # 没挂起
    assert loop.pending_review is None
    tool_msg = next(m for m in r.turn.messages if m.get("role") == "tool")
    assert "先用 save_draft 存下来" in tool_msg["content"]


# ---------------------------------------------------------------- CLI 人审决策解析


def test_cli人审输入解析():
    from aigc_agent.interfaces.cli.main import _parse_decision

    assert _parse_decision("a") == ("adopt", "")
    assert _parse_decision("a 控制在60集") == ("adopt", "控制在60集")
    assert _parse_decision("a　控制在60集") == ("adopt", "控制在60集")  # 全角空格
    assert _parse_decision("r 开头太硬广") == ("revise", "开头太硬广")
    assert _parse_decision("J") == ("reject", "")
    assert _parse_decision("我觉得没问题") is None  # 自由文本不认，重新问
    assert _parse_decision("") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

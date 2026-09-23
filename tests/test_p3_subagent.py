"""M10 子代理运行时验收 —— 最小能力：单个子代理拉起 + 独立上下文 + 结构化回传。

盯四件事：
  1. 只看得到、只调得了被允许的工具（全局注册表的裁剪视图）
  2. L-external 永远不给 —— 子代理背后没有人，不可逆动作没人确认
  3. 输出按 schema 校验，主 Agent 只收结构化结论
  4. 无状态：每次都是新上下文；只有显式 stateless=False 才保留
"""

from __future__ import annotations

from typing import Any

from aigc_agent.capabilities.subagents import (
    SubAgentDef,
    SubAgentRunner,
    parse_json,
    validate_output,
)
from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.functions.content import ContentFunctions
from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.model.gateway import ModelResponse, ToolCall, Usage
from aigc_agent.harness.tools.builtin import builtin
from aigc_agent.harness.tools.provider import PermissionLevel
from aigc_agent.harness.tools.registry import ToolRegistry


class _Gw:
    """按剧本返回；记录每次收到的 messages 与 tools。"""

    def __init__(self, script: list[ModelResponse]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, role, messages, tools=None, **kw):
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "tools": [t["function"]["name"] for t in (tools or [])],
            }
        )
        return self.script.pop(0) if len(self.script) > 1 else self.script[0]


def _tc(cid: str, name: str, args: str = "{}") -> ToolCall:
    return ToolCall(id=cid, name=name, arguments=args)


def _text(t: str) -> ModelResponse:
    return ModelResponse(text=t, usage=Usage(3, 3))


async def _registry(store: AssetStore | None = None):
    bus = EventBus()
    reg = ToolRegistry(bus)
    reg.register(builtin)
    # AssetStore 定义了 __len__，空库是假值 —— 不能写 `store or AssetStore()`
    reg.register(ContentFunctions(store if store is not None else AssetStore()))
    await reg.refresh()
    return reg, bus


DEF = SubAgentDef(
    name="probe",
    system_prompt="你是探针",
    tools=["now", "calc"],
    role="subagent",
    output_schema={
        "type": "object",
        "required": ["answer"],
        "properties": {"answer": {"type": "string"}},
    },
)


async def test_子代理只看到被允许的工具():
    reg, bus = await _registry()
    gw = _Gw([_text('{"answer":"ok"}')])
    r = await SubAgentRunner(gw, reg, bus).run(DEF, "报时")
    assert r.ok, r.error
    assert r.data == {"answer": "ok"}
    assert set(gw.calls[0]["tools"]) == {"now", "calc"}, "只带被允许的工具 schema"
    system = gw.calls[0]["messages"][0]["content"]
    assert "你是探针" in system and "输出契约" in system
    assert "save_draft" not in system, "目录里也不该出现未授权的工具"
    assert gw.calls[0]["role"] == "subagent"


async def test_调未授权工具被拒但子代理不崩():
    reg, bus = await _registry()
    gw = _Gw(
        [
            ModelResponse(
                tool_calls=[_tc("c1", "save_draft", '{"content":"x","kind":"copy"}')],
                usage=Usage(1, 1),
            ),
            _text('{"answer":"done"}'),
        ]
    )
    r = await SubAgentRunner(gw, reg, bus).run(DEF, "存一下")
    assert r.ok
    tool_msgs = [m for m in gw.calls[1]["messages"] if m.get("role") == "tool"]
    assert tool_msgs and "save_draft" in tool_msgs[-1]["content"]
    assert "未知工具" in tool_msgs[-1]["content"] or "无权" in tool_msgs[-1]["content"]


async def test_L_external永远不给():
    reg, bus = await _registry()
    d = SubAgentDef(
        name="pub",
        system_prompt="x",
        tools=["publish_demo"],
        allowed=[PermissionLevel.READ, PermissionLevel.EXTERNAL],  # 写了也没用
    )
    gw = _Gw(
        [
            ModelResponse(
                tool_calls=[_tc("c1", "publish_demo", '{"platform":"a","content":"b"}')],
                usage=Usage(1, 1),
            ),
            _text("done"),
        ]
    )
    r = await SubAgentRunner(gw, reg, bus).run(d, "发布")
    assert r.ok  # 子代理本身跑完了，只是工具被拒
    tool_msgs = [m for m in gw.calls[1]["messages"] if m.get("role") == "tool"]
    assert "策略禁止" in tool_msgs[-1]["content"]
    assert any(e.type is EventType.PERMISSION_DENY for e in bus.history)


async def test_显式给的写权限能用():
    store = AssetStore()
    reg, bus = await _registry(store)
    d = SubAgentDef(
        name="w",
        system_prompt="x",
        tools=["save_draft"],
        allowed=[PermissionLevel.READ, PermissionLevel.WRITE],
    )
    gw = _Gw(
        [
            ModelResponse(
                tool_calls=[_tc("c1", "save_draft", '{"content":"正文","kind":"copy"}')],
                usage=Usage(1, 1),
            ),
            _text("存好了"),
        ]
    )
    r = await SubAgentRunner(gw, reg, bus).run(d, "存")
    assert r.ok and len(store) == 1
    tool_msgs = [m for m in gw.calls[1]["messages"] if m.get("role") == "tool"]
    assert "已存为" in tool_msgs[-1]["content"]


async def test_输出按schema校验():
    reg, bus = await _registry()
    r = await SubAgentRunner(_Gw([_text('{"nope":1}')]), reg, bus).run(DEF, "x")
    assert not r.ok and "缺少必填字段" in (r.error or "") and r.data == {"nope": 1}

    r = await SubAgentRunner(_Gw([_text("我不会输出 JSON")]), reg, bus).run(DEF, "x")
    assert not r.ok and "合法 JSON" in (r.error or "")

    r = await SubAgentRunner(_Gw([_text('{"answer": 42}')]), reg, bus).run(DEF, "x")
    assert not r.ok and "应为 string" in (r.error or "")


async def test_没有schema时回传原文():
    reg, bus = await _registry()
    d = SubAgentDef(name="free", system_prompt="随意")
    gw = _Gw([_text("随便说")])
    r = await SubAgentRunner(gw, reg, bus).run(d, "x")
    assert r.ok and r.data == "随便说"
    assert "输出契约" not in gw.calls[0]["messages"][0]["content"]


async def test_无状态_每次都是新上下文():
    reg, bus = await _registry()
    gw = _Gw([_text('{"answer":"1"}')])
    runner = SubAgentRunner(gw, reg, bus)
    await runner.run(DEF, "第一件事")
    await runner.run(DEF, "第二件事")
    second = gw.calls[1]["messages"]
    assert not any("第一件事" in str(m.get("content")) for m in second)


async def test_有状态子代理才保留上一轮():
    reg, bus = await _registry()
    gw = _Gw([_text('{"answer":"1"}')])
    runner = SubAgentRunner(gw, reg, bus)
    d = DEF.model_copy(update={"name": "stateful", "stateless": False})
    await runner.run(d, "第一件事")
    await runner.run(d, "第二件事")
    second = gw.calls[1]["messages"]
    assert any("第一件事" in str(m.get("content")) for m in second)


async def test_子代理不能请人审():
    store = AssetStore()
    a = store.create("候选", summary="候选")
    reg, bus = await _registry(store)
    d = SubAgentDef(
        name="rev",
        system_prompt="x",
        tools=["request_review"],
        allowed=[PermissionLevel.READ, PermissionLevel.WRITE],
    )
    gw = _Gw(
        [
            ModelResponse(
                tool_calls=[
                    _tc("c1", "request_review", f'{{"asset_ids":["{a.id}"],"question":"行吗"}}')
                ],
                usage=Usage(1, 1),
            ),
            _text("?"),
        ]
    )
    r = await SubAgentRunner(gw, reg, bus).run(d, "请审")
    assert not r.ok and "不能请人审" in (r.error or "")


async def test_事件与耗时():
    reg, bus = await _registry()
    r = await SubAgentRunner(_Gw([_text('{"answer":"ok"}')]), reg, bus).run(DEF, "x")
    starts = [e for e in bus.history if e.type is EventType.SUBAGENT_START]
    ends = [e for e in bus.history if e.type is EventType.SUBAGENT_END]
    assert starts and starts[0].data["name"] == "probe"
    assert starts[0].data["tools"] == ["now", "calc"]
    assert ends and ends[0].data["ok"] and ends[0].data["iterations"] >= 1
    assert r.duration_ms >= 0


def test_parse_json容忍围栏与前后文():
    assert parse_json('前言 ```json\n{"a": 1}\n``` 后记') == {"a": 1}
    assert parse_json("结论如下：[1, 2]") == [1, 2]
    assert parse_json("什么都没有") is None
    assert parse_json('{"broken": ') is None


def test_validate_output三条检查():
    schema = {
        "type": "object",
        "required": ["a"],
        "properties": {"a": {"type": "array"}, "b": {"type": "number"}},
    }
    assert validate_output({"a": [], "b": 1.5}, schema) == ""
    assert "顶层" in validate_output([], schema)
    assert "缺少必填字段" in validate_output({"b": 1}, schema)
    assert "应为 array" in validate_output({"a": "x"}, schema)
    assert validate_output("anything", {}) == ""

"""P0 验收测试 —— 不依赖真实 API。

验收标准（ARCHITECTURE.md 建设顺序 P0）：
  能跑通一个多轮工具调用的最小 loop；工具注册走 Provider 而非硬编码。

模型调用用打桩替换，其余（工具注册/并发调度/权限闸门/上下文装配/
窗口驱逐）全部跑真实代码路径。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from aigc_agent.harness.context.assembler import ContextAssembler
from aigc_agent.harness.context.window import ShortTermMemory, Turn, WindowPolicy
from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.execution.loop import LoopRuntime, StopReason
from aigc_agent.harness.model.gateway import ModelResponse, ToolCall, Usage
from aigc_agent.harness.permission.gate import PermissionGate
from aigc_agent.harness.tools.builtin import builtin
from aigc_agent.harness.tools.dispatcher import ToolDispatcher
from aigc_agent.harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolResult,
    ToolSpec,
)
from aigc_agent.harness.tools.registry import ToolRegistry


class ScriptedGateway:
    """按剧本返回的假 Gateway。签名与 ModelGateway.chat 一致。"""

    def __init__(self, script: list[ModelResponse]) -> None:
        self.script = script
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(self, role, messages, tools=None, **kw) -> ModelResponse:
        self.calls.append(messages)
        return self.script[min(len(self.calls) - 1, len(self.script) - 1)]


def _tc(cid: str, name: str, args: str = "{}") -> ToolCall:
    return ToolCall(id=cid, name=name, arguments=args)


async def _build(script: list[ModelResponse], asker=None, policy=None):
    bus = EventBus(session_id="test")
    registry = ToolRegistry(bus)
    registry.register(builtin)
    await registry.refresh()

    gate = PermissionGate(bus, policy=policy, asker=asker)
    dispatcher = ToolDispatcher(registry, gate, bus, timeout=10)
    memory = ShortTermMemory(policy=WindowPolicy())
    loop = LoopRuntime(
        gateway=ScriptedGateway(script),  # type: ignore[arg-type]
        registry=registry,
        dispatcher=dispatcher,
        assembler=ContextAssembler(bus),
        memory=memory,
        bus=bus,
    )
    return bus, registry, memory, loop


class _SlowProvider:
    """两个同样睡 N 秒的工具：一个用调度器默认超时，一个自带大超时。

    批量渲染（drama_render_shots 等）一次调用跑几十分钟，默认 120s
    会把它误杀 —— 所以超时是按工具声明的。
    """

    name = "slow"
    namespaced = False

    def __init__(self) -> None:
        self._specs = [
            ToolSpec(
                name="slow_default", summary="慢任务（默认超时）", permission=PermissionLevel.READ
            ),
            ToolSpec(
                name="slow_long",
                summary="慢任务（自带 1s 超时）",
                permission=PermissionLevel.READ,
                timeout=1.0,
            ),
        ]

    async def list_tools(self):
        return [s.meta(self.name) for s in self._specs]

    async def get_schema(self, tool: str):
        return next(s for s in self._specs if s.name == tool).to_openai(tool)

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        await asyncio.sleep(float(args.get("seconds", 0.3)))
        return ToolResult(content="done")

    async def health(self):
        return ProviderHealth(ok=True)


async def test_调度器超时按工具覆盖():
    """没带超时的工具用调度器默认；带了的用它自己的 —— 长调用不被误杀。"""
    bus = EventBus()
    registry = ToolRegistry(bus)
    registry.register(_SlowProvider())
    await registry.refresh()
    disp = ToolDispatcher(registry, PermissionGate(bus), bus, timeout=0.1)

    r = await disp.run([_tc("c1", "slow_default", '{"seconds": 0.3}')])
    assert not r[0][1].ok and "超时" in (r[0][1].error or "")

    r = await disp.run([_tc("c2", "slow_long", '{"seconds": 0.3}')])
    assert r[0][1].ok and r[0][1].content == "done"


# ---------------------------------------------------------------- M4 注册表


async def test_registry_两级披露():
    bus = EventBus()
    registry = ToolRegistry(bus)
    registry.register(builtin)
    await registry.refresh()

    catalog = registry.catalog()
    assert {m.name for m in catalog} == {"now", "calc", "sleep_demo", "publish_demo"}

    # 内置 provider 不加命名空间前缀
    assert all("__" not in m.name for m in catalog)

    # 权限等级从 ToolSpec 带上来
    assert registry.meta("publish_demo").permission is PermissionLevel.EXTERNAL
    assert registry.meta("now").permission is PermissionLevel.READ

    # 目录级 vs 全 schema：目录显著更小，这是两级披露的意义
    digest = registry.catalog_digest()
    schemas = await registry.schemas()
    assert len(digest) < len(str(schemas))
    assert schemas[0]["type"] == "function"

    # 按子集展开
    subset = await registry.schemas(["calc"])
    assert len(subset) == 1 and subset[0]["function"]["name"] == "calc"


async def test_registry_未知工具不抛异常():
    bus = EventBus()
    registry = ToolRegistry(bus)
    registry.register(builtin)
    await registry.refresh()
    r = await registry.invoke("不存在的工具", {})
    assert not r.ok and "未知工具" in r.error


# ---------------------------------------------------------------- M1 Loop


async def test_loop_多轮工具调用后收敛():
    script = [
        ModelResponse(tool_calls=[_tc("c1", "now")], usage=Usage(10, 5)),
        ModelResponse(
            tool_calls=[_tc("c2", "calc", '{"expression": "(1+2)*3"}')], usage=Usage(20, 5)
        ),
        ModelResponse(text="现在时间已查到，计算结果是 9。", usage=Usage(30, 12)),
    ]
    bus, _, memory, loop = await _build(script)
    result = await loop.run_turn("现在几点，顺便算下 (1+2)*3")

    assert result.stop_reason is StopReason.NO_TOOL_CALLS
    assert result.iterations == 3
    assert "9" in result.text

    # 一问一答算 1 轮 —— 中间的工具调用不单独计轮
    assert len(memory.turns) == 1

    tool_events = [e for e in bus.history if e.type is EventType.TOOL_RESULT]
    assert {e.data["tool"] for e in tool_events} == {"now", "calc"}

    # 消息回填完整：user + (assistant+tool)×2 + assistant
    roles = [m["role"] for m in result.turn.messages]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]


async def test_loop_达到最大轮次会停():
    # 剧本永远要调工具，靠 max_iterations 兜底
    script = [ModelResponse(tool_calls=[_tc("c", "now")], usage=Usage(1, 1))]
    bus, _, _, loop = await _build(script)
    loop.max_iterations = 3
    result = await loop.run_turn("死循环测试")
    assert result.stop_reason is StopReason.MAX_ITERATIONS
    assert result.iterations == 3


async def test_同轮多工具是并发执行的():
    """内容场景的性能命门：同时生成 6 张图 vs 串行 6 次，差 6 倍。"""
    calls = [
        _tc(f"c{i}", "sleep_demo", f'{{"seconds": 0.4, "label": "t{i}"}}') for i in range(4)
    ]
    script = [
        ModelResponse(tool_calls=calls, usage=Usage(10, 5)),
        ModelResponse(text="都跑完了", usage=Usage(10, 5)),
    ]
    _, _, _, loop = await _build(script)

    started = time.perf_counter()
    result = await loop.run_turn("并发测试")
    elapsed = time.perf_counter() - started

    assert result.stop_reason is StopReason.NO_TOOL_CALLS
    # 并发 ≈ 0.4s；串行会是 1.6s。留足余量，只要明显低于串行即可
    assert elapsed < 1.0, f"看起来是串行执行的，耗时 {elapsed:.2f}s"

    tool_msgs = [m for m in result.turn.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 4


async def test_工具异常不打断loop():
    script = [
        # calc 参数非法，工具会失败但 loop 应继续
        ModelResponse(
            tool_calls=[_tc("c1", "calc", '{"expression": "import os"}')], usage=Usage(5, 5)
        ),
        ModelResponse(text="表达式有问题，换一个吧", usage=Usage(5, 5)),
    ]
    _, _, _, loop = await _build(script)
    result = await loop.run_turn("算一下")

    assert result.stop_reason is StopReason.NO_TOOL_CALLS
    tool_msg = next(m for m in result.turn.messages if m["role"] == "tool")
    assert "工具执行失败" in tool_msg["content"]


async def test_参数不是合法json时给出可读错误():
    script = [
        ModelResponse(tool_calls=[_tc("c1", "calc", "{不是json")], usage=Usage(5, 5)),
        ModelResponse(text="收到", usage=Usage(5, 5)),
    ]
    _, _, _, loop = await _build(script)
    result = await loop.run_turn("测试")
    tool_msg = next(m for m in result.turn.messages if m["role"] == "tool")
    assert "不是合法 JSON" in tool_msg["content"]


# ---------------------------------------------------------------- M5 权限


async def test_L_external_必须经人工确认():
    asked: list[str] = []

    async def approve(meta, args):
        asked.append(meta.name)
        return True

    script = [
        ModelResponse(
            tool_calls=[_tc("c1", "publish_demo", '{"platform":"抖音","content":"测试"}')],
            usage=Usage(5, 5),
        ),
        ModelResponse(text="已发布", usage=Usage(5, 5)),
    ]
    _, _, _, loop = await _build(script, asker=approve)
    result = await loop.run_turn("发一条")

    assert asked == ["publish_demo"]
    tool_msg = next(m for m in result.turn.messages if m["role"] == "tool")
    assert "已发布到 抖音" in tool_msg["content"]


async def test_拒绝后工具不执行且loop继续():
    async def deny(meta, args):
        return False

    script = [
        ModelResponse(
            tool_calls=[_tc("c1", "publish_demo", '{"platform":"抖音","content":"x"}')],
            usage=Usage(5, 5),
        ),
        ModelResponse(text="好的，不发了", usage=Usage(5, 5)),
    ]
    bus, _, _, loop = await _build(script, asker=deny)
    result = await loop.run_turn("发一条")

    tool_msg = next(m for m in result.turn.messages if m["role"] == "tool")
    assert "用户拒绝" in tool_msg["content"]
    assert result.stop_reason is StopReason.NO_TOOL_CALLS
    assert any(e.type is EventType.PERMISSION_DENY for e in bus.history)


async def test_没有询问入口时L_external被拒():
    """无人可问 = 拒绝。半自动定位下不允许静默放行不可逆动作。"""
    script = [
        ModelResponse(
            tool_calls=[_tc("c1", "publish_demo", '{"platform":"x","content":"y"}')],
            usage=Usage(5, 5),
        ),
        ModelResponse(text="ok", usage=Usage(5, 5)),
    ]
    _, _, _, loop = await _build(script, asker=None)
    result = await loop.run_turn("发")
    tool_msg = next(m for m in result.turn.messages if m["role"] == "tool")
    assert "需要人工确认" in tool_msg["content"]


async def test_只读工具自动放行不询问():
    asked: list[str] = []

    async def spy(meta, args):
        asked.append(meta.name)
        return True

    script = [
        ModelResponse(tool_calls=[_tc("c1", "now")], usage=Usage(5, 5)),
        ModelResponse(text="ok", usage=Usage(5, 5)),
    ]
    _, _, _, loop = await _build(script, asker=spy)
    await loop.run_turn("几点了")
    assert asked == []


async def test_registry公开入口必须过权限闸门():
    """回归：闸门原本只装在 ToolDispatcher 里，任何直接调 registry.invoke()
    的代码（CLI 命令、Graph 的 tool 节点）都能静默执行 L-external 工具。
    默认必须安全。"""
    bus = EventBus()
    registry = ToolRegistry(bus)
    registry.register(builtin)
    await registry.refresh()
    registry.gate = PermissionGate(bus, asker=None)  # 无人可问

    r = await registry.invoke("publish_demo", {"platform": "x", "content": "y"})
    assert not r.ok and "人工确认" in r.error

    # 只读工具不受影响
    assert (await registry.invoke("now", {})).ok


async def test_ungated入口仅供dispatcher自用():
    """dispatcher 已经串行过闸门，再问一遍会让人重复确认。"""
    bus = EventBus()
    registry = ToolRegistry(bus)
    registry.register(builtin)
    await registry.refresh()
    registry.gate = PermissionGate(bus, asker=None)

    r = await registry.invoke_ungated("publish_demo", {"platform": "x", "content": "y"})
    assert r.ok, "invoke_ungated 不该过闸门"


async def test_同一个L_external工具只确认一次():
    asked = []

    async def spy(meta, args):
        asked.append(meta.name)
        return True

    script = [
        ModelResponse(
            tool_calls=[_tc("c1", "publish_demo", '{"platform":"x","content":"y"}')],
            usage=Usage(5, 5),
        ),
        ModelResponse(text="ok", usage=Usage(5, 5)),
    ]
    _, registry, _, loop = await _build(script, asker=spy)
    registry.gate = loop.dispatcher.gate  # 与 dispatcher 共用同一个闸门
    await loop.run_turn("发一条")
    assert asked == ["publish_demo"], f"确认了 {len(asked)} 次，应该只有 1 次"


# ---------------------------------------------------------------- M3 窗口


def _turns(n: int, tokens: int = 100) -> list[Turn]:
    return [
        Turn(index=i, messages=[{"role": "user", "content": "x"}], tokens=tokens)
        for i in range(n)
    ]


def test_批量驱逐_不到阈值不动():
    p = WindowPolicy(window_turns=10, evict_at=15)
    for n in range(1, 15):
        keep, evicted = p.split(_turns(n))
        assert evicted == [], f"{n} 轮时不该驱逐"
        assert len(keep) == n


def test_批量驱逐_到15轮一次性剔回10():
    p = WindowPolicy(window_turns=10, evict_at=15)
    keep, evicted = p.split(_turns(15))
    assert len(keep) == 10
    assert len(evicted) == 5
    # 剔的是最老的 5 轮
    assert [t.index for t in evicted] == [0, 1, 2, 3, 4]
    assert [t.index for t in keep] == list(range(5, 15))


def test_token兜底能在轮数未到时触发():
    """保险丝：某一轮里连调几十次工具且结果没走资产引用。"""
    p = WindowPolicy(window_turns=10, evict_at=15, max_tokens=1000)
    keep, evicted = p.split(_turns(6, tokens=500))  # 6×500 = 3000 > 1000
    assert evicted, "token 超限时应触发驱逐"
    assert sum(t.tokens for t in keep) <= 1000 or len(keep) == 1


def test_pin不占窗口配额且不被驱逐():
    mem = ShortTermMemory(policy=WindowPolicy(window_turns=10, evict_at=15))
    mem.pin("tone", "这个号不要用感叹号", position="pre_input")
    for _ in range(15):
        t = mem.new_turn()
        t.tokens = 100
    evicted = mem.maybe_evict()

    assert len(evicted) == 5
    assert len(mem.turns) == 10
    # 第 1 轮定的调子被剔出窗口了，但 pin 还在 —— 这正是 pin 存在的理由
    assert "感叹号" in mem.pins["tone"].content
    assert len(mem.pins_at("pre_input")) == 1


async def test_loop结束会触发驱逐并发出事件():
    script = [ModelResponse(text="好", usage=Usage(5, 5))]
    bus, _, memory, loop = await _build(script)
    memory.policy = WindowPolicy(window_turns=3, evict_at=5)

    captured: list[list[Turn]] = []
    memory.on_evict = captured.append

    for i in range(5):
        await loop.run_turn(f"第 {i} 条")

    assert len(memory.turns) == 3
    assert captured and len(captured[0]) == 2
    assert any(e.type is EventType.WINDOW_EVICT for e in bus.history)


# ---------------------------------------------------------------- M3 装配


async def test_上下文排布顺序():
    """长期记忆召回放短期记忆之后、pin 紧贴当前输入 —— 见 assembler 文档。"""
    bus = EventBus()
    mem = ShortTermMemory()
    mem.pin("hard", "不要用感叹号", position="pre_input")
    mem.pin("brand", "品牌调性：克制", position="system")

    old = mem.new_turn()
    old.messages = [
        {"role": "user", "content": "历史提问"},
        {"role": "assistant", "content": "历史回答"},
    ]
    cur = mem.new_turn()
    cur.messages = [{"role": "user", "content": "当前提问"}]

    msgs = await ContextAssembler(bus).assemble(
        mem, cur, tool_catalog="- now：取时间", recalled="用户偏好口语化开头"
    )

    contents = [str(m.get("content") or "") for m in msgs]
    joined = "\n".join(contents)

    # system 位 pin 并入首条系统消息
    assert "品牌调性" in contents[0]
    assert "now：取时间" in contents[0]

    # 顺序：历史 → 召回 → pre_input pin → 当前输入
    i_hist = next(i for i, c in enumerate(contents) if "历史提问" in c)
    i_recall = next(i for i, c in enumerate(contents) if "偏好口语化" in c)
    i_pin = next(i for i, c in enumerate(contents) if "不要用感叹号" in c)
    i_cur = next(i for i, c in enumerate(contents) if "当前提问" in c)
    assert i_hist < i_recall < i_pin < i_cur, joined


async def test_装配不包含被驱逐的轮次():
    bus = EventBus()
    mem = ShortTermMemory(policy=WindowPolicy(window_turns=2, evict_at=3))
    for i in range(3):
        t = mem.new_turn()
        t.messages = [{"role": "user", "content": f"第{i}条"}]
        t.tokens = 10
    mem.maybe_evict()

    cur = mem.turns[-1]
    msgs = await ContextAssembler(bus).assemble(mem, cur)
    joined = "\n".join(str(m.get("content") or "") for m in msgs)
    assert "第0条" not in joined
    assert "第2条" in joined


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

"""2026-09-23 审查的内核问题（主循环与上下文）。

  · 估算不含工具 schema（89 份约 1.5 万 token），校准比值被顶到 3.0；历史轮又被校准两次
  · 按 token 驱逐时累加的是未折叠原文，而且没有回差 —— 一过 12 万每轮剔一轮
  · 两级披露对内置工具没生效；load_tool_schema 展开的当轮不生效；演示工具进了生产
  · 超时被当连接错误重试 6 次（CLI 再续 3 次），每次整包重发
  · continue_turn 不修补悬空调用、不校验是不是最新一轮、丢掉召回的记忆、不做驱逐
  · L0 内核写满了短剧领域知识
"""

from __future__ import annotations

import httpx2 as httpx
import pytest
from openai import APITimeoutError

from aigc_agent.domain.lines import get_line, line_providers
from aigc_agent.domain.system_prompt import MAJOR_STAGES, SYSTEM_PROMPT
from aigc_agent.harness.context.assembler import DEFAULT_SYSTEM_PROMPT, ContextAssembler
from aigc_agent.harness.context.window import ShortTermMemory, Turn, WindowPolicy
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.execution.loop import LoopRuntime, StopReason
from aigc_agent.harness.model.gateway import ModelResponse, ToolCall
from aigc_agent.harness.permission.gate import PermissionGate
from aigc_agent.harness.tools.builtin import builtin, demo
from aigc_agent.harness.tools.disclosure import DisclosureProvider
from aigc_agent.harness.tools.dispatcher import ToolDispatcher
from aigc_agent.harness.tools.registry import ToolRegistry
from tests.test_net_resilience import Flaky, _gateway

# ---------------------------------------------------------------- 估算与驱逐


def test_估算计入工具schema_历史轮不双重校准():
    asm = ContextAssembler(EventBus(), calibration=2.0)
    msgs = [{"role": "user", "content": "你好"}]
    before = asm.tokens_of(msgs)
    asm.set_tools([{"type": "function", "function": {"name": "x", "description": "说明" * 500}}])
    assert asm.tokens_of(msgs) > before + 500, "每次请求都带的 schema 要算进去"

    t = Turn(index=0, messages=[{"role": "user", "content": "字" * 300}], tokens=400)
    asm._history_view(t)  # noqa: SLF001
    # t.tokens 已经是校准后的数：400 ≤ 阈值 → 不折；之前再乘 2.0 会被误判成大轮
    assert asm._fold_decisions[0] is (400 > asm.compaction.history_turn_tokens)  # noqa: SLF001


def test_驱逐按折叠后的大小_一次剔到低水位():
    p = WindowPolicy(window_turns=10, evict_at=15, max_tokens=1000, low_tokens=500)
    # 原文很大、折叠后很小的轮：按原文算会误触发兜底
    big_but_folded = [Turn(index=i, tokens=900, view_tokens=100) for i in range(5)]
    assert not p.should_evict(big_but_folded)
    turns = [Turn(index=i, tokens=300) for i in range(5)]  # 1500 > 1000
    keep, evicted = p.split(turns)
    assert sum(t.weight for t in keep) <= 500, "一次剔到低水位，不是刚好压到上限下"
    assert len(keep) == 1 and len(evicted) == 4


# ---------------------------------------------------------------- 工具披露


async def _registry() -> ToolRegistry:
    bus = EventBus()
    reg = ToolRegistry(bus)
    reg.register(builtin)
    reg.register(demo)
    reg.register(DisclosureProvider(reg))
    await reg.refresh()
    return reg


async def test_按产线收窄全披露_紧凑目录只列要展开的():
    reg = await _registry()
    sent = {s["function"]["name"] for s in await reg.schemas_for_context()}
    assert {"now", "publish_demo"} <= sent
    reg.set_focus({"builtin", "meta"})
    sent = {s["function"]["name"] for s in await reg.schemas_for_context()}
    assert "now" in sent and "publish_demo" not in sent, "别的 provider 只上目录"
    d = reg.catalog_digest(compact=True)
    assert "publish_demo*" in d and "- now" not in d and "个工具的完整定义已随请求提供" in d
    assert "- now（L-read）" in reg.catalog_digest(), "完整目录照旧"
    reg.set_focus(None)
    assert "publish_demo" in {s["function"]["name"] for s in await reg.schemas_for_context()}


def test_产线的工具范围():
    drama = line_providers(get_line("drama"))
    assert {"drama", "content", "files", "media"} <= drama and "short_video" not in drama
    design = line_providers(get_line("design"))
    assert "poster" in design and "drama" not in design


def test_演示工具不在生产目录():
    names = set(builtin._specs)  # noqa: SLF001
    assert "publish_demo" not in names and "sleep_demo" not in names
    assert {"publish_demo", "sleep_demo"} <= set(demo._specs)  # noqa: SLF001


class _Gw:
    def __init__(self, script: list[ModelResponse]) -> None:
        self.script = script
        self.tools_seen: list[set[str]] = []

    async def chat(self, role, messages, tools=None, **kw) -> ModelResponse:
        self.tools_seen.append({t["function"]["name"] for t in tools or []})
        return self.script[min(len(self.tools_seen) - 1, len(self.script) - 1)]


def _tc(cid: str, name: str, args: str = "{}") -> ToolCall:
    return ToolCall(id=cid, name=name, arguments=args)


async def _loop(reg: ToolRegistry, script: list[ModelResponse]) -> LoopRuntime:
    bus = reg.bus
    return LoopRuntime(
        gateway=_Gw(script),  # type: ignore[arg-type]
        registry=reg,
        dispatcher=ToolDispatcher(reg, PermissionGate(bus), bus),
        assembler=ContextAssembler(bus),
        memory=ShortTermMemory(),
        bus=bus,
    )


async def test_展开的工具当轮就能用():
    reg = await _registry()
    reg.set_focus({"builtin", "meta"})
    loop = await _loop(reg, [
        ModelResponse(tool_calls=[_tc("c1", "load_tool_schema", '{"names": ["sleep_demo"]}')]),
        ModelResponse(tool_calls=[_tc("c2", "sleep_demo", '{"seconds": 0}')]),
        ModelResponse(text="好"),
    ])
    r = await loop.run_turn("跑")
    seen = loop.gateway.tools_seen  # type: ignore[attr-defined]
    assert "sleep_demo" not in seen[0] and "sleep_demo" in seen[1], "展开后下一次迭代就带上"
    assert r.stop_reason is StopReason.NO_TOOL_CALLS and not r.tool_failures


# ---------------------------------------------------------------- continue_turn


async def test_续跑只许最新一轮_修补悬空_沿用召回():
    reg = await _registry()
    loop = await _loop(reg, [ModelResponse(text="好")])
    r1 = await loop.run_turn("第一轮", recalled="记忆：别用感叹号")
    turn = r1.turn
    # 模拟被 /stop 打断：留一条没回填的工具调用
    turn.messages.append({"role": "assistant", "content": None, "tool_calls": [
        {"id": "x1", "type": "function", "function": {"name": "now", "arguments": "{}"}}]})
    r2 = await loop.continue_turn(turn)
    assert r2.stop_reason is StopReason.NO_TOOL_CALLS
    assert any(m.get("tool_call_id") == "x1" for m in turn.messages), "先补上悬空的调用"
    assert loop._recalled == "记忆：别用感叹号"  # noqa: SLF001
    await loop.run_turn("第二轮")
    with pytest.raises(RuntimeError, match="最新"):
        await loop.continue_turn(turn)


# ---------------------------------------------------------------- 超时


async def test_超时最多重试两次():
    bus = EventBus()
    gw, provider = _gateway(bus, max_attempts=3, connect_max_attempts=6)
    timeout = Flaky(fails=10, exc=lambda: APITimeoutError(request=httpx.Request("POST", "https://x")))
    gw._call_once = timeout  # type: ignore[method-assign]
    with pytest.raises(APITimeoutError):
        await gw._with_retry(provider, {}, use_stream=False)
    assert timeout.calls == 2, "超时不是网络抖动，原样整包重发没用"


# ---------------------------------------------------------------- L0 去领域化


def test_内核不认识领域():
    assert "短剧" not in DEFAULT_SYSTEM_PROMPT and "drama_" not in DEFAULT_SYSTEM_PROMPT
    assert "短剧" in SYSTEM_PROMPT and "drama_render_shots" in SYSTEM_PROMPT
    assert "剧本" in MAJOR_STAGES
    loop_defaults = LoopRuntime.__init__.__defaults__ or ()
    assert all("剧本" not in str(d) for d in loop_defaults)

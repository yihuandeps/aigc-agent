"""网络抖动不该让整轮白做（2026-09-19 用户实测：重试 2 次共等 1.5 秒就把 6 次迭代 ¥1 的活判死）。

两层：
  · 网关：连接类错误单独给更深的重试预算 + 上限封顶 + 抖动；状态码类照旧
  · 循环：失败带上类别，网络类的可以 continue_turn 在**同一轮**里接着跑（进度不丢）
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx2 as httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.execution.loop import LoopResult, StopReason
from aigc_agent.harness.model.config import (
    ModelsConfig,
    ProviderConfig,
    RetryConfig,
    TextConfig,
)
from aigc_agent.harness.model.gateway import ModelGateway, ModelResponse, classify_model_error


def _conn_error() -> APIConnectionError:
    return APIConnectionError(request=httpx.Request("POST", "https://api.example.com/v1"))


def _status_error(code: int) -> APIStatusError:
    req = httpx.Request("POST", "https://api.example.com/v1")
    return APIStatusError("boom", response=httpx.Response(code, request=req), body=None)


# ---------------------------------------------------------------- 配置


def test_连接类错误的重试预算更深():
    cfg = RetryConfig()
    assert cfg.max_attempts == 3 and cfg.connect_max_attempts >= 5
    assert cfg.attempts_for(connection=False) == 3
    assert cfg.attempts_for(connection=True) == cfg.connect_max_attempts
    assert cfg.max_delay_ms >= 5_000 and 0 <= cfg.jitter < 1
    # 真实配置也得是这套
    from pathlib import Path

    real = ModelsConfig.load(Path(__file__).resolve().parents[1] / "config" / "models.yaml")
    provider, _ = real.text.resolve("main_agent")
    assert provider.retry.connect_max_attempts >= 5


def test_失败类别_连接与超时():
    kind, hint = classify_model_error(_conn_error())
    assert kind == "connection" and "代理" in hint
    timeout = APITimeoutError(request=httpx.Request("POST", "https://x"))
    assert classify_model_error(timeout)[0] == "timeout"
    assert classify_model_error(_status_error(500))[0] == "other"


# ---------------------------------------------------------------- 网关重试


class Flaky:
    """前 n 次抛错，之后成功。"""

    def __init__(self, fails: int, exc: Any = None) -> None:
        self.fails = fails
        self.calls = 0
        self.exc = exc or _conn_error

    async def __call__(self, provider: Any, kwargs: dict[str, Any]) -> ModelResponse:
        self.calls += 1
        if self.calls <= self.fails:
            raise self.exc()
        return ModelResponse(text="ok")


def _provider(retry: RetryConfig) -> ProviderConfig:
    return ProviderConfig(
        key="p", model="m", api_key="sk-x", base_url="https://api.example.com/v1",
        retry=retry, stream=False,
    )


def _gateway(bus: EventBus, **retry: Any) -> tuple[ModelGateway, ProviderConfig]:
    provider = _provider(RetryConfig(initial_delay_ms=1, max_delay_ms=4, jitter=0.0, **retry))
    cfg = ModelsConfig(text=TextConfig(providers={"p": provider}, roles={"main_agent": "p"}))
    return ModelGateway(cfg, bus), provider


async def test_连接抖动_多等几次最终成功():
    bus = EventBus()
    events: list[Any] = []
    bus.subscribe(lambda e: events.append(e) if e.type is EventType.MODEL_RETRY else None)
    gw, provider = _gateway(bus, max_attempts=3, connect_max_attempts=6)
    flaky = Flaky(fails=4)  # 超过 max_attempts=3，但在 connect 预算内
    gw._call_once = flaky  # type: ignore[method-assign]
    resp = await gw._with_retry(provider, {}, use_stream=False)
    assert resp.text == "ok" and flaky.calls == 5
    assert [e.data["max_attempts"] for e in events] == [6, 6, 6, 6], "重试提示按连接预算显示"


async def test_状态码类错误不享受更深预算():
    bus = EventBus()
    gw, provider = _gateway(bus, max_attempts=3, connect_max_attempts=6)
    flaky = Flaky(fails=9, exc=lambda: _status_error(503))
    gw._call_once = flaky  # type: ignore[method-assign]
    with pytest.raises(APIStatusError):
        await gw._with_retry(provider, {}, use_stream=False)
    assert flaky.calls == 3


async def test_不可重试的状态码一次就抛():
    bus = EventBus()
    gw, provider = _gateway(bus)
    flaky = Flaky(fails=9, exc=lambda: _status_error(400))
    gw._call_once = flaky  # type: ignore[method-assign]
    with pytest.raises(APIStatusError):
        await gw._with_retry(provider, {}, use_stream=False)
    assert flaky.calls == 1


async def test_等待指数退避_封顶_带抖动(monkeypatch: Any):
    """不真的睡：把 sleep 换掉，直接看它打算等多久。"""
    import aigc_agent.harness.model.gateway as mod

    waits: list[float] = []

    async def fake_sleep(s: float) -> None:
        waits.append(s)

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)
    provider = _provider(
        RetryConfig(connect_max_attempts=8, initial_delay_ms=500, max_delay_ms=15_000, jitter=0.25)
    )
    cfg = ModelsConfig(text=TextConfig(providers={"p": provider}, roles={"main_agent": "p"}))
    gw = ModelGateway(cfg, EventBus())
    flaky = Flaky(fails=7)
    gw._call_once = flaky  # type: ignore[method-assign]
    await gw._with_retry(provider, {}, use_stream=False)
    assert len(waits) == 7 and flaky.calls == 8
    assert all(w <= 15.0 * 1.25 + 1e-9 for w in waits), "封顶生效"
    assert waits[3] > waits[0], "指数退避"
    assert len({round(w, 6) for w in waits[-3:]}) == 3, "到顶之后仍有抖动，不是完全一样"
    assert sum(waits) > 25, "连接类错误总共能容忍几十秒的断网"


# ---------------------------------------------------------------- 循环层


def _result(kind: str, stop: StopReason = StopReason.ERROR) -> LoopResult:
    return LoopResult(text="x", turn=None, iterations=6, stop_reason=stop, error_kind=kind)  # type: ignore[arg-type]


def test_网络类失败可原地续跑_其余不可():
    assert _result("connection").resumable
    assert _result("timeout").resumable
    assert not _result("quota").resumable
    assert not _result("context_overflow").resumable
    assert not _result("", StopReason.NO_TOOL_CALLS).resumable


async def test_循环把失败类别带出来():
    from types import SimpleNamespace

    from aigc_agent.harness.context.assembler import ContextAssembler
    from aigc_agent.harness.context.window import ShortTermMemory
    from aigc_agent.harness.execution.loop import LoopRuntime
    from aigc_agent.harness.tools.dispatcher import ToolDispatcher
    from aigc_agent.harness.tools.registry import ToolRegistry

    bus = EventBus()
    registry = ToolRegistry(bus)
    await registry.refresh()

    class Gw:
        async def chat(self, *a: Any, **k: Any) -> Any:
            raise _conn_error()

    loop = LoopRuntime(
        gateway=Gw(),  # type: ignore[arg-type]
        registry=registry,
        dispatcher=ToolDispatcher(registry, SimpleNamespace(check=_ok), bus),  # type: ignore[arg-type]
        assembler=ContextAssembler(bus),
        memory=ShortTermMemory(),
        bus=bus,
    )
    r = await loop.run_turn("写第 6 集")
    assert r.stop_reason is StopReason.ERROR and r.error_kind == "connection"
    assert r.resumable and loop.last_error_kind == "connection"
    assert "模型调用失败" in r.text and "代理" in r.text
    # 同一轮还在窗口里，可以 continue_turn 接着跑（这里网关一直挂，只验它不新建轮次）
    before = len(loop.memory.turns)
    again = await loop.continue_turn(r.turn)
    assert len(loop.memory.turns) == before and again.turn is r.turn


async def _ok(meta: Any, args: Any) -> tuple[bool, str]:
    return True, ""


def test_重试间隔配置():
    from aigc_agent.interfaces.cli.main import _NET_RETRIES, _NET_WAIT

    assert _NET_RETRIES >= 2 and _NET_WAIT >= 10
    assert asyncio.iscoroutinefunction(__import__(
        "aigc_agent.interfaces.cli.main", fromlist=["_retry_after"]
    )._retry_after)

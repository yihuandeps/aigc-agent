"""缺口 C（2026-09-23 审查）：钱花了拿不到结果、同一个镜头付两份钱。

审查结论：
  · task_id 只活在内存里 —— /stop、工具超时、轮询放弃之后，服务端照跑照扣费，本地什么都没留
  · 轮询遇到一次 5xx/429 就判死；上层再把「轮询失败」当网络抖动**重新提交**
  · 读超时后也重提（请求可能已经送达），一个镜头最多被提交 6 次
  · 中断时同一批里已经完成的调用结果一起丢，被补记成「没有执行结果」
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx2 as httpx
import pytest

from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.model.media import (
    MediaGateway,
    MediaKind,
    MediaTask,
    PollTransientError,
    TaskStatus,
)
from aigc_agent.harness.model.task_ledger import MediaTaskLedger

CATALOG = Path(__file__).resolve().parents[1] / "config" / "media_models.yaml"


class Scripted:
    """每个任务独立 id；可以脚本化「轮询第几次抛什么」。"""

    name = "scripted"

    def __init__(self, polls_needed: int = 2, poll_errors: dict[int, Exception] | None = None,
                 submit_errors: list[Exception | MediaTask] | None = None) -> None:
        self.polls_needed = polls_needed
        self.poll_errors = dict(poll_errors or {})
        self.submit_errors = list(submit_errors or [])
        self.submits = 0
        self.polls: dict[str, int] = {}

    async def submit(self, kind, model, prompt, **params):
        self.submits += 1
        if self.submit_errors:
            e = self.submit_errors.pop(0)
            if isinstance(e, MediaTask):
                return e
            raise e
        return MediaTask(task_id=f"t{self.submits}", kind=kind, model=model)

    async def poll(self, task):
        n = self.polls[task.task_id] = self.polls.get(task.task_id, 0) + 1
        task.polls += 1
        if n in self.poll_errors:
            raise self.poll_errors[n]
        if n >= self.polls_needed:
            task.status = TaskStatus.SUCCEEDED
            task.urls = [f"https://cdn/{task.task_id}.mp4"]
        else:
            task.status = TaskStatus.RUNNING
        return task

    async def close(self):
        return None


def _gw(p, tmp_path: Path, **kw) -> MediaGateway:
    return MediaGateway(
        {"x": p}, EventBus(), poll_interval=0.01, poll_backoff=1.0, max_poll_interval=0.01,
        ledger=MediaTaskLedger(tmp_path / "media_tasks.jsonl"), **kw,
    )


# ---------------------------------------------------------------- 台账本身


def test_台账按task_id折叠_坏行跳过_交付后不再可取回(tmp_path: Path):
    led = MediaTaskLedger(tmp_path / "t.jsonl")
    led.submitted("a1", kind="video", model="m", provider="x", fingerprint="fp", prompt="p")
    led.update("a1", "timeout", error="没等到")
    with (tmp_path / "t.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"task_id": "a1", "status": "succ')  # 进程被杀写了半行
    rec = led.get("a1")
    assert rec is not None and rec.status == "timeout" and rec.undelivered
    assert led.recoverable("fp").task_id == "a1"
    assert led.recoverable("fp", exclude={"a1"}) is None  # 本进程正在等它
    led.delivered("a1")
    assert led.recoverable("fp") is None and led.pending() == []


# ---------------------------------------------------------------- 轮询


async def test_轮询遇到5xx和429当抖动接着问_不判死(tmp_path: Path):
    p = Scripted(polls_needed=4, poll_errors={
        1: PollTransientError("HTTP 502：bad gateway"),
        2: PollTransientError("HTTP 429：too many"),
    })
    task = await _gw(p, tmp_path).generate("x", MediaKind.VIDEO, "m", "镜头")
    assert task.ok, task.error
    assert p.submits == 1  # 没有重新提交


async def test_本地放弃等待_任务记进台账_同一请求再来先取回不重提(tmp_path: Path):
    p = Scripted(polls_needed=10**9)  # 永远没生成完
    gw = _gw(p, tmp_path, max_polls=3)
    t1 = await gw.generate("x", MediaKind.VIDEO, "m", "同一个镜头", duration=15)
    assert t1.status is TaskStatus.TIMEOUT and t1.retryable is False and t1.stage == "poll"
    assert "media_recover" in (t1.error or "") and t1.task_id == "t1"
    assert gw.ledger.get("t1").status == "timeout"

    # 服务端后来生成完了；上层用同一份参数「重试」—— 应该取回 t1，而不是再提交一次
    p.polls_needed = 1
    t2 = await gw.generate("x", MediaKind.VIDEO, "m", "同一个镜头", duration=15)
    assert t2.ok and t2.recovered and t2.task_id == "t1"
    assert p.submits == 1, "没有重新付费"

    # 参数不同就是另一份请求：正常提交
    t3 = await gw.generate("x", MediaKind.VIDEO, "m", "同一个镜头", duration=10)
    assert t3.ok and not t3.recovered and p.submits == 2


async def test_被取消时把task_id记成abandoned(tmp_path: Path):
    p = Scripted(polls_needed=10**9)
    gw = _gw(p, tmp_path)
    job = asyncio.create_task(gw.generate("x", MediaKind.VIDEO, "m", "镜头"))
    await asyncio.sleep(0.1)
    job.cancel()
    with pytest.raises(asyncio.CancelledError):
        await job
    rec = gw.ledger.get("t1")
    assert rec is not None and rec.status == "abandoned" and rec.undelivered


# ---------------------------------------------------------------- 提交


async def test_提交读超时不重提_连不上才重提(tmp_path: Path):
    p = Scripted(submit_errors=[httpx.ReadTimeout("read")])
    t = await _gw(p, tmp_path).generate("x", MediaKind.VIDEO, "m", "镜头")
    assert t.status is TaskStatus.FAILED and not t.retryable and p.submits == 1
    assert "可能已经建了任务" in (t.error or "")

    p2 = Scripted(polls_needed=1, submit_errors=[httpx.ConnectError("no route")])
    t2 = await _gw(p2, tmp_path, submit_retries=2).generate("x", MediaKind.VIDEO, "m", "镜头2")
    assert t2.ok and p2.submits == 2


async def test_提交被429拒收_退避后再试(tmp_path: Path, monkeypatch):
    slept: list[float] = []
    real_sleep = asyncio.sleep

    async def fast_sleep(s: float) -> None:
        slept.append(s)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)
    busy = MediaTask(task_id="", kind=MediaKind.VIDEO, model="m", status=TaskStatus.FAILED,
                     error="HTTP 429：busy", http_status=429, stage="submit", retryable=True)
    p = Scripted(polls_needed=1, submit_errors=[busy, busy])
    t = await _gw(p, tmp_path).generate("x", MediaKind.VIDEO, "m", "镜头")
    assert t.ok and p.submits == 3
    assert any(s >= 5 for s in slept), "429 要退避，不是立刻重打"


# ---------------------------------------------------------------- 并发


async def test_按模态共享并发上限_两批同时跑也不翻倍(tmp_path: Path):
    running = peak = 0

    class Slow(Scripted):
        async def submit(self, kind, model, prompt, **params):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            try:
                return await super().submit(kind, model, prompt, **params)
            finally:
                await asyncio.sleep(0.05)
                running -= 1

    p = Slow(polls_needed=1)
    gw = _gw(p, tmp_path, concurrency={"video": 2})
    await asyncio.gather(*(gw.generate("x", MediaKind.VIDEO, "m", f"镜头{i}") for i in range(6)))
    assert peak <= 2


# ---------------------------------------------------------------- 工具层 + 分发器


async def test_media_tasks与media_recover把钱花了没拿到的结果取回来(tmp_path: Path):
    from aigc_agent.domain.assets.store import AssetStore
    from aigc_agent.domain.functions.media import MediaFunctions
    from aigc_agent.domain.generators.catalog import MediaCatalog

    catalog = MediaCatalog.load(CATALOG)
    p = Scripted(polls_needed=10**9)
    gw = MediaGateway({catalog.provider: p}, EventBus(), poll_interval=0.01, poll_backoff=1.0,
                      max_poll_interval=0.01, max_polls=2,
                      ledger=MediaTaskLedger(tmp_path / "m.jsonl"))
    fns = MediaFunctions(gw, catalog, AssetStore(tmp_path / "assets"))
    fns.video_lock = next(iter(catalog.video)).id  # 有锁：不触发选型
    r = await fns._fn_gen_video("镜头", allow_no_refs=True)
    assert not r.ok and r.meta.get("retryable") is False and r.meta.get("task_id") == "t1"

    listed = await fns._fn_media_tasks()
    assert "t1" in listed.content and "media_recover" in listed.content
    p.polls_needed = 1
    got = await fns._fn_media_recover("t1", summary="取回的镜头")
    assert got.ok and "没有重新付费" in got.content and p.submits == 1
    assert fns.store.get(got.asset_ref).gen_params.get("task_id") == "t1"
    again = await fns._fn_media_recover("t1")
    assert not again.ok and "已经登记过" in (again.error or "")


async def test_中断时同一批已完成的结果不丢(tmp_path: Path):
    from aigc_agent.harness.context.assembler import ContextAssembler
    from aigc_agent.harness.context.window import ShortTermMemory
    from aigc_agent.harness.execution.loop import LoopRuntime
    from aigc_agent.harness.model.gateway import ToolCall
    from aigc_agent.harness.permission.gate import PermissionGate
    from aigc_agent.harness.tools.dispatcher import ToolDispatcher
    from aigc_agent.harness.tools.provider import (
        PermissionLevel,
        ProviderHealth,
        ToolResult,
        ToolSpec,
    )
    from aigc_agent.harness.tools.registry import ToolRegistry

    class Tools:
        name = "t"
        namespaced = False

        async def list_tools(self):
            return [ToolSpec(name=n, summary=n, permission=PermissionLevel.READ).meta("t")
                    for n in ("fast", "slow")]

        async def get_schema(self, tool):
            spec = ToolSpec(name=tool, summary=tool, permission=PermissionLevel.READ)
            return spec.to_openai(tool)

        async def invoke(self, tool, args):
            if tool == "slow":
                await asyncio.sleep(10)
            return ToolResult(content=f"{tool} 已生成 as_done")

        async def health(self):
            return ProviderHealth()

    bus = EventBus()
    reg = ToolRegistry(bus)
    reg.register(Tools())
    await reg.refresh()
    disp = ToolDispatcher(reg, PermissionGate(bus), bus)
    calls = [ToolCall("c1", "fast", "{}"), ToolCall("c2", "slow", "{}")]
    job = asyncio.create_task(disp.run(calls))
    await asyncio.sleep(0.2)
    job.cancel()
    with pytest.raises(asyncio.CancelledError):
        await job

    mem = ShortTermMemory()
    turn = mem.new_turn()
    turn.messages.append({"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "fast", "arguments": "{}"}},
        {"id": "c2", "type": "function", "function": {"name": "slow", "arguments": "{}"}},
    ]})
    loop = LoopRuntime(gateway=None, registry=reg, dispatcher=disp,  # type: ignore[arg-type]
                       assembler=ContextAssembler(bus), memory=mem, bus=bus)
    assert loop._repair_interrupted_calls() == 2
    by_id = {m["tool_call_id"]: m["content"] for m in turn.messages if m.get("role") == "tool"}
    assert "fast 已生成 as_done" in by_id["c1"], "先跑完的那个回填真实结果"
    assert "media_tasks" in by_id["c2"], "没跑完的提醒先核对，别直接重做"
    _ = json  # 保持导入稳定

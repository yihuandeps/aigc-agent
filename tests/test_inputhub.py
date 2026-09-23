"""输入枢纽（2026-09-18）：生成过程中输入框不消失。

一轮在跑时打字 → 排队、本轮结束后自动发送；/stop 立刻停当前；/now 停当前并插队；
问人的提示（权限/决策）等的是新输入，不吃排队消息；停止优先于任何正在等的提示。
"""

from __future__ import annotations

import asyncio

import pytest
from rich.console import Console

from aigc_agent.interfaces.cli.inputhub import InputHub


def _hub() -> tuple[InputHub, Console]:
    console = Console(record=True, width=120, force_terminal=False)
    hub = InputHub(console, use_thread=False)
    hub.start()
    return hub, console


async def _slow(result: str = "done", seconds: float = 0.3) -> str:
    await asyncio.sleep(seconds)
    return result


async def test_空闲时问答_和排队消息优先发送():
    hub, console = _hub()
    loop = asyncio.get_running_loop()
    loop.call_later(0.02, hub.feed, "你好")
    assert await hub.next_message("你 ") == "你好"
    # 排队里有东西：next_message 直接取，不等人打
    hub.pending.append("下一条")
    assert await hub.next_message("你 ") == "下一条"
    assert "排队消息" in console.export_text()
    hub.close()


async def test_跑的时候打字会排队_本轮结束后按序发送():
    hub, console = _hub()
    loop = asyncio.get_running_loop()
    loop.call_later(0.02, hub.feed, "第一句")
    loop.call_later(0.04, hub.feed, "第二句")
    loop.call_later(0.06, hub.feed, "/queue")
    result = await hub.watch(_slow("ok"))
    assert result == "ok" and hub.pending == ["第一句", "第二句"]
    out = console.export_text()
    assert "已排队（第 1 条）" in out and "已排队（第 2 条）" in out and "1. 第一句" in out
    assert await hub.next_message("你 ") == "第一句"
    assert await hub.next_message("你 ") == "第二句"
    hub.close()


async def test_stop立刻停掉当前一轮():
    hub, console = _hub()
    loop = asyncio.get_running_loop()
    loop.call_later(0.02, hub.feed, "/stop")
    with pytest.raises(asyncio.CancelledError):
        await hub.watch(_slow("never", seconds=5))
    assert hub.stopped_by_user and not hub.running
    assert "正在停止" in console.export_text()
    hub.close()


async def test_now停掉当前并插队():
    hub, _ = _hub()
    loop = asyncio.get_running_loop()
    hub.pending.append("原来排着的")
    loop.call_later(0.02, hub.feed, "/now 先发这条")
    with pytest.raises(asyncio.CancelledError):
        await hub.watch(_slow("never", seconds=5))
    assert hub.stopped_by_user and hub.pending == ["先发这条", "原来排着的"]
    assert await hub.next_message("你 ") == "先发这条"
    hub.close()


async def test_问人时等的是新输入_不吃排队():
    hub, _ = _hub()
    loop = asyncio.get_running_loop()
    hub.pending.append("排队的话")
    loop.call_later(0.02, hub.feed, "y")
    assert await hub.ask("执行吗？") == "y"
    assert hub.pending == ["排队的话"]
    hub.close()


async def test_正在问权限时stop_也能把任务停掉():
    hub, _ = _hub()
    loop = asyncio.get_running_loop()

    async def turn() -> str:
        ans = await hub.ask("执行吗？")  # 任务内部问人
        await asyncio.sleep(5)
        return ans

    loop.call_later(0.05, hub.feed, "/stop")
    with pytest.raises(asyncio.CancelledError):
        await hub.watch(turn())
    assert hub.stopped_by_user
    hub.close()


async def test_stdin关了_ask抛EOF():
    hub, _ = _hub()
    loop = asyncio.get_running_loop()
    loop.call_later(0.02, hub.feed, None)
    with pytest.raises(EOFError):
        await hub.ask("你 ")
    assert hub.eof
    with pytest.raises(EOFError):
        await hub.ask("你 ")
    hub.close()


async def test_外部取消不算人停的():
    hub, _ = _hub()
    task = asyncio.ensure_future(hub.watch(_slow("x", seconds=5)))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not hub.stopped_by_user
    hub.close()

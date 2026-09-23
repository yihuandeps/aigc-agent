"""2026-09-23 审查的人机交互问题。

  · 后台提问（流水线经闸门问人）顶掉了主循环等着的输入：你打的字被当成那个提问的回答，
    主循环再也不返回【实测·模拟】
  · 读一行就进出一次 patch_stdout、进度窗又反复重定向 stdout：最终回复可能永远不显示
  · /stop 不清排队：停完立刻开下一轮接着花钱
  · 命令表和实现对不上：/help 没实现，拼错的 /xxx 原样发给模型
  · 会话回放分不清哪些是 /auto 自动采纳的；日志里的方括号能把回放带崩
  · 调度前的拒绝、人点头放行、质检门判定都不发事件
"""

from __future__ import annotations

import asyncio

import pytest
from rich.console import Console

from aigc_agent.harness.events.bus import Event, EventBus, EventType
from aigc_agent.harness.model.gateway import ToolCall
from aigc_agent.harness.permission.gate import PermissionGate
from aigc_agent.harness.tools.builtin import builtin
from aigc_agent.harness.tools.dispatcher import ToolDispatcher
from aigc_agent.harness.tools.registry import ToolRegistry
from aigc_agent.interfaces.cli import sessions_cmd
from aigc_agent.interfaces.cli.inputhub import InputHub
from aigc_agent.interfaces.cli.progress import ProgressBoard
from aigc_agent.interfaces.cli.promptbox import COMMANDS, PromptReader, help_text, known_command


def _hub() -> tuple[InputHub, Console]:
    console = Console(record=True, width=120, force_terminal=False)
    hub = InputHub(console, use_thread=False)
    hub.start()
    return hub, console


async def test_后台提问不顶掉主循环的输入():
    hub, console = _hub()
    loop = asyncio.get_running_loop()
    main = asyncio.ensure_future(hub.next_message("你 "))
    await asyncio.sleep(0.01)
    # 主循环正等着「你」的时候，后台问了一句
    background = asyncio.ensure_future(hub.ask("执行吗？(y/N) "))
    await asyncio.sleep(0.01)
    loop.call_soon(hub.feed, "y")
    assert await background == "y", "最后问的那个先拿到下一行"
    assert not main.done(), "主循环还在等，没有被顶掉"
    loop.call_soon(hub.feed, "继续写第 6 集")
    assert await asyncio.wait_for(main, 1) == "继续写第 6 集"
    assert console.export_text().count("你 ") >= 2, "问完后把主循环的提示再打一遍"
    hub.close()


async def test_stop清掉排队():
    hub, console = _hub()
    loop = asyncio.get_running_loop()
    hub.pending += ["再渲第 3 集", "再渲第 4 集"]
    stops: list[str] = []
    hub.on_stop = stops.append
    loop.call_later(0.02, hub.feed, "/stop")

    async def slow() -> None:
        await asyncio.sleep(5)

    with pytest.raises(asyncio.CancelledError):
        await hub.watch(slow())
    assert hub.pending == [] and stops == ["/stop"]
    assert "排队的 2 条也清掉了" in console.export_text()
    hub.close()


def test_命令表_帮助_未知命令():
    names = {c for c, _ in COMMANDS}
    assert {"/help", "/line", "/retry", "/budget", "/out"} <= names
    assert known_command("/budget set 金额 300") and known_command("/停")
    assert not known_command("/bugdet") and not known_command("/xyz 参数")
    assert "/rollback" in help_text()


def test_输入框整个会话只进一次patch():
    entered: list[str] = []

    class Ctx:
        def __init__(self, raw: bool) -> None:
            assert raw

        def __enter__(self) -> None:
            entered.append("in")

        def __exit__(self, *a: object) -> None:
            entered.append("out")

    r = PromptReader(lambda: "hi\n", Ctx)
    r.start()
    assert r() == "hi\n" and r() == "hi\n"
    r.start()  # 重入不再进一次
    r.stop()
    r.stop()
    assert entered == ["in", "out"]


def test_进度窗不重定向stdout():
    board = ProgressBoard(Console(record=True))
    with board.running():
        live = board._live  # noqa: SLF001
        if live is not None:
            assert not live._redirect_stdout and not live._redirect_stderr  # noqa: SLF001


async def test_调度前被拒_和人点头放行_都发事件():
    bus = EventBus()
    reg = ToolRegistry(bus)
    reg.register(builtin)
    await reg.refresh()

    async def yes(meta: object, args: object) -> bool:
        return True

    disp = ToolDispatcher(reg, PermissionGate(bus, asker=yes), bus)
    await disp.run([ToolCall(id="c1", name="calc", arguments='{"expression": "1+')])
    await disp.run([ToolCall(id="c2", name="nope", arguments="{}")])
    await disp.run([ToolCall(id="c3", name="publish_demo", arguments='{"text": "x"}')])
    types = [e.type for e in bus.history]
    assert types.count(EventType.TOOL_REJECTED) == 2
    assert EventType.PERMISSION_GRANT in types


def test_回放显示谁定的_方括号不炸():
    ev = Event(type=EventType.CHECKPOINT_DECIDED,
               data={"decision": "adopt", "reason": "[/] 好", "decided_by": "auto"})
    line = sessions_cmd._line(ev)  # noqa: SLF001
    assert line is not None and "auto" in line
    Console(file=None).render_str(line)  # 不抛 MarkupError
    stop = sessions_cmd._line(Event(type=EventType.USER_STOP, data={"how": "/stop"}))  # noqa: SLF001
    assert stop is not None and "人叫停" in stop

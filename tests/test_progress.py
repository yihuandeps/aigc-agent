"""任务进度窗 —— 事件驱动的状态更新与渲染。

ProgressBoard 只读事件总线不改状态，所以测试不碰终端、不开 Live：
驱动真实 EventBus，断言板上的状态与渲染文本。
"""

from __future__ import annotations

from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.interfaces.cli.progress import ProgressBoard


class _FakeConsole:
    is_terminal = False  # 非终端不开 Live


def _board() -> ProgressBoard:
    return ProgressBoard(_FakeConsole())  # type: ignore[arg-type]


async def test_批量进度事件驱动进度条():
    bus = EventBus()
    board = _board()
    bus.subscribe(board.on_event)

    await bus.emit(EventType.LOOP_START, turn=0)
    await bus.emit(EventType.BATCH_PROGRESS, stage="渲染分镜视频", done=0, total=10)
    await bus.emit(EventType.TOOL_CALL, tool="gen_video")
    await bus.emit(EventType.BATCH_PROGRESS, stage="渲染分镜视频", done=3, total=10)
    await bus.emit(EventType.TOOL_RESULT, tool="gen_video")

    assert board.stage == "渲染分镜视频"
    assert board.done == 3 and board.total == 10
    assert "gen_video" not in board.active

    text = str(board.__rich__().renderable)
    assert "渲染分镜视频" in text and "3/10" in text and "█" in text


async def test_工具运行计数与阶段映射():
    bus = EventBus()
    board = _board()
    bus.subscribe(board.on_event)

    await bus.emit(EventType.TOOL_CALL, tool="gen_image")
    await bus.emit(EventType.TOOL_CALL, tool="gen_image")
    assert board.active["gen_image"] == 2
    assert board.stage == "生成图片"

    await bus.emit(EventType.TOOL_ERROR, tool="gen_image", error="x")
    assert board.active["gen_image"] == 1
    await bus.emit(EventType.TOOL_RESULT, tool="gen_image")
    assert board.active == {}


async def test_成本累计与迭代显示():
    bus = EventBus()
    board = _board()
    bus.subscribe(board.on_event)

    await bus.emit(EventType.ITERATION_START, turn=0, iteration=3)
    await bus.emit(EventType.COST, cost=0.5)
    await bus.emit(EventType.COST, cost=0.7)

    assert board.iterations == 3
    assert abs(board.cost - 1.2) < 1e-9
    text = str(board.__rich__().renderable)
    assert "迭代 3" in text and "¥1.200" in text


async def test_每轮开始重置进度():
    bus = EventBus()
    board = _board()
    bus.subscribe(board.on_event)

    await bus.emit(EventType.BATCH_PROGRESS, stage="渲染参考图", done=5, total=5)
    await bus.emit(EventType.LOOP_START, turn=1)
    assert board.done == 0 and board.total == 0 and board.stage == "准备中…"


def test_非终端不开Live():
    board = _board()
    with board.running():
        pass
    assert board._live is None
    # paused 在没开 Live 时是空操作，不炸
    with board.paused():
        pass

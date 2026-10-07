"""2026-09-29 优化审查 1.9：发事件的 create_task 没保存任务引用。

资产落库、集长提示、/stop 这几处都是在同步代码里 `create_task(bus.emit(...))`、不留返回值 ——
事件循环对任务只持弱引用，任务理论上可能跑到一半被回收：资产事件丢了，按集流水漏接。
现在统一走 EventBus.emit_soon：引用由总线留着，跑完才放；任务本身出错交给事件循环的
异常处理记一笔，不静默丢，也不拖到回收时才冒「Task exception was never retrieved」。
"""

from __future__ import annotations

import asyncio
import gc
import re
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.harness.events.bus import Event, EventBus, EventType

SRC = Path(__file__).resolve().parents[1] / "src" / "aigc_agent"


async def test_emit_soon_引用留到跑完为止():
    bus = EventBus()
    got: list[str] = []

    async def slow(ev: Event) -> None:
        await asyncio.sleep(0)
        got.append(ev.data["message"])

    bus.subscribe(slow)
    task = bus.emit_soon(EventType.WARNING, message="集长改了")
    assert task is not None and task in bus._soon  # noqa: SLF001 — 跑着的时候总线拿着它
    ev = await task
    await asyncio.sleep(0)
    assert got == ["集长改了"] and ev.type is EventType.WARNING
    assert bus.history == [ev]
    assert not bus._soon, "跑完要放掉，不然越攒越多"  # noqa: SLF001


async def test_emit_soon_发出去的任务不会被中途回收():
    """调用点都不留返回值。订阅者等的东西只有任务自己拿着时，没人留引用的任务一次 gc 就没了。"""
    bus = EventBus()
    waits: list[weakref.ref[asyncio.Future[None]]] = []
    done: list[str] = []

    async def handler(ev: Event) -> None:
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        waits.append(weakref.ref(fut))  # 只留弱引用：除了任务本身没人拿着它
        await fut
        done.append(ev.data["id"])

    bus.subscribe(handler)
    bus.emit_soon(EventType.ASSET_CREATED, id="as_1")  # 和调用点一样，不留返回值
    await asyncio.sleep(0)  # 让任务跑到 await fut
    gc.collect()
    fut = waits[0]()
    assert fut is not None, "任务跑到一半被回收，事件丢在半路"
    fut.set_result(None)
    for _ in range(5):
        await asyncio.sleep(0)
    assert done == ["as_1"]


def test_emit_soon_没有运行中的事件循环就不发():
    bus = EventBus()
    assert bus.emit_soon(EventType.WARNING, message="x") is None
    assert bus.history == [] and not bus._soon  # noqa: SLF001


async def test_emit_soon_任务出错_交给事件循环记一笔_不等回收时才报():
    loop = asyncio.get_running_loop()
    seen: list[dict[str, Any]] = []
    old = loop.get_exception_handler()
    loop.set_exception_handler(lambda _lp, ctx: seen.append(ctx))
    try:
        bus = EventBus()
        task = bus.emit_soon("不是事件类型", x=1)  # type: ignore[arg-type]
        assert task is not None
        for _ in range(5):
            await asyncio.sleep(0)
        assert task.done() and not bus._soon  # noqa: SLF001
        assert len(seen) == 1, seen
        assert "没发成" in seen[0]["message"] and seen[0]["exception"] is not None
        assert seen[0]["task"] is task
        del task
        gc.collect()
        await asyncio.sleep(0)
        assert len(seen) == 1, "已经报过了，回收时不该再冒一句 never retrieved"
        assert bus.history == []
    finally:
        loop.set_exception_handler(old)


async def test_订阅者崩了照旧兜住_不算任务出错():
    loop = asyncio.get_running_loop()
    seen: list[dict[str, Any]] = []
    old = loop.get_exception_handler()
    loop.set_exception_handler(lambda _lp, ctx: seen.append(ctx))
    try:
        bus = EventBus()
        got: list[EventType] = []

        def bad(ev: Event) -> None:
            raise ValueError("订阅者自己的 bug")

        bus.subscribe(bad)
        bus.subscribe(lambda ev: got.append(ev.type))
        task = bus.emit_soon(EventType.USER_STOP, how="/stop")
        assert task is not None
        await task
        assert got == [EventType.USER_STOP] and seen == []
    finally:
        loop.set_exception_handler(old)


# ---------------------------------------------------------------- 调用点


async def test_资产落库事件走emit_soon_跑的时候总线拿着引用():
    bus = EventBus()
    seen: list[tuple[str, int]] = []

    async def handler(ev: Event) -> None:
        if ev.type is EventType.ASSET_CREATED:
            seen.append((ev.data["id"], len(bus._soon)))  # noqa: SLF001

    bus.subscribe(handler)
    store = AssetStore()
    store.bus = bus
    a = store.create("正文", summary="第1集")
    for _ in range(5):
        await asyncio.sleep(0)
    assert seen == [(a.id, 1)]
    assert not bus._soon  # noqa: SLF001


async def test_线程里建的资产_事件投回主循环_任务建在主循环上且留着引用():
    bus = EventBus()
    main = asyncio.get_running_loop()
    main_thread = threading.get_ident()
    seen: list[tuple[str, bool, bool, int]] = []

    async def handler(ev: Event) -> None:
        if ev.type is EventType.ASSET_CREATED:
            seen.append((
                ev.data["id"],
                asyncio.get_running_loop() is main,
                threading.get_ident() == main_thread,
                len(bus._soon),  # noqa: SLF001
            ))

    bus.subscribe(handler)
    store = AssetStore()
    store.bus = bus
    store.bind_loop()
    a = await asyncio.to_thread(lambda: store.create("x", summary="线程里建的"))
    for _ in range(50):
        if seen:
            break
        await asyncio.sleep(0.01)
    assert seen == [(a.id, True, True, 1)], seen
    await asyncio.sleep(0)
    assert not bus._soon  # noqa: SLF001


async def test_集长提示走emit_soon():
    from aigc_agent.app import Agent

    bus = EventBus()
    holder = SimpleNamespace(bus=bus)
    Agent._length_note(holder, "每集 8 分钟")  # type: ignore[arg-type]  # noqa: SLF001
    assert len(bus._soon) == 1  # noqa: SLF001
    for _ in range(3):
        await asyncio.sleep(0)
    notes = [e.data["message"] for e in bus.history if e.type is EventType.WARNING]
    assert notes == ["这个项目的集长设为每集 8 分钟，/length 可改"]
    assert not bus._soon  # noqa: SLF001


def test_集长提示_没有事件循环也不炸():
    from aigc_agent.app import Agent

    bus = EventBus()
    Agent._length_note(SimpleNamespace(bus=bus), "每集 8 分钟")  # type: ignore[arg-type]  # noqa: SLF001
    assert bus.history == []


_SKIP = {
    "harness/events/bus.py",  # emit_soon 自己的说明里写着旧写法
}


def test_没有地方再用不留引用的create_task发总线事件():
    """/stop 那处（interfaces/cli/main.py）在交互循环里，单测够不着，按源码兜一道。"""
    bad = re.compile(r"(?:create_task|ensure_future)\(\s*[\w.()]*\bbus\.emit\(")
    hits = []
    for f in sorted(SRC.rglob("*.py")):
        rel = f.relative_to(SRC).as_posix()
        if rel in _SKIP:
            continue
        if bad.search(f.read_text(encoding="utf-8")):
            hits.append(rel)
    assert hits == [], f"改用 bus.emit_soon：{hits}"

"""RPA 浏览器目录同一时间只给一个会话（2026-09-29 优化审查 1.5）。

9-23 同一轮里并发调用 douyin_hot_rpa 和 xhs_collect，两个都用 workspace/rpa/profile：
第二个启动时浏览器提示「正在现有的浏览器会话中打开」，随后 TargetClosedError，模型对用户说
是环境配置问题。browser.py 之前没有任何锁，启动失败也不 stop playwright。

现在：同一进程按 profile 目录排队；别的窗口占着立刻给一句人话；持锁的窗口死了锁跟着
没了；启动失败停掉 playwright、放锁、说人话。全程用假 playwright，不起真浏览器。
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest

from aigc_agent.domain.assets.store import AssetStore


class TargetClosedError(Exception):
    """和 playwright 的同名：浏览器一启动就退出（profile 被占）时抛的就是它。"""

    def __init__(self, message: str = "") -> None:
        super().__init__(message or "Target page, context or browser has been closed")


class _Ctx:
    def __init__(self, pw: _PW) -> None:
        self.pw = pw
        self.pages: list[Any] = []

    async def close(self) -> None:
        self.pw.log.append("close")


class _Chromium:
    def __init__(self, pw: _PW) -> None:
        self.pw = pw

    async def launch_persistent_context(self, user_data_dir: str, **opts: Any) -> _Ctx:
        self.pw.launches.append(opts.get("channel", ""))
        await asyncio.sleep(0.01)  # 让并发的另一个有机会插进来
        if self.pw.fail:
            raise self.pw.fail.pop(0)
        self.pw.log.append("launch")
        return _Ctx(self.pw)


class _PW:
    def __init__(self, fail: list[BaseException]) -> None:
        self.log: list[str] = []
        self.launches: list[str] = []
        self.fail = fail
        self.stopped = 0
        self.chromium = _Chromium(self)

    async def stop(self) -> None:
        self.stopped += 1


def _fake_playwright(monkeypatch: pytest.MonkeyPatch, *fail: BaseException) -> _PW:
    """假的 playwright.async_api：BrowserSession 照常 import，拿到的是这里的桩。"""
    from aigc_agent.domain.rpa import browser

    pw = _PW(list(fail))

    class _Starter:
        async def start(self) -> _PW:
            pw.log.append("start")
            return pw

    mod = types.ModuleType("playwright.async_api")
    mod.async_playwright = _Starter  # type: ignore[attr-defined]
    if "playwright" not in sys.modules:
        monkeypatch.setitem(sys.modules, "playwright", types.ModuleType("playwright"))
    monkeypatch.setitem(sys.modules, "playwright.async_api", mod)
    # 别依赖这台机器装没装 Chrome
    monkeypatch.setattr(browser, "system_browser", lambda: ("chrome", ""))
    return pw


async def _open_close(profile: Path) -> None:
    from aigc_agent.domain.rpa import browser

    async with browser.BrowserSession(profile):
        pass


# ---------------------------------------------------------------- 同一进程排队


async def test_同一进程两个RPA调用按浏览器目录排队(tmp_path: Path, monkeypatch):
    from aigc_agent.domain.rpa import browser

    pw = _fake_playwright(monkeypatch)
    profile = tmp_path / "rpa" / "profile"
    trace: list[str] = []

    async def use(tag: str) -> None:
        async with browser.BrowserSession(profile):
            trace.append(f"{tag}+")
            await asyncio.sleep(0.05)
            trace.append(f"{tag}-")

    await asyncio.gather(use("a"), use("b"))
    assert trace in (["a+", "a-", "b+", "b-"], ["b+", "b-", "a+", "a-"]), trace
    # 第二个是在第一个关掉浏览器之后才起的
    assert pw.log == ["start", "launch", "close", "start", "launch", "close"], pw.log
    assert pw.stopped == 2


async def test_不同浏览器目录互不排队(tmp_path: Path, monkeypatch):
    from aigc_agent.domain.rpa import browser

    _fake_playwright(monkeypatch)
    trace: list[str] = []

    async def use(profile: Path, tag: str) -> None:
        async with browser.BrowserSession(profile):
            trace.append(f"{tag}+")
            await asyncio.sleep(0.05)
            trace.append(f"{tag}-")

    await asyncio.gather(use(tmp_path / "p1", "a"), use(tmp_path / "p2", "b"))
    assert trace[:2] == ["a+", "b+"], trace


async def test_排队太久给人话不死等(tmp_path: Path, monkeypatch):
    from aigc_agent.domain.rpa import browser

    _fake_playwright(monkeypatch)
    monkeypatch.setattr(browser, "QUEUE_WAIT", 0.1)
    profile = tmp_path / "rpa" / "profile"
    entered = asyncio.Event()
    leave = asyncio.Event()

    async def holder() -> None:
        async with browser.BrowserSession(profile):
            entered.set()
            await leave.wait()

    task = asyncio.create_task(holder())
    await entered.wait()
    try:
        with pytest.raises(browser.BrowserUnavailable) as ei:
            await _open_close(profile)
        assert "排" in str(ei.value) and "再" in str(ei.value)
    finally:
        leave.set()
        await task
    await _open_close(profile)  # 前一个用完了，后面的照常能用


# ---------------------------------------------------------------- 跨窗口


async def test_别的窗口占着浏览器目录_立刻说人话不干等(tmp_path: Path, monkeypatch):
    from aigc_agent.capabilities.memory.session import SessionLock
    from aigc_agent.domain.rpa import browser

    pw = _fake_playwright(monkeypatch)
    profile = tmp_path / "rpa" / "profile"
    other = SessionLock(browser.lock_path(profile))  # 另一个窗口手里的锁
    assert other.acquire()
    try:
        t0 = time.perf_counter()
        with pytest.raises(browser.BrowserUnavailable) as ei:
            await _open_close(profile)
        assert time.perf_counter() - t0 < 2, "被别的窗口占着要立刻报，不能干等"
        msg = str(ei.value)
        assert "另一个" in msg and "窗口" in msg and "再试" in msg
        assert pw.log == [], "别的窗口占着就不该再起浏览器"
    finally:
        other.release()
    await _open_close(profile)  # 那边用完了，这边就能用
    assert pw.log == ["start", "launch", "close"]


_HOLDER = """
import sys, time
from pathlib import Path
from aigc_agent.capabilities.memory.session import SessionLock
lock = SessionLock(Path(sys.argv[1]))
Path(sys.argv[2]).write_text("locked" if lock.acquire() else "busy", encoding="utf-8")
time.sleep(120)
"""


def test_持锁的窗口死了_锁自动接管(tmp_path: Path, monkeypatch):
    """另一个窗口崩了 / 被杀了：锁文件还留在磁盘上，但锁跟着进程没了，不用手动删。"""
    from aigc_agent.domain.rpa import browser

    _fake_playwright(monkeypatch)
    profile = tmp_path / "rpa" / "profile"
    lock_file = browser.lock_path(profile)
    ready = tmp_path / "ready.txt"
    env = {**os.environ, "AIGC_WORKSPACE": str(tmp_path / "ws")}
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(lock_file), str(ready)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 60
        while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists(), "持锁的子进程没起来"
        assert ready.read_text(encoding="utf-8") == "locked"
        with pytest.raises(browser.BrowserUnavailable):
            asyncio.run(_open_close(profile))
    finally:
        proc.kill()
        proc.wait(timeout=30)
    assert lock_file.exists(), "死进程不会收拾锁文件"
    # 系统回收死进程的锁可能慢一拍
    deadline = time.monotonic() + 10
    while True:
        try:
            asyncio.run(_open_close(profile))
            break
        except browser.BrowserUnavailable:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)


# ---------------------------------------------------------------- 启动失败


async def test_浏览器一启动就退出_停playwright放锁说人话(tmp_path: Path, monkeypatch):
    from aigc_agent.capabilities.memory.session import SessionLock
    from aigc_agent.domain.rpa import browser

    pw = _fake_playwright(
        monkeypatch,
        TargetClosedError(
            "BrowserType.launch_persistent_context: Target page, context or browser has been "
            "closed\nBrowser logs:\n正在现有的浏览器会话中打开。"
        ),
    )
    profile = tmp_path / "rpa" / "profile"
    with pytest.raises(browser.BrowserUnavailable) as ei:
        await _open_close(profile)
    msg = str(ei.value)
    assert "占着" in msg and "关掉" in msg and "不是环境配置问题" in msg
    assert "TargetClosedError" not in msg and "Browser logs" not in msg
    assert pw.stopped == 1, "启动失败也要停掉 playwright"
    assert pw.launches == ["chrome"], "目录被占时换自带 Chromium 也是一样被占，不该再试"
    # 锁放了：别的窗口马上能占，同一进程的下一次也不会卡在排队上
    other = SessionLock(browser.lock_path(profile))
    assert other.acquire()
    other.release()
    await _open_close(profile)


async def test_浏览器一启动就没了但没说为什么_也给人话(tmp_path: Path, monkeypatch):
    from aigc_agent.domain.rpa import browser

    pw = _fake_playwright(monkeypatch, TargetClosedError(), TargetClosedError())
    with pytest.raises(browser.BrowserUnavailable) as ei:
        await _open_close(tmp_path / "rpa" / "profile")
    msg = str(ei.value)
    assert "一启动就退出" in msg and "占着" in msg and "关掉" in msg
    assert "TargetClosedError" not in msg and "has been closed" not in msg
    assert pw.launches == ["chrome", ""], "没说是目录被占，照旧退回自带 Chromium 试一次"
    assert pw.stopped == 1


async def test_系统浏览器起不来_退回自带Chromium(tmp_path: Path, monkeypatch):
    pw = _fake_playwright(monkeypatch, RuntimeError("chrome 版本太旧"))
    await _open_close(tmp_path / "rpa" / "profile")
    assert pw.launches == ["chrome", ""]
    assert pw.log == ["start", "launch", "close"]


async def test_哪个浏览器都没有_说清楚装什么(tmp_path: Path, monkeypatch):
    from aigc_agent.domain.rpa import browser

    pw = _fake_playwright(
        monkeypatch,
        RuntimeError("Executable doesn't exist at C:\\Program Files\\Google\\chrome.exe"),
        RuntimeError(
            "BrowserType.launch_persistent_context: Executable doesn't exist at C:\\ms-playwright"
            "\\chromium\\chrome.exe\nLooks like Playwright was just installed or updated."
        ),
    )
    with pytest.raises(browser.BrowserUnavailable) as ei:
        await _open_close(tmp_path / "rpa" / "profile")
    msg = str(ei.value)
    assert "Chrome" in msg and "playwright install" in msg
    assert pw.stopped == 1


# ---------------------------------------------------------------- 工具层


def _rpa(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, collect: Any):
    from aigc_agent.domain.functions import rpa as rpa_mod

    monkeypatch.setattr(rpa_mod, "have_playwright", lambda: True)
    monkeypatch.setattr(rpa_mod, "collect_xiaohongshu", collect)
    return rpa_mod.RpaFunctions(AssetStore(tmp_path / "assets"), tmp_path / "ws")


async def test_RPA工具遇到别的窗口占着_把人话原样给模型(tmp_path: Path, monkeypatch):
    from aigc_agent.capabilities.memory.session import SessionLock
    from aigc_agent.domain.rpa import browser

    _fake_playwright(monkeypatch)
    called: list[str] = []

    async def collect(*a: Any, **k: Any):
        called.append("collect")
        return [], ""

    fns = _rpa(monkeypatch, tmp_path, collect)
    other = SessionLock(browser.lock_path(fns.profile))
    assert other.acquire()
    try:
        r = await fns.invoke("xhs_collect", {"keyword": "露营"})
    finally:
        other.release()
    assert not r.ok and called == []
    assert "另一个" in r.error and "BrowserUnavailable" not in r.error


async def test_RPA工具采集中途浏览器被关_不甩原始异常(tmp_path: Path, monkeypatch):
    _fake_playwright(monkeypatch)

    async def collect(*a: Any, **k: Any):
        raise TargetClosedError()

    r = await _rpa(monkeypatch, tmp_path, collect).invoke("xhs_collect", {})
    assert not r.ok
    assert "TargetClosedError" not in r.error and "has been closed" not in r.error
    assert "关" in r.error and "不是环境配置问题" in r.error


async def test_RPA工具的超时留够排队的时间(tmp_path: Path):
    from aigc_agent.domain.functions.rpa import RpaFunctions
    from aigc_agent.domain.rpa import browser

    fns = RpaFunctions(AssetStore(tmp_path / "assets"), tmp_path / "ws")
    metas = await fns.list_tools()
    assert metas and all(m.timeout > browser.QUEUE_WAIT for m in metas), metas


async def test_登录命令遇到目录被占_说人话不甩traceback(tmp_path: Path, monkeypatch, capsys):
    import typer

    from aigc_agent.domain.rpa.browser import BrowserUnavailable
    from aigc_agent.interfaces.cli import rpa_cmd

    class Busy:
        def __init__(self, *a: Any, **k: Any) -> None: ...

        async def __aenter__(self) -> Any:
            raise BrowserUnavailable("另一个 Agent 窗口正在用这个浏览器目录，等那边用完再试")

        async def __aexit__(self, *a: Any) -> None: ...

    monkeypatch.setattr(rpa_cmd, "BrowserSession", Busy)
    monkeypatch.setattr(rpa_cmd, "detect", lambda: None)
    monkeypatch.setattr(rpa_cmd, "have_playwright", lambda: True)
    monkeypatch.setattr(rpa_cmd, "system_browser", lambda: ("chrome", ""))
    monkeypatch.setattr(rpa_cmd, "workspace_root", lambda: tmp_path)
    with pytest.raises(typer.Exit) as ei:
        await rpa_cmd._login("xhs", 1)
    assert ei.value.exit_code == 1
    assert "另一个 Agent 窗口正在用这个浏览器目录" in capsys.readouterr().out

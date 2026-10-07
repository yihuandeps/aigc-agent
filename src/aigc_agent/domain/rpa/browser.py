"""RPA 浏览器底座 —— 抖音 / 小红书。

**什么时候该用 RPA，什么时候不该**（这个判断比代码本身重要）：

| 需求 | 走哪条 | 原因 |
|---|---|---|
| 抖音热榜、搜索、视频详情 | TikHub API | 稳定、无需登录、不怕改版 |
| 小红书任何数据 | **RPA** | TikHub 该平台端点全部 402，免费额度不覆盖 |
| 个性化推荐流（你自己的首页） | **RPA** | API 拿不到「你的」推荐 |

RPA 是**兜底手段不是首选**：它依赖登录态、怕页面改版、速度慢。
能走 API 的绝不用它。

登录态持久化在 workspace/rpa/profile/，**第一次要你手动扫码**，
之后复用。不读取、不导出浏览器 Cookie 数据库。

两条实测约束（2026-09-11）：
  · **必须可见窗口**：headless 下小红书返回「安全限制」页，抖音同理
  · **用系统 Chrome/Edge**：Playwright 自带 Chromium 的 CDN 国内下不动，
    而且系统浏览器本来就是真实浏览器

**一个 profile 目录同一时间只能开一个浏览器**（2026-09-29 审查 1.5）：9-23 同一轮里并发调
douyin_hot_rpa 和 xhs_collect，第二个启动时浏览器提示「正在现有的浏览器会话中打开」随即退出，
报 TargetClosedError，模型对用户说成「环境配置问题」。所以 BrowserSession 按目录加锁：
  · 同一进程：asyncio 锁排队，前一个用完后一个再开
  · 别的窗口（另开的 Agent、agent rpa login）占着：立刻报一句人话，不干等
  · 跨窗口用的是操作系统文件锁（和会话快照的「第二个窗口锁」同一个），进程没了锁自动放，
    死进程留下的锁文件不会挡路
"""

from __future__ import annotations

import asyncio
import os
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 首次访问要等页面渲染，比一般站点慢
NAV_TIMEOUT = 45_000
SETTLE_MS = 2500

# 同一进程里排队最多等多久（秒）。单次采集按 config/rpa.yaml 最多跑 8–10 分钟；
# RPA 工具的调度超时要比它长（functions/rpa.py），排不上时给人话，而不是被调度器判超时
QUEUE_WAIT = 600.0


class BrowserUnavailable(RuntimeError):
    """浏览器这次用不了（目录被占、排不上、起不来）。消息是说给人听的整句，上层原样转给模型。"""


@dataclass
class Post:
    """抓到的一条内容。两个平台归一成同一形状。"""

    platform: str
    title: str = ""
    url: str = ""
    post_id: str = ""
    author: str = ""
    likes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def line(self, i: int) -> str:
        bits = [f"{i:2}. {self.title[:44]}"]
        if self.author:
            bits.append(f"@{self.author}")
        if self.likes:
            bits.append(f"赞 {self.likes}")
        return "  ".join(bits) + (f"\n     {self.url}" if self.url else "")


# 优先用系统已装的浏览器，不下 Playwright 自带的 Chromium。
# 原因：那个包 150MB+ 且 CDN 在国内常年下不动；而 Chrome/Edge 基本都有。
_CHANNELS = ["chrome", "msedge"]
_BROWSER_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def have_playwright() -> bool:
    try:
        import playwright  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def system_browser() -> tuple[str, str]:
    """返回 (channel, executable_path)，都为空表示只能用自带 Chromium。"""
    import shutil  # noqa: PLC0415

    for path in _BROWSER_PATHS:
        if Path(path).exists():
            return ("chrome" if "chrome.exe" in path.lower() else "msedge"), path
    for ch in _CHANNELS:
        if shutil.which(ch):
            return ch, ""
    return "", ""


def lock_path(profile_dir: Path) -> Path:
    """跨窗口锁文件：放在 profile 目录旁边（rpa/profile.lock），不往浏览器自己的数据目录里塞。"""
    return profile_dir.parent / f"{profile_dir.name}.lock"


# 同一进程里按 profile 目录排队的锁。asyncio.Lock 绑事件循环（测试、CLI 每次 asyncio.run
# 都是新循环），所以按循环分开存，循环没了跟着回收
_QUEUES: weakref.WeakKeyDictionary[Any, dict[str, asyncio.Lock]] = weakref.WeakKeyDictionary()


def _queue_of(profile_dir: Path) -> asyncio.Lock:
    try:
        key = os.path.normcase(str(profile_dir.resolve()))
    except OSError:
        key = os.path.normcase(str(profile_dir.absolute()))
    locks = _QUEUES.setdefault(asyncio.get_running_loop(), {})
    if key not in locks:
        locks[key] = asyncio.Lock()
    return locks[key]


# 浏览器自己说的：这个 profile 已经有浏览器开着，新进程把网址交给它就退出了
_HANDOFF_SIGNS = ("existing browser session", "现有的浏览器会话")
# 浏览器一启动就没了（目录被占时最常见，崩溃也是这个错）
_CLOSED_SIGNS = ("targetclosederror", "has been closed")
_NO_BROWSER_SIGNS = ("executable doesn't exist", "playwright install")


def _first_line(err: BaseException) -> str:
    """异常的第一行，去掉 playwright 的「BrowserType.xxx: 」前缀 —— 后面的浏览器日志和
    调用栈不给模型看。"""
    lines = [s.strip() for s in str(err).splitlines() if s.strip()]
    line = lines[0] if lines else type(err).__name__
    head, sep, rest = line.partition(": ")
    if sep and head.startswith("BrowserType."):
        line = rest
    return line[:160]


def _looks(err: BaseException, signs: tuple[str, ...]) -> bool:
    text = f"{type(err).__name__} {err}".lower()
    return any(s in text for s in signs)


def _launch_hint(errors: list[BaseException], profile_dir: Path) -> str:
    """浏览器起不来时说给人听的话：发生了什么、该怎么办。原始异常不往上甩。"""
    if any(_looks(e, _HANDOFF_SIGNS) for e in errors):
        return (
            f"RPA 登录态目录 {profile_dir} 已经被另一个浏览器占着"
            "（浏览器提示「正在现有的浏览器会话中打开」后就退出了）——多半是上次没关掉的"
            " RPA 浏览器窗口，或者另一个 Agent 窗口还在采集。"
            "把那个浏览器窗口关掉再试；这不是环境配置问题，不用改设置。"
        )
    if any(_looks(e, _CLOSED_SIGNS) for e in errors):
        return (
            "浏览器一启动就退出了。最常见的原因是 RPA 登录态目录"
            f" {profile_dir} 正被另一个浏览器窗口占着（上次没关掉的 RPA 浏览器，"
            "或者另一个 Agent 窗口在采集）：把它关掉再试。"
            "还不行就跑 agent rpa status 看看浏览器和登录态。"
        )
    if all(_looks(e, _NO_BROWSER_SIGNS) for e in errors):
        return (
            "找不到能用的浏览器：系统的 Chrome / Edge 起不来，Playwright 自带的 Chromium 也没下载。"
            "装上 Chrome 或 Edge 再试（或者跑 playwright install chromium）。"
        )
    return (
        f"浏览器没能启动（{_first_line(errors[0])}）。"
        "先跑 agent rpa status 看浏览器和登录态，确认没问题再重试。"
    )


class BrowserSession:
    """持久化上下文的浏览器会话。

    用 launch_persistent_context 而不是每次新建：小红书/抖音都要登录，
    每次重新扫码没法用。profile 存在 workspace 下，不碰系统浏览器的数据。

    同一个 profile 同一时间只给一个会话（见模块说明）：进入时先占锁，退出或启动失败时放。
    """

    def __init__(self, profile_dir: Path, headless: bool = False) -> None:
        self.profile_dir = profile_dir
        # 实测 2026-09-11：headless 下小红书直接返回「安全限制」页，
        # 可见窗口一切正常。抖音同理。所以这里**恒为可见**——
        # 后台静默跑不通，接受这个约束比想办法绕过它更合适。
        self.headless = False
        if headless:
            self._headless_requested = True
        self._pw: Any = None
        self._ctx: Any = None
        self._queue: asyncio.Lock | None = None  # 同一进程排队用的锁（占到了才有）
        self._window: Any = None  # 跨窗口的文件锁（占到了才有）

    async def __aenter__(self) -> BrowserSession:
        await self._claim()
        try:
            await self._launch()
        except BaseException:
            # 起不来也要停掉 playwright、放锁：之前 driver 进程会一直挂着，锁也没人放
            await self._shutdown()
            raise
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self._shutdown()

    async def _claim(self) -> None:
        """占住 profile 目录：同一进程排队；别的窗口占着就立刻报，不干等。"""
        queue = _queue_of(self.profile_dir)
        try:
            async with asyncio.timeout(QUEUE_WAIT):
                await queue.acquire()
        except TimeoutError:
            raise BrowserUnavailable(
                f"前面还有 RPA 采集在用这个浏览器（登录态目录 {self.profile_dir}），"
                f"排了 {QUEUE_WAIT:.0f} 秒还没轮到。等它跑完再调；"
                "同一轮里别同时发好几个 RPA 工具，一个一个来。"
            ) from None
        self._queue = queue

        # 和会话快照的「第二个窗口锁」同一个实现：操作系统级文件锁，进程没了自动放
        from ...capabilities.memory.session import SessionLock  # noqa: PLC0415

        window = SessionLock(lock_path(self.profile_dir))
        if not window.acquire():
            self._release()
            raise BrowserUnavailable(
                f"另一个窗口正在用 RPA 浏览器（登录态目录 {self.profile_dir}）——"
                "另开的 Agent 窗口在采集，或者 agent rpa login 还没关。"
                "同一个目录同一时间只能开一个浏览器：等那边用完再试，这不是环境配置问题。"
            )
        self._window = window

    async def _launch(self) -> None:
        from playwright.async_api import async_playwright  # noqa: PLC0415

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._pw = await async_playwright().start()
        except Exception as e:  # noqa: BLE001
            raise BrowserUnavailable(
                f"Playwright 没能启动（{_first_line(e)}）。"
                "重装一下再试：pip install -U playwright"
            ) from e

        opts: dict[str, Any] = {
            "headless": self.headless,
            "viewport": {"width": 1440, "height": 900},
            "locale": "zh-CN",
            "timezone_id": "Asia/Shanghai",
        }
        attempts = []
        channel, exe = system_browser()
        if channel or exe:
            sys_opts = dict(opts)
            if channel:
                sys_opts["channel"] = channel
            if exe:
                sys_opts["executable_path"] = exe
            attempts.append(sys_opts)
        attempts.append(opts)  # 系统浏览器起不来就退回自带 Chromium（若已下载）

        errors: list[BaseException] = []
        for o in attempts:
            try:
                self._ctx = await self._pw.chromium.launch_persistent_context(
                    str(self.profile_dir), **o
                )
                return
            except Exception as e:  # noqa: BLE001
                errors.append(e)
                if _looks(e, _HANDOFF_SIGNS):
                    break  # 目录被占：换自带 Chromium 也是同一个目录，一样被占
        raise BrowserUnavailable(_launch_hint(errors, self.profile_dir)) from errors[0]

    async def _shutdown(self) -> None:
        """关浏览器、停 playwright、放锁。每一步单独兜住：前一步出错后面照样要做，
        也不让收尾的错盖掉采集本身的结果或异常。"""
        ctx, self._ctx = self._ctx, None
        pw, self._pw = self._pw, None
        try:
            if ctx is not None:
                try:
                    await ctx.close()
                except Exception:  # noqa: BLE001 — 窗口被手动关了时 close 会报错，不要紧
                    pass
            if pw is not None:
                try:
                    await pw.stop()
                except Exception:  # noqa: BLE001
                    pass
        finally:
            self._release()

    def _release(self) -> None:
        window, self._window = self._window, None
        if window is not None:
            window.release()
        queue, self._queue = self._queue, None
        if queue is not None and queue.locked():
            queue.release()

    @property
    def context(self) -> Any:
        """底层 BrowserContext。授予剪贴板权限等场景要用。"""
        return self._ctx

    async def page(self) -> Any:
        pages = self._ctx.pages
        p = pages[0] if pages else await self._ctx.new_page()
        p.set_default_timeout(NAV_TIMEOUT)
        return p

    async def goto(self, url: str, settle_ms: int = SETTLE_MS) -> Any:
        p = await self.page()
        await p.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
        await asyncio.sleep(settle_ms / 1000)
        return p

    async def scroll(self, page: Any, times: int = 3, pace: Any = None) -> None:
        """滚动加载更多。两个平台都是无限流，不滚只能拿到首屏几条。

        节奏由 humanize 控制 —— 固定距离 + 固定间隔的机械滚动
        是最容易被风控盯上的模式。
        """
        from .humanize import Pace, human_scroll  # noqa: PLC0415

        await human_scroll(page, pace or Pace(), times)


LOGIN_HINTS = ("扫码登录", "登录后查看", "立即登录", "手机号登录", "新用户扫码登录")

# 登录后才会出现的元素。只看「有没有登录提示」不够 ——
# 游客能浏览的页面（如抖音首页）没有提示，会被误判成已登录。
_SIGNED_IN = (
    "img[class*=avatar]",
    "[class*=user-info]",
    "[class*=avatar] img",
    "a[href*='/user/self']",
)


async def is_logged_in(session: BrowserSession, platform: str = "") -> tuple[bool, str]:
    """判登录态，返回 (是否已登录, 依据)。

    两个信号都看：
      · 页面上还在劝你登录   → 明确没登
      · 找不到头像类元素     → 也算没登（游客态）
    实测：抖音首页允许游客浏览、无登录提示，但头像元素为 0；
    真登录后小红书能找到 30 个。单看提示会误判。
    """
    page = await session.page()
    try:
        content = await page.content()
    except Exception:  # noqa: BLE001
        return False, "页面读不到"

    if any(h in content for h in LOGIN_HINTS):
        return False, "页面仍在提示登录"

    found = 0
    for sel in _SIGNED_IN:
        try:
            found += len(await page.query_selector_all(sel))
        except Exception:  # noqa: BLE001
            continue
    if found == 0:
        return False, "没有登录提示，但也找不到头像元素（多半是游客态）"
    return True, f"找到 {found} 个登录态元素"

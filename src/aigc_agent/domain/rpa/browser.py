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
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 首次访问要等页面渲染，比一般站点慢
NAV_TIMEOUT = 45_000
SETTLE_MS = 2500


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


class BrowserSession:
    """持久化上下文的浏览器会话。

    用 launch_persistent_context 而不是每次新建：小红书/抖音都要登录，
    每次重新扫码没法用。profile 存在 workspace 下，不碰系统浏览器的数据。
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

    async def __aenter__(self) -> BrowserSession:
        from playwright.async_api import async_playwright  # noqa: PLC0415

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._pw = await async_playwright().start()

        opts: dict[str, Any] = {
            "headless": self.headless,
            "viewport": {"width": 1440, "height": 900},
            "locale": "zh-CN",
            "timezone_id": "Asia/Shanghai",
        }
        channel, exe = system_browser()
        if channel:
            opts["channel"] = channel
        if exe:
            opts["executable_path"] = exe

        try:
            self._ctx = await self._pw.chromium.launch_persistent_context(
                str(self.profile_dir), **opts
            )
        except Exception:
            # 系统浏览器起不来就退回自带 Chromium（若已下载）
            opts.pop("channel", None)
            opts.pop("executable_path", None)
            self._ctx = await self._pw.chromium.launch_persistent_context(
                str(self.profile_dir), **opts
            )
        return self

    async def __aexit__(self, *_: Any) -> None:
        try:
            if self._ctx:
                await self._ctx.close()
        finally:
            if self._pw:
                await self._pw.stop()

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

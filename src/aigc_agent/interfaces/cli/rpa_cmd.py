"""`agent rpa` —— 登录与采集。

登录态存在 workspace/rpa/profile/，扫一次之后复用。
"""

from __future__ import annotations

import asyncio
import time

import typer
from rich.console import Console
from rich.panel import Panel

from ...domain.rpa.browser import (
    BrowserSession,
    have_playwright,
    is_logged_in,
    system_browser,
)
from ...envdetect import PROJECT_ROOT, detect, workspace_root  # noqa: F401

console = Console()
app = typer.Typer(help="浏览器采集（小红书/抖音）", no_args_is_help=True)

SITES = {
    "xhs": ("小红书", "https://www.xiaohongshu.com/explore"),
    "douyin": ("抖音", "https://www.douyin.com/"),
}

@app.command()
def login(
    site: str = typer.Argument("xhs", help="xhs / douyin"),
    wait: int = typer.Option(300, "--wait", "-w", help="最多等你几秒"),
) -> None:
    """打开浏览器扫码登录。登录态会记住，之后采集不用再扫。"""
    if site not in SITES:
        console.print(f"[red]不认识 {site!r}，可选：{', '.join(SITES)}[/]")
        raise typer.Exit(1)
    asyncio.run(_login(site, wait))


async def _login(site: str, wait: int) -> None:
    detect()
    if not have_playwright():
        console.print("[red]未装 playwright[/]：pip install playwright")
        raise typer.Exit(1)

    name, url = SITES[site]
    browser, _ = system_browser()
    profile = workspace_root() / "rpa" / "profile"

    console.print(
        Panel(
            f"即将打开 [bold]{browser or 'chromium'}[/] 访问 [bold]{name}[/]\n"
            f"[yellow]请在弹出的窗口里扫码/登录[/]，我会自动检测，登录成功就收工。\n"
            f"[dim]登录态存在 {profile}，之后复用。最多等 {wait} 秒。[/]",
            title="RPA 登录",
            border_style="cyan",
        )
    )

    async with BrowserSession(profile) as s:
        await s.goto(url, settle_ms=3000)
        t0 = time.perf_counter()
        was_login_wall = False
        last = ""

        while time.perf_counter() - t0 < wait:
            ok, why = await is_logged_in(s, site)
            was_login_wall = was_login_wall or not ok
            state = ("已登录 — " if ok else "等待登录 — ") + why
            if state != last:
                console.print(f"  [dim]{int(time.perf_counter() - t0):3}s  {state}[/]")
                last = state

            if ok:
                # 连续两次确认，避免页面还没渲染完就误判
                await asyncio.sleep(4)
                again, _ = await is_logged_in(s, site)
                if again:
                    console.print(
                        f"\n[green]✓ {name} 已登录[/]（用时 {time.perf_counter() - t0:.0f}s）"
                    )
                    console.print("[dim]登录态已保存，现在可以跑采集了：[/]")
                    console.print(f'  agent rpa collect --site {site} --keyword "关键词"')
                    return
            await asyncio.sleep(3)

        console.print(f"\n[yellow]等了 {wait}s 还没检测到登录[/]")
        if not was_login_wall:
            console.print("[dim]（页面上没出现登录提示，可能本来就登录着，直接试采集看看）[/]")


@app.command()
def collect(
    site: str = typer.Option("xhs", "--site", "-s", help="xhs / douyin"),
    keyword: str = typer.Option("", "--keyword", "-k", help="搜索词，留空抓推荐流"),
    limit: int = typer.Option(20, "--limit", "-n"),
    pace: str = typer.Option("default", "--pace", "-p", help="default/cautious/brisk"),
) -> None:
    """采集热门内容。节奏见 config/rpa.yaml。"""
    asyncio.run(_collect(site, keyword, limit, pace))


async def _collect(site: str, keyword: str, limit: int, pace: str) -> None:
    from ...app import Agent

    agent = Agent.create()
    await agent.setup(mcp=False)

    tool = "xhs_collect" if site == "xhs" else "douyin_hot_rpa"
    args = {"limit": limit} if site != "xhs" else {
        "keyword": keyword, "limit": limit, "pace": pace
    }

    console.print(f"[dim]采集中（{pace} 节奏，会比较慢，这是有意的）…[/]")
    r = await agent.registry.invoke(tool, args)
    if not r.ok:
        console.print(f"[red]✗ {r.error}[/]")
        raise typer.Exit(1)
    console.print(Panel(r.content[:3000], title=f"采集结果（{r.duration_ms / 1000:.0f}s）",
                        border_style="green"))


@app.command()
def status() -> None:
    """看登录态和浏览器就绪情况。"""
    detect()
    profile = workspace_root() / "rpa" / "profile"
    browser, path = system_browser()
    console.print(f"playwright: {'✓' if have_playwright() else '✗ 未安装'}")
    console.print(f"浏览器:     {browser or '（无系统浏览器，用自带 Chromium）'}")
    if path:
        console.print(f"            [dim]{path}[/]")
    has = (profile / "Default").exists()
    console.print(f"登录态:     {'✓ 有' if has else '✗ 无，先 agent rpa login'}")
    console.print(f"            [dim]{profile}[/]")

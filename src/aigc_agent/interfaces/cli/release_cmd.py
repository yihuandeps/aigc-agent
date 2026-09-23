"""`agent release` —— 待发布包的运维面。

上传是人的动作。这里只做三件事：看有哪些包、看某个包的上传清单、上传完把链接记回来。
"""

from __future__ import annotations

import asyncio
import datetime as _dt

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...app import Agent
from .quiet import install as quiet_shutdown_noise

console = Console()
app = typer.Typer(help="待发布包：列表 / 上传清单 / 记回发布链接", no_args_is_help=True)


@app.command("list")
def list_cmd() -> None:
    """列出已打的待发布包及发布状态。"""
    asyncio.run(_list())


async def _list() -> None:
    quiet_shutdown_noise()
    agent = Agent.create()
    try:
        rows = agent.packager.packages()
        if not rows:
            console.print(
                "[dim]还没有待发布包。chat 里让模型 build_release_package，或跑 graph run copy[/]"
            )
            return
        t = Table(show_header=True, header_style="bold")
        for col in ("资产 id", "平台", "类型", "标题", "机审", "状态", "链接 / 目录"):
            t.add_column(col)
        for a, m in rows:
            c = m.compliance.get("counts") or {}
            if not m.compliance:
                check = "[dim]未审[/]"
            elif m.compliance.get("passed"):
                check = "[green]通过[/]"
            else:
                check = f"[red]block {c.get('block', '?')}[/]"
            t.add_row(
                a.id, m.platform, m.kind, m.title[:24], check,
                "[green]已发布[/]" if m.status == "published" else "待上传",
                m.published_url or str(a.uri or ""),
            )
        console.print(t)
    finally:
        await agent.aclose()


@app.command("show")
def show_cmd(asset_id: str = typer.Argument(..., help="包的资产 id")) -> None:
    """看一个包的上传清单。"""
    asyncio.run(_show(asset_id))


async def _show(asset_id: str) -> None:
    quiet_shutdown_noise()
    agent = Agent.create()
    try:
        a = agent.assets.get(asset_id)
        m = agent.packager.manifest_of(a)
        console.print(
            Panel(agent.assets.content(asset_id), title=f"{a.id} · {a.uri}", border_style="cyan")
        )
        if m.published_url:
            when = _dt.datetime.fromtimestamp(m.published_at).strftime("%m-%d %H:%M")
            console.print(f"[green]已发布[/] {m.published_url}（{when}）")
    except (KeyError, ValueError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e
    finally:
        await agent.aclose()


@app.command("published")
def published_cmd(
    asset_id: str = typer.Argument(..., help="包的资产 id"),
    url: str = typer.Option(..., "--url", help="发布后的链接"),
) -> None:
    """人上传完，把链接记回包上。之后录数据靠它对号。"""
    asyncio.run(_published(asset_id, url))


async def _published(asset_id: str, url: str) -> None:
    quiet_shutdown_noise()
    agent = Agent.create(auto_approve=True)
    try:
        await agent.setup(mcp=False)
        r = await agent.registry.invoke(
            "mark_published", {"package_asset_id": asset_id, "url": url}
        )
        if not r.ok:
            console.print(f"[red]{r.error}[/]")
            raise typer.Exit(1)
        console.print(f"[green]✓[/] {r.content}")
    finally:
        await agent.aclose()

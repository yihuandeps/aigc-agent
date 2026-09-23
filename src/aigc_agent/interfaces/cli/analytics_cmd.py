"""`agent analytics` —— 数据回流的运维面（MVP：人工录入）。

  record   录一次发布后的数据。录入即复盘，满 3 个作品后自动提炼进账号层记忆
  list     看已录入的数据
  review   看复盘
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.panel import Panel

from ...app import Agent
from .quiet import install as quiet_shutdown_noise

console = Console()
app = typer.Typer(help="数据回流：录入 / 列表 / 复盘", no_args_is_help=True)


@app.command("record")
def record_cmd(
    package_id: str = typer.Argument(..., help="待发布包的资产 id"),
    views: int = typer.Option(..., "--views", help="播放/阅读"),
    likes: int = typer.Option(0, "--likes"),
    comments: int = typer.Option(0, "--comments"),
    shares: int = typer.Option(0, "--shares"),
    saves: int = typer.Option(0, "--saves"),
    follows: int = typer.Option(0, "--follows"),
    completion: float = typer.Option(-1.0, "--completion", help="完播率 0-1，不填就不记"),
    url: str = typer.Option("", "--url"),
    note: str = typer.Option("", "--note"),
) -> None:
    """录入一个作品的数据。"""
    asyncio.run(
        _record(package_id, views, likes, comments, shares, saves, follows, completion, url, note)
    )


async def _record(
    package_id: str, views: int, likes: int, comments: int, shares: int, saves: int,
    follows: int, completion: float, url: str, note: str,
) -> None:
    quiet_shutdown_noise()
    agent = Agent.create(auto_approve=True)
    try:
        await agent.setup(mcp=False)
        args = {
            "package_asset_id": package_id, "views": views, "likes": likes, "comments": comments,
            "shares": shares, "saves": saves, "follows": follows, "url": url, "note": note,
        }
        if completion >= 0:
            args["completion_rate"] = completion
        r = await agent.registry.invoke("record_metrics", args)
        if not r.ok:
            console.print(f"[red]{r.error}[/]")
            raise typer.Exit(1)
        console.print(Panel(r.content, title="已录入", border_style="green"))
    finally:
        await agent.aclose()


@app.command("list")
def list_cmd(platform: str = typer.Option("", "--platform")) -> None:
    """看已录入的数据。"""
    asyncio.run(_simple("list_metrics", {"platform": platform}, "数据"))


@app.command("review")
def review_cmd(platform: str = typer.Option("", "--platform")) -> None:
    """复盘：相对账号中位数的倍率、强/弱作品。"""
    asyncio.run(_simple("review_performance", {"platform": platform}, "复盘"))


async def _simple(tool: str, args: dict, title: str) -> None:
    quiet_shutdown_noise()
    agent = Agent.create(auto_approve=True)
    try:
        await agent.setup(mcp=False)
        r = await agent.registry.invoke(tool, args)
        if not r.ok:
            console.print(f"[red]{r.error}[/]")
            raise typer.Exit(1)
        console.print(Panel(r.content, title=title, border_style="cyan"))
    finally:
        await agent.aclose()

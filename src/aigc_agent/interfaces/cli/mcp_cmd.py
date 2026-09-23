"""`agent mcp` —— 看 MCP server 连没连上、工具权限审计、手动调一次。

M20 的运维面。三件事：
  status  哪些 server 连上了、各暴露几个工具、有没有熔断
  audit   哪些外部工具还卡在 L-external —— 确认只读后去 mcp_servers.yaml 降级
  call    手动调一次某个工具，验证连通与返回形状（走同一个注册表和闸门）
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...app import Agent
from ...harness.tools.provider import ToolMeta
from .quiet import install as quiet_shutdown_noise

console = Console()
app = typer.Typer(help="MCP server 状态、权限审计与手动调用", no_args_is_help=True)


async def _ask(meta: ToolMeta, args: dict[str, Any]) -> bool:
    console.print(
        Panel(
            f"[bold]{meta.name}[/]  [dim]{meta.permission.value}[/]\n{meta.summary}\n\n"
            f"[dim]参数：{json.dumps(args, ensure_ascii=False)[:400]}[/]",
            title="[yellow]需要确认[/]",
            border_style="yellow",
        )
    )
    answer = await asyncio.to_thread(typer.prompt, "执行吗？(y/N)", default="n", show_default=False)
    return answer.strip().lower() in {"y", "yes"}


async def _connected() -> Agent:
    agent = Agent.create(asker=_ask)
    await agent.setup()
    return agent


@app.command("status")
def status_cmd() -> None:
    """列出已声明的 server：连接状态、工具数、L-external 数、熔断。"""
    asyncio.run(_status())


async def _status() -> None:
    quiet_shutdown_noise()
    agent = await _connected()
    try:
        if agent.mcp is None or not agent.mcp.providers:
            console.print("[dim]config/mcp_servers.yaml 里没有启用的 server[/]")
            return
        t = Table(show_header=True, header_style="bold")
        for col in ("alias", "传输", "状态", "工具", "L-external", "熔断"):
            t.add_column(col)
        for s in await agent.mcp.status():
            t.add_row(
                str(s["alias"]),
                str(s["transport"]),
                f"[green]✓ {s['detail']}[/]" if s["ok"] else f"[red]✗ {s['detail']}[/]",
                str(s["tools"]),
                str(s["externals"]),
                "[red]开[/]" if s["circuit_open"] else "[dim]关[/]",
            )
        console.print(t)

        console.print("\n[bold]注册进 M4 的外部工具[/]（与内置工具同一注册表、同一套权限）")
        aliases = set(agent.mcp.providers)
        for m in agent.registry.catalog():
            if m.provider in aliases:
                console.print(f"  [dim]{m.permission.value:12}[/] {m.name} — {m.summary}")
    finally:
        await agent.aclose()


@app.command("audit")
def audit_cmd() -> None:
    """列出仍是 L-external 的外部工具，供人核对后在配置里降级。"""
    asyncio.run(_audit())


async def _audit() -> None:
    quiet_shutdown_noise()
    agent = await _connected()
    try:
        if agent.mcp is None or not agent.mcp.providers:
            console.print("[dim]没有启用的 server[/]")
            return
        ext = agent.mcp.audit_permissions()
        if not ext:
            console.print("[green]✓[/] 没有外部工具停留在 L-external")
            return
        console.print(
            f"[yellow]{len(ext)} 个外部工具仍是 L-external[/]（每次调用都要人点头）：\n"
        )
        for name in ext:
            console.print(f"  {name}")
        console.print(
            "\n[dim]确认某个工具只读/可撤销后，在 config/mcp_servers.yaml 对应 server 的 "
            "tool_overrides 里把它降到 L-read / L-write。破坏性的（删除、发布）保持不动。[/]"
        )
    finally:
        await agent.aclose()


@app.command("call")
def call_cmd(
    tool: str = typer.Argument(..., help="完整工具名，如 material_lib__search_materials"),
    args: str = typer.Argument("{}", help="JSON 参数，如 '{\"query\": \"日落\"}'"),
) -> None:
    """手动调一次外部工具。走同一个注册表和权限闸门 —— L-external 会问你。"""
    asyncio.run(_call(tool, args))


async def _call(tool: str, raw: str) -> None:
    quiet_shutdown_noise()
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as e:
        console.print(f"[red]参数不是合法 JSON：{e}[/]")
        raise typer.Exit(1) from e
    agent = await _connected()
    try:
        r = await agent.registry.invoke(tool, parsed)
        if r.ok:
            console.print(Panel(r.content or "（空）", title=f"{tool} · {r.duration_ms}ms",
                                border_style="green"))
        else:
            console.print(f"[red]✗ {r.error}[/]")
            raise typer.Exit(1)
    finally:
        await agent.aclose()

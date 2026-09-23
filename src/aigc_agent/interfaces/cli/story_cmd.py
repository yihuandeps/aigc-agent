"""`agent story` —— 角色护照与分镜拆解。

这些能力（移植自 ai-character-passport）本来只有主循环能用，
终端里够不着。但"给角色建档""把剧本拆成分镜"是要反复跑的日常动作，
每次都开对话去让模型调工具太绕 —— 该有直接入口。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...app import Agent
from .quiet import install as quiet_shutdown_noise

console = Console()
app = typer.Typer(help="角色护照与分镜", no_args_is_help=True)


@app.command("who")
def who_cmd() -> None:
    """看有哪些角色护照。"""
    asyncio.run(_who())


async def _who() -> None:
    quiet_shutdown_noise()
    agent = Agent.create()
    try:
        await agent.setup(mcp=False)
        r = await agent.registry.invoke("list_characters", {})
        console.print(r.content if r.ok else f"[red]{r.error}[/]")
    finally:
        await agent.aclose()


@app.command("new")
def new_cmd(
    name: str = typer.Argument(..., help="角色名，如 林夏"),
    who: str = typer.Option("", "--who", "-w", help="人物是谁，一句话"),
    look: str = typer.Option("", "--look", "-l", help="外观标签，逗号分隔"),
    style: str = typer.Option("", "--style", "-s", help="风格与光影"),
) -> None:
    """建一张角色护照。

    外观标签是跨镜头一致性的主力，写得越具体越稳：
    `--look "短黑发, 圆框眼镜, 灰色连帽衫, 左耳银色耳钉"`
    """
    asyncio.run(_new(name, who, look, style))


async def _new(name: str, who: str, look: str, style: str) -> None:
    quiet_shutdown_noise()
    agent = Agent.create()
    try:
        await agent.setup(mcp=False)
        r = await agent.registry.invoke(
            "save_character",
            {"name": name, "base_prompt": who, "appearance": look, "style_lighting": style},
        )
        if r.ok:
            console.print(Panel(r.content, border_style="green"))
        else:
            console.print(f"[red]✗ {r.error}[/]")
            raise typer.Exit(1)
    finally:
        await agent.aclose()


@app.command("shots")
def shots_cmd(
    script: str = typer.Argument("", help="剧本原文。留空则从标准输入读"),
    character: str = typer.Option("", "--as", "-a", help="用哪个角色（逐镜注入外貌）"),
    seconds: int = typer.Option(8, "--seconds", help="单镜时长秒，默认 8（veo3.1 上限）"),
    count: int = typer.Option(0, "--count", "-n", help="要几镜。省略由剧本长度决定"),
    from_file: str = typer.Option("", "--file", "-f", help="从文件读剧本"),
) -> None:
    """把剧本拆成分镜（英文画面提示词 + 中文旁白）。

    长剧本用管道或文件：
      type script.txt | agent story shots --as 林夏
      agent story shots -f script.txt --as 林夏
    """
    text = script
    if from_file:
        text = Path(from_file).read_text(encoding="utf-8")
    elif not text and not sys.stdin.isatty():
        text = sys.stdin.read()
    if not text.strip():
        console.print("[red]没有剧本内容。用参数、--file 或管道传进来。[/]")
        raise typer.Exit(1)
    asyncio.run(_shots(text, character, seconds, count))


async def _shots(script: str, character: str, seconds: int, count: int) -> None:
    quiet_shutdown_noise()
    agent = Agent.create()
    try:
        await agent.setup(mcp=False)
        console.print(f"[dim]拆解中（{len(script)} 字）…[/]")
        r = await agent.registry.invoke(
            "breakdown_script",
            {
                "script": script,
                "character": character,
                "seconds_each": seconds,
                "shot_count": count,
            },
        )
        if not r.ok:
            console.print(f"[red]✗ {r.error}[/]")
            raise typer.Exit(1)
        console.print(r.content)
    finally:
        await agent.aclose()


@app.command("voices")
def voices_cmd() -> None:
    """看当前 TTS provider 有哪些音色可用。"""
    asyncio.run(_voices())


async def _voices() -> None:
    quiet_shutdown_noise()
    agent = Agent.create()
    try:
        await agent.setup(mcp=False)
        cat = agent.catalog
        if cat is None or not cat.speech_voices:
            console.print("[dim]目录里没有音色[/]")
            return
        t = Table(show_header=True, header_style="bold")
        t.add_column("音色 id")
        t.add_column("适合什么内容")
        for v in cat.speech_voices:
            t.add_row(v.name, v.note)
        console.print(t)
        console.print(
            f"[dim]provider {cat.speech_provider} · 默认 {cat.speech_default_voice}\n"
            "配方里 voice: auto 会让模型按文案语气自动挑[/]"
        )
    finally:
        await agent.aclose()

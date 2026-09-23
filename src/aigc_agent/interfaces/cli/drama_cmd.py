"""`agent drama` —— 短剧三段式流水线。

    剧本 ──① script──→ 分镜脚本
      └───② assets──→ 资产库 ──④ render──→ 参考图
                          └──────┬───────────┘
                            ③ shots → ⑤ film → 成片

音频**直接来自视频生成模型**（seedance 原生出声：台词 + 环境音），
不做单独 TTS 配音 —— 那会盖掉原声，口型也对不上。

每步的产物都是资产 id，下一步按 id 取。`run` 把整条串起来，
但**默认只跑到第③步**（拆解）—— 后面两步要花钱花时间，
方向没确认就批量生成是最贵的错误。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console

from ...app import Agent
from ...domain.drama import ETHNICITIES, LANGUAGES, normalize
from .quiet import install as quiet_shutdown_noise

console = Console()
app = typer.Typer(help="短剧：剧本 → 分镜 → 资产 → 成片", no_args_is_help=True)


def _read(text: str, file: str) -> str:
    if file:
        return Path(file).read_text(encoding="utf-8")
    if text:
        return text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def _show(label: str, r) -> bool:
    secs = r.duration_ms / 1000
    if not r.ok:
        console.print(f"[red]✗ {label}[/] [dim]{secs:.0f}s[/]")
        for line in (r.error or "").splitlines():
            console.print(f"  [red]{line}[/]")
        return False
    console.print(f"[green]✓ {label}[/] [dim]{secs:.0f}s[/]")
    for line in (r.content or "").splitlines():
        style = "yellow" if line.strip().startswith("⚠") else "dim"
        console.print(f"  [{style}]{line}[/]")
    return True


# ---------------------------------------------------------------- 单步


@app.command("script")
def script_cmd(
    text: str = typer.Argument("", help="剧本原文，留空则从 --file 或管道读"),
    file: str = typer.Option("", "--file", "-f", help="剧本文件"),
    face: str = typer.Option(
        "", "--face", help="面孔：asian/chinese/caucasian/african/latino/mixed"
    ),
    lang: str = typer.Option("", "--lang", help="台词语言：zh/en/keep"),
) -> None:
    """① 拆分镜脚本。台词逐字保留。"""
    script = _need(_read(text, file))
    e, ln = _choose(face, lang)
    asyncio.run(
        _one(
            "drama_storyboard",
            {"script": script, "ethnicity": e, "language": ln},
            "① 分镜脚本",
        )
    )


@app.command("assets")
def assets_cmd(
    text: str = typer.Argument("", help="剧本原文"),
    file: str = typer.Option("", "--file", "-f", help="剧本文件"),
    face: str = typer.Option(
        "", "--face", help="面孔：asian/chinese/caucasian/african/latino/mixed"
    ),
    lang: str = typer.Option("", "--lang", help="台词语言：zh/en/keep"),
) -> None:
    """② 拆资产库：角色主形象 / 服装 / 场景 / 道具。"""
    script = _need(_read(text, file))
    e, ln = _choose(face, lang)
    asyncio.run(
        _one("drama_assets", {"script": script, "ethnicity": e, "language": ln}, "② 资产库")
    )


@app.command("shots")
def shots_cmd(
    storyboard_id: str = typer.Argument(..., help="① 的资产 id"),
    assets_id: str = typer.Argument(..., help="② 的资产 id"),
    episode: int = typer.Option(0, "--episode", "-e", help="只处理第几集"),
) -> None:
    """③ 合成 seedance 视频提示词（把镜头和资产 ID 绑死）。"""
    asyncio.run(
        _one(
            "drama_shots",
            {"storyboard_id": storyboard_id, "assets_id": assets_id, "episode": episode},
            "③ 视频提示词",
        )
    )


@app.command("render")
def render_cmd(
    assets_id: str = typer.Argument(..., help="② 的资产 id"),
    only: str = typer.Option("all", "--only", help="characters/costumes/scenes/props"),
) -> None:
    """④ 资产生图。角色主形象先出，服装拿它当参考图。"""
    asyncio.run(_one("drama_render_assets", {"assets_id": assets_id, "only": only}, "④ 资产生图"))


@app.command("film")
def film_cmd(
    shots_id: str = typer.Argument(..., help="③ 的资产 id"),
    rendered_id: str = typer.Option("", "--refs", "-r", help="④ 的资产 id，用来取参考图"),
    limit: int = typer.Option(0, "--limit", "-n", help="只跑前几段，0=全部"),
    out: str = typer.Option("", "--out", "-o", help="成片导出目录"),
    name: str = typer.Option("", "--name", help="成片文件名"),
    no_compose: bool = typer.Option(False, "--no-compose", help="只出片段不拼整集"),
) -> None:
    """⑤ 分镜生视频并拼成整集。很慢很贵，建议先 -n 1 看方向。

    片段自带音轨（seedance 原生出声），拼接会保留 —— 不需要另做配音。
    """
    asyncio.run(
        _one(
            "drama_render_shots",
            {
                "shots_id": shots_id,
                "rendered_id": rendered_id,
                "limit": limit,
                "compose": not no_compose,
                "out_dir": out,
                "filename": name,
            },
            "⑤ 分镜生视频",
        )
    )


# ---------------------------------------------------------------- 串起来


@app.command("run")
def run_cmd(
    text: str = typer.Argument("", help="剧本原文"),
    file: str = typer.Option("", "--file", "-f", help="剧本文件"),
    render: bool = typer.Option(False, "--render", help="继续跑资产生图"),
    film: int = typer.Option(0, "--film", help="继续跑生视频，值=跑几段（要先 --render）"),
    face: str = typer.Option(
        "", "--face", help="面孔：asian/chinese/caucasian/african/latino/mixed"
    ),
    lang: str = typer.Option("", "--lang", help="台词语言：zh/en/keep"),
) -> None:
    """整条跑。**默认只到第③步**（拆解，不花生成的钱）。

    确认方向后再加 --render 出参考图，再加 --film 2 试两段视频。
    """
    script = _need(_read(text, file))
    e, ln = _choose(face, lang)
    asyncio.run(_run(script, render, film, e, ln))


def _choose(opt_e: str, opt_l: str) -> tuple[str, str]:
    """没在命令行给就当场问。

    不设默认值 —— 族裔和语言会贯穿角色形象和全部分镜视频，
    选错等于整条链重跑。让人明确选一次，比事后返工便宜得多。
    """
    o = normalize(opt_e, opt_l)
    if o.ready:
        return o.ethnicity, o.language

    if not o.ethnicity:
        console.print()
        console.print("[bold]画面里的面孔？[/]")
        keys = list(ETHNICITIES)
        for i, k in enumerate(keys, 1):
            console.print(f"  [cyan]{i}[/] {k:<10} {ETHNICITIES[k][1]}")
        pick = console.input("选一个（序号或名字）：").strip()
        opt_e = keys[int(pick) - 1] if pick.isdigit() and 0 < int(pick) <= len(keys) else pick

    if not o.language:
        console.print()
        console.print("[bold]台词语言？[/]")
        keys = list(LANGUAGES)
        for i, k in enumerate(keys, 1):
            console.print(f"  [cyan]{i}[/] {k:<6} {LANGUAGES[k][1]}")
        pick = console.input("选一个（序号或名字）：").strip()
        opt_l = keys[int(pick) - 1] if pick.isdigit() and 0 < int(pick) <= len(keys) else pick

    o = normalize(opt_e, opt_l)
    if not o.ready:
        console.print("[red]没认出你选的项，再跑一次并用 --face / --lang 明确指定。[/]")
        raise typer.Exit(1)
    console.print(f"[dim]已选：{o.brief()}[/]")
    console.print()
    return o.ethnicity, o.language


def _need(script: str) -> str:
    if not script.strip():
        console.print("[red]没有剧本内容。用参数、--file 或管道传进来。[/]")
        raise typer.Exit(1)
    return script


async def _one(tool: str, args: dict, label: str) -> None:
    quiet_shutdown_noise()
    agent = Agent.create()
    try:
        await agent.setup(mcp=False)
        r = await agent.registry.invoke(tool, args)
        if not _show(label, r):
            raise typer.Exit(1)
    finally:
        await agent.aclose()


async def _run(script: str, render: bool, film: int, ethnicity: str, language: str) -> None:
    quiet_shutdown_noise()
    agent = Agent.create()
    try:
        await agent.setup(mcp=False)

        opts = {"ethnicity": ethnicity, "language": language}
        r1 = await agent.registry.invoke("drama_storyboard", {"script": script, **opts})
        if not _show("① 分镜脚本", r1):
            raise typer.Exit(1)

        r2 = await agent.registry.invoke("drama_assets", {"script": script, **opts})
        if not _show("② 资产库", r2):
            raise typer.Exit(1)

        r3 = await agent.registry.invoke(
            "drama_shots", {"storyboard_id": r1.asset_ref, "assets_id": r2.asset_ref}
        )
        if not _show("③ 视频提示词", r3):
            raise typer.Exit(1)

        if not render:
            console.print(
                f"\n[dim]拆解完成。确认方向后继续：\n"
                f"  agent drama render {r2.asset_ref}\n"
                f"  agent drama film {r3.asset_ref} --refs <上一步的 id> -n 1[/]"
            )
            return

        r4 = await agent.registry.invoke("drama_render_assets", {"assets_id": r2.asset_ref})
        if not _show("④ 资产生图", r4):
            raise typer.Exit(1)

        if not film:
            console.print(
                f"\n[dim]参考图就绪。试几段视频：\n"
                f"  agent drama film {r3.asset_ref} --refs {r4.asset_ref} -n 1[/]"
            )
            return

        r5 = await agent.registry.invoke(
            "drama_render_shots",
            {"shots_id": r3.asset_ref, "rendered_id": r4.asset_ref, "limit": film},
        )
        _show("⑤ 分镜生视频", r5)
    finally:
        await agent.aclose()

"""`agent video` —— 一条命令出成片。

流程固定（写文案 → 分镜 → 生成 → 配音 → 字幕 → 合成），细节全在配方里，
改 yaml 就能迭代。见 config/recipes/。
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...app import Agent
from ...domain.media.naming import recipe_shot_name
from ...domain.pipeline.recipe import Recipe, list_recipes, load_recipe, parse_pick, parse_plan
from ...domain.pipeline.subtitle import align_script
from ...domain.pipeline.voice_pick import parse_pick as parse_voice
from ...domain.pipeline.voice_pick import pick_prompt as voice_prompt
from .quiet import install as quiet_shutdown_noise

console = Console()
app = typer.Typer(help="短视频生产", no_args_is_help=True)


@app.command("list")
def list_cmd() -> None:
    """看有哪些配方。"""
    rs = list_recipes()
    if not rs:
        console.print("[dim]还没有配方，去 config/recipes/ 建一个[/]")
        return
    t = Table(show_header=True, header_style="bold")
    t.add_column("配方")
    t.add_column("说明")
    t.add_column("素材")
    t.add_column("成片")
    t.add_column("单镜")
    t.add_column("档位")
    for r in rs:
        t.add_row(
            r.path.stem if r.path else r.name,
            r.description,
            f"{r.shot_count}×{r.seconds_each}s",
            f"{r.total_seconds}s",
            f"≤{r.cut_max:g}s" if r.cut_max else "整段",
            str(r.models.get("video_tier", "fast")),
        )
    console.print(t)
    console.print("[dim]改效果直接编辑 config/recipes/*.yaml，不用改代码[/]")


@app.command()
def make(
    topic: str = typer.Argument(..., help="视频主题"),
    recipe: str = typer.Option("tech-short", "--recipe", "-r", help="配方名"),
    out: str = typer.Option("", "--out", "-o", help="导出目录"),
    duration: int = typer.Option(0, "--duration", "-d", help="总时长秒，覆盖配方"),
    tier: str = typer.Option("", "--tier", help="fast/balanced/quality，覆盖配方"),
    voice: str = typer.Option("", "--voice", help="音色，覆盖配方"),
    no_voiceover: bool = typer.Option(False, "--no-voiceover", help="不配音"),
    no_subtitle: bool = typer.Option(False, "--no-subtitle", help="不加字幕"),
    yes: bool = typer.Option(False, "--yes", "-y", help="跳过人审，全自动"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只出文案和分镜，不生成视频"),
    no_ground: bool = typer.Option(False, "--no-ground", help="不拉真实热榜，直接用你给的主题"),
    max_cut: float = typer.Option(
        0.0, "--max-cut", help="单个镜头最长几秒（快切）。覆盖配方，如 --max-cut 2"
    ),
) -> None:
    """生成一条短视频。默认会先拉真实热榜来定选题。"""
    asyncio.run(
        _make(
            topic, recipe, out, duration, tier, voice,
            no_voiceover, no_subtitle, yes, dry_run, no_ground, max_cut,
        )
    )


async def _make(
    topic: str,
    recipe_name: str,
    out: str,
    duration: int,
    tier: str,
    voice: str,
    no_vo: bool,
    no_sub: bool,
    yes: bool,
    dry: bool,
    no_ground: bool = False,
    max_cut: float = 0.0,
) -> None:
    quiet_shutdown_noise()
    try:
        r = load_recipe(recipe_name)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e

    r = r.override(
        duration=duration,
        tier=tier,
        voice=voice,
        no_voiceover=no_vo,
        no_subtitle=no_sub,
        max_cut=max_cut,
    )
    vtier = str(r.models.get("video_tier", "fast"))

    pace_note = f" · 快切 ≤{r.cut_max:g}s" if r.cut_max else ""
    console.print(
        Panel(
            f"""[bold]{topic}[/]
配方 {r.name} · 素材 {r.shot_count}×{r.seconds_each}s · 成片 {r.total_seconds}s{pace_note}
{r.output.get("aspect_ratio", "9:16")} · 档位 {vtier}
口播 {"开" if r.voiceover.get("enabled") and not no_vo else "关"} · """
            f"""字幕 {"开" if r.subtitle.get("enabled") and not no_sub else "关"}""",
            title="短视频生产",
            border_style="cyan",
        )
    )


    agent = Agent.create()
    await agent.setup(mcp=False)
    # 每条退出路径都要关连接：dry-run 早返回、用户放弃、Exit(1) 都会漏，
    # 漏了进程退出时 httpx 连接池会刷一屏 athrow 堆栈，把真正的输出淹掉。
    try:
        t0 = time.perf_counter()

        # ---- 0. 接真实热点 ----
        # 没有这步，所谓「热点」就是模型按常识编的 —— 上一版成片效果差的根因。
        if r.grounded and not no_ground:
            console.print()
            console.print("[bold]0/5[/] 拉真实热榜定选题…")
            topic, facts = await _ground(r, agent, topic)
        else:
            facts = ""

        # ---- 1. 文案 + 分镜 ----
        console.print("\n[bold]1/5[/] 写文案与分镜…")
        resp = await agent.gateway.chat(
            "main_agent", [{"role": "user", "content": r.script_prompt(topic, facts)}]
        )
        script, shots, warn = parse_plan(resp.text, r.shot_count)
        if not script or not shots:
            console.print(f"[red]✗ {warn or '模型没给出可用的文案/分镜'}[/]")
            console.print(f"[dim]{resp.text[:400]}[/]")
            raise typer.Exit(1)
        if warn:
            console.print(f"[yellow]⚠ {warn}[/]")

        # 字数超了成片就会变长（画面必须盖住旁白，否则旁白被截断）。
        # 放在生成视频**之前**报，这时候还没花钱，人审那步可以直接否掉重来。
        if len(script) > r.script_chars * 1.15:
            est = len(script) / 4.5
            console.print(
                f"[yellow]⚠ 文案 {len(script)} 字，超出 {r.script_chars} 字的目标；"
                f"配音约 {est:.0f}s，成片会从 {r.total_seconds}s 拉长到这个长度[/]"
            )

        console.print(Panel(script, title=f"口播（{len(script)} 字）", border_style="blue"))
        for i, s in enumerate(shots, 1):
            console.print(f"  [cyan]第{i}镜[/] {s}")

        # ---- 人审断点 ----
        if r.review.get("after_script") and not yes and not dry:
            console.print()
            ans = await asyncio.to_thread(
                console.input, "[bold]继续生成视频？[/] (y=继续 / n=放弃 / 回车=继续) "
            )
            if ans.strip().lower() in ("n", "no"):
                console.print("[dim]已放弃。调整 config/recipes/ 里的 prompt_hint 再试。[/]")
                return

        if dry:
            console.print("\n[dim]--dry-run：到此为止，未生成视频[/]")
            return

        # ---- 2. 生成分镜 ----
        console.print(f"\n[bold]2/5[/] 生成 {len(shots)} 段视频（这步最慢，每段 1-4 分钟）…")
        clip_ids: list[str] = []
        for i, shot in enumerate(shots, 1):
            res = await agent.registry.invoke(
                "gen_video",
                {
                    "prompt": r.shot_prompt(shot),
                    "prefer": vtier,
                    "aspect_ratio": r.output.get("aspect_ratio", "9:16"),
                    "duration": r.seconds_each,
                    "resolution": r.output.get("resolution", "720p"),
                    "summary": f"{topic}·第{i}镜",
                    "local_name": recipe_shot_name(topic, i),
                },
            )
            if res.ok:
                clip_ids.append(res.asset_ref)
                console.print(f"  [green]✓[/] 第{i}镜 [dim]{res.duration_ms / 1000:.0f}s[/]")
            else:
                console.print(f"  [red]✗[/] 第{i}镜 {res.error[:120]}")

        if not clip_ids:
            console.print("[red]全部分镜生成失败，无法合成[/]")
            raise typer.Exit(1)
        if len(clip_ids) < len(shots):
            console.print(
                f"[yellow]⚠ {len(shots) - len(clip_ids)} 段失败，用剩下 {len(clip_ids)} 段继续[/]"
            )

        # ---- 3. 配音 ----
        audio_id = ""
        if r.voiceover.get("enabled"):
            console.print("\n[bold]3/5[/] 配音…")
            voice, speed = await _pick_voice(r, agent, script, topic)
            res = await agent.registry.invoke(
                "tts",
                {
                    "text": script,
                    "voice": voice,
                    "speed": speed,
                    "instruct": r.voiceover.get("instruct", ""),
                    "summary": f"{topic}·口播",
                },
            )
            if res.ok:
                audio_id = res.asset_ref
                console.print("  [green]✓[/] 已合成")
            else:
                console.print(f"  [yellow]⚠ 配音失败，成片将无声：{res.error[:100]}[/]")
        else:
            console.print("\n[dim]3/5 跳过配音[/]")

        # ---- 4. 字幕 ----
        sub_id = ""
        if r.subtitle.get("enabled") and audio_id:
            console.print("\n[bold]4/5[/] 字幕（从配音转写，时间轴最准）…")
            res = await agent.registry.invoke(
                "transcribe",
                {
                    "asset_id": audio_id,
                    "format": "srt",
                    "language": r.subtitle.get("language", "zh"),
                },
            )
            if res.ok:
                sub_id = res.asset_ref
                # ASR 只用来拿时间轴，文字回贴原稿。
                # 它再准也是二次识别，同音字必错在专业名词上 ——
                # 实测一条讲氢能的片子，"氢"全程被听成"芯"，还烧进了画面。
                fixed, changed = align_script(script, agent.assets.content(sub_id))
                if changed:
                    rev = agent.assets.revise(
                        sub_id, fixed, summary="字幕·按原稿校正", creator="pipeline:align"
                    )
                    sub_id = rev.id
                    console.print(f"  [green]✓[/] 已生成 [dim]（按原稿校正 {changed} 条）[/]")
                else:
                    console.print("  [green]✓[/] 已生成")
            else:
                console.print(f"  [yellow]⚠ 字幕失败，成片将无字幕：{res.error[:100]}[/]")
        else:
            console.print("\n[dim]4/5 跳过字幕[/]")

        # ---- 5. 合成 ----
        console.print("\n[bold]5/5[/] 下载素材并合成…")
        res = await agent.registry.invoke(
            "compose_video",
            {
                "clips": clip_ids,
                "audio_id": audio_id,
                "subtitle_id": sub_id,
                "out_dir": out,
                "filename": r.filename(topic, vtier),
                "max_cut_seconds": r.cut_max,
                "min_cut_seconds": r.cut_min,
                "total_seconds": r.total_seconds,
            },
        )
        dt = time.perf_counter() - t0

        if not res.ok:
            console.print(f"[red]✗ 合成失败：{res.error}[/]")
            raise typer.Exit(1)

        console.print()
        console.print(Panel(res.content, title=f"完成（{dt / 60:.1f} 分钟）", border_style="green"))
        console.print(
            "[dim]效果要调？改 config/recipes/"
            f"{Path(recipe_name).stem}.yaml 的 style / prompt_hint / instruct，再跑一次[/]"
        )
    finally:
        await agent.aclose()



async def _ground(r: Recipe, agent: Agent, topic: str) -> tuple[str, str]:
    """拿真实热榜让模型挑选题，返回 (选题, 热榜原文)。

    拉不到就退回用户给的主题，**并明说降级了** —— 热榜是加分项，不该因为
    它挂了就出不了片；但悄悄退回去会让人以为内容是有依据的，那更糟。
    """
    g = r.grounding
    hot = await agent.registry.invoke(
        "douyin_hot_list",
        {
            "count": int(g.get("top", 15)),
            "days": int(g.get("days", 1)),
            "category": str(g.get("category") or ""),
        },
    )
    if not hot.ok:
        console.print("  [yellow]⚠ 热榜没拉到，退回你给的主题；内容将没有真实数据支撑[/]")
        console.print(f"  [dim]{hot.error[:160]}[/]")
        return topic, ""

    resp = await agent.gateway.chat(
        "main_agent", [{"role": "user", "content": r.pick_prompt(topic, hot.content)}]
    )
    picked, why = parse_pick(resp.text)
    if not picked:
        console.print("  [yellow]⚠ 没解析出选题，退回你给的主题[/]")
        return topic, hot.content

    console.print(f"  [green]✓[/] 选题：[bold]{picked}[/]")
    if why:
        console.print(f"  [dim]{why}[/]")
    return picked, hot.content


async def _pick_voice(r: Recipe, agent: Agent, script: str, topic: str) -> tuple[str, float]:
    """配方里写 auto 就让模型按文案语气挑，否则用写死的那个。

    同一个音色配所有内容，是"听着假"的一个隐性来源：用新闻女声念一条生活
    种草，语气和内容对不上，哪怕音色本身很自然，听感也别扭。
    """
    want = str(r.voiceover.get("voice") or "")
    speed = float(r.voiceover.get("speed") or 1.0)
    if want.lower() != "auto":
        return want, speed

    voices = [(v.name, v.note) for v in agent.catalog.speech_voices]
    fallback = agent.catalog.speech_default_voice
    if not voices:
        return fallback, speed

    resp = await agent.gateway.chat(
        "voice_select", [{"role": "user", "content": voice_prompt(script, voices, topic)}]
    )
    pick = parse_voice(resp.text, [n for n, _ in voices], fallback, speed)
    note = next((n for v, n in voices if v == pick.voice), "")
    console.print(f"  [green]✓[/] 音色：[bold]{note or pick.voice}[/] · 语速 {pick.speed:g}")
    if pick.why:
        console.print(f"  [dim]{pick.why}[/]")
    return pick.voice, pick.speed

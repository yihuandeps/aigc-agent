"""`agent video` —— 一条命令出成片。

走的是和对话里同一组工具：（拉热榜）→ short_video_brief 写简报 → short_video_produce 出片。
细节全在配方里（config/recipes/），改 yaml 就能迭代。

2026-09-23 审查：这里之前自己把文案 / 分镜 / 生成 / 配音 / 字幕 / 合成又写了一遍，对话那条链
后来补上的东西它全没有 —— 出片前报成本、口播出镜、产品图身份锁、抽帧查字、按简报顺序排刀、
失败不重复付费。现在命令行只管交互，活都交给同一组工具，两条链不会再各改各的。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ...app import Agent
from ...domain.pipeline.recipe import Recipe, list_recipes, load_recipe
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
    t.add_column("时长")
    t.add_column("单镜")
    t.add_column("档位")
    t.add_column("热点")
    for r in rs:
        lo, hi = r.duration_bounds
        t.add_row(
            r.key,
            r.description,
            f"{lo}–{hi}s",
            f"≤{r.cut_max:g}s",
            str(r.models.get("video_tier", "fast")),
            "接热榜" if r.grounded else "—",
        )
    console.print(t)
    console.print("[dim]改效果直接编辑 config/recipes/*.yaml，不用改代码[/]")


@app.command()
def make(
    topic: str = typer.Argument(..., help="视频主题"),
    recipe: str = typer.Option("tech-short", "--recipe", "-r", help="配方名"),
    out: str = typer.Option("", "--out", "-o", help="导出目录"),
    duration: int = typer.Option(0, "--duration", "-d", help="成片大概多少秒（在配方范围内）"),
    tier: str = typer.Option("", "--tier", help="fast/balanced/quality，覆盖配方"),
    voice: str = typer.Option("", "--voice", help="音色，覆盖配方"),
    no_voiceover: bool = typer.Option(False, "--no-voiceover", help="不配音"),
    no_subtitle: bool = typer.Option(False, "--no-subtitle", help="不加字幕"),
    yes: bool = typer.Option(False, "--yes", "-y", help="跳过出片前确认，全自动"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只出简报（口播和分镜），不生成视频"),
    no_ground: bool = typer.Option(False, "--no-ground", help="不拉真实热榜，直接用你给的主题"),
    max_cut: float = typer.Option(
        0.0, "--max-cut", help="（已不单独覆盖）快切上限按配方 cut.max_seconds，全局 ≤3 秒"
    ),
    ratio: str = typer.Option(
        "", "--ratio", help="视频画幅：16:9 横屏 / 9:16 竖屏 / 1:1 方形；不填按项目设置或配方"
    ),
) -> None:
    """生成一条短视频。配方开了接热点就先拉真实热榜。"""
    asyncio.run(
        _make(
            topic, recipe, out, duration, tier, voice,
            no_voiceover, no_subtitle, yes, dry_run, no_ground, max_cut, ratio,
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
    ratio: str = "",
) -> None:
    quiet_shutdown_noise()
    try:
        r = load_recipe(recipe_name)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e

    lo, hi = r.duration_bounds
    vtier = tier or str(r.models.get("video_tier", "fast"))
    vo = "开" if r.voiceover.get("enabled") and not no_vo else "关"
    if vo == "关" and not no_vo and str(r.style.get("footage") or "") == "real":
        # 实拍为主的配方没给出镜素材时，出片那步用 TTS 口播兜底（不然成片没声音）
        vo = "开（没给出镜素材，TTS 口播兜底）"
    sub = "开" if r.subtitle.get("enabled") and not no_sub else "关"
    console.print(
        Panel(
            Text(
                f"{topic}\n配方 {r.name} · 成片 {lo}–{hi}s · 快切 ≤{r.cut_max:g}s · "
                f"{r.output.get('aspect_ratio', '9:16')} · 档位 {vtier}\n口播 {vo} · 字幕 {sub}"
            ),
            title="短视频生产",
            border_style="cyan",
        )
    )
    if max_cut:
        console.print(
            f"[yellow]--max-cut {max_cut:g} 不再单独生效：快切上限按配方 cut.max_seconds"
            f"（这次 ≤{r.cut_max:g}s），全局不超过 3 秒[/]"
        )

    agent = Agent.create(asker=_ask)
    await agent.setup(mcp=False)
    # 每条退出路径都要关连接：dry-run 早返回、用户放弃、Exit(1) 都会漏，
    # 漏了进程退出时 httpx 连接池会刷一屏 athrow 堆栈，把真正的输出淹掉。
    try:
        t0 = time.perf_counter()

        # ---- 0. 接真实热点 ----
        # 没有这步，所谓「热点」就是模型按常识编的 —— 上一版成片效果差的根因。
        sources: list[str] = []
        if r.grounded and not no_ground:
            console.print("\n[bold]0/3[/] 拉真实热榜…")
            hot = await _ground(r, agent)
            if hot:
                sources.append(hot)

        # ---- 1. 简报：选题角度 + 口播 + 分镜（和对话里是同一个工具）----
        console.print("\n[bold]1/3[/] 写简报（选题角度、口播、分镜）…")
        notes = f"成片按 {duration} 秒左右出" if duration else ""
        b = await agent.registry.invoke(
            "short_video_brief",
            # 热榜上面已经拉过（或 --no-ground 不要）：简报工具别再自己拉一次
            {"keyword": topic, "style": recipe_name, "sources": sources, "notes": notes,
             "grounding": False},
        )
        if not b.ok:
            console.print(f"[red]✗ 简报没写成：{escape(str(b.error))}[/]")
            raise typer.Exit(1)
        # 配方要求文案先给人看时工具会挂起；终端里人就在看这块面板，下一步的出片确认就是拍板
        console.print(
            Panel(Text(str(b.content).split("\n\n已暂停")[0]), title="简报", border_style="blue")
        )
        if dry:
            console.print("\n[dim]--dry-run：到此为止，未生成视频[/]")
            return

        # ---- 2. 出片前确认：要生成几段、多少秒、用哪个模型 ----
        args = {
            "brief_id": b.asset_ref,
            "tier": tier,
            "voice": voice,
            "no_voiceover": no_vo,
            "no_subtitle": no_sub,
            "out_dir": out,
        }
        if ratio:
            args["aspect_ratio"] = ratio
        console.print("\n[bold]2/3[/] 核对要生成多少…")
        res = await agent.registry.invoke("short_video_produce", args)
        if res.suspend:
            q = str((res.suspend_payload or {}).get("question") or res.content)
            console.print(Panel(Text(q), title="出片前确认", border_style="yellow"))
            if not yes:
                ans = await asyncio.to_thread(
                    console.input, "[bold]开始生成？[/] (y=开始 / n=放弃 / 回车=开始) "
                )
                if ans.strip().lower() in ("n", "no"):
                    console.print(
                        "[dim]已放弃。调整 config/recipes/ 里的 prompt_hint 再试；"
                        f"简报留着（{b.asset_ref}），对话里也能接着用。[/]"
                    )
                    return
            # 人在终端点了头（或开工时就给了 --yes）：记下这次确认，confirm=true 才生效
            fns = getattr(agent, "short_video_fns", None)
            if fns is not None:
                fns.approve(str(b.asset_ref))
            # ---- 3. 生成 → 配音 → 字幕 → 合成 ----
            console.print("\n[bold]3/3[/] 生成并合成（这步最慢，每段 1–4 分钟）…")
            res = await agent.registry.invoke("short_video_produce", {**args, "confirm": True})
        dt = time.perf_counter() - t0

        if not res.ok:
            console.print(f"[red]✗ 出片失败：{escape(str(res.error))}[/]")
            raise typer.Exit(1)
        console.print()
        if (res.meta or {}).get("complete") is False:
            # 有镜头没生成出来：不合成（缺镜头的成片会让人以为做完了）
            console.print(Panel(Text(res.content), title="未成片", border_style="yellow"))
            console.print("[dim]再跑一次同样的命令：已生成的镜头会复用，只补缺的。[/]")
            raise typer.Exit(1)
        title = f"完成（{dt / 60:.1f} 分钟）"
        body = str(res.content)
        if res.suspend:
            # 配方要求成片审核（产品广告 / 口播出镜）：终端里就是人在看
            title = f"成片待你审（{dt / 60:.1f} 分钟）"
            body = body.split("\n\n已暂停")[0]
        console.print(Panel(Text(body), title=title, border_style="green"))
        console.print(
            f"[dim]效果要调？改 config/recipes/{r.key}.yaml 的 style / prompt_hint / instruct，"
            "再跑一次[/]"
        )
    finally:
        await agent.aclose()


async def _ask(meta: Any, args: dict[str, Any]) -> bool:
    """闸门要人点头时（超出开工额度、L-external 工具）在终端问一句。

    之前没接询问器：超出额度的镜头直接被拒，出片照样合成、成片缺镜（2026-09-24 审查）。
    --yes 只跳过出片前的成本确认，超额度照样要问。读不到输入（管道 / 非交互）按不同意。"""
    console.print(
        Panel(Text(f"{meta.name}\n{meta.summary}"), title="需要确认", border_style="yellow")
    )
    try:
        ans = await asyncio.to_thread(console.input, "执行吗？(y/N) ")
    except (EOFError, KeyboardInterrupt):
        return False
    return ans.strip().lower() in ("y", "yes")


async def _ground(r: Recipe, agent: Agent) -> str:
    """拉真实热榜，返回热榜资产 id（简报按它写，口播里的数字才有出处）。

    拉不到就返回空 —— 热榜是加分项，不该因为它挂了就出不了片；但要**明说降级了**，
    悄悄退回去会让人以为内容是有依据的，那更糟。
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
    if not hot.ok or not str(hot.content or "").strip():
        console.print("  [yellow]⚠ 热榜没拉到，按你给的主题写；内容将没有真实数据支撑[/]")
        if hot.error:
            console.print(f"  [dim]{hot.error[:160]}[/]")
        return ""
    if getattr(hot, "asset_ref", ""):
        console.print("  [green]✓[/] 已拉到热榜")
        return str(hot.asset_ref)
    a = agent.assets.create(hot.content, summary="抖音热榜", creator="tool:douyin_hot_list")
    console.print("  [green]✓[/] 已拉到热榜")
    return a.id

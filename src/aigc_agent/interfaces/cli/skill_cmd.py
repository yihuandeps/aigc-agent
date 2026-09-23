"""`agent skill` —— 运营维护方法论的运维面。

运营直接改 skills/*.md，存盘即生效（热加载）。配套两件事让"放开写"不出事：
版本记录（谁在什么时候改了什么）和一键回滚（改坏了退回上一版）。
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...capabilities.skill_hub import SkillHub, parse_skill, skill_files
from ...envdetect import PROJECT_ROOT

console = Console()
app = typer.Typer(help="方法论 skill：列表 / 查看 / 版本 / 回滚 / 体检", no_args_is_help=True)

SKILLS_DIR = PROJECT_ROOT / "skills"
HISTORY_DIR = PROJECT_ROOT / "workspace" / "skills_history"


def _hub() -> SkillHub:
    hub = SkillHub(SKILLS_DIR, history_dir=HISTORY_DIR)
    hub.load()
    return hub


@app.command("list")
def list_cmd(
    all_: bool = typer.Option(False, "--all", "-a", help="连 draft/placeholder 一起列"),
) -> None:
    """列出 skill：状态、作用域、优先级、版本、正文 token 数。"""
    hub = _hub()
    rows = hub.skills if all_ else hub.available
    if not rows:
        console.print("[dim]没有 skill[/]")
        return
    t = Table(show_header=True, header_style="bold")
    for col in ("name", "状态", "scope", "优先级", "版本", "token", "参考", "sha", "owner"):
        t.add_column(col)
    for s in sorted(rows, key=lambda x: (-x.priority, x.name)):
        status = f"[green]{s.status}[/]" if s.active else f"[dim]{s.status}[/]"
        t.add_row(
            s.name, status, s.scope, str(s.priority), s.version,
            f"{s.tokens:,}", str(len(s.references)), s.sha[:7], s.owner or "—",
        )
    console.print(t)
    console.print(f"[dim]{len(hub.available)} 个生效 · 版本记录在 {HISTORY_DIR}[/]")


@app.command("show")
def show_cmd(name: str = typer.Argument(..., help="skill 名")) -> None:
    """看一份 skill 的元数据与正文。"""
    hub = _hub()
    s = next((x for x in hub.skills if x.name == name), None)
    if s is None:
        console.print(f"[red]没有 skill {name!r}[/]")
        raise typer.Exit(1)
    console.print(
        Panel(
            f"{s.description}\n\n[dim]scope {s.scope} · 适用 {', '.join(s.applies_to) or '全部'} · "
            f"阶段 {', '.join(s.stage) or '全部'} · 优先级 {s.priority} · v{s.version} · "
            f"{s.tokens:,} token · sha {s.sha} · {s.path}[/]",
            title=f"{s.name}（{s.status}）",
            border_style="cyan",
        )
    )
    console.print(s.body.strip())
    if s.references:
        console.print(Panel(s.reference_index(), border_style="dim"))


@app.command("history")
def history_cmd(name: str = typer.Argument(..., help="skill 名")) -> None:
    """版本记录。每次内容变化记一条，按内容哈希去重。"""
    hub = _hub()
    entries = hub.history(name)
    if not entries:
        console.print(f"[dim]{name} 没有版本记录[/]")
        return
    current = next((x.sha for x in hub.skills if x.name == name), "")
    t = Table(show_header=True, header_style="bold")
    for col in ("#", "记录时间", "文件修改时间", "sha", "版本", "状态", "token", "owner", ""):
        t.add_column(col)
    for i, e in enumerate(entries, 1):
        mark = "[green]← 当前[/]" if e["sha"] == current else ""
        t.add_row(
            str(i),
            _dt.datetime.fromtimestamp(e["recorded_at"]).strftime("%m-%d %H:%M"),
            _dt.datetime.fromtimestamp(e["mtime"]).strftime("%m-%d %H:%M"),
            e["sha"][:7], e.get("version", ""), e.get("status", ""),
            f"{e.get('tokens', 0):,}", e.get("owner") or "—", mark,
        )
    console.print(t)
    console.print("[dim]回滚：agent skill rollback <name> [--to sha][/]")


@app.command("rollback")
def rollback_cmd(
    name: str = typer.Argument(..., help="skill 名"),
    to: str = typer.Option("", "--to", help="回到哪个 sha（见 history）。留空 = 上一版"),
) -> None:
    """把某个历史版本写回文件。当前版本本身也在记录里，回滚错了可以再滚回来。"""
    hub = _hub()
    try:
        s = hub.rollback(name, to)
    except (KeyError, ValueError, RuntimeError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e
    console.print(f"[green]✓[/] {name} 已回滚到 {s.sha}（v{s.version}），文件 {s.path}")


@app.command("check")
def check_cmd() -> None:
    """体检：哪些文件解析不了、哪些正文超长、哪些还没写 owner。"""
    problems: list[str] = []
    for f in skill_files(SKILLS_DIR):
        if f.parent.name == "references":
            continue
        s = parse_skill(f)
        if s is None:
            problems.append(f"{f.relative_to(PROJECT_ROOT)}：frontmatter 解析失败或缺 name")
            continue
        if s.active and s.tokens > 5000:
            problems.append(f"{s.name}：正文约 {s.tokens:,} token，超过 5K 建议拆分")
        if s.active and (not s.owner or s.owner.upper() == "TODO"):
            problems.append(f"{s.name}：没写 owner，出问题找不到人")
        if s.active and len(s.description) < 10:
            problems.append(f"{s.name}：description 太短，模型没法判断何时该用它")
    if not problems:
        console.print("[green]✓[/] 所有 skill 文件正常")
        return
    for p in problems:
        console.print(f"[yellow]⚠[/] {p}")
    raise typer.Exit(1)


def _path(name: str) -> Path:
    return SKILLS_DIR / f"{name}.md"

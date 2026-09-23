"""`agent memory` —— 记忆库的运维面。

三件只有人能做的事：
  brief    看现在的 Memory Brief 长什么样（规则版，不调模型）
  promote  把一条记忆晋升到账号层 —— 推测不得自动晋升，只有人能点这个
  forget   作废一条错记忆 —— 记忆会被反复注入上下文，错一条影响后面每一轮
"""

from __future__ import annotations

import datetime as _dt

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...capabilities.memory.brief import build_brief
from ...capabilities.memory.store import Layer, MemoryStore
from ...envdetect import PROJECT_ROOT, workspace_root  # noqa: F401

console = Console()
app = typer.Typer(help="记忆库：列表 / 简报 / 晋升 / 作废", no_args_is_help=True)

MEM_DIR = workspace_root() / "memory"


def _store() -> MemoryStore:
    return MemoryStore(MEM_DIR)


@app.command("list")
def list_cmd(
    layer: str = typer.Option("", "--layer", help="account / project / session"),
    project: str = typer.Option("", "--project", help="只看某个项目/会话"),
    limit: int = typer.Option(30, "--limit"),
) -> None:
    """列出记忆：层、来源、极性、内容、命中次数。"""
    store = _store()
    lay = Layer(layer) if layer else None
    items = store.all(layer=lay, project_id=project)[-limit:]
    if not items:
        console.print("[dim]没有记忆[/]")
        return
    t = Table(show_header=True, header_style="bold")
    for col in ("id", "时间", "层", "来源", "极性", "内容", "命中"):
        t.add_column(col)
    for m in items:
        pol = {k.polarity.value for k in m.keywords}
        t.add_row(
            m.id,
            _dt.datetime.fromtimestamp(m.created_at).strftime("%m-%d %H:%M"),
            m.layer.value,
            m.source.value,
            "/".join(sorted(pol)) or "—",
            m.content[:60],
            str(m.hit_count),
        )
    console.print(t)
    console.print(f"[dim]共 {len(store)} 条，存于 {MEM_DIR}[/]")


@app.command("brief")
def brief_cmd(
    topic: str = typer.Option("", "--topic", help="按主题过滤建议与参考（避雷不过滤）"),
    stage: str = typer.Option("", "--stage", help="当前环节，如 正文 / 分镜"),
    project: str = typer.Option("", "--project", help="项目/会话 id"),
) -> None:
    """看规则版 Memory Brief：必须 / 避雷 / 建议 / 参考 / 待裁决的矛盾。"""
    b = build_brief(_store(), project_id=project, topic=topic, stage=stage)
    if b.empty:
        console.print("[dim]简报为空：没有约束、打回记录或偏好[/]")
        return
    console.print(Panel(b.render(), title="Memory Brief", border_style="magenta"))
    console.print(
        Panel(b.pin_text() or "（无硬约束）", title="pin 进上下文的形态", border_style="dim")
    )


@app.command("promote")
def promote_cmd(
    memory_id: str = typer.Argument(..., help="记忆 id，见 list"),
    by: str = typer.Option("human", "--by", help="human / data。模型不能晋升"),
) -> None:
    """晋升到账号层：从"这个项目这么要求"变成"这个号一直这么要求"。"""
    store = _store()
    try:
        m = store.promote(memory_id, by=by)
    except (KeyError, ValueError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e
    console.print(
        f"[green]✓[/] {m.id} 已晋升到 {m.layer.value} 层（来源 {m.source.value}）：{m.content}"
    )


@app.command("forget")
def forget_cmd(memory_id: str = typer.Argument(..., help="记忆 id")) -> None:
    """作废一条记忆（标记为已被人覆盖，不物理删除）。"""
    store = _store()
    try:
        m = store.forget(memory_id)
    except KeyError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e
    console.print(f"[green]✓[/] 已作废 {m.id}：{m.content}")

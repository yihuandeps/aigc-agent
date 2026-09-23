"""`agent assets` —— 产物文件改名与清单。

产物目录里的图片/视频原来一律叫 `as_xxx.png/mp4`，剪辑时排不出顺序。
`rename` 按 domain/media/naming.py 的规则补改成带序号的可读名（只动仍叫 as_… 的文件，
重复跑幂等），`manifest` 出一份「文件 ↔ 资产 id ↔ 说明」的清单。
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ...capabilities.memory.session import SessionSnapshot
from ...domain.assets.store import AssetStore
from ...domain.media.naming import apply_renames, plan_renames, write_manifest
from ...envdetect import PROJECT_ROOT

console = Console()
app = typer.Typer(help="产物文件：改成带序号的可读文件名 / 产物清单", no_args_is_help=True)
WORKSPACE = PROJECT_ROOT / "workspace"


def _root(session: str, out: str) -> Path:
    """产物目录：--out 优先，否则用该 session 记住的目录，再否则默认目录。"""
    if out:
        return Path(out)
    snap = SessionSnapshot(WORKSPACE / "memory" / "sessions", session or "default")
    if snap.output_dir:
        return Path(snap.output_dir)
    return WORKSPACE / "output" / (session or "default")


@app.command("rename")
def rename_cmd(
    session: str = typer.Option("default", "--session", help="哪个会话的产物目录"),
    out: str = typer.Option("", "--out", "-o", help="直接指定产物目录"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只看改名计划，不动文件"),
) -> None:
    """把已生成的图片/视频改成带序号的可读文件名（第01集-03_2场_镜9-18.mp4 这种）。"""
    root = _root(session, out)
    if not root.exists():
        console.print(f"[red]产物目录不存在：{root}[/]")
        raise typer.Exit(1)
    store = AssetStore(WORKSPACE / "assets")
    plan = plan_renames(store, root)
    if not plan:
        console.print(f"[dim]{root} 里没有需要改名的文件（都已是可读名，或没有图/视频）[/]")
        return

    t = Table(show_header=True, header_style="bold")
    t.add_column("现在")
    t.add_column("改成")
    t.add_column("类型")
    t.add_column("资产")
    for r in plan:
        t.add_row(r.old.name, r.new.name, r.why, r.asset_id)
    console.print(t)
    if dry_run:
        console.print(f"[dim]预览 {len(plan)} 项，未改动。去掉 --dry-run 执行[/]")
        return

    report = apply_renames(store, plan)
    manifest = write_manifest(store, root)
    console.print(f"[green]已改名 {len(report.done)} 个文件[/]")
    if report.failed:
        console.print(f"[yellow]{len(report.failed)} 个没改成：[/]\n{report.render_failed()}")
    if manifest:
        console.print(f"[dim]清单：{manifest}[/]")


@app.command("manifest")
def manifest_cmd(
    session: str = typer.Option("default", "--session"),
    out: str = typer.Option("", "--out", "-o"),
) -> None:
    """生成产物清单（文件 ↔ 资产 id ↔ 说明），不改名。"""
    root = _root(session, out)
    path = write_manifest(AssetStore(WORKSPACE / "assets"), root)
    if path is None:
        console.print(f"[dim]{root} 里没有可列的图/视频[/]")
        return
    console.print(f"[green]已写出 {path}[/]")

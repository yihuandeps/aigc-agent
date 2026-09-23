"""`agent assets` —— 产物文件改名与清单；存量资产分项目（迁移）。

产物目录里的图片/视频原来一律叫 `as_xxx.png/mp4`，剪辑时排不出顺序。
`rename` 按 domain/media/naming.py 的规则补改成带序号的可读名（只动仍叫 as_… 的文件，
重复跑幂等），`manifest` 出一份「文件 ↔ 资产 id ↔ 说明」的清单。
`migrate` 把加项目维度之前的存量资产分到各自的项目（2026-09-23 审查缺口 A），可撤销。
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ...capabilities.memory.session import SessionSnapshot
from ...domain.assets.migrate import (
    agents_running,
    apply_migration,
    plan_migration,
    restore_migration,
)
from ...domain.assets.store import AssetStore
from ...domain.media.naming import apply_renames, plan_renames, write_manifest
from ...domain.output import default_root
from ...domain.project import project_key
from ...envdetect import PROJECT_ROOT, workspace_root

console = Console()
app = typer.Typer(help="产物文件：改名 / 清单；存量资产分项目", no_args_is_help=True)
WORKSPACE = workspace_root()


def _root(session: str, out: str) -> Path:
    """产物目录：--out 优先；指定了 --session 用它记住的目录；否则当前文件夹（同 chat）。"""
    if out:
        return Path(out)
    if session:
        snap = SessionSnapshot(WORKSPACE / "memory" / "sessions", session)
        if snap.output_dir:
            return Path(snap.output_dir)
    return default_root(WORKSPACE, session, PROJECT_ROOT)


def _store(root: Path) -> AssetStore:
    store = AssetStore(WORKSPACE / "assets")
    store.project = project_key(root)  # 只动这个项目的产物
    return store


@app.command("rename")
def rename_cmd(
    session: str = typer.Option("", "--session", help="哪个会话的产物目录（默认当前文件夹）"),
    out: str = typer.Option("", "--out", "-o", help="直接指定产物目录"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只看改名计划，不动文件"),
) -> None:
    """把已生成的图片/视频改成带序号的可读文件名（第01集-03_2场_镜9-18.mp4 这种）。"""
    root = _root(session, out)
    if not root.exists():
        console.print(f"[red]产物目录不存在：{root}[/]")
        raise typer.Exit(1)
    store = _store(root)
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
    session: str = typer.Option("", "--session"),
    out: str = typer.Option("", "--out", "-o"),
) -> None:
    """生成产物清单（文件 ↔ 资产 id ↔ 说明），不改名。"""
    root = _root(session, out)
    path = write_manifest(_store(root), root)
    if path is None:
        console.print(f"[dim]{root} 里没有可列的图/视频[/]")
        return
    console.print(f"[green]已写出 {path}[/]")


@app.command("migrate")
def migrate_cmd(
    apply: bool = typer.Option(False, "--apply", help="按计划执行（默认只看计划）"),
    force: bool = typer.Option(False, "--force", help="检测到疑似有 Agent 在跑时也执行"),
    restore: str = typer.Option("", "--restore", help="撤销某次迁移：传它的 manifest.json 路径"),
) -> None:
    """把加项目维度之前的存量资产分到各自的项目；测试桩移进回收站。全部可撤销。"""
    if restore:
        done = restore_migration(Path(restore))
        console.print(f"[green]已撤销[/] {done}")
        return
    plan = plan_migration(WORKSPACE)
    console.print(escape(plan.render()))
    if not apply:
        console.print(
            "\n[dim]只是计划，没有改任何东西。确认后：agent assets migrate --apply"
            "（先关掉所有 Agent 窗口）[/]"
        )
        return
    busy = agents_running(WORKSPACE)
    if busy and not force:
        console.print(
            f"[red]最近两分钟还有会话日志在写（{', '.join(busy[:3])}），像是有 Agent 在跑。[/]\n"
            "[dim]老进程内存里的资产没有项目键，它一回写就把迁移结果盖掉。关掉所有 Agent 窗口"
            "再来；确定没有在跑就加 --force。[/]"
        )
        raise typer.Exit(1)
    manifest = apply_migration(plan, WORKSPACE)
    console.print(
        f"[green]迁移完成[/]。撤销：agent assets migrate --restore \"{manifest}\""
    )

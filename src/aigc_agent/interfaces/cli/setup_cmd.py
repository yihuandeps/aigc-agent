"""`agent setup` —— 换台机器后逐条告诉你缺什么。

本机能自动就位的（代理、已配的密钥）不打扰你；缺的**不猜也不静默降级**，
明确说少了哪个能力、去哪申请、填到哪。
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...envdetect import PROJECT_ROOT, detect

console = Console()


def run_setup() -> None:
    r = detect()

    console.print(Panel("环境检查", border_style="cyan"))

    # ---- 基础 ----
    t = Table(show_header=False, box=None)
    t.add_row(
        "[green]✓[/]" if r.dotenv_loaded else "[yellow]○[/]",
        ".env",
        f"加载 {len(r.dotenv_loaded)} 个变量" if r.dotenv_loaded else "不存在或为空",
    )
    t.add_row(
        "[green]✓[/]" if r.proxy else "[yellow]○[/]",
        "代理",
        f"{r.proxy}  [dim]（来自 {r.proxy_source}）[/]" if r.proxy else
        "未配。国内网络下多数外部 API 连不通，多半要配",
    )
    t.add_row(
        "[green]✓[/]" if r.ffmpeg else "[red]✗[/]",
        "ffmpeg",
        "可用" if r.ffmpeg else "缺失 —— 视频拼接/配音/字幕都用不了",
    )
    console.print(t)

    # ---- 密钥 ----
    console.print()
    kt = Table(show_header=True, header_style="bold")
    kt.add_column("", width=2)
    kt.add_column("密钥")
    kt.add_column("影响的能力")
    kt.add_column("申请地址")
    for key in r.present:
        what = next((w for k, w, _, _ in _reqs() if k == key), "")
        kt.add_row("[green]✓[/]", key, what, "[dim]已配置[/]")
    for key, what, where, required in r.missing:
        kt.add_row("[red]✗[/]" if required else "[yellow]○[/]", key, what, where)
    console.print(kt)

    # ---- 结论 ----
    console.print()
    if r.ready and r.ffmpeg:
        console.print("[green]环境就绪[/]，可以直接跑：")
        console.print("  [bold]agent chat[/]                  对话")
        console.print('  [bold]agent video make "主题"[/]      出一条短视频')
        if r.missing:
            names = "、".join(w for _, w, _, _ in r.missing)
            console.print(f"\n[dim]未配置的可选能力：{names}[/]")
        return

    console.print("[yellow]还差一些东西：[/]")
    if not r.ffmpeg:
        console.print("\n[bold]ffmpeg[/]（视频合成必需）")
        console.print("  winget install Gyan.FFmpeg   或   choco install ffmpeg")
        console.print("  装完重开终端，用 ffmpeg -version 验证")

    if r.blocking or r.missing:
        console.print(f"\n[bold]密钥[/] —— 编辑 {PROJECT_ROOT / '.env'}：")
        console.print()
        for key, what, where, required in r.missing:
            mark = "必需" if required else "可选"
            console.print(f"  [dim]# {what}（{mark}）· {where}[/]")
            console.print(f"  {key}=你的key")
        console.print()
        console.print("[dim]没有 .env 的话先复制模板：copy .env.example .env[/]")

    if not r.proxy:
        console.print("\n[bold]代理[/]（若外部 API 连不通）")
        console.print("  在 .env 里加 HTTPS_PROXY=http://127.0.0.1:端口")
        console.print("  [dim]这一项没配会伪装成「key 无效」，很难查，建议先排除[/]")

    console.print("\n配完再跑一次 [bold]agent setup[/] 确认。")


def _reqs():
    from ...envdetect import REQUIREMENTS

    return REQUIREMENTS


def register(app: typer.Typer) -> None:
    @app.command()
    def setup(
        write: bool = typer.Option(False, "--write", help="从 .env.example 创建 .env"),
    ) -> None:
        """检查环境，列出还缺什么。换台机器先跑这个。"""
        if write:
            src, dst = PROJECT_ROOT / ".env.example", PROJECT_ROOT / ".env"
            if dst.exists():
                console.print(f"[yellow]{dst} 已存在，不覆盖[/]")
            elif src.exists():
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                console.print(f"[green]已创建 {dst}[/]，填入你的密钥后再跑 agent setup")
            else:
                console.print("[red]找不到 .env.example[/]")
            console.print()
        run_setup()

"""`agent new` —— 引导式入口，先问清楚做哪条产线。

现在有两条产线，流程、配方、成本完全不同：

    短视频   30 秒资讯口播，配方驱动，单人口播，无角色
    短剧     带剧情对白的连续剧，三段式，有角色/服装/跨集一致性

走错一条不是"效果差一点"，是整条链白跑 —— 而且要等到出片才看得出来。
所以入口处先问，不猜。

已经知道自己要什么的人不用走这里，直接 `agent video` / `agent drama`。
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()


def _read(text: str, file: str) -> str:
    if file:
        return Path(file).read_text(encoding="utf-8")
    if text:
        return text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


LINES = {
    "1": ("video", "短视频", "30 秒资讯口播 · 拉真实热榜定选题 · 快切 · 自动选音色"),
    "2": ("drama", "短剧", "带剧情对白 · 角色主形象保一致性 · 视频模型原生出声"),
}


def register(app: typer.Typer) -> None:
    @app.command("new")
    def new_cmd(
        topic: str = typer.Argument("", help="主题（短视频）或剧本（短剧），可留空"),
        file: str = typer.Option("", "--file", "-f", help="剧本文件（短剧用）"),
    ) -> None:
        """开始做内容。会先问你要做短视频还是短剧。"""
        console.print(
            Panel(
                "\n".join(
                    f"[cyan]{k}[/] [bold]{name}[/]\n   [dim]{note}[/]"
                    for k, (_, name, note) in LINES.items()
                ),
                title="要做哪种内容？",
                border_style="cyan",
            )
        )
        pick = console.input("选一个（1/2）：").strip().lower()
        chosen = LINES.get(pick)
        if chosen is None:
            # 也认名字，省得记序号
            for _k, (key, name, _n) in LINES.items():
                if pick in (key, name) or (pick and pick in name):
                    chosen = (key, name, "")
                    break
        if chosen is None:
            console.print("[red]没认出来。直接跑 agent video 或 agent drama 也行。[/]")
            raise typer.Exit(1)

        key, name, _ = chosen
        console.print(f"[dim]→ {name}[/]\n")

        if key == "video":
            _go_video(topic)
        else:
            _go_drama(topic, file)


def _go_video(topic: str) -> None:
    from .video_cmd import make as video_make

    if not topic:
        topic = console.input("想做什么方向？（如：科技 / 职场 / 健康）").strip()
    if not topic:
        console.print("[red]没有方向就没法拉热榜定选题。[/]")
        raise typer.Exit(1)
    console.print(f"[dim]跑的是：agent video make \"{topic}\"[/]\n")
    video_make(topic=topic)


def _go_drama(topic: str, file: str) -> None:
    from .drama_cmd import run_cmd as drama_run

    script = _read(topic, file)
    if not script.strip():
        console.print(
            "[yellow]短剧要先有剧本。[/]\n"
            "  · 已经有了：[bold]agent new -f 剧本.txt[/]\n"
            "  · 还没有：  [bold]agent chat[/] 里说「我要做一部XX题材的短剧」，"
            "它会用 drama-script 方法论带你立项、写人设、出大纲、分集"
        )
        raise typer.Exit(1)
    console.print("[dim]跑的是：agent drama run（默认只到拆解，不花生成的钱）[/]\n")
    drama_run(text=script, file="", render=False, film=0, face="", lang="")

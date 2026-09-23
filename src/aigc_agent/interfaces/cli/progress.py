"""任务进度面板 —— 终端里的实时小窗口。

数据源是事件总线：工具开始/结束（TOOL_CALL/TOOL_RESULT）、批量进度
（BATCH_PROGRESS，渲染函数报 done/total）、成本（COST）。只读事件不改状态。

Live 面板只在一轮对话执行期间打开；问人（权限确认/预算护栏/人审决策）时
要暂停 —— 终端输入提示和 Live 刷新区域叠在一起就没法看了。
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from ...harness.events.bus import Event, EventType

# 工具 → 阶段名：给非批量调用一个可读的「当前在干嘛」
_STAGE = {
    "save_draft": "撰写 / 存稿",
    "request_review": "等待人审",
    "drama_write": "写剧本",
    "drama_expand": "扩写剧本",
    "drama_storyboard": "拆分镜脚本",
    "drama_assets": "生成资产库",
    "drama_shots": "生成视频提示词",
    "drama_render_assets": "渲染参考图",
    "drama_render_shots": "渲染分镜视频",
    "compose_video": "拼接成片",
    "gen_image": "生成图片",
    "gen_video": "生成视频",
    "tts": "合成配音",
    "transcribe": "转写 / 字幕",
    "fan_out_candidates": "并行出候选",
    "check_compliance": "合规机审",
    "build_release_package": "打待发布包",
    "search_library": "检索",
}

_BAR_WIDTH = 20


class ProgressBoard:
    """进度窗状态 + Live 生命周期。on_event 是纯状态更新，可脱离终端测试。"""

    def __init__(self, console: Console) -> None:
        self.console = console
        # 非终端（管道/重定向）下不开 Live：转义码只会把输出搞花
        self.enabled = console.is_terminal
        self._live: Live | None = None
        self.reset()

    def reset(self) -> None:
        self.stage = "准备中…"
        self.done = 0
        self.total = 0
        self.active: dict[str, int] = {}  # 运行中的工具计数
        self.iterations = 0
        self.started = time.perf_counter()
        self.cost = 0.0

    # ---------- 事件订阅 ----------

    def on_event(self, ev: Event) -> None:
        d = ev.data
        if ev.type is EventType.LOOP_START:
            self.reset()
        elif ev.type is EventType.ITERATION_START:
            self.iterations = d.get("iteration", self.iterations)
        elif ev.type is EventType.TOOL_CALL:
            name = d.get("tool", "?")
            self.active[name] = self.active.get(name, 0) + 1
            self.stage = _STAGE.get(name, self.stage)
        elif ev.type in (EventType.TOOL_RESULT, EventType.TOOL_ERROR):
            n = self.active.get(d.get("tool", "?"), 0) - 1
            if n > 0:
                self.active[d.get("tool", "?")] = n
            else:
                self.active.pop(d.get("tool", "?"), None)
        elif ev.type is EventType.BATCH_PROGRESS:
            self.stage = d.get("stage", self.stage)
            self.done = d.get("done", self.done)
            self.total = d.get("total", self.total)
        elif ev.type is EventType.COST:
            self.cost += d.get("cost") or 0.0

    # ---------- 渲染 ----------

    def __rich__(self) -> Panel:  # Live 每次刷新都会调它
        lines: list[str] = []
        if self.total:
            pct = min(1.0, self.done / self.total)
            bar = "█" * round(_BAR_WIDTH * pct) + "░" * (_BAR_WIDTH - round(_BAR_WIDTH * pct))
            lines.append(
                f"[bold]{self.stage}[/]  {bar} [cyan]{self.done}/{self.total}[/]（{pct:.0%}）"
            )
        else:
            lines.append(f"[bold]{self.stage}[/]")
        if self.active:
            lines.append(
                "运行中：" + "，".join(f"{n} ×{c}" for n, c in sorted(self.active.items()))
            )
        m, s = divmod(int(time.perf_counter() - self.started), 60)
        cost = f" · ¥{self.cost:.3f}" if self.cost else ""
        lines.append(f"[dim]迭代 {self.iterations} · 用时 {m}m{s:02d}s{cost}[/]")
        return Panel("\n".join(lines), title="任务进度", border_style="cyan")

    # ---------- Live 生命周期 ----------

    @contextmanager
    def running(self) -> Iterator[None]:
        """一轮对话执行期间打开进度窗。重入安全（人审恢复会嵌套进来）。"""
        if not self.enabled or self._live is not None:
            yield
            return
        # 不重定向 stdout/stderr：输入框已经整会话接管了 sys.stdout，Live 再换一次，
        # 停下时可能换回一个失效的代理，回复就再也显示不出来（2026-09-23 审查）
        self._live = Live(
            self, console=self.console, refresh_per_second=4,
            redirect_stdout=False, redirect_stderr=False,
        )
        self._live.start()
        try:
            yield
        finally:
            self._live.stop()
            self._live = None

    @contextmanager
    def paused(self) -> Iterator[None]:
        """问人时暂停 Live —— 输入提示不能和刷新中的面板叠在一起。"""
        live = self._live
        if live is None:
            yield
            return
        live.stop()
        try:
            yield
        finally:
            live.start()

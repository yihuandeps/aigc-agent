"""`agent sessions` —— 会话回放、成本看板、痕迹重建（M6 的运维面）。

事件流已按会话落盘（workspace/logs/sessions/*.jsonl）。同一份数据三种看法：
  replay  按时间顺序看发生了什么（工具调用、人审、预算拦截、子代理…）
  cost    按角色/模型/工具/媒体算成本与耗时
  trace   把事件重新喂给 ExecutionTrace，得到当时那张 DAG
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...envdetect import PROJECT_ROOT
from ...harness.events.bus import Event, EventType
from ...harness.events.log import EventLog, find_session, list_sessions
from ...harness.events.stats import render_stats, session_stats
from ...harness.execution.trace import ExecutionTrace

console = Console()
app = typer.Typer(help="会话回放 / 成本看板 / 痕迹重建", no_args_is_help=True)

LOG_DIR = PROJECT_ROOT / "workspace" / "logs" / "sessions"

_QUIET = {
    EventType.ITERATION_START,
    EventType.CONTEXT_ASSEMBLED,
    EventType.MODEL_REQUEST,
    EventType.MODEL_RESPONSE,
    EventType.COST,
    EventType.CAPABILITY_BUDGET,
    EventType.ROUTE,
}


def _brief(value: Any, limit: int = 80) -> str:
    if value in (None, "", {}, []):
        return ""
    s = str(value).replace("\n", " ")
    return s if len(s) <= limit else s[:limit] + "…"


def _line(ev: Event) -> str | None:
    d = ev.data
    t = ev.type
    if t is EventType.LOOP_START:
        return f"[bold cyan]▶ 第 {d.get('turn')} 轮[/] {_brief(d.get('input_preview'))}"
    if t is EventType.LOOP_END:
        cost = f" ¥{d['cost']:.4f}" if d.get("cost") else ""
        return (
            f"[cyan]■ 轮结束[/] {d.get('stop_reason')} · {d.get('iterations')} 次迭代{cost} "
            f"{_brief(d.get('text_preview'))}"
        )
    if t is EventType.TOOL_CALL:
        return f"  → {d.get('tool')} [dim]{_brief(d.get('args'))}[/]"
    if t is EventType.TOOL_RESULT:
        return f"  ← {d.get('tool')} [dim]{d.get('duration_ms', 0)}ms {_brief(d.get('preview'))}[/]"
    if t is EventType.TOOL_ERROR:
        return f"  [red]✗ {d.get('tool')}[/] {_brief(d.get('error'))}"
    if t is EventType.CHECKPOINT_REACHED:
        return f"  [yellow]⏸ 人审[/] {d.get('stage') or d.get('node')} {_brief(d.get('question'))}"
    if t is EventType.CHECKPOINT_DECIDED:
        return f"  [yellow]✎ 决策 {d.get('decision')}[/] {_brief(d.get('reason'))}"
    if t is EventType.BUDGET_EXCEEDED:
        return f"  [red]¥ 预算护栏[/] {d.get('tool')} {_brief(d.get('reason'))}"
    if t is EventType.PERMISSION_DENY:
        return f"  [red]⛔ {d.get('tool')}[/] {_brief(d.get('reason'))}"
    if t is EventType.WARNING:
        return f"  [yellow]⚠ {_brief(d.get('message'))}[/]"
    if t is EventType.SUBAGENT_END:
        mark = "✓" if d.get("ok") else "✗"
        return (
            f"  {mark} 子代理 {d.get('name')} {d.get('iterations')} 次 "
            f"[dim]{d.get('duration_ms')}ms[/]"
        )
    if t is EventType.FANOUT_END:
        return (
            f"  ⇶ 扇出 {d.get('name')} {d.get('ok')}/{d.get('total')} "
            f"[dim]{d.get('duration_ms')}ms[/]"
        )
    if t is EventType.PARALLEL_END:
        return (
            f"  ⇶ 并行 {d.get('node')} {d.get('branches')} 支 → {d.get('join')} "
            f"[dim]{d.get('elapsed_ms')}ms[/]"
        )
    if t is EventType.NODE_START:
        return f"  ▷ 节点 {d.get('node')} ({d.get('type')})"
    if t is EventType.NODE_DONE:
        return f"  ◁ 节点 {d.get('node')} → {', '.join(d.get('outputs') or []) or '—'}"
    if t is EventType.GRAPH_ROLLBACK:
        return (
            f"  [magenta]↩ 回退 {d.get('from_node')} → {d.get('to_node')}[/] "
            f"{_brief(d.get('reason'))}"
        )
    if t is EventType.TRACE_ROLLBACK:
        n = len(d.get("superseded") or [])
        return f"  [magenta]↩ 回退到 {d.get('asset')}[/] 作废 {n} 次调用"
    if t is EventType.WINDOW_EVICT:
        return f"  [magenta]⇤ 驱逐 {d.get('count')} 轮[/]"
    if t is EventType.SKILL_RELOAD:
        return f"  [magenta]↻ {_brief(d.get('message'))}[/]"
    if t is EventType.COMPLIANCE_CHECKED:
        return f"  ⚖ 机审 {'通过' if d.get('passed') else 'block ' + str(d.get('block'))}"
    if t is EventType.PACKAGE_BUILT:
        return f"  📦 待发布包 {d.get('package')} {d.get('platform')}"
    if t is EventType.METRICS_RECORDED:
        return f"  📈 数据 {d.get('package')} 播放 {d.get('views')}"
    if t in _QUIET:
        return None
    return f"  · {t.value} {_brief(d, 60)}"


def _resolve(key: str) -> Path:
    p = find_session(LOG_DIR, key)
    if p is None:
        console.print(f"[red]找不到会话 {key!r}。用 agent sessions list 看有哪些[/]")
        raise typer.Exit(1)
    return p


@app.command("list")
def list_cmd(limit: int = typer.Option(30, "--limit")) -> None:
    """按时间倒序列出会话：事件数、轮数、成本。"""
    infos = list_sessions(LOG_DIR, limit)
    if not infos:
        console.print(f"[dim]还没有会话记录（{LOG_DIR}）[/]")
        return
    t = Table(show_header=True, header_style="bold")
    for col in ("开始", "session", "事件", "轮", "成本", "大小"):
        t.add_column(col)
    for i in infos:
        t.add_row(
            i.started_text, i.session_id, str(i.events), str(i.turns),
            f"¥{i.cost:.4f}" if i.cost else "—", f"{i.size / 1024:.0f}KB",
        )
    console.print(t)
    console.print(
        "[dim]回放：agent sessions replay <session>   成本：agent sessions cost <session>[/]"
    )


@app.command("replay")
def replay_cmd(
    key: str = typer.Argument(..., help="session id 或文件名片段"),
    tail: int = typer.Option(0, "--tail", help="只看最后 N 条"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="连模型请求/成本事件一起看"),
) -> None:
    """按时间顺序回放一个会话。"""
    events = EventLog.read(_resolve(key))
    if tail:
        events = events[-tail:]
    shown = 0
    for ev in events:
        if verbose and ev.type in _QUIET:
            line = f"  · {ev.type.value} {_brief(ev.data, 100)}"
        else:
            line = _line(ev)
        if line is None:
            continue
        stamp = _dt.datetime.fromtimestamp(ev.ts).strftime("%H:%M:%S")
        console.print(f"[dim]{stamp}[/] {line}")
        shown += 1
    console.print(f"[dim]{shown} 条（共 {len(events)} 个事件）[/]")


@app.command("cost")
def cost_cmd(key: str = typer.Argument(..., help="session id 或文件名片段")) -> None:
    """成本看板：按角色、媒体、工具。"""
    events = EventLog.read(_resolve(key))
    console.print(Panel(render_stats(session_stats(events)), title=key, border_style="cyan"))


@app.command("trace")
def trace_cmd(
    key: str = typer.Argument(..., help="session id 或文件名片段"),
    mermaid: bool = typer.Option(False, "--mermaid", help="输出 mermaid 而不是文本视图"),
) -> None:
    """把事件重新喂给执行痕迹，重建当时的 DAG。"""
    events = EventLog.read(_resolve(key))
    trace = ExecutionTrace()
    EventLog.replay(events, trace._on_event)  # noqa: SLF001 — 回放就是走它在线时的入口
    console.print(Panel(trace.to_mermaid() if mermaid else trace.to_text(), title="执行痕迹",
                        border_style="magenta"))
    console.print(f"[dim]{trace.summary()}[/]")

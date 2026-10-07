"""M18 CLI —— 入口。

  agent doctor   自检：配置、密钥、工具注册、MCP、连通性；--cache 实测前缀缓存
  agent models   拉服务端实际可用的模型 id（核对 models.yaml 填得对不对）
  agent chat     交互式对话，跑主 Loop
  agent mcp      MCP server 状态 / 权限审计 / 手动调用
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Any

# Windows 中文环境下 stdout 默认是 GBK，编不了 ✓ ✗ → 之类的符号。
# 在最早处改掉，比要求用户设 PYTHONUTF8=1 可靠。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import typer  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.markup import escape  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

from ...app import PROJECT_ROOT, Agent, load_dotenv  # noqa: E402
from ...domain.aspect import (  # noqa: E402
    VIDEO_ASPECTS,
    aspect_label,
    parse_aspect,
    supported_aspects,
)
from ...domain.lines import get_line, is_off, parse_line  # noqa: E402
from ...domain.lines import menu as lines_menu  # noqa: E402
from ...domain.media.naming import apply_renames, plan_renames, write_manifest  # noqa: E402
from ...domain.project import normalize_root  # noqa: E402
from ...harness.events.bus import Event, EventType  # noqa: E402
from ...harness.execution.loop import LoopResult, StopReason  # noqa: E402
from ...harness.model.config import ModelsConfig  # noqa: E402
from ...harness.tools.provider import ToolMeta  # noqa: E402
from .analytics_cmd import app as analytics_app  # noqa: E402
from .assets_cmd import app as assets_app  # noqa: E402
from .budget_prompt import parse_budget, render_budget  # noqa: E402
from .drama_cmd import app as drama_app  # noqa: E402
from .graph_cmd import app as graph_app  # noqa: E402
from .inputhub import InputHub  # noqa: E402
from .mcp_cmd import app as mcp_app  # noqa: E402
from .memory_cmd import app as memory_app  # noqa: E402
from .progress import ProgressBoard  # noqa: E402
from .promptbox import help_text, known_command, make_reader, search  # noqa: E402
from .quiet import install as quiet_shutdown_noise  # noqa: E402
from .release_cmd import app as release_app  # noqa: E402
from .rpa_cmd import app as rpa_app  # noqa: E402
from .sessions_cmd import app as sessions_app  # noqa: E402
from .setup_cmd import register as register_setup  # noqa: E402
from .skill_cmd import app as skill_app  # noqa: E402
from .start_cmd import register as register_start  # noqa: E402
from .story_cmd import app as story_app  # noqa: E402
from .video_cmd import app as video_app  # noqa: E402

app = typer.Typer(
    help="AIGC 内容创作 Agent —— 短视频/图文/文案生产",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
app.add_typer(graph_app, name="graph")
app.add_typer(video_app, name="video")
app.add_typer(rpa_app, name="rpa")
app.add_typer(story_app, name="story")
app.add_typer(drama_app, name="drama")
app.add_typer(mcp_app, name="mcp")
app.add_typer(skill_app, name="skill")
app.add_typer(memory_app, name="memory")
app.add_typer(release_app, name="release")
app.add_typer(analytics_app, name="analytics")
app.add_typer(sessions_app, name="sessions")
app.add_typer(assets_app, name="assets")
register_setup(app)
register_start(app)  # agent new：先问做短视频还是短剧


# ---------------------------------------------------------------- 事件渲染

VERBOSE_EVENTS = {
    EventType.MODEL_REQUEST,
    EventType.CONTEXT_ASSEMBLED,
    EventType.ITERATION_START,
    EventType.LOOP_START,
    EventType.LOOP_END,
}


def make_renderer(verbose: bool):
    def render(ev: Event) -> None:
        d = ev.data
        if ev.type is EventType.TOOL_CALL:
            console.print(f"  [cyan]→ {d['tool']}[/] [dim]{_brief(d.get('args'))}[/]")
        elif ev.type is EventType.TOOL_RESULT:
            mark = "[dim]（已截断）[/]" if d.get("truncated") else ""
            console.print(
                f"  [green]← {d['tool']}[/] [dim]{d.get('duration_ms', 0)}ms[/] "
                f"{_brief(d.get('preview'))}{mark}"
            )
        elif ev.type is EventType.TOOL_ERROR:
            console.print(f"  [red]✗ {escape(str(d['tool']))}[/] {escape(str(d.get('error')))}")
        elif ev.type is EventType.MODEL_RETRY:
            console.print(
                f"  [yellow]↻ 重试 {d['attempt']}/{d['max_attempts']}"
                f"（{d['delay_s']}s 后）[/] [dim]{escape(str(d.get('error')))}[/]"
            )
        elif ev.type is EventType.WINDOW_EVICT:
            why = "上下文超预算，预检提前剔" if d.get("reason") == "token_budget" else "批量驱逐"
            console.print(
                f"  [magenta]⇤ {why} {d['count']} 轮[/] "
                f"[dim]保留 {d['remaining']} 轮，已入队待提取关键词[/]"
            )
        elif ev.type is EventType.CONTEXT_COMPACTED and verbose:
            mode = "应急" if d.get("shrink") else "常规"
            console.print(
                f"  [dim]⇣ 轮内压缩（{mode}）折叠 {d.get('folded')} 处，"
                f"约 {d.get('tokens', 0):,} token[/]"
            )
        elif ev.type is EventType.WARNING:
            console.print(f"  [yellow]⚠ {escape(str(d.get('message')))}[/]")
        elif ev.type is EventType.SKILL_RELOAD:
            console.print(f"  [magenta]↻ {escape(str(d.get('message')))}[/]")
        elif ev.type is EventType.COMPLIANCE_CHECKED:
            mark = "[green]通过[/]" if d.get("passed") else f"[red]block {d.get('block')}[/]"
            console.print(f"  [cyan]⚖ 机审 {d.get('asset')}[/] {mark} · warn {d.get('warn')}")
        elif ev.type is EventType.PACKAGE_BUILT:
            console.print(f"  [green]📦 待发布包 {d.get('package')}[/] [dim]{d.get('folder')}[/]")
        elif ev.type is EventType.METRICS_RECORDED:
            n = len(d.get("distilled") or [])
            console.print(
                f"  [cyan]📈 数据已录入[/] 样本 {d.get('samples')}"
                + (f" · 提炼 {n} 条记忆" if n else "")
            )
        elif ev.type is EventType.FANOUT_END:
            console.print(
                f"  [cyan]⇶ 扇出 {d.get('name')} {d.get('ok')}/{d.get('total')}[/] "
                f"[dim]{d.get('duration_ms', 0)}ms[/]"
            )
        elif ev.type is EventType.SUBAGENT_END:
            mark = "[green]✓[/]" if d.get("ok") else "[red]✗[/]"
            console.print(
                f"  {mark} 子代理 {escape(str(d.get('name')))} {d.get('iterations')} 次迭代 "
                f"[dim]{d.get('duration_ms', 0)}ms {escape(str(d.get('error') or ''))}[/]"
            )
        elif ev.type is EventType.BUDGET_EXCEEDED:
            console.print(
                f"  [red]¥ 预算护栏：{escape(str(d.get('tool')))}[/] "
                f"{escape(str(d.get('reason')))} [dim]（{escape(str(d.get('usage')))}）[/]"
            )
        elif ev.type is EventType.PERMISSION_DENY:
            console.print(
                f"  [red]⛔ {escape(str(d.get('tool')))} 被拒绝[/] "
                f"[dim]{escape(str(d.get('reason', '')))}[/]"
            )
        elif ev.type is EventType.LOOP_STOP_REASON:
            detail = f" — {escape(str(d['detail']))}" if d.get("detail") else ""
            console.print(f"  [yellow]■ 停机：{d.get('reason')}{detail}[/]")
        elif ev.type is EventType.CHECKPOINT_REACHED:
            console.print(
                f"  [yellow]⚑ 人审 · {d.get('stage') or '未标注环节'}[/] "
                f"[dim]{_brief(d.get('question'))}[/]"
            )
        elif ev.type is EventType.CHECKPOINT_DECIDED:
            verdict = {"adopt": "采纳", "revise": "打回", "reject": "退回"}.get(
                d.get("decision"), d.get("decision")
            )
            by = "（auto 自动）" if d.get("decided_by") == "auto" else ""
            reason = f" [dim]{_brief(d.get('reason'))}[/]" if d.get("reason") else ""
            console.print(f"  [yellow]⚐ 决策：{verdict}{by}[/]{reason}")
        elif verbose and ev.type in VERBOSE_EVENTS:
            console.print(f"  [dim]· {ev.type.value} {_brief(d, 160)}[/]")

    return render


def _brief(value: Any, limit: int = 90) -> str:
    """一行摘要，**已转义**可以直接拼进 rich 标记（工具参数 / 报错里的方括号会被当成
    标记解析，[/] 这类直接抛 MarkupError 把整个 chat 带崩 —— 2026-09-23 审查）。"""
    if value in (None, "", {}, []):
        return ""
    s = str(value).replace("\n", " ")
    return escape(s if len(s) <= limit else s[:limit] + "…")


def _line_label(agent: Any) -> str:
    line = get_line(getattr(agent, "content_line", ""))
    return line.label if line else "不限定"


async def _settle_output_dir(agent: Agent, hub: InputHub, explicit_session: bool) -> None:
    """开工时定产物目录（= 项目）。

    不带 --session 时会话按文件夹分，以**当前文件夹**为准；这个文件夹上次用 /out 改到了
    别处，就问一次用哪个（2026-09-23 审查：之前只要快照里记着目录就悄悄覆盖当前文件夹 ——
    default 快照里是 E:\\西游记，换任何文件夹启动产物都落进西游记，面板却写「当前文件夹」）。
    带 --session 时会话带着它记住的产物目录走。
    """
    store = agent.session_store
    if store is None or agent.output_prefs is None:
        return
    here = agent.output_prefs.root
    saved = store.output_dir
    if not saved:
        store.set_output_dir(str(here))
        return
    if normalize_root(saved) == normalize_root(here):
        return
    if explicit_session:
        agent.switch_project(Path(saved))
        return
    console.print(
        Panel(
            Text(
                f"这个文件夹上次用 /out 把产物目录改到了：{saved}\n"
                f"这次用哪个？回车 = 当前文件夹 {here}；2 = {saved}"
            ),
            title="产物目录",
            border_style="cyan",
        )
    )
    try:
        pick = (await hub.ask("> ")).strip()
    except EOFError:
        pick = ""
    if pick == "2":
        agent.switch_project(Path(saved))
    else:
        store.set_output_dir(str(here))


def _warn_store(agent: Agent) -> None:
    """资产库自检：读不出来的文件、还没分项目的旧资产 —— 开工时说一声，别让人以为东西没了。"""
    errs = list(getattr(agent.assets, "load_errors", []) or [])
    if errs:
        shown = "、".join(errs[:3]) + (f" 等 {len(errs)} 个" if len(errs) > 3 else "")
        console.print(f"[yellow]⚠ 资产库有文件读不出来（{escape(shown)}），这些资产暂时看不到[/]")
    unassigned = agent.assets.projects().get("", 0)
    if unassigned:
        console.print(
            f"[yellow]⚠ 资产库里还有 {unassigned} 份旧资产没分项目，会出现在每个项目里"
            "（新剧可能看到旧剧的剧本和资产库）。[/]\n"
            "[dim]  关掉其它 Agent 窗口后运行 agent assets migrate 先看分配计划，"
            "确认后 agent assets migrate --apply（可撤销）[/]"
        )


def _ratio_brief(agent: Agent) -> str:
    """一句话说清这个项目的视频画幅，以及锁定的视频模型支持哪些。"""
    where = "本项目设置" if agent.aspect_custom else "默认；短视频没设时按配方"
    lock = getattr(agent.media_fns, "video_lock", "") or ""
    listed = supported_aspects(agent.catalog, lock) if lock else []
    tail = f"；锁定的视频模型 {lock} 支持 {' / '.join(listed)}" if listed else ""
    return f"视频画幅 {aspect_label(agent.aspect_ratio)}（{where}）{tail}"


def _cut_brief(agent: Agent) -> str:
    """一句话说清这个项目镜头超 3 秒怎么处理。"""
    if agent.cut_block:
        return (
            "镜头超 3 秒：拦（本项目设置）—— 超了自动重生成 1 次，仍超的段不进成片、"
            "等你放行或重渲；/cut 标 改回只标出来"
        )
    return (
        "镜头超 3 秒：只标出来、照样进成片（默认；检测准不准还没验证）—— 第 1 集渲完抽看几段，"
        "确实超了就 /cut 拦"
    )


def _length_brief(agent: Agent) -> str:
    """一句话说清这个项目的集长：几分钟、分镜总长区间、每集大约几段视频。"""
    fmt = agent.episode_fmt
    if fmt.follow_script:
        return (
            "集长跟剧本走（本项目设置）：分镜总长按台词念完 + 必要的动作定，不按集长凑；"
            "每集段数按分镜时长算"
        )
    lo, hi = fmt.duration_range
    seg_lo, seg_hi = fmt.shot_range
    base = agent.base_episode_fmt.minutes
    where = "默认" if fmt.minutes == base else f"本项目设置，默认 {base:g} 分钟"
    return (
        f"一集 {fmt.minutes:g} 分钟（{where}）：分镜总长 {lo}–{hi} 秒，"
        f"每集约 {seg_lo}–{seg_hi} 段视频"
    )


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _ensure_dir(text: str) -> Path:
    """校验并创建产物目录。同步函数，经 to_thread 调，别在事件循环里直接做文件操作。
    先 resolve：相对路径（/out 蜘蛛精）存进快照后换个目录启动会解析成另一个项目
    （2026-09-24 审查）。"""
    p = Path(text).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


async def _ask_permission(meta: ToolMeta, args: dict[str, Any], hub: InputHub) -> bool:
    """L-external 的人工确认闸门；预算超限时也走这里（summary 会带上原因）。"""
    console.print()
    console.print(
        Panel(
            f"[bold]{meta.name}[/]  [dim]{meta.permission.value}[/]\n"
            f"{meta.summary}\n\n"
            f"[dim]参数：{_brief(args, 400)}[/]",
            title="[yellow]需要确认[/]",
            border_style="yellow",
        )
    )
    answer = await hub.ask("执行吗？(y/N) ")
    return answer.strip().lower() in {"y", "yes"}


def _candidate_view(agent: Agent, asset: Any) -> tuple[str, Path | None]:
    """人审面板里一个候选怎么显示：文本给开头 + 字数 + 全文在哪；图片/视频给本地路径和链接。

    之前一律打印 content[:1500] —— 图片/视频的 content 是空的，人看到一个空框
    （2026-09-23 审查）。
    """
    from ...domain.assets.store import local_copy

    local = local_copy(asset)
    kind = asset.type.value
    if kind in ("image", "video", "audio"):
        where = str(local) if local is not None else "（没有本地副本）"
        return f"{kind} · {asset.summary}\n本地文件：{where}\n链接：{asset.uri or '—'}", local
    body = agent.assets.content(asset.id)
    if len(body) > 1500:
        body = body[:1500] + (
            f"\n……（共 {len(body)} 字；全文在产物目录 texts/ 里，或让我 read_asset）"
        )
    return body or "（空）", None


def _open_file(p: Path) -> None:
    """用系统默认程序打开（看图/看视频）。打不开就把路径打出来。"""
    import os
    import subprocess

    try:
        if hasattr(os, "startfile"):
            os.startfile(str(p))  # noqa: S606 — 打开的是资产库登记过的本地产物
        else:
            subprocess.Popen(["xdg-open", str(p)])  # noqa: S603, S607
    except OSError:
        console.print(f"[dim]打不开，文件在：{escape(str(p))}[/]")


async def _confirm_budget(agent: Agent, hub: InputHub) -> None:
    """开工额度确认：把这次开工的额度列出来，回车确认或当场改（用户 2026-09-13 定的规则）。"""
    if agent.guard is None:
        return
    current = agent.budget_defaults()
    console.print(
        Panel(
            render_budget(current, agent.media_priced, agent.config.text.estimated_providers()),
            title="开工额度（这次开工最多花这些，超了会停下来问你）",
            border_style="yellow",
        )
    )
    raw = (
        await hub.ask(
            "[bold yellow]回车按这个开工；要改就输入"
            "（如：金额 300 视频秒 900 视频 80 图 100）：[/] "
        )
    ).strip()
    changes = parse_budget(raw) if raw else {}
    if raw and not changes:
        console.print("[dim]没认出来，按上面的额度开工；/budget set 随时改[/]")
    agent.apply_budget({**current, **changes})
    console.print(f"[green]额度已确认[/] [dim]{agent.guard.brief()}[/]\n")


async def _watched(hub: InputHub, coro: Any) -> LoopResult | None:
    """跑一轮并让输入枢纽盯着：期间能排队/停止/插队。被人停掉返回 None。"""
    try:
        return await hub.watch(coro)
    except asyncio.CancelledError:
        if not hub.stopped_by_user:
            raise
        console.print(
            "[yellow]■ 本轮已停止[/][dim]（没执行完的调用会在下一轮开始时补记为「已中断」；"
            "已提交给远端的生成任务在服务端照跑、照计费 —— 同样的请求再来会先取回，"
            "不重新付费；让我 media_tasks 可以看到它们）[/]"
        )
        return None


# ---------------------------------------------------------------- 人审决策（agentic）

_DECISIONS = {
    "a": "adopt",
    "adopt": "adopt",
    "r": "revise",
    "revise": "revise",
    "j": "reject",
    "reject": "reject",
}


def _parse_decision(raw: str) -> tuple[str, str] | None:
    """把人审输入解析成 (decision, 附言)：'a 控制在60集' → ('adopt', '控制在60集')。

    认不出（没以 a/r/j 开头）返回 None，让调用方重新问。
    全角空格按半角处理 —— 中文输入法下 'a　备注' 太常见了。
    """
    head, _, tail = raw.strip().replace("　", " ").partition(" ")
    decision = _DECISIONS.get(head.lower())
    if decision is None:
        return None
    return decision, tail.strip()


_MEM_ID = re.compile(r"mem_[0-9a-f]{10}")
_EVERYWHERE = ("全局 ", "全局：", "全局:", "所有项目 ", "--all ")


async def _remember(agent: Agent, arg: str) -> None:
    """/remember 一句话 → 这个项目的规则（人亲口定的，每轮 pin）；「/remember 全局 …」→ 所有项目；
    /remember mem_xxx → 把一条没有项目键、被隔离的旧记忆认领到这个项目。"""
    mem = agent.mem_agent
    if mem is None:
        console.print("[dim]没有装配记忆代理[/]")
        return
    if not arg:
        console.print(
            "[dim]用法：/remember 一句话（只对这个项目，每轮都带上）· /remember 全局 一句话"
            "（所有项目）· /remember mem_xxx（把一条待认领的旧记忆认领到这个项目）[/]"
        )
        return
    if _MEM_ID.fullmatch(arg):
        try:
            m = mem.store.claim(arg, mem.project_id)
        except (KeyError, ValueError) as e:
            console.print(f"[red]{escape(str(e))}[/]")
            return
        console.print(f"[green]已认领到这个项目[/]：{escape(m.content)}")
        return
    everywhere = False
    for p in _EVERYWHERE:
        if arg.startswith(p):
            everywhere, arg = True, arg[len(p):].strip()
            break
    try:
        m = mem.store.remember(arg, project_id=mem.project_id, everywhere=everywhere)
    except ValueError as e:
        console.print(f"[red]{escape(str(e))}[/]")
        return
    where = "所有项目" if everywhere else "这个项目"
    console.print(
        f"[green]记住了（{where}，每轮都会带上）[/]：{escape(m.content)}"
        f" [dim]{m.id}；不要了用 /forget {m.id}[/]"
    )


async def _forget(agent: Agent, hub: InputHub, arg: str) -> None:
    """/forget 关键词或记忆 id：先列出匹配的记忆，人确认 y 才作废（标记不删，可复盘）。"""
    mem = agent.mem_agent
    if mem is None:
        console.print("[dim]没有装配记忆代理[/]")
        return
    if not arg:
        console.print("[dim]用法：/forget 关键词 或 /forget mem_xxx（先列出来，你确认了才作废）[/]")
        return
    found = mem.store.find(arg, mem.project_id)
    if not found:
        console.print("[dim]没找到匹配的记忆（/brief 看这个项目现在带着哪些）[/]")
        return
    if len(found) > 10:
        console.print(
            f"[yellow]匹配到 {len(found)} 条，太多了 —— 关键词再具体一点，或直接给记忆 id[/]"
        )
        return
    for m in found:
        scope = m.project_id or ("全局" if m.layer.value == "account" else "待认领")
        console.print(f"  {m.id}  [dim]{escape(scope)}[/]  {escape(m.content[:80])}")
    ans = (await hub.ask(f"作废这 {len(found)} 条？(y/N) ")).strip().lower()
    if ans not in ("y", "yes"):
        console.print("[dim]没动[/]")
        return
    for m in found:
        mem.store.forget(m.id)
    console.print(f"[green]已作废 {len(found)} 条[/] [dim]（标记作废、不删文件，可复盘）[/]")


def _print_result(result: LoopResult) -> None:
    console.print()
    # 模型回复原样显示，不当 rich 标记解析（回复里的 [/] [/budget allow 50] 之类会让
    # rich 抛 MarkupError，整个 chat 直接退出 —— 2026-09-23 审查）
    body = Text(result.text) if result.text else Text("（无文本输出）", style="dim")
    console.print(Panel(body, border_style="green"))
    cost = f" · ¥{result.cost:.4f}" if result.cost else ""
    console.print(f"[dim]{result.iterations} 次迭代 · {result.stop_reason.value}{cost}[/]")
    fails = getattr(result, "tool_failures", None) or []
    if fails:
        # 不管模型最后怎么说，失败的步骤都列出来给人看
        shown = "\n".join(f"  ✗ {escape(f)}" for f in fails[:8])
        more = f"\n  …另有 {len(fails) - 8} 次" if len(fails) > 8 else ""
        console.print(f"[yellow]本轮有 {len(fails)} 次工具调用失败：[/]\n{shown}{more}")


async def _decide_review(agent: Agent, board: ProgressBoard, hub: InputHub) -> LoopResult | None:
    """人审决策交互：展示问题与候选 → 收 a/r/j → resume_turn 结案。

    与图模式 graph_cmd._ask 同一套约定。a 可带补充要求（'a 控制在60集'）；
    r/j 必须给理由，可同行给（'r 开头太硬广'），没给就追问 ——
    不记原因的话下一版会犯一模一样的错。
    决策口也认 /auto on：当场开自动模式，本次按采纳结案，后续人审不再问。
    决策收完后 resume 会再跑起来，那段执行期间进度窗重新打开。
    """
    pending = agent.loop.pending_review
    if pending is None:  # 调用方保证有挂起，这里只是防御
        raise RuntimeError("当前没有挂起的人审请求")
    stage = pending.get("stage") or "未标注环节"
    console.print()
    console.print(
        Panel(
            Text(str(pending.get("question", "")), style="bold"),
            title=f"[yellow]人审 · {escape(str(stage))}[/]",
            border_style="yellow",
        )
    )
    openable: list[Path] = []
    for aid in pending.get("assets", []):
        try:
            asset = agent.assets.get(aid)
        except KeyError:
            console.print(f"  [red]候选 {escape(str(aid))} 已不存在[/]")
            continue
        body, local = _candidate_view(agent, asset)
        if local is not None:
            openable.append(local)
        console.print(Panel(Text(body), title=escape(asset.brief()), border_style="blue"))
    console.print(
        "[bold]决策[/]  [green]a[/]=采纳（可带补充：a 控制在60集）  "
        "[yellow]r[/]=打回重写  [red]j[/]=方向不对退回  [dim]理由可直接跟在后面[/]"
        + ("  [cyan]o[/]=打开候选文件" if openable else "")
        + "  [dim]/stop 先不定[/]"
    )
    async def _resume(decision: str, reason: str = "", decided_by: str = "human") -> LoopResult:
        with board.running():
            return await agent.loop.resume_turn(decision, reason=reason, decided_by=decided_by)

    while True:
        try:
            raw = await hub.ask("> ")
        except EOFError:
            return None  # Ctrl+D：先不定，挂起的人审留着，下一条输入会再问
        low = raw.strip().lower()
        if low in {"o", "open", "打开"} and openable:
            for p in openable:
                _open_file(p)
            continue
        if low in {"/stop", "/pause", "/停", "/exit", "/quit"}:
            console.print("[dim]先不定：这条人审挂着，下一条输入会先问它[/]")
            return None
        if raw.strip().lower() in {"/auto on", "/auto 开"} and agent.second_window:
            console.print(f"[yellow]{_SECOND_WINDOW_AUTO}[/]")
            continue
        if raw.strip().lower() in {"/auto on", "/auto 开"}:
            # 在决策口开 /auto：后续小节点人审不再逐条问
            agent.loop.auto_review = True
            agent.pipeline.enabled = True
            agent.pipeline.on_enable()
            if agent.loop._is_major_review(pending):  # noqa: SLF001
                # 2026-09-23 审查：之前这里不分大小节点一律按采纳结案 —— 挂着的若是「视频模型
                # 切换」，锁就被换了，而屏幕上说的是「大节点仍会问你」。大节点这条照样要人定。
                console.print(
                    "[green]自动模式已开[/]（之后的小节点自动采纳）。"
                    "[yellow]但眼前这条是大节点，仍需你来定：[/]"
                )
                continue
            console.print(
                "[green]自动模式已开[/]：本次按采纳结案，后续小节点人审自动采纳，"
                "大节点（剧本/视频生成/图片生成/换模型）仍会停下来问你一次。"
            )
            return await _watched(hub, _resume("adopt", decided_by="auto"))
        if low in {"/auto stop", "/auto 停"}:
            # 人审挂着时流水线照样在后台跑：硬停不用先答完这一条
            n = agent.pipeline.stop()
            agent.loop.auto_review = False
            console.print(
                f"[yellow]■ 流水线已停[/][dim]（取消了 {n} 个在跑的环节）。"
                "眼前这条人审仍要你定：[/]"
            )
            continue
        if low in {"/auto go", "/auto 放行"}:
            what = agent.pipeline.pass_gate()
            console.print(
                f"[green]已放行[/]：{escape(what)}[dim]。眼前这条人审仍要你定：[/]" if what
                else "[dim]流水线现在没有停在停点上。眼前这条人审仍要你定：[/]"
            )
            continue
        parsed = _parse_decision(raw)
        if parsed is None:
            console.print("[dim]请输入 a / r / j（理由可直接跟在后面）[/]")
            continue
        decision, reason = parsed
        while decision != "adopt" and not reason:
            reason = (await hub.ask("[yellow]打回理由（必填）[/] ")).strip()
            if reason.startswith("/"):
                # 「/stop」之类是命令不是理由（之前 Ctrl+C 产生的 /stop 被当打回理由存进了记忆）
                console.print("[red]这是命令不是理由；请写一句为什么打回。[/]")
                reason = ""
                continue
            again = _parse_decision(reason)
            if again is not None:
                # 追问理由时又把决定字母敲了一遍（「r 不允许换」）：只留后面的理由 ——
                # 之前原样当理由存，记忆库里是「r 不允许换」（2026-09-26）
                reason = again[1]
            if not reason:
                console.print("[red]理由不能为空。不记原因的话下一版会犯一模一样的错。[/]")
        return await _watched(hub, _resume(decision, reason=reason))


_SECOND_WINDOW_AUTO = (
    "这个窗口是同一个文件夹的第二个窗口，/auto 只能在主窗口开 —— 两个窗口的流水线会重复派发渲染、"
    "重复花钱。要在这里开，先关掉另一个窗口再重开这个"
)

# 网络断了在同一轮里自动重试几次、每次等多久（第 N 次等 N×_NET_WAIT 秒）
_NET_RETRIES = 3
_NET_WAIT = 20


async def _retry_after(
    wait: int, agent: Agent, board: ProgressBoard, turn: Any
) -> LoopResult:
    """等一会儿再在同一轮里接着跑。等待期间不开进度窗，/stop 能取消。"""
    await asyncio.sleep(wait)
    console.print("[dim]↻ 重新连接，接着跑这一轮…[/]")
    with board.running():
        return await agent.loop.continue_turn(turn)


async def _run_tail(
    agent: Agent, board: ProgressBoard, hub: InputHub, result: LoopResult | None
) -> LoopResult | None:
    """一轮跑完后的收尾循环，直到真正停下；返回最后一次结果（被人停掉返回 None）。

    · 模型请人审 → 当场收决策结案（auto 模式不挂起，这里空转）
    · auto 模式撞单轮迭代上限 → 同一轮自动续跑（/stop 或 Ctrl+C 可停）
    人审没结案不能开新轮（上下文缺 tool 响应，API 会 400）。
    """

    async def _continue(turn: Any) -> LoopResult:
        with board.running():
            return await agent.loop.continue_turn(turn)

    retried = 0
    while result is not None:
        while agent.loop.pending_review is not None:
            try:
                result = await _decide_review(agent, board, hub)
            except KeyboardInterrupt:
                console.print("[yellow]已中断 —— 人审还没结案，下一条输入会先补这个决策[/]")
                return result
            if result is None:
                return None
            _print_result(result)
        if agent.loop.pending_review is not None:
            break
        # 网络断了：在**同一轮**里等着接着跑 —— 本轮做过的活（已写的集、已生成的图）不重来。
        # /stop 随时叫停；等的时候进度窗关着，能看清倒计时。
        # 断网多等几次；超时只再试一次 —— 超时多半是上下文太大，Loop 已经压缩后重试过，
        # 再原样重发只会整包再超一次（2026-09-23 审查：最坏整包重发 24 次）
        tries = _NET_RETRIES if result.error_kind == "connection" else 1
        if result.resumable and retried < tries:
            retried += 1
            wait = _NET_WAIT * retried
            why = "连不上模型服务" if result.error_kind == "connection" else "模型响应超时"
            console.print(
                f"[yellow]⚠ {why}[/][dim]（{result.iterations} 次迭代的进度保留着）"
                f"　{wait}s 后在同一轮里自动重试 {retried}/{_NET_RETRIES}，/stop 叫停[/]"
            )
            try:
                result = await _watched(hub, _retry_after(wait, agent, board, result.turn))
            except KeyboardInterrupt:
                console.print("[yellow]已中断[/]")
                break
            if result is None:
                return None
            _print_result(result)
            continue
        net_dead = result.stop_reason is StopReason.ERROR and result.error_kind in (
            "connection",
            "timeout",
        )
        if net_dead:
            console.print(
                "[red]网络一直不通，先停在这里。[/][dim]修好网络/代理后打 /retry，"
                "会在同一轮接着跑，本轮进度不丢（agent doctor 可查代理连通性）[/]"
            )
            break
        if agent.loop.auto_review and result.stop_reason is StopReason.MAX_ITERATIONS:
            console.print("[dim]↻ 单轮迭代上限已到，auto 同一轮自动续跑（/stop 可停）[/]")
            try:
                result = await _watched(hub, _continue(result.turn))
            except KeyboardInterrupt:
                console.print("[yellow]已中断[/]")
                break
            if result is not None:
                _print_result(result)
            continue
        break
    return result


# ---------------------------------------------------------------- 命令


@app.command()
def doctor(
    cache: bool = typer.Option(
        False, "--cache", help="实测前缀缓存是否命中：同一长前缀连发两次小请求，看 cached_tokens"
    ),
) -> None:
    """自检：配置能不能读、密钥有没有、工具与 MCP 有没有注册上、能不能连通。"""
    asyncio.run(_doctor(cache))


async def _doctor(cache: bool = False) -> None:
    quiet_shutdown_noise()
    console.print("[bold]配置与连通性自检[/]\n")
    loaded = load_dotenv()
    console.print(f"[green]✓[/] .env 加载 {len(loaded)} 个变量：{', '.join(loaded) or '（无）'}")

    try:
        config = ModelsConfig.load(PROJECT_ROOT / "config" / "models.yaml")
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]✗[/] 读配置失败：{e}")
        raise typer.Exit(1) from e
    console.print("[green]✓[/] config/models.yaml 解析成功")

    t = Table(show_header=True, header_style="bold")
    t.add_column("provider")
    t.add_column("model")
    t.add_column("密钥")
    t.add_column("窗口")
    t.add_column("定价")
    for key, p in config.text.providers.items():
        t.add_row(
            key,
            p.model,
            "[green]已解析[/]" if p.api_key_resolved else "[red]缺失[/]",
            f"{p.context_window:,}",
            "[green]已配[/]" if p.pricing.known else "[yellow]未配[/]",
        )
    console.print(t)
    console.print(f"角色映射：{len(config.text.roles)} 个 → {', '.join(sorted(config.text.roles))}")

    if config.cache.mode is None:
        console.print(
            "[yellow]⚠[/] cache.mode 未确认。这决定批量驱逐(evict_at=15)能否拿到预期收益。"
            "跑 [bold]agent doctor --cache[/] 实测后回填 config/models.yaml。"
        )
    else:
        console.print(f"[green]✓[/] cache.mode = {config.cache.mode}")

    cg = config.cost_guard

    def _money(v: float | None) -> str:
        return f"¥{v:.2f}" if v is not None else "[yellow]未设[/]"

    console.print(
        f"[green]✓[/] Cost Guard：单次任务 {_money(cg.per_task_limit)} · "
        f"单项目 {_money(cg.per_project_limit)} · 单日 {_money(cg.daily_limit)} · 次数上限 "
        f"{', '.join(f'{k} {v}' for k, v in sorted(cg.call_limits.items())) or '默认'} · "
        f"超限 {cg.on_exceed}"
    )

    agent = Agent.create()
    try:
        if not agent.media_priced:
            # 之前这里照样打绿勾（2026-09-23 审查）—— 可媒体没单价时金额上限根本看不见视频
            console.print(
                "[yellow]⚠[/] 媒体模型都没配单价（config/media_models.yaml 的 price / "
                "price_per_second）：金额上限只管文本，视频靠「段数 / 秒数」上限管"
            )
        await agent.setup()
        tools = agent.registry.catalog()
        console.print(f"[green]✓[/] 工具注册表：{len(tools)} 个")
        for m in tools:
            console.print(f"    [dim]{m.permission.value:12}[/] {m.name} — {m.summary}")

        # ---- MCP ----
        console.print("\n[bold]MCP server[/]")
        if agent.mcp is not None and agent.mcp.providers:
            for s in await agent.mcp.status():
                mark = "[green]✓[/]" if s["ok"] else "[red]✗[/]"
                console.print(
                    f"{mark} {s['alias']}（{s['transport']}）{s['detail']} · "
                    f"L-external {s['externals']} 个"
                )
            ext = agent.mcp.audit_permissions()
            if ext:
                console.print(
                    f"    [dim]仍是 L-external 的外部工具：{', '.join(ext)} —— "
                    "确认只读后可在 config/mcp_servers.yaml 里降级[/]"
                )
        else:
            console.print("[dim]○ config/mcp_servers.yaml 里没有启用的 server[/]")

        # ---- 连通性：每个 provider 只和它自己的端点比 ----
        console.print("\n[bold]连通性测试[/]")
        main_provider = config.text.roles.get("main_agent", "")
        main_failed = False
        for key, p in config.text.providers.items():
            try:
                ids = await agent.gateway.list_models_for(key)
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]✗[/] {key}：{type(e).__name__}: {e}")
                main_failed = main_failed or key == main_provider
                continue
            if p.model in ids:
                console.print(f"[green]✓[/] {key}：API 可达，model {p.model!r} 在服务端可用")
            else:
                shown = ", ".join(ids[:30]) + (f" …共 {len(ids)} 个" if len(ids) > 30 else "")
                console.print(
                    f"[red]✗[/] {key}：model {p.model!r} 不在服务端列表里。可用的有：\n"
                    f"    {shown}\n"
                    f"    → 改 config/models.yaml 的 text.providers.{key}.model"
                )
                main_failed = main_failed or key == main_provider

        # ---- 前缀缓存实测 ----
        if cache:
            console.print("\n[bold]前缀缓存探测[/]（同一长前缀连发两次，看第二次 cached_tokens）")
            try:
                r = await agent.gateway.probe_cache()
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]✗[/] 探测失败：{type(e).__name__}: {e}")
            else:
                console.print(
                    f"    模型 {r['model']} · 前缀 {r['prompt_tokens']:,} tokens · "
                    f"命中 第一次 {r['cached_first']:,} / 第二次 {r['cached_second']:,}"
                    f"（{r['hit_ratio']:.0%}）"
                )
                if r["mode"] == "auto_prefix":
                    console.print(
                        "[green]✓[/] 自动前缀缓存生效 → config/models.yaml 填 "
                        "[bold]cache.mode: auto_prefix[/]；批量驱逐的收益成立"
                    )
                else:
                    console.print(
                        "[yellow]○[/] 两次都没命中 → 填 [bold]cache.mode: none[/]"
                        "（或该端点不回传 cached_tokens；批量驱逐仍无害，只是省不到钱）"
                    )

        if main_failed:
            raise typer.Exit(1)
    finally:
        await agent.aclose()


@app.command()
def models(role: str = typer.Option("main_agent", help="用哪个角色的凭据去查")) -> None:
    """列出服务端实际可用的模型 id。"""
    asyncio.run(_models(role))


async def _models(role: str) -> None:
    quiet_shutdown_noise()
    agent = Agent.create()
    try:
        ids = await agent.gateway.list_models(role)
    finally:
        await agent.aclose()
    console.print(f"[bold]服务端可用模型（{len(ids)}）[/]")
    for i in ids:
        console.print(f"  {i}")


@app.command()
def chat(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="显示模型请求与上下文装配细节"),
    role: str = typer.Option("main_agent", help="使用的角色"),
    session: str = typer.Option(
        "", "--session",
        help="会话名。不填 = 按当前文件夹（一个文件夹一个项目 / 会话，记忆和项目预算都跟着它）",
    ),
) -> None:
    """交互式对话。输入 /exit 退出，/stat 看窗口与成本。"""
    asyncio.run(_chat(verbose, role, session))


async def _chat(verbose: bool, role: str, session: str = "") -> None:
    quiet_shutdown_noise()
    board = ProgressBoard(console)
    # 输入枢纽：stdin 由一根线程常驻读，一轮在跑时打字会排队，/stop 停、/now 插队。
    # 2026-09-22 用户要的：底部钉一个不消失的输入框，敲 / 弹命令菜单、模糊搜索。
    # 终端不支持或 stdin 被重定向时 make_reader 返回 None，退回原来的 readline。
    reader = make_reader()
    hub = InputHub(console, reader=reader)
    if reader is not None and hasattr(reader, "start"):
        reader.start()
    hub.start()

    async def _asker(meta: ToolMeta, args: dict[str, Any]) -> bool:
        # 超出开工时确认的额度：一律停下来问人，/auto 也问（2026-09-23 审查：之前 /auto 下
        # 次数超限自动放行，视频又没有单价 —— 视频花费等于没有刹车）。额度是人开工时
        # 点过头的，超出它就得人再点一次头。
        # 进度窗开着的时候先暂停再提问 —— 否则输入提示和刷新区域叠在一起
        with board.paused():
            return await _ask_permission(meta, args, hub)

    agent = Agent.create(asker=_asker, role=role, session_id=session)

    async def _budget_asker(reason: str) -> float | None:
        # 金额护栏：问一次追加多少，人点头就抬上限继续。auto 模式下也问 —— 钱是最后一道闸。
        with board.paused():
            console.print()
            console.print(
                Panel(
                    f"{reason}\n\n[dim]回车 = 停下来；输入数字 = 临时追加该金额（元）后继续；"
                    "y = 追加 50 元。追加只在本次会话生效，不改配置。[/]",
                    title="[yellow]金额护栏[/]",
                    border_style="yellow",
                )
            )
            raw = (await hub.ask("追加多少？> ")).strip().lower()
        if raw in {"y", "yes"}:
            return 50.0
        try:
            return float(raw) if raw else None
        except ValueError:
            return None

    agent.loop.budget_asker = _budget_asker

    def _on_stop(how: str) -> None:
        # 复盘时分得清是人停的（之前 /stop 不发任何事件）
        agent.bus.emit_soon(EventType.USER_STOP, how=how)

    hub.on_stop = _on_stop
    agent.bus.subscribe(make_renderer(verbose))
    agent.bus.subscribe(board.on_event)
    # 按集流水管线的派发/完成提示（/auto on 时才开始派发，提示随时可见）
    agent.pipeline.note = lambda m: console.print(f"[dim]{escape(str(m))}[/]")
    # 两个停点（参考图看脸、第 1 集看片）：有人在控制台前才开 —— 停下来等的是 /auto go
    agent.pipeline.gates = True
    await agent.setup()
    restored = await agent.restore_session()

    # 产物目录（2026-09-22 用户定）：**默认就是他打开 Agent 的这个文件夹**，不再问一次 ——
    # 内容都在本地跑，生成的东西该落在他眼前的目录里。产物目录 = 项目（缺口 A）。
    # 开工前先把这个文件夹清点一遍报给他：有什么、没什么。
    await _settle_output_dir(agent, hub, explicit_session=bool(session))
    if agent.second_window:
        console.print(
            Panel(
                Text(
                    "这个文件夹已经有一个 Agent 窗口在用（它的对话、挂起的人审、模型锁归它）。\n"
                    f"这个窗口另起了会话「{agent.session_store.name}」：项目设置从主窗口抄了一份，"
                    "在这里改只对这个窗口生效；/auto 只能在主窗口开"
                    "（两个窗口的流水线会重复派发渲染）。"
                ),
                title="第二个窗口",
                border_style="yellow",
            )
        )
    _warn_store(agent)
    if getattr(agent, "local_materials", None) is not None:
        try:
            note = await asyncio.to_thread(lambda: agent.local_materials.index().inventory())
            console.print(f"[dim]{note}[/]")
        except Exception as e:  # noqa: BLE001
            console.print(f"[dim]产物目录清点失败（{type(e).__name__}），不影响使用[/]")

    # 产线标签（2026-09-23 用户要的）：让用户自己选这次做哪类内容。老 session 沿用记住的；
    # 回车 = 不限定（模型按系统提示词自己判断）；之后 /type 随时改。
    if agent.session_store is not None and not agent.content_line:
        console.print(
            Panel(lines_menu(), title="这次做哪类内容？（回车 = 不限定，之后 /type 可改）",
                  border_style="cyan")
        )
        pick = (await hub.ask("[bold cyan]选一个（1/2/3/4）：[/] ")).strip()
        chosen = parse_line(pick)
        if chosen is not None:
            agent.set_content_line(chosen.key)
            console.print(f"[green]产线：{chosen.label}[/]\n")
        elif not is_off(pick):
            console.print("[dim]没认出来，先不限定；/type 随时选[/]\n")

    # 开工额度（用户规则：每次开工前确认上限 —— 金额 / 次数 / 视频秒数）
    await _confirm_budget(agent, hub)

    provider, _ = agent.config.text.resolve(role)
    console.print(
        Panel(
            f"模型 [bold]{provider.model}[/]  角色 [bold]{role}[/]\n"
            f"工具 {len(agent.registry.catalog())} 个 · "
            f"窗口 {agent.memory.policy.window_turns} 轮"
            f"（涨到 {agent.memory.policy.evict_at} 轮批量驱逐）\n"
            f"skill {len(agent.skills.available)} 篇（改 skills/*.md 存盘即生效）\n"
            f"产线 [bold]{_line_label(agent)}[/]（短剧 / 抖音短视频 / 广告 / 设计，/type 可改）\n"
            f"画幅 [bold]{aspect_label(agent.aspect_ratio)}[/]"
            "（/ratio 可改：16:9 横屏 / 9:16 竖屏 / 1:1 方形）\n"
            f"产物目录 [bold]{agent.output_prefs.root}[/]"
            "（当前文件夹；生成的东西默认都落这里，/out 可改）\n"
            f"项目 [bold]{escape(agent.project_title)}[/] [dim]{agent.project}[/]"
            "（资产、记忆、单项目预算都按它分开；一个文件夹一个项目）\n"
            f"视频模型 [bold]{getattr(agent.media_fns, 'video_lock', '') or '未锁定'}[/] · "
            f"生图模型 [bold]{getattr(agent.media_fns, 'image_lock', '') or '未锁定'}[/]"
            "（换模型会先问你：对话里说要换哪个，确认 a 才生效）\n"
            f"[dim]/exit 退出   /stat 状态   /tools 工具   /trace 痕迹   /mermaid 流程图\n"
            f"/rollback <资产id> 回退到某版本重来   /budget 各级用量与额度\n"
            f"/budget set 金额 300 视频秒 900 视频 80 图 100 改本次开工的额度"
            f"   /budget allow 50 临时追加 50 元\n"
            f"/budget reset 新开一段工（清零本次开工的用量；单日 / 单项目累计在台账里，不清）\n"
            f"/rename 把已生成的图/视频改成带序号的可读文件名（/rename dry 只看计划）\n"
            f"/brief 记忆简报   /skills 已加载的 skill 与能力区占用\n"
            f"模型请人审时会暂停等你决策：a 采纳（可带补充）· r 打回（必填理由）· j 退回\n"
            f"/auto on 后小节点人审自动采纳、大节点（剧本/视频生成/图片生成）仍问你一次，"
            f"渲图 / 渲一集之前整批报价问你一次；\n"
            f"按集流水在参考图渲完、第 1 集渲完各停一次，/auto go 放行；/auto stop 硬停"
            f"（在跑的渲染当场取消）\n"
            f"挂起/撞线的活当场接着跑不用再发话触发（长剧批量出稿用），Ctrl+C 随时叫停\n"
            f"生成过程中照样能打字：会排队、本轮结束后自动发送；/stop 立即停止当前一轮，\n"
            f"/now <消息> 停止当前并立即发送这条；/queue 看排队、/queue clear 清空\n"
            f"断网/超时会在同一轮里自动等着重试，仍失败后 /retry 接着跑（本轮进度不丢）[/]",
            title="AIGC Agent",
            border_style="cyan",
        )
    )
    if restored:
        console.print(
            f"[green]↺ 已恢复这个文件夹上次的 {restored} 轮对话[/]"
            "[dim]（会话按文件夹分：在别的文件夹启动就是另一个项目）[/]"
        )
    if agent.loop.pending_review is not None:
        # 上次退出前有一条人审没结案 —— 之前重启后被悄悄补成「已中断」，这里接着问
        console.print("[yellow]上次退出前有一条人审还没结案，先把它定下来：[/]")
        try:
            pending_result = await _decide_review(agent, board, hub)
            if pending_result is not None:
                _print_result(pending_result)
        except KeyboardInterrupt:
            console.print("[yellow]已中断 —— 下一条输入会先补这个决策[/]")

    # 上一轮怎么停下的。/auto on 要看它决定要不要把停下的活当场接着跑。
    last_result: LoopResult | None = None
    loop_errors = 0  # 连续出错次数（读到一条输入就清零）
    try:
        while True:
            try:
                try:
                    text = await hub.next_message("\n[bold cyan]你[/] ")
                except (EOFError, KeyboardInterrupt):
                    break

                text = text.strip()
                loop_errors = 0
                if not text:
                    continue
                if text in {"/exit", "/quit"}:
                    break
                if text.lower() in {"/help", "/?", "/帮助"}:
                    console.print(Panel(Text(help_text()), title="命令", border_style="cyan"))
                    continue
                if text.startswith("/") and not known_command(text):
                    # 拼错的命令不当聊天发给模型（之前 /xxx 原样发出去，还会被当成要求去执行）
                    near = [c for c, _ in search(text.split(" ", 1)[0])][:4]
                    console.print(
                        f"[yellow]没有这个命令：{escape(text.split(' ', 1)[0])}[/]"
                        + (f"[dim]　是不是：{' '.join(near)}[/]" if near else "")
                        + "[dim]　/help 看全部命令[/]"
                    )
                    continue
                if text.lower().startswith(("/now ", "/插队 ")):
                    # 空闲时没有队列可插：把 /now 去掉当普通一句话发（之前原样连 /now 发给模型）
                    text = text.split(" ", 1)[1].strip()
                    if not text:
                        continue
                if text.lower() in {"/stop", "/pause", "/停", "/暂停", "/停止"}:
                    console.print("[dim]现在没有在跑的任务[/]")
                    continue
                if text.lower() in {"/retry", "/重试", "/继续跑"}:
                    # 在**同一轮**里接着跑：网络中断、撞迭代上限之后用它，本轮进度不用重来
                    if agent.loop.pending_review is not None:
                        console.print("[yellow]上一条人审还没结案，先对它做决定。[/]")
                        continue
                    if last_result is None:
                        console.print("[dim]没有可接着跑的轮次[/]")
                        continue
                    if not (
                        last_result.resumable
                        or last_result.stop_reason
                        in (StopReason.MAX_ITERATIONS, StopReason.BUDGET_EXCEEDED)
                    ):
                        # 正常结束的、被内容过滤拦下的，接着跑只会让模型把同样的话再说一遍
                        console.print(
                            f"[dim]上一轮是「{last_result.stop_reason.value}」停的，没有可以接着跑的"
                            " —— 直接说下一句话[/]"
                        )
                        continue
                    console.print("[dim]↻ 在同一轮接着跑…[/]")

                    async def _resume_same(turn: Any = last_result.turn) -> LoopResult:
                        with board.running():
                            return await agent.loop.continue_turn(turn)

                    try:
                        again = await _watched(hub, _resume_same())
                    except KeyboardInterrupt:
                        console.print("[yellow]已中断[/]")
                        continue
                    if again is not None:
                        _print_result(again)
                        last_result = await _run_tail(agent, board, hub, again)
                    continue
                if text.lower() == "/queue":
                    hub.print_queue()
                    continue
                if text.lower() in {"/queue clear", "/queue 清空"}:
                    hub.pending.clear()
                    console.print("[dim]排队已清空[/]")
                    continue
                if text == "/stat":
                    _print_stat(agent)
                    continue
                if text == "/trace":
                    console.print(
                        Panel(agent.trace.to_text(), title="执行痕迹", border_style="magenta")
                    )
                    console.print(f"[dim]{agent.trace.summary()}[/]")
                    continue
                if text == "/mermaid":
                    console.print(
                        Panel(
                            agent.trace.to_mermaid(),
                            title="自动生成的流程图",
                            border_style="magenta",
                        )
                    )
                    continue
                if text == "/tools":
                    for m in agent.registry.catalog():
                        console.print(f"  [dim]{m.permission.value:12}[/] {m.name} — {m.summary}")
                    continue
                if text in ("/brief full", "/brief 整理"):
                    # 完整模式：规则版简报再交给子代理合并重复、标出矛盾（花一次模型调用）
                    if agent.mem_agent is None:
                        console.print("[dim]没有装配记忆代理[/]")
                        continue
                    with board.running():
                        b = await agent.mem_agent.consolidate()
                    if b.empty:
                        console.print("[dim]简报为空[/]")
                    else:
                        console.print(Panel(Text(b.render()), title="Memory Brief（整理后）",
                                            border_style="magenta"))
                    continue
                if text == "/brief":
                    b = agent.mem_agent.brief() if agent.mem_agent else None
                    if b is None or b.empty:
                        console.print("[dim]简报为空：没有约束、打回记录或偏好[/]")
                    else:
                        console.print(Panel(Text(b.render()), title="Memory Brief",
                                            border_style="magenta"))
                    continue
                if text in ("/remember", "/记住") or text.startswith(("/remember ", "/记住 ")):
                    # 人亲口定的规则：这个项目每轮都带上（2026-09-26）。之前对话里说的规则只被
                    # 记忆提取当成「推测」，从不 pin；打回理由反而永久 pin
                    await _remember(agent, text.split(" ", 1)[1].strip() if " " in text else "")
                    continue
                if text in ("/forget", "/忘掉") or text.startswith(("/forget ", "/忘掉 ")):
                    await _forget(agent, hub, text.split(" ", 1)[1].strip() if " " in text else "")
                    continue
                if text == "/skills":
                    if agent.allocator is None:
                        console.print("[dim]没有装配能力预算[/]")
                    else:
                        active = ", ".join(agent.allocator.active) or "无"
                        console.print(f"[dim]已加载：{active}[/]")
                        console.print(f"[dim]{agent.allocator.brief()}[/]")
                    continue
                if text == "/budget" or text.startswith("/budget "):
                    arg = text[len("/budget"):].strip()
                    if agent.guard is None:
                        console.print("[dim]没有装配 Cost Guard[/]")
                    elif arg == "reset":
                        agent.guard.reset()
                        agent.pipeline.kick()  # 预算恢复 → 流水线渲染刹车自动解除
                        console.print(
                            "[dim]本次开工的用量已清零（额度不变）；"
                            "台账里的单日 / 单项目累计不清 —— 撞的是那两级就用 /budget allow[/]"
                        )
                    elif arg.startswith("set"):
                        changes = parse_budget(arg[len("set"):])
                        if not changes:
                            console.print(
                                "[yellow]用法：/budget set 金额 300 视频秒 900 视频 80 图 100"
                                "（写哪项改哪项）[/]"
                            )
                            continue
                        agent.apply_budget({**agent.budget_defaults(), **changes})
                        agent.pipeline.kick()
                        console.print(f"[green]额度已改[/] [dim]{agent.guard.brief()}[/]")
                    elif arg.startswith("allow"):
                        amount = arg[len("allow"):].strip() or "50"
                        more = parse_budget(amount) if not _is_number(amount) else {}
                        if more:
                            # 「/budget allow 视频 20 视频秒 300 图 10」：本次开工和单日口径一起抬
                            # （单日次数上限之前没有任何出口，撞上就永久暂停，2026-09-24 审查）
                            parts: list[str] = []
                            if "video_calls" in more or "video_seconds" in more:
                                agent.guard.allow_more(
                                    "video", n=int(more.get("video_calls", 0)),
                                    seconds=float(more.get("video_seconds", 0.0)),
                                )
                                if more.get("video_calls"):
                                    parts.append(f"视频 {int(more['video_calls'])} 段")
                                if more.get("video_seconds"):
                                    parts.append(f"视频 {int(more['video_seconds'])} 秒")
                            if "image_calls" in more:
                                agent.guard.allow_more("image", n=int(more["image_calls"]))
                                parts.append(f"图片 {int(more['image_calls'])} 张")
                            if "money" in more:
                                agent.guard.allow_more_money(float(more["money"]))
                                parts.append(f"金额 ¥{float(more['money']):.0f}")
                            agent.pipeline.kick()
                            console.print(
                                f"[green]已临时追加：{'、'.join(parts)}[/]"
                                "[dim]（本次开工与单日口径一起抬，配置不变）[/]"
                            )
                            continue
                        try:
                            total = agent.guard.allow_more_money(float(amount))
                        except ValueError:
                            console.print(
                                "[yellow]用法：/budget allow 50（追加金额，元）或 "
                                "/budget allow 视频 20 视频秒 300 图 10[/]"
                            )
                            continue
                        agent.pipeline.kick()
                        console.print(
                            f"[green]已临时追加 ¥{float(amount):.0f}[/]"
                            f"[dim]（本次会话累计追加 ¥{total:.0f}，配置不变）[/]"
                        )
                    else:
                        console.print(f"[dim]{agent.guard.brief()}[/]")
                    continue
                if text == "/auto" or text.startswith("/auto "):
                    arg = text[len("/auto"):].strip().lower()
                    if arg in {"on", "开"} and agent.second_window:
                        console.print(f"[yellow]{_SECOND_WINDOW_AUTO}[/]")
                        continue
                    if arg in {"on", "开"}:
                        agent.loop.auto_review = True
                        agent.pipeline.enabled = True
                        agent.pipeline.on_enable()  # 扫存量剧本按批补派，on 之前写的不漏
                    elif arg in {"off", "关"}:
                        agent.loop.auto_review = False
                        agent.pipeline.enabled = False  # 在跑的跑完，不再派新任务
                        if agent.pipeline.busy:
                            console.print(
                                "[dim]流水线上在跑的环节会跑完；要当场停掉用 /auto stop[/]"
                            )
                    elif arg in {"stop", "停"}:
                        # 硬停（2026-09-26 用户定的）：/auto off 让在跑的跑完，两集并行时还有
                        # 几十段照渲照付
                        agent.loop.auto_review = False
                        n = agent.pipeline.stop()
                        console.print(
                            f"[yellow]■ 流水线已停[/]：取消了 {n} 个在跑的环节，不再派新活。"
                            "[dim]已经提交的渲染任务留在台账上，"
                            "下次 /auto on 重渲这一集时按段取回，"
                            "不重付。[/]"
                            if n else "[yellow]■ 流水线已停[/][dim]（没有在跑的环节）[/]"
                        )
                        continue
                    elif arg in {"go", "放行", "继续"}:
                        what = agent.pipeline.pass_gate()
                        if not what:
                            console.print("[dim]流水线现在没有停在停点上[/]")
                        elif not agent.pipeline.enabled:
                            console.print(
                                f"[green]已放行[/]（{escape(what)}）"
                                "[yellow]/auto 没开，开了才派[/]"
                            )
                        else:
                            console.print(f"[green]已放行[/]：{escape(what)}")
                        continue
                    elif arg in {"retry", "重试"}:
                        # 失败的环节之前只在重启后才会重派（2026-09-23 审查）
                        keys = agent.pipeline.retry()
                        if not keys:
                            console.print("[dim]流水线没有失败的环节[/]")
                        else:
                            names = escape("、".join(keys))
                            off = "" if agent.pipeline.enabled else "（/auto 没开，开了才派）"
                            console.print(
                                f"[green]重派 {len(keys)} 个失败的环节[/]：{names}  "
                                f"[yellow]{off}[/]"
                            )
                        continue
                    elif arg:
                        console.print(
                            "[yellow]用法：/auto on 开 · /auto off 关（在跑的跑完）· "
                            "/auto stop 硬停（在跑的当场取消）· /auto go 放行停点 · "
                            "/auto retry 重派失败的环节[/]"
                        )
                        continue
                    if not agent.loop.auto_review:
                        console.print("[dim]自动模式已关：每条人审都会停下来问你，流水线暂停派发。[/]")
                        continue
                    console.print(
                        "[green]自动模式已开[/]：小节点人审自动采纳不再逐条问你，"
                        "每条决策都会打印出来；大节点（剧本/视频生成/图片生成）"
                        "仍会停下来等你拍板一次。\n"
                        "[dim]按集流水已开：每落一集剧本就拆这一集的分镜；全剧齐后拆资产库、"
                        "渲参考图，再逐集出提示词、渲染。渲参考图、渲每一集之前整批报价问你一次。\n"
                        "两个停点：参考图渲完停一次（看脸，可以说「定音」给角色定音色）、"
                        "第 1 集单独渲、"
                        "渲完停一次（看片）；都用 /auto go 放行，之后其余各集自动并行渲。\n"
                        "随时 Ctrl+C 打断，/auto off 回到逐条确认（在跑的跑完），"
                        "/auto stop 当场硬停。"
                        "注意：L-external 不可逆操作与预算超限也仍会问你。[/]"
                    )
                    if agent.pipeline.holding:
                        console.print(f"[yellow]{escape(agent.pipeline.holding)}[/]")
                    # 开 auto 这一刻就该自己跑起来，不用再发一句话触发：
                    # · 有挂着的人审 → 按采纳结案续跑
                    # · 上一轮撞单轮迭代上限停下 → 同一轮接着跑
                    # （预算超限仍由人拍板，不自动续 —— 钱是最后一道闸）
                    resumed: LoopResult | None = None

                    async def _auto_resume() -> LoopResult:
                        with board.running():
                            return await agent.loop.resume_turn("adopt", decided_by="auto")

                    async def _auto_continue(turn: Any) -> LoopResult:
                        with board.running():
                            return await agent.loop.continue_turn(turn)

                    try:
                        pending = agent.loop.pending_review
                        if pending is not None and agent.loop._is_major_review(pending):  # noqa: SLF001
                            # 大节点（换模型、剧本、视频生成）开了 /auto 也要人定 ——
                            # 之前这里一律按采纳结案，挂着的「换视频模型」就这么被换了
                            # （2026-09-23 审查）
                            console.print(
                                "[yellow]挂着的人审是大节点，仍需你来定：发任意一句话会先问它[/]"
                            )
                        elif pending is not None:
                            console.print("[dim]↻ 有挂起的人审，按采纳结案继续跑[/]")
                            resumed = await _watched(hub, _auto_resume())
                        elif (
                            last_result is not None
                            and last_result.stop_reason is StopReason.MAX_ITERATIONS
                        ):
                            console.print("[dim]↻ 上一轮撞迭代上限停下，同一轮自动续跑[/]")
                            resumed = await _watched(hub, _auto_continue(last_result.turn))
                    except KeyboardInterrupt:
                        console.print("[yellow]已中断[/]")
                        continue
                    if resumed is not None:
                        _print_result(resumed)
                        last_result = await _run_tail(agent, board, hub, resumed)
                    continue
                if text == "/rename" or text.startswith("/rename "):
                    # 已生成的图/视频改成带序号的可读文件名；/rename dry 只看计划
                    arg = text[len("/rename"):].strip().lower()
                    root = agent.output_prefs.root
                    plan = await asyncio.to_thread(plan_renames, agent.assets, root)
                    if not plan:
                        console.print(f"[dim]{root} 里没有需要改名的文件[/]")
                        continue
                    for r in plan:
                        console.print(
                            f"  [dim]{r.old.name}[/] → [bold]{r.new.name}[/]  "
                            f"[dim]{r.why} · {r.asset_id}[/]"
                        )
                    if arg in {"dry", "预览", "--dry-run"}:
                        console.print(f"[dim]预览 {len(plan)} 项，未改动。/rename 执行[/]")
                        continue
                    report = await asyncio.to_thread(apply_renames, agent.assets, plan)
                    manifest = await asyncio.to_thread(write_manifest, agent.assets, root)
                    console.print(
                        f"[green]已改名 {len(report.done)} 个文件[/]"
                        + (f"[dim]，清单 {manifest}[/]" if manifest else "")
                    )
                    if report.failed:
                        console.print(
                            f"[yellow]{len(report.failed)} 个没改成：[/]\n{report.render_failed()}"
                        )
                    continue
                if text in ("/type", "/line") or text.startswith(("/type ", "/line ")):
                    arg = text.split(" ", 1)[1].strip() if " " in text else ""
                    if not arg:
                        console.print(
                            f"[dim]当前产线：{_line_label(agent)}[/]\n{lines_menu()}\n"
                            "[dim]/type <序号或名字> 切换，/type off 不限定[/]"
                        )
                        continue
                    if is_off(arg):
                        agent.set_content_line("")
                        console.print("[green]产线已改为：不限定[/]")
                        continue
                    chosen = parse_line(arg)
                    if chosen is None:
                        console.print("[yellow]没认出来。可选：短剧 / 抖音短视频 / 广告 / 设计[/]")
                        continue
                    agent.set_content_line(chosen.key)
                    console.print(
                        f"[green]产线已改为：{chosen.label}[/] [dim]（本会话记住，重启沿用）[/]"
                    )
                    continue
                if text in ("/ratio", "/比例", "/画幅") or text.startswith(
                    ("/ratio ", "/比例 ", "/画幅 ")
                ):
                    arg = text.split(" ", 1)[1].strip() if " " in text else ""
                    if arg.lower() in ("reset", "默认", "off"):
                        agent.set_aspect_ratio(None)
                        console.print(f"[green]画幅已恢复默认[/] [dim]{_ratio_brief(agent)}[/]")
                        continue
                    if arg:
                        ratio = parse_aspect(arg)
                        if not ratio:
                            console.print(
                                f"[yellow]用法：/ratio 16:9（可选 {' / '.join(VIDEO_ASPECTS)}，"
                                "也可以写 横屏 / 竖屏 / 方形）· /ratio reset 恢复默认[/]"
                            )
                            continue
                        lock = getattr(agent.media_fns, "video_lock", "") or ""
                        listed = supported_aspects(agent.catalog, lock) if lock else []
                        if listed and ratio not in listed:
                            console.print(
                                f"[yellow]锁定的视频模型 {lock} 不支持 {ratio}"
                                f"（支持 {' / '.join(listed)}）。换一个比例，"
                                "或先在对话里说要换哪个视频模型[/]"
                            )
                            continue
                        agent.set_aspect_ratio(ratio)
                        console.print(
                            f"[green]这个项目的视频改成 {aspect_label(ratio)}[/] "
                            "[dim]短剧渲染、短视频出片、gen_video 没传比例时都按它"
                            "（本项目记住，重启沿用）。\n已经渲好的片段是原来的比例，"
                            "不会被拼进新比例的成片，会按新比例重新生成[/]"
                        )
                        continue
                    console.print(
                        f"[dim]{_ratio_brief(agent)}（/ratio 16:9 改，/ratio reset 恢复默认）[/]"
                    )
                    continue
                if text in ("/cut", "/镜头") or text.startswith(("/cut ", "/镜头 ")):
                    # 镜头超 3 秒拦不拦成片（2026-09-27 用户定的：先只标，第 1 集抽看过再定）
                    arg = text.split(" ", 1)[1].strip().lower() if " " in text else ""
                    if arg in ("拦", "on", "block"):
                        agent.set_cut_block(True)
                    elif arg in ("标", "off", "mark"):
                        agent.set_cut_block(False)
                    elif arg:
                        console.print(
                            "[yellow]用法：/cut 拦（超 3 秒不进成片）· /cut 标（只标出来）[/]"
                        )
                        continue
                    console.print(f"[dim]{_cut_brief(agent)}[/]")
                    continue
                if text in ("/length", "/集长") or text.startswith(("/length ", "/集长 ")):
                    arg = text.split(" ", 1)[1].strip() if " " in text else ""
                    if arg.lower() in ("reset", "默认", "off"):
                        agent.set_episode_minutes(None)
                        console.print(f"[green]集长已恢复默认[/] [dim]{_length_brief(agent)}[/]")
                        continue
                    if arg.lower() in ("auto", "自动", "跟剧本", "按剧本"):
                        agent.set_episode_minutes(None, auto=True)
                        console.print(
                            "[green]这个项目的集长改成跟剧本走[/] "
                            f"[dim]{_length_brief(agent)}"
                            "\n拆分镜时不再要求凑到某个时长，"
                            "只查台词念不念得完、单镜 ≤3 秒、开场高潮点；"
                            "视频提示词的时长跟着分镜走。已经拆好的不会自动重做[/]"
                        )
                        continue
                    if arg:
                        try:
                            minutes = float(arg.removesuffix("分钟").removesuffix("分"))
                        except ValueError:
                            minutes = 0.0
                        if not 1 <= minutes <= 30:
                            console.print(
                                "[yellow]用法：/length 8（1–30 分钟）· /length auto（跟剧本走）"
                                " · /length reset[/]"
                            )
                            continue
                        agent.set_episode_minutes(minutes)
                        console.print(
                            f"[green]这个项目一集改成 {minutes:g} 分钟[/] "
                            f"[dim]{_length_brief(agent)}"
                            "\n已经拆好的分镜 / 视频提示词不会自动重做；"
                            "要按新集长重拆哪一集，跟我说一声。"
                            "\n每集要生成的视频段数跟着变，开工额度不够时用 /budget set 调[/]"
                        )
                        continue
                    console.print(
                        f"[dim]{_length_brief(agent)}（/length 8 改，/length auto 跟剧本走，"
                        "/length reset 恢复默认）[/]"
                    )
                    continue
                if text == "/out" or text.startswith("/out "):
                    arg = text[len("/out"):].strip().strip('"')
                    if not arg:
                        console.print(f"[dim]产物目录：{agent.output_prefs.root}[/]")
                        continue
                    try:
                        p = await asyncio.to_thread(_ensure_dir, arg)
                    except OSError as e:
                        console.print(f"[red]目录不可用：{e}[/]")
                        continue
                    before = agent.session_store.name if agent.session_store else ""
                    try:
                        # 换项目 = 换会话（2026-09-26 用户定的）：对话窗口和项目设置跟着换
                        key = agent.switch_project(p, session=True)
                    except RuntimeError as e:
                        console.print(f"[yellow]{e}[/]")
                        continue
                    swapped = bool(agent.session_store) and agent.session_store.name != before
                    if swapped:
                        await agent.restore_skills()
                        last_result = None  # 上一轮是旧项目的，/retry 不能拿它在新项目里接着跑
                    console.print(
                        f"[green]产物目录已改为 {p}[/]，"
                        f"项目 [bold]{escape(agent.project_title)}[/] "
                        f"[dim]{key}（资产 / 记忆 / 单项目预算都换成这个项目的）[/]"
                    )
                    if swapped:
                        n = len(agent.memory.turns)
                        console.print(
                            "[dim]对话也换成这个项目的"
                            + (f"（接上它上次的 {n} 轮）" if n else "（这个项目还没有对话）")
                            + "，原来的对话存回原项目，/out 回去就接着聊。"
                            f"\n产线 {_line_label(agent)} · {_length_brief(agent)} · "
                            f"{_ratio_brief(agent)} · 视频模型 "
                            f"{getattr(agent.media_fns, 'video_lock', '') or '未锁定'} · 生图模型 "
                            f"{getattr(agent.media_fns, 'image_lock', '') or '未锁定'}[/]"
                        )
                        if agent.loop.pending_review is not None:
                            console.print("[yellow]这个项目有一条挂着的人审，下一条输入会先问它[/]")
                    continue
                if text == "/rollback" or text.startswith("/rollback "):
                    target = text[len("/rollback"):].strip()
                    if not target:
                        console.print(
                            "[yellow]用法：/rollback <资产id>。"
                            "先用 /trace 或让模型 list_assets 找到 id[/]"
                        )
                        continue
                    plan = await agent.rollback(target)
                    if not plan.get("ok"):
                        console.print(f"[red]{escape(str(plan.get('error')))}[/]")
                        continue
                    console.print(
                        Panel(
                            f"{plan['note']}\n\n"
                            f"[dim]作废调用 {len(plan['supersede'])} 次 · "
                            f"丢弃资产 {', '.join(plan['lost_assets']) or '无'} · "
                            f"下一句话模型会以此为起点[/]",
                            title="已回退",
                            border_style="magenta",
                        )
                    )
                    continue

                console.print()

                async def _turn(user_text: str) -> LoopResult:
                    with board.running():
                        return await agent.chat(user_text)

                try:
                    if agent.loop.pending_review is not None:
                        # 上一条人审没结案（比如上次决策时被打断）。带着挂起的
                        # tool_calls 开新轮，上下文缺 tool 响应，API 会 400。
                        console.print("[yellow]上一条人审还没结案，先对它做决定。[/]")
                        result = await _decide_review(agent, board, hub)
                    else:
                        result = await _watched(hub, _turn(text))
                    if result is None:
                        last_result = None
                        continue
                    _print_result(result)
                    last_result = await _run_tail(agent, board, hub, result)
                except KeyboardInterrupt:
                    console.print("[yellow]已中断[/]")
                    continue
                except Exception as e:  # noqa: BLE001 — 任何没接住的异常都不许把整个 chat 带走
                    # 2026-09-23 审查：之前一个 MarkupError / 渲染异常就直接退出 chat，挂着的人审
                    # 只在内存里、跟着丢了。现在报出来、存快照、回到输入
                    console.print(
                        f"[red]这一轮出错了：{escape(type(e).__name__)}: {escape(str(e)[:300])}[/]"
                        "\n[dim]对话现场已保存，可以接着说；反复出现请把这段报错发给维护者[/]"
                    )
                    try:
                        agent.session_store.save(
                            agent.memory, pending_review=agent.loop.pending_review
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    last_result = None
                    continue
            except EOFError:
                break
            except Exception as e:  # noqa: BLE001
                # 命令处理（/rollback /rename /brief full /out …）或读输入时没接住的异常：报出来、
                # 回到输入，不把整个 chat 带走（2026-09-24 复审：之前只有对话轮包了 try）。
                # 连读输入都接连出错（输入枢纽坏了）就别硬撑，免得刷屏死循环
                loop_errors += 1
                console.print(
                    f"[red]出错了：{escape(type(e).__name__)}: {escape(str(e)[:300])}[/]"
                    "\n[dim]回到输入；反复出现请把这段报错发给维护者[/]"
                )
                if loop_errors >= 5:
                    raise
                continue

        _print_stat(agent)
        console.print("[dim]再见[/]")
    finally:
        hub.close()
        if reader is not None and hasattr(reader, "stop"):
            reader.stop()
        await agent.aclose()


def _print_stat(agent: Agent) -> None:
    pin, pout = agent.bus.total_tokens()
    cached = sum(
        e.data.get("cached_tokens", 0) or 0
        for e in agent.bus.history
        if e.type is EventType.COST
    )
    cost = agent.bus.total_cost()
    hit = f" · 缓存命中 {cached / pin:.0%}" if pin and cached else ""
    console.print(
        f"[dim]窗口 {len(agent.memory.turns)}/{agent.memory.policy.window_turns} 轮"
        f"（驱逐阈值 {agent.memory.policy.evict_at}）· "
        f"约 {agent.memory.token_estimate:,} tokens · "
        f"累计 in {pin:,} / out {pout:,}{hit}"
        f"{f' · ¥{cost:.4f}' if cost else ' · 成本未知（pricing 未配）'}[/]"
    )
    asm = agent.assembler
    console.print(
        f"[dim]上下文：上次请求约 {asm.last_tokens:,} token（校准 ×{asm.calibration:.2f}，"
        f"真实 {agent.memory.observed_prompt_tokens:,}）· 预算 {asm.token_budget:,} · "
        f"折叠 {asm.last_folded} 处{'（应急档）' if asm.shrink else ''}[/]"
    )
    console.print(f"[dim]产线：{_line_label(agent)}（/type 可改）[/]")
    console.print(f"[dim]集长：{_length_brief(agent)}（/length 可改）[/]")
    console.print(f"[dim]画幅：{_ratio_brief(agent)}（/ratio 可改）[/]")
    console.print(f"[dim]{_cut_brief(agent)}[/]")
    counts = agent.assets.projects()
    console.print(
        f"[dim]项目：{escape(agent.project_title)}（{agent.project}）· "
        f"本项目资产 {counts.get(agent.project, 0)} 份 · 产物目录 {agent.output_prefs.root}[/]"
    )
    if agent.guard is not None:
        console.print(f"[dim]预算：{agent.guard.brief()}[/]")
    if agent.loop.auto_review:
        console.print("[dim]自动模式：开（小节点人审自动采纳，大节点仍问，/auto off 关闭）[/]")
    notices = agent.pipeline.notices() if agent.pipeline is not None else ""
    if notices:
        console.print(f"[yellow]{escape(notices)}[/]")
    if agent.allocator is not None:
        console.print(f"[dim]{agent.allocator.brief()}[/]")


def main() -> None:
    """控制台入口。

    显式给 prog_name：不给的话 --help 里会显示
    `python -m aigc_agent.interfaces.cli.main`，用户照着抄是跑不通的。
    """
    app(prog_name="agent")


if __name__ == "__main__":
    main()

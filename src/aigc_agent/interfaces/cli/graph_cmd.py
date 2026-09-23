"""M19 人审台 —— P1 的 CLI 形态。

P1 的验收（打回能否正确回退重跑）在终端里就能完成，别为了做界面推迟核心验证。
P2 换成 FastAPI + Web，P3 加 React Flow 图视图 —— 这里的决策逻辑不用改。
"""

from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...app import PROJECT_ROOT, Agent
from ...envdetect import workspace_root
from ...capabilities.memory.brief import build_brief
from ...capabilities.memory.recorder import RejectionRecorder
from ...capabilities.memory.store import Layer, MemoryStore
from ...domain.assets.store import AssetStore
from ...domain.pipeline.executors import AgentNodeExecutor, ToolNodeExecutor
from ...domain.pipeline.registry import GraphRegistry
from ...harness.events.bus import Event, EventType
from ...harness.execution.graph.models import Decision, GraphState, NodeType
from ...harness.execution.graph.runtime import GraphRuntime, make_state_summary

console = Console()
app = typer.Typer(help="图执行与人审", no_args_is_help=True)

WORKSPACE = workspace_root()
RUNS_DIR = WORKSPACE / "runs"
MEM_DIR = WORKSPACE / "memory"


def _render(ev: Event) -> None:
    d = ev.data
    if ev.type is EventType.NODE_START:
        console.print(f"  [cyan]▶ {d['node']}[/] [dim]({d['type']})[/]")
    elif ev.type is EventType.NODE_DONE:
        console.print(f"  [green]✓ {d['node']}[/] [dim]→ {', '.join(d['outputs']) or '无产物'}[/]")
    elif ev.type is EventType.CHECKPOINT_REACHED:
        console.print(f"  [yellow]⏸ 停在人审节点 {d['node']}[/]")
    elif ev.type is EventType.GRAPH_ROLLBACK:
        console.print(
            f"  [magenta]↩ 回退 {d['from_node']} → {d['to_node']}[/]  "
            f"[dim]快照已恢复={d['restored']}，剩余槽位 {d['slots_after']}[/]"
        )
    elif ev.type is EventType.GRAPH_HALT:
        console.print(f"  [red]■ 护栏触发：{d['reason']}（{d['node']}）[/]")
    elif ev.type is EventType.ROUTE:
        console.print(f"  [dim]· 路由 → {d['route']} {d.get('graph', '')}（{d['reason']}）[/]")


@app.command("list")
def list_graphs() -> None:
    """列出已加载的图。"""
    reg = GraphRegistry()
    reg.load_all()
    t = Table(show_header=True, header_style="bold")
    t.add_column("id")
    t.add_column("名称")
    t.add_column("节点")
    t.add_column("回退边")
    t.add_column("触发词")
    for g in reg.all():
        fb = sum(1 for e in g.edges if e.type.value == "fallback")
        t.add_row(g.id, g.name, str(len(g.nodes)), str(fb), " ".join(g.triggers[:4]))
    console.print(t)
    for name, err in reg.errors.items():
        console.print(f"[red]✗ {name}.yaml 加载失败[/]\n{err}")


@app.command()
def run(
    graph_id: str = typer.Argument(..., help="图 id，见 graph list"),
    topic: str = typer.Option(..., "--topic", "-t", help="选题 / 输入"),
    fake: bool = typer.Option(False, "--fake", help="用假执行器跑，不调模型"),
) -> None:
    """跑一张图。命中人审节点会停下来问你。"""
    asyncio.run(_run(graph_id, topic, fake))


async def _run(graph_id: str, topic: str, fake: bool) -> None:
    reg = GraphRegistry()
    reg.load_all()
    graph = reg.get(graph_id)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    assets = AssetStore(WORKSPACE / "assets")
    memories = MemoryStore(MEM_DIR)

    agent = Agent.create()
    await agent.setup()
    agent.bus.subscribe(_render)

    # 打回理由自动落库：L0 发事件，L1 订阅写库，L0 不知道有记忆这回事
    recorder = RejectionRecorder(memories, project_id=graph_id)
    recorder.attach(agent.bus)

    if fake:
        stub = _StubExecutor(assets)
        executors = {NodeType.AGENT: stub, NodeType.TOOL: stub}
    else:
        # 节点启动前拿 Memory Brief（M8.2 慢路径）：打回理由 pin 进节点内 Loop，
        # 第二版才不会重犯第一版的错
        async def _brief(node, state):  # noqa: ARG001
            return build_brief(memories, project_id=graph_id, topic=topic, stage=node.stage)

        executors = {
            NodeType.AGENT: AgentNodeExecutor(
                agent.gateway, agent.registry, agent.dispatcher, agent.bus, assets,
                brief_fn=_brief,
            ),
            NodeType.TOOL: ToolNodeExecutor(agent.registry, assets),
        }

    rt = GraphRuntime(agent.bus, executors)
    state = rt.new_state(graph)
    state.slots["topic"] = assets.create(topic, summary=topic[:30], creator="human:cli").id

    console.print(
        Panel(
            f"[bold]{graph.name}[/]（{graph.id}）\n选题：{topic}\nrun：{state.run_id}",
            border_style="cyan",
        )
    )

    state = await rt.run(graph, state)

    while state.status == "awaiting_review":
        _save(state)
        _show_candidates(state, assets)
        decision, reason = await _ask()
        if decision is None:
            console.print("[dim]已挂起，稍后可继续[/]")
            break
        state = await rt.resume(graph, state, decision, reason=reason)

    _save(state)
    console.print()
    console.print(
        Panel(
            json.dumps(make_state_summary(state), ensure_ascii=False, indent=2),
            title=f"运行结束：{state.status}",
            border_style="green",
        )
    )

    if "final" in state.slots:
        console.print(
            Panel(
                assets.content(state.slots["final"]),
                title="待发布包（上传清单）",
                border_style="green",
            )
        )

    recorded = memories.all(layer=Layer.PROJECT, project_id=graph_id)
    if recorded:
        console.print(
            f"\n[dim]本次落库 {len(recorder.recorded)} 条打回理由，库内共 {len(recorded)} 条[/]"
        )
        console.print(memories.render_brief(recorded[-3:]))


def _show_candidates(state: GraphState, assets: AssetStore) -> None:
    console.print()
    for slot, ref in state.slots.items():
        if slot == "topic":
            continue
        a = assets.get(ref)
        console.print(
            Panel(assets.content(ref)[:1200], title=f"{slot}  {a.brief()}", border_style="blue")
        )


async def _ask() -> tuple[Decision | None, str]:
    console.print(
        "[bold]决策[/]  [green]a[/]=采纳   [yellow]r[/]=打回重写(revise)   "
        "[red]j[/]=方向不对退回上游(reject)   [dim]q[/]=挂起退出"
    )
    while True:
        raw = (await asyncio.to_thread(console.input, "> ")).strip().lower()
        if raw in {"q", "quit"}:
            return None, ""
        if raw in {"a", "adopt"}:
            return Decision.ADOPT, ""
        if raw in {"r", "revise", "j", "reject"}:
            d = Decision.REVISE if raw in {"r", "revise"} else Decision.REJECT
            # 打回必须填理由 —— 不填的话 Agent 第二次会重犯同样的错
            reason = ""
            while not reason.strip():
                reason = await asyncio.to_thread(console.input, "[yellow]打回理由（必填）[/] ")
                if not reason.strip():
                    console.print("[red]理由不能为空。不记原因的话下一版会犯一模一样的错。[/]")
            return d, reason.strip()
        console.print("[dim]请输入 a / r / j / q[/]")


def _save(state: GraphState) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{state.run_id}.json").write_text(
        state.model_dump_json(indent=2), encoding="utf-8"
    )


@app.command()
def show(run_id: str = typer.Argument(..., help="运行实例 id")) -> None:
    """查看一次运行的状态。"""
    f = RUNS_DIR / f"{run_id}.json"
    if not f.exists():
        console.print(f"[red]找不到运行记录 {run_id}[/]")
        raise typer.Exit(1)
    state = GraphState.model_validate_json(f.read_text(encoding="utf-8"))
    console.print(
        Panel(
            json.dumps(make_state_summary(state), ensure_ascii=False, indent=2),
            title=run_id,
            border_style="cyan",
        )
    )


@app.command()
def memory(limit: int = typer.Option(20, help="显示条数")) -> None:
    """查看已落库的打回理由。"""
    store = MemoryStore(MEM_DIR)
    items = store.all(layer=Layer.PROJECT)[-limit:]
    if not items:
        console.print("[dim]还没有打回记录[/]")
        return
    t = Table(show_header=True, header_style="bold")
    t.add_column("时间", width=16)
    t.add_column("来源")
    t.add_column("内容")
    t.add_column("命中")
    import datetime as _dt

    for m in items:
        t.add_row(
            _dt.datetime.fromtimestamp(m.created_at).strftime("%m-%d %H:%M"),
            m.source.value,
            m.content,
            str(m.hit_count),
        )
    console.print(t)
    console.print(f"[dim]共 {len(store)} 条，存于 {MEM_DIR}[/]")


class _StubExecutor:
    """--fake 用。不调模型，产可区分的假产物。

    模型还没打通时，用它就能把「跑图 → 人审 → 打回 → 回退重跑 → 理由落库」
    整条链路手动走一遍，验证的是流程不是文笔。
    """

    def __init__(self, store: AssetStore) -> None:
        self.store = store
        self.runs: dict[str, int] = {}

    async def execute(self, node, state, inputs):
        n = self.runs[node.id] = self.runs.get(node.id, 0) + 1
        topic = self.store.content(state.slots["topic"]) if "topic" in state.slots else ""
        lines = [f"【{node.stage or node.id} · 第 {n} 版】", f"选题：{topic}"]
        if inputs:
            lines.append("上游：" + ", ".join(f"{k}={v}" for k, v in inputs.items()))
        goal = node.contract.get("目标", "—")
        lines.append(f"（stub 产物，未调模型。契约目标：{goal}）")
        body = "\n".join(lines)
        asset = self.store.create(
            body, parents=list(inputs.values()), creator="stub", summary=f"{node.id} v{n}"
        )
        return {node.out_slots[0]: asset.id} if node.out_slots else {}

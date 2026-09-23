"""M1 Graph Runtime —— 图的执行器。

核心性质：**可挂起、可恢复、可回退。**

  run(graph, state)              跑到 human 节点或终点就停下并返回
  resume(graph, state, decision) 人做完决策后从断点继续

回退边（fallback）让「打回重做」变成图上的一条普通边：
人在 human 节点选 revise/reject → 沿 fallback 边回到上游节点 →
**恢复该节点执行前的快照** → 重跑。不是散落在各处的特殊逻辑。

GraphState 与 Loop Context 的分工（见 ARCHITECTURE.md M1）：
  · 节点启动：从 GraphState 提取节点契约 → 组装成 Loop 的初始上下文
  · 节点结束：Loop 产出结构化结果 → 写回 GraphState → **上下文整个丢弃**
所以跑 20 个节点和跑 2 个节点，上下文占用是一样的。

并行（P5）：`parallel` 节点把每条出边当一个分支，各分支顺序执行到同一个 `join`
节点为止，分支之间并发。分支里不能有 human / parallel / terminal —— 人审要挂起
整张图，和"等所有分支"说不清先后；嵌套并行等真有需要再加。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Protocol

from ...events.bus import EventBus, EventType
from .models import (
    AssetRef,
    Checkpoint,
    Decision,
    EdgeType,
    GraphDef,
    GraphEdge,
    GraphNode,
    GraphState,
    NodeStatus,
    NodeType,
)

_NOT_IN_BRANCH = (NodeType.HUMAN, NodeType.PARALLEL, NodeType.TERMINAL)


class NodeExecutor(Protocol):
    """节点执行器。按 NodeType 注册。

    agent 节点的实现会在内部拉起一个 Loop（模式 A：Graph 套 Loop）；
    tool 节点直接调工具、不过模型。
    """

    async def execute(
        self, node: GraphNode, state: GraphState, inputs: dict[str, AssetRef]
    ) -> dict[str, AssetRef]:
        """返回 out_slots 的产物引用。"""
        ...


class GraphRuntime:
    def __init__(
        self,
        bus: EventBus,
        executors: dict[NodeType, NodeExecutor],
    ) -> None:
        self.bus = bus
        self.executors = executors

    # ---------- 入口 ----------

    def new_state(self, graph: GraphDef) -> GraphState:
        return GraphState(
            graph_id=graph.id,
            run_id=uuid.uuid4().hex[:12],
            current=graph.entry,
            status="pending",
        )

    async def run(self, graph: GraphDef, state: GraphState) -> GraphState:
        """从 state.current 开始跑，直到命中 human 节点、终点或护栏。"""
        state.status = "running"
        await self.bus.emit(
            EventType.GRAPH_START, graph=graph.id, run=state.run_id, node=state.current
        )

        while state.current:
            node = graph.node(state.current)

            # ---- 护栏 ----
            if state.nodes_run >= graph.max_total_nodes:
                return await self._halt(state, "max_total_nodes", node.id)
            if graph.max_cost is not None and state.cost >= graph.max_cost:
                return await self._halt(state, "max_cost", node.id)

            # ---- human 节点：挂起，交给 M19 人审台 ----
            if node.type is NodeType.HUMAN:
                state.node_status[node.id] = NodeStatus.AWAITING_REVIEW
                state.status = "awaiting_review"
                candidates = [state.slots[s] for s in node.in_slots if s in state.slots]
                state.checkpoints.append(Checkpoint(node_id=node.id, candidates=candidates))
                await self.bus.emit(
                    EventType.CHECKPOINT_REACHED,
                    graph=graph.id,
                    run=state.run_id,
                    node=node.id,
                    candidates=candidates,
                )
                return state

            # ---- terminal ----
            if node.type is NodeType.TERMINAL:
                state.node_status[node.id] = NodeStatus.DONE
                state.status = "done"
                state.current = ""
                await self.bus.emit(EventType.GRAPH_END, graph=graph.id, run=state.run_id)
                return state

            # ---- parallel：扇出到所有出边，汇聚到同一个 join ----
            if node.type is NodeType.PARALLEL:
                try:
                    join_id = await self._run_parallel(graph, node, state)
                except RuntimeError as e:
                    return await self._halt(state, f"parallel_failed: {e}", node.id)
                state.current = join_id
                continue

            # ---- join：分支已在 parallel 里跑完，这里只是汇合点 ----
            if node.type is NodeType.JOIN:
                state.node_status[node.id] = NodeStatus.DONE
                state.nodes_run += 1
                state.snapshot(node.id)
                await self.bus.emit(
                    EventType.NODE_DONE,
                    node=node.id,
                    outputs=[],
                    snapshot_seq=len(state.snapshots) - 1,
                )
                nxt, _ = self._next(graph, node.id, state)
                if nxt is None:
                    return await self._halt(state, "no_outgoing_edge", node.id)
                state.current = nxt
                continue

            # ---- 执行节点 ----
            await self._execute(graph, node, state)

            # ---- 选下一条边 ----
            nxt, _ = self._next(graph, node.id, state)
            if nxt is None:
                return await self._halt(state, "no_outgoing_edge", node.id)
            state.current = nxt

        state.status = "done"
        return state

    async def resume(
        self,
        graph: GraphDef,
        state: GraphState,
        decision: Decision,
        reason: str = "",
        decided_by: str = "human",
    ) -> GraphState:
        """人在 human 节点做完决策后继续。

        reason 是必填的：打回理由要写进 Memory 项目层（M8），
        否则第二次会生成几乎一样的东西。
        """
        if state.status != "awaiting_review":
            raise RuntimeError(f"当前状态 {state.status!r} 不在等待人审，无法 resume")
        if decision is not Decision.ADOPT and not reason.strip():
            raise ValueError("打回必须填理由 —— 不填的话 Agent 第二次会重犯同样的错")

        node = graph.node(state.current)
        cp = state.checkpoints[-1]
        cp.decision = decision
        cp.reason = reason
        cp.decided_by = decided_by
        cp.decided_at = time.time()

        state.node_status[node.id] = (
            NodeStatus.DONE if decision is Decision.ADOPT else NodeStatus.REJECTED
        )

        nxt, edge = self._next(graph, node.id, state, signal=decision.value)

        # 先算出目标节点再发事件：订阅方（如 M8 的打回理由落库）需要知道
        # 这条理由该挂在**哪个即将重跑的节点**上，而不是挂在人审节点上。
        await self.bus.emit(
            EventType.CHECKPOINT_DECIDED,
            graph=graph.id,
            run=state.run_id,
            node=node.id,
            target_node=nxt or "",
            decision=decision.value,
            reason=reason,
            decided_by=decided_by,
            candidates=cp.candidates,
        )

        if nxt is None:
            return await self._halt(state, "no_matching_edge", node.id)

        # ---- 打回：作废目标节点及其下游的产物，再重跑 ----
        # 按图依赖算，不整体恢复到某个快照：并行分支里打回正文，配图要留着
        if edge is not None and edge.type is EdgeType.FALLBACK:
            doomed = {nxt} | graph.downstream(nxt)
            restored = state.rollback_to(nxt, doomed, graph.out_slots_of(doomed))
            await self.bus.emit(
                EventType.GRAPH_ROLLBACK,
                run=state.run_id,
                from_node=node.id,
                to_node=nxt,
                restored=restored,
                reason=reason,
                slots_after=list(state.slots),
            )

        state.current = nxt
        return await self.run(graph, state)

    # ---------- 内部 ----------

    async def _execute(self, graph: GraphDef, node: GraphNode, state: GraphState) -> None:
        executor = self.executors.get(node.type)
        if executor is None:
            raise RuntimeError(f"没有为节点类型 {node.type.value!r} 注册执行器")

        state.node_status[node.id] = NodeStatus.RUNNING
        await self.bus.emit(
            EventType.NODE_START, graph=graph.id, node=node.id, type=node.type.value
        )

        inputs = {s: state.slots[s] for s in node.in_slots if s in state.slots}
        missing = [s for s in node.in_slots if s not in state.slots]
        if missing:
            raise RuntimeError(f"节点 {node.id!r} 缺少输入槽位 {missing}")

        outputs = await executor.execute(node, state, inputs)

        state.slots.update(outputs)
        state.node_status[node.id] = NodeStatus.DONE
        state.nodes_run += 1
        state.snapshot(node.id)  # ← 每个节点完成后快照，支撑单步重跑

        await self.bus.emit(
            EventType.NODE_DONE,
            node=node.id,
            outputs=list(outputs),
            snapshot_seq=len(state.snapshots) - 1,
        )

    async def _run_parallel(self, graph: GraphDef, node: GraphNode, state: GraphState) -> str:
        """每条出边一个分支，顺序执行到 join 为止；分支之间并发。返回 join 节点 id。"""
        heads = [e.to for e in graph.out_edges(node.id)]
        if len(heads) < 2:
            raise RuntimeError(f"parallel 节点 {node.id!r} 只有 {len(heads)} 条出边")
        state.node_status[node.id] = NodeStatus.RUNNING
        await self.bus.emit(
            EventType.PARALLEL_START, graph=graph.id, run=state.run_id, node=node.id, branches=heads
        )

        async def branch(head: str) -> str:
            cur = head
            steps = 0
            while True:
                n = graph.node(cur)
                if n.type is NodeType.JOIN:
                    return n.id
                if n.type in _NOT_IN_BRANCH:
                    raise RuntimeError(f"并行分支里不能有 {n.type.value} 节点（{n.id}）")
                await self._execute(graph, n, state)
                steps += 1
                if steps > graph.max_total_nodes:
                    raise RuntimeError(f"分支 {head} 走了 {steps} 步还没到 join")
                nxt, _ = self._next(graph, cur, state)
                if nxt is None:
                    raise RuntimeError(f"分支节点 {cur!r} 没有出边")
                cur = nxt

        started = time.perf_counter()
        joins = await asyncio.gather(*(branch(h) for h in heads))
        if len(set(joins)) != 1:
            raise RuntimeError(f"并行分支必须汇聚到同一个 join 节点，实际：{sorted(set(joins))}")

        state.node_status[node.id] = NodeStatus.DONE
        state.nodes_run += 1
        await self.bus.emit(
            EventType.PARALLEL_END,
            node=node.id,
            join=joins[0],
            branches=len(heads),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return joins[0]

    def _next(
        self, graph: GraphDef, node_id: str, state: GraphState, signal: str = ""
    ) -> tuple[str | None, GraphEdge | None]:
        """按信号选出边，返回 (下一节点, 走的那条边)。

        conditional / fallback 匹配 when，否则取第一条无条件边。
        """
        edges = graph.out_edges(node_id)

        if signal:
            for e in edges:
                if e.when == signal and self._can_traverse(e, state):
                    self._count(e, state)
                    return e.to, e

        for e in edges:
            if not e.when and self._can_traverse(e, state):
                self._count(e, state)
                return e.to, e
        return None, None

    @staticmethod
    def _key(e: GraphEdge) -> str:
        return f"{e.from_}->{e.to}:{e.when or '_'}"

    def _can_traverse(self, e: GraphEdge, state: GraphState) -> bool:
        if e.type is not EdgeType.LOOP:
            return True
        return state.traversals.get(self._key(e), 0) < e.max_traversals

    def _count(self, e: GraphEdge, state: GraphState) -> None:
        k = self._key(e)
        state.traversals[k] = state.traversals.get(k, 0) + 1

    async def _halt(self, state: GraphState, reason: str, node_id: str) -> GraphState:
        """任一护栏触发即挂起交给人，不静默继续也不静默降级。"""
        state.status = "halted"
        await self.bus.emit(
            EventType.GRAPH_HALT, reason=reason, node=node_id, run=state.run_id
        )
        return state


def make_state_summary(state: GraphState) -> dict[str, Any]:
    """给 CLI / 人审台看的紧凑状态。"""
    return {
        "run": state.run_id,
        "status": state.status,
        "current": state.current,
        "slots": list(state.slots),
        "nodes_run": state.nodes_run,
        "snapshots": len(state.snapshots),
        "checkpoints": [
            {"node": c.node_id, "decision": c.decision, "reason": c.reason}
            for c in state.checkpoints
        ],
    }

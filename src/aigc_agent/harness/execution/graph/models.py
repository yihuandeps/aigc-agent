"""M1 Graph Runtime —— 图的数据结构。

这里是**通用图运行时**，不认识「选题」「分镜」。具体的内容生产图定义在
M11（domain/pipeline/graphs/*.yaml），是数据不是代码。

L0 只持有 **AssetRef（字符串 id）**，不引用 L2 的 Asset 模型
—— 依赖方向必须单向向下，由 lint-imports 强制。
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

AssetRef = str  # 不透明引用。L0 不关心它背后是什么，解析是 L2 的事


class NodeType(StrEnum):
    AGENT = "agent"  # 创作类，内部跑一个 Loop
    TOOL = "tool"  # 确定性操作，不过模型，省钱省时
    HUMAN = "human"  # Checkpoint，挂起等人决策
    ROUTER = "router"  # 条件分支，优先用规则判
    SUBGRAPH = "subgraph"  # 嵌套图，复用
    PARALLEL = "parallel"
    JOIN = "join"
    TERMINAL = "terminal"  # 图的出口


class EdgeType(StrEnum):
    SEQ = "seq"
    CONDITIONAL = "conditional"
    FALLBACK = "fallback"  # 人审打回 → 回上游重跑。半自动定位下最重要的一条
    LOOP = "loop"  # 自我迭代，必须带 max_traversals


class NodeStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    DONE = "done"
    REJECTED = "rejected"


class Decision(StrEnum):
    ADOPT = "adopt"
    REVISE = "revise"
    REJECT = "reject"


class GraphNode(BaseModel):
    id: str
    type: NodeType
    stage: str = ""
    in_slots: list[str] = Field(default_factory=list)
    out_slots: list[str] = Field(default_factory=list)
    # agent 节点：注入节点内 Loop 的唯一内容。不继承完整聊天历史。
    contract: dict[str, Any] = Field(default_factory=dict)
    max_iterations: int = 12
    # tool 节点用
    tool: str = ""
    tool_args: dict[str, Any] = Field(default_factory=dict)
    # 候选数：关键节点默认产 N 个候选供人选，而不是给一个答案
    candidates: int = 1


class GraphEdge(BaseModel):
    from_: str = Field(alias="from")
    to: str
    type: EdgeType = EdgeType.SEQ
    # conditional / fallback 用：匹配 human 节点的 decision 或 router 的输出
    when: str = ""
    max_traversals: int = 3

    model_config = {"populate_by_name": True}


class Checkpoint(BaseModel):
    node_id: str
    candidates: list[AssetRef] = Field(default_factory=list)
    decision: Decision | None = None
    reason: str = ""  # ← 必填。半自动模式下人反复打回，不记原因就会重犯
    decided_by: str = ""
    decided_at: float = 0.0


class Snapshot(BaseModel):
    """节点完成后的状态快照 —— 「从第 3 步重跑」的底座。"""

    node_id: str
    slots: dict[str, AssetRef]
    seq: int


class GraphState(BaseModel):
    graph_id: str
    run_id: str
    status: str = "pending"  # pending|running|awaiting_review|done|failed|halted
    current: str = ""
    slots: dict[str, AssetRef] = Field(default_factory=dict)  # 放引用，不放内容
    node_status: dict[str, NodeStatus] = Field(default_factory=dict)
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    snapshots: list[Snapshot] = Field(default_factory=list)
    traversals: dict[str, int] = Field(default_factory=dict)  # 边 -> 走过几次
    cost: float = 0.0
    nodes_run: int = 0

    def snapshot(self, node_id: str) -> None:
        self.snapshots.append(
            Snapshot(node_id=node_id, slots=dict(self.slots), seq=len(self.snapshots))
        )

    def rollback_to(self, node_id: str, doomed_nodes: set[str], doomed_slots: set[str]) -> bool:
        """回退到 node_id 执行之前：只清掉它和它下游的产物与快照，兄弟分支的保留。

        比 restore_before 更准。并行分支各自快照、谁先完成谁在前，整体恢复到
        "目标节点快照的前一个"会把同时完成的兄弟分支产物一起抹掉 —— 打回正文
        不该连配图也丢了。下游集合由运行时按图算（不走回退边），这里只做减法。
        """
        if not any(s.node_id == node_id for s in self.snapshots):
            return False
        self.slots = {k: v for k, v in self.slots.items() if k not in doomed_slots}
        kept: list[Snapshot] = []
        for s in self.snapshots:
            if s.node_id in doomed_nodes:
                continue
            s.slots = {k: v for k, v in s.slots.items() if k not in doomed_slots}
            kept.append(s)
        self.snapshots = kept
        for n in doomed_nodes:
            if n in self.node_status and n != node_id:
                self.node_status[n] = NodeStatus.PENDING
        self.node_status[node_id] = NodeStatus.PENDING
        return True

    def restore_before(self, node_id: str) -> bool:
        """回退到 node_id **执行之前**的槽位状态。

        找该节点最近一次快照的前一个。找不到就回到初始（空槽位之后的最早状态）。
        """
        idx = None
        for i in range(len(self.snapshots) - 1, -1, -1):
            if self.snapshots[i].node_id == node_id:
                idx = i
                break
        if idx is None:
            return False
        self.slots = dict(self.snapshots[idx - 1].slots) if idx > 0 else {}
        del self.snapshots[idx:]
        return True


class GraphDef(BaseModel):
    id: str
    name: str = ""
    description: str = ""
    # 触发词。Router 用它做规则匹配 —— 规则能判就别花模型调用
    triggers: list[str] = Field(default_factory=list)
    entry: str
    nodes: list[GraphNode]
    edges: list[GraphEdge] = Field(default_factory=list)
    # 护栏：图会循环，必须有
    max_total_nodes: int = 50
    max_cost: float | None = None

    def node(self, node_id: str) -> GraphNode:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(f"图 {self.id!r} 里没有节点 {node_id!r}")

    def out_edges(self, node_id: str) -> list[GraphEdge]:
        return [e for e in self.edges if e.from_ == node_id]

    def downstream(self, node_id: str) -> set[str]:
        """沿顺序边与条件边可达的全部节点（**不走回退边和循环边**）。

        回退用：打回到 draft 时要作废的是 draft 之后的东西，
        而 review→outline 这条回退边不能把 outline 也算成"下游"。
        """
        seen: set[str] = set()
        frontier = [node_id]
        while frontier:
            cur = frontier.pop()
            for e in self.out_edges(cur):
                if e.type in (EdgeType.FALLBACK, EdgeType.LOOP) or e.to in seen:
                    continue
                seen.add(e.to)
                frontier.append(e.to)
        return seen

    def out_slots_of(self, node_ids: set[str]) -> set[str]:
        out: set[str] = set()
        for n in self.nodes:
            if n.id in node_ids:
                out.update(n.out_slots)
        return out

    def validate_graph(self) -> list[str]:
        """静态检查。图定义是手写 yaml，写错了要在加载时就报，不要跑到一半才炸。"""
        problems: list[str] = []
        ids = {n.id for n in self.nodes}
        if len(ids) != len(self.nodes):
            problems.append("存在重复的节点 id")
        if self.entry not in ids:
            problems.append(f"entry {self.entry!r} 不是已定义的节点")
        for e in self.edges:
            if e.from_ not in ids:
                problems.append(f"边的起点 {e.from_!r} 不存在")
            if e.to not in ids:
                problems.append(f"边的终点 {e.to!r} 不存在")
        for n in self.nodes:
            if n.type is NodeType.TERMINAL:
                continue
            if not self.out_edges(n.id):
                problems.append(f"节点 {n.id!r} 没有出边，也不是 terminal —— 会走进死胡同")
            if n.type is NodeType.TOOL and not n.tool:
                problems.append(f"tool 节点 {n.id!r} 未指定 tool")
            if n.type is NodeType.PARALLEL and len(self.out_edges(n.id)) < 2:
                n_out = len(self.out_edges(n.id))
                problems.append(f"parallel 节点 {n.id!r} 至少要两条出边，现在 {n_out} 条")
        return problems

    @classmethod
    def load(cls, path: str | Path) -> GraphDef:
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        graph = cls.model_validate(data)
        problems = graph.validate_graph()
        if problems:
            raise ValueError(f"图定义 {path.name} 有问题：\n  - " + "\n  - ".join(problems))
        return graph

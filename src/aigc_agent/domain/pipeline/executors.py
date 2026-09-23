"""节点执行器 —— 把 L0 的 GraphRuntime 接到真实能力上。

模式 A（Graph 套 Loop）在这里落地：agent 节点内部拉起一个**节点内 Loop**，
它的上下文只从节点契约组装，**不继承主对话历史**，节点结束即销毁。
这就是为什么跑 20 个节点和跑 2 个节点，上下文占用是一样的。

P3 加的一样东西：**节点启动前拿到 Memory Brief**（M8.2 慢路径）。
Brief 的完整形态进节点契约，must / must_not 另外 pin 在节点内 Loop 的 pre_input 位
—— 打回理由放在注意力最强的位置，第二版才不会重犯第一版的错。
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from ...harness.context.assembler import ContextAssembler
from ...harness.context.window import ShortTermMemory, WindowPolicy
from ...harness.events.bus import EventBus
from ...harness.execution.graph.models import AssetRef, GraphNode, GraphState
from ...harness.execution.loop import LoopRuntime
from ...harness.model.gateway import ModelGateway
from ...harness.tools.dispatcher import ToolDispatcher
from ...harness.tools.registry import ToolRegistry
from ..assets.store import AssetStore, AssetType

# 节点启动前产出 Brief。返回值要有 empty / render() / pin_text()（MemoryBrief 的形状），
# 这里鸭子类型，不 import L1 的具体类。
BriefFn = Callable[[GraphNode, GraphState], Awaitable[Any]]
BRIEF_PIN = "memory_brief"


def render_contract(
    node: GraphNode, store: AssetStore, inputs: dict[str, AssetRef], brief: Any = None
) -> str:
    """节点契约 → 注入节点内 Loop 的初始上下文。

    上游产物**按引用带进来**，正文由工具按需取回；契约本身只有约束和验收标准。
    Memory Brief 作为契约的一部分进来（ARCHITECTURE M8.2「核心设计」）。
    """
    parts = [f"## 当前任务：{node.stage or node.id}"]

    if node.contract:
        parts.append(json.dumps(node.contract, ensure_ascii=False, indent=2))

    if brief is not None and not getattr(brief, "empty", True):
        parts.append("## 记忆简报\n" + brief.render())

    if inputs:
        lines = []
        for slot, ref in inputs.items():
            a = store.get(ref)
            lines.append(f"### {slot}  {a.brief()}\n{store.content(ref)}")
        parts.append("## 上游产物\n" + "\n\n".join(lines))

    if node.candidates > 1:
        parts.append(
            f"## 输出要求\n产出 {node.candidates} 个**方向不同**的候选，"
            f"用 `--- 候选 N ---` 分隔，并在每个候选后一句话说明它的取舍。"
        )
    return "\n\n".join(parts)


class AgentNodeExecutor:
    """agent 节点：内部跑一个独立上下文的 Loop。"""

    def __init__(
        self,
        gateway: ModelGateway,
        registry: ToolRegistry,
        dispatcher: ToolDispatcher,
        bus: EventBus,
        store: AssetStore,
        role: str = "node_agent",
        brief_fn: BriefFn | None = None,
    ) -> None:
        self.gateway = gateway
        self.registry = registry
        self.dispatcher = dispatcher
        self.bus = bus
        self.store = store
        self.role = role
        self.brief_fn = brief_fn

    async def execute(
        self, node: GraphNode, state: GraphState, inputs: dict[str, AssetRef]
    ) -> dict[str, AssetRef]:
        # 节点边界：先拿 Memory Brief。拿不到不阻塞节点 —— 记忆是加分项
        brief = None
        if self.brief_fn is not None:
            try:
                brief = await self.brief_fn(node, state)
            except Exception:  # noqa: BLE001
                brief = None

        # 每个节点开独立子上下文 —— 不继承主对话历史，结束即丢弃
        memory = ShortTermMemory(policy=WindowPolicy(window_turns=10, evict_at=15))
        if brief is not None and not getattr(brief, "empty", True):
            text = brief.pin_text()
            if text:
                memory.pin(BRIEF_PIN, text, position="pre_input")

        loop = LoopRuntime(
            gateway=self.gateway,
            registry=self.registry,
            dispatcher=self.dispatcher,
            assembler=ContextAssembler(self.bus),
            memory=memory,
            bus=self.bus,
            role=self.role,
            max_iterations=node.max_iterations,
        )

        result = await loop.run_turn(render_contract(node, self.store, inputs, brief))
        state.cost += result.cost or 0.0

        asset = self.store.create(
            content=result.text,
            type_=_slot_type(node.out_slots[0] if node.out_slots else ""),
            parents=list(inputs.values()),
            creator=f"model:{self.role}",
            gen_params={"node": node.id, "iterations": result.iterations},
            gen_cost=result.cost,
        )
        # 上下文在这里整个丢弃，只有结构化结果写回 GraphState
        return {node.out_slots[0]: asset.id} if node.out_slots else {}


class ToolNodeExecutor:
    """tool 节点：直接调工具，**不过模型**，省钱省时。

    tool_args 里以 `$` 开头的字符串引用 GraphState 的槽位（如 `$draft`），执行时替换成
    该槽位的资产 id —— 图定义是数据，节点要用上游产物就得有这个口子。
    工具自己已经落了资产（asset_ref）的，直接把它接到出槽位上，不再包一层。
    """

    def __init__(self, registry: ToolRegistry, store: AssetStore) -> None:
        self.registry = registry
        self.store = store

    async def execute(
        self, node: GraphNode, state: GraphState, inputs: dict[str, AssetRef]
    ) -> dict[str, AssetRef]:
        args: dict[str, Any] = {}
        for key, value in node.tool_args.items():
            if isinstance(value, str) and value.startswith("$"):
                slot = value[1:]
                if slot not in state.slots:
                    raise RuntimeError(
                        f"节点 {node.id!r} 的参数 {key} 引用了不存在的槽位 {slot!r}"
                    )
                args[key] = state.slots[slot]
            else:
                args[key] = value
        result = await self.registry.invoke(node.tool, args)
        if not result.ok:
            raise RuntimeError(f"节点 {node.id!r} 的工具 {node.tool!r} 失败：{result.error}")
        if result.asset_ref:
            return {node.out_slots[0]: result.asset_ref} if node.out_slots else {}

        asset = self.store.create(
            content=result.content,
            parents=list(inputs.values()),
            creator=f"tool:{node.tool}",
            gen_params={"node": node.id, "args": args},
        )
        return {node.out_slots[0]: asset.id} if node.out_slots else {}


def _slot_type(slot: str) -> AssetType:
    return {
        "outline": AssetType.OUTLINE,
        "script": AssetType.SCRIPT,
        "storyboard": AssetType.STORYBOARD,
    }.get(slot, AssetType.TEXT)

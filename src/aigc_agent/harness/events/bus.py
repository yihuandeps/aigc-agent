"""M6 Event Bus —— 结构化事件流。

所有上层都依赖它，且它必须在任何模块崩溃时仍然能记录。
事件流同时是 UI 渲染的数据源（流式展示）和事后复盘的数据源，一套双用。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EventType(StrEnum):
    # Loop 生命周期
    LOOP_START = "loop.start"
    LOOP_END = "loop.end"
    ITERATION_START = "iteration.start"
    LOOP_STOP_REASON = "loop.stop_reason"

    # 模型
    MODEL_REQUEST = "model.request"
    MODEL_RESPONSE = "model.response"
    MODEL_RETRY = "model.retry"
    MODEL_ERROR = "model.error"
    TEXT_DELTA = "text.delta"

    # 工具
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    TOOL_ERROR = "tool.error"
    # 调度前就被拒（参数不是合法 JSON、抄回折叠占位、未知工具）：之前不发事件，复盘看不到
    TOOL_REJECTED = "tool.rejected"
    # 质检门的判定（字幕 / 人物一致性 / 镜头时长）：过没过、第几版、为什么
    GATE_VERDICT = "gate.verdict"
    # 批量任务的进度汇报（done/total）。渲染批量图/视频时由调用方发，
    # 给 CLI 的进度窗用 —— 单个工具调用自己不知道一批有几个。
    BATCH_PROGRESS = "batch.progress"
    # 资产落库（fire-and-forget，载荷带 seq 保序）。
    # 按集流水管线（EpisodePipeline）靠它做环节间的实时交接。
    ASSET_CREATED = "asset.created"

    # 权限
    PERMISSION_ASK = "permission.ask"
    PERMISSION_DENY = "permission.deny"
    # 问了人、人点了头（权限 / 超预算放行）。之前只记拒绝，复盘分不清哪些是人放的
    PERMISSION_GRANT = "permission.grant"

    # 图执行
    GRAPH_START = "graph.start"
    GRAPH_END = "graph.end"
    GRAPH_HALT = "graph.halt"
    NODE_START = "node.start"
    NODE_DONE = "node.done"
    CHECKPOINT_REACHED = "checkpoint.reached"
    CHECKPOINT_DECIDED = "checkpoint.decided"
    GRAPH_ROLLBACK = "graph.rollback"
    PARALLEL_START = "parallel.start"  # parallel 节点扇出
    PARALLEL_END = "parallel.end"
    TRACE_ROLLBACK = "trace.rollback"  # agentic 模式：以某份资产为新起点，其后作废
    ROUTE = "route"

    # 上下文
    CONTEXT_ASSEMBLED = "context.assembled"
    CONTEXT_COMPACTED = "context.compacted"  # 轮内压缩：工具参数/结果折叠成存根
    WINDOW_EVICT = "window.evict"
    CAPABILITY_BUDGET = "capability.budget"  # 能力区实际占用（M7+M20 共管）
    SKILL_RELOAD = "skill.reload"  # 运营改了 markdown，已热加载

    # 子代理（M10）
    SUBAGENT_START = "subagent.start"
    SUBAGENT_END = "subagent.end"
    FANOUT_START = "fanout.start"  # 并行扇出（M10）
    FANOUT_END = "fanout.end"

    # 发布闭环（P4）
    COMPLIANCE_CHECKED = "compliance.checked"
    PACKAGE_BUILT = "package.built"
    PACKAGE_PUBLISHED = "package.published"
    METRICS_RECORDED = "metrics.recorded"

    # 成本
    COST = "cost"
    BUDGET_EXCEEDED = "budget.exceeded"  # Cost Guard 拦下了一次调用

    WARNING = "warning"
    USER_STOP = "user.stop"  # 人按了 /stop、/now 或 Ctrl+C


class Event(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: EventType
    ts: float = Field(default_factory=time.time)
    session_id: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


Handler = Callable[[Event], None | Awaitable[None]]


class EventBus:
    """极简发布订阅。

    刻意不做持久化 —— P1 接 SQLite 时挂一个 handler 进来即可，
    不用改任何 emit 调用点。
    """

    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self._handlers: list[Handler] = []
        self.history: list[Event] = []

    def subscribe(self, handler: Handler) -> None:
        self._handlers.append(handler)

    async def emit(self, type_: EventType, **data: Any) -> Event:
        ev = Event(type=type_, session_id=self.session_id, data=data)
        self.history.append(ev)
        for h in self._handlers:
            try:
                r = h(ev)
                if asyncio.iscoroutine(r):
                    await r
            except Exception:  # noqa: BLE001 — 事件总线不能因为某个订阅者崩溃而挂掉
                pass
        return ev

    def total_cost(self) -> float:
        return sum(e.data.get("cost", 0.0) or 0.0 for e in self.history if e.type == EventType.COST)

    def total_tokens(self) -> tuple[int, int]:
        pin = sum(e.data.get("prompt_tokens", 0) for e in self.history if e.type == EventType.COST)
        pout = sum(
            e.data.get("completion_tokens", 0) for e in self.history if e.type == EventType.COST
        )
        return pin, pout

"""M1 Router —— 决定这次用哪种执行模型。

    请求进入
       ├ 有正在等人审的运行实例？        → GRAPH_RESUME（定位节点，重跑或回退）
       ├ 匹配到已注册的 Graph 模板？      → GRAPH
       ├ 探索/问答/临时任务？            → LOOP
       └ 多步但无模板？                  → PROPOSE_GRAPH（模型出图 → 人确认 → 执行）

判断标准就一条：**这个流程你会跑第二次吗？**
会，就值得定义成图；不会，就用 Loop。

**规则优先。** 能用 if 判的分支绝不花一次模型调用。规则判不了才走
`router_decision` 角色兜底 —— 这个兜底是可注入的，默认不启用。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ..events.bus import EventBus, EventType


class Route(StrEnum):
    LOOP = "loop"
    GRAPH = "graph"
    GRAPH_RESUME = "graph_resume"
    PROPOSE_GRAPH = "propose_graph"


@dataclass
class RoutingDecision:
    route: Route
    graph_id: str = ""
    run_id: str = ""
    reason: str = ""
    by: str = "rule"  # rule | model


class GraphCatalog(Protocol):
    """图目录。由 L2 的 GraphRegistry 实现 —— L0 不认识具体的图。"""

    def triggers(self) -> dict[str, list[str]]:
        """graph_id -> 触发词列表。"""
        ...

    def has(self, graph_id: str) -> bool: ...


# 可注入的模型兜底：规则判不了时调一次，返回 graph_id 或空串
ModelFallback = Callable[[str, dict[str, list[str]]], Awaitable[str]]

# 明确的探索性信号 —— 命中就直接走 Loop，不用问模型
_LOOP_HINTS = (
    "为什么", "怎么看", "什么是", "解释", "分析一下", "帮我看看",
    "查一下", "搜一下", "对比", "建议", "?", "？",
)


class ExecutionRouter:
    def __init__(
        self,
        bus: EventBus,
        catalog: GraphCatalog | None = None,
        model_fallback: ModelFallback | None = None,
    ) -> None:
        self.bus = bus
        self.catalog = catalog
        self.model_fallback = model_fallback

    async def route(
        self,
        user_input: str,
        pending_run: tuple[str, str] | None = None,
    ) -> RoutingDecision:
        """pending_run: (graph_id, run_id)，存在则说明有实例卡在人审。"""
        text = user_input.strip()

        # ---- 1. 有实例等着人审，优先回到它 ----
        if pending_run:
            d = RoutingDecision(
                route=Route.GRAPH_RESUME,
                graph_id=pending_run[0],
                run_id=pending_run[1],
                reason="有运行实例正卡在人审节点，先处理它",
            )
            return await self._emit(d, text)

        triggers = self.catalog.triggers() if self.catalog else {}

        # ---- 2. 显式指定图：「跑 copy 图」「用短视频流程」----
        for gid in triggers:
            if gid in text:
                return await self._emit(
                    RoutingDecision(Route.GRAPH, gid, reason=f"输入里直接点名了图 {gid!r}"), text
                )

        # ---- 3. 探索性信号 → Loop（规则先判，省一次模型调用）----
        if any(h in text for h in _LOOP_HINTS):
            return await self._emit(
                RoutingDecision(Route.LOOP, reason="含探索/问答信号，不走固定流程"), text
            )

        # ---- 4. 触发词匹配 ----
        best, score = "", 0
        for gid, words in triggers.items():
            n = sum(1 for w in words if w and w in text)
            if n > score:
                best, score = gid, n
        if score:
            return await self._emit(
                RoutingDecision(Route.GRAPH, best, reason=f"命中 {score} 个触发词"), text
            )

        # ---- 5. 规则判不了，才考虑问模型 ----
        if self.model_fallback and triggers:
            gid = (await self.model_fallback(text, triggers)).strip()
            if gid and (not self.catalog or self.catalog.has(gid)):
                return await self._emit(
                    RoutingDecision(Route.GRAPH, gid, reason="规则未命中，模型判定", by="model"),
                    text,
                )

        # ---- 6. 兜底走 Loop ----
        # 注意这里**不**自动走 PROPOSE_GRAPH：动态生成图必须人确认后才执行，
        # 不能因为一句没匹配上的话就悄悄编排出一条流水线。
        return await self._emit(
            RoutingDecision(Route.LOOP, reason="无匹配图模板，按临时任务处理"), text
        )

    async def _emit(self, d: RoutingDecision, text: str) -> RoutingDecision:
        await self.bus.emit(
            EventType.ROUTE,
            route=d.route.value,
            graph=d.graph_id,
            run=d.run_id,
            by=d.by,
            reason=d.reason,
            input_preview=text[:60],
        )
        return d

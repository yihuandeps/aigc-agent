"""内置工具 Provider —— P0 的第一个 ToolProvider 实现。

这里的几个工具是为了验证 loop 能跑通（多轮、并发、权限分级），
不是最终的领域工具。真正的生成/剪辑/发布工具在 L2 domain 层，
到 P2 再作为独立 Provider 注册进来。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from .provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
    ToolSpec,
)

Handler = Callable[..., Awaitable[Any]]


class BuiltinProvider:
    """装饰器注册式的内置工具集。"""

    name = "builtin"
    namespaced = False  # 内置工具不加前缀

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Handler] = {}

    def tool(self, spec: ToolSpec) -> Callable[[Handler], Handler]:
        def deco(fn: Handler) -> Handler:
            self._specs[spec.name] = spec
            self._handlers[spec.name] = fn
            return fn

        return deco

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            out = await self._handlers[tool](**args)
            return ToolResult(
                ok=True,
                content=out if isinstance(out, str) else str(out),
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as e:  # noqa: BLE001 — 工具异常不能冒泡打断 loop
            return ToolResult(
                ok=False,
                error=f"{type(e).__name__}: {e}",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

    async def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail=f"{len(self._specs)} 个内置工具")


# ---------------------------------------------------------------- 工具定义

builtin = BuiltinProvider()


@builtin.tool(
    ToolSpec(
        name="now",
        summary="获取当前日期和时间",
        permission=PermissionLevel.READ,
        parameters={"type": "object", "properties": {}},
    )
)
async def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


@builtin.tool(
    ToolSpec(
        name="calc",
        summary="计算一个算术表达式",
        permission=PermissionLevel.READ,
        description="计算算术表达式并返回结果。只支持数字和 + - * / ( ) ** % 运算。",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "如 (1+2)*3"}
            },
            "required": ["expression"],
        },
    )
)
async def _calc(expression: str) -> str:
    allowed = set("0123456789+-*/(). %")
    if not set(expression) <= allowed:
        raise ValueError("表达式含不允许的字符，只支持数字和 + - * / ( ) . % 空格")
    return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 — 已做字符白名单


# 演示工具（验证并发和 L-external 确认闸门用）：单独一个 provider，**只在测试里注册** ——
# 之前混在 builtin 里进了生产工具目录，模型真的会去调「模拟发布」（2026-09-23 审查）
demo = BuiltinProvider()
demo.name = "demo"


@demo.tool(
    ToolSpec(
        name="sleep_demo",
        summary="等待指定秒数，用于验证工具并发执行",
        permission=PermissionLevel.COMPUTE,
        description=(
            "等待 seconds 秒后返回。用来验证同一轮内的多个工具调用是并发执行的"
            "——并发时总耗时约等于最慢的那个，而不是求和。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "description": "等待秒数，0-5"},
                "label": {"type": "string", "description": "标记名，便于区分"},
            },
            "required": ["seconds"],
        },
    )
)
async def _sleep_demo(seconds: float, label: str = "task") -> str:
    seconds = max(0.0, min(float(seconds), 5.0))
    await asyncio.sleep(seconds)
    return f"{label} 完成，耗时 {seconds}s"


@demo.tool(
    ToolSpec(
        name="publish_demo",
        summary="模拟发布内容到外部平台（不可逆动作，需人工确认）",
        permission=PermissionLevel.EXTERNAL,
        description="模拟一次对外发布。用于验证 L-external 等级会触发人工确认闸门。",
        parameters={
            "type": "object",
            "properties": {
                "platform": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["platform", "content"],
        },
    )
)
async def _publish_demo(platform: str, content: str) -> str:
    return f"[模拟] 已发布到 {platform}：{content[:60]}"

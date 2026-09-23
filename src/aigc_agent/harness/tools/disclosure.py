"""`load_tool_schema` 元工具 —— 两级披露的模型侧入口。

模型先在目录里看到「有这么个工具、大概干什么」（约 30 token），
真要用了再调这个元工具把完整参数定义拉进上下文（300-800 token）。

60 个工具：目录 1.8K vs 全量 30K，差 17 倍。这就是它存在的全部理由。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from .provider import PermissionLevel, ProviderHealth, ToolMeta, ToolResult, ToolSpec

if TYPE_CHECKING:
    from .registry import ToolRegistry

SPEC = ToolSpec(
    name="load_tool_schema",
    summary="展开某个工具的完整参数定义，之后才能调用它",
    permission=PermissionLevel.READ,
    description=(
        "工具目录里标了「需先 load_tool_schema 展开参数」的工具，必须先用本函数"
        "把它的完整参数定义载入，然后才能调用。一次可传多个名字。"
        "已经能直接看到参数的工具不需要展开。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "工具名（目录里的完整名，含 server 前缀）",
            }
        },
        "required": ["names"],
    },
)


class DisclosureProvider:
    """只提供一个元工具。持有 registry 引用，属于 M4 内部，不跨层。"""

    name = "meta"
    namespaced = False
    disclosure = "full"  # 元工具本身必须始终可见，否则模型没法展开任何东西

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def list_tools(self) -> list[ToolMeta]:
        return [
            ToolMeta(
                name=SPEC.name,
                summary=SPEC.summary,
                permission=SPEC.permission,
                provider=self.name,
            )
        ]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return SPEC.to_openai(SPEC.name)

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        names = args.get("names") or []
        if isinstance(names, str):
            names = [names]
        if not names:
            return ToolResult(ok=False, error="names 不能为空")

        lines: list[str] = []
        any_ok = False
        for n in names:
            ok, msg = self.registry.expand(str(n))
            any_ok = any_ok or ok
            lines.append(("✓ " if ok else "✗ ") + msg)

        return ToolResult(
            ok=any_ok,
            content="\n".join(lines),
            error=None if any_ok else "\n".join(lines),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    async def health(self) -> ProviderHealth:
        pending = len(self.registry.pending_expansion)
        return ProviderHealth(ok=True, detail=f"{pending} 个工具待展开")

"""按需展开的工具：追加在末尾、满额收回最久没用的（2026-09-29 审查 2.5）。

之前展开的工具按目录顺序插进请求里的工具列表中间：展开一个，它后面的工具定义整段错位，
前缀缓存从那里起全部失效。满额时 set.pop() 收回的是随便哪一个，可能正是刚在用的。
"""

from __future__ import annotations

from typing import Any

from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.tools.provider import PermissionLevel, ToolResult, ToolSpec
from aigc_agent.harness.tools.registry import ToolRegistry


class _Provider:
    namespaced = False

    def __init__(self, name: str, tools: list[str], disclosure: str = "full") -> None:
        self.name = name
        self.disclosure = disclosure
        self._specs = {
            t: ToolSpec(name=t, summary=f"{t} 工具", permission=PermissionLevel.READ) for t in tools
        }

    async def list_tools(self) -> list[Any]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        return ToolResult(content=f"{tool} ok")

    async def health(self) -> Any:
        return None


async def _registry(max_expanded: int = 8) -> ToolRegistry:
    reg = ToolRegistry(EventBus())
    reg.max_expanded = max_expanded
    reg.register(_Provider("core", ["a", "b", "c"]))
    reg.register(_Provider("ext", ["x1", "x2", "x3", "x4"], disclosure="digest"))
    await reg.refresh()
    return reg


async def _sent(reg: ToolRegistry) -> list[str]:
    return [s["function"]["name"] for s in await reg.schemas_for_context()]


async def test_展开的工具按展开先后追加在末尾_前面的不挪位置():
    reg = await _registry()
    before = await _sent(reg)
    assert before == ["a", "b", "c"]
    reg.expand("x3")
    reg.expand("x1")
    after = await _sent(reg)
    assert after == ["a", "b", "c", "x3", "x1"], "不按目录顺序插到中间"
    reg.expand("x3")  # 已经展开的再展开一次：位置不动
    assert await _sent(reg) == after


async def test_满额收回最久没用的_调用也算用到():
    reg = await _registry(max_expanded=2)
    reg.expand("x1")
    reg.expand("x2")
    await reg.invoke_ungated("x1", {})  # x1 刚用过，x2 最久没用
    ok, msg = reg.expand("x3")
    assert ok and "x2" in msg
    assert await _sent(reg) == ["a", "b", "c", "x1", "x3"]
    assert reg.expanded == ["x1", "x3"]


async def test_收回后照样能再展开():
    reg = await _registry(max_expanded=1)
    reg.expand("x1")
    assert reg.collapse("x1") and not reg.collapse("x1")
    assert await _sent(reg) == ["a", "b", "c"]
    reg.expand("x4")
    assert await _sent(reg) == ["a", "b", "c", "x4"]

"""MCP 客户端 —— 传输层抽象。

抽成协议是为了两件事：
  1. 测试不用真起进程（FakeMcpClient 可注入工具和故障）
  2. 将来加新传输（streamable http）不动上层
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import ServerSpec


@dataclass
class RemoteTool:
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )

    def __post_init__(self) -> None:
        if not self.input_schema:
            self.input_schema = {"type": "object", "properties": {}}


class McpClient(Protocol):
    async def connect(self, timeout: float) -> None:  # noqa: ASYNC109
        ...
    async def list_tools(self) -> list[RemoteTool]: ...
    async def call_tool(self, name: str, args: dict[str, Any]) -> str: ...
    async def close(self) -> None: ...


class SdkMcpClient:
    """基于官方 mcp SDK 的实现。stdio 与 sse 两种传输。"""

    def __init__(self, spec: ServerSpec) -> None:
        self.spec = spec
        self._stack: AsyncExitStack | None = None
        self._session: Any = None

    async def connect(self, timeout: float) -> None:  # noqa: ASYNC109
        # 延迟导入：没装 mcp 也不影响其余功能可用
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        stack = AsyncExitStack()
        try:
            if self.spec.transport == "stdio":
                params = StdioServerParameters(
                    command=self.spec.command,
                    args=list(self.spec.args),
                    env={**self.spec.env} or None,
                    cwd=self.spec.cwd,
                )
                read, write = await asyncio.wait_for(
                    stack.enter_async_context(stdio_client(params)), timeout=timeout
                )
            else:
                from mcp.client.sse import sse_client

                read, write = await asyncio.wait_for(
                    stack.enter_async_context(
                        sse_client(self.spec.url, headers=self.spec.headers or None)
                    ),
                    timeout=timeout,
                )

            session = await stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(session.initialize(), timeout=timeout)
        except BaseException:
            await stack.aclose()
            raise

        self._stack = stack
        self._session = session

    async def list_tools(self) -> list[RemoteTool]:
        if self._session is None:
            raise RuntimeError("尚未连接")
        resp = await self._session.list_tools()
        return [
            RemoteTool(
                name=t.name,
                description=t.description or "",
                input_schema=dict(_field(t, "input_schema", "inputSchema") or {}),
            )
            for t in resp.tools
        ]

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        if self._session is None:
            raise RuntimeError("尚未连接")
        result = await self._session.call_tool(name, args)
        if _field(result, "is_error", "isError"):
            raise RuntimeError(_render_content(result) or "server 返回错误")
        return _render_content(result)

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None


def _field(obj: Any, *names: str) -> Any:
    """mcp SDK 2.x 把 inputSchema / isError 改成了 snake_case，1.x 是 camelCase。
    两个名字都认，装哪个版本都能跑 —— 这条路之前从没被真 server 验过。"""
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


def _render_content(result: Any) -> str:
    """把 MCP 的 content 块拍平成文本。

    返回内容一律当**不可信数据**处理，不是指令 —— 调用方负责截断和标注。
    """
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
            continue
        kind = getattr(block, "type", "?")
        # 图片/音频等非文本块：只留占位，避免把 base64 塞进上下文
        parts.append(f"[{kind} 内容，未内联]")
    return "\n".join(parts)


class FakeMcpClient:
    """测试用。可注入工具、返回值、以及各类故障。"""

    def __init__(
        self,
        tools: list[RemoteTool] | None = None,
        results: dict[str, str] | None = None,
        fail_connect: bool = False,
        fail_list: bool = False,
        fail_calls: set[str] | None = None,
        hang_seconds: float = 0.0,
    ) -> None:
        self.tools = tools or []
        self.results = results or {}
        self.fail_connect = fail_connect
        self.fail_list = fail_list
        self.fail_calls = fail_calls or set()
        self.hang_seconds = hang_seconds
        self.connected = False
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def connect(self, timeout: float) -> None:  # noqa: ASYNC109
        if self.fail_connect:
            raise ConnectionError("模拟连接失败")
        self.connected = True

    async def list_tools(self) -> list[RemoteTool]:
        if self.fail_list:
            raise RuntimeError("模拟列举失败")
        return list(self.tools)

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        self.calls.append((name, args))
        if self.hang_seconds:
            await asyncio.sleep(self.hang_seconds)
        if name in self.fail_calls:
            raise RuntimeError(f"模拟 {name} 调用失败")
        return self.results.get(name, f"{name} 已执行")

    async def close(self) -> None:
        self.connected = False

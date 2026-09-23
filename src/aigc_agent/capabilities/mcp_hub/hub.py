"""M20 MCP Hub —— 管理多个 server 的连接与生命周期。

Hub 本身不是 ToolProvider，它是**生产 Provider 的管理器**：
每个 server 产出一个 McpServerProvider 注册进 M4。

这样命名空间、冲突检测、按健康摘除全部复用 M4 已有的逻辑，
不在 Hub 里重复实现一遍。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from ...harness.events.bus import EventBus, EventType
from ...harness.tools.registry import ToolRegistry
from .client import McpClient, SdkMcpClient
from .config import McpConfig, ServerSpec
from .provider import McpServerProvider

ClientFactory = Callable[[ServerSpec], McpClient]


class McpHub:
    def __init__(
        self,
        config: McpConfig,
        bus: EventBus,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.config = config
        self.bus = bus
        self.client_factory = client_factory or SdkMcpClient
        self.providers: dict[str, McpServerProvider] = {}

    @classmethod
    def from_file(
        cls, path: str | Path, bus: EventBus, client_factory: ClientFactory | None = None
    ) -> McpHub:
        return cls(McpConfig.load(path), bus, client_factory)

    async def connect_all(self) -> dict[str, bool]:
        """并行连接所有启用的 server。

        单个失败不影响其余 —— 也不阻塞启动。失败的 server 其工具不进注册表，
        模型看不到必然失败的工具。
        """
        for p in self.config.problems:
            await self.bus.emit(EventType.WARNING, message=f"[MCP 配置] {p}")

        specs = self.config.enabled_servers
        if not specs:
            return {}

        async def one(spec: ServerSpec) -> tuple[str, bool]:
            provider = McpServerProvider(
                spec, self.client_factory(spec), self.config.settings
            )
            ok = await provider.connect()
            self.providers[spec.alias] = provider
            health = await provider.health()
            await self.bus.emit(
                EventType.WARNING if not ok else EventType.ROUTE,
                kind="mcp_connect",
                server=spec.alias,
                ok=ok,
                transport=spec.transport,
                detail=health.detail,
                message=None if ok else f"[MCP] server {spec.alias!r} 连接失败：{health.detail}",
            )
            return spec.alias, ok

        if self.config.settings.parallel_connect:
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(one(s)) for s in specs]
            results = [t.result() for t in tasks]
        else:
            results = [await one(s) for s in specs]

        return dict(results)

    def register_into(self, registry: ToolRegistry) -> list[str]:
        """把连上的 server 注册进 M4。返回注册成功的 alias。

        **MCP 工具不走独立通路** —— 和内置工具、Skill 工具共用同一个注册表、
        同一套权限、同一套成本记账、同一套日志。
        """
        registered: list[str] = []
        for alias, provider in self.providers.items():
            try:
                registry.register(provider)
            except ValueError:
                continue  # 已注册过
            registered.append(alias)
        return registered

    async def close_all(self) -> None:
        for p in self.providers.values():
            try:
                await p.close()
            except Exception:  # noqa: BLE001 — 关闭失败不该冒泡
                pass
        self.providers.clear()

    # ---------- 观察 ----------

    async def status(self) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for alias, p in self.providers.items():
            h = await p.health()
            tools = await p.list_tools()
            out.append(
                {
                    "alias": alias,
                    "transport": p.spec.transport,
                    "ok": h.ok,
                    "detail": h.detail,
                    "tools": len(tools),
                    "externals": len(p.externals),
                    "circuit_open": p.circuit.open,
                }
            )
        return out

    def audit_permissions(self) -> list[str]:
        """列出所有仍是 L-external 的外部工具，供人核对。

        默认最严是对的，但如果一个只读工具一直卡在 L-external，
        每次调用都要人点一下，用起来会很烦 —— 这个清单是让人去降级的。
        """
        lines: list[str] = []
        for alias, p in self.providers.items():
            for tool in p.externals:
                lines.append(f"{alias}__{tool}")
        return lines

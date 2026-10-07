"""M4 工具注册表 —— 聚合所有 Provider，统一命名与两级披露。

统一注册表是本架构的一条硬规则：MCP 工具、内置工具、Skill 工具走
同一条路。否则会得到三套权限逻辑、三套成本记账、三套日志
—— 这类系统最常见的架构腐化起点。
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import Any

from ..events.bus import EventBus, EventType
from .provider import PermissionLevel, ProviderHealth, ToolMeta, ToolProvider, ToolResult

NAMESPACE_SEP = "__"


class ToolRegistry:
    def __init__(self, bus: EventBus, gate: Any = None) -> None:
        self.bus = bus
        # 权限闸门。**默认安全**：公开的 invoke() 一律过闸门。
        # 之前闸门只装在 ToolDispatcher 里，导致任何直接调 registry.invoke()
        # 的代码（CLI 命令、Graph 的 tool 节点）全部绕过了权限检查 ——
        # L-external 的工具可以被静默执行。这是不该有的默认。
        self.gate = gate
        self._providers: dict[str, ToolProvider] = {}
        self._catalog: dict[str, ToolMeta] = {}  # 最终名 -> meta
        self._origin: dict[str, tuple[str, str]] = {}  # 最终名 -> (provider, 原始名)
        self._down: set[str] = set()  # 熔断中的 provider
        # 两级披露：digest 级 provider 的工具默认只上目录，全 schema 按需展开。
        # 不做这个的话，接 5 个 MCP server（60-80 个工具）光工具定义就吃掉
        # 30-50K token —— 占 256K 窗口的 20%，一行活还没干。
        self._digest_only: set[str] = set()  # 最终名
        self._expanded: set[str] = set()  # 已展开全 schema 的
        self.max_expanded = 8
        # 按产线收窄：不在这里的 provider 的工具也只上目录、按需展开。None = 不收窄。
        # 2026-09-23 审查：两级披露对内置工具没生效 —— 89 份 schema 每次请求全量发送
        self._focus: set[str] | None = None

    def set_focus(self, providers: Iterable[str] | None) -> None:
        """按产线收窄全披露范围（/type 选了产线时由装配层调）。"""
        self._focus = set(providers) if providers is not None else None

    def _is_digest(self, name: str) -> bool:
        if name in self._digest_only:
            return True
        if self._focus is None:
            return False
        return self._origin.get(name, ("", ""))[0] not in self._focus

    def register(self, provider: ToolProvider) -> None:
        if provider.name in self._providers:
            raise ValueError(f"provider {provider.name!r} 重复注册")
        self._providers[provider.name] = provider

    async def refresh(self) -> None:
        """重建目录。provider 挂了就跳过它，不拖垮整体。"""
        self._catalog.clear()
        self._origin.clear()

        for pname, provider in self._providers.items():
            try:
                metas = await provider.list_tools()
            except Exception as e:  # noqa: BLE001
                self._down.add(pname)
                await self.bus.emit(
                    EventType.WARNING,
                    message=f"provider {pname!r} 列举工具失败，其工具已临时摘除：{e}",
                )
                continue

            self._down.discard(pname)
            for meta in metas:
                final = (
                    f"{pname}{NAMESPACE_SEP}{meta.name}"
                    if getattr(provider, "namespaced", True)
                    else meta.name
                )
                if final in self._catalog:
                    await self.bus.emit(
                        EventType.WARNING,
                        message=f"工具名冲突：{final!r} 已存在，来自 {pname!r} 的被跳过",
                    )
                    continue
                self._catalog[final] = meta.model_copy(update={"name": final, "provider": pname})
                self._origin[final] = (pname, meta.name)
                if getattr(provider, "disclosure", "full") == "digest":
                    self._digest_only.add(final)

    # ---------- 两级披露 ----------

    def catalog(self) -> list[ToolMeta]:
        """L1 目录级：name + 一句话，常驻上下文。"""
        return list(self._catalog.values())

    expand_tool_name = "load_tool_schema"

    def catalog_digest(self, names: Iterable[str] | None = None, compact: bool = False) -> str:
        """渲染成给模型看的紧凑目录。约 30 token/工具，常驻上下文。
        compact=True：只列要展开的（随请求发的系统区用），全披露的只报个数。

        names 给了就只渲染这个子集（子代理的裁剪视图用）。

        注意：需要展开的说明**只在开头讲一次**，每个工具只用一个 `*` 标记。
        把这句话在每行重复一遍的话，60 个工具就是 2400 字符的纯废话，
        正好把两级披露想省的 token 又吃回去 —— 这里被测试盯着。
        """
        allow = set(names) if names is not None else None
        lines: list[str] = []
        needs_expand = False
        full = 0
        for m in self._catalog.values():
            if allow is not None and m.name not in allow:
                continue
            star = ""
            if self._is_digest(m.name) and m.name not in self._expanded:
                star = "*"
                needs_expand = True
            elif compact:
                # 紧凑版（随请求发的系统区）：完整定义已在 tools 参数里的不再列一遍
                full += 1
                continue
            lines.append(f"- {m.name}{star}（{m.permission.value}）：{m.summary}")

        if needs_expand:
            lines.insert(
                0,
                f"标 * 的工具需先调 {self.expand_tool_name}(names=[...]) 载入参数定义才能使用。",
            )
        if compact and full:
            lines.insert(0, f"另有 {full} 个工具的完整定义已随请求提供，直接调用。")
        return "\n".join(lines)

    def expand(self, name: str) -> tuple[bool, str]:
        """展开某个工具的完整 schema，使其可被调用。"""
        if name not in self._catalog:
            near = [n for n in self._catalog if name.lower() in n.lower()][:5]
            hint = f"。你是想找：{', '.join(near)}？" if near else ""
            return False, f"没有名为 {name!r} 的工具{hint}"
        if name in self._expanded:
            return True, f"{name} 的完整定义已在上下文里，直接调用即可"
        if len(self._expanded) >= self.max_expanded:
            dropped = self._expanded.pop()
            self._expanded.add(name)
            return True, f"已展开 {name}（达上限，收回了 {dropped}）"
        self._expanded.add(name)
        return True, f"已展开 {name}，现在可以调用它了"

    async def schemas_for_context(
        self, names: Iterable[str] | None = None
    ) -> list[dict[str, Any]]:
        """真正随请求发出去的工具定义。

        = 全披露 provider 的全部工具 + 已展开的 digest 工具。
        未展开的 digest 工具只出现在目录里，不占 schema 预算。
        names 给了就只取这个子集。
        """
        allow = set(names) if names is not None else None
        picked = [
            n
            for n in self._catalog
            if (allow is None or n in allow) and (not self._is_digest(n) or n in self._expanded)
        ]
        return await self.schemas(picked)

    def collapse(self, name: str) -> bool:
        """收回一个已展开的 digest 工具（能力预算降级用）。"""
        if name in self._expanded:
            self._expanded.discard(name)
            return True
        return False

    @property
    def expanded(self) -> list[str]:
        return sorted(self._expanded)

    @property
    def digest_only(self) -> list[str]:
        return sorted(n for n in self._catalog if self._is_digest(n))

    @property
    def full_names(self) -> list[str]:
        """全披露的工具（schema 常驻请求体）。"""
        return [n for n in self._catalog if not self._is_digest(n)]

    @property
    def pending_expansion(self) -> list[str]:
        return sorted(n for n in self._catalog if self._is_digest(n) and n not in self._expanded)

    def scoped(self, names: Iterable[str]) -> ScopedRegistry:
        """裁剪出一个子集视图给子代理（M10）。"""
        return ScopedRegistry(self, names)

    async def schemas(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """L2 全 schema：按需展开。names 为 None 时展开全部。

        P0 工具少，默认全展开。P2 接入 MCP 后工具会到几十个，
        那时必须传 names 做子集，否则光工具定义就吃掉 30K+ token。
        """
        out: list[dict[str, Any]] = []
        for final in names if names is not None else self._catalog:
            if final not in self._origin:
                continue
            pname, orig = self._origin[final]
            schema = await self._providers[pname].get_schema(orig)
            # 覆盖成带命名空间的最终名
            schema.setdefault("function", {})["name"] = final
            out.append(schema)
        return out

    # ---------- 调用 ----------

    def meta(self, name: str) -> ToolMeta | None:
        return self._catalog.get(name)

    def meta_for_call(self, name: str, args: dict[str, Any]) -> ToolMeta | None:
        """这一次调用该按什么等级过闸门。

        静态等级写在目录里；但有些工具的风险取决于参数 —— 同一个 fs_write，写进产物目录
        是可回滚的 L-write，写进 Agent 自己的配置/代码/台账就等于让模型改自己的规则
        （2026-09-23 审查：模型能改 models.yaml 的预算、往 mcp_servers.yaml 加任意命令、
        清零台账，全都不经确认）。Provider 可选实现 permission_for(tool, args)，
        返回 (等级, 原因) 就按那个等级过闸门，原因写进询问提示。L0 不认识「路径」，
        只认这个钩子。
        """
        meta = self._catalog.get(name)
        if meta is None or name not in self._origin:
            return meta
        pname, orig = self._origin[name]
        provider = self._providers.get(pname)
        update: dict[str, Any] = {}
        # 花钱的工具：按参数预估这次要几段 / 几秒 / 多少钱，闸门据此事前拦
        est_hook = getattr(provider, "estimate_cost", None)
        if meta.cost_kind and est_hook is not None:
            try:
                est = est_hook(orig, args)
            except Exception:  # noqa: BLE001 — 预估失败就按次数口径走
                est = None
            if est:
                update["estimate"] = dict(est)
        # 超时按参数放宽：批量渲染要生成多少张事先数得出来，写死的超时撞上就是已付费的白丢
        # （2026-09-29 审查 1.4）。只放宽、不收紧
        to_hook = getattr(provider, "timeout_for", None)
        if to_hook is not None:
            try:
                t = to_hook(orig, args)
            except Exception:  # noqa: BLE001 — 钩子出错按目录里的超时走
                t = None
            if t and t > (meta.timeout or 0):
                update["timeout"] = float(t)
        hook = getattr(provider, "permission_for", None)
        raised = None
        if hook is not None:
            try:
                raised = hook(orig, args)
            except Exception:  # noqa: BLE001 — 钩子出错按静态等级走，不能让它拖垮调用
                raised = None
        if raised:
            level, why = raised
            if level != meta.permission:
                update["permission"] = PermissionLevel(level)
                update["summary"] = f"{why}（{meta.summary}）"
        return meta.model_copy(update=update) if update else meta

    async def invoke(self, name: str, args: dict[str, Any]) -> ToolResult:
        """公开入口，**过权限闸门**，并发 TOOL_CALL / TOOL_RESULT 事件。

        发事件是为了执行痕迹（M1 Trace）：配方流水线（agent video）和图的 tool
        节点都是直接调这里、不经过 Dispatcher，之前这些调用**不进 DAG** ——
        跑完一条短视频，/trace 里一个节点都没有。Dispatcher 走 invoke_ungated()
        并自己发事件，两条路各发各的，不会重复。
        """
        meta = self.meta_for_call(name, args)
        if meta is not None and self.gate is not None:
            ok, reason = await self.gate.check(meta, args)
            if not ok:
                return ToolResult(ok=False, error=reason)

        call_id = f"direct_{uuid.uuid4().hex[:8]}"
        await self.bus.emit(EventType.TOOL_CALL, tool=name, args=args, call_id=call_id)
        result = await self.invoke_ungated(name, args)
        settle = getattr(self.gate, "settle", None)
        if meta is not None and callable(settle):
            settle(meta, args, result)  # 没真花钱的（被拦 / 挂起 / 提交被拒）退回额度
        await self.bus.emit(
            EventType.TOOL_RESULT if result.ok else EventType.TOOL_ERROR,
            tool=name,
            call_id=call_id,
            ok=result.ok,
            error=result.error,
            duration_ms=result.duration_ms,
            truncated=result.truncated,
            preview=result.content[:200],
            asset_ref=result.asset_ref,
            suspend=result.suspend,
            args=args,
        )
        return result

    async def invoke_ungated(self, name: str, args: dict[str, Any]) -> ToolResult:
        """不过闸门。**只给已经自行校验过权限的调用方用**（如 ToolDispatcher，
        它对一批工具串行过闸门后再并发执行）。其余场景一律用 invoke()。
        """
        if name not in self._origin:
            return ToolResult(
                ok=False,
                error=(
                    f"未知工具 {name!r}。当前可用："
                    f"{', '.join(sorted(self._catalog)) or '（无）'}"
                ),
            )
        pname, orig = self._origin[name]
        if pname in self._down:
            return ToolResult(ok=False, error=f"provider {pname!r} 当前不可用，该工具已临时摘除")
        return await self._providers[pname].invoke(orig, args)

    async def health(self) -> dict[str, ProviderHealth]:
        out: dict[str, ProviderHealth] = {}
        for pname, provider in self._providers.items():
            try:
                out[pname] = await provider.health()
            except Exception as e:  # noqa: BLE001
                out[pname] = ProviderHealth(ok=False, detail=str(e))
        return out


class ScopedRegistry:
    """全局注册表的一个子集视图 —— 子代理（M10）用。

    子代理只看得到、只调得了它被允许的那几个工具；权限闸门另配（gate 属性）。
    不复制任何东西：目录、schema、调用都委托给父注册表，只是过滤名字。
    LoopRuntime / ToolDispatcher 需要的接口它都有，换进去就能跑。
    """

    expand_tool_name = ToolRegistry.expand_tool_name

    def __init__(self, parent: ToolRegistry, names: Iterable[str]) -> None:
        self.parent = parent
        self.names = set(names)
        self.bus = parent.bus
        self.gate: Any = None

    def catalog(self) -> list[ToolMeta]:
        return [m for m in self.parent.catalog() if m.name in self.names]

    def catalog_digest(self, names: Iterable[str] | None = None, compact: bool = False) -> str:
        allow = self.names if names is None else (self.names & set(names))
        return self.parent.catalog_digest(allow, compact=compact)

    async def schemas_for_context(
        self, names: Iterable[str] | None = None
    ) -> list[dict[str, Any]]:
        allow = self.names if names is None else (self.names & set(names))
        return await self.parent.schemas_for_context(allow)

    async def schemas(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        picked = [n for n in (names if names is not None else self.names) if n in self.names]
        return await self.parent.schemas(picked)

    def meta(self, name: str) -> ToolMeta | None:
        return self.parent.meta(name) if name in self.names else None

    def meta_for_call(self, name: str, args: dict[str, Any]) -> ToolMeta | None:
        return self.parent.meta_for_call(name, args) if name in self.names else None

    def expand(self, name: str) -> tuple[bool, str]:
        if name not in self.names:
            return False, f"子代理无权使用 {name!r}"
        return self.parent.expand(name)

    @property
    def pending_expansion(self) -> list[str]:
        return [n for n in self.parent.pending_expansion if n in self.names]

    async def invoke(self, name: str, args: dict[str, Any]) -> ToolResult:
        meta = self.meta_for_call(name, args)
        if meta is None:
            return self._forbidden(name)
        if self.gate is not None:
            ok, reason = await self.gate.check(meta, args)
            if not ok:
                return ToolResult(ok=False, error=reason)
        result = await self.parent.invoke_ungated(name, args)
        settle = getattr(self.gate, "settle", None)
        if callable(settle):
            settle(meta, args, result)
        return result

    async def invoke_ungated(self, name: str, args: dict[str, Any]) -> ToolResult:
        if name not in self.names:
            return self._forbidden(name)
        return await self.parent.invoke_ungated(name, args)

    def _forbidden(self, name: str) -> ToolResult:
        allowed = ", ".join(sorted(self.names)) or "（无）"
        return ToolResult(ok=False, error=f"子代理无权调用 {name!r}。可用：{allowed}")

    async def health(self) -> dict[str, ProviderHealth]:
        return await self.parent.health()

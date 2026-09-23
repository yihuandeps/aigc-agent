"""M5 Permission Gate —— 拦截工具调用，按副作用等级放行/询问/拒绝。

半自动定位下的硬规则：**L-external 永远不提供自动放行选项。**
它是不可逆的对外动作，与其冒险不如多点一次确认。

Cost Guard 也挂在这里：L-compute 的表格里写的是「预算内放行，超限询问」，
"预算内"这三个字就是在闸门处判的。之前 COMPUTE 一律 ALLOW、注释说"护栏在
Cost Guard"，而 Cost Guard 没接任何地方 —— 等于没有。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from ..events.bus import EventBus, EventType
from ..model.budget import CostGuard
from ..tools.provider import PermissionLevel, ToolMeta


def _cost_units(meta: ToolMeta, args: dict[str, Any]) -> int:
    """这次调用要记几次。批量工具看它声明的数组参数有几项，普通工具就是 1。"""
    if not meta.cost_units_arg:
        return 1
    value = args.get(meta.cost_units_arg)
    return max(1, len(value)) if isinstance(value, (list, tuple)) else 1


def _charge(meta: ToolMeta, args: dict[str, Any]) -> tuple[int, float, float | None]:
    """这次调用记多少：(次数, 视频秒数, 金额)。provider 给了预估用预估，没有就只按次数。"""
    est = meta.estimate or {}
    try:
        units = int(est.get("units") or 0) or _cost_units(meta, args)
    except (TypeError, ValueError):
        units = _cost_units(meta, args)
    try:
        seconds = float(est.get("seconds") or 0.0)
    except (TypeError, ValueError):
        seconds = 0.0
    money = est.get("money")
    try:
        money = float(money) if money is not None else None
    except (TypeError, ValueError):
        money = None
    return max(1, units), max(0.0, seconds), money


class Decision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


# 默认策略。可按 会话 / 项目 / 全局 三级覆盖。
DEFAULT_POLICY: dict[PermissionLevel, Decision] = {
    PermissionLevel.READ: Decision.ALLOW,
    PermissionLevel.COMPUTE: Decision.ALLOW,  # 预算内放行；超限由下面的 _budget 询问
    PermissionLevel.WRITE: Decision.ALLOW,
    PermissionLevel.EXTERNAL: Decision.ASK,
}

# 询问器：收到 (meta, args) 返回 True 放行。CLI 里接终端输入，Web 里接人审台。
Asker = Callable[[ToolMeta, dict[str, Any]], Awaitable[bool]]


class PermissionGate:
    def __init__(
        self,
        bus: EventBus,
        policy: dict[PermissionLevel, Decision] | None = None,
        asker: Asker | None = None,
        guard: CostGuard | None = None,
    ) -> None:
        self.bus = bus
        self.policy = {**DEFAULT_POLICY, **(policy or {})}
        self.asker = asker
        self.guard = guard
        self._session_allow: set[str] = set()  # 本次会话已提权的工具
        # 问人必须串行。批量渲染改成并发执行后，多个调用可能同时触发询问，
        # 并发问会让终端提示交错、人看不清在批准什么 —— dispatcher 的
        # 「权限串行、执行并发」在调用方层面的同款约束，这里在闸门内兜底。
        self._ask_lock = asyncio.Lock()

    def grant_for_session(self, tool_name: str) -> None:
        """临时提权：本次会话内该工具不再询问。"""
        self._session_allow.add(tool_name)

    async def check(self, meta: ToolMeta, args: dict[str, Any]) -> tuple[bool, str]:
        ok, why = await self._policy(meta, args)
        if not ok:
            return ok, why
        # 权限过了再问预算。提权只免"问不问"，不免"花不花得起"。
        return await self._budget(meta, args, why)

    async def _policy(self, meta: ToolMeta, args: dict[str, Any]) -> tuple[bool, str]:
        # 会话提权只免 L-write / L-compute 的询问。L-external（含按参数提上来的，比如改
        # Agent 自己的配置）永远逐次问 —— 否则提权一次 fs_write，改配置也跟着静默放行了
        if meta.name in self._session_allow and meta.permission is not PermissionLevel.EXTERNAL:
            return True, "本次会话已提权"

        decision = self.policy.get(meta.permission, Decision.ASK)

        if decision is Decision.ALLOW:
            return True, ""

        if decision is Decision.DENY:
            await self.bus.emit(
                EventType.PERMISSION_DENY, tool=meta.name, level=meta.permission.value
            )
            return False, f"策略禁止调用 {meta.name}（{meta.permission.value}）"

        # ASK
        await self.bus.emit(
            EventType.PERMISSION_ASK, tool=meta.name, level=meta.permission.value, args=args
        )
        if self.asker is None:
            await self.bus.emit(
                EventType.PERMISSION_DENY,
                tool=meta.name,
                level=meta.permission.value,
                reason="无人可询问",
            )
            return False, f"{meta.name} 需要人工确认，但当前没有可用的询问入口"

        async with self._ask_lock:
            approved = await self.asker(meta, args)
        if not approved:
            await self.bus.emit(
                EventType.PERMISSION_DENY, tool=meta.name, reason="用户拒绝"
            )
            return False, f"用户拒绝执行 {meta.name}"
        return True, "用户已确认"

    async def _budget(
        self, meta: ToolMeta, args: dict[str, Any], why: str
    ) -> tuple[bool, str]:
        """L-compute 且带 cost_kind 的工具，放行前先过 Cost Guard 的次数/金额口径。

        超限时按 on_exceed：
          pause_and_ask —— 问人，人点头就**只放行这一次**（加一次额度，不改配置）
          其它          —— 直接拒绝
        没有询问入口时同样拒绝：不可逆的花钱动作不该在无人看着时静默放行。
        """
        if (
            self.guard is None
            or meta.permission is not PermissionLevel.COMPUTE
            or not meta.cost_kind
        ):
            return True, why

        # 这次调用要花多少：provider 按参数给的预估（段数 / 视频秒数 / 金额），没有就只按
        # 次数算。批量工具一次生成 N 个就要记 N 次，否则 gen_videos 跑 20 段只算 1 次，
        # 预算护栏等于没有（2026-09-19）；秒数和金额在**放行前**就算进去 —— 并发的一批
        # 完成前互相看不见，事后记账拦不住一整批（2026-09-23 审查）。
        units, seconds, money = _charge(meta, args)
        verdict = self.guard.check(meta.cost_kind, units=units, money=money or 0.0, seconds=seconds)
        if verdict:
            self.guard.record_call(meta.cost_kind, n=units, seconds=seconds, money=money)
            return True, why

        await self.bus.emit(
            EventType.BUDGET_EXCEEDED,
            tool=meta.name,
            kind=meta.cost_kind,
            reason=verdict.reason,
            dimension=verdict.dimension,
            level=verdict.level,
            usage=self.guard.usage.brief(),
        )
        if self.guard.on_exceed != "pause_and_ask" or self.asker is None:
            await self.bus.emit(
                EventType.PERMISSION_DENY, tool=meta.name, reason=f"预算护栏：{verdict.reason}"
            )
            return False, f"预算护栏拦下 {meta.name}：{verdict.reason}"

        asked = meta.model_copy(
            update={
                "summary": f"⚠ 预算护栏：超出开工时确认的额度 —— {verdict.reason}",
                "budget_ask": True,
                "budget_money": verdict.money,
            }
        )
        async with self._ask_lock:
            approved = await self.asker(asked, args)
        if not approved:
            await self.bus.emit(
                EventType.PERMISSION_DENY, tool=meta.name, reason="超预算，用户拒绝"
            )
            return False, f"预算护栏拦下 {meta.name}：{verdict.reason}（用户未放行）"

        self.guard.allow_more(meta.cost_kind, n=units, seconds=seconds)
        if money:
            self.guard.allow_more_money(money)
        self.guard.record_call(meta.cost_kind, n=units, seconds=seconds, money=money)
        return True, "超预算，用户已确认本次放行"

    def settle(self, meta: ToolMeta, args: dict[str, Any], result: Any) -> None:
        """执行完结算：放行时记的账里，**没真花出去**的退回来。

        之前闸门一放行就计次，模型切换挂起问人、参考图门拦下、提交被服务端拒收这些
        一分钱没花的调用也占着额度，重新调一次再记一遍 —— 次数护栏误报（2026-09-23 审查）。
        工具在 result.meta 里说明：charged=False（整次没花）或 refund_units / refund_seconds /
        refund_money（批量里没花出去的那部分）。挂起问人一律全退。
        """
        if (
            self.guard is None
            or meta.permission is not PermissionLevel.COMPUTE
            or not meta.cost_kind
            or result is None
        ):
            return
        rmeta = getattr(result, "meta", None) or {}
        units, seconds, money = _charge(meta, args)
        if getattr(result, "suspend", False) or rmeta.get("charged") is False:
            self.guard.refund(meta.cost_kind, n=units, seconds=seconds, money=money)
            return
        ru = int(rmeta.get("refund_units") or 0)
        rs = float(rmeta.get("refund_seconds") or 0.0)
        rm = rmeta.get("refund_money")
        if ru or rs or rm:
            self.guard.refund(
                meta.cost_kind, n=min(ru, units), seconds=min(rs, seconds),
                money=float(rm) if rm else None,
            )

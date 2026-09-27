"""M5 Permission Gate —— 拦截工具调用，按副作用等级放行/询问/拒绝。

半自动定位下的硬规则：**L-external 永远不提供自动放行选项。**
它是不可逆的对外动作，与其冒险不如多点一次确认。

Cost Guard 也挂在这里：L-compute 的表格里写的是「预算内放行，超限询问」，
"预算内"这三个字就是在闸门处判的。之前 COMPUTE 一律 ALLOW、注释说"护栏在
Cost Guard"，而 Cost Guard 没接任何地方 —— 等于没有。
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..events.bus import EventBus, EventType
from ..model.budget import CostGuard
from ..tools.provider import PermissionLevel, ToolMeta


@dataclass
class BatchPass:
    """人确认过的一整批花费（2026-09-26 用户定的：渲参考图、渲每一集之前整批报价，确认一次）。

    之前超出额度后每张图、每段视频各问一次 y/N（一套 174 张的参考图会弹一百多次），答 N 只拒
    那一张、其余照问。现在调用方先把这一批的最坏情况（含质检重生成）报给人，人确认一次，
    批内的调用照样过闸门、照样记账，只是超出额度时在这张单子的范围内不再逐次问人。
    单子用完（比最坏情况还多）才回到逐次询问：那时人答 y，这一批剩下的都不再问；答 N，
    整批停下，后面的一个都不发。

    放在 contextvar 里（batch_scope）：只对这一批自己发起的调用生效，两集并行渲互不相干。
    """

    label: str
    units: dict[str, int] = field(default_factory=dict)  # 类别 → 这批最多还能放行几次
    seconds: float = 0.0
    money: float | None = None  # None = 目录没填单价，金额不在单子里
    go_on: bool = False  # 批内被问到时人答了 y：这一批剩下的不再问
    stopped: bool = False  # 批内被问到时人答了 N：整批停

    def covers(self, kind: str, units: int, seconds: float, money: float | None,
               money_verdict: bool) -> bool:
        if self.go_on:
            return True
        if self.units.get(kind, 0) < units or self.seconds + 1e-6 < seconds:
            return False
        if money_verdict:
            # 撞的是金额线：单子里没有金额（没单价）就不替人拍板，照常问
            return self.money is not None and self.money + 1e-6 >= (money or 0.0)
        return True

    def take(self, kind: str, units: int, seconds: float, money: float | None) -> None:
        self.units[kind] = max(0, self.units.get(kind, 0) - units)
        self.seconds = max(0.0, self.seconds - seconds)
        if self.money is not None and money:
            self.money = max(0.0, self.money - money)

    def give_back(self, kind: str, units: int, seconds: float, money: float | None) -> None:
        self.units[kind] = self.units.get(kind, 0) + units
        self.seconds += seconds
        if self.money is not None and money:
            self.money += money


_BATCH: contextvars.ContextVar[BatchPass | None] = contextvars.ContextVar(
    "batch_pass", default=None
)


@contextlib.contextmanager
def batch_scope(p: BatchPass | None) -> Iterator[BatchPass | None]:
    """这一段代码（含它创建的并发任务）发起的花钱调用都按这张单子过闸门。p=None 不开。"""
    token = _BATCH.set(p)
    try:
        yield p
    finally:
        _BATCH.reset(token)


def current_batch() -> BatchPass | None:
    return _BATCH.get()


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
        await self.bus.emit(
            EventType.PERMISSION_GRANT, tool=meta.name, level=meta.permission.value,
            reason="用户确认",
        )
        return True, "用户已确认"

    async def _budget(
        self, meta: ToolMeta, args: dict[str, Any], why: str
    ) -> tuple[bool, str]:
        """L-compute 且带 cost_kind 的工具，放行前先过 Cost Guard 的次数/金额口径。

        超限时按 on_exceed：
          pause_and_ask —— 问人，人点头就**只放行这一次**（加一次额度，不改配置）
          其它          —— 直接拒绝
        没有询问入口时同样拒绝：不可逆的花钱动作不该在无人看着时静默放行。

        按 cost_kind 判、不按等级判：花钱的工具被 permission_for 按参数提到 L-external
        （比如放开画面禁字）时，人点了头也照样计次计秒 —— 之前只认 L-compute，
        一提权就连预算护栏一起跳过了（2026-09-26）。
        """
        if self.guard is None or not meta.cost_kind:
            return True, why

        # 这次调用要花多少：provider 按参数给的预估（段数 / 视频秒数 / 金额），没有就只按
        # 次数算。批量工具一次生成 N 个就要记 N 次，否则 gen_videos 跑 20 段只算 1 次，
        # 预算护栏等于没有（2026-09-19）；秒数和金额在**放行前**就算进去 —— 并发的一批
        # 完成前互相看不见，事后记账拦不住一整批（2026-09-23 审查）。
        units, seconds, money = _charge(meta, args)
        kind = meta.cost_kind
        bp = current_batch()
        if bp is not None and bp.stopped:
            return False, (
                f"预算护栏拦下 {meta.name}：这一批（{bp.label}）已经被你叫停，后面的不再发"
            )
        verdict = self.guard.check(kind, units=units, money=money or 0.0, seconds=seconds)
        if verdict:
            self.guard.record_call(kind, n=units, seconds=seconds, money=money)
            if bp is not None:
                bp.take(kind, units, seconds, money)
            return True, why
        # 超出额度，但人已经整批确认过（报价含最坏情况）：在单子范围内照常放行、照常记账
        if bp is not None and bp.covers(kind, units, seconds, money, verdict.money):
            self.guard.record_call(kind, n=units, seconds=seconds, money=money)
            bp.take(kind, units, seconds, money)
            return True, f"整批已确认（{bp.label}）"

        await self.bus.emit(
            EventType.BUDGET_EXCEEDED,
            tool=meta.name,
            kind=kind,
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

        summary = f"⚠ 预算护栏：超出开工时确认的额度 —— {verdict.reason}"
        if bp is not None:
            summary += (
                f"\n（这一批「{bp.label}」已经超出你确认的最坏情况。回 y：这一批剩下的都不再问；"
                "回 N：整批停下，后面的一个都不发）"
            )
        asked = meta.model_copy(
            update={"summary": summary, "budget_ask": True, "budget_money": verdict.money}
        )
        async with self._ask_lock:
            approved = await self.asker(asked, args)
        if not approved:
            if bp is not None:
                bp.stopped = True  # 中途答 N：整批停（2026-09-26 用户定的）
            await self.bus.emit(
                EventType.PERMISSION_DENY, tool=meta.name, reason="超预算，用户拒绝"
            )
            return False, f"预算护栏拦下 {meta.name}：{verdict.reason}（用户未放行）"
        await self.bus.emit(
            EventType.PERMISSION_GRANT, tool=meta.name, reason=f"超预算放行：{verdict.reason}"
        )
        if bp is not None:
            bp.go_on = True  # 批内答 y：这一批剩下的不再逐个问

        self.guard.allow_more(kind, n=units, seconds=seconds)
        if money:
            self.guard.allow_more_money(money)
        self.guard.record_call(kind, n=units, seconds=seconds, money=money)
        return True, "超预算，用户已确认本次放行"

    async def confirm_batch(self, tool: str, text: str, args: dict[str, Any]) -> bool | None:
        """整批报价问人一次（渲参考图、渲一集之前）。返回 True 确认 / False 拒绝；
        没有询问入口（脚本 / 测试）返回 None —— 不开整批，照常逐次过闸门。"""
        if self.asker is None:
            return None
        meta = ToolMeta(
            name=tool, summary=text, permission=PermissionLevel.EXTERNAL, provider="整批报价"
        )
        await self.bus.emit(EventType.PERMISSION_ASK, tool=tool, level="batch", args=args)
        async with self._ask_lock:
            approved = bool(await self.asker(meta, args))
        await self.bus.emit(
            EventType.PERMISSION_GRANT if approved else EventType.PERMISSION_DENY,
            tool=tool,
            reason="整批报价：" + ("确认" if approved else "没有确认"),
        )
        return approved

    def settle(self, meta: ToolMeta, args: dict[str, Any], result: Any) -> None:
        """执行完结算：放行时记的账里，**没真花出去**的退回来。

        之前闸门一放行就计次，模型切换挂起问人、参考图门拦下、提交被服务端拒收这些
        一分钱没花的调用也占着额度，重新调一次再记一遍 —— 次数护栏误报（2026-09-23 审查）。
        工具在 result.meta 里说明：charged=False（整次没花）或 refund_units / refund_seconds /
        refund_money（批量里没花出去的那部分）。挂起问人一律全退。
        """
        if self.guard is None or not meta.cost_kind or result is None:
            return
        rmeta = getattr(result, "meta", None) or {}
        units, seconds, money = _charge(meta, args)
        bp = current_batch()
        if getattr(result, "suspend", False) or rmeta.get("charged") is False:
            self.guard.refund(meta.cost_kind, n=units, seconds=seconds, money=money)
            if bp is not None:
                bp.give_back(meta.cost_kind, units, seconds, money)  # 没花出去的还给这一批
            return
        ru = int(rmeta.get("refund_units") or 0)
        rs = float(rmeta.get("refund_seconds") or 0.0)
        rm = rmeta.get("refund_money")
        if ru or rs or rm:
            self.guard.refund(
                meta.cost_kind, n=min(ru, units), seconds=min(rs, seconds),
                money=float(rm) if rm else None,
            )
            if bp is not None:
                bp.give_back(
                    meta.cost_kind, min(ru, units), min(rs, seconds), float(rm) if rm else None
                )

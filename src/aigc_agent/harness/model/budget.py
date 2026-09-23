"""Cost Guard —— 花钱的刹车。

**接线之前，这道防线整个是不存在的**，而且不是"少报个数字"：

  1. `Pricing.cost()` 在 pricing 没填时返回 None → `turn_cost` 恒为 0
     → `0 >= limit` 永假 → **预算停机判定永不触发**
  2. 更要命的是，`turn_cost` 只累加**文本模型**的 token 成本。
     生图、生视频、TTS 完全不进账 —— 而这个 agent 真正烧钱的恰恰是这侧：
     一次 drama film 六段 seedance，一次 video make 六段 veo3.1。
     **就算文本 pricing 填全了，也拦不住这条路。**

所以这里按两套口径同时拦：

  · 金额  —— 有 pricing 就按钱算，这是最准的
  · 次数  —— 没 pricing 也能拦。媒体生成本来就是**按次计费**不是按 token，
             次数才是它的自然单位；而且次数口径在不知道单价时依然有效，
             这让防线不依赖"运营有没有把价格填对"

三级预算（ARCHITECTURE 横切模块 Cost Guard）：
  · 单次任务  进程内累计（usage）
  · 单个项目  台账按 project 求和（P5 起，跨会话）
  · 单日      台账按天求和（P5 起，跨会话）
台账是 CostLedger（harness/model/ledger.py），JSONL 落盘；没给台账就只剩单任务口径。

接线方式（都在 L0，不侵入业务代码）：

  · 媒体调用在 **权限闸门（M5）** 处计次：L-compute 且带 `cost_kind` 的工具，
    放行前先问这里，超了就按 on_exceed 处理。**在发起调用之前问** ——
    事后拦没有意义，钱已经花了。
  · 文本调用没有闸门可挂，从 **事件总线的 COST 事件** 里记金额与次数。
    目录里填了单价的媒体调用也走这条补金额。
  · Loop 每轮工具执行后问一次累计口径，超了**挂起交给人**，不静默降级。

宁可拦错也不要不拦：超限是挂起交给人决定，人确认后可以放行**这一次**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..events.bus import Event, EventBus, EventType
from .ledger import CostLedger, LedgerEntry, today


class Kind(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


# 各类媒体调用的默认次数上限（单次任务内）。
# 取值依据是实际链路的用量：一集短剧 6 段视频 + 7 张资产图，
# 一条短视频 6 段视频。**留一倍余量**，正常跑不会撞到，
# 跑飞了（模型反复重试、循环调用）会。可在 models.yaml 的 cost_guard.call_limits 覆盖。
DEFAULT_CALL_LIMITS: dict[str, int] = {
    Kind.IMAGE: 24,
    Kind.VIDEO: 14,
    Kind.AUDIO: 40,
}


@dataclass
class Usage:
    """一次任务内的累计用量。"""

    money: float = 0.0  # 已知单价的部分，元
    unpriced: int = 0  # 算不出钱的调用次数（文本没 pricing，或媒体目录没填单价）
    calls: dict[str, int] = field(default_factory=dict)

    def add_call(self, kind: str) -> None:
        self.calls[kind] = self.calls.get(kind, 0) + 1

    def add_cost(self, cost: float | None) -> None:
        if cost is None:
            self.unpriced += 1
        else:
            self.money += cost

    def add(self, kind: str, cost: float | None = None) -> None:
        """一次调用：计次 + 记金额。"""
        self.add_call(kind)
        self.add_cost(cost)

    def brief(self) -> str:
        parts = [f"{k} {n} 次" for k, n in sorted(self.calls.items()) if n]
        head = f"¥{self.money:.4f}" if self.money else "金额未知"
        tail = f"（{self.unpriced} 次调用无单价）" if self.unpriced else ""
        return f"{head} · {' / '.join(parts) if parts else '无调用'}{tail}"


@dataclass
class Verdict:
    ok: bool
    reason: str = ""
    # True = 金额口径超限（单次任务/项目/单日）。False = 次数口径。
    # auto 模式据此区分：次数护栏可自动放行，金额护栏永远问人 —— 钱是最后一道闸。
    money: bool = False

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class CostGuard:
    """按金额和次数两套口径、任务/项目/日三级拦。

    两套是**或**的关系：任一超限就拦。金额口径在 pricing 缺失时自动失效，
    次数口径始终有效 —— 这是有意的，防线不该依赖配置填得全不全。
    """

    money_limit: float | None = None
    call_limits: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_CALL_LIMITS))
    usage: Usage = field(default_factory=Usage)
    on_exceed: str = "pause_and_ask"  # pause_and_ask | deny
    # 人在超限时批准的加量：kind -> 额外次数。只加这一次，不改配置。
    extra: dict[str, int] = field(default_factory=dict)
    # 人在金额超限时批准的临时追加（元）。三级金额上限都加上它 —— 人说了
    # "继续"就是继续，不该换个口径再拦一次。reset() 清零，不改配置。
    extra_money: float = 0.0
    # ---- 跨会话（P5）----
    project_limit: float | None = None
    daily_limit: float | None = None
    daily_call_limits: dict[str, int] = field(default_factory=dict)
    ledger: CostLedger | None = None
    project_id: str = "default"
    session_id: str = ""

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        ledger: CostLedger | None = None,
        project_id: str = "default",
        session_id: str = "",
    ) -> CostGuard:
        """从 ModelsConfig.cost_guard 建。没填的项用默认值，金额上限允许为空。"""
        limits = dict(DEFAULT_CALL_LIMITS)
        limits.update(
            {str(k): int(v) for k, v in (getattr(cfg, "call_limits", None) or {}).items()}
        )
        daily = {
            str(k): int(v) for k, v in (getattr(cfg, "daily_call_limits", None) or {}).items()
        }
        return cls(
            money_limit=getattr(cfg, "per_task_limit", None),
            call_limits=limits,
            on_exceed=str(getattr(cfg, "on_exceed", "pause_and_ask") or "pause_and_ask"),
            project_limit=getattr(cfg, "per_project_limit", None),
            daily_limit=getattr(cfg, "daily_limit", None),
            daily_call_limits=daily,
            ledger=ledger,
            project_id=project_id or "default",
            session_id=session_id,
        )

    # ---------- 接线 ----------

    def attach(self, bus: EventBus) -> None:
        """订阅 COST 事件。文本调用没有闸门可挂，只能从这里数。"""
        bus.subscribe(self._on_event)

    def _on_event(self, ev: Event) -> None:
        if ev.type is not EventType.COST:
            return
        kind = str(ev.data.get("modality") or Kind.TEXT)
        cost = ev.data.get("cost")
        # 媒体调用已经在闸门处计过次，这里只补金额；文本调用两样都在这记。
        if kind == Kind.TEXT:
            self.usage.add_call(kind)
        self.usage.add_cost(cost)
        self._ledger(
            kind,
            cost,
            calls=1 if kind == Kind.TEXT else 0,
            tokens=int(ev.data.get("prompt_tokens") or 0)
            + int(ev.data.get("completion_tokens") or 0),
            role=str(ev.data.get("role") or ""),
            model=str(ev.data.get("model") or ""),
        )

    def _ledger(self, kind: str, cost: float | None, calls: int, **extra: Any) -> None:
        if self.ledger is None:
            return
        self.ledger.append(
            LedgerEntry(
                project=self.project_id,
                session=self.session_id,
                kind=kind,
                cost=cost,
                calls=calls,
                **extra,
            )
        )

    def record(self, kind: str, cost: float | None = None) -> None:
        self.usage.add(kind, cost)
        self._ledger(kind, cost, calls=1)

    def record_call(self, kind: str, n: int = 1) -> None:
        """闸门放行媒体调用时记 n 次。金额（若目录有单价）由 COST 事件补。

        n > 1 是批量工具：一次 gen_videos 生成几段就记几次，不然预算护栏拦不住。
        """
        for _ in range(max(1, int(n))):
            self.usage.add_call(kind)
        self._ledger(kind, None, calls=max(1, int(n)))

    # ---------- 判定 ----------

    def limit_for(self, kind: str) -> int | None:
        base = self.call_limits.get(kind)
        return None if base is None else base + self.extra.get(kind, 0)

    def check(self, kind: str = "", units: int = 1) -> Verdict:
        """**在发起调用之前**问。事后拦没有意义 —— 钱已经花了。

        顺序：单任务 → 单日 → 单项目。哪一级先超就报哪一级。
        units > 1 是批量调用：要一次性放得下这么多，不能放一半进去。
        """
        units = max(1, int(units))
        bonus = self.extra_money
        if self.money_limit is not None and self.usage.money >= self.money_limit + bonus:
            return Verdict(
                False,
                f"已花 ¥{self.usage.money:.4f}，达到单次任务上限 ¥{self.money_limit + bonus:.2f}",
                money=True,
            )
        if kind:
            limit = self.limit_for(kind)
            used = self.usage.calls.get(kind, 0)
            if limit is not None and used + units > limit:
                more = f"，本次要 {units} 个" if units > 1 else ""
                return Verdict(
                    False,
                    f"{kind} 已调用 {used} 次{more}，达到单次任务上限 {limit} 次。"
                    "正常链路用不到这么多，多半是循环调用或反复重试。",
                )
        if self.ledger is not None:
            day = today()
            spent_today = self.ledger.money(day=day)
            if self.daily_limit is not None and spent_today >= self.daily_limit + bonus:
                return Verdict(
                    False,
                    f"今日已花 ¥{spent_today:.4f}，达到单日上限 ¥{self.daily_limit + bonus:.2f}",
                    money=True,
                )
            spent_project = self.ledger.money(project=self.project_id)
            if self.project_limit is not None and spent_project >= self.project_limit + bonus:
                return Verdict(
                    False,
                    f"项目 {self.project_id} 已花 ¥{spent_project:.4f}，"
                    f"达到单项目上限 ¥{self.project_limit + bonus:.2f}",
                    money=True,
                )
            if kind and kind in self.daily_call_limits:
                used = self.ledger.calls(kind, day=day)
                limit = self.daily_call_limits[kind]
                if used >= limit + self.extra.get(f"day:{kind}", 0):
                    return Verdict(
                        False, f"{kind} 今日已调用 {used} 次，达到单日上限 {limit} 次"
                    )
        return Verdict(True)

    def allow_more(self, kind: str, n: int = 1) -> None:
        """人确认后放行：只给这一类加 n 次额度，不动配置。日级次数同样临时加。"""
        self.extra[kind] = self.extra.get(kind, 0) + n
        self.extra[f"day:{kind}"] = self.extra.get(f"day:{kind}", 0) + n

    def allow_more_money(self, amount: float) -> float:
        """人确认后临时追加金额（元），三级上限一起抬。返回累计追加。

        之前金额超限只有 /budget reset 一条出口，而且用户不知道有它 ——
        实测一个会话里 50 多轮每轮只跑一次迭代就停，人说"别管花费继续写"也没用。
        """
        self.extra_money += max(0.0, float(amount))
        return self.extra_money

    def reset(self) -> None:
        self.usage = Usage()
        self.extra = {}
        self.extra_money = 0.0

    @property
    def blind(self) -> bool:
        """金额口径是否失效（pricing 没配）。用来提醒运营。"""
        return self.money_limit is None or self.usage.unpriced > 0

    def brief(self) -> str:
        limits = " / ".join(
            f"{k} ≤{self.limit_for(k)}" for k in sorted(self.call_limits) if self.limit_for(k)
        )
        money = f"¥{self.money_limit:.2f}" if self.money_limit is not None else "未设"
        if self.extra_money:
            money += f"（临时追加 ¥{self.extra_money:.0f}）"
        text = f"{self.usage.brief()} · 上限 金额 {money} · 次数 {limits}"
        if self.ledger is not None:
            s = self.ledger.summary(self.project_id)
            daily = f"¥{self.daily_limit:.2f}" if self.daily_limit is not None else "未设"
            proj = f"¥{self.project_limit:.2f}" if self.project_limit is not None else "未设"
            calls = " / ".join(f"{k} {n}" for k, n in sorted(s["today_calls"].items())) or "无"
            text += (
                f"\n今日 ¥{s['today_money']:.4f}（上限 {daily}）· 媒体 {calls} · "
                f"项目 {self.project_id} ¥{s['project_money']:.4f}（上限 {proj}）"
            )
        return text

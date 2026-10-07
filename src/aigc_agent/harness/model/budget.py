"""Cost Guard —— 花钱的刹车。

**接线之前，这道防线整个是不存在的**，而且不是"少报个数字"：

  1. `Pricing.cost()` 在 pricing 没填时返回 None → `turn_cost` 恒为 0
     → `0 >= limit` 永假 → **预算停机判定永不触发**
  2. 更要命的是，`turn_cost` 只累加**文本模型**的 token 成本。
     生图、生视频、TTS 完全不进账 —— 而这个 agent 真正烧钱的恰恰是这侧：
     一次 drama film 六段 seedance，一次 video make 六段 veo3.1。
     **就算文本 pricing 填全了，也拦不住这条路。**

所以这里按两套口径同时拦：

  · 金额  —— 有 pricing 就按钱算，这是最准的。文本 provider 没配 pricing 的按保守估价
             计入（text.pricing_estimate），各处标「估算」—— 之前按 0 记，gemini 三天约
             ¥90–100 金额上限看不见（2026-10-07，09-29 审查 2.1）
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


# 各类媒体调用的默认次数上限（单次开工内）。
# 2026-09-23 审查：之前按「一集 6 段视频 + 7 张图」定成 14 / 24，而一集 4 分钟是 16–28 段、
# 一套参考图包 40 张上下 —— 每集渲到第 15 段必停，还提示「多半是循环调用」，真实会话里
# 36 分钟弹了 40 次，人被训练成一路点「是」。现在按一集的规格 + 质检重生成留余量；
# 开工时会让人确认/改（CLI），可在 models.yaml 的 cost_guard.call_limits 覆盖。
DEFAULT_CALL_LIMITS: dict[str, int] = {
    Kind.IMAGE: 80,
    Kind.VIDEO: 60,
    Kind.AUDIO: 40,
}


@dataclass
class Usage:
    """一次任务内的累计用量。"""

    money: float = 0.0  # 算得出钱的部分，元（含下面按估价算的）
    # money 里按保守估价算的部分：文本 provider 没配单价（2026-10-07）。照样计入、照样拦，
    # 只是展示时要说清多少是估算、是哪几家。estimated_by：provider -> 估算金额
    estimated: float = 0.0
    estimated_by: dict[str, float] = field(default_factory=dict)
    unpriced: int = 0  # 连估价都算不出钱的调用次数（文本没 pricing 又关了估价）
    calls: dict[str, int] = field(default_factory=dict)
    # 生成的视频总秒数。**不依赖单价**的视频刹车：媒体目录没填价格时金额口径看不见视频，
    # 段数又不反映长短，秒数才是视频花费的自然单位（2026-09-23 审查）
    seconds: float = 0.0

    def add_call(self, kind: str) -> None:
        self.calls[kind] = self.calls.get(kind, 0) + 1

    def add_cost(
        self, cost: float | None, estimate: float | None = None, source: str = ""
    ) -> None:
        """记一次调用的金额。cost 按真实单价算；没有时用 estimate（按保守估价算的，source 是
        哪家 provider），两样都没有才记一次「无单价」。"""
        if cost is not None:
            self.money += cost
        elif estimate is not None:
            self.money += estimate
            self.estimated += estimate
            key = source or "?"
            self.estimated_by[key] = self.estimated_by.get(key, 0.0) + estimate
        else:
            self.unpriced += 1

    def add(self, kind: str, cost: float | None = None) -> None:
        """一次调用：计次 + 记金额。"""
        self.add_call(kind)
        self.add_cost(cost)

    def estimate_note(self) -> str:
        """「（其中估算 ¥3.2000：gemini 没配单价）」；金额里没有估算的部分返回空串。"""
        if not self.estimated_by:
            return ""
        who = "、".join(sorted(self.estimated_by))
        return f"（其中估算 ¥{self.estimated:.4f}：{who} 没配单价）"

    def brief(self) -> str:
        parts = [f"{k} {n} 次" for k, n in sorted(self.calls.items()) if n]
        if self.seconds:
            parts.append(f"视频 {self.seconds:.0f} 秒")
        head = f"¥{self.money:.4f}" if self.money or self.estimated_by else "金额未知"
        tail = f"（{self.unpriced} 次调用无单价）" if self.unpriced else ""
        parts_text = " / ".join(parts) if parts else "无调用"
        return f"{head}{self.estimate_note()} · {parts_text}{tail}"


@dataclass
class Verdict:
    ok: bool
    reason: str = ""
    # True = 金额口径超限（单次任务/项目/单日）。False = 次数 / 秒数口径。
    money: bool = False
    # 哪一维撞线：money / calls / seconds；哪一级：task（本次开工）/ day / project
    dimension: str = ""
    level: str = ""

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class CostGuard:
    """按金额和次数两套口径、任务/项目/日三级拦。

    两套是**或**的关系：任一超限就拦。次数口径始终有效 —— 这是有意的，防线不该依赖
    配置填得全不全。金额口径：文本 provider 没配 pricing 的按保守估价计入（标「估算」，
    2026-10-07），不再因此失效；媒体目录没填单价的看不见，靠次数 / 秒数口径管。
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
    # ---- 视频秒数口径（2026-09-23）----
    seconds_limit: float | None = None  # 本次开工
    daily_seconds_limit: float | None = None  # 单日（跨会话，靠台账）
    extra_seconds: float = 0.0
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
            seconds_limit=getattr(cfg, "seconds_limit", None),
            daily_seconds_limit=getattr(cfg, "daily_seconds_limit", None),
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
        d = ev.data
        kind = str(d.get("modality") or Kind.TEXT)
        cost = d.get("cost")
        # 没配单价的文本 provider：网关按保守估价另给一份 est_cost（cost 仍是 None）。
        # 照样计入金额、照样进台账，只是标「估算」（2026-10-07，09-29 审查 2.1）
        est = d.get("est_cost") if cost is None else None
        estimate = None if est is None else float(est)
        provider = str(d.get("provider") or "")
        # 媒体调用已经在闸门处计过次，这里只补金额；文本调用两样都在这记。
        if kind == Kind.TEXT:
            self.usage.add_call(kind)
        self.usage.add_cost(cost, estimate, source=provider or str(d.get("model") or ""))
        prompt = int(d.get("prompt_tokens") or 0)
        completion = int(d.get("completion_tokens") or 0)
        self._ledger(
            kind,
            cost if cost is not None else estimate,
            calls=1 if kind == Kind.TEXT else 0,
            estimated=cost is None and estimate is not None,
            tokens=prompt + completion,
            # 输入、输出分开记：以后补了真实单价，估算的那几笔能按 token 回算
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=int(d.get("cached_tokens") or 0),
            role=str(d.get("role") or ""),
            provider=provider,
            model=str(d.get("model") or ""),
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

    def record_call(
        self, kind: str, n: int = 1, seconds: float = 0.0, money: float | None = None
    ) -> None:
        """闸门放行媒体调用时记账：n 次、视频 seconds 秒、预估金额 money（目录有单价时）。

        n > 1 是批量工具：一次 gen_videos 生成几段就记几次，不然预算护栏拦不住。
        金额在**放行时**按目录单价记（提交即计费），不再等生成完从 COST 事件补 ——
        并发的一批在完成前都看不见彼此的花费，事后记账拦不住一整批。
        """
        n = max(1, int(n))
        for _ in range(n):
            self.usage.add_call(kind)
        self.usage.seconds += max(0.0, float(seconds or 0.0))
        if money is not None:
            self.usage.money += max(0.0, float(money))
        self._ledger(kind, money, calls=n, seconds=float(seconds or 0.0))

    def refund(
        self, kind: str, n: int = 0, seconds: float = 0.0, money: float | None = None
    ) -> None:
        """放行后发现没真花钱（被拦下、挂起问人、提交被拒）：把记的账退回来。"""
        n = max(0, int(n))
        if n:
            self.usage.calls[kind] = max(0, self.usage.calls.get(kind, 0) - n)
        self.usage.seconds = max(0.0, self.usage.seconds - max(0.0, float(seconds or 0.0)))
        if money:
            self.usage.money = max(0.0, self.usage.money - float(money))
        if n or seconds or money:
            self._ledger(
                kind, -float(money) if money else None, calls=-n, seconds=-float(seconds or 0.0)
            )

    # ---------- 判定 ----------

    def limit_for(self, kind: str) -> int | None:
        base = self.call_limits.get(kind)
        return None if base is None else base + self.extra.get(kind, 0)

    def check(
        self, kind: str = "", units: int = 1, money: float = 0.0, seconds: float = 0.0
    ) -> Verdict:
        """**在发起调用之前**问。事后拦没有意义 —— 钱已经花了。

        顺序：本次开工 → 单日 → 单项目。哪一级先超就报哪一级。
        units / money / seconds 是**这次调用**的预估（批量调用按总量）：
        要一次性放得下，不能放一半进去。
        """
        units = max(1, int(units))
        money = max(0.0, float(money or 0.0))
        seconds = max(0.0, float(seconds or 0.0))
        bonus = self.extra_money
        this = f"（这次预估 ¥{money:.2f}）" if money else ""
        cap = (self.money_limit or 0.0) + bonus
        # 已花的钱里有按估价算的就说清楚（多少、哪几家）：人决定追不追加时要知道这部分不是真账
        if self.money_limit is not None and _over(self.usage.money, money, cap):
            return Verdict(
                False,
                f"本次开工已花 ¥{self.usage.money:.4f}{self.usage.estimate_note()}{this}，"
                f"超过上限 ¥{self.money_limit + bonus:.2f}",
                money=True, dimension="money", level="task",
            )
        if kind:
            limit = self.limit_for(kind)
            used = self.usage.calls.get(kind, 0)
            if limit is not None and used + units > limit:
                more = f"，这次要 {units} 个" if units > 1 else ""
                return Verdict(
                    False,
                    f"{_KIND_LABEL.get(kind, kind)}本次开工已用 {used} 次{more}，"
                    f"超过确认的上限 {limit} 次",
                    dimension="calls", level="task",
                )
        if seconds and self.seconds_limit is not None:
            cap = self.seconds_limit + self.extra_seconds
            if self.usage.seconds + seconds > cap:
                return Verdict(
                    False,
                    f"视频本次开工已生成 {self.usage.seconds:.0f} 秒，这次要 {seconds:.0f} 秒，"
                    f"超过确认的上限 {cap:.0f} 秒",
                    dimension="seconds", level="task",
                )
        if self.ledger is not None:
            day = today()
            spent_today = self.ledger.money(day=day)
            if self.daily_limit is not None and _over(spent_today, money, self.daily_limit + bonus):
                est = _paren(_of_which(self.ledger.estimated(day=day)))
                return Verdict(
                    False,
                    f"今日已花 ¥{spent_today:.4f}{est}{this}，达到单日上限 "
                    f"¥{self.daily_limit + bonus:.2f}",
                    money=True, dimension="money", level="day",
                )
            spent_project = self.ledger.money(project=self.project_id)
            if self.project_limit is not None and _over(
                spent_project, money, self.project_limit + bonus
            ):
                est = _paren(_of_which(self.ledger.estimated(project=self.project_id)))
                return Verdict(
                    False,
                    f"项目「{self.project_id}」已花 ¥{spent_project:.4f}{est}{this}，"
                    f"达到单项目上限 ¥{self.project_limit + bonus:.2f}",
                    money=True, dimension="money", level="project",
                )
            if kind and kind in self.daily_call_limits:
                used = self.ledger.calls(kind, day=day)
                limit = self.daily_call_limits[kind]
                if used + units > limit + self.extra.get(f"day:{kind}", 0):
                    return Verdict(
                        False,
                        f"{_KIND_LABEL.get(kind, kind)}今日已调用 {used} 次，"
                        f"达到单日上限 {limit} 次",
                        dimension="calls", level="day",
                    )
            if seconds and self.daily_seconds_limit is not None:
                used_s = self.ledger.seconds(day=day)
                cap = self.daily_seconds_limit + self.extra_seconds
                if used_s + seconds > cap:
                    return Verdict(
                        False,
                        f"视频今日已生成 {used_s:.0f} 秒，这次要 {seconds:.0f} 秒，"
                        f"达到单日上限 {cap:.0f} 秒",
                        dimension="seconds", level="day",
                    )
        return Verdict(True)

    def room(self, kind: str) -> dict[str, float | None]:
        """还剩多少额度：calls / seconds / money，本次开工和单日（金额再加单项目）取更紧的。
        None = 这一维不限。整批报价时给人看（2026-09-26）。"""

        def tighter(a: float | None, b: float | None) -> float | None:
            return b if a is None else a if b is None else min(a, b)

        calls: float | None = None
        limit = self.limit_for(kind)
        if limit is not None:
            calls = limit - self.usage.calls.get(kind, 0)
        seconds: float | None = None
        if self.seconds_limit is not None:
            seconds = self.seconds_limit + self.extra_seconds - self.usage.seconds
        money: float | None = None
        if self.money_limit is not None:
            money = self.money_limit + self.extra_money - self.usage.money
        if self.ledger is not None:
            day = today()
            if kind in self.daily_call_limits:
                calls = tighter(
                    calls,
                    self.daily_call_limits[kind] + self.extra.get(f"day:{kind}", 0)
                    - self.ledger.calls(kind, day=day),
                )
            if self.daily_seconds_limit is not None:
                seconds = tighter(
                    seconds,
                    self.daily_seconds_limit + self.extra_seconds - self.ledger.seconds(day=day),
                )
            if self.daily_limit is not None:
                money = tighter(
                    money, self.daily_limit + self.extra_money - self.ledger.money(day=day)
                )
            if self.project_limit is not None:
                money = tighter(
                    money,
                    self.project_limit + self.extra_money
                    - self.ledger.money(project=self.project_id),
                )
        return {
            "calls": None if calls is None else max(0.0, float(calls)),
            "seconds": None if seconds is None else max(0.0, float(seconds)),
            "money": None if money is None else max(0.0, float(money)),
        }

    def allow_more(self, kind: str, n: int = 1, seconds: float = 0.0) -> None:
        """人确认后放行：只给这一类加 n 次（视频再加 seconds 秒）额度，不动配置。"""
        self.extra[kind] = self.extra.get(kind, 0) + n
        self.extra[f"day:{kind}"] = self.extra.get(f"day:{kind}", 0) + n
        self.extra_seconds += max(0.0, float(seconds or 0.0))

    # ---------- 开工额度（2026-09-23：用户规则「每次开工前确认上限」）----------

    def limits(self) -> dict[str, float | int | None]:
        """当前的本次开工额度（给 CLI 展示 / 写进会话快照）。"""
        return {
            "money": self.money_limit,
            "video_calls": self.call_limits.get(Kind.VIDEO),
            "video_seconds": self.seconds_limit,
            "image_calls": self.call_limits.get(Kind.IMAGE),
        }

    def set_limits(
        self,
        money: float | None = None,
        video_calls: int | None = None,
        video_seconds: float | None = None,
        image_calls: int | None = None,
    ) -> None:
        """人确认过的本次开工额度。给了哪项改哪项。"""
        if money is not None:
            self.money_limit = float(money)
        if video_calls is not None:
            self.call_limits[Kind.VIDEO] = int(video_calls)
        if video_seconds is not None:
            self.seconds_limit = float(video_seconds)
        if image_calls is not None:
            self.call_limits[Kind.IMAGE] = int(image_calls)

    def allow_more_money(self, amount: float) -> float:
        """人确认后临时追加金额（元），三级上限一起抬。返回累计追加。

        之前金额超限只有 /budget reset 一条出口，而且用户不知道有它 ——
        实测一个会话里 50 多轮每轮只跑一次迭代就停，人说"别管花费继续写"也没用。
        """
        self.extra_money += max(0.0, float(amount))
        return self.extra_money

    def reset(self) -> None:
        """清零**本次开工**的用量（新开一段工）。台账里的单日 / 单项目累计不动。"""
        self.usage = Usage()
        self.extra = {}
        self.extra_money = 0.0
        self.extra_seconds = 0.0

    @property
    def blind(self) -> bool:
        """金额口径是否失效：没设金额上限，或有文本调用连估价都算不出钱。用来提醒运营。

        文本 provider 没配 pricing 时按保守估价计入（2026-10-07），不再算失效 —— 金额照样
        作数、照样拦，只是其中一部分是估算：多少、哪几家见 usage.estimate_note() 和 brief()。
        媒体目录没填单价的金额口径看不见，靠次数 / 秒数口径管，不在这里判。
        """
        return self.money_limit is None or self.usage.unpriced > 0

    def brief(self) -> str:
        limits = " / ".join(
            f"{k} ≤{self.limit_for(k)}" for k in sorted(self.call_limits) if self.limit_for(k)
        )
        money = f"¥{self.money_limit:.2f}" if self.money_limit is not None else "未设"
        if self.extra_money:
            money += f"（临时追加 ¥{self.extra_money:.0f}）"
        secs = ""
        if self.seconds_limit is not None:
            secs = f" · 视频 ≤{self.seconds_limit + self.extra_seconds:.0f} 秒"
        text = f"本次开工 {self.usage.brief()} · 上限 金额 {money} · 次数 {limits}{secs}"
        if self.ledger is not None:
            s = self.ledger.summary(self.project_id)
            daily = f"¥{self.daily_limit:.2f}" if self.daily_limit is not None else "未设"
            proj = f"¥{self.project_limit:.2f}" if self.project_limit is not None else "未设"
            calls = " / ".join(f"{k} {n}" for k, n in sorted(s["today_calls"].items())) or "无"
            # 金额里有按估价算的就写在括号里（「其中估算 ¥x；上限 ¥y」），没有时和原来一样
            day_est = _of_which(float(s["today_estimated"]))
            proj_est = _of_which(float(s["project_estimated"]))
            text += (
                f"\n今日 ¥{s['today_money']:.4f}（{day_est + '；' if day_est else ''}上限 {daily}）"
                f"· 媒体 {calls} · 项目 {self.project_id} ¥{s['project_money']:.4f}"
                f"（{proj_est + '；' if proj_est else ''}上限 {proj}）"
            )
        return text


_KIND_LABEL = {"image": "生图", "video": "视频", "audio": "配音/转写", "text": "文本"}


def _over(spent: float, this: float, cap: float) -> bool:
    """已经花到上限（不管这次多少都拦），或者这次的预估会把它推过上限。"""
    return spent >= cap or (this > 0 and spent + this > cap)


def _of_which(estimated: float) -> str:
    """金额里按估价算的那部分：「其中估算 ¥x」；没有返回空串。"""
    return f"其中估算 ¥{estimated:.4f}" if estimated > 0 else ""


def _paren(text: str) -> str:
    return f"（{text}）" if text else ""

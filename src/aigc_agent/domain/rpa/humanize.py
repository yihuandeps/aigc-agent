"""行为节奏拟人化 —— 保护你自己的账号。

做法是**把速度放慢并且让它不规律**，不是伪装身份：
  ✅ 随机化延迟、变距滚动、阅读停顿、会话配额、冷却期
  ❌ 指纹伪装 / 验证码识别 / 多身份轮换 / 签名伪造

后者属于对抗平台的检测机制，不做。前者既保账号也真的降低了对平台的压力
—— 一个每 200ms 翻一屏的脚本，本身就是不该有的行为。

为什么不用 `random.uniform()` 就完事：均匀分布的间隔其实很不像人。
人的操作间隔是**右偏的长尾**——大部分时候挺快，偶尔卡很久（看到感兴趣
的内容、走神、切窗口）。对数正态分布更接近，代价只是几行代码。
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field


@dataclass
class Pace:
    """节奏参数。都可以在 config/rpa.yaml 里调。

    默认值偏保守：宁可慢，不要把账号搭进去。
    """

    # 动作间隔（秒）。median 是中位数，sigma 越大长尾越明显
    action_median: float = 1.8
    action_sigma: float = 0.55
    action_min: float = 0.6
    action_max: float = 9.0

    # 滚动一屏后的停顿 —— 人要看内容
    scroll_median: float = 2.4
    scroll_sigma: float = 0.5

    # 每滚几次会停下来"读一会儿"
    read_every: tuple[int, int] = (3, 6)
    read_seconds: tuple[float, float] = (4.0, 12.0)

    # 偶尔往回滚一点（看漏了/回看），这是人很典型的动作
    scrollback_chance: float = 0.18

    # 每次滚动的像素距离，不是固定值
    scroll_distance: tuple[int, int] = (600, 1800)

    # 会话配额：够了就停，不要一直挂着
    max_items: int = 60
    max_minutes: float = 8.0

    # 页面之间的冷却
    page_cooldown: tuple[float, float] = (3.0, 9.0)


@dataclass
class SessionBudget:
    """会话配额。超了就停，并如实告诉上层为什么停。"""

    pace: Pace
    started: float = field(default_factory=time.monotonic)
    items: int = 0
    actions: int = 0

    @property
    def elapsed_minutes(self) -> float:
        return (time.monotonic() - self.started) / 60

    def exhausted(self) -> str:
        if self.items >= self.pace.max_items:
            return f"已达单次上限 {self.pace.max_items} 条"
        if self.elapsed_minutes >= self.pace.max_minutes:
            return f"已达单次时长上限 {self.pace.max_minutes:.0f} 分钟"
        return ""


def _lognormal(median: float, sigma: float, lo: float, hi: float) -> float:
    """右偏的间隔：多数时候接近 median，偶尔明显更久。"""
    import math

    v = median * math.exp(random.gauss(0, sigma))
    return max(lo, min(v, hi))


async def act_pause(pace: Pace, budget: SessionBudget | None = None) -> float:
    """一次普通动作之后的停顿。"""
    d = _lognormal(pace.action_median, pace.action_sigma, pace.action_min, pace.action_max)
    if budget:
        budget.actions += 1
    await asyncio.sleep(d)
    return d


async def human_scroll(page, pace: Pace, times: int, budget: SessionBudget | None = None) -> None:
    """拟人滚动：变距、变停顿、偶尔回滚、间歇性停下来读。

    对比机械滚动（固定 2400px + 固定 1.2s），这个的轨迹看起来杂乱得多，
    而且**总耗时明显更长** —— 这正是目的。
    """
    next_read = random.randint(*pace.read_every)

    for i in range(times):
        if budget and budget.exhausted():
            return

        dist = random.randint(*pace.scroll_distance)
        await page.mouse.wheel(0, dist)
        await asyncio.sleep(
            _lognormal(pace.scroll_median, pace.scroll_sigma, 0.8, 15.0)
        )

        # 偶尔往回滚一点：看漏了想回看，人常这么干
        if random.random() < pace.scrollback_chance:
            await page.mouse.wheel(0, -random.randint(150, 500))
            await asyncio.sleep(_lognormal(1.2, 0.4, 0.4, 4.0))

        # 间歇性"读一会儿"
        if i + 1 >= next_read:
            await asyncio.sleep(random.uniform(*pace.read_seconds))
            next_read = i + 1 + random.randint(*pace.read_every)


async def page_cooldown(pace: Pace) -> float:
    """换页/换关键词之间的冷却。连着切页是最容易被盯上的模式。"""
    d = random.uniform(*pace.page_cooldown)
    await asyncio.sleep(d)
    return d


async def move_like_reading(page, pace: Pace) -> None:
    """鼠标在内容区域轻微移动。

    人看页面时鼠标不会一直不动。这一步很便宜，聊胜于无。
    """
    try:
        w, h = 1440, 900
        for _ in range(random.randint(1, 3)):
            await page.mouse.move(
                random.randint(int(w * 0.2), int(w * 0.8)),
                random.randint(int(h * 0.25), int(h * 0.75)),
                steps=random.randint(8, 25),
            )
            await asyncio.sleep(_lognormal(0.6, 0.5, 0.15, 3.0))
    except Exception:  # noqa: BLE001 — 鼠标动作失败不影响采集
        pass


async def type_like_human(page, selector: str, text: str, pace: Pace) -> bool:
    """逐字输入，字间隔不等长，偶尔停顿。

    一次性 fill() 整个词是很明显的机器行为。
    """
    try:
        el = await page.query_selector(selector)
        if not el:
            return False
        await el.click()
        await act_pause(pace)
        for ch in text:
            await el.type(ch, delay=0)
            await asyncio.sleep(_lognormal(0.14, 0.6, 0.04, 1.2))
            if random.random() < 0.08:  # 偶尔想一下
                await asyncio.sleep(random.uniform(0.4, 1.6))
        return True
    except Exception:  # noqa: BLE001
        return False


def load_pace(profile: str = "", config_path=None) -> Pace:
    """从 config/rpa.yaml 载入节奏。profile 可选 cautious / brisk。

    配置读不到就用默认值 —— 节奏配置缺失不该让采集整个失败，
    而默认值本身就是保守的。
    """
    from pathlib import Path  # noqa: PLC0415

    import yaml  # noqa: PLC0415

    path = Path(config_path) if config_path else (
        Path(__file__).resolve().parents[3].parent / "config" / "rpa.yaml"
    )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return Pace()

    base = dict(raw.get("pace") or {})
    if profile:
        base.update((raw.get("profiles") or {}).get(profile) or {})

    fields = {f for f in Pace.__dataclass_fields__}
    kw = {}
    for k, v in base.items():
        if k not in fields:
            continue
        kw[k] = tuple(v) if isinstance(v, list) else v
    return Pace(**kw)

"""行为节奏拟人化验收。

验的是**节奏真的不规律、真的够慢、真的会自己停**——
这三条是保护账号的实际手段，不是写在注释里的愿望。
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import pytest

from aigc_agent.domain.rpa.humanize import (
    Pace,
    SessionBudget,
    _lognormal,
    act_pause,
    human_scroll,
    load_pace,
    page_cooldown,
)

CONFIG = Path(__file__).resolve().parents[1] / "config" / "rpa.yaml"


class FakePage:
    """记录滚动距离与鼠标动作，用来检查轨迹是否规律。"""

    def __init__(self) -> None:
        self.wheel: list[int] = []
        self.moves: list[tuple[int, int]] = []
        self.mouse = self

    async def wheel(self, dx: int, dy: int) -> None:  # noqa: ARG002
        self.wheel.append(dy)

    async def move(self, x: int, y: int, steps: int = 1) -> None:  # noqa: ARG002
        self.moves.append((x, y))


class _Mouse:
    def __init__(self, rec: list[int]) -> None:
        self.rec = rec

    async def wheel(self, dx: int, dy: int) -> None:  # noqa: ARG002
        self.rec.append(dy)


class Page:
    def __init__(self) -> None:
        self.scrolls: list[int] = []
        self.mouse = _Mouse(self.scrolls)


# ---------------------------------------------------------------- 分布


def test_间隔是右偏长尾不是均匀分布():
    """均匀随机的间隔其实很不像人。人是大部分快、偶尔很慢。"""
    xs = [_lognormal(2.0, 0.6, 0.5, 20.0) for _ in range(4000)]
    mean, median = statistics.mean(xs), statistics.median(xs)

    assert 1.7 < median < 2.3, f"中位数应贴近 2.0，实际 {median:.2f}"
    # 右偏的标志：均值明显大于中位数
    assert mean > median * 1.08, f"没有右偏：mean={mean:.2f} median={median:.2f}"
    # 存在长尾
    assert max(xs) > median * 3, "缺少长尾，看起来还是太规律"


def test_间隔被夹在上下界内():
    xs = [_lognormal(2.0, 1.5, 0.5, 6.0) for _ in range(2000)]
    assert min(xs) >= 0.5 and max(xs) <= 6.0


def test_每次取值都不同():
    """固定间隔是最容易被识别的模式。"""
    xs = [_lognormal(2.0, 0.5, 0.1, 30.0) for _ in range(50)]
    assert len(set(xs)) == 50


# ---------------------------------------------------------------- 滚动


async def test_滚动距离不是固定值():
    page = Page()
    pace = Pace(
        scroll_median=0.001, scroll_sigma=0.1, read_every=(99, 99),
        scroll_distance=(600, 1800), scrollback_chance=0.0,
    )
    await human_scroll(page, pace, times=25)

    assert len(page.scrolls) == 25
    assert len(set(page.scrolls)) > 15, "滚动距离重复太多，看起来像机器"
    assert all(600 <= d <= 1800 for d in page.scrolls)


async def test_会往回滚():
    """看漏了回看是很典型的人类动作。"""
    page = Page()
    pace = Pace(
        scroll_median=0.001, scroll_sigma=0.1, read_every=(99, 99), scrollback_chance=1.0
    )
    await human_scroll(page, pace, times=5)
    assert any(d < 0 for d in page.scrolls), "从不回滚"


async def test_滚动确实耗时不是一滚到底():
    page = Page()
    pace = Pace(scroll_median=0.05, scroll_sigma=0.3, read_every=(99, 99), scrollback_chance=0.0)
    t0 = time.perf_counter()
    await human_scroll(page, pace, times=10)
    assert time.perf_counter() - t0 > 0.25, "几乎没停顿，等于机械滚动"


# ---------------------------------------------------------------- 配额


def test_到条数上限会停():
    b = SessionBudget(Pace(max_items=5))
    assert not b.exhausted()
    b.items = 5
    assert "5 条" in b.exhausted()


def test_到时长上限会停():
    p = Pace(max_minutes=0.0001)
    b = SessionBudget(p)
    time.sleep(0.02)
    assert "分钟" in b.exhausted()


async def test_配额耗尽后滚动立即中止():
    page = Page()
    b = SessionBudget(Pace(max_items=1))
    b.items = 1
    await human_scroll(page, Pace(scroll_median=0.001), times=10, budget=b)
    assert page.scrolls == [], "配额已满却还在滚"


# ---------------------------------------------------------------- 配置


def test_三档节奏从配置读出且快慢有别():
    d, c, b = load_pace("", CONFIG), load_pace("cautious", CONFIG), load_pace("brisk", CONFIG)
    assert c.action_median > d.action_median > b.action_median
    assert c.max_items < d.max_items <= b.max_items
    assert c.page_cooldown[0] > d.page_cooldown[0]


def test_配置缺失时回退到保守默认值():
    """节奏配置读不到不该让采集失败，而默认值本身是保守的。"""
    p = load_pace("", Path("完全不存在的路径.yaml"))
    assert p.action_median >= 1.5
    assert p.max_items <= 60


def test_默认节奏不会快到不像人():
    """防止有人把配置调成 0.1 秒还以为安全。"""
    p = load_pace("", CONFIG)
    assert p.action_median >= 1.0
    assert p.scroll_median >= 1.5
    assert p.action_min >= 0.3


async def test_换页冷却真的会等():
    t0 = time.perf_counter()
    await page_cooldown(Pace(page_cooldown=(0.05, 0.12)))
    assert 0.04 <= time.perf_counter() - t0 <= 0.5


async def test_动作停顿会计入配额():
    b = SessionBudget(Pace(action_median=0.001, action_min=0.001, action_max=0.01))
    await act_pause(b.pace, b)
    await act_pause(b.pace, b)
    assert b.actions == 2


# ---------------------------------------------------------------- 边界


def test_不做身份伪装():
    """明确的能力边界：放慢节奏可以，对抗检测机制不做。"""
    src = (
        Path(__file__).resolve().parents[1]
        / "src/aigc_agent/domain/rpa/humanize.py"
    ).read_text(encoding="utf-8")
    for banned in ["undetected", "stealth", "fingerprint", "解验证码", "打码"]:
        assert banned not in src.lower(), f"越界了：{banned}"
    assert "不是伪装身份" in src


async def test_采集器接受节奏参数():
    """节奏必须能一路传到采集流程，不能只是个摆设。"""
    import inspect  # noqa: PLC0415

    from aigc_agent.domain.rpa.collectors import collect_douyin_hot, collect_xiaohongshu

    for fn in (collect_xiaohongshu, collect_douyin_hot):
        assert "pace" in inspect.signature(fn).parameters, f"{fn.__name__} 没接节奏"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

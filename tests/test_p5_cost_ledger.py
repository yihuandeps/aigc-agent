"""Cost Guard 项目级 / 日级预算验收（P5）。

单任务口径活在进程里，跨会话的两级靠台账落盘。盯：
  1. 台账按天、按项目、按类别求和，落盘后读回一致
  2. 单日上限跨会话累计：上一个会话花的算数
  3. 单项目上限按 project 归档，别的项目不受影响
  4. 单日媒体次数在没单价时也拦得住，人点头只放行一次
"""

from __future__ import annotations

from pathlib import Path

from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.model.budget import CostGuard
from aigc_agent.harness.model.config import CostGuard as CostGuardConfig
from aigc_agent.harness.model.config import ModelsConfig
from aigc_agent.harness.model.ledger import CostLedger, LedgerEntry, today

ROOT = Path(__file__).resolve().parents[1]


def test_台账按天按项目求和且落盘可读回(tmp_path: Path):
    path = tmp_path / "costs" / "ledger.jsonl"
    led = CostLedger(path)
    led.append(LedgerEntry(project="p1", kind="text", cost=0.5))
    led.append(LedgerEntry(project="p1", kind="video", cost=None, calls=1))
    led.append(LedgerEntry(project="p2", kind="text", cost=0.25))
    led.append(LedgerEntry(project="p1", kind="text", cost=1.0, day="2000-01-01"))
    assert abs(led.money(day=today()) - 0.75) < 1e-9
    assert abs(led.money(project="p1") - 1.5) < 1e-9
    assert led.calls("video", day=today()) == 1 and led.calls("text", project="p2") == 1
    assert led.entries == 4

    again = CostLedger(path)
    assert again.entries == 4 and abs(again.money(project="p1") - 1.5) < 1e-9
    s = again.summary("p1")
    assert s["today_money"] == 0.75 and s["today_calls"]["video"] == 1 and s["project_money"] == 1.5


async def test_单日上限跨会话累计(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    # 上一个会话花了 60
    bus1 = EventBus()
    g1 = CostGuard(daily_limit=100, ledger=CostLedger(path), project_id="p1", session_id="s1")
    g1.attach(bus1)
    await bus1.emit(EventType.COST, cost=60.0, prompt_tokens=10, completion_tokens=5)
    assert g1.check()

    # 新会话、新进程：台账从盘上读回来
    bus2 = EventBus()
    g2 = CostGuard(daily_limit=100, ledger=CostLedger(path), project_id="p2", session_id="s2")
    g2.attach(bus2)
    assert g2.check(), "还没到 100"
    await bus2.emit(EventType.COST, cost=50.0)
    v = g2.check()
    assert not v and "单日上限" in v.reason and "110" in v.reason
    assert g2.usage.money == 50.0, "单任务口径只记自己这一次"


async def test_单项目上限按project归档(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    led = CostLedger(path)
    led.append(LedgerEntry(project="campaign-a", kind="text", cost=9.0))
    a = CostGuard(project_limit=10, ledger=CostLedger(path), project_id="campaign-a")
    bus = EventBus()
    a.attach(bus)
    assert a.check()
    await bus.emit(EventType.COST, cost=2.0)
    v = a.check()
    assert not v and "单项目上限" in v.reason and "campaign-a" in v.reason

    b = CostGuard(project_limit=10, ledger=CostLedger(path), project_id="campaign-b")
    assert b.check(), "别的项目不受影响"


def test_单日媒体次数没单价也拦(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    g = CostGuard(daily_call_limits={"video": 2}, ledger=CostLedger(path), project_id="p")
    g.record_call("video")
    g.record_call("video")
    v = g.check("video")
    assert not v and "单日上限" in v.reason
    assert g.check("image"), "别的类别不受影响"
    assert g.check(), "不带类别的金额口径没超"

    g.allow_more("video")
    assert g.check("video"), "人点头只放行一次"
    g.record_call("video")
    assert not g.check("video")

    # 新进程读回台账，昨天的账还在
    h = CostGuard(daily_call_limits={"video": 2}, ledger=CostLedger(path), project_id="p")
    assert not h.check("video")


def test_单任务口径先于日级报():
    g = CostGuard(money_limit=1.0, daily_limit=100, ledger=CostLedger())
    g.usage.add_cost(1.5)
    assert "单次任务上限" in g.check().reason


def test_没有台账时只剩单任务口径():
    g = CostGuard(daily_limit=0.0, project_limit=0.0)  # 上限 0 但没台账 → 不生效
    assert g.check()
    assert "今日" not in g.brief()


def test_from_config读日级次数与三级金额():
    cfg = CostGuardConfig(
        per_task_limit=3.0, per_project_limit=30.0, daily_limit=60.0,
        call_limits={"video": 3}, daily_call_limits={"video": 9},
    )
    g = CostGuard.from_config(cfg, ledger=CostLedger(), project_id="x", session_id="s")
    assert g.money_limit == 3.0 and g.project_limit == 30.0 and g.daily_limit == 60.0
    assert g.daily_call_limits == {"video": 9} and g.limit_for("video") == 3
    assert g.project_id == "x" and g.session_id == "s"


def test_brief含今日与项目(tmp_path: Path):
    g = CostGuard(
        daily_limit=100, project_limit=50, ledger=CostLedger(tmp_path / "l.jsonl"), project_id="p1"
    )
    g.record("text", 0.5)
    g.record_call("video")
    text = g.brief()
    assert "今日 ¥0.5000（上限 ¥100.00）" in text
    assert "video 1" in text and "项目 p1 ¥0.5000（上限 ¥50.00）" in text


def test_真实配置三级都有值():
    cg = ModelsConfig.load(ROOT / "config" / "models.yaml").cost_guard
    assert cg.per_task_limit == 200 and cg.per_project_limit == 400 and cg.daily_limit == 800
    assert cg.daily_call_limits["video"] == 60 and cg.call_limits["video"] == 14


def test_Agent装配把台账接进了闸门():
    import inspect

    from aigc_agent.app import Agent

    src = inspect.getsource(Agent.create)
    assert "CostLedger(" in src and "ledger=ledger" in src

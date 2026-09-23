"""M6 可观测验收（P5）：事件流落盘、会话回放、成本看板、痕迹重建。

P5 验收原话：「可复盘」。一套数据三种看法，都从同一份 JSONL 来。
"""

from __future__ import annotations

import inspect
from pathlib import Path

from aigc_agent.harness.events.bus import Event, EventBus, EventType
from aigc_agent.harness.events.log import EventLog, find_session, list_sessions
from aigc_agent.harness.events.stats import render_stats, session_stats
from aigc_agent.harness.execution.trace import ExecutionTrace


async def test_事件落盘并可读回_大字段截断_delta不落(tmp_path: Path):
    bus = EventBus(session_id="abc12345")
    log = EventLog(tmp_path / "sessions", bus.session_id, max_field_chars=50)
    log.attach(bus)
    await bus.emit(EventType.LOOP_START, turn=0, role="main_agent", input_preview="你好")
    for _ in range(20):
        await bus.emit(EventType.TEXT_DELTA, text="x")
    await bus.emit(
        EventType.TOOL_CALL, tool="save_draft", call_id="c1", args={"content": "正" * 500}
    )
    await bus.emit(
        EventType.TOOL_RESULT, tool="save_draft", call_id="c1", ok=True, asset_ref="as_0123456789"
    )

    assert log.written == 3 and log.failed == 0
    events = EventLog.read(log.path)
    assert [e.type for e in events] == [
        EventType.LOOP_START, EventType.TOOL_CALL, EventType.TOOL_RESULT,
    ]
    assert events[0].session_id == "abc12345" and events[0].data["input_preview"] == "你好"
    args = events[1].data["args"]["content"]
    assert len(args) < 100 and "…(+450)" in args, "大字段截断并注明截了多少"
    assert events[2].data["asset_ref"] == "as_0123456789"


async def test_写失败不影响主链路(tmp_path: Path):
    blocker = tmp_path / "notadir"
    blocker.write_text("x", encoding="utf-8")  # 文件占了目录的位置，mkdir 必然失败
    bus = EventBus()
    log = EventLog(blocker, bus.session_id)
    log.attach(bus)
    await bus.emit(EventType.WARNING, message="x")
    assert log.failed == 1 and log.written == 0


async def test_回放重建痕迹与在线一致(tmp_path: Path):
    bus = EventBus()
    live = ExecutionTrace()
    live.attach(bus)
    log = EventLog(tmp_path / "s", bus.session_id)
    log.attach(bus)

    await bus.emit(EventType.LOOP_START, turn=0)
    await bus.emit(EventType.TOOL_CALL, tool="save_draft", call_id="c1", args={"content": "v1"})
    review_args = {"asset_ids": ["as_aaaaaaaaaa"], "stage": "正文"}
    await bus.emit(
        EventType.TOOL_RESULT, tool="save_draft", call_id="c1", ok=True,
        asset_ref="as_aaaaaaaaaa", args={"content": "v1"},
    )
    await bus.emit(EventType.TOOL_CALL, tool="request_review", call_id="c2", args=review_args)
    await bus.emit(
        EventType.TOOL_RESULT, tool="request_review", call_id="c2", ok=True,
        suspend=True, args=review_args,
    )
    await bus.emit(EventType.CHECKPOINT_DECIDED, decision="revise", reason="太硬广")

    rebuilt = ExecutionTrace()
    EventLog.replay(EventLog.read(log.path), rebuilt._on_event)  # noqa: SLF001
    assert rebuilt.to_mermaid() == live.to_mermaid()
    assert rebuilt.summary() == live.summary()
    assert rebuilt.node("c2").decision == "revise" and rebuilt.node("c2").reason == "太硬广"
    assert [(e.src, e.dst, e.asset) for e in rebuilt.edges] == [("c1", "c2", "as_aaaaaaaaaa")]


def _ev(t: EventType, **data) -> Event:
    return Event(type=t, data=data)


def test_会话统计与渲染():
    events = [
        _ev(EventType.LOOP_START, turn=0),
        _ev(EventType.ITERATION_START, iteration=1),
        _ev(
            EventType.COST, role="main_agent", model="k3", prompt_tokens=1000,
            completion_tokens=100, cached_tokens=500, cost=0.5,
        ),
        _ev(EventType.ITERATION_START, iteration=2),
        _ev(EventType.COST, role="memory_extract", model="k3", prompt_tokens=200,
            completion_tokens=20, cached_tokens=0, cost=None),
        _ev(
            EventType.MODEL_RESPONSE, modality="image", model="qwen", status="succeeded",
            elapsed_s=3.0,
        ),
        _ev(EventType.COST, modality="image", model="qwen", cost=0.2),
        _ev(EventType.TOOL_RESULT, tool="save_draft", ok=True, duration_ms=12),
        _ev(EventType.TOOL_RESULT, tool="save_draft", ok=True, duration_ms=30),
        _ev(EventType.TOOL_ERROR, tool="gen_video", ok=False, duration_ms=900, error="超时"),
        _ev(EventType.CHECKPOINT_REACHED, stage="正文"),
        _ev(EventType.CHECKPOINT_DECIDED, decision="adopt"),
        _ev(EventType.BUDGET_EXCEEDED, tool="gen_video", reason="x"),
        _ev(EventType.LOOP_STOP_REASON, reason="budget_exceeded"),
        _ev(EventType.SUBAGENT_END, name="memory_consolidate", ok=True, iterations=1),
        _ev(EventType.LOOP_END, turn=0, iterations=2, stop_reason="budget_exceeded"),
    ]
    s = session_stats(events)
    assert s["turns"] == 1 and s["iterations"] == 2 and s["checkpoints"] == 1
    assert s["budget_hits"] == 1 and s["decisions"] == {"adopt": 1}
    assert abs(s["total_cost"] - 0.7) < 1e-9
    assert s["by_role"]["main_agent"]["cost"] == 0.5
    assert s["by_role"]["memory_extract"]["unpriced"] == 1
    assert s["prompt_tokens"] == 1200 and s["cached_tokens"] == 500
    assert abs(s["cache_hit"] - 0.417) < 0.01
    assert s["media"]["image"]["calls"] == 1 and s["media"]["image"]["cost"] == 0.2
    assert s["tools"]["save_draft"]["calls"] == 2 and s["tools"]["save_draft"]["max_ms"] == 30
    assert s["tools"]["gen_video"]["failed"] == 1
    assert s["stops"] == {"budget_exceeded": 1}
    assert s["subagents"]["memory_consolidate"] == {"runs": 1, "ok": 1}

    text = render_stats(s)
    assert "总成本 ¥0.7000" in text and "按角色" in text and "媒体" in text and "工具" in text
    assert "main_agent" in text and "gen_video" in text and "memory_consolidate 1/1" in text


async def test_会话列表与查找(tmp_path: Path):
    root = tmp_path / "sessions"
    for sid, cost in (("aaaa1111", 0.3), ("bbbb2222", 0.0)):
        bus = EventBus(session_id=sid)
        EventLog(root, sid).attach(bus)
        await bus.emit(EventType.LOOP_START, turn=0)
        await bus.emit(EventType.COST, cost=cost or None)
    infos = list_sessions(root)
    assert {i.session_id for i in infos} == {"aaaa1111", "bbbb2222"}
    a = next(i for i in infos if i.session_id == "aaaa1111")
    assert a.turns == 1 and a.events == 2 and abs(a.cost - 0.3) < 1e-9 and a.started > 0
    assert find_session(root, "bbbb").name.endswith("_bbbb2222.jsonl")
    assert find_session(root, "zzz") is None
    assert list_sessions(tmp_path / "nope") == []


def test_装配时挂了事件日志并把输入输出预览写进事件():
    from aigc_agent.app import Agent
    from aigc_agent.harness.execution.loop import LoopRuntime

    assert "EventLog(" in inspect.getsource(Agent.create)
    src = inspect.getsource(LoopRuntime.run_turn)
    assert "input_preview" in src and "text_preview" in src, "回放要能看到用户说了什么"

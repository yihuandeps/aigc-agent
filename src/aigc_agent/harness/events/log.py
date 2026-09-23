"""M6 事件流落盘 —— 会话回放与事后复盘的数据源。

EventBus 刻意不做持久化，P1 的注释说"接 SQLite 时挂一个 handler 进来即可"。
这就是那个 handler：每个会话一个 JSONL 文件，一行一个事件，追加写、不阻塞。

两条取舍：
  · **截断大字段**。TOOL_CALL 的 args 里可能带整篇正文，TEXT_DELTA 一轮几百条；
    回放要的是"发生了什么"，不是把全部内容再存一遍 —— 正文在 Asset 里。
  · **写失败不上抛**。可观测不能成为主链路的故障点（ARCHITECTURE M6：
    它必须能在任何模块崩溃时仍然记录，反过来它自己崩了也不能带走别人）。

回放：把文件里的事件按序喂给任意订阅者（渲染器、ExecutionTrace、统计），
和它们当初在线时收到的一样 —— 一套双用。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .bus import Event, EventBus, EventType, Handler

# 逐条事件不落盘的类型：流式文本 delta 一轮几百条，回放用不上，成本统计也不靠它
_SKIP = {EventType.TEXT_DELTA}


def _truncate(value: Any, max_chars: int, max_items: int = 50) -> Any:
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + f"…(+{len(value) - max_chars})"
    if isinstance(value, dict):
        return {str(k): _truncate(v, max_chars, max_items) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_truncate(v, max_chars, max_items) for v in list(value)[:max_items]]
        if len(value) > max_items:
            items.append(f"…(+{len(value) - max_items})")
        return items
    return value


class EventLog:
    """订阅总线，逐行写 JSONL。文件名带启动时间，列表时能按时间排。"""

    def __init__(self, root: Path, session_id: str, max_field_chars: int = 2000) -> None:
        self.root = root
        self.session_id = session_id
        self.max_field_chars = max_field_chars
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = root / f"{stamp}_{session_id}.jsonl"
        self.written = 0
        self.failed = 0

    def attach(self, bus: EventBus) -> None:
        bus.subscribe(self._on_event)

    def _on_event(self, ev: Event) -> None:
        if ev.type in _SKIP:
            return
        row = {
            "id": ev.id,
            "type": ev.type.value,
            "ts": ev.ts,
            "session": ev.session_id,
            "data": _truncate(ev.data, self.max_field_chars),
        }
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            self.written += 1
        except Exception:  # noqa: BLE001 — 可观测不能成为主链路的故障点
            self.failed += 1

    # ---------- 读回 ----------

    @staticmethod
    def read(path: Path) -> list[Event]:
        out: list[Event] = []
        if not path.exists():
            return out
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                out.append(
                    Event(
                        id=str(row.get("id") or ""),
                        type=EventType(row["type"]),
                        ts=float(row.get("ts") or 0.0),
                        session_id=str(row.get("session") or ""),
                        data=dict(row.get("data") or {}),
                    )
                )
            except Exception:  # noqa: BLE001 — 单行坏数据跳过
                continue
        return out

    @staticmethod
    def replay(events: list[Event], *handlers: Handler) -> None:
        """把事件按序喂给订阅者。只支持同步订阅者 —— 渲染器、Trace、统计都是同步的。"""
        for ev in events:
            for h in handlers:
                try:
                    h(ev)
                except Exception:  # noqa: BLE001
                    continue


@dataclass
class SessionInfo:
    path: Path
    session_id: str
    started: float
    events: int
    turns: int
    cost: float
    size: int

    @property
    def started_text(self) -> str:
        return datetime.fromtimestamp(self.started).strftime("%m-%d %H:%M") if self.started else "—"


def list_sessions(root: Path, limit: int = 30) -> list[SessionInfo]:
    """按启动时间倒序。只扫文件头尾，不把每个文件全读进来。"""
    if not root.exists():
        return []
    infos: list[SessionInfo] = []
    for p in sorted(root.glob("*.jsonl"), reverse=True)[:limit]:
        events = EventLog.read(p)
        turns = sum(1 for e in events if e.type is EventType.LOOP_START)
        cost = sum(float(e.data.get("cost") or 0.0) for e in events if e.type is EventType.COST)
        started = events[0].ts if events else p.stat().st_mtime
        sid = p.stem.split("_", 1)[1] if "_" in p.stem else p.stem
        infos.append(
            SessionInfo(
                path=p, session_id=sid, started=started, events=len(events),
                turns=turns, cost=cost, size=p.stat().st_size,
            )
        )
    return infos


def find_session(root: Path, key: str) -> Path | None:
    """按 session id 前缀或文件名片段找。"""
    if not root.exists():
        return None
    hits = [p for p in root.glob("*.jsonl") if key in p.stem]
    if not hits:
        return None
    return sorted(hits)[-1]


def now_stamp() -> float:
    return time.time()

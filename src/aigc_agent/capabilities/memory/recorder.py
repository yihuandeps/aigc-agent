"""把打回理由自动落库 —— 通过事件总线，不侵入 L0。

GraphRuntime（L0）只管发 CHECKPOINT_DECIDED 事件，它不知道有记忆这回事。
这里（L1）订阅事件并写库。依赖方向 L1 → L0，符合分层约束。
"""

from __future__ import annotations

from ...harness.events.bus import Event, EventBus, EventType
from .store import MemoryStore


class RejectionRecorder:
    """订阅 checkpoint 决策，把打回理由写进项目层记忆。"""

    def __init__(self, store: MemoryStore, project_id: str = "") -> None:
        self.store = store
        self.project_id = project_id
        self.recorded: list[str] = []

    def attach(self, bus: EventBus) -> None:
        bus.subscribe(self._on_event)

    def _on_event(self, ev: Event) -> None:
        if ev.type is not EventType.CHECKPOINT_DECIDED:
            return
        d = ev.data
        if d.get("decision") == "adopt":
            return  # 采纳不产生「避雷」记忆
        reason = (d.get("reason") or "").strip()
        if not reason:
            return
        m = self.store.record_rejection(
            reason=reason,
            node_id=d.get("node", ""),
            run_id=d.get("run", ""),
            target_node=d.get("target_node", ""),
            project_id=self.project_id,
            decision=d.get("decision", "revise"),
            candidates=d.get("candidates"),
        )
        self.recorded.append(m.id)

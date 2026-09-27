"""把打回理由自动落库 —— 通过事件总线，不侵入 L0。

GraphRuntime（L0）只管发 CHECKPOINT_DECIDED 事件，它不知道有记忆这回事。
这里（L1）订阅事件并写库。依赖方向 L1 → L0，符合分层约束。

2026-09-26（用户拍板）：
  · **决策类环节的打回不存避雷**：换模型、面容冲突、素材、出片确认、主形象被拒 —— 那是对
    一次操作的决定（「这次不换」「这段先不要」），不是内容上的偏好。文案审核、成片审核的打回
    多半是内容意见（「口播太硬」「产品特写太暗」），照常存成避雷。
    之前一律存成永久避雷、每轮 pin：「视频模型切换 … 不允许换」「已充值用 seedance 2.0 fast」
    在别的剧里还 pin 着
  · 内容类打回（剧本 / 大纲 / 分镜 / 正文…）照存，挂到项目和环节上，默认 30 天后过期
"""

from __future__ import annotations

from ...harness.events.bus import Event, EventBus, EventType
from .store import REJECTION_TTL_DAYS, MemoryStore

# 决策型环节（人审 stage 名）。系统挂起问人时用的就是这些名字：
# 视频 / 生图模型切换（media.py）、角色面容冲突 / 参考图·主形象被拒（drama.py）、
# 素材 / 出片确认（short_video.py）
DECISION_STAGES: tuple[str, ...] = (
    "视频模型切换",
    "生图模型切换",
    "角色面容冲突",
    "素材",
    "出片确认",
    "参考图·主形象被拒",
)


def is_decision_stage(node: str) -> bool:
    n = (node or "").strip()
    return any(n == s or n.startswith(s) for s in DECISION_STAGES)


class RejectionRecorder:
    """订阅 checkpoint 决策，把内容类的打回理由写进项目层记忆。"""

    def __init__(
        self, store: MemoryStore, project_id: str = "", ttl_days: float | None = REJECTION_TTL_DAYS
    ) -> None:
        self.store = store
        self.project_id = project_id
        # 内容类打回的有效期（天）；None = 永久（旧行为）
        self.ttl_days = ttl_days
        self.recorded: list[str] = []
        self.skipped: list[str] = []  # 决策类环节的打回：没存（给测试和复盘看）

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
        node = str(d.get("node") or "")
        target = str(d.get("target_node") or "")
        if is_decision_stage(node) or is_decision_stage(target):
            self.skipped.append(node or target)
            return
        m = self.store.record_rejection(
            reason=reason,
            node_id=node,
            run_id=d.get("run", ""),
            target_node=target,
            project_id=self.project_id,
            decision=d.get("decision", "revise"),
            candidates=d.get("candidates"),
            ttl_days=self.ttl_days,
        )
        self.recorded.append(m.id)

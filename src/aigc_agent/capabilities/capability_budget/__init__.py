"""能力预算 —— Skill Hub (M7) 与 MCP Hub (M20) 共管的那一个池子。

引入两个 Hub 后最容易失控的地方：两边各自看着都合理，加起来就超了。
Skill Hub 多加载一个 SOP，MCP Hub 就要少展开几个 schema。没有这个共同上限，
"我能做什么"还没干活就吃掉窗口的五分之一。

这里管五样东西的 token（config/capability_budget.yaml）：

    常驻  skill 目录 · 工具目录 · 内置工具全 schema
    激活  已加载的 skill 正文 · 已展开的外部工具 schema

超预算按 on_over_budget 的顺序降级，直到回到预算内：
    drop_lowest_priority_skill → collapse_expanded_tools → shrink_skill_digest
合规类 skill（priority ≥ 100）永不被挤出；Pin 区不在这里管。

实现上它是 L1 的一个订阅者 + 调度者：skill 正文和目录以 pin 的形式挂在
主循环的 ShortTermMemory 系统区，展开的工具 schema 由 M4 注册表管着；
分配器只做加减法和决定卸谁，不碰装配顺序（那是 M3 的事）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ...harness.context.window import ShortTermMemory
from ...harness.events.bus import Event, EventBus, EventType
from ...harness.model.gateway import estimate_tokens
from ...harness.tools.registry import ToolRegistry
from ..skill_hub import Skill, SkillHub

DIGEST_PIN = "skill_digest"
BODIES_PIN = "skill_bodies"

DEFAULT_ORDER = [
    "drop_lowest_priority_skill",
    "collapse_expanded_tools",
    "shrink_skill_digest",
]


class BudgetConfig(BaseModel):
    context_window: int = 256_000
    capability_budget: int = 30_000
    soft_target: int = 80_000
    resident: dict[str, int] = Field(default_factory=dict)
    active: dict[str, int] = Field(default_factory=dict)
    max_active_skills: int = 3
    max_skill_body_tokens: int = 5_000
    max_expanded_tools: int = 8
    on_over_budget: list[str] = Field(default_factory=lambda: list(DEFAULT_ORDER))
    warn_at_ratio: float = 0.9

    @classmethod
    def load(cls, path: str | Path) -> BudgetConfig:
        path = Path(path)
        if not path.exists():
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        alloc = raw.get("allocation") or {}
        limits = raw.get("limits") or {}
        tele = raw.get("telemetry") or {}
        return cls(
            context_window=int(raw.get("context_window", 256_000)),
            capability_budget=int(raw.get("capability_budget", 30_000)),
            soft_target=int(raw.get("soft_target", 80_000)),
            resident={k: int(v) for k, v in (alloc.get("resident") or {}).items()},
            active={k: int(v) for k, v in (alloc.get("active") or {}).items()},
            max_active_skills=int(limits.get("max_active_skills", 3)),
            max_skill_body_tokens=int(limits.get("max_skill_body_tokens", 5_000)),
            max_expanded_tools=int(limits.get("max_expanded_tools", 8)),
            on_over_budget=[str(x) for x in (raw.get("on_over_budget") or DEFAULT_ORDER)],
            warn_at_ratio=float(tele.get("warn_at_ratio", 0.9)),
        )


@dataclass
class Usage:
    skill_digest: int = 0
    tool_digest: int = 0
    builtin_tool_schema: int = 0
    skill_body: int = 0
    expanded_tool_schema: int = 0

    @property
    def resident(self) -> int:
        return self.skill_digest + self.tool_digest + self.builtin_tool_schema

    @property
    def active(self) -> int:
        return self.skill_body + self.expanded_tool_schema

    @property
    def total(self) -> int:
        return self.resident + self.active

    def as_dict(self) -> dict[str, int]:
        return {
            "skill_digest": self.skill_digest,
            "tool_digest": self.tool_digest,
            "builtin_tool_schema": self.builtin_tool_schema,
            "skill_body": self.skill_body,
            "expanded_tool_schema": self.expanded_tool_schema,
            "total": self.total,
        }

    def brief(self, budget: int) -> str:
        ratio = f"{self.total / budget:.0%}" if budget else "—"
        return (
            f"能力区 {self.total:,}/{budget:,} token（{ratio}）· "
            f"常驻 {self.resident:,}"
            f"（skill目录 {self.skill_digest:,} / 工具目录 {self.tool_digest:,} / "
            f"内置schema {self.builtin_tool_schema:,}）· "
            f"激活 {self.active:,}"
            f"（skill正文 {self.skill_body:,} / 展开schema {self.expanded_tool_schema:,}）"
        )


class CapabilityAllocator:
    """两个 Hub 的共同上限。"""

    def __init__(
        self,
        cfg: BudgetConfig,
        registry: ToolRegistry,
        hub: SkillHub,
        memory: ShortTermMemory,
        bus: EventBus,
    ) -> None:
        self.cfg = cfg
        self.registry = registry
        self.hub = hub
        self.memory = memory
        self.bus = bus
        self.active: dict[str, Skill] = {}
        self.digest_limit = 0  # 0 = 不截；shrink_skill_digest 会减半
        self.last: Usage | None = None
        self.content_type = ""
        self.stage = ""

    # ---------- 接线 ----------

    def attach(self, bus: EventBus) -> None:
        """模型展开了外部工具的 schema 之后，回来核一次预算。"""
        bus.subscribe(self._on_event)

    async def _on_event(self, ev: Event) -> None:
        if ev.type is not EventType.TOOL_RESULT:
            return
        if ev.data.get("tool") == self.registry.expand_tool_name:
            await self.enforce()

    # ---------- 目录与正文的 pin ----------

    async def refresh(self, content_type: str = "", stage: str = "") -> list[str]:
        """每轮开始前调：热加载 skill、重挂目录与正文、核预算。返回变了的 skill。"""
        if content_type or stage:
            self.content_type, self.stage = content_type, stage
        changed = self.hub.refresh_if_changed()
        if changed:
            for name in changed:
                if name in self.active:
                    fresh = self.hub.get(name)
                    if fresh is None:
                        self.active.pop(name, None)  # 删掉了或改成 draft，卸下
                    else:
                        self.active[name] = fresh
            await self.bus.emit(
                EventType.SKILL_RELOAD,
                changed=changed,
                active=list(self.active),
                message=f"[skill] 热加载：{', '.join(changed)}",
            )
        self._pin_digest()
        self._pin_bodies()
        await self.enforce()
        return changed

    def _pin_digest(self) -> None:
        digest = self.hub.catalog_digest(self.content_type, self.stage, limit=self.digest_limit)
        if not digest:
            self.memory.unpin(DIGEST_PIN)
            return
        self.memory.pin(
            DIGEST_PIN,
            "## 可用的方法论 skill\n"
            "看到和任务相关的，先用 load_skill(names=[...]) 加载正文再动手。\n" + digest,
            position="system",
        )

    def _pin_bodies(self) -> None:
        if not self.active:
            self.memory.unpin(BODIES_PIN)
            return
        self.memory.pin(BODIES_PIN, self.hub.render(list(self.active.values())), position="system")

    # ---------- 激活 / 卸载 ----------

    async def activate_skill(self, skill: Skill) -> str:
        """load_skill 的回调：把正文挂进系统区，然后核预算。返回给模型看的一句话。"""
        self.active[skill.name] = skill
        note = f"已加载，约 {skill.tokens:,} token"
        if skill.tokens > self.cfg.max_skill_body_tokens:
            note += f"（超过单篇上限 {self.cfg.max_skill_body_tokens:,}，建议拆分）"
        self._pin_bodies()
        actions = await self.enforce()
        if skill.name not in self.active:
            return f"加载后立即被能力预算挤出（{'；'.join(actions)}）。先卸下别的再试。"
        if actions:
            note += "；为腾预算已" + "，".join(actions)
        return note

    def deactivate_skill(self, name: str) -> bool:
        if name not in self.active:
            return False
        self.active.pop(name)
        self._pin_bodies()
        return True

    # ---------- 度量与执行 ----------

    async def measure(self) -> Usage:
        u = Usage()
        pin = self.memory.pins.get(DIGEST_PIN)
        u.skill_digest = estimate_tokens(pin.content) if pin else 0
        pin = self.memory.pins.get(BODIES_PIN)
        u.skill_body = estimate_tokens(pin.content) if pin else 0
        u.tool_digest = estimate_tokens(self.registry.catalog_digest())
        full = await self.registry.schemas(self.registry.full_names)
        u.builtin_tool_schema = estimate_tokens(json.dumps(full, ensure_ascii=False))
        expanded = await self.registry.schemas(self.registry.expanded)
        u.expanded_tool_schema = estimate_tokens(json.dumps(expanded, ensure_ascii=False))
        return u

    async def enforce(self) -> list[str]:
        """超预算就按顺序降级，直到回到预算内或无可降级。返回做过的动作。"""
        actions: list[str] = []

        # 硬限制先：同时激活的正文数
        while len(self.active) > self.cfg.max_active_skills:
            name = self._lowest()
            if name is None:
                break
            self.active.pop(name)
            actions.append(f"卸下 skill {name}（超过同时激活上限）")
        if actions:
            self._pin_bodies()

        usage = await self.measure()
        budget = self.cfg.capability_budget
        while usage.total > budget:
            progressed = False
            for step in self.cfg.on_over_budget:
                did = self._apply(step)
                if did:
                    progressed = True
                    actions.append(did)
                    usage = await self.measure()
                    if usage.total <= budget:
                        break
            if not progressed:
                await self.bus.emit(
                    EventType.WARNING,
                    message=(
                        f"能力区 {usage.total:,} token 超预算 {budget:,}，且已无可降级项 —— "
                        "内置工具太多或合规 skill 太长，需要调配置"
                    ),
                )
                break

        self.last = usage
        await self.bus.emit(
            EventType.CAPABILITY_BUDGET,
            **usage.as_dict(),
            budget=budget,
            ratio=round(usage.total / budget, 3) if budget else 0.0,
            active_skills=list(self.active),
            expanded_tools=self.registry.expanded,
            actions=actions,
        )
        if budget and usage.total >= self.cfg.warn_at_ratio * budget:
            await self.bus.emit(
                EventType.WARNING,
                message=f"能力区已用 {usage.total:,}/{budget:,} token，接近上限",
            )
        return actions

    def _lowest(self) -> str | None:
        """最该被挤出的：rank 最低且不受保护的。"""
        pool = [s for s in self.active.values() if not s.protected]
        if not pool:
            return None
        return min(pool, key=lambda s: s.rank).name

    def _apply(self, step: str) -> str:
        if step == "drop_lowest_priority_skill":
            name = self._lowest()
            if name is None:
                return ""
            self.active.pop(name)
            self._pin_bodies()
            return f"卸下 skill {name}"
        if step == "collapse_expanded_tools":
            expanded = self.registry.expanded
            if not expanded:
                return ""
            self.registry.collapse(expanded[-1])
            return f"收回工具 {expanded[-1]} 的 schema"
        if step == "shrink_skill_digest":
            n = len(self.hub.candidates(self.content_type, self.stage))
            current = self.digest_limit or n
            if current <= 1:
                return ""
            self.digest_limit = max(1, current // 2)
            self._pin_digest()
            return f"skill 目录截到 {self.digest_limit} 条"
        return ""

    def brief(self) -> str:
        if self.last is None:
            return "能力区尚未度量"
        active = ", ".join(self.active) or "无"
        return f"{self.last.brief(self.cfg.capability_budget)} · 激活 skill：{active}"

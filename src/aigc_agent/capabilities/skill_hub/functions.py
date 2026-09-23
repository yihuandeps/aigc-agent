"""Skill 相关的 function —— 三级披露的模型侧入口。

  L1 目录   常驻，name + description
  L2 主文档 `load_skill` 拉进来
  L3 参考   `load_skill_reference` 按需拉

第三级是给大型多阶段 skill 用的。一套完整创作方法论动辄 20K token，
整篇塞进去会把能力预算吃光；拆成「主文档说流程 + 参考说细节」之后，
任一时刻只有当前阶段用得上的那 1–2 篇在上下文里。

正文进上下文有两种方式：
  · 没接能力预算时（on_load=None）：正文直接作为工具返回值，进对话历史
  · 接了能力预算时：正文由分配器 **pin 进系统区常驻**，工具只返回一句确认。
    这样正文不会随滑窗被剔掉、也不会被重复加载两份，而且预算超了分配器能把它卸下来。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from ...harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
    ToolSpec,
)
from . import Skill, SkillHub

MAX_REF_CHARS = 12000

# 分配器回调：收到被选中的 skill，返回给模型看的一句说明
OnLoad = Callable[[Skill], Awaitable[str]]


class SkillFunctions:
    name = "skill"
    namespaced = False
    disclosure = "full"  # 就两个，且没它们就没法加载任何方法论

    def __init__(self, hub: SkillHub, on_load: OnLoad | None = None) -> None:
        self.hub = hub
        self.on_load = on_load
        self._specs: dict[str, ToolSpec] = {
            "load_skill": ToolSpec(
                name="load_skill",
                summary="加载一份方法论 skill 的正文",
                permission=PermissionLevel.READ,
                description=(
                    "目录里看到相关的 skill 就先加载它再动手。"
                    "带参考文档的 skill 会同时给出参考目录，按阶段再用 "
                    "load_skill_reference 取。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "skill 名，见目录",
                        }
                    },
                    "required": ["names"],
                },
            ),
            "load_skill_reference": ToolSpec(
                name="load_skill_reference",
                summary="加载某个 skill 的一篇参考文档",
                permission=PermissionLevel.READ,
                description=(
                    "**一次只拉当前阶段用得上的那一两篇，不要一次全拉**"
                    "——参考文档单篇 2–4K token，全拉会把上下文挤爆。"
                    "主文档的参考目录里写了每篇的用途和加载时机。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "skill": {"type": "string"},
                        "name": {"type": "string", "description": "参考文档名，可省略 .md"},
                    },
                    "required": ["skill", "name"],
                },
            ),
        }

    # ---------- ToolProvider 协议 ----------

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        n = len(self.hub.available)
        refs = sum(len(s.references) for s in self.hub.available)
        return ProviderHealth(ok=True, detail=f"{n} 个 skill · {refs} 篇参考")

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            result = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
            result = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result

    # ---------- 实现 ----------

    async def _fn_load_skill(self, names: list[str] | str) -> ToolResult:
        if isinstance(names, str):
            names = [names]
        # 运营可能刚改过文件：先看一眼有没有变，再取正文
        self.hub.refresh_if_changed()
        found, missing = self.hub.select(names)
        if not found:
            avail = ", ".join(s.name for s in self.hub.available) or "（无）"
            return ToolResult(ok=False, error=f"没有这些 skill：{missing}。可用：{avail}")

        parts: list[str] = []
        if self.on_load is None:
            parts.append(self.hub.render(found))
        else:
            notes = [f"✓ {s.name}：{await self.on_load(s)}" for s in found]
            parts.append("\n".join(notes))
            parts.append("正文已常驻在系统区（能力预算管理），不必再次加载，直接按它做。")
        for s in found:
            idx = s.reference_index()
            if idx:
                parts.append(idx)
        body = "\n\n".join(parts)
        if missing:
            body += f"\n\n（未找到：{', '.join(missing)}）"
        return ToolResult(content=body)

    async def _fn_load_skill_reference(self, skill: str, name: str) -> ToolResult:
        sk = next((s for s in self.hub.available if s.name == skill), None)
        if sk is None:
            avail = ", ".join(s.name for s in self.hub.available) or "（无）"
            return ToolResult(ok=False, error=f"没有 skill {skill!r}。可用：{avail}")

        ref = sk.reference(name)
        if ref is None:
            names = ", ".join(r.name for r in sk.references) or "（该 skill 没有参考文档）"
            return ToolResult(ok=False, error=f"{skill} 里没有参考文档 {name!r}。可用：{names}")

        text = ref.read()
        if not text:
            return ToolResult(ok=False, error=f"{ref.path} 读不出内容")

        truncated = len(text) > MAX_REF_CHARS
        return ToolResult(
            content=f"# {skill} / {ref.name}\n\n{text[:MAX_REF_CHARS]}",
            truncated=truncated,
        )

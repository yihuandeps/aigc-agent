"""并行候选 function（M10 扇出的模型侧入口）。

  fan_out_candidates   同一个任务、N 个角度，N 个子代理并发各写一版，每版落一个资产，
                       汇总时给出每版独有的差异点 —— 供人对比，不替人挑

典型用法（ARCHITECTURE M10）：一个选题并行生成 5 个不同角度的脚本方案。
主 Agent 拿到候选 id 后接 request_review，人在候选里选。
"""

from __future__ import annotations

import time
from typing import Any

from ...capabilities.subagents import SubAgentDef, SubAgentRunner
from ...harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
    ToolSpec,
)
from ..assets.store import AssetStore, AssetType

MAX_ANGLES = 8

_TYPE_BY_KIND = {
    "outline": AssetType.OUTLINE,
    "script": AssetType.SCRIPT,
    "storyboard": AssetType.STORYBOARD,
    "copy": AssetType.TEXT,
}

CANDIDATE_DEF = SubAgentDef(
    name="candidate_writer",
    description="按指定角度独立写一版候选成稿",
    system_prompt=(
        "你是内容创作子代理。按给定的角度独立写一版**成稿正文**。\n"
        "只输出正文本身：不要解释思路、不要列大纲、不要加「以下是…」之类的引导语。\n"
        "风格、长度、平台规范以任务说明为准；任务说明里的硬约束（禁止项）必须遵守。"
    ),
    tools=[],  # 纯生成，不碰工具；要查资料由主 Agent 先查好放进 context
    allowed=[PermissionLevel.READ],
    role="subagent",
    stateless=True,
    max_iterations=2,
)


class FanOutFunctions:
    name = "fanout"
    namespaced = False
    disclosure = "full"

    def __init__(self, runner: SubAgentRunner, assets: AssetStore) -> None:
        self.runner = runner
        self.assets = assets
        self._specs = {
            "fan_out_candidates": ToolSpec(
                name="fan_out_candidates",
                summary="并行按多个角度各写一版候选，每版落资产并给出差异点，供人对比选择",
                permission=PermissionLevel.COMPUTE,
                description=(
                    "关键节点要给人多个候选时用它：一次给 2-8 个**方向不同**的角度，"
                    "子代理并发各写一版，互相看不见。返回每版的资产 id 和它独有的表达，"
                    "然后用 request_review 交给人选。每个角度都会花一次模型调用。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "写什么、给谁、多长、什么规范"},
                        "angles": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "每版的切入角度，如「反差开场」「数据开场」「故事开场」",
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["outline", "copy", "script", "storyboard"],
                            "description": "候选的资产类型，默认 copy",
                        },
                        "context": {
                            "type": "string",
                            "description": "共享的背景资料（热榜原文、记忆简报的硬约束等）",
                        },
                        "parent_id": {
                            "type": "string",
                            "description": "基于哪份资产（大纲/脚本）展开，用于血缘",
                        },
                        "concurrency": {"type": "integer", "description": "并发数，默认 4"},
                    },
                    "required": ["task", "angles"],
                },
            )
        }

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail=f"最多 {MAX_ANGLES} 个角度并发")

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            r = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
            r = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        r.duration_ms = int((time.perf_counter() - started) * 1000)
        return r

    async def _fn_fan_out_candidates(
        self,
        task: str,
        angles: list[str],
        kind: str = "copy",
        context: str = "",
        parent_id: str = "",
        concurrency: int = 4,
    ) -> ToolResult:
        angles = [str(a).strip() for a in (angles or []) if str(a).strip()]
        if not angles:
            return ToolResult(ok=False, error="angles 不能为空：至少给一个角度")
        if len(angles) > MAX_ANGLES:
            return ToolResult(
                ok=False, error=f"一次最多 {MAX_ANGLES} 个角度（给了 {len(angles)} 个），先挑一挑"
            )
        if parent_id:
            self.assets.get(parent_id)  # 不存在就在这里报，不要打完一批才发现血缘挂不上

        tasks = [f"{task.strip()}\n\n本版角度：{a}" for a in angles]
        summary = await self.runner.run_many(
            CANDIDATE_DEF, tasks, context=context, concurrency=max(1, int(concurrency or 4))
        )

        lines = [
            f"并行产出 {summary.ok}/{summary.total} 个候选，耗时 {summary.duration_ms / 1000:.1f}s"
            + (f"，¥{summary.cost:.4f}" if summary.cost is not None else "")
        ]
        ids: list[str] = []
        for i, (angle, r, diff) in enumerate(
            zip(angles, summary.results, summary.diffs, strict=True), 1
        ):
            text = (r.text or "").strip()
            if not r.ok or not text:
                lines.append(f"- ✗ 候选{i} 角度「{angle}」失败：{r.error or '空输出'}")
                continue
            a = self.assets.create(
                text,
                type_=_TYPE_BY_KIND.get(kind, AssetType.TEXT),
                summary=f"候选{i}·{angle[:14]}",
                parents=[parent_id] if parent_id else [],
                creator="model:subagent",
                gen_params={
                    "angle": angle,
                    "iterations": r.iterations,
                    "cost": r.cost,
                    "fanout": True,
                },
                gen_cost=r.cost,
            )
            ids.append(a.id)
            lines.append(
                f"- {a.id} 候选{i} 角度「{angle}」{len(text)} 字 · 独有：{', '.join(diff) or '—'}"
            )
        if not ids:
            return ToolResult(ok=False, error="\n".join(lines))
        lines.append("下一步：用 request_review 把候选交给人选，或 compare_assets 两两对比。")
        return ToolResult(content="\n".join(lines), asset_ref=ids[0])

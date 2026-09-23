"""数据回流 function（M17 的模型侧入口）。

  record_metrics      录一次发布后的数据（人工录入；将来 MetricsSource 接 API 也走这条落库）
  review_performance  复盘：相对账号中位数的倍率，强/弱作品
  list_metrics        看已录入的数据

录入即回流：每录一条就重算复盘，显著优于/劣于中位数的作品提炼成 DATA 来源的账号层记忆，
下一次 Brief 的建议区就能看到 —— 这是让 Agent 越用越准的那条闭环。
"""

from __future__ import annotations

import time
from typing import Any

from ...capabilities.memory.store import Memory
from ...harness.events.bus import EventBus, EventType
from ...harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
    ToolSpec,
)
from ..analytics import Feedback, Metrics, MetricsStore, Reviewer, review_to_params
from ..assets.store import AssetStore, AssetType
from ..distribution import Packager


class AnalyticsFunctions:
    name = "analytics"
    namespaced = False
    disclosure = "full"

    def __init__(
        self,
        store: MetricsStore,
        reviewer: Reviewer,
        feedback: Feedback,
        packager: Packager,
        assets: AssetStore,
        bus: EventBus | None = None,
    ) -> None:
        self.store = store
        self.reviewer = reviewer
        self.feedback = feedback
        self.packager = packager
        self.assets = assets
        self.bus = bus
        self._specs = {
            "record_metrics": ToolSpec(
                name="record_metrics",
                summary="录入一个已发布作品的数据（播放/完播/赞评转藏），自动复盘并提炼进账号记忆",
                permission=PermissionLevel.WRITE,
                description=(
                    "发布后有数据了就录。同一个包可以多次录（1 天 / 7 天），复盘取最新一条。"
                    "满 3 个作品后，显著优于或劣于中位数的会写进账号层记忆。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "package_asset_id": {"type": "string", "description": "待发布包的资产 id"},
                        "views": {"type": "integer"},
                        "likes": {"type": "integer"},
                        "comments": {"type": "integer"},
                        "shares": {"type": "integer"},
                        "saves": {"type": "integer"},
                        "follows": {"type": "integer"},
                        "completion_rate": {"type": "number", "description": "完播率 0-1"},
                        "url": {"type": "string"},
                        "note": {"type": "string"},
                    },
                    "required": ["package_asset_id", "views"],
                },
            ),
            "review_performance": ToolSpec(
                name="review_performance",
                summary="复盘已录入的数据：相对账号中位数的倍率、强/弱作品",
                permission=PermissionLevel.WRITE,
                parameters={
                    "type": "object",
                    "properties": {"platform": {"type": "string", "description": "留空看全部"}},
                },
            ),
            "list_metrics": ToolSpec(
                name="list_metrics",
                summary="列出已录入的发布数据",
                permission=PermissionLevel.READ,
                parameters={
                    "type": "object",
                    "properties": {
                        "platform": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                },
            ),
        }

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail=f"{len(self.store)} 条数据")

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            r = await getattr(self, f"_fn_{tool}")(**args)
        except (ValueError, KeyError) as e:
            r = ToolResult(ok=False, error=str(e))
        except Exception as e:  # noqa: BLE001
            r = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        r.duration_ms = int((time.perf_counter() - started) * 1000)
        return r

    async def _fn_record_metrics(
        self,
        package_asset_id: str,
        views: int,
        likes: int = 0,
        comments: int = 0,
        shares: int = 0,
        saves: int = 0,
        follows: int = 0,
        completion_rate: float | None = None,
        url: str = "",
        note: str = "",
    ) -> ToolResult:
        pkg = self.assets.get(package_asset_id)
        manifest = self.packager.manifest_of(pkg)
        if completion_rate is not None and not 0 <= float(completion_rate) <= 1:
            return ToolResult(ok=False, error="completion_rate 是 0-1 的比例，不是百分数")
        m = self.store.add(
            Metrics(
                package_id=package_asset_id,
                platform=manifest.platform,
                title=manifest.title,
                tags=list(manifest.tags),
                url=url or manifest.published_url,
                views=int(views),
                likes=int(likes),
                comments=int(comments),
                shares=int(shares),
                saves=int(saves),
                follows=int(follows),
                completion_rate=float(completion_rate) if completion_rate is not None else None,
                note=note,
            )
        )
        review = self.reviewer.summarize(manifest.platform)
        new = self.feedback.distill(review)
        if self.bus is not None:
            await self.bus.emit(
                EventType.METRICS_RECORDED,
                package=package_asset_id,
                platform=manifest.platform,
                views=m.views,
                samples=review.samples,
                distilled=[x.id for x in new],
            )
        lines = [
            f"已记录 {m.id}：{manifest.title[:24]} · 播放 {m.views} · 互动率 {m.engagement:.2%}"
        ]
        lines.append("")
        lines.append(review.render())
        if new:
            lines.append("")
            lines.append("提炼进账号层记忆：")
            lines += [f"- {x.id}：{x.content}" for x in new]
        return ToolResult(content="\n".join(lines))

    async def _fn_review_performance(self, platform: str = "") -> ToolResult:
        review = self.reviewer.summarize(platform)
        text = review.render()
        if review.samples == 0:
            return ToolResult(content=text)  # 没数据就不落报告资产，免得留一堆空报告
        asset = self.assets.create(
            text,
            type_=AssetType.REPORT,
            summary=f"复盘·{platform or '全部'}·{review.samples} 个作品",
            creator="tool:review_performance",
            gen_params=review_to_params(review),
        )
        return ToolResult(content=f"报告 {asset.id}\n{text}", asset_ref=asset.id)

    async def _fn_list_metrics(self, platform: str = "", limit: int = 20) -> ToolResult:
        items = self.store.all(platform)[-max(1, int(limit or 20)) :]
        if not items:
            return ToolResult(content="还没有录入数据")
        lines = [
            f"- {m.id} · {m.platform} · {m.title[:20]} · 播放 {m.views} · 赞 {m.likes} · "
            f"评 {m.comments} · 转 {m.shares}"
            + (f" · 完播 {m.completion_rate:.0%}" if m.completion_rate is not None else "")
            for m in items
        ]
        return ToolResult(content="\n".join(lines))


def memories_brief(mems: list[Memory]) -> str:
    return "\n".join(f"- {m.content}" for m in mems)

"""分发 function（M16 的模型侧入口）。

  list_platforms          各平台格式规格
  build_release_package   机审 + 格式适配 + 落成待发布包（目录 + manifest）
  mark_published          人上传完把链接记回来 —— 数据回流靠它对上号
  list_packages           已打的包与发布状态

没有 publish 这个 function：发布是不可逆的 L-external，这版只产待发布包，由人上传。
"""

from __future__ import annotations

import time
from typing import Any

from ...harness.events.bus import EventBus, EventType
from ...harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
    ToolSpec,
)
from ..assets.store import AssetStore
from ..compliance import ComplianceChecker
from ..distribution import Packager
from .compliance import run_check


class DistributionFunctions:
    name = "distribution"
    namespaced = False
    disclosure = "full"

    def __init__(
        self,
        packager: Packager,
        checker: ComplianceChecker,
        assets: AssetStore,
        bus: EventBus | None = None,
    ) -> None:
        self.packager = packager
        self.checker = checker
        self.assets = assets
        self.bus = bus
        self._specs = {
            "list_platforms": ToolSpec(
                name="list_platforms",
                summary="各平台的格式规格：标题/正文长度、话题数、视频比例与时长",
                permission=PermissionLevel.READ,
                parameters={"type": "object", "properties": {}},
            ),
            "build_release_package": ToolSpec(
                name="build_release_package",
                summary="机审后按平台规格打成待发布包（目录 + 上传清单 + manifest），由人上传",
                permission=PermissionLevel.WRITE,
                description=(
                    "**这是交付的最后一步。** 会先跑机审：有 block 级问题就拒绝打包，"
                    "改完再来；warn 会写进上传清单让人判断。标题/正文超长不会被截断，"
                    "而是列为待改项。打好的包不等于已发布 —— 上传是人的动作，"
                    "上传后用 mark_published 记回链接。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "content_asset_id": {"type": "string", "description": "正文资产 id"},
                        "platform": {
                            "type": "string",
                            "description": "douyin / xiaohongshu / wechat / zhihu / generic",
                        },
                        "title": {"type": "string", "description": "留空取正文第一行"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "media_asset_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "成片/图片资产，决定包的类型（video/image/text）",
                        },
                        "cover_asset_id": {"type": "string"},
                        "override_reason": {
                            "type": "string",
                            "description": (
                                "机审有 block 仍要打包时，人给出的放行理由（会记进 manifest）"
                            ),
                        },
                    },
                    "required": ["content_asset_id", "platform"],
                },
            ),
            "mark_published": ToolSpec(
                name="mark_published",
                summary="人上传完后，把发布链接记到待发布包上",
                permission=PermissionLevel.WRITE,
                parameters={
                    "type": "object",
                    "properties": {
                        "package_asset_id": {"type": "string"},
                        "url": {"type": "string"},
                    },
                    "required": ["package_asset_id", "url"],
                },
            ),
            "list_packages": ToolSpec(
                name="list_packages",
                summary="列出已打的待发布包及其发布状态",
                permission=PermissionLevel.READ,
                parameters={"type": "object", "properties": {}},
            ),
        }

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            ok=True,
            detail=f"{len(self.packager.catalog.specs)} 个平台 · 包存于 {self.packager.dir}",
        )

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            r = await getattr(self, f"_fn_{tool}")(**args)
        except (PermissionError, ValueError, KeyError) as e:
            r = ToolResult(ok=False, error=str(e))
        except Exception as e:  # noqa: BLE001
            r = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        r.duration_ms = int((time.perf_counter() - started) * 1000)
        return r

    async def _fn_list_platforms(self) -> ToolResult:
        return ToolResult(content=self.packager.catalog.render())

    async def _fn_build_release_package(
        self,
        content_asset_id: str,
        platform: str,
        title: str = "",
        tags: list[str] | None = None,
        media_asset_ids: list[str] | None = None,
        cover_asset_id: str = "",
        override_reason: str = "",
    ) -> ToolResult:
        report, report_asset = run_check(
            self.checker, self.assets, content_asset_id, platform, media_asset_ids
        )
        asset, manifest = self.packager.build(
            content_asset_id,
            platform,
            title=title,
            tags=tags or [],
            media_asset_ids=media_asset_ids or [],
            cover_asset_id=cover_asset_id,
            report=report,
            report_asset_id=report_asset.id,
            override_reason=override_reason,
        )
        if self.bus is not None:
            await self.bus.emit(
                EventType.PACKAGE_BUILT,
                package=asset.id,
                platform=platform,
                kind=manifest.kind,
                passed=report.passed,
                issues=len(manifest.issues),
                folder=str(asset.uri),
            )
        return ToolResult(
            content=f"包资产 {asset.id} → {asset.uri}\n\n{self.assets.content(asset.id)}",
            asset_ref=asset.id,
        )

    async def _fn_mark_published(self, package_asset_id: str, url: str) -> ToolResult:
        if not url.strip():
            return ToolResult(ok=False, error="url 不能为空")
        m = self.packager.mark_published(package_asset_id, url)
        if self.bus is not None:
            await self.bus.emit(
                EventType.PACKAGE_PUBLISHED, package=package_asset_id, platform=m.platform, url=url
            )
        return ToolResult(
            content=f"{package_asset_id} 已标记为已发布：{url}（{m.platform}·{m.title}）",
            asset_ref=package_asset_id,
        )

    async def _fn_list_packages(self) -> ToolResult:
        rows = self.packager.packages()
        if not rows:
            return ToolResult(content="还没有待发布包")
        lines = [
            f"- {a.id} · {m.platform} · {m.kind} · {m.title[:24]} · {m.status}"
            + (f" · {m.published_url}" if m.published_url else "")
            for a, m in rows
        ]
        return ToolResult(content="\n".join(lines))

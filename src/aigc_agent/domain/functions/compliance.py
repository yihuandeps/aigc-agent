"""合规机审 function（M15 的模型侧入口）。

  check_compliance        对一份或一批资产跑机审，风险清单存成 report 资产（血缘挂在被审资产下）
  list_compliance_rules   看现在生效的规则（运营改 config/compliance.yaml 后可核对）

机审只出清单不改字。block 级问题会拦住打包，但正文一个字不动 —— 改是人的事。

2026-09-17：规则按资产类型生效（剧本台词不过广告法极限词），并加批量入口 ——
之前审 60 集要 60 次工具调用、每次一份完整报告回上下文，一轮 34 次调用把窗口撑爆。
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
from ..assets.store import Asset, AssetStore, AssetType, rights_of
from ..compliance import ComplianceChecker, ComplianceReport, report_to_params

MAX_BATCH = 100


def run_check(
    checker: ComplianceChecker,
    assets: AssetStore,
    asset_id: str,
    platform: str = "generic",
    media_asset_ids: list[str] | None = None,
) -> tuple[ComplianceReport, Asset]:
    """跑机审并落一份 report 资产。打包（M16）也走这里，两边一致。"""
    target = assets.get(asset_id)
    content = assets.content(asset_id)
    media_rights = [(mid, rights_of(assets.get(mid))) for mid in (media_asset_ids or [])]
    report = checker.check(
        content,
        platform=platform,
        generated=rights_of(target) == "generated",
        media_rights=media_rights,
        asset_id=asset_id,
        asset_type=target.type.value,
    )
    c = report.counts()
    verdict = "通过" if report.passed else f"block {c['block']}"
    asset = assets.create(
        report.render(),
        type_=AssetType.REPORT,
        summary=f"合规机审·{platform}·{verdict}·warn {c['warn']}",
        parents=[asset_id, *(media_asset_ids or [])],
        creator="tool:check_compliance",
        gen_params=report_to_params(report),
    )
    return report, asset


class ComplianceFunctions:
    name = "compliance"
    namespaced = False
    disclosure = "full"

    def __init__(
        self, checker: ComplianceChecker, assets: AssetStore, bus: EventBus | None = None
    ) -> None:
        self.checker = checker
        self.assets = assets
        self.bus = bus
        self._specs = {
            "check_compliance": ToolSpec(
                name="check_compliance",
                summary="发布前机审：违禁词、数字出处、AIGC 标识、素材版权、账号禁忌，只出风险清单",
                permission=PermissionLevel.WRITE,
                max_result_chars=12_000,
                description=(
                    "打包发布之前必跑。返回分级的风险清单（block 不改不能发 / warn 需人判断），"
                    "带行号、片段和改法。**它不会改正文**，改完再存一版重新跑。"
                    "审一批（如整部剧本）用 asset_ids，一次调用返回汇总，每份仍各落一份报告。"
                    "规则按资产类型生效：剧本（script）不过广告法极限词那一类。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "asset_id": {"type": "string", "description": "要审的文本资产 id"},
                        "asset_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "批量：一次审多份（最多 100），与 asset_id 二选一",
                        },
                        "platform": {
                            "type": "string",
                            "description": "douyin / xiaohongshu / wechat / zhihu / generic",
                        },
                        "media_asset_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "随稿使用的图/视频资产，用来查版权状态",
                        },
                    },
                },
            ),
            "list_compliance_rules": ToolSpec(
                name="list_compliance_rules",
                summary="看当前生效的机审规则与分级",
                permission=PermissionLevel.READ,
                parameters={"type": "object", "properties": {}},
            ),
        }

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        n = sum(len(r.words) for r in self.checker.rules.word_rules)
        version = self.checker.rules.version or "—"
        return ProviderHealth(ok=True, detail=f"规则 {version} · {n} 个词")

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            r = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
            r = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        r.duration_ms = int((time.perf_counter() - started) * 1000)
        return r

    async def _emit(self, asset_id: str, report: ComplianceReport, asset: Asset, platform: str):
        if self.bus is not None:
            await self.bus.emit(
                EventType.COMPLIANCE_CHECKED,
                asset=asset_id,
                report=asset.id,
                platform=platform,
                passed=report.passed,
                **report.counts(),
            )

    async def _fn_check_compliance(
        self,
        asset_id: str = "",
        asset_ids: list[str] | None = None,
        platform: str = "generic",
        media_asset_ids: list[str] | None = None,
    ) -> ToolResult:
        ids = [i for i in (asset_ids or []) if i] or ([asset_id] if asset_id else [])
        if not ids:
            return ToolResult(ok=False, error="要给 asset_id 或 asset_ids")
        if len(ids) > MAX_BATCH:
            return ToolResult(ok=False, error=f"一次最多审 {MAX_BATCH} 份（给了 {len(ids)} 份）")

        # 单份：完整报告
        if len(ids) == 1:
            report, asset = run_check(
                self.checker, self.assets, ids[0], platform, media_asset_ids
            )
            await self._emit(ids[0], report, asset, platform)
            return ToolResult(content=f"报告 {asset.id}\n{report.render()}", asset_ref=asset.id)

        # 批量：每份各落一份报告，回上下文的只有汇总 + block 明细
        lines: list[str] = []
        failed: list[str] = []
        blocks = warns = 0
        last: Asset | None = None
        for aid in ids:
            try:
                report, asset = run_check(self.checker, self.assets, aid, platform, media_asset_ids)
            except KeyError as e:
                failed.append(f"- {aid}：{e}")
                continue
            await self._emit(aid, report, asset, platform)
            last = asset
            c = report.counts()
            blocks += c["block"]
            warns += c["warn"]
            mark = "✓" if report.passed else "✗"
            target = self.assets.get(aid)
            line = (
                f"{mark} {aid}（{target.summary[:24]}）→ 报告 {asset.id}："
                f"block {c['block']} · warn {c['warn']}"
            )
            for f in report.blocks[:3]:
                line += f"\n    ✗ 第 {f.line} 行 {f.message}"
            if len(report.blocks) > 3:
                line += f"\n    …还有 {len(report.blocks) - 3} 条 block，见报告"
            lines.append(line)
        head = (
            f"批量机审（{platform}）{len(ids)} 份：block 共 {blocks} · warn 共 {warns}"
            + ("，全部通过" if not blocks and not failed else "")
        )
        body = "\n".join([head, *lines, *failed])
        return ToolResult(ok=bool(lines), content=body, asset_ref=last.id if last else None,
                          error=None if lines else body)

    async def _fn_list_compliance_rules(self) -> ToolResult:
        return ToolResult(content=self.checker.rules.summary())

"""检索 function（M9 的模型侧入口）+ 历史内容库来源。

`search_library` 一个入口查所有库：历史内容（Asset）、记忆（品牌资料 / 打回理由）、
素材库（MCP server）。返回带来源与版权状态的命中表，正文用 read_asset 按需取。

AssetSource 放在 L2 而不是 capabilities/retrieval：资产模型住在 L2，
L1 不能 import 它 —— 依赖方向单向向下，由 lint-imports 强制。
"""

from __future__ import annotations

import time
from typing import Any

from ...capabilities.retrieval import Hit, RetrievalHub, keyword_score
from ...harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
    ToolSpec,
)
from ..assets.store import AssetStore, AssetType, rights_of

_KIND = {
    AssetType.IMAGE: "image",
    AssetType.VIDEO: "video",
    AssetType.AUDIO: "audio",
    AssetType.SUBTITLE: "doc",
    AssetType.REPORT: "doc",
    AssetType.PACKAGE: "doc",
}


class AssetSource:
    """历史内容库：按摘要与内联正文做关键词匹配。二进制资产只匹配摘要。"""

    name = "assets"

    def __init__(self, store: AssetStore) -> None:
        self.store = store

    async def search(self, query: str, kind: str, limit: int) -> list[Hit]:
        if kind == "memory":
            return []
        hits: list[Hit] = []
        for a in self.store.find(newest_first=False):  # 当前项目、active（缺口 A）
            k = _KIND.get(a.type, "text")
            if kind != "all" and kind != k:
                continue
            text = f"{a.summary} {a.inline or ''}"
            s = keyword_score(query, text)
            if s <= 0:
                continue
            hits.append(
                Hit(
                    source=self.name,
                    ref=a.id,
                    title=a.brief(),
                    kind=k,
                    rights=rights_of(a),
                    score=s + 0.01 * a.version,  # 同分时新版本靠前
                    extra={"creator": a.creator, "version": a.version},
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]


class RetrievalFunctions:
    name = "retrieval"
    namespaced = False
    disclosure = "full"

    def __init__(self, hub: RetrievalHub) -> None:
        self.hub = hub
        self._specs = {
            "search_library": ToolSpec(
                name="search_library",
                summary="一个入口检索历史内容、记忆和素材库，返回带来源与版权状态的命中",
                permission=PermissionLevel.READ,
                description=(
                    "找素材、找以前做过的东西、查品牌资料都用这个。"
                    "结果只给引用和摘要：资产用 read_asset 取正文，素材用路径。"
                    "**版权状态是 unknown 的素材不要直接用进成片**，先问人。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "关键词，多个用空格分开"},
                        "kind": {
                            "type": "string",
                            "enum": ["all", "text", "image", "video", "audio", "doc", "memory"],
                            "description": "只要某一类时填，默认 all",
                        },
                        "limit": {"type": "integer", "description": "最多几条，默认 10"},
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "只查某几个来源（assets / memory / material_lib）",
                        },
                    },
                    "required": ["query"],
                },
            )
        }

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        names = ", ".join(s.name for s in self.hub.sources) or "无"
        return ProviderHealth(ok=bool(self.hub.sources), detail=f"来源：{names}")

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            r = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
            r = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        r.duration_ms = int((time.perf_counter() - started) * 1000)
        return r

    async def _fn_search_library(
        self,
        query: str,
        kind: str = "all",
        limit: int = 10,
        sources: list[str] | None = None,
    ) -> ToolResult:
        if not query.strip():
            return ToolResult(ok=False, error="query 不能为空")
        hits = await self.hub.search(
            query, kind=kind or "all", limit=max(1, min(int(limit or 10), 50)), sources=sources
        )
        head = f"「{query}」命中 {len(hits)} 条" + (f"（{kind}）" if kind != "all" else "")
        return ToolResult(content=head + "\n" + self.hub.render(hits))

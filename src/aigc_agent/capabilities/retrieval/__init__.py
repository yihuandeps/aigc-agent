"""M9 Retrieval —— 素材与知识的统一检索。

热点库、竞品库、历史内容库、素材库、品牌资料库，模型不该记住每个库各有各的查法。
这里给一个入口：`search_library(query, kind)` 扇出到所有已接的来源，合并成一张
带**来源**与**版权状态**的命中表。

P3 的形态是关键词检索，够内部工具起步；以图搜图、按情绪搜 BGM 这类多模态查询
要等向量/多模态索引，接口上已经给了 kind 这个口子，到时加来源即可，模型侧不变。

来源（RetrievalSource）是可插拔的：
  · MemorySource   记忆库（品牌资料 = 账号层记忆）       —— L1，这里
  · ToolSource     任何注册进 M4 的检索类工具（如素材库 MCP）—— L1，这里
  · AssetSource    历史内容库（Asset Store）               —— L2，domain/functions/retrieval.py
                                                             （资产模型住在 L2，L1 不能 import 它）

**每条命中都带 rights 字段**：generated（本系统生成）/ human（人给的）/ licensed /
unknown。素材库里的东西默认 unknown —— 版权说不清的素材不该悄悄进成片。
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ...harness.tools.registry import ToolRegistry
from ..memory.store import MemoryStore

MEDIA_KINDS = {"image", "video", "audio", "doc"}


class Hit(BaseModel):
    source: str
    ref: str  # asset id / 记忆 id / 素材相对路径
    title: str = ""
    kind: str = "text"  # text | image | video | audio | doc | memory | other
    rights: str = "unknown"  # generated | human | licensed | unknown | inferred | data
    score: float = 1.0
    extra: dict[str, Any] = Field(default_factory=dict)

    def line(self) -> str:
        return f"- [{self.source}] {self.ref} · {self.kind} · 版权:{self.rights} · {self.title}"


class RetrievalSource(Protocol):
    name: str

    async def search(self, query: str, kind: str, limit: int) -> list[Hit]: ...


def keyword_score(query: str, text: str) -> float:
    """关键词命中率。中文没空格：整句也算一个词，子串命中就给分。"""
    q = (query or "").strip()
    if not q or not text:
        return 0.0
    low = text.lower()
    terms = [t for t in re.split(r"[\s,，;；]+", q.lower()) if t]
    if not terms:
        return 0.0
    hit = sum(1 for t in terms if t in low)
    score = hit / len(terms)
    # 整句连着出现再加分。中文查询里的空格只是分词习惯，比对时去掉
    compact = re.sub(r"\s+", "", q.lower())
    if len(terms) > 1 and compact in low.replace(" ", ""):
        score += 0.5
    return score


class MemorySource:
    """品牌资料库 = 账号层记忆；项目层记忆也一并可查。"""

    name = "memory"

    def __init__(self, store: MemoryStore, project_id: str = "") -> None:
        self.store = store
        self.project_id = project_id

    async def search(self, query: str, kind: str, limit: int) -> list[Hit]:
        if kind not in ("all", "memory"):
            return []
        terms = self.store.match_terms(query)
        if not terms:
            return []
        hits = self.store.recall(terms, limit=limit, project_id=self.project_id)
        return [
            Hit(
                source=self.name,
                ref=m.id,
                title=m.content,
                kind="memory",
                rights=m.source.value,
                score=float(m.weight),
                extra={"layer": m.layer.value},
            )
            for m in hits
        ]


class ToolSource:
    """把任何注册进 M4 的检索类工具当作一个来源（典型：素材库 MCP server）。

    走 registry.invoke()：权限、计次、痕迹和别的调用一样，不开后门。
    工具没接上（server 没连）就当没有这个来源，不报错。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        tool: str,
        name: str = "",
        query_arg: str = "query",
        kind_arg: str = "kind",
        limit_arg: str = "limit",
        items_key: str = "items",
    ) -> None:
        self.registry = registry
        self.tool = tool
        self.name = name or tool.split("__", 1)[0]
        self.query_arg, self.kind_arg, self.limit_arg = query_arg, kind_arg, limit_arg
        self.items_key = items_key

    async def search(self, query: str, kind: str, limit: int) -> list[Hit]:
        if kind == "memory" or self.registry.meta(self.tool) is None:
            return []
        args = {
            self.query_arg: query,
            self.kind_arg: kind if kind in MEDIA_KINDS else "all",
            self.limit_arg: limit,
        }
        r = await self.registry.invoke(self.tool, args)
        if not r.ok:
            raise RuntimeError(r.error or "调用失败")
        data = _parse(r.content)
        items = data.get(self.items_key) if isinstance(data, dict) else data
        out: list[Hit] = []
        for it in items or []:
            if not isinstance(it, dict):
                continue
            ref = str(it.get("path") or it.get("id") or it.get("ref") or "")
            if not ref:
                continue
            out.append(
                Hit(
                    source=self.name,
                    ref=ref,
                    title=str(it.get("name") or it.get("title") or ref),
                    kind=str(it.get("kind") or "other"),
                    # 外部素材默认说不清版权。除非来源自己带了字段
                    rights=str(it.get("rights") or "unknown"),
                    extra={k: v for k, v in it.items() if k in ("size", "modified", "duration")},
                )
            )
        return out


def _parse(text: str) -> Any:
    raw = (text or "").strip()
    i = min([x for x in (raw.find("{"), raw.find("[")) if x >= 0], default=-1)
    if i < 0:
        return {}
    try:
        return json.loads(raw[i:])
    except json.JSONDecodeError:
        return {}


class RetrievalHub:
    """扇出到所有来源，合并、排序、截断。单个来源挂了不影响其余。"""

    def __init__(self, sources: list[RetrievalSource] | None = None) -> None:
        self.sources: list[RetrievalSource] = list(sources or [])
        self.last_errors: list[str] = []

    def add(self, source: RetrievalSource) -> None:
        self.sources.append(source)

    async def search(
        self,
        query: str,
        kind: str = "all",
        limit: int = 10,
        sources: list[str] | None = None,
    ) -> list[Hit]:
        picked = [s for s in self.sources if not sources or s.name in sources]
        if not picked or not query.strip():
            return []
        results = await asyncio.gather(
            *(s.search(query, kind, limit) for s in picked), return_exceptions=True
        )
        hits: list[Hit] = []
        errors: list[str] = []
        for s, r in zip(picked, results, strict=True):
            if isinstance(r, BaseException):
                errors.append(f"{s.name}: {type(r).__name__}: {r}")
            else:
                hits.extend(r)
        self.last_errors = errors
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def render(self, hits: list[Hit]) -> str:
        lines = [h.line() for h in hits]
        if self.last_errors:
            lines.append("（部分来源不可用：" + "；".join(self.last_errors) + "）")
        return "\n".join(lines) if lines else "没有匹配的结果"

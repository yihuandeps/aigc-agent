"""M9 统一检索验收。

一个入口查历史内容（Asset）、记忆（品牌资料/打回理由）、素材库（MCP 工具），
每条命中带来源与版权状态。单个来源挂了不影响其余。
"""

from __future__ import annotations

import json

from aigc_agent.capabilities.memory.store import (
    Category,
    Keyword,
    Layer,
    Memory,
    MemoryStore,
    Source,
)
from aigc_agent.capabilities.retrieval import (
    Hit,
    MemorySource,
    RetrievalHub,
    ToolSource,
    keyword_score,
)
from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions.retrieval import AssetSource, RetrievalFunctions
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.tools.provider import PermissionLevel, ProviderHealth, ToolMeta, ToolResult
from aigc_agent.harness.tools.registry import ToolRegistry


class _Lib:
    """假素材库（形状照 material_lib MCP server）。"""

    name = "material_lib"
    namespaced = True
    disclosure = "digest"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def list_tools(self):
        return [
            ToolMeta(name="search_materials", summary="搜素材", permission=PermissionLevel.READ)
        ]

    async def get_schema(self, tool):
        return {"type": "function", "function": {"name": tool, "parameters": {"type": "object"}}}

    async def invoke(self, tool, args):
        self.calls.append(args)
        return ToolResult(
            content=json.dumps(
                {
                    "items": [
                        {"path": "bgm/轻快.mp3", "name": "轻快.mp3", "kind": "audio", "size": 1},
                        {"path": "海边日落.mp4", "name": "海边日落.mp4", "kind": "video"},
                    ]
                },
                ensure_ascii=False,
            )
        )

    async def health(self):
        return ProviderHealth(ok=True)


class _Broken:
    name = "broken"

    async def search(self, query, kind, limit):
        raise RuntimeError("库炸了")


async def _hub():
    store = AssetStore()
    a1 = store.create(
        "露营装备种草文案 第一版", type_=AssetType.TEXT, summary="露营 文案", creator="model:k3"
    )
    a2 = store.revise(a1.id, "露营装备种草文案 第二版 更口语", creator="human:me")
    img = store.create("", type_=AssetType.IMAGE, summary="露营帐篷夜景", creator="model:qwen")
    store.create("无关的公众号长文", type_=AssetType.TEXT, summary="职场", creator="model:k3")

    memories = MemoryStore()
    memories.put(
        Memory(
            layer=Layer.ACCOUNT,
            content="品牌调性：露营内容也要讲安全",
            keywords=[Keyword(term="露营", category=Category.PREFERENCE)],
            source=Source.HUMAN,
        )
    )

    bus = EventBus()
    reg = ToolRegistry(bus)
    lib = _Lib()
    reg.register(lib)
    await reg.refresh()
    hub = RetrievalHub(
        [
            AssetSource(store),
            MemorySource(memories),
            ToolSource(reg, "material_lib__search_materials"),
        ]
    )
    return hub, store, (a1, a2, img), lib, reg


async def test_资产按关键词命中并标版权():
    hub, store, (a1, a2, img), _, _ = await _hub()
    hits = await hub.search("露营", sources=["assets"])
    by_ref = {h.ref: h for h in hits}
    assert {a1.id, a2.id, img.id} <= set(by_ref)
    assert by_ref[a1.id].rights == "generated"
    assert by_ref[a2.id].rights == "human"
    assert by_ref[img.id].kind == "image"
    assert all("职场" not in h.title for h in hits)
    assert hits[0].ref == a2.id, "同分时新版本靠前"


async def test_按类型过滤():
    hub, store, (a1, a2, img), _, _ = await _hub()
    assert [h.ref for h in await hub.search("露营", kind="image", sources=["assets"])] == [img.id]
    text = {h.ref for h in await hub.search("露营", kind="text", sources=["assets"])}
    assert text == {a1.id, a2.id}


async def test_记忆来源带层与来源():
    hub, *_ = await _hub()
    (h,) = await hub.search("露营", sources=["memory"])
    assert h.kind == "memory" and h.rights == "human" and h.extra["layer"] == "account"


async def test_素材库来源走注册表_版权默认unknown():
    hub, _, _, lib, _ = await _hub()
    hits = await hub.search("轻快", kind="audio", sources=["material_lib"])
    assert lib.calls == [{"query": "轻快", "kind": "audio", "limit": 10}]
    assert {h.ref for h in hits} == {"bgm/轻快.mp3", "海边日落.mp4"}
    assert all(h.rights == "unknown" for h in hits)
    assert next(h for h in hits if h.ref.endswith(".mp3")).kind == "audio"


async def test_素材工具没接上就当没有():
    _, _, _, _, reg = await _hub()
    assert await ToolSource(reg, "nope__search").search("x", "all", 5) == []


async def test_单个来源挂了不影响其余():
    hub, store, (a1, *_), _, _ = await _hub()
    hub.add(_Broken())
    hits = await hub.search("露营")
    assert any(h.ref == a1.id for h in hits)
    assert hub.last_errors and "broken" in hub.last_errors[0]
    assert "部分来源不可用" in hub.render(hits)


async def test_合并排序与截断():
    hub, *_ = await _hub()
    hits = await hub.search("露营", limit=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


async def test_function入口():
    hub, *_ = await _hub()
    bus = EventBus()
    reg = ToolRegistry(bus)
    reg.register(RetrievalFunctions(hub))
    await reg.refresh()
    assert reg.meta("search_library").permission is PermissionLevel.READ
    r = await reg.invoke("search_library", {"query": "露营"})
    assert r.ok and "命中" in r.content and "版权:" in r.content and "[assets]" in r.content
    r = await reg.invoke("search_library", {"query": "露营", "kind": "memory"})
    assert "[assets]" not in r.content and "[memory]" in r.content
    r = await reg.invoke("search_library", {"query": "  "})
    assert not r.ok
    schema = (await reg.schemas(["search_library"]))[0]
    assert "unknown" in schema["function"]["description"]


def test_keyword_score():
    assert keyword_score("露营 装备", "露营装备种草") == 1.5
    assert keyword_score("露营", "职场") == 0.0
    assert keyword_score("", "x") == 0.0
    assert Hit(source="s", ref="r", title="t").line().startswith("- [s] r")

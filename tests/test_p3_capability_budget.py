"""M7 Skill Hub + 能力预算验收。

P3 验收原话：「运营改 markdown 即可影响输出；能力区占用受控在 30K 内」。

之前的实际状况：skill 目录**从没进过上下文**（catalog_digest 没有任何调用方），
模型不知道有哪些方法论可以加载；正文作为工具返回值进对话历史，会随滑窗被剔掉，
预算超了也没人能卸它。这组测试盯：
  1. 目录 pin 进系统区；正文 pin 进系统区而不是塞进工具返回值
  2. 超预算按 drop → collapse → shrink 的顺序降级，合规 skill 永不被挤出
  3. 改文件 → 下一轮生效（热加载）；版本有记录；回滚一键
  4. 真实配置 + 真实 skills 目录下，能力区在 30K 内
"""

from __future__ import annotations

from pathlib import Path

from aigc_agent.capabilities.capability_budget import (
    BODIES_PIN,
    DIGEST_PIN,
    BudgetConfig,
    CapabilityAllocator,
)
from aigc_agent.capabilities.skill_hub import SkillHub
from aigc_agent.capabilities.skill_hub.functions import SkillFunctions
from aigc_agent.harness.context.window import ShortTermMemory
from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.tools.builtin import builtin
from aigc_agent.harness.tools.disclosure import DisclosureProvider
from aigc_agent.harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
)
from aigc_agent.harness.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]


def _skill_md(name: str, priority: int = 50, body_chars: int = 600, status: str = "active") -> str:
    body = "这是方法论正文，讲清楚该怎么做。" * max(1, body_chars // 16)
    return (
        f"---\nname: {name}\ndescription: {name} 的方法论，做 {name} 相关内容时先读\n"
        f"scope: content_type\napplies_to: [短视频]\nstage: []\npriority: {priority}\n"
        f"version: 1.0.0\nowner: 测试\nstatus: {status}\n---\n\n# {name}\n\n{body}\n"
    )


def _make_skills(tmp_path: Path, specs: list[tuple[str, dict]]) -> Path:
    d = tmp_path / "skills"
    d.mkdir()
    for name, kw in specs:
        (d / f"{name}.md").write_text(_skill_md(name, **kw), encoding="utf-8")
    return d


class _Ext:
    """外部风格的 provider：digest 披露，schema 很大。"""

    name = "ext"
    namespaced = True
    disclosure = "digest"

    async def list_tools(self):
        return [
            ToolMeta(name=n, summary=f"{n} 工具", permission=PermissionLevel.READ, provider="ext")
            for n in ("alpha", "beta", "gamma")
        ]

    async def get_schema(self, tool):
        return {
            "type": "function",
            "function": {
                "name": tool,
                "description": "很长的说明。" * 80,
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        }

    async def invoke(self, tool, args):
        return ToolResult(content="ok")

    async def health(self):
        return ProviderHealth(ok=True)


async def _setup(tmp_path: Path, specs: list[tuple[str, dict]], **cfg):
    hub = SkillHub(_make_skills(tmp_path, specs), history_dir=tmp_path / "hist")
    hub.load()
    bus = EventBus()
    reg = ToolRegistry(bus)
    reg.register(builtin)
    reg.register(DisclosureProvider(reg))
    reg.register(_Ext())
    await reg.refresh()
    memory = ShortTermMemory()
    alloc = CapabilityAllocator(BudgetConfig(**cfg), reg, hub, memory, bus)
    alloc.attach(bus)
    return hub, reg, memory, alloc, bus


# ---------------------------------------------------------------- pin


async def test_目录pin进系统区(tmp_path: Path):
    hub, reg, memory, alloc, bus = await _setup(tmp_path, [("a", {}), ("b", {})])
    await alloc.refresh()
    pin = memory.pins[DIGEST_PIN]
    assert pin.position == "system"
    assert "- a" in pin.content and "- b" in pin.content and "load_skill" in pin.content
    assert any(e.type is EventType.CAPABILITY_BUDGET for e in bus.history)


async def test_加载正文走pin而不是工具返回值(tmp_path: Path):
    hub, reg, memory, alloc, bus = await _setup(tmp_path, [("a", {})])
    await alloc.refresh()
    fns = SkillFunctions(hub, on_load=alloc.activate_skill)
    r = await fns.invoke("load_skill", {"names": ["a"]})
    assert r.ok, r.error
    assert "已加载" in r.content and "常驻" in r.content
    assert "这是方法论正文" not in r.content, "正文不该重复出现在工具返回值里"
    body = memory.pins[BODIES_PIN]
    assert body.position == "system" and "这是方法论正文" in body.content
    assert "a" in alloc.active


async def test_没接预算时保持旧行为_正文在返回值里(tmp_path: Path):
    hub, *_ = await _setup(tmp_path, [("a", {})])
    r = await SkillFunctions(hub).invoke("load_skill", {"names": ["a"]})
    assert "这是方法论正文" in r.content


# ---------------------------------------------------------------- 降级


async def test_超预算先卸最低优先级的skill(tmp_path: Path):
    hub, reg, memory, alloc, bus = await _setup(
        tmp_path, [("low", {"priority": 40}), ("high", {"priority": 60})]
    )
    await alloc.refresh()
    await alloc.activate_skill(hub.get("low"))
    usage = await alloc.measure()
    # 预算刚好容得下一篇，第二篇进来必须挤出一篇
    alloc.cfg.capability_budget = usage.total + 50
    note = await alloc.activate_skill(hub.get("high"))
    assert "high" in alloc.active and "low" not in alloc.active
    assert "卸下 skill low" in note
    assert alloc.last is not None and alloc.last.total <= alloc.cfg.capability_budget
    ev = [e for e in bus.history if e.type is EventType.CAPABILITY_BUDGET][-1]
    assert ev.data["actions"] and ev.data["active_skills"] == ["high"]


async def test_合规skill永不被挤出(tmp_path: Path):
    hub, reg, memory, alloc, bus = await _setup(
        tmp_path, [("plain", {"priority": 50}), ("compliance", {"priority": 100})]
    )
    await alloc.refresh()
    await alloc.activate_skill(hub.get("compliance"))
    await alloc.activate_skill(hub.get("plain"))
    alloc.cfg.capability_budget = 10  # 怎么都超
    await alloc.enforce()
    assert "compliance" in alloc.active and "plain" not in alloc.active
    warns = [e for e in bus.history if e.type is EventType.WARNING]
    assert any("无可降级项" in str(e.data.get("message")) for e in warns)


async def test_同时激活上限(tmp_path: Path):
    hub, reg, memory, alloc, bus = await _setup(
        tmp_path, [("a", {"priority": 40}), ("b", {"priority": 60})], max_active_skills=1
    )
    await alloc.refresh()
    await alloc.activate_skill(hub.get("a"))
    note = await alloc.activate_skill(hub.get("b"))
    assert list(alloc.active) == ["b"] and "同时激活上限" in note


async def test_收回展开的外部工具schema(tmp_path: Path):
    hub, reg, memory, alloc, bus = await _setup(
        tmp_path, [("a", {})], on_over_budget=["collapse_expanded_tools"]
    )
    await alloc.refresh()
    ok, _ = reg.expand("ext__alpha")
    assert ok and reg.expanded == ["ext__alpha"]
    usage = await alloc.measure()
    assert usage.expanded_tool_schema > 0
    alloc.cfg.capability_budget = usage.total - 1
    actions = await alloc.enforce()
    assert reg.expanded == [] and any("收回工具 ext__alpha" in a for a in actions)


async def test_展开工具后自动核预算(tmp_path: Path):
    """模型调 load_tool_schema 展开之后，分配器通过事件总线收到通知。"""
    hub, reg, memory, alloc, bus = await _setup(tmp_path, [("a", {})])
    await alloc.refresh()
    before = len([e for e in bus.history if e.type is EventType.CAPABILITY_BUDGET])
    r = await reg.invoke("load_tool_schema", {"names": ["ext__beta"]})
    assert r.ok
    after = len([e for e in bus.history if e.type is EventType.CAPABILITY_BUDGET])
    assert after == before + 1


async def test_最后一级截断skill目录(tmp_path: Path):
    specs = [(f"s{i}", {"priority": 10 + i}) for i in range(6)]
    hub, reg, memory, alloc, bus = await _setup(
        tmp_path, specs, on_over_budget=["shrink_skill_digest"]
    )
    await alloc.refresh()
    full = memory.pins[DIGEST_PIN].content
    usage = await alloc.measure()
    alloc.cfg.capability_budget = usage.total - 1
    await alloc.enforce()
    shrunk = memory.pins[DIGEST_PIN].content
    assert alloc.digest_limit and alloc.digest_limit < 6
    assert len(shrunk) < len(full)
    assert "- s5" in shrunk, "截断保留优先级最高的"


async def test_单篇超长提示拆分(tmp_path: Path):
    hub, reg, memory, alloc, bus = await _setup(tmp_path, [("huge", {"body_chars": 40000})])
    await alloc.refresh()
    note = await alloc.activate_skill(hub.get("huge"))
    assert "建议拆分" in note


# ---------------------------------------------------------------- 热加载 / 版本 / 回滚


async def test_改文件下一轮生效并记录版本(tmp_path: Path):
    hub, reg, memory, alloc, bus = await _setup(tmp_path, [("a", {})])
    await alloc.refresh()
    await alloc.activate_skill(hub.get("a"))
    old_sha = hub.get("a").sha
    assert len(hub.history("a")) == 1

    path = tmp_path / "skills" / "a.md"
    extra = "\n新增的一条规矩：结尾不喊口号。\n"
    path.write_text(path.read_text(encoding="utf-8") + extra, encoding="utf-8")
    changed = await alloc.refresh()
    assert changed == ["a"]
    assert hub.get("a").sha != old_sha
    assert "结尾不喊口号" in memory.pins[BODIES_PIN].content, "已激活的正文要换成新版"
    assert len(hub.history("a")) == 2
    assert any(e.type is EventType.SKILL_RELOAD for e in bus.history)
    assert not await alloc.refresh(), "没改就不重复报"


async def test_一键回滚(tmp_path: Path):
    hub = SkillHub(_make_skills(tmp_path, [("a", {})]), history_dir=tmp_path / "hist")
    hub.load()
    original = (tmp_path / "skills" / "a.md").read_text(encoding="utf-8")
    old_sha = hub.get("a").sha
    (tmp_path / "skills" / "a.md").write_text(original + "\n改坏了\n", encoding="utf-8")
    assert hub.refresh_if_changed() == ["a"]

    rolled = hub.rollback("a")
    assert rolled.sha == old_sha
    assert (tmp_path / "skills" / "a.md").read_text(encoding="utf-8") == original
    assert hub.get("a").sha == old_sha


async def test_改成draft就从目录和激活集里消失(tmp_path: Path):
    hub, reg, memory, alloc, bus = await _setup(tmp_path, [("a", {}), ("b", {})])
    await alloc.refresh()
    await alloc.activate_skill(hub.get("a"))
    path = tmp_path / "skills" / "a.md"
    drafted = path.read_text(encoding="utf-8").replace("status: active", "status: draft")
    path.write_text(drafted, encoding="utf-8")
    await alloc.refresh()
    assert "a" not in alloc.active
    assert "- a" not in memory.pins[DIGEST_PIN].content and "- b" in memory.pins[DIGEST_PIN].content


# ---------------------------------------------------------------- 真实配置


def test_默认配置文件可加载():
    cfg = BudgetConfig.load(ROOT / "config" / "capability_budget.yaml")
    assert cfg.capability_budget == 30_000
    assert cfg.max_active_skills == 3 and cfg.max_expanded_tools == 8
    assert cfg.on_over_budget[0] == "drop_lowest_priority_skill"
    assert cfg.warn_at_ratio == 0.9


async def test_真实装配下能力区在预算内():
    from aigc_agent.app import Agent

    agent = Agent.create()
    try:
        await agent.setup(mcp=False)
        assert agent.allocator is not None and agent.allocator.last is not None
        assert agent.allocator.last.total <= 30_000
        assert DIGEST_PIN in agent.memory.pins, "skill 目录进了系统区"
        assert "drama-script" in agent.memory.pins[DIGEST_PIN].content
    finally:
        await agent.aclose()

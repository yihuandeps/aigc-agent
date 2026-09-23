"""Cost Guard 接线验收。

budget.py 早就写好了，但在 P2 收尾前**没接任何地方** —— 没有一处 import 它。
这组测试盯的是"接上了"这件事本身：

  · 媒体调用在权限闸门处计次，超限先问人，无人可问就拒
  · 文本花费从 COST 事件累计，媒体金额（目录有单价时）也从 COST 事件补
  · Loop 在累计口径超限时**挂起交给人**，不静默降级
  · 人点头只放行这一次，不改配置
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.functions.audio import AudioFunctions
from aigc_agent.domain.functions.media import MediaFunctions
from aigc_agent.domain.generators.catalog import MediaCatalog
from aigc_agent.harness.context.assembler import ContextAssembler
from aigc_agent.harness.context.window import ShortTermMemory
from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.execution.loop import LoopRuntime, StopReason
from aigc_agent.harness.model.audio import AudioGateway, FakeAudioProvider
from aigc_agent.harness.model.budget import DEFAULT_CALL_LIMITS, CostGuard
from aigc_agent.harness.model.config import CostGuard as CostGuardConfig
from aigc_agent.harness.model.gateway import ModelResponse, ToolCall, Usage
from aigc_agent.harness.model.media import FakeMediaProvider, MediaGateway, MediaKind
from aigc_agent.harness.permission.gate import PermissionGate
from aigc_agent.harness.tools.dispatcher import ToolDispatcher
from aigc_agent.harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolResult,
    ToolSpec,
)
from aigc_agent.harness.tools.registry import ToolRegistry

CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "media_models.yaml"


class _Paid:
    """假 Provider：一个计次的花钱工具，一个不计次的 compute 工具，一个只读工具。"""

    name = "paid"
    namespaced = False

    def __init__(self) -> None:
        self.specs = {
            "gen": ToolSpec(
                name="gen",
                summary="花钱生图",
                permission=PermissionLevel.COMPUTE,
                cost_kind="image",
            ),
            "free": ToolSpec(name="free", summary="本地合成", permission=PermissionLevel.COMPUTE),
            "look": ToolSpec(name="look", summary="只读", permission=PermissionLevel.READ),
        }
        self.calls: list[str] = []

    async def list_tools(self):
        return [s.meta(self.name) for s in self.specs.values()]

    async def get_schema(self, tool):
        return self.specs[tool].to_openai(tool)

    async def invoke(self, tool, args):
        self.calls.append(tool)
        return ToolResult(content=f"{tool} ok")

    async def health(self):
        return ProviderHealth(ok=True)


async def _registry(guard: CostGuard, asker=None):
    bus = EventBus()
    registry = ToolRegistry(bus)
    paid = _Paid()
    registry.register(paid)
    await registry.refresh()
    registry.gate = PermissionGate(bus, asker=asker, guard=guard)
    return registry, paid, bus


# ---------------------------------------------------------------- 闸门计次


async def test_计次工具超限后被闸门拦下():
    guard = CostGuard(call_limits={"image": 2})
    registry, paid, bus = await _registry(guard)

    assert (await registry.invoke("gen", {})).ok
    assert (await registry.invoke("gen", {})).ok
    third = await registry.invoke("gen", {})
    assert not third.ok
    assert "预算护栏" in third.error and "image" in third.error

    assert paid.calls == ["gen", "gen"], "拦下的那次不能真的发出去"
    assert guard.usage.calls["image"] == 2
    assert any(e.type is EventType.BUDGET_EXCEEDED for e in bus.history)


async def test_不计次的compute和只读工具不受影响():
    guard = CostGuard(call_limits={"image": 0})  # 图一张都不许
    registry, paid, _ = await _registry(guard)

    assert not (await registry.invoke("gen", {})).ok
    for _ in range(5):
        assert (await registry.invoke("free", {})).ok
        assert (await registry.invoke("look", {})).ok
    assert "image" not in guard.usage.calls
    assert guard.usage.calls == {}, "不带 cost_kind 的工具不进计数"


async def test_超限时问人_点头只放行这一次():
    asked: list[Any] = []

    async def yes(meta, args):
        asked.append(meta)
        return True

    guard = CostGuard(call_limits={"image": 1})
    registry, paid, _ = await _registry(guard, asker=yes)

    assert (await registry.invoke("gen", {})).ok  # 额度内，不问
    assert (await registry.invoke("gen", {})).ok  # 超限，问了，放行一次
    assert (await registry.invoke("gen", {})).ok  # 又超了，再问
    assert len(asked) == 2
    assert "预算护栏" in asked[0].summary, "问人时要说清楚是为什么"
    assert guard.limit_for("image") == 3, "临时额度只加一次"
    assert guard.call_limits["image"] == 1, "配置不动"
    assert paid.calls == ["gen"] * 3


async def test_人拒绝则拦下():
    async def no(meta, args):
        return False

    guard = CostGuard(call_limits={"image": 0})
    registry, paid, _ = await _registry(guard, asker=no)
    r = await registry.invoke("gen", {})
    assert not r.ok and "未放行" in r.error
    assert paid.calls == []


async def test_无人可问时直接拒绝():
    guard = CostGuard(call_limits={"image": 0})
    registry, paid, bus = await _registry(guard)
    r = await registry.invoke("gen", {})
    assert not r.ok
    denies = [e for e in bus.history if e.type is EventType.PERMISSION_DENY]
    assert denies and "预算护栏" in str(denies[-1].data.get("reason"))


async def test_deny策略下超限不问人():
    asked = []

    async def yes(meta, args):
        asked.append(meta)
        return True

    guard = CostGuard(call_limits={"image": 0}, on_exceed="deny")
    registry, _, _ = await _registry(guard, asker=yes)
    assert not (await registry.invoke("gen", {})).ok
    assert asked == []


# ---------------------------------------------------------------- 事件总线记账


async def test_文本花费从COST事件累计():
    bus = EventBus()
    guard = CostGuard()
    guard.attach(bus)

    await bus.emit(EventType.COST, cost=0.5, prompt_tokens=100, completion_tokens=10)
    await bus.emit(EventType.COST, cost=None, prompt_tokens=100, completion_tokens=10)
    assert guard.usage.money == 0.5
    assert guard.usage.unpriced == 1
    assert guard.usage.calls["text"] == 2


async def test_媒体金额从COST事件补但不重复计次():
    bus = EventBus()
    guard = CostGuard()
    guard.attach(bus)
    await bus.emit(EventType.COST, modality="image", model="x", cost=1.2)
    assert guard.usage.money == 1.2
    assert guard.usage.calls.get("image", 0) == 0, "次数在闸门记过了，这里只补金额"


async def test_目录填了单价就写进资产并发COST事件():
    catalog = MediaCatalog.load(CATALOG_PATH)
    model = catalog.get(MediaKind.IMAGE, catalog.image_defaults["fast"])
    assert model is not None
    model.price = 0.1

    bus = EventBus()
    guard = CostGuard()
    guard.attach(bus)
    store = AssetStore()
    gw = MediaGateway(
        {"apimart": FakeMediaProvider(urls=["https://x/a.png", "https://x/b.png"])},
        bus, poll_interval=0.01, max_poll_interval=0.02,
    )
    registry = ToolRegistry(bus)
    registry.register(MediaFunctions(gw, catalog, store))
    await registry.refresh()
    registry.gate = PermissionGate(bus, guard=guard)

    r = await registry.invoke("gen_image", {"prompt": "x", "prefer": "fast", "n": 2})
    assert r.ok, r.error
    assert store.get(r.asset_ref).gen_cost == 0.1
    assert abs(guard.usage.money - 0.2) < 1e-9, "两张图 × 0.1"
    assert guard.usage.calls["image"] == 1, "一次调用计一次，不按张数"


async def test_没填单价只计次不记金额():
    catalog = MediaCatalog.load(CATALOG_PATH)
    bus = EventBus()
    guard = CostGuard()
    guard.attach(bus)
    store = AssetStore()
    gw = MediaGateway(
        {"apimart": FakeMediaProvider()}, bus, poll_interval=0.01, max_poll_interval=0.02
    )
    registry = ToolRegistry(bus)
    registry.register(MediaFunctions(gw, catalog, store))
    await registry.refresh()
    registry.gate = PermissionGate(bus, guard=guard)

    r = await registry.invoke("gen_video", {"prompt": "x", "prefer": "fast"})
    assert r.ok, r.error
    assert guard.usage.calls["video"] == 1
    assert guard.usage.money == 0.0
    assert not any(e.type is EventType.COST for e in bus.history)


# ---------------------------------------------------------------- function 带上分类


async def test_媒体与语音function带cost_kind():
    catalog = MediaCatalog.load(CATALOG_PATH)
    bus = EventBus()
    registry = ToolRegistry(bus)
    store = AssetStore()
    registry.register(
        MediaFunctions(MediaGateway({"apimart": FakeMediaProvider()}, bus), catalog, store)
    )
    fake = FakeAudioProvider()
    registry.register(
        AudioFunctions(
            AudioGateway({catalog.provider: fake, catalog.speech_provider: fake}, bus),
            catalog, store,
        )
    )
    await registry.refresh()
    kinds = {m.name: m.cost_kind for m in registry.catalog()}
    assert kinds["gen_image"] == "image"
    assert kinds["gen_video"] == "video"
    assert kinds["tts"] == "audio"
    assert kinds["transcribe"] == "audio"
    assert kinds["list_media_models"] == ""  # 只读不计


# ---------------------------------------------------------------- Loop 挂起


class _Gw:
    def __init__(self) -> None:
        self.n = 0

    async def chat(self, role, messages, tools=None, **kw):
        self.n += 1
        if self.n == 1:
            return ModelResponse(
                tool_calls=[ToolCall(id="c1", name="look", arguments="{}")], usage=Usage(5, 5)
            )
        return ModelResponse(text="完成", usage=Usage(5, 5))


async def test_累计金额超限时loop挂起交给人():
    bus = EventBus()
    guard = CostGuard(money_limit=1.0)
    guard.attach(bus)
    registry = ToolRegistry(bus)
    registry.register(_Paid())
    await registry.refresh()
    gate = PermissionGate(bus, guard=guard)
    loop = LoopRuntime(
        gateway=_Gw(),  # type: ignore[arg-type]
        registry=registry,
        dispatcher=ToolDispatcher(registry, gate, bus),
        assembler=ContextAssembler(bus),
        memory=ShortTermMemory(),
        bus=bus,
        guard=guard,
    )
    # 之前的媒体调用已经把钱花到上限
    await bus.emit(EventType.COST, modality="video", cost=1.5)

    r = await loop.run_turn("继续")
    assert r.stop_reason is StopReason.BUDGET_EXCEEDED
    assert "达到单次任务上限" in r.text
    stops = [e for e in bus.history if e.type is EventType.LOOP_STOP_REASON]
    assert stops and stops[-1].data["reason"] == "budget_exceeded"


async def test_不超限时loop正常跑完():
    bus = EventBus()
    guard = CostGuard(money_limit=1.0)
    guard.attach(bus)
    registry = ToolRegistry(bus)
    registry.register(_Paid())
    await registry.refresh()
    loop = LoopRuntime(
        gateway=_Gw(),  # type: ignore[arg-type]
        registry=registry,
        dispatcher=ToolDispatcher(registry, PermissionGate(bus, guard=guard), bus),
        assembler=ContextAssembler(bus),
        memory=ShortTermMemory(),
        bus=bus,
        guard=guard,
    )
    r = await loop.run_turn("继续")
    assert r.stop_reason is StopReason.NO_TOOL_CALLS


# ---------------------------------------------------------------- 配置


def test_from_config合并默认与覆盖():
    cfg = CostGuardConfig(per_task_limit=3.0, call_limits={"video": 3}, on_exceed="deny")
    g = CostGuard.from_config(cfg)
    assert g.money_limit == 3.0
    assert g.limit_for("video") == 3
    assert g.limit_for("image") == DEFAULT_CALL_LIMITS["image"]
    assert g.on_exceed == "deny"


def test_reset清空用量并撤销临时加量():
    g = CostGuard(call_limits={"image": 1})
    g.record_call("image")
    g.allow_more("image")
    assert g.limit_for("image") == 2
    g.reset()
    assert g.usage.calls == {} and g.limit_for("image") == 1


def test_brief可读():
    g = CostGuard(money_limit=10, call_limits={"image": 24})
    g.record("image")
    g.record("text", 0.25)
    s = g.brief()
    assert "¥0.2500" in s and "image 1 次" in s and "≤24" in s

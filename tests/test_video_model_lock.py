"""视频模型锁（2026-09-20，用户定的规则）：切换视频模型之前必须先问用户。

生图模型 2026-09-22 起走同一套门（见 test_image_model_lock.py），这里只钉视频那侧。

真实事故（20260921-000907 会话）：短剧用的是用户指定的 seedance-2.0；模型补段时 gen_videos 的
model 留空、prefer=quality，自动选型换成了 veo3.1-quality —— 用户没同意过，质量与体验都很差。

规则落在媒体层，不靠模型自觉：
  · 本会话锁定一个视频模型（会话快照记住的 > 短剧配置里用户指定的 > 第一次用的）
  · model 留空一律用锁定的，不再按 prefer 自动选型
  · 传了不同的模型 → 挂起问人（major，/auto 也停）；人采纳（总线决策事件）才换锁，打回不换
  · 短剧链的渲染模型跟着锁走；锁写回会话快照，重启沿用
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aigc_agent.capabilities.memory.session import SessionSnapshot
from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.functions.drama import DramaFunctions
from aigc_agent.domain.functions.media import VIDEO_SWITCH_STAGE, MediaFunctions
from aigc_agent.domain.generators.catalog import MediaCatalog
from aigc_agent.domain.output import OutputPrefs
from aigc_agent.harness.context.assembler import ContextAssembler
from aigc_agent.harness.context.window import ShortTermMemory
from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.execution.loop import LoopRuntime, StopReason
from aigc_agent.harness.model.gateway import ModelResponse, ToolCall, Usage
from aigc_agent.harness.model.media import FakeMediaProvider, MediaGateway, MediaKind
from aigc_agent.harness.permission.gate import PermissionGate
from aigc_agent.harness.tools.dispatcher import ToolDispatcher
from aigc_agent.harness.tools.registry import ToolRegistry

CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "media_models.yaml"


def _media(tmp_path: Path, bus: EventBus | None = None) -> tuple[MediaFunctions, FakeMediaProvider]:
    provider = FakeMediaProvider(urls=["https://example.com/out.mp4"])
    gw = MediaGateway(
        {"apimart": provider}, bus or EventBus(), poll_interval=0.01, max_poll_interval=0.02
    )

    async def dl(url: str, dest: Path) -> bool:
        return False

    fns = MediaFunctions(gw, MediaCatalog.load(CATALOG_PATH), AssetStore(tmp_path / "a"),
                         prefs=OutputPrefs(tmp_path / "out", downloader=dl))
    return fns, provider


def _decided(decision: str) -> Any:
    return SimpleNamespace(
        type=EventType.CHECKPOINT_DECIDED,
        data={"node": VIDEO_SWITCH_STAGE, "decision": decision, "decided_by": "human"},
    )


# ---------------------------------------------------------------- 媒体层


async def test_留空用锁定的模型_不再自动选型(tmp_path: Path):
    fns, provider = _media(tmp_path)
    fns.video_lock = "seedance-2.0"
    r = await fns._fn_gen_video("海边日落", prefer="quality", allow_no_refs=True)
    assert r.ok, r.error
    assert provider.submitted[-1]["model"] == "seedance-2.0", "锁定优先于 quality 档默认"
    r = await fns._fn_gen_video("海边日落", model="seedance-2.0", allow_no_refs=True)
    assert r.ok and provider.submitted[-1]["model"] == "seedance-2.0"
    assert "已锁定" not in r.content, "本来就锁着，不用再说一遍"


async def test_要换模型_挂起问人_采纳才换_打回不换(tmp_path: Path):
    fns, provider = _media(tmp_path)
    fns.video_lock = "seedance-2.0"
    saved: list[str] = []
    fns.on_video_lock = saved.append

    r = await fns._fn_gen_video("补镜15", model="veo3.1-quality", summary="第11集镜15",
                                allow_no_refs=True)
    assert r.suspend and not provider.submitted, "没问人之前一段都不生成"
    p = r.suspend_payload
    assert p["stage"] == VIDEO_SWITCH_STAGE and p["major"] is True
    assert p["model_from"] == "seedance-2.0" and p["model_to"] == "veo3.1-quality"
    assert "seedance-2.0 换成 veo3.1-quality" in p["question"] and "第11集镜15" in p["question"]
    assert "回复 a 同意换" in p["question"]
    assert "已暂停等待用户决定" in r.content
    assert fns.video_lock == "seedance-2.0"
    assert fns._pending_switch == {VIDEO_SWITCH_STAGE: "veo3.1-quality"}

    # 人打回：不换，锁不动
    fns.on_event(_decided("revise"))
    assert fns.video_lock == "seedance-2.0" and not fns._pending_switch and saved == []
    r = await fns._fn_gen_video("补镜15", allow_no_refs=True)
    assert r.ok and provider.submitted[-1]["model"] == "seedance-2.0"

    # 再问一次，人采纳：换锁并持久化，之后留空也用新模型
    r = await fns._fn_gen_video("补镜15", model="veo3.1-quality", allow_no_refs=True)
    assert r.suspend
    fns.on_event(_decided("adopt"))
    assert fns.video_lock == "veo3.1-quality" and saved == ["veo3.1-quality"]
    r = await fns._fn_gen_video("补镜15", model="veo3.1-quality", allow_no_refs=True)
    assert r.ok and provider.submitted[-1]["model"] == "veo3.1-quality"
    r = await fns._fn_gen_video("补镜16", allow_no_refs=True)
    assert r.ok and provider.submitted[-1]["model"] == "veo3.1-quality"

    # 别的环节的决策事件不相干
    fns._pending_switch = {VIDEO_SWITCH_STAGE: "seedance-2.5"}
    fns.on_event(SimpleNamespace(type=EventType.CHECKPOINT_DECIDED,
                                 data={"node": "正文", "decision": "adopt"}))
    assert fns.video_lock == "veo3.1-quality"
    fns.on_event(SimpleNamespace(type=EventType.LOOP_END, data={}))
    assert fns.video_lock == "veo3.1-quality"

    # 不认识的模型：直接报错，不挂起
    r = await fns._fn_gen_video("x", model="不存在的模型", allow_no_refs=True)
    assert not r.ok and not r.suspend and "未知模型" in r.error


async def test_没锁定时第一次用的成为锁(tmp_path: Path):
    fns, provider = _media(tmp_path)
    saved: list[str] = []
    fns.on_video_lock = saved.append
    r = await fns._fn_gen_video("x", model="seedance-2.5", allow_no_refs=True)
    assert r.ok and fns.video_lock == "seedance-2.5" and saved == ["seedance-2.5"]
    assert "视频模型已锁定为 seedance-2.5" in r.content and "要换会先问用户" in r.content

    fns2, provider2 = _media(tmp_path / "b")
    r2 = await fns2._fn_gen_video("x", prefer="fast", allow_no_refs=True)
    fast = fns2.catalog.defaults(MediaKind.VIDEO)["fast"]
    assert r2.ok and fns2.video_lock == fast and provider2.submitted[-1]["model"] == fast


async def test_批量生成_任一项要换模型整批先问人(tmp_path: Path):
    fns, provider = _media(tmp_path)
    fns.video_lock = "seedance-2.0"
    r = await fns._fn_gen_videos(
        [{"prompt": "a", "summary": "镜1"},
         {"prompt": "b", "summary": "镜2", "model": "veo3.1-quality"}],
        allow_no_refs=True,
    )
    assert r.suspend and not provider.submitted
    assert r.suspend_payload["model_to"] == "veo3.1-quality"
    assert "镜2" in r.suspend_payload["question"]

    r = await fns._fn_gen_videos([{"prompt": "a"}, {"prompt": "b"}], prefer="quality",
                                 allow_no_refs=True)
    assert r.ok, r.error
    assert [s["model"] for s in provider.submitted] == ["seedance-2.0", "seedance-2.0"]

    r = await fns._fn_gen_videos([{"prompt": "c"}], model="veo3.1-quality", allow_no_refs=True)
    assert r.suspend and len(provider.submitted) == 2


# ---------------------------------------------------------------- 短剧链跟着锁走


def test_短剧渲染模型跟着会话锁走():
    fns = DramaFunctions(None, AssetStore(), registry=None,
                         catalog=SimpleNamespace(drama={"video_model": "seedance-2.0"}))
    assert fns.video_model == "seedance-2.0"
    fns.video_lock_source = lambda: "veo3.1-quality"
    assert fns.video_model == "veo3.1-quality", "用户同意换了，整条链一起换"
    fns.video_lock_source = lambda: ""
    assert fns.video_model == "seedance-2.0"


# ---------------------------------------------------------------- 会话快照


def test_会话快照记住视频模型_重启沿用(tmp_path: Path):
    s = SessionSnapshot(tmp_path, "default")
    s.set_output_dir("E:/x")
    s.set_video_model("seedance-2.5")
    again = SessionSnapshot(tmp_path, "default")
    assert again.video_model == "seedance-2.5" and again.output_dir == "E:/x"
    again.set_output_dir("E:/y")  # 改产物目录不能把模型冲掉
    assert SessionSnapshot(tmp_path, "default").video_model == "seedance-2.5"
    again.save(ShortTermMemory())  # 整体重写也要带上
    data = json.loads((tmp_path / "default.json").read_text(encoding="utf-8"))
    assert data["video_model"] == "seedance-2.5" and data["output_dir"] == "E:/y"


# ---------------------------------------------------------------- 主循环：挂起 → 人拍板 → 换锁


class ScriptedGateway:
    def __init__(self, script: list[ModelResponse]) -> None:
        self.script = list(script)
        self.n = 0

    async def chat(self, role: str, messages: list[dict], tools: Any = None, **kw: Any) -> Any:
        r = self.script[min(self.n, len(self.script) - 1)]
        self.n += 1
        return r


def _tc(cid: str, name: str, args: str) -> ToolCall:
    return ToolCall(id=cid, name=name, arguments=args)


async def _loop(tmp_path: Path, script: list[ModelResponse]):
    bus = EventBus(session_id="lock")
    media, provider = _media(tmp_path, bus)
    media.video_lock = "seedance-2.0"
    bus.subscribe(media.on_event)  # 装配层就是这么接的
    registry = ToolRegistry(bus)
    registry.register(media)
    await registry.refresh()
    loop = LoopRuntime(
        gateway=ScriptedGateway(script),  # type: ignore[arg-type]
        registry=registry,
        dispatcher=ToolDispatcher(registry, PermissionGate(bus), bus),
        assembler=ContextAssembler(bus),
        memory=ShortTermMemory(),
        bus=bus,
    )
    return loop, media, provider, bus


VEO_CALL = (
    '{"prompt": "补镜15", "model": "veo3.1-quality", "summary": "第11集镜15", '
    '"allow_no_refs": true}'
)


async def test_主循环_换模型挂起等人_采纳后换锁再调才生成(tmp_path: Path):
    script = [
        ModelResponse(tool_calls=[_tc("c1", "gen_video", VEO_CALL)], usage=Usage(1, 1)),
        ModelResponse(tool_calls=[_tc("c2", "gen_video", VEO_CALL)], usage=Usage(1, 1)),
        ModelResponse(text="补好了", usage=Usage(1, 1)),
    ]
    loop, media, provider, bus = await _loop(tmp_path, script)
    r = await loop.run_turn("把镜 15 用 veo 补一下")
    assert r.stop_reason is StopReason.AWAITING_REVIEW
    assert loop.pending_review is not None and loop.pending_review["stage"] == VIDEO_SWITCH_STAGE
    assert not provider.submitted, "人没点头之前不生成"
    assert media.video_lock == "seedance-2.0"

    r2 = await loop.resume_turn("adopt")
    assert r2.stop_reason is StopReason.NO_TOOL_CALLS and "补好了" in r2.text
    assert media.video_lock == "veo3.1-quality"
    assert [s["model"] for s in provider.submitted] == ["veo3.1-quality"]
    verdicts = [
        m["content"] for m in r2.turn.messages
        if m.get("role") == "tool" and "人的决策" in m["content"]
    ]
    assert verdicts and "已采纳" in verdicts[0]


async def test_主循环_打回就不换_留空继续用锁定的(tmp_path: Path):
    script = [
        ModelResponse(tool_calls=[_tc("c1", "gen_video", VEO_CALL)], usage=Usage(1, 1)),
        ModelResponse(
            tool_calls=[_tc("c2", "gen_video", '{"prompt": "补镜15", "allow_no_refs": true}')],
            usage=Usage(1, 1),
        ),
        ModelResponse(text="按原模型补好了", usage=Usage(1, 1)),
    ]
    loop, media, provider, bus = await _loop(tmp_path, script)
    r = await loop.run_turn("补镜 15")
    assert r.stop_reason is StopReason.AWAITING_REVIEW
    r2 = await loop.resume_turn("revise", reason="不换，继续用 seedance")
    assert r2.stop_reason is StopReason.NO_TOOL_CALLS
    assert media.video_lock == "seedance-2.0"
    assert [s["model"] for s in provider.submitted] == ["seedance-2.0"]


async def test_auto模式下换模型也要停(tmp_path: Path):
    script = [
        ModelResponse(tool_calls=[_tc("c1", "gen_video", VEO_CALL)], usage=Usage(1, 1)),
        ModelResponse(text="x", usage=Usage(1, 1)),
    ]
    loop, media, provider, bus = await _loop(tmp_path, script)
    loop.auto_review = True
    r = await loop.run_turn("补镜 15")
    assert r.stop_reason is StopReason.AWAITING_REVIEW, "/auto 也不能替用户换模型"
    assert not provider.submitted

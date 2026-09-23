"""生图模型锁（2026-09-22，用户定的规则）：切换生图模型之前必须先问用户。

真实来由：西游记那条线里，8 岁角色「阿蛛」的主形象被 gpt-image-2 的内容护栏拒了。
换一家模型是对的工程选择（实测 qwen / seedream / flux 都能出，gemini 系同样拒），
但画风、质感会跟着变，而且之后所有图都用新模型 —— 这种事只能人拍板。

规则和视频那套完全一样，共用 MediaFunctions._model_gate：
  · 本会话锁定一个生图模型（会话快照记住的 > 短剧配置里的 image_model > 第一次用的）
  · model 留空一律用锁定的，不再按 prefer 自动选型
  · 传了不同的模型 → 挂起问人（major，/auto 也停）；人采纳才换锁，打回不换
  · 短剧链的生图模型跟着锁走；锁写回会话快照，重启沿用

模型**可以主动提议**换（拒稿时本来就该提议），禁止的只是自己换了算。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aigc_agent.capabilities.memory.session import SessionSnapshot
from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.functions.drama import DramaFunctions
from aigc_agent.domain.functions.media import (
    IMAGE_SWITCH_STAGE,
    VIDEO_SWITCH_STAGE,
    MediaFunctions,
)
from aigc_agent.domain.generators.catalog import MediaCatalog
from aigc_agent.domain.output import OutputPrefs
from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.model.media import FakeMediaProvider, MediaGateway

CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "media_models.yaml"


def _media(tmp_path: Path) -> tuple[MediaFunctions, FakeMediaProvider]:
    provider = FakeMediaProvider(urls=["https://example.com/out.png"])
    gw = MediaGateway({"apimart": provider}, EventBus(), poll_interval=0.01,
                      max_poll_interval=0.02)

    async def dl(url: str, dest: Path) -> bool:
        return False

    fns = MediaFunctions(gw, MediaCatalog.load(CATALOG_PATH), AssetStore(tmp_path / "a"),
                         prefs=OutputPrefs(tmp_path / "out", downloader=dl))
    return fns, provider


def _decided(decision: str, node: str = IMAGE_SWITCH_STAGE) -> Any:
    return SimpleNamespace(
        type=EventType.CHECKPOINT_DECIDED,
        data={"node": node, "decision": decision, "decided_by": "human"},
    )


# ---------------------------------------------------------------- 媒体层


async def test_留空用锁定的生图模型_不再自动选型(tmp_path: Path):
    fns, provider = _media(tmp_path)
    fns.image_lock = "gpt-image-2"
    r = await fns._fn_gen_image("白底证件照", prefer="quality")
    assert r.ok, r.error
    assert provider.submitted[-1]["model"] == "gpt-image-2", "prefer 不该把锁顶掉"


async def test_要换生图模型_挂起问人_采纳才换_打回不换(tmp_path: Path):
    fns, provider = _media(tmp_path)
    fns.image_lock = "gpt-image-2"
    saved: list[str] = []
    fns.on_image_lock = saved.append

    r = await fns._fn_gen_image("阿蛛 白底正面半身", model="qwen-image-3.0", summary="角色·阿蛛")
    assert r.suspend and not provider.submitted, "没问人之前一张都不生成"
    p = r.suspend_payload
    assert p["stage"] == IMAGE_SWITCH_STAGE and p["major"] is True
    assert p["model_from"] == "gpt-image-2" and p["model_to"] == "qwen-image-3.0"
    assert "gpt-image-2 换成 qwen-image-3.0" in p["question"] and "角色·阿蛛" in p["question"]
    assert "回复 a 同意换" in p["question"]
    assert "已暂停等待用户决定" in r.content
    assert fns._pending_switch == {IMAGE_SWITCH_STAGE: "qwen-image-3.0"}

    # 人打回：不换，锁不动
    fns.on_event(_decided("revise"))
    assert fns.image_lock == "gpt-image-2" and not fns._pending_switch and saved == []
    r = await fns._fn_gen_image("接着画")
    assert r.ok and provider.submitted[-1]["model"] == "gpt-image-2"

    # 再问一次，人采纳：换锁并持久化，之后留空也用新模型
    r = await fns._fn_gen_image("阿蛛", model="qwen-image-3.0")
    assert r.suspend
    fns.on_event(_decided("adopt"))
    assert fns.image_lock == "qwen-image-3.0" and saved == ["qwen-image-3.0"]
    r = await fns._fn_gen_image("阿蛛")
    assert r.ok and provider.submitted[-1]["model"] == "qwen-image-3.0"


async def test_视频的决策事件不会换掉生图的锁(tmp_path: Path):
    """两把锁共用一套门，但决策事件必须各认各的环节名，不能串。"""
    fns, _ = _media(tmp_path)
    fns.image_lock, fns.video_lock = "gpt-image-2", "seedance-2.0"
    r = await fns._fn_gen_image("x", model="qwen-image-3.0")
    assert r.suspend
    fns.on_event(_decided("adopt", node=VIDEO_SWITCH_STAGE))
    assert fns.image_lock == "gpt-image-2", "视频那边的决策把生图的锁换了"
    assert fns._pending_switch == {IMAGE_SWITCH_STAGE: "qwen-image-3.0"}
    fns.on_event(_decided("adopt"))
    assert fns.image_lock == "qwen-image-3.0"


async def test_没锁定时第一次用的成为生图锁(tmp_path: Path):
    fns, provider = _media(tmp_path)
    assert fns.image_lock == ""
    r = await fns._fn_gen_image("一张图", prefer="fast")
    assert r.ok and fns.image_lock == provider.submitted[-1]["model"]
    assert "生图模型已锁定为" in r.content
    first = fns.image_lock
    r = await fns._fn_gen_image("再来一张", prefer="quality")
    assert r.ok and provider.submitted[-1]["model"] == first, "锁定后 prefer 不再改模型"


async def test_批量生图_任一项要换模型整批先问人(tmp_path: Path):
    fns, provider = _media(tmp_path)
    fns.image_lock = "gpt-image-2"
    r = await fns._fn_gen_images(
        [{"prompt": "a", "summary": "图1"},
         {"prompt": "b", "summary": "图2", "model": "qwen-image-3.0"}]
    )
    assert r.suspend and not provider.submitted
    assert r.suspend_payload["model_to"] == "qwen-image-3.0"
    assert "图2" in r.suspend_payload["question"]

    r = await fns._fn_gen_images([{"prompt": "a"}, {"prompt": "b"}], prefer="quality")
    assert r.ok, r.error
    assert [s["model"] for s in provider.submitted] == ["gpt-image-2", "gpt-image-2"]


async def test_不认识的生图模型直接报错不挂起(tmp_path: Path):
    fns, _ = _media(tmp_path)
    fns.image_lock = "gpt-image-2"
    r = await fns._fn_gen_image("x", model="并不存在的模型")
    assert not r.ok and not r.suspend and "未知模型" in r.error


# ---------------------------------------------------------------- 短剧链跟着锁走


def test_短剧生图模型跟着会话锁走():
    fns = DramaFunctions(None, AssetStore(), registry=None,
                         catalog=SimpleNamespace(drama={"image_model": "gpt-image-2"}))
    assert fns.image_model == "gpt-image-2", "没接锁时用配置里的"
    locked = {"m": ""}
    fns.image_lock_source = lambda: locked["m"]
    assert fns.image_model == "gpt-image-2", "锁是空的就回落到配置"
    locked["m"] = "qwen-image-3.0"
    assert fns.image_model == "qwen-image-3.0", "用户同意换了，整条短剧链一起换"


def test_取锁抛异常时不拖垮渲染():
    fns = DramaFunctions(None, AssetStore(), registry=None,
                         catalog=SimpleNamespace(drama={"image_model": "gpt-image-2"}))

    def boom() -> str:
        raise RuntimeError("装配层没接好")

    fns.image_lock_source = boom
    assert fns.image_model == "gpt-image-2"


# ---------------------------------------------------------------- 会话快照


def test_会话快照记住生图模型_重启沿用(tmp_path: Path):
    s = SessionSnapshot(tmp_path, "西游记")
    assert s.image_model == ""
    s.set_image_model("qwen-image-3.0")
    s.set_video_model("seedance-2.0")
    again = SessionSnapshot(tmp_path, "西游记")
    assert again.image_model == "qwen-image-3.0" and again.video_model == "seedance-2.0"
    data = json.loads((tmp_path / "西游记.json").read_text(encoding="utf-8"))
    assert data["image_model"] == "qwen-image-3.0"

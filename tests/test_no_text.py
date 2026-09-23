"""视频画面里不许出现字幕/文字（2026-09-18，用户定的最高优先级约束）。

两道保险：所有 gen_video 的提示词最外层包一段最高优先级禁令；短剧渲染每段生成后抽帧给
视觉模型看，发现字幕就用更硬的提示词重生成，还有就标出来交人复核。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions.drama import DramaFunctions
from aigc_agent.domain.functions.media import MediaFunctions
from aigc_agent.domain.generators.catalog import MediaCatalog
from aigc_agent.domain.media.no_text import (
    NO_TEXT_HEAD,
    NO_TEXT_RETRY,
    NO_TEXT_TAIL,
    no_text_prompt,
    no_text_retry,
    parse_subtitle_verdict,
    subtitle_check_messages,
)
from aigc_agent.domain.output import OutputPrefs
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.model.media import FakeMediaProvider, MediaGateway
from aigc_agent.harness.tools.provider import ToolResult

CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "media_models.yaml"

# ---------------------------------------------------------------- 提示词


def test_禁令在最外层_开头最高优先级_末尾复述_幂等():
    p = no_text_prompt("【音色锁定】…\n陆离推门进屋")
    assert p.startswith(NO_TEXT_HEAD) and p.endswith(NO_TEXT_TAIL)
    assert p.index("最高优先级") < p.index("音色锁定")
    assert no_text_prompt(p) == p
    for kw in ("字幕", "水印", "台词只以声音呈现", "subtitles", "captions", "overrides everything"):
        assert kw in NO_TEXT_HEAD, kw


def test_重生成时禁令后面插一句上一版出了字幕():
    p = no_text_retry(no_text_prompt("正文"))
    assert p.startswith(NO_TEXT_HEAD)
    assert p.index(NO_TEXT_RETRY) < p.index("正文")
    assert no_text_retry(p) == p
    assert no_text_retry("裸提示词").startswith(NO_TEXT_RETRY)


def test_检查消息与解析():
    msgs = subtitle_check_messages(["data:image/jpeg;base64,AA", "data:image/jpeg;base64,BB"])
    parts = msgs[0]["content"]
    assert parts[0]["type"] == "text" and "招牌" in parts[0]["text"]
    assert [p["type"] for p in parts[1:]] == ["image_url", "image_url"]
    assert parse_subtitle_verdict('{"text_found": true, "where": "底部字幕"}') == (True, "底部字幕")
    assert parse_subtitle_verdict('```json\n{"text_found": false}\n```') == (False, "")
    assert parse_subtitle_verdict("看不出来")[0] is None


async def test_gen_video统一包禁令_allow_text才放开(tmp_path: Path):
    provider = FakeMediaProvider(urls=["https://example.com/out.mp4"])
    gw = MediaGateway({"apimart": provider}, EventBus(), poll_interval=0.01, max_poll_interval=0.02)

    async def dl(url: str, dest: Path) -> bool:
        return False

    fns = MediaFunctions(gw, MediaCatalog.load(CATALOG_PATH), AssetStore(tmp_path / "a"),
                         prefs=OutputPrefs(tmp_path / "out", downloader=dl))
    r = await fns._fn_gen_video("海边日落", model="seedance-2.0")
    assert r.ok, r.error
    assert provider.submitted[-1]["prompt"].startswith(NO_TEXT_HEAD)
    assert provider.submitted[-1]["prompt"].endswith(NO_TEXT_TAIL)
    await fns._fn_gen_video("片头标题卡", model="seedance-2.0", allow_text=True)
    assert provider.submitted[-1]["prompt"] == "片头标题卡"


# ---------------------------------------------------------------- 字幕门


class Registry:
    def __init__(self, store: AssetStore) -> None:
        self.store = store
        self.calls: list[dict] = []

    async def invoke(self, name: str, args: dict) -> ToolResult:
        if name == "compose_video":
            a = self.store.create("mp4", summary="成片", creator="fake")
            a.uri = "https://fake/composed.mp4"
            self.store.put(a)
            return ToolResult(ok=True, content="拼好了", asset_ref=a.id)
        self.calls.append(dict(args))
        a = self.store.create("", type_=AssetType.VIDEO, summary=args["summary"], creator="model:x")
        a.uri = f"https://fake/{a.id}.mp4"
        self.store.put(a)
        return ToolResult(ok=True, content="ok", asset_ref=a.id)


SHOTS = [
    {"scene_index": "[第1集-1场]", "video_name": "1-2", "video_duration": "8s",
     "description": "开场 (安妮)"},
    {"scene_index": "[第1集-2场]", "video_name": "3-4", "video_duration": "8s",
     "description": "收尾 (安妮)"},
]


def _fns(store: AssetStore, verdicts: dict[str, list[bool]], retries: str = "1") -> Any:
    """verdicts：summary → 每次检查的结果序列（True = 发现字幕）。"""
    reg = Registry(store)
    fns = DramaFunctions(SimpleNamespace(chat=None), store, registry=reg, catalog=None)
    fns.catalog = SimpleNamespace(
        drama={"subtitle_gate": "true", "subtitle_retries": retries},
        max_concurrency=lambda k: 0,
    )
    checked: list[str] = []

    async def check(asset_id: str) -> tuple[bool, str]:
        summary = store.get(asset_id).summary
        checked.append(summary)
        seq = verdicts.get(summary)
        if seq is None:
            return False, ""
        return seq.pop(0), ""

    fns._check_subtitles = check  # type: ignore[method-assign]
    return fns, reg, checked


def _shots(store: AssetStore) -> str:
    return store.create(json.dumps(SHOTS, ensure_ascii=False), summary="提示词", creator="t").id


async def test_发现字幕_用更硬的提示词重生成_第二版干净():
    store = AssetStore()
    fns, reg, checked = _fns(store, {"[第1集-1场] 1-2": [True, False]})
    r = await fns._fn_drama_render_shots(_shots(store))
    assert r.ok, r.error
    first = [c for c in reg.calls if c["summary"] == "[第1集-1场] 1-2"]
    assert len(first) == 2 and NO_TEXT_RETRY in first[1]["prompt"]
    assert NO_TEXT_RETRY not in first[0]["prompt"]
    assert "画面出现字幕，已重生成" in r.content and "重生成后无字幕" in r.content
    assert checked.count("[第1集-1场] 1-2") == 2 and checked.count("[第1集-2场] 3-4") == 1
    # 成片用的是第二版
    compose_clips = [a for a in store.all() if a.summary == "[第1集-1场] 1-2"]
    assert len(compose_clips) == 2
    render = next(a for a in store.all() if a.creator == "tool:drama_render_shots")
    kept = json.loads(store.content(render.id))
    assert kept[0]["asset"] == compose_clips[1].id


async def test_重生成后仍有字幕_保留并标出来交人复核():
    store = AssetStore()
    fns, reg, _ = _fns(store, {"[第1集-2场] 3-4": [True, True]})
    r = await fns._fn_drama_render_shots(_shots(store))
    assert r.ok, r.error
    assert sum(1 for c in reg.calls if c["summary"] == "[第1集-2场] 3-4") == 2
    assert "仍有字幕/文字" in r.content and "需人工复核" in r.content
    assert "[第1集-2场] 3-4" in r.content.split("需人工复核")[1]


async def test_只检查不重生成():
    store = AssetStore()
    fns, reg, _ = _fns(store, {"[第1集-1场] 1-2": [True]}, retries="0")
    r = await fns._fn_drama_render_shots(_shots(store))
    assert r.ok, r.error
    assert sum(1 for c in reg.calls if c["summary"] == "[第1集-1场] 1-2") == 1
    assert "仍有字幕/文字" in r.content


async def test_检查做不了只记备注不拦生成():
    store = AssetStore()
    fns, reg, _ = _fns(store, {})

    async def cannot(asset_id: str) -> tuple[bool, str]:
        return False, "片段没有本地副本，跳过字幕检查"

    fns._check_subtitles = cannot  # type: ignore[method-assign]
    r = await fns._fn_drama_render_shots(_shots(store))
    assert r.ok and len(reg.calls) == 2
    assert "跳过字幕检查" in r.content and "仍有字幕" not in r.content


async def test_没有文本网关就不检查():
    store = AssetStore()
    reg = Registry(store)
    fns = DramaFunctions(None, store, registry=reg, catalog=None)
    assert fns._subtitle_cfg() == (False, 0)
    r = await fns._fn_drama_render_shots(_shots(store))
    assert r.ok and "字幕检查" not in r.content

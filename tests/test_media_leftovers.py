"""2026-09-23 审查 · 媒体网关剩下的几处。

  · 海报底图不读本地副本、直接重下远端链接（>24h 必挂），下载后又把 uri 改成本地路径 ——
    这张图以后当参考图会被当成「不是公网链接」拒掉
  · 轮询状态还是 running 时一见 URL 就判成功（有的实现先吐预览链接）
  · 模型锁：采纳就永久换，没有「只这一次」；有锁时 prefer / tier 被悄悄忽略
  · seedance-2.0-fast 目录只列 720p，工具描述却推荐 480p
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2 as httpx

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions import poster as poster_mod
from aigc_agent.domain.functions.media import VIDEO_SWITCH_STAGE, MediaFunctions, _adopt_once
from aigc_agent.domain.functions.poster import PosterFunctions
from aigc_agent.domain.functions.short_video import ShortVideoFunctions
from aigc_agent.domain.generators.catalog import MediaCatalog
from aigc_agent.domain.output import OutputPrefs
from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.model.media import (
    ApiMartProvider,
    FakeMediaProvider,
    MediaGateway,
    MediaKind,
    MediaTask,
    TaskStatus,
)

CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "media_models.yaml"
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64


# ---------------------------------------------------------------- 海报底图


def _remote_image(store: AssetStore, **gp: Any):
    a = store.create("", type_=AssetType.IMAGE, summary="底图", creator="model",
                     gen_params=dict(gp))
    a.uri = "https://expired.example.com/bg.png"
    store.put(a)
    return a


async def test_海报底图先用本地副本_不重下_不改uri(tmp_path: Path, monkeypatch):
    store = AssetStore(tmp_path / "assets")
    local = tmp_path / "out" / "images" / "底图.png"
    local.parent.mkdir(parents=True)
    local.write_bytes(PNG)
    a = _remote_image(store, local=str(local))

    async def boom(url: str, target: Path) -> tuple[bool, str]:
        raise AssertionError("有本地副本还去下远端链接")

    monkeypatch.setattr(poster_mod, "download", boom)
    got = await PosterFunctions(store)._background(a.id)
    assert got == PNG
    assert store.get(a.id).uri == "https://expired.example.com/bg.png"


async def test_海报底图只有远端链接_抓下来记blob_uri不动(tmp_path: Path, monkeypatch):
    store = AssetStore(tmp_path / "assets")
    a = _remote_image(store)

    def write(target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(PNG)

    async def fake(url: str, target: Path) -> tuple[bool, str]:
        await asyncio.to_thread(write, target)
        return True, ""

    monkeypatch.setattr(poster_mod, "download", fake)
    assert await PosterFunctions(store)._background(a.id) == PNG
    again = store.get(a.id)
    assert again.uri.startswith("https://"), "uri 改成本地路径，参考图链就用不了它了"
    assert await asyncio.to_thread(Path(again.gen_params["blob"]).exists)

    async def dead(url: str, target: Path) -> tuple[bool, str]:
        return False, "HTTP 403"

    b = _remote_image(store)
    monkeypatch.setattr(poster_mod, "download", dead)
    try:
        await PosterFunctions(store)._background(b.id)
    except RuntimeError as e:
        assert "没有本地副本" in str(e) and "24 小时" in str(e)
    else:
        raise AssertionError("下载失败要报清楚")


# ---------------------------------------------------------------- 轮询


def _provider(payload: dict[str, Any]) -> ApiMartProvider:
    p = ApiMartProvider("https://api.example.com", "k")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    p._client = httpx.AsyncClient(base_url=p.base_url, transport=httpx.MockTransport(handler))
    return p


async def _poll(payload: dict[str, Any]) -> MediaTask:
    p = _provider(payload)
    try:
        return await p.poll(MediaTask(task_id="t1", kind=MediaKind.VIDEO, model="m"))
    finally:
        await p.close()


async def test_轮询_running带URL_进度没到头不算成功():
    url = {"url": "https://cdn.example.com/preview.mp4"}
    t = await _poll({"data": {"status": "processing", "progress": 45, "result": url}})
    assert t.status is TaskStatus.RUNNING, "进度 45 就判成功，拿到的是预览"
    t = await _poll({"data": {"status": "processing", "progress": "60%", "result": url}})
    assert t.status is TaskStatus.RUNNING
    t = await _poll({"data": {"status": "processing", "progress": 0.5, "result": url}})
    assert t.status is TaskStatus.RUNNING
    # 进度到头 / 没有进度字段：照旧以有无产物为准
    for extra in ({"progress": 100}, {"progress": 1.0}, {}):
        t = await _poll({"data": {"status": "processing", **extra, "result": url}})
        assert t.status is TaskStatus.SUCCEEDED, extra
        assert t.urls == ["https://cdn.example.com/preview.mp4"]


# ---------------------------------------------------------------- 模型锁


def _media(tmp_path: Path) -> tuple[MediaFunctions, FakeMediaProvider]:
    provider = FakeMediaProvider(urls=["https://example.com/out.mp4"])
    gw = MediaGateway({"apimart": provider}, EventBus(), poll_interval=0.01,
                      max_poll_interval=0.02)

    async def dl(url: str, dest: Path) -> bool:
        return False

    fns = MediaFunctions(gw, MediaCatalog.load(CATALOG_PATH), AssetStore(tmp_path / "a"),
                         prefs=OutputPrefs(tmp_path / "out", downloader=dl))
    return fns, provider


def _decided(decision: str, reason: str = "") -> Any:
    return SimpleNamespace(
        type=EventType.CHECKPOINT_DECIDED,
        data={"node": VIDEO_SWITCH_STAGE, "decision": decision, "reason": reason},
    )


def _loop_end(stop: str) -> Any:
    return SimpleNamespace(type=EventType.LOOP_END, data={"stop_reason": stop})


def test_只这一次的说法():
    for s in ("只这一次", "只这次", "仅此一次", "只换这一次", "只用一次", "就这次", "只这一镜",
              "临时用一下", "暂时先这样", "just once"):
        assert _adopt_once(s), s
    for s in ("", "控制在60集", "之后都用它", "只这一次吗？不，之后都换", "这次效果好就行"):
        assert not _adopt_once(s), s


async def test_只这一次_这一轮放行_锁不动_轮结束收回(tmp_path: Path):
    fns, provider = _media(tmp_path)
    fns.video_lock = "seedance-2.0"
    saved: list[str] = []
    fns.on_video_lock = saved.append

    r = await fns._fn_gen_video("试一镜", model="seedance-2.0-fast", allow_no_refs=True)
    assert r.suspend and "a 只这一次" in r.suspend_payload["question"]
    fns.on_event(_decided("adopt", "只这一次"))
    assert fns.video_lock == "seedance-2.0" and saved == [], "只这一次不许动锁、不写快照"

    r = await fns._fn_gen_video("试一镜", model="seedance-2.0-fast", allow_no_refs=True)
    assert r.ok and provider.submitted[-1]["model"] == "seedance-2.0-fast"
    assert "仍锁定为 seedance-2.0" in r.content
    r = await fns._fn_gen_video("下一镜", allow_no_refs=True)
    assert r.ok and provider.submitted[-1]["model"] == "seedance-2.0", "留空仍用锁定的"

    # 挂起等人审的 LOOP_END 不算这一轮结束
    fns.on_event(_loop_end("awaiting_review"))
    r = await fns._fn_gen_video("再试", model="seedance-2.0-fast", allow_no_refs=True)
    assert r.ok
    # 这一轮真结束：收回，再要用 fast 又得问
    fns.on_event(_loop_end("no_tool_calls"))
    r = await fns._fn_gen_video("再试", model="seedance-2.0-fast", allow_no_refs=True)
    assert r.suspend

    # 不带「只这一次」的采纳照旧永久换
    fns.on_event(_decided("adopt", "画面更稳就行"))
    assert fns.video_lock == "seedance-2.0-fast" and saved == ["seedance-2.0-fast"]


async def test_有锁时prefer不生效要说出来_并指同一家的那一档(tmp_path: Path):
    fns, provider = _media(tmp_path)
    fns.video_lock = "seedance-2.0"
    r = await fns._fn_gen_video("试方向", prefer="fast", allow_no_refs=True)
    assert r.ok and provider.submitted[-1]["model"] == "seedance-2.0"
    assert "prefer=fast 没生效" in r.content and "model=seedance-2.0-fast" in r.content
    r = await fns._fn_gen_videos([{"prompt": "a"}], prefer="fast", allow_no_refs=True)
    assert r.ok and "prefer=fast 没生效" in r.content
    r = await fns._fn_gen_video("x", prefer="quality", allow_no_refs=True)
    assert "没生效" not in r.content, "档位本来就一致，不用说"
    r = await fns._fn_gen_video("x", allow_no_refs=True)
    assert "没生效" not in r.content, "balanced 是默认值，不当成有意传的"


def test_短视频档位和会话锁对账():
    fns = ShortVideoFunctions(None, AssetStore(), catalog=MediaCatalog.load(CATALOG_PATH))
    assert fns._tier_note("fast", explicit=True) == ("", "")
    fns.video_lock_source = lambda: "seedance-2.0"
    locked, note = fns._tier_note("fast", explicit=False)
    assert locked == "seedance-2.0"
    assert "档位 fast（配方默认）" in note and "没生效" in note and "quality 档" in note
    assert fns._tier_note("quality", explicit=True) == ("seedance-2.0", "")


# ---------------------------------------------------------------- 分辨率


async def test_fast档480p进目录_目录没列的分辨率要提示(tmp_path: Path):
    catalog = MediaCatalog.load(CATALOG_PATH)
    fast = catalog.get(MediaKind.VIDEO, "seedance-2.0-fast")
    assert fast is not None and "480p" in fast.resolutions
    assert "480p" in fast.brief(), "list_media_models 要看得到支持哪些分辨率"

    fns, provider = _media(tmp_path)
    fns.video_lock = "seedance-2.0"
    r = await fns._fn_gen_video("x", resolution="480p", allow_no_refs=True)
    assert r.ok and "没列 480p" in r.content
    fns.video_lock = "seedance-2.0-fast"
    r = await fns._fn_gen_video("x", resolution="480p", allow_no_refs=True)
    assert r.ok and "没列" not in r.content

"""视频生成的健壮性（2026-09-18，按真实失败日志）。

那天一集 4 段里 3 段失败，两类原因：
  · HTTP 400 "Invalid format for video_urls[0]"：拼完第 1 集后片段 uri 被换成本地路径，
    下一轮把它当音色锚点传给模型 —— 接口只收 http(s) 链接
  · 轮询 2 次 ConnectError 就把任务判死：服务端还在跑、照样计费，片段却丢了
再加上每次失败都要整集重生成。这里钉住四件事：
  ① 下载本地副本不改 uri；本地路径的片段不会被当参考视频传出去
  ② 轮询：固定间隔、次数上限、网络抖动容忍、提交重试
  ③ 重跑只补失败的段，成功的复用
  ④ 单段网络类失败自动重试，400 不重试
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx2 as httpx

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions.drama import DramaFunctions, _transient_error
from aigc_agent.domain.functions.video_edit import VideoEditFunctions
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.model.media import (
    FakeMediaProvider,
    MediaGateway,
    MediaKind,
    MediaTask,
    TaskStatus,
)
from aigc_agent.harness.tools.provider import ToolResult

# ---------------------------------------------------------------- ① uri 不被本地路径覆盖


async def test_下载本地副本不改uri_本地路径记在gen_params(tmp_path: Path):
    store = AssetStore(tmp_path / "assets")
    a = store.create("", type_=AssetType.VIDEO, summary="片段", creator="model:seedance-2.0")
    a.uri = "https://cdn/clip.mp4"
    store.put(a)
    fns = VideoEditFunctions(store, tmp_path / "ws")

    async def fake_download(url: str, target: Path) -> tuple[bool, str]:
        def write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"mp4")

        await asyncio.to_thread(write)
        return True, ""

    import aigc_agent.domain.functions.video_edit as mod

    orig = mod.ffmpeg.download
    mod.ffmpeg.download = fake_download  # type: ignore[assignment]
    try:
        p, err = await fns._localize(a.id)
    finally:
        mod.ffmpeg.download = orig  # type: ignore[assignment]
    assert not err and p and p.exists()
    again = store.get(a.id)
    assert again.uri == "https://cdn/clip.mp4", "远端链接是后面镜头的参考链，不能被本地路径顶掉"
    assert again.gen_params["local"] == str(p)
    # 第二次直接用本地副本，不再下载
    p2, err2 = await fns._localize(a.id)
    assert not err2 and p2 == p


def test_本地路径的片段不当参考视频():
    store = AssetStore()
    fns = DramaFunctions(None, store, registry=None, catalog=None)
    a = store.create("", type_=AssetType.VIDEO, summary="片段", creator="model:x")
    a.uri = r"E:\手搓Agent\workspace\assets\blobs\as_x.mp4"
    store.put(a)
    url, why = fns._clip_ref_url(a.id, fns._voice_cfg())
    assert url == "" and "本地路径" in why
    b = store.create("", type_=AssetType.VIDEO, summary="片段", creator="model:x")
    b.uri = "https://cdn/ok.mp4"
    store.put(b)
    assert fns._clip_ref_url(b.id, fns._voice_cfg()) == ("https://cdn/ok.mp4", "")


# ---------------------------------------------------------------- ② 轮询


class FlakyProvider(FakeMediaProvider):
    """前几次轮询抛网络错误，之后正常。"""

    def __init__(self, flaky_polls: int, **kw):
        super().__init__(**kw)
        self.flaky_polls = flaky_polls
        self.poll_calls = 0

    async def poll(self, task):
        self.poll_calls += 1
        if self.poll_calls <= self.flaky_polls:
            raise httpx.ConnectError("boom")
        return await super().poll(task)


class FlakySubmit(FakeMediaProvider):
    def __init__(self, fail_times: int, **kw):
        super().__init__(**kw)
        self.fail_times = fail_times
        self.submit_calls = 0

    async def submit(self, kind, model, prompt, **params):
        self.submit_calls += 1
        if self.submit_calls <= self.fail_times:
            raise httpx.ConnectError("no route")
        return await super().submit(kind, model, prompt, **params)


def _gw(provider, **kw) -> MediaGateway:
    return MediaGateway(
        {"apimart": provider}, EventBus(), poll_interval=0.01, poll_backoff=1.0,
        max_poll_interval=0.01, **kw,
    )


async def test_轮询网络抖动不判死_连续超限才失败():
    p = FlakyProvider(flaky_polls=3, polls_needed=2, urls=["https://x/1.mp4"])
    task = await _gw(p, max_transient=5).generate("apimart", MediaKind.VIDEO, "m", "x")
    assert task.ok, task.error
    assert p.poll_calls == 3 + 2  # 3 次网络错误 + 2 次正常

    p2 = FlakyProvider(flaky_polls=99, polls_needed=2)
    task2 = await _gw(p2, max_transient=4).generate("apimart", MediaKind.VIDEO, "m", "x")
    assert task2.status is TaskStatus.FAILED and "连续 5 次网络错误" in (task2.error or "")


async def test_轮询次数上限_固定间隔():
    p = FakeMediaProvider(polls_needed=50)
    gw = _gw(p, max_polls=7)
    task = await gw.generate("apimart", MediaKind.VIDEO, "m", "x", max_wait_s=30)
    assert task.status is TaskStatus.TIMEOUT
    assert "轮询 7 次" in (task.error or "") and task.polls == 7
    # backoff 1.0：间隔不再增长
    assert gw.poll_backoff == 1.0


async def test_提交网络错误自动重试_超过次数才报():
    p = FlakySubmit(fail_times=2, polls_needed=1, urls=["https://x/1.mp4"])
    task = await _gw(p, submit_retries=2).generate("apimart", MediaKind.VIDEO, "m", "x")
    assert task.ok and p.submit_calls == 3

    p2 = FlakySubmit(fail_times=5, polls_needed=1)
    task2 = await _gw(p2, submit_retries=1).generate("apimart", MediaKind.VIDEO, "m", "x")
    assert task2.status is TaskStatus.FAILED and "提交失败" in (task2.error or "")
    assert isinstance(task2, MediaTask)


def test_配置里的口径_3秒一次200次():
    from aigc_agent.domain.generators.catalog import MediaCatalog

    c = MediaCatalog.load(Path(__file__).resolve().parents[1] / "config" / "media_models.yaml")
    assert c.polling.interval == 3.0 and c.polling.backoff == 1.0 and c.polling.max_polls == 200
    assert c.polling.max_transient >= 5 and c.polling.submit_retries >= 1


# ---------------------------------------------------------------- ③④ 重跑只补失败 + 单段重试


class Registry:
    """gen_video 按 summary 决定成败：fail_once 里的先失败一次再成功；always_fail 永远失败。"""

    def __init__(self, store: AssetStore, fail_once: dict[str, str] | None = None,
                 always_fail: dict[str, str] | None = None):
        self.store = store
        self.fail_once = dict(fail_once or {})
        self.always_fail = dict(always_fail or {})
        self.calls: list[tuple[str, dict]] = []

    def videos(self) -> list[dict]:
        return [a for n, a in self.calls if n == "gen_video"]

    async def invoke(self, name: str, args: dict) -> ToolResult:
        self.calls.append((name, dict(args)))
        if name == "compose_video":
            a = self.store.create("mp4", summary="成片", creator="fake")
            a.uri = "https://fake/composed.mp4"
            self.store.put(a)
            return ToolResult(ok=True, content="拼好了", asset_ref=a.id)
        key = args["summary"]
        if key in self.always_fail:
            return ToolResult(ok=False, error=self.always_fail[key])
        if key in self.fail_once:
            return ToolResult(ok=False, error=self.fail_once.pop(key))
        a = self.store.create("", type_=AssetType.VIDEO, summary=key, creator="model:x")
        a.uri = f"https://fake/{a.id}.mp4"
        self.store.put(a)
        return ToolResult(ok=True, content="ok", asset_ref=a.id)


SHOTS = [
    {"scene_index": "[第1集-1场]", "video_name": "1-2", "video_duration": "8s",
     "description": "开场 (安妮)"},
    {"scene_index": "[第1集-2场]", "video_name": "3-4", "video_duration": "8s",
     "description": "接上场 {第1集-1场} (安妮)"},
    {"scene_index": "[第1集-3场]", "video_name": "5-6", "video_duration": "8s",
     "description": "收尾 (安妮)"},
]


def _shots(store: AssetStore) -> str:
    return store.create(json.dumps(SHOTS, ensure_ascii=False), summary="提示词", creator="t").id


def test_网络类错误才重试():
    assert _transient_error("seedance-2.0 生成失败（failed）：轮询失败 ConnectError: x")
    assert _transient_error("提交失败（网络错误，已重试 2 次）")
    assert _transient_error("HTTP 502：bad gateway")
    assert not _transient_error("HTTP 400: Invalid format for video_urls[0]")
    assert not _transient_error("生成超时（开跑后 >600s）")


async def test_单段网络失败自动重试一次():
    store = AssetStore()
    reg = Registry(store, fail_once={"[第1集-1场] 1-2": "轮询失败 ConnectError: boom"})
    r = await DramaFunctions(None, store, registry=reg)._fn_drama_render_shots(_shots(store))
    assert r.ok, r.error
    assert "失败" not in r.content.split("\n\n")[0]
    assert sum(1 for v in reg.videos() if v["summary"] == "[第1集-1场] 1-2") == 2


async def test_400不重试_直接记失败():
    store = AssetStore()
    bad = "HTTP 400: Invalid format for video_urls[0]"
    reg = Registry(store, always_fail={"[第1集-3场] 5-6": bad})
    r = await DramaFunctions(None, store, registry=reg)._fn_drama_render_shots(_shots(store))
    assert r.ok, r.error
    assert sum(1 for v in reg.videos() if v["summary"] == "[第1集-3场] 5-6") == 1
    assert "失败：" in r.content and "只会补这些失败的段" in r.content


async def test_重跑只补失败的段_成功的复用():
    store = AssetStore()
    reg = Registry(store, always_fail={"[第1集-3场] 5-6": "轮询失败 ConnectError: x"})
    fns = DramaFunctions(None, store, registry=reg, catalog=None)
    shots_id = _shots(store)
    r1 = await fns._fn_drama_render_shots(shots_id)
    assert r1.ok and "[第1集-3场] 5-6" in r1.content.split("失败：")[1]
    names = [a["summary"] for a in reg.videos()]
    assert len(names) == 4 and len(set(names)) == 3  # 第 3 段重试过（同名两次），前两段各一次

    # 修好网络再跑：前两段复用，只生成第 3 段；成片按原序拼 3 段
    reg.always_fail.clear()
    reg.calls.clear()
    r2 = await fns._fn_drama_render_shots(shots_id)
    assert r2.ok, r2.error
    assert [v["summary"] for v in reg.videos()] == ["[第1集-3场] 5-6"]
    assert "复用 2，新生成 1" in r2.content and "↺ [第1集-1场] 1-2" in r2.content
    compose = next(a for n, a in reg.calls if n == "compose_video")
    assert len(compose["clips"]) == 3
    kept = next(a for a in store.all() if a.summary == "[第1集-1场] 1-2")
    assert kept.id == compose["clips"][0], "复用的是上次那段，不是新生成的"

    # reuse=False 全部重生成
    reg.calls.clear()
    r3 = await fns._fn_drama_render_shots(shots_id, reuse=False)
    assert r3.ok and len(reg.videos()) == 3 and "复用" not in r3.content.split("\n")[0]


async def test_复用的段链接过期就不再当前序参考():
    import time

    store = AssetStore()
    reg = Registry(store)
    fns = DramaFunctions(None, store, registry=reg, catalog=None)
    shots_id = _shots(store)
    assert (await fns._fn_drama_render_shots(shots_id)).ok
    old = next(a for a in store.all() if a.summary == "[第1集-1场] 1-2")
    old.created_at = time.time() - 30 * 3600
    store.put(old)
    # 只让第 2 段重生成：把它从上次结果里抹掉
    prev = next(a for a in store.all() if a.creator == "tool:drama_render_shots")
    rows = [r for r in json.loads(store.content(prev.id)) if r["name"] != "3-4"]
    prev.inline = json.dumps(rows, ensure_ascii=False)
    store.put(prev)

    reg.calls.clear()
    r = await fns._fn_drama_render_shots(shots_id)
    assert r.ok, r.error
    second = next(v for v in reg.videos() if v["summary"] == "[第1集-2场] 3-4")
    assert "video_urls" not in second, "过期链接不传给模型（传了只会 400）"

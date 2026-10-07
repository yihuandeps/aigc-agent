"""看图 / 看视频（2026-09-18）。

不打真实视觉模型：假网关记下发出去的消息，验证图片以 data URL 进消息、视频按帧带时间戳、
问题附在提示词后面、来源支持路径 / 资产 id / 链接、路径受文件系统边界约束、转写可选。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions.files import FileFunctions, FsPolicy
from aigc_agent.domain.functions.vision import IMAGE_PROMPT, VIDEO_PROMPT, VisionFunctions
from aigc_agent.harness.tools.provider import ToolResult


class FakeGateway:
    def __init__(self, text: str = "看到了一个人") -> None:
        self.text = text
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    async def chat(self, role: str, messages: list[dict[str, Any]], **_: Any) -> Any:
        self.calls.append((role, messages))
        return SimpleNamespace(text=self.text)


def _setup(tmp_path: Path, gateway: Any = None) -> tuple[VisionFunctions, AssetStore, Path]:
    store = AssetStore(tmp_path / "assets")
    out = tmp_path / "out"
    out.mkdir()
    files = FileFunctions(store, tmp_path / "ws", out, FsPolicy(), project_root=tmp_path / "proj")
    fns = VisionFunctions(gateway or FakeGateway(), store, files)
    return fns, store, out


def _parts(gw: FakeGateway) -> list[dict[str, Any]]:
    return gw.calls[-1][1][-1]["content"]


# ---------------------------------------------------------------- 图片


async def test_看本地图片_dataURL_带问题(tmp_path: Path):
    fns, _, out = _setup(tmp_path)
    (out / "a.png").write_bytes(b"\x89PNG fake")
    r = await fns.invoke("view_image", {"source": str(out / "a.png"), "question": "有几个人？"})
    assert r.ok, r.error
    assert r.content.startswith("【a.png】") and "看到了一个人" in r.content
    gw: FakeGateway = fns.gateway  # type: ignore[assignment]
    role, messages = gw.calls[-1]
    assert role == "vision" and messages[0]["role"] == "system"
    parts = _parts(gw)
    assert parts[0]["text"].startswith(IMAGE_PROMPT) and "有几个人？" in parts[0]["text"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[1]["image_url"]["detail"] == "high"


async def test_看资产图片与链接图片(tmp_path: Path, monkeypatch):
    from aigc_agent.domain.media import vision_input

    async def fake_fetch(url: str) -> tuple[bytes, str, str]:
        if url == "https://cdn/x.png":
            return b"\x89PNG\r\n\x1a\nfake", "image/png", ""
        return b"", "", "HTTP 404"

    monkeypatch.setattr(vision_input, "fetch_image", fake_fetch)
    fns, store, out = _setup(tmp_path)
    (out / "p.jpg").write_bytes(b"jpg")
    a = store.create("", type_=AssetType.IMAGE, summary="主形象")
    a.gen_params["local"] = str(out / "p.jpg")
    store.put(a)
    r = await fns.invoke("view_image", {"source": a.id, "detail": "low"})
    assert r.ok and r.content.startswith(f"【{a.id}】")
    gw: FakeGateway = fns.gateway  # type: ignore[assignment]
    assert _parts(gw)[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert _parts(gw)[1]["image_url"]["detail"] == "low"
    # 链接先下载再转 data URL（2026-09-29 审查 1.2：直接发链接，视觉模型多半拉不到）
    r = await fns.invoke("view_image", {"source": "https://cdn/x.png"})
    assert r.ok and _parts(gw)[1]["image_url"]["url"].startswith("data:image/png;base64,")
    calls = len(gw.calls)
    r = await fns.invoke("view_image", {"source": "https://cdn/gone.png"})
    assert not r.ok and "下载不下来" in r.error and "HTTP 404" in r.error
    assert len(gw.calls) == calls, "图拿不到就不调视觉模型"
    r = await fns.invoke("view_image", {"source": "as_nope"})
    assert not r.ok and "没有资产" in r.error


async def test_路径受文件系统边界约束_视频给错工具(tmp_path: Path):
    fns, _, out = _setup(tmp_path)
    (tmp_path / "outside.png").write_bytes(b"x")
    r = await fns.invoke("view_image", {"source": str(tmp_path / "outside.png")})
    assert not r.ok and "不在允许访问的目录内" in r.error
    (out / "c.mp4").write_bytes(b"x")
    r = await fns.invoke("view_image", {"source": str(out / "c.mp4")})
    assert not r.ok and "view_video" in r.error
    (out / "a.png").write_bytes(b"x")
    r = await fns.invoke("view_video", {"source": str(out / "a.png")})
    assert not r.ok and "view_image" in r.error


async def test_没配视觉角色或没网关_报清楚(tmp_path: Path):
    class NoRole:
        async def chat(self, role: str, messages: Any, **_: Any) -> Any:
            raise KeyError(role)

    fns, _, out = _setup(tmp_path, NoRole())
    (out / "a.png").write_bytes(b"x")
    r = await fns.invoke("view_image", {"source": str(out / "a.png")})
    assert not r.ok and "vision" in r.error
    fns.gateway = None
    r = await fns.invoke("view_image", {"source": str(out / "a.png")})
    assert not r.ok and "网关" in r.error


# ---------------------------------------------------------------- 视频


def _with_frames(
    fns: VisionFunctions, frames: list[tuple[float, bytes]], dur: float = 12.0
) -> None:
    async def fake_frames(path: Path, count: int) -> tuple[list[tuple[float, bytes]], Any]:
        return frames[:count], SimpleNamespace(duration=dur, width=720, height=1280, has_audio=True)

    fns._video_frames = fake_frames  # type: ignore[method-assign]


async def test_看视频_按帧带时间戳_信息与问题进提示词(tmp_path: Path):
    fns, _, out = _setup(tmp_path, FakeGateway("第一帧两人对话…"))
    (out / "clip.mp4").write_bytes(b"x")
    _with_frames(fns, [(0.75, b"f1"), (2.25, b"f2"), (3.75, b"f3")])
    args = {"source": str(out / "clip.mp4"), "frames": 3, "question": "有字幕吗"}
    r = await fns.invoke("view_video", args)
    assert r.ok, r.error
    assert r.content.startswith("【clip.mp4】总长 12.0s，720x1280，有音轨，看了 3 帧")
    gw: FakeGateway = fns.gateway  # type: ignore[assignment]
    parts = _parts(gw)
    head = parts[0]["text"]
    assert head.startswith(VIDEO_PROMPT) and "第1帧 @ 00:00" in head and "第3帧 @ 00:03" in head
    assert "有字幕吗" in head and "音轨转写" not in head
    images = [p for p in parts if p["type"] == "image_url"]
    assert len(images) == 3 and all(p["image_url"]["detail"] == "low" for p in images)
    assert images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    labels = [p["text"] for p in parts if p["type"] == "text"][1:]
    assert labels == ["第 00:00 帧：", "第 00:02 帧：", "第 00:03 帧："]


async def test_帧数上限与抽不出帧(tmp_path: Path):
    fns, _, out = _setup(tmp_path)
    (out / "clip.mp4").write_bytes(b"x")
    _with_frames(fns, [(i * 0.5, b"f") for i in range(40)])
    r = await fns.invoke("view_video", {"source": str(out / "clip.mp4"), "frames": 99})
    assert r.ok and "看了 24 帧" in r.content
    _with_frames(fns, [])
    r = await fns.invoke("view_video", {"source": str(out / "clip.mp4")})
    assert not r.ok and "抽不出画面帧" in r.error


async def test_转写音轨走transcribe工具(tmp_path: Path):
    fns, store, out = _setup(tmp_path)
    clip = store.create("", type_=AssetType.VIDEO, summary="[第1集-1场] 1-4")
    (out / "c.mp4").write_bytes(b"x")
    clip.gen_params["local"] = str(out / "c.mp4")
    store.put(clip)
    _with_frames(fns, [(1.0, b"f")])

    calls: list[tuple[str, dict]] = []

    class Reg:
        async def invoke(self, name: str, args: dict) -> ToolResult:
            calls.append((name, args))
            return ToolResult(ok=True, content="陆离：别回头。")

    fns.registry = Reg()

    async def fake_audio(path: Path) -> bytes:
        return b"mp3"

    fns._extract_audio = fake_audio  # type: ignore[method-assign]
    r = await fns.invoke("view_video", {"source": clip.id, "transcribe": True})
    assert r.ok, r.error
    assert calls and calls[0][0] == "transcribe" and calls[0][1]["format"] == "text"
    audio = store.get(calls[0][1]["asset_id"])
    assert audio.type is AssetType.AUDIO and audio.parent_ids == [clip.id]
    gw: FakeGateway = fns.gateway  # type: ignore[assignment]
    assert "音轨转写" in _parts(gw)[0]["text"] and "别回头" in _parts(gw)[0]["text"]
    assert r.content.startswith(f"【{clip.id}】")

    # 转写做不了：结果里标出来，不报错
    async def no_audio(path: Path) -> bytes:
        return b""

    fns._extract_audio = no_audio  # type: ignore[method-assign]
    r = await fns.invoke("view_video", {"source": clip.id, "transcribe": True})
    assert r.ok and "音轨没转写成功" in r.content


async def test_权限与目录(tmp_path: Path):
    fns, _, _ = _setup(tmp_path)
    metas = {m.name: m for m in await fns.list_tools()}
    assert set(metas) == {"view_image", "view_video"}
    assert all(m.permission.value == "L-compute" for m in metas.values())
    assert metas["view_video"].timeout == 600

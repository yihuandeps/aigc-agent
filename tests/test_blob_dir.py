"""2026-09-29 优化审查 4.5：测试往仓库根目录写文件。

三处各拼 `(root or Path(".")) / "blobs"`（资产库 create_blob、短视频图片转镜头、海报抓底图）：
没传 root 的资产库（内存库，测试里最常见）把字节写进当前目录的 blobs/ —— 跑测试时当前目录
是仓库根目录，blobs/板卡照片_镜02.mp4 每跑一次全量测试被重写一遍。
现在资产库只留一个 blob_dir：有 root 是 root/blobs，内存库用临时目录；另外两处都用它。
conftest 再兜一道：每个测试都在自己的 tmp_path 里跑。
"""

from __future__ import annotations

import asyncio
import gc
from pathlib import Path
from typing import Any

import pytest

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions import poster as poster_mod
from aigc_agent.domain.functions import short_video as short_video_mod
from aigc_agent.domain.functions.poster import PosterFunctions
from aigc_agent.domain.functions.short_video import ShortVideoFunctions

ROOT = Path(__file__).resolve().parents[1]
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64


@pytest.fixture
def cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """单独切到一个空目录：看得出谁往当前目录写了东西。"""
    d = tmp_path / "当前目录"
    d.mkdir()
    monkeypatch.chdir(d)
    return d


def _under(path: Path, folder: Path) -> bool:
    return path.resolve().is_relative_to(folder.resolve())


def test_测试默认在自己的临时目录里跑_不在仓库根目录(tmp_path: Path):
    assert Path.cwd().resolve() == tmp_path.resolve()
    assert Path.cwd().resolve() != ROOT


def test_有root的库_blob落在root下的blobs(tmp_path: Path):
    store = AssetStore(tmp_path / "assets")
    assert store.blob_dir == tmp_path / "assets" / "blobs"
    a = store.create_blob(b"mp3", ".mp3", summary="配音")
    assert Path(a.uri or "").parent == store.blob_dir
    assert store.blob(a.id) == b"mp3"


def test_内存库的blob落临时目录_不写当前目录(cwd: Path):
    store = AssetStore()
    a = store.create_blob(b"mp3", "mp3", summary="配音")
    path = Path(a.uri or "")
    assert path.is_absolute() and path.parent == store.blob_dir
    assert store.blob(a.id) == b"mp3"
    assert not (cwd / "blobs").exists(), "内存库把字节写进了当前目录"
    assert not _under(path, cwd) and not _under(path, ROOT)


def test_root中途置空_blob也不写当前目录(tmp_path: Path, cwd: Path):
    """装配层测试常见写法：Agent.create 之后 assets.root = None，免得往资产库写。"""
    store = AssetStore(tmp_path / "assets")
    store.root = None
    a = store.create_blob(b"png", ".png", type_=AssetType.IMAGE)
    path = Path(a.uri or "")
    assert path.parent == store.blob_dir
    assert not _under(path, tmp_path / "assets") and not (cwd / "blobs").exists()


def test_内存库回收后临时目录跟着删():
    store = AssetStore()
    store.create_blob(b"x", ".bin")
    d = store.blob_dir
    assert d.is_dir() and any(d.iterdir())
    del store
    gc.collect()
    assert not d.exists(), "临时目录没人删，每个内存库漏一个"


async def test_短视频图片转镜头_没产物目录时落资产库的blob_dir(tmp_path: Path, cwd: Path,
                                                monkeypatch: pytest.MonkeyPatch):
    store = AssetStore()
    img = tmp_path / "board.png"
    await asyncio.to_thread(img.write_bytes, PNG)
    photo = store.create("", type_=AssetType.IMAGE, summary="板卡照片",
                         gen_params={"local": str(img)})

    async def fake_still(image: Path, out: Path, seconds: float, size: Any = None
                         ) -> tuple[bool, str]:
        def write() -> None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"clip")

        await asyncio.to_thread(write)
        return True, ""

    monkeypatch.setattr(short_video_mod.ffmpeg, "still_to_clip", fake_still)
    fns = ShortVideoFunctions(None, store)  # 没接产物目录（output=None）
    clip_id, why = await fns._material_clip(photo.id, 3.0, 2)  # noqa: SLF001
    assert clip_id, why
    out = Path(store.get(clip_id).gen_params["local"])
    assert out.parent == store.blob_dir and out.name == "板卡照片_镜02.mp4"
    assert not (cwd / "blobs").exists(), "图片转的镜头写进了当前目录"


async def test_海报底图抓远端_内存库落blob_dir(cwd: Path, monkeypatch: pytest.MonkeyPatch):
    store = AssetStore()
    a = store.create("", type_=AssetType.IMAGE, summary="底图", creator="model")
    a.uri = "https://expired.example.com/bg.png"
    store.put(a)

    async def fake_download(url: str, target: Path) -> tuple[bool, str]:
        def write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(PNG)

        await asyncio.to_thread(write)
        return True, ""

    monkeypatch.setattr(poster_mod, "download", fake_download)
    assert await PosterFunctions(store)._background(a.id) == PNG  # noqa: SLF001
    blob = Path(store.get(a.id).gen_params["blob"])
    assert blob.parent == store.blob_dir and blob.name == f"{a.id}.png"
    assert not (cwd / "blobs").exists(), "抓下来的底图写进了当前目录"

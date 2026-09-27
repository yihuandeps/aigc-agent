"""人定的音色锚点在本地永久存一份
（2026-09-27 用户要的：「定音的锚点我需要保存在本地进行永久的保存」）。

- 定音采纳 / 手动 pin：片段存进 <产物目录>/音色锚点/<角色>.mp4，Agent 在资产库 blobs/ 再留一份
- 重新定音：旧文件挪进 音色锚点/旧版/，不删
- 产物目录那份被整理掉了，还能从 Agent 那份找回；链接过期后从它重新上传（配了托管的话）
- drama_voice_anchors list 显示人定的锚点存在哪
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aigc_agent.domain.assets.store import AssetStore, AssetType, local_copy
from aigc_agent.domain.functions.drama import DramaFunctions
from aigc_agent.domain.media.hosting import Hosting, HostingConfig
from tests.test_hosting import FakeUploader
from tests.test_voice import FakeRegistry, _seed
from tests.test_voice_casting_0926 import _board, _decide, _expire_casting


def _setup(tmp_path: Path) -> tuple[AssetStore, DramaFunctions, str, str, str, Path]:
    store = AssetStore(tmp_path / "assets")
    lib_id, shots_id, pack_id = _seed(store)
    _board(store)
    fns = DramaFunctions(None, store, registry=FakeRegistry(store), catalog=None)
    out = tmp_path / "不渡"
    fns.files = SimpleNamespace(output_root=out)
    return store, fns, lib_id, shots_id, pack_id, out


def _newest_clip(store: AssetStore, name: str) -> Any:
    clips = [a for a in store.find(type_=AssetType.VIDEO) if a.summary == f"定音·{name}"]
    return max(clips, key=lambda a: a.seq)


def _downloaded(store: AssetStore, out: Path, tag: str) -> dict[str, Path]:
    """模拟 gen_video 把刚渲的定音片段下载进产物目录 videos/。"""
    got: dict[str, Path] = {}
    for name in ("陆离", "小满"):
        clip = _newest_clip(store, name)
        p = out / "videos" / f"定音_{name}-{tag}.mp4"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"{tag}-{name}".encode())
        clip.gen_params["local"] = str(p)
        store.put(clip)
        got[name] = p
    return got


async def test_定音采纳后_产物目录和Agent各存一份_产物目录那份删了也找得回(tmp_path: Path):
    store, fns, lib_id, _, _, out = _setup(tmp_path)
    r = await fns._fn_drama_voice_casting(lib_id)
    assert r.suspend
    assert f"并在 {out / '音色锚点'} 里永久存一份" in r.suspend_payload["question"]
    videos = _downloaded(store, out, "v1")
    await _decide(fns, "adopt")

    home = out / "音色锚点" / "陆离.mp4"
    assert home.read_bytes() == "v1-陆离".encode()
    clip = _newest_clip(store, "陆离")
    assert clip.gen_params["local"] == str(home), "本地副本改指到永久那份"
    assert clip.gen_params["voice_anchor"] == "陆离"
    blob = Path(clip.gen_params["blob"])
    assert blob.parent == tmp_path / "assets" / "blobs"
    assert await asyncio.to_thread(blob.read_bytes) == "v1-陆离".encode()

    videos["陆离"].unlink()  # 用户整理了 videos/
    home.unlink()  # 连 音色锚点/ 那份也删了
    assert local_copy(store.get(clip.id)) == blob, "Agent 自留的那份还在"
    assert Hosting.local_file(store.get(clip.id)) == blob, "重新上传也找得到它"


async def test_重新定音_旧文件挪进旧版_不删(tmp_path: Path):
    store, fns, lib_id, _, _, out = _setup(tmp_path)
    await fns._fn_drama_voice_casting(lib_id)
    _downloaded(store, out, "v1")
    await _decide(fns, "adopt")

    await fns._fn_drama_voice_casting(lib_id, characters=["陆离"])
    _downloaded(store, out, "v2")
    await _decide(fns, "adopt")
    home = out / "音色锚点"
    assert (home / "陆离.mp4").read_bytes() == "v2-陆离".encode()
    old = list((home / "旧版").glob("陆离-*.mp4"))
    assert len(old) == 1 and old[0].read_bytes() == "v1-陆离".encode()
    assert (home / "小满.mp4").read_bytes() == "v1-小满".encode(), "没重新定音的角色不动"


async def test_人定的锚点链接过期_从本地那份重新上传_不用临时锚点(tmp_path: Path):
    store, fns, lib_id, shots_id, pack_id, out = _setup(tmp_path)
    await fns._fn_drama_voice_casting(lib_id)
    _downloaded(store, out, "v1")
    await _decide(fns, "adopt")
    _expire_casting(store)
    (out / "音色锚点" / "陆离.mp4").unlink()  # 产物目录那份被整理掉了

    up = FakeUploader()
    fns.hosting = Hosting(HostingConfig(type="command", command="x"), uploader=up)
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.ok, r.error
    uploaded = {p.read_bytes() for p, _ in up.calls}
    assert uploaded == {"v1-陆离".encode(), "v1-小满".encode()}, (
        "陆离从 Agent 自留的那份传、小满从 音色锚点/ 传"
    )
    assert "已用本地副本重新托管" in r.content
    assert "临时用本集的独白段" not in r.content, "传上去了就用人定的，不用临时锚点"
    clip = _newest_clip(store, "陆离")
    assert clip.uri.startswith("https://cdn.example.com/") and clip.gen_params["hosted"]


async def test_锚点列表显示人定的存在哪_没配托管过期了照实说(tmp_path: Path):
    store, fns, lib_id, _, _, out = _setup(tmp_path)
    await fns._fn_drama_voice_casting(lib_id)
    _downloaded(store, out, "v1")
    await _decide(fns, "adopt")
    r = await fns._fn_drama_voice_anchors("list")
    assert f"♪ 陆离（人定的）本地：{out / '音色锚点' / '陆离.mp4'}" in r.content

    _expire_casting(store)
    r2 = await fns._fn_drama_voice_anchors("list")
    assert "没配托管（config/hosting.yaml）" in r2.content
    assert "下次渲染会用新片段接替" not in r2.content, "人定的不会被接替"


async def test_手动pin也存一份(tmp_path: Path):
    store, fns, _, _, _, out = _setup(tmp_path)
    clip = store.create("", type_=AssetType.VIDEO, summary="[第1集-2场] 5-6", creator="model:x")
    p = out / "videos" / "第01集-02.mp4"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"pin")
    clip.uri = "https://fake/pin.mp4"
    clip.gen_params["local"] = str(p)
    store.put(clip)
    r = await fns._fn_drama_voice_anchors("pin", character="陆离", clip_id=clip.id)
    assert r.ok and "本地永久存一份：♪ 陆离" in r.content
    assert (out / "音色锚点" / "陆离.mp4").read_bytes() == b"pin"

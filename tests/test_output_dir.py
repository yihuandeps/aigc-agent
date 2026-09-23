"""产物目录（OutputPrefs）—— 文本镜像 / 媒体本地副本 / 成片缺省导出。

用户定的规则：生成的文本、图片、视频都要落到用户指定的文件夹；
开工前问一次，同名 session 记住。所有消费点不传 prefs 时行为与原来一致。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions.media import MediaFunctions
from aigc_agent.domain.functions.video_edit import VideoEditFunctions
from aigc_agent.domain.generators.catalog import MediaCatalog
from aigc_agent.domain.output import OutputPrefs
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.model.media import FakeMediaProvider, MediaGateway

CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "media_models.yaml"


async def _fake_download(url: str, dest: Path) -> bool:
    dest.write_bytes(f"fake-bytes:{url}".encode())  # noqa: ASYNC240 — 测试替身，两字节
    return True


async def _fail_download(url: str, dest: Path) -> bool:  # noqa: ARG001
    return False


# ---------------------------------------------------------------- 文本镜像


def test_文本资产镜像成可读md(tmp_path):
    prefs = OutputPrefs(tmp_path / "out")
    store = AssetStore(tmp_path / "assets")
    store.mirror = prefs

    a = store.create("# 第1集\n正文……", type_=AssetType.SCRIPT, summary="第1集·开场", creator="t")
    files = list(prefs.dir_for("texts").glob("*.md"))
    assert len(files) == 1
    assert "第1集·开场".replace("·", "-") in files[0].name or a.id in files[0].name
    assert files[0].read_text(encoding="utf-8") == "# 第1集\n正文……"

    # 中间产物（分镜 JSON 等 STORYBOARD 类型）不镜像
    store.create('{"x": 1}', type_=AssetType.STORYBOARD, summary="中间产物")
    assert len(list(prefs.dir_for("texts").glob("*.md"))) == 1

    # 改写产新版本 → 新文件，旧的留着（血缘不断，文件也一样）
    store.revise(a.id, "# 第1集 v2", summary="第1集·改")
    assert len(list(prefs.dir_for("texts").glob("*.md"))) == 2


def test_没挂mirror就不写任何地方(tmp_path):
    store = AssetStore(tmp_path / "assets")
    store.create("正文", summary="s")
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------- 媒体本地副本


async def test_生成图下载本地副本_uri保持远端(tmp_path):
    prefs = OutputPrefs(tmp_path / "out", downloader=_fake_download)
    store = AssetStore(tmp_path / "assets")
    catalog = MediaCatalog.load(CATALOG_PATH)
    gw = MediaGateway(
        {"apimart": FakeMediaProvider()},
        EventBus(),
        poll_interval=0.01,
        max_poll_interval=0.02,
    )
    fns = MediaFunctions(gw, catalog, store, prefs=prefs)

    r = await fns._fn_gen_image("一只橘猫", model="qwen-image-2.0")

    assert r.ok
    files = list(prefs.dir_for("images").glob("*.png"))
    assert len(files) == 1
    assert files[0].read_bytes().startswith(b"fake-bytes:")
    a = store.latest(AssetType.IMAGE)
    # 远端 URL 不动 —— 后续镜头的参考图链要它；本地路径在 gen_params
    assert a.uri.startswith("https://")
    assert a.gen_params["local"] == str(files[0])
    assert "本地副本" in r.content


async def test_本地下载失败不拖累生成(tmp_path):
    prefs = OutputPrefs(tmp_path / "out", downloader=_fail_download)
    store = AssetStore(tmp_path / "assets")
    catalog = MediaCatalog.load(CATALOG_PATH)
    gw = MediaGateway(
        {"apimart": FakeMediaProvider()},
        EventBus(),
        poll_interval=0.01,
        max_poll_interval=0.02,
    )
    fns = MediaFunctions(gw, catalog, store, prefs=prefs)

    r = await fns._fn_gen_image("一只橘猫", model="qwen-image-2.0")

    assert r.ok  # 生成本身不受影响
    assert "本地副本" not in r.content
    assert list(prefs.dir_for("images").glob("*.png")) == []


# ---------------------------------------------------------------- 成片导出目录


def test_成片导出目录的优先级(tmp_path):
    store = AssetStore(tmp_path / "assets")
    prefs = OutputPrefs(tmp_path / "out")

    with_prefs = VideoEditFunctions(store, tmp_path / "ws", prefs=prefs)
    assert with_prefs._export_dir("") == prefs.dir_for("exports")
    assert with_prefs._export_dir("D:/指定") == Path("D:/指定")  # 显式传参优先

    legacy = VideoEditFunctions(store, tmp_path / "ws")
    assert legacy._export_dir("") == tmp_path / "ws" / "exports"  # 不传 prefs 行为不变


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

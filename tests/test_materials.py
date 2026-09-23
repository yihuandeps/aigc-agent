"""联网找可用素材（2026-09-18）。

只接明确可商用的素材站（Pexels / Pixabay）；下载后授权信息随资产保存；
任意链接下载授权记 unknown 并提醒；网页抓正文。不打真实网络：注入假的 http_get / downloader。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions.materials import MaterialFunctions, strip_html

PEXELS_VIDEOS = {
    "videos": [
        {
            "id": 1, "duration": 12, "width": 1080, "height": 1920,
            "user": {"name": "Ann"}, "url": "https://www.pexels.com/video/1/",
            "video_files": [
                {"file_type": "video/mp4", "width": 1080, "height": 1920, "link": "https://dl/1.mp4"},
                {"file_type": "video/mp4", "width": 2160, "height": 3840, "link": "https://dl/4k.mp4"},
                {"file_type": "video/webm", "width": 1080, "height": 1920, "link": "https://dl/1.webm"},
            ],
        }
    ]
}
PIXABAY_VIDEOS = {
    "hits": [
        {
            "id": 9, "duration": 8, "user": "Bob", "pageURL": "https://pixabay.com/videos/9/",
            "videos": {"large": {"url": "https://dl/9.mp4", "width": 1080, "height": 1920}},
        }
    ]
}
PEXELS_PHOTOS = {
    "photos": [
        {"id": 5, "width": 2000, "height": 3000, "photographer": "Cy",
         "url": "https://www.pexels.com/photo/5/", "src": {"large2x": "https://dl/5.jpg"}}
    ]
}
HTML = (
    "<html><head><title>芯片 新闻</title><style>x{}</style></head><body>"
    "<script>alert(1)</script><h1>产能</h1><p>第一段&amp;内容</p><div>第二段</div></body></html>"
)


def _fake_get(calls: list[tuple[str, dict[str, Any]]]):
    async def get(
        url: str, headers: dict[str, str], params: dict[str, Any]
    ) -> tuple[int, Any, str]:
        calls.append((url, dict(params)))
        if "api.pexels.com/videos" in url:
            return 200, PEXELS_VIDEOS, ""
        if "api.pexels.com/v1" in url:
            return 200, PEXELS_PHOTOS, ""
        if "pixabay.com/api/videos" in url:
            return 200, PIXABAY_VIDEOS, ""
        if "pixabay.com/api" in url:
            return 200, {"hits": []}, ""
        if url.startswith("https://news/"):
            return 200, None, HTML
        return 404, None, "nope"

    return get


async def _fake_download(url: str, target: Path) -> tuple[bool, str]:
    def write() -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"data:" + url.encode())

    await asyncio.to_thread(write)
    return True, ""


def _fns(
    tmp_path: Path, env: dict[str, str] | None = None
) -> tuple[MaterialFunctions, list, AssetStore]:
    store = AssetStore(tmp_path / "assets")
    calls: list[tuple[str, dict[str, Any]]] = []
    fns = MaterialFunctions(
        store, tmp_path / "ws", http_get=_fake_get(calls), downloader=_fake_download,
        env=env if env is not None else {"PEXELS_API_KEY": "k1", "PIXABAY_API_KEY": "k2"},
    )
    return fns, calls, store


async def test_搜视频_两站合并_挑不超过1080p的mp4(tmp_path: Path):
    fns, calls, _ = _fns(tmp_path)
    r = await fns.invoke("stock_media_search", {"query": "gpu card close up"})
    assert r.ok, r.error
    assert "pexels:v:1" in r.content and "pixabay:v:9" in r.content
    assert "12s · 1080x1920 · by Ann · Pexels License" in r.content
    assert fns._found["pexels:v:1"]["url"] == "https://dl/1.mp4", "4K 和 webm 都不选"
    assert calls[0][1]["orientation"] == "portrait" and calls[0][1]["per_page"] == 5
    args = {"query": "gpu", "kind": "image", "source": "pexels"}
    r2 = await fns.invoke("stock_media_search", args)
    assert r2.ok and "pexels:p:5" in r2.content and "pixabay" not in r2.content


async def test_下载登记_授权随资产保存(tmp_path: Path):
    fns, _, store = _fns(tmp_path)
    await fns.invoke("stock_media_search", {"query": "gpu card"})
    r = await fns.invoke("fetch_stock_media", {"handle": "pexels:v:1", "name": "板卡特写"})
    assert r.ok, r.error
    a = store.get(r.asset_ref)
    assert a.type is AssetType.VIDEO and a.gen_params["license"].startswith("Pexels")
    assert a.gen_params["author"] == "Ann" and a.gen_params["source_url"].startswith("https://www.pexels.com")
    local = Path(a.gen_params["local"])
    assert await asyncio.to_thread(local.exists)
    assert local.parent.name == "pexels" and local.name == "板卡特写.mp4"
    assert a.uri == str(local) and "无需署名" in r.content
    bad = await fns.invoke("fetch_stock_media", {"handle": "pexels:v:404"})
    assert not bad.ok and "先 stock_media_search" in bad.error


async def test_没配key_说清楚怎么配(tmp_path: Path):
    fns, _, _ = _fns(tmp_path, env={})
    r = await fns.invoke("stock_media_search", {"query": "gpu"})
    assert not r.ok and "PEXELS_API_KEY" in r.error and "PIXABAY_API_KEY" in r.error
    h = await fns.health()
    assert not h.ok


async def test_任意链接下载_授权unknown要提醒(tmp_path: Path):
    fns, _, store = _fns(tmp_path)
    args = {"url": "https://cdn/x/clip.mp4?sig=1", "name": "官方物料"}
    r = await fns.invoke("fetch_media_url", args)
    assert r.ok and "unknown" in r.content and "确认权利" in r.content
    a = store.get(r.asset_ref)
    assert a.type is AssetType.VIDEO and a.gen_params["source"] == "url"
    assert Path(a.gen_params["local"]).name == "官方物料.mp4"
    r2 = await fns.invoke("fetch_media_url", {"url": "ftp://x"})
    assert not r2.ok


async def test_网页抓正文(tmp_path: Path):
    fns, _, _ = _fns(tmp_path)
    r = await fns.invoke("web_fetch", {"url": "https://news/1"})
    assert r.ok and r.content.startswith("芯片 新闻")
    assert "产能" in r.content and "第一段&内容" in r.content and "alert" not in r.content
    title, body = strip_html("<p>a</p><p>b</p>", 3)
    assert title == "" and body == "a\nb"
    r2 = await fns.invoke("web_fetch", {"url": "https://nope/"})
    assert not r2.ok and "404" in r2.error


def test_权限():
    store = AssetStore()
    fns = MaterialFunctions(store, Path("."), env={})
    metas = {s.name: s.permission.value for s in fns._specs.values()}
    assert metas["stock_media_search"] == "L-read" and metas["web_fetch"] == "L-read"
    assert metas["fetch_stock_media"] == "L-write" and metas["fetch_media_url"] == "L-write"

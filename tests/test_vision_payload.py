"""给视觉模型的图一律先转成 data URL（2026-09-29 审查 1.2）。

之前没有本地副本的图直接把远端链接发给视觉模型，Kimi 拉不到很多图床（9-21 一个会话 8 次
unsupported image url）；调用失败的真实感校验按「通过」记，服装图人物一致没查成也写成「一致」。
现在：本地文件读出来转、链接先下载再转、都拿不到就记「没查成」，报告里单列。
不连真网络：下载一律换成假的。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions.drama import DramaFunctions, _realism_note
from aigc_agent.domain.media import vision_input

PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 16
# conftest 默认把下载换成了假的（测试不连网）；测下载本身的判断时用真函数、假 HTTP 客户端
_real_fetch = vision_input.fetch_image


@pytest.fixture
def fetched(monkeypatch) -> list[str]:
    """假下载：/ok 结尾的链接给一张 PNG，其余 404。返回下载过的链接。"""
    seen: list[str] = []

    async def fake(url: str) -> tuple[bytes, str, str]:
        seen.append(url)
        if url.endswith("/ok.png"):
            return PNG, "image/png", ""
        return b"", "", "HTTP 404"

    monkeypatch.setattr(vision_input, "fetch_image", fake)
    return seen


def _fns(store: AssetStore, replies: list[dict[str, Any]] | None = None) -> DramaFunctions:
    calls: list[list[dict[str, Any]]] = []
    seq = list(replies or [])

    async def chat(role: str, messages: list[dict[str, Any]], **_: Any) -> Any:
        calls.append(messages)
        return SimpleNamespace(text=json.dumps(seq.pop(0) if seq else {"score": 9, "pass": True}))

    fns = DramaFunctions(SimpleNamespace(chat=chat), store)
    fns._calls = calls  # type: ignore[attr-defined]
    return fns


def _images(messages: list[dict[str, Any]]) -> list[str]:
    return [
        p["image_url"]["url"]
        for m in messages if isinstance(m.get("content"), list)
        for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url"
    ]


def _remote_image(store: AssetStore, url: str) -> str:
    a = store.create("", type_=AssetType.IMAGE, summary="服装")
    a.uri = url
    store.put(a)
    return a.id


# ---------------------------------------------------------------- 转换本身


async def test_本地文件_链接_dataURL_都转成dataURL(tmp_path: Path, fetched: list[str]):
    p = tmp_path / "a.png"
    p.write_bytes(PNG)
    url, why = await vision_input.image_data_url(str(p))
    assert why == "" and url.startswith("data:image/png;base64,")
    url, why = await vision_input.image_data_url("https://img.cdn/ok.png")
    assert why == "" and url.startswith("data:image/png;base64,")
    assert await vision_input.image_data_url("data:image/jpeg;base64,AAAA") == (
        "data:image/jpeg;base64,AAAA", ""
    )


async def test_拿不到的说清楚原因(tmp_path: Path, fetched: list[str]):
    url, why = await vision_input.image_data_url("https://img.cdn/gone.png")
    assert url == "" and "下载不下来" in why and "HTTP 404" in why
    url, why = await vision_input.image_data_url(str(tmp_path / "nope.png"))
    assert url == "" and "不是本地文件也不是链接" in why
    url, why = await vision_input.image_data_url("")
    assert url == "" and why


async def test_过期链接回的错误页不当成图(monkeypatch):
    class Resp:
        status_code = 200
        content = b"<html>expired</html>"
        headers = {"content-type": "text/html; charset=utf-8"}

    class Client:
        def __init__(self, **_: Any) -> None: ...
        async def __aenter__(self) -> Client:
            return self
        async def __aexit__(self, *_: Any) -> None: ...
        async def get(self, url: str) -> Resp:
            return Resp()

    monkeypatch.setattr(vision_input.httpx, "AsyncClient", Client)
    data, mime, why = await _real_fetch("https://img.cdn/x.png")
    assert data == b"" and "不是图片" in why and "text/html" in why


# ---------------------------------------------------------------- 短剧的视觉门


async def test_真实感校验_没本地副本的图先下载再发(fetched: list[str]):
    store = AssetStore()
    aid = _remote_image(store, "https://img.cdn/ok.png")
    fns = _fns(store, [{"score": 8, "pass": True}])
    passed, score, _, _, _, unchecked = await fns._check_realism(aid, "")
    assert passed and score == 8 and not unchecked
    assert fetched == ["https://img.cdn/ok.png"]
    sent = _images(fns._calls[0])  # type: ignore[attr-defined]
    assert sent and all(u.startswith("data:image/png;base64,") for u in sent), "不发远端链接"


async def test_真实感校验_图拿不到记没查成_不调视觉模型(fetched: list[str]):
    store = AssetStore()
    aid = _remote_image(store, "https://img.cdn/gone.png")
    fns = _fns(store)
    passed, score, _, note, _, unchecked = await fns._check_realism(aid, "")
    assert passed and score == -1 and unchecked and "没查成" in note
    assert not fns._calls, "图都拿不到，不花这次视觉调用"  # type: ignore[attr-defined]


async def test_人物一致_参考图是链接或本地路径都转成dataURL(tmp_path: Path, fetched: list[str]):
    store = AssetStore()
    target = tmp_path / "cos.png"
    target.write_bytes(PNG)
    t = store.create("", type_=AssetType.IMAGE, summary="服装")
    t.gen_params["local"] = str(target)
    store.put(t)
    face = tmp_path / "face.png"
    face.write_bytes(PNG)
    fns = _fns(store, [{"score": 9}])
    refs = [("参考图1", "https://img.cdn/ok.png"), ("参考图2", str(face))]
    v = await fns._check_identity(t.id, refs, is_video=False, pass_score=7)
    assert v.passed and not v.unchecked
    sent = _images(fns._calls[0])  # type: ignore[attr-defined]
    assert len(sent) == 3 and all(u.startswith("data:") for u in sent)


async def test_人物一致_生成图拿不到_记没查成(fetched: list[str]):
    store = AssetStore()
    aid = _remote_image(store, "https://img.cdn/gone.png")
    fns = _fns(store)
    v = await fns._check_identity(aid, [("参考图1", "https://img.cdn/ok.png")], False, 7)
    assert v.unchecked and "生成图拿不到" in v.note and "HTTP 404" in v.note


async def test_服装图人物一致没查成_报告单列_不算一致(fetched: list[str]):
    store = AssetStore()
    aid = _remote_image(store, "https://img.cdn/gone.png")
    fns = _fns(store)
    fns.catalog = SimpleNamespace(drama={"identity_gate": "true", "identity_retries": "1"})
    entry: dict[str, Any] = {"name": "服装·夜行衣", "notes": [], "checked": True, "pass": True}
    out = await fns._identity_gate_image({}, "提示词", ["https://img.cdn/ok.png"], entry, aid, "")
    assert out == (aid, "")  # 照旧放行（拦不拦待用户拍板）
    assert entry["identity"]["unchecked"] is True
    note = _realism_note([entry])
    assert "0 张与主形象一致" in note and "1 张没查成" in note and "服装·夜行衣" in note


def test_真实感没查成_报告单列_不算一次通过():
    log = [
        {"name": "角色·A", "notes": [], "checked": True, "pass": True, "attempts": 1},
        {"name": "角色·B", "notes": [], "checked": True, "pass": True, "attempts": 1,
         "unchecked": True, "note": "没查成：图片链接下载不下来（HTTP 404）"},
    ]
    note = _realism_note(log)
    assert "一次通过 1" in note and "· 没查成 1" in note
    assert "角色·B" in note[note.index("⚠ 没查成 1 张"):]

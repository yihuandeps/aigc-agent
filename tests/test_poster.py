"""海报 / 图文卡片（2026-09-23 从 E:\\aigc-agent 移植，设计产线用）。

图文要的是"一张有字的图"：生图模型写不好中文，字本地叠。盯：
  1. 三个模板都能出图，底图盖满不变形，长标题自动缩字换行，找不到字体不静默
  2. make_poster 落 image 资产、挂血缘；底图不是图片 / 不存在都报错；本地渲染不花钱
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions.poster import PosterFunctions
from aigc_agent.domain.media import poster as P
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.tools.provider import PermissionLevel
from aigc_agent.harness.tools.registry import ToolRegistry


def _png(w: int = 600, h: int = 400, color: tuple[int, int, int] = (30, 60, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()


def _size(png: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(png)).size


# ---------------------------------------------------------------- 渲染


@pytest.mark.parametrize("template", list(P.TEMPLATES))
def test_三个模板都能出图(template: str):
    r = P.render(
        P.PosterSpec(
            title="五个妈妈一个娃",
            subtitle="第1集 · 天上掉下个蛛妹妹\n每集 4 分钟\n开场 15 秒高潮点",
            template=template,
            brand="@西游记",
        )
    )
    assert r.png[:8] == b"\x89PNG\r\n\x1a\n" and _size(r.png) == (1080, 1440)
    assert r.title_lines >= 1 and r.template == template and r.font


def test_底图盖满不变形_比例与错误():
    r = P.render(P.PosterSpec(title="标题", background=_png(300, 900), aspect_ratio="1:1"))
    assert _size(r.png) == (1080, 1080)
    assert _size(P.render(P.PosterSpec(title="标题", aspect_ratio="9:16")).png) == (1080, 1920)
    with pytest.raises(ValueError, match="比例"):
        P.render(P.PosterSpec(title="x", aspect_ratio="2:1"))
    with pytest.raises(ValueError, match="模板"):
        P.render(P.PosterSpec(title="x", template="neon"))
    with pytest.raises(ValueError, match="标题"):
        P.render(P.PosterSpec(title="   "))
    with pytest.raises(ValueError, match="底图"):
        P.render(P.PosterSpec(title="x", background=b"not an image"))


def test_长标题自动缩字并换行():
    long = "这是一个特别特别长的标题用来测试自动换行和缩小字号是否正常工作到底行不行"
    r = P.render(P.PosterSpec(title=long, template="clean"))
    assert 1 < r.title_lines <= 3
    assert P.render(P.PosterSpec(title=long, template="card")).title_lines <= 2


def test_换行按像素宽_中文逐字英文按词():
    draw = ImageDraw.Draw(Image.new("RGB", (400, 100)))
    font = P.load_font(P.find_font(), 24)
    lines = P.wrap(draw, "海报怎么做 poster design guide", font, 200)
    assert len(lines) >= 2
    assert all(draw.textlength(ln, font=font) <= 200 for ln in lines)
    assert not any(ln.startswith(" ") for ln in lines)
    assert P.wrap(draw, "a\nb", font, 1000) == ["a", "b"]


def test_颜色解析与字体查找(monkeypatch, tmp_path: Path):
    assert P.parse_color("#ff5733", (0, 0, 0)) == (255, 87, 51)
    assert P.parse_color("1, 2,3", (0, 0, 0)) == (1, 2, 3)
    assert P.parse_color("", (9, 9, 9)) == (9, 9, 9)
    with pytest.raises(ValueError):
        P.parse_color("red", (0, 0, 0))
    monkeypatch.setenv("AIGC_POSTER_FONT", str(tmp_path / "nope.ttf"))
    assert P.render(P.PosterSpec(title="x")).font, "指定的字体不存在就回落，不报错"
    assert P.load_font(tmp_path / "nope.ttf", 20) is not None


# ---------------------------------------------------------------- function


async def test_make_poster落资产并挂血缘(tmp_path: Path):
    store = AssetStore(tmp_path / "assets")
    bg = store.create_blob(
        _png(), ".png", type_=AssetType.IMAGE, mime="image/png", summary="底图", creator="m"
    )
    draft = store.create("正文", creator="model:fake")
    fn = PosterFunctions(store)
    reg = ToolRegistry(EventBus())
    reg.register(fn)
    await reg.refresh()
    assert reg.meta("make_poster").permission is PermissionLevel.WRITE, "本地渲染，不花钱"

    r = await reg.invoke(
        "make_poster",
        {
            "title": "五个妈妈一个娃",
            "subtitle": "三条要点",
            "template": "card",
            "background_asset_id": bg.id,
            "parent_id": draft.id,
            "brand": "@西游记",
            "accent": "#2563eb",
        },
    )
    assert r.ok, r.error
    a = store.get(r.asset_ref)
    assert a.type is AssetType.IMAGE and a.mime == "image/png"
    assert a.parent_ids == [bg.id, draft.id] and a.gen_params["template"] == "card"
    assert Image.open(a.uri or "").size == (1080, 1440)
    assert "海报已生成" in r.content and a.id in r.content

    r = await reg.invoke("make_poster", {"title": "x", "background_asset_id": draft.id})
    assert not r.ok and "image" in r.error
    r = await reg.invoke("make_poster", {"title": "x", "background_asset_id": "as_nope"})
    assert not r.ok
    r = await reg.invoke("make_poster", {"title": "x", "template": "neon"})
    assert not r.ok and "模板" in r.error
    assert (await fn.health()).ok

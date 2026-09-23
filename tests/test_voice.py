"""角色音色锁定（2026-09-18）—— 多集视频里同一角色的声音不能漂。

用户发现：集数一多，同一角色的声音会漂。根因是 seedance 逐段发声，每段各自"发明"
一个声音。两道锁：
  ② 资产库：每个角色一张音色卡（voice）
  ⑤ 渲染：说话角色的音色卡锁进提示词开头；该角色的锚点片段（第一段独白）当参考视频
     （@视频N 取音色），锚点存成资产跨集沿用；链接过期用新片段接替
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.drama import (
    Anchor,
    assets_system,
    choose_anchor_shot,
    dump_anchors,
    normalize,
    parse_anchors,
    parse_assets,
    plan_anchors,
    speakers_of,
    voice_block,
)
from aigc_agent.domain.drama.models import ShotPrompt, as_dict
from aigc_agent.domain.drama.voice import ANCHORS_CREATOR
from aigc_agent.domain.functions.drama import DramaFunctions
from aigc_agent.domain.functions.media import MediaFunctions
from aigc_agent.domain.generators.catalog import MediaCatalog
from aigc_agent.domain.output import OutputPrefs
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.model.media import FakeMediaProvider, MediaGateway
from aigc_agent.harness.tools.provider import ToolResult

LIB = {
    "characters": [
        {
            "baseRoleName": "陆离",
            "roleTotalDesc": "男 | 35岁 [强制白底证件照]",
            "voice": "男声 | 35岁听感 | 中低音、略沙哑 | 语速慢、句尾下压 | 普通话无口音",
            "roleCostumeList": [{"costumeName": "陆离-风衣-[全集]", "costumeDesc": "风衣"}],
        },
        {
            "baseRoleName": "小满",
            "roleTotalDesc": "女 | 22岁 [强制白底证件照]",
            "voiceDesc": "女声 | 20出头 | 偏高、清亮带气声 | 语速快、爱抢话 | 轻微川普口音",
            "roleCostumeList": [{"costumeName": "小满-连帽衫-[全集]", "costumeDesc": "连帽衫"}],
        },
        {"baseRoleName": "老鬼", "roleTotalDesc": "男 | 60岁", "roleCostumeList": []},
    ],
    "scenes": [{"name": "地铁车厢", "description": "x"}],
    "props": [],
}


def _lib():
    lib, err = parse_assets(json.dumps(LIB, ensure_ascii=False))
    assert not err, err
    return lib


def _shot(scene: str, name: str, desc: str) -> ShotPrompt:
    return ShotPrompt(scene_index=scene, video_name=name, duration="10s", description=desc)


# ---------------------------------------------------------------- ② 音色卡


def test_解析音色卡_兼容几种字段名_落库带voice():
    lib = _lib()
    luli, xiaoman, laogui = lib.characters
    assert luli.voice.startswith("男声")
    assert xiaoman.voice.startswith("女声"), "voiceDesc 也认"
    assert laogui.voice == ""
    assert as_dict(luli)["voice"] == luli.voice


def test_资产库提示词要求音色卡():
    text = assets_system(normalize("asian", "zh"))
    assert "音色卡协议" in text and '"voice"' in text
    for kw in ("音高", "语速", "口音", "互相区分"):
        assert kw in text, kw


# ---------------------------------------------------------------- 谁在说话


def test_台词归最近的前一个角色_L_Cut也算():
    lib = _lib()
    s = _shot(
        "[第1集-1场]", "1-4",
        "场景设定: (地铁车厢)。(陆离-风衣-[全集]) 压低声音说：“别回头。”"
        "(小满-连帽衫-[全集]) 一愣：“为什么？” 又追问：“你到底是谁？”"
        "(陆离-风衣-[全集])(L-Cut Voice-over)说：“数到三。”",
    )
    assert speakers_of(s, lib) == {"陆离": 2, "小满": 2}
    assert speakers_of(_shot("[第1集-2场]", "5-6", "(地铁车厢) 空镜，雨声。"), lib) == {}
    # 引号前没出现角色（旁白）不计；直引号也认
    s2 = _shot("[第1集-3场]", "7", '“很久以前……” (老鬼) 咳了两声："都过去了。"')
    assert speakers_of(s2, lib) == {"老鬼": 1}


def test_锚点镜头_独白优先_其次台词最多_最后取最早():
    by_shot = [{"陆离": 1, "小满": 2}, {"陆离": 1}, {"陆离": 3}, {"小满": 5, "老鬼": 1}]
    assert choose_anchor_shot(by_shot, "陆离") == 2, "独白里台词最多的"
    assert choose_anchor_shot(by_shot, "小满") == 3, "没独白就挑台词最多的"
    assert choose_anchor_shot(by_shot, "老鬼") == 3
    assert choose_anchor_shot(by_shot, "路人") is None


def test_锚点计划_沿用_过期重定_缺音色卡():
    lib = _lib()
    by_shot = [{"陆离": 1, "小满": 1}, {"陆离": 2}, {"老鬼": 1}]
    existing = {
        "陆离": Anchor(character="陆离", asset="as_old"),
        "小满": Anchor(character="小满", asset="as_stale"),
    }
    plan = plan_anchors(by_shot, lib, existing, usable={"陆离"})
    assert list(plan.ready) == ["陆离"]
    assert plan.stale == ["小满"] and plan.births["小满"] == 0
    assert plan.births["老鬼"] == 2
    assert plan.unvoiced == ["老鬼"]
    assert plan.anchor_shots == {0, 2}


def test_音色段_锚点编号与前序编号():
    lib = _lib()
    text = voice_block(["陆离", "小满"], lib, {"陆离": 2}, [("第1集-1场", 1)])
    assert text.startswith("【音色锁定】")
    assert "(陆离)：男声" in text and "@视频2" in text
    assert "(小满)：女声" in text and "@视频1" not in text.split("(小满)")[1].split("。")[0]
    assert "{第1集-1场} 即 @视频1" in text
    assert "只取声音" in text
    assert voice_block([], lib, {}, []) == ""


def test_锚点表序列化往返():
    anchors = {"陆离": Anchor(character="陆离", asset="as_1", scene="[第1集-1场]", solo=True)}
    back = parse_anchors(dump_anchors(anchors))
    assert back["陆离"].asset == "as_1" and back["陆离"].solo is True
    assert parse_anchors("not json") == {}


# ---------------------------------------------------------------- ⑤ 渲染链路


class FakeRegistry:
    def __init__(self, store: AssetStore) -> None:
        self.store = store
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
        a = self.store.create("", type_=AssetType.VIDEO, summary=args["summary"], creator="fake")
        a.uri = f"https://fake/{a.id}.mp4"
        self.store.put(a)
        return ToolResult(ok=True, content="ok", asset_ref=a.id)


SHOTS = [
    {  # 两人对话：不是任何人的独白
        "scene_index": "[第1集-1场]",
        "video_name": "1-4",
        "video_duration": "10s",
        "description": (
            "(地铁车厢) (陆离-风衣-[全集]) 说：“别回头。” (小满-连帽衫-[全集]) 问：“为什么？”"
        ),
    },
    {  # 陆离独白 → 陆离的锚点
        "scene_index": "[第1集-2场]",
        "video_name": "5-6",
        "video_duration": "10s",
        "description": "(地铁车厢) (陆离-风衣-[全集]) 自语：“数到三。” 又说：“三。”",
    },
    {  # 小满独白，且引入前序 → 小满的锚点，排在依赖层
        "scene_index": "[第1集-3场]",
        "video_name": "7-8",
        "video_duration": "10s",
        "description": "{第1集-2场} (地铁车厢) (小满-连帽衫-[全集]) 喊：“等等我！”",
    },
]


def _seed(store: AssetStore, shots=SHOTS) -> tuple[str, str, str]:
    lib_id = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库", creator="t").id
    sb_id = store.create("分镜", summary="分镜", creator="t").id
    shots_id = store.create(
        json.dumps(shots, ensure_ascii=False),
        summary="提示词",
        creator="tool:drama_shots",
        parents=[sb_id, lib_id],
    ).id
    # 2026-09-20 引用门：分镜引用到的角色/场景都得在包里，否则花钱之前整批拦
    pack = {"陆离": {"asset": "x", "url": "https://img/luli", "kind": "角色"},
            "小满": {"asset": "y", "url": "https://img/xiaoman", "kind": "角色"},
            "老鬼": {"asset": "z", "url": "https://img/laogui", "kind": "角色"},
            "地铁车厢": {"asset": "s", "url": "https://img/car", "kind": "场景"}}
    pack_id = store.create(
        json.dumps(pack, ensure_ascii=False),
        summary="参考图包",
        creator="tool:drama_render_assets",
        parents=[lib_id],
    ).id
    return lib_id, shots_id, pack_id


def _fns(store: AssetStore, fake: FakeRegistry) -> DramaFunctions:
    return DramaFunctions(None, store, registry=fake, catalog=None)


async def test_首集_独白段先渲当锚点_对话段带两人锚点():
    store = AssetStore()
    _, shots_id, pack_id = _seed(store)
    fake = FakeRegistry(store)
    r = await _fns(store, fake)._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.ok, r.error

    by = {a["summary"]: a for a in fake.videos()}
    order = [a["summary"] for a in fake.videos()]
    # 陆离独白先渲（第 0 层）；小满独白带前序引入排第 1 层；两人对话等两个锚点都有了才渲
    keys = ("[第1集-2场] 5-6", "[第1集-3场] 7-8", "[第1集-1场] 1-4")
    solo, xm_solo, duo_ = (order.index(k) for k in keys)
    assert solo < xm_solo < duo_

    luli_clip = next(a for a in store.all() if a.summary == "[第1集-2场] 5-6").uri
    xm_clip = next(a for a in store.all() if a.summary == "[第1集-3场] 7-8").uri
    duo = by["[第1集-1场] 1-4"]
    assert duo["video_urls"] == [luli_clip, xm_clip]
    assert "(陆离)：男声" in duo["prompt"] and "@视频1" in duo["prompt"]
    assert "(小满)：女声" in duo["prompt"] and "@视频2" in duo["prompt"]
    assert duo["prompt"].index("【音色锁定】") < duo["prompt"].index("别回头")
    # 小满独白：前序片段 @视频1，小满自己还没锚点
    xm = by["[第1集-3场] 7-8"]
    assert xm["video_urls"] == [luli_clip]
    assert "{第1集-2场} 即 @视频1" in xm["prompt"] and "@视频2" not in xm["prompt"]
    # 锚点表落库
    anchors = next(a for a in store.all() if a.creator == ANCHORS_CREATOR)
    table = json.loads(store.content(anchors.id))
    assert table["陆离"]["solo"] is True and table["陆离"]["scene"] == "[第1集-2场]"
    assert table["小满"]["scene"] == "[第1集-3场]"
    assert "本次新定 2 人" in r.content and "音色 2 人" in r.content


async def test_第二集_沿用锚点_不再重定():
    store = AssetStore()
    _, shots_id, pack_id = _seed(store)
    fake = FakeRegistry(store)
    fns = _fns(store, fake)
    assert (await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)).ok
    luli_clip = next(a for a in store.all() if a.summary == "[第1集-2场] 5-6").uri

    ep2 = [{
        "scene_index": "[第2集-1场]", "video_name": "1-3", "video_duration": "10s",
        "description": "(地铁车厢) (陆离-风衣-[全集]) 低声：“又见面了。”",
    }]
    _, shots2, _ = _seed(store, ep2)
    fake.calls.clear()
    r = await fns._fn_drama_render_shots(shots2, rendered_id=pack_id, episode=2)
    assert r.ok, r.error
    v = fake.videos()[0]
    assert v["video_urls"] == [luli_clip], "第 2 集对着第 1 集定下的同一段声音"
    assert "锚点沿用 1 人" in r.content and "新定" not in r.content
    assert len([a for a in store.all() if a.creator == ANCHORS_CREATOR]) == 1, "没变就不重写锚点表"


async def test_锚点链接过期_用新片段接替并写回():
    store = AssetStore()
    _, shots_id, pack_id = _seed(store)
    fake = FakeRegistry(store)
    fns = _fns(store, fake)
    assert (await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)).ok
    old = next(a for a in store.all() if a.summary == "[第1集-2场] 5-6")
    old.created_at = time.time() - 30 * 3600  # 过了 24h
    store.put(old)

    # 默认 reuse=true 会复用上次全部片段：没有新片段，锚点接替不了，要说清楚
    fake.calls.clear()
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.ok, r.error
    assert "没有新片段可接替" in r.content and "陆离" in r.content
    assert not fake.videos(), "全部复用，没花钱"

    # 重生成：独白段重新出片，接替为新锚点并写回
    fake.calls.clear()
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, reuse=False)
    assert r.ok, r.error
    assert "旧锚点链接过期已接替" in r.content and "陆离" in r.content
    latest = [a for a in store.all() if a.creator == ANCHORS_CREATOR][-1]
    table = json.loads(store.content(latest.id))
    assert table["陆离"]["asset"] != old.id and table["陆离"]["rolled_from"] == old.id


async def test_人工指定锚点_渲染就用它():
    store = AssetStore()
    _, shots_id, pack_id = _seed(store)
    fake = FakeRegistry(store)
    fns = _fns(store, fake)
    clip = store.create("", type_=AssetType.VIDEO, summary="[第0集-1场] 试音", creator="fake")
    clip.uri = "https://fake/pinned.mp4"
    store.put(clip)
    r = await fns._fn_drama_voice_anchors("pin", character="陆离", clip_id=clip.id)
    assert r.ok, r.error
    listed = await fns._fn_drama_voice_anchors("list")
    assert "陆离" in listed.content and "人工指定" in listed.content

    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.ok, r.error
    duo = next(a for a in fake.videos() if a["summary"] == "[第1集-1场] 1-4")
    assert "https://fake/pinned.mp4" in duo["video_urls"]
    # 陆离沿用人工锚点，独白段不再作为锚点镜头抢先
    assert "锚点沿用 1 人（陆离）" in r.content

    r = await fns._fn_drama_voice_anchors("clear", character="陆离")
    assert r.ok
    listed = (await fns._fn_drama_voice_anchors("list")).content.split("：", 1)[1]
    assert "陆离" not in listed
    bad = await fns._fn_drama_voice_anchors("pin", character="陆离", clip_id="as_nope")
    assert not bad.ok


async def test_没有资产库_不锁音色但明确警告():
    store = AssetStore()
    shots_id = store.create(json.dumps(SHOTS, ensure_ascii=False), summary="提示词", creator="t").id
    fake = FakeRegistry(store)
    r = await _fns(store, fake)._fn_drama_render_shots(shots_id)
    assert r.ok, r.error
    assert "没有锁音色" in r.content
    assert all("video_urls" not in a or a["video_urls"] for a in fake.videos())
    assert "音色锁定】" not in fake.videos()[0]["prompt"]


async def test_关掉参考视频模式_仍锁音色卡文字():
    store = AssetStore()
    _, shots_id, pack_id = _seed(store)
    fake = FakeRegistry(store)
    fns = DramaFunctions(None, store, registry=fake, catalog=None)
    fns.catalog = type(
        "C", (), {"drama": {"voice_anchor": "false"}, "max_concurrency": lambda s, k: 0}
    )()
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.ok, r.error
    duo = next(a for a in fake.videos() if a["summary"] == "[第1集-1场] 1-4")
    assert "video_urls" not in duo and "(陆离)：男声" in duo["prompt"]
    assert "参考视频模式已关" in r.content
    assert not [a for a in store.all() if a.creator == ANCHORS_CREATOR]


# ---------------------------------------------------------------- gen_video 字段


CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "media_models.yaml"


async def _fake_download(url: str, dest: Path) -> bool:
    await asyncio.to_thread(dest.write_bytes, b"x")
    return True


async def test_gen_video_参考视频走配置的字段名(tmp_path):
    catalog = MediaCatalog.load(CATALOG_PATH)
    assert catalog.video_ref_field == "video_urls" and catalog.image_ref_field == "image"
    provider = FakeMediaProvider(urls=["https://example.com/out.mp4"])
    gw = MediaGateway({"apimart": provider}, EventBus(), poll_interval=0.01, max_poll_interval=0.02)
    fns = MediaFunctions(gw, catalog, AssetStore(tmp_path / "a"),
                         prefs=OutputPrefs(tmp_path / "out", downloader=_fake_download))
    r = await fns._fn_gen_video(
        "x", model="seedance-2.0", image=["https://i/1"], video_urls=["https://v/1"],
        audio_urls=["https://a/1"],
    )
    assert r.ok, r.error
    sent = provider.submitted[-1]
    assert sent["image"] == ["https://i/1"]
    assert sent["video_urls"] == ["https://v/1"] and sent["audio_urls"] == ["https://a/1"]

    # 字段名改成和图片同名时合并成一个列表（接口只认 image 的情况）
    catalog.video_ref_field = "image"
    await fns._fn_gen_video("x", model="seedance-2.0", image=["https://i/1"], video_urls=["https://v/1"])
    assert provider.submitted[-1]["image"] == ["https://i/1", "https://v/1"]

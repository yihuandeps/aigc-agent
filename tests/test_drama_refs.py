"""分镜渲染的参考图绑定（2026-09-18 线上失败的回归）。

真实情况：模型先 drama_render_assets(only="characters") 省钱，包里只有「小满」「陆离」这类
主形象；分镜提示词引用的却是服装名 (小满-旧连帽衫-[前10集]) 和场景名 (地铁车厢)。
逐镜按名字精确匹配全空，4 段视频「参考 0 张」静默渲完 —— 钱花了，脸没保住。

盯五件事：
  1. 服装名在包里没有 → 退回该角色的主形象（保脸）
  2. 包和分镜一个都对不上 → 花钱之前就报错
  3. 没传 rendered_id → 按分镜的血缘自动找这套资产库最新的包；传成资产库 id → 自动换算
  4. 参考图顺序：人物 → 前序片段 → 场景道具
  5. drama_render_assets 增量复用：补跑只生成缺的，主形象不重花钱、不换脸
"""

from __future__ import annotations

import json

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions.drama import DramaFunctions, _resolve_ref
from aigc_agent.harness.tools.provider import ToolResult


class FakeRegistry:
    def __init__(self, store: AssetStore) -> None:
        self.store = store
        self.calls: list[tuple[str, dict]] = []

    async def invoke(self, name, args):
        self.calls.append((name, args))
        if name == "compose_video":
            a = self.store.create("mp4", summary="成片", creator="fake")
            a.uri = "https://fake/composed.mp4"
            self.store.put(a)
            return ToolResult(ok=True, content="拼好了", asset_ref=a.id)
        a = self.store.create("", summary=args.get("summary", name), creator="fake")
        a.uri = f"https://fake/{a.id}"
        self.store.put(a)
        return ToolResult(ok=True, content="ok", asset_ref=a.id)

    def videos(self) -> list[dict]:
        return [a for n, a in self.calls if n == "gen_video"]

    def images(self) -> list[dict]:
        return [a for n, a in self.calls if n == "gen_image"]


LIB = {
    "characters": [
        {
            "baseRoleName": "小满",
            "roleTotalDesc": "12岁女孩",
            "roleCostumeList": [{"costumeName": "小满-旧连帽衫-[前10集]", "costumeDesc": "旧卫衣"}],
        },
        {
            "baseRoleName": "陆离",
            "roleTotalDesc": "守门人",
            "roleCostumeList": [{"costumeName": "陆离-深色风衣-[8/1]", "costumeDesc": "风衣"}],
        },
    ],
    "scenes": [{"name": "地铁车厢", "description": "末班车"}],
    "props": [{"name": "暗色骨簪", "description": "骨簪"}],
}

SHOTS = [
    {
        "scene_index": "[第1集-1场]",
        "video_name": "1-4",
        "video_duration": "14s",
        "description": "场景设定: (地铁车厢) [夜] [内]。(小满-旧连帽衫-[前10集]) 蜷缩在角落。",
    },
    {
        "scene_index": "[第1集-2场]",
        "video_name": "5-8",
        "video_duration": "14s",
        "description": "(陆离-深色风衣-[8/1]) 拿起 (暗色骨簪)，接上 {第1集-1场}。",
    },
]


def _seed(store: AssetStore) -> tuple[str, str]:
    lib_id = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库",
                          creator="tool:drama_assets", type_=AssetType.STORYBOARD).id
    sb_id = store.create("[]", summary="分镜脚本", creator="tool:drama_storyboard").id
    shots_id = store.create(
        json.dumps(SHOTS, ensure_ascii=False), summary="提示词", creator="tool:drama_shots",
        parents=[sb_id, lib_id], type_=AssetType.STORYBOARD,
    ).id
    return lib_id, shots_id


def _pack(store: AssetStore, lib_id: str, names: dict[str, str]) -> str:
    """手工造一个参考图包（老格式，没有 kind）。"""
    images = {n: {"asset": f"as_{i:010x}", "url": u} for i, (n, u) in enumerate(names.items())}
    return store.create(
        json.dumps(images, ensure_ascii=False), summary=f"资产参考图·{len(images)}张",
        creator="tool:drama_render_assets", parents=[lib_id], type_=AssetType.STORYBOARD,
    ).id


def _fns(store: AssetStore, fake: FakeRegistry) -> DramaFunctions:
    return DramaFunctions(None, store, registry=fake)


# ---------------------------------------------------------------- 匹配规则


def test_服装名退回角色主形象():
    images = {"小满": {"url": "u1"}, "地铁车厢": {"url": "u2", "kind": "场景"}}
    m = _resolve_ref("小满-旧连帽衫-[前10集]", images)
    assert m is not None and m.key == "小满" and m.url == "u1" and m.person
    m2 = _resolve_ref("地铁车厢", images)
    assert m2 is not None and m2.key == "地铁车厢" and not m2.person
    assert _resolve_ref("暗色骨簪", images) is None
    assert _resolve_ref("小 满", images).key == "小满"  # 空格/大小写容错


# ---------------------------------------------------------------- render_shots


async def test_只渲了主形象的包_服装引用退回主形象且有提示():
    """线上那次：包里只有 7 个主形象，分镜引用服装名 —— 之前参考 0 张。"""
    store = AssetStore()
    lib_id, shots_id = _seed(store)
    pack = _pack(store, lib_id, {"小满": "https://img/xiaoman", "陆离": "https://img/luli"})
    fake = FakeRegistry(store)

    # 2026-09-20 用户定的规则：引用了包里没有的东西（场景、道具）→ 花钱之前整批拦下
    r = await _fns(store, fake)._fn_drama_render_shots(shots_id, rendered_id=pack)
    assert not r.ok and fake.videos() == [], "缺参考图不能先生成再说"
    assert "没有发起任何生成" in r.error and "「地铁车厢」" in r.error and "「暗色骨簪」" in r.error
    assert f'drama_render_assets(assets_id="{lib_id}", reuse=true)' in r.error

    # 补齐场景和道具后：服装名仍退回主形象（保脸），并有提示
    full = _pack(store, lib_id, {
        "小满": "https://img/xiaoman", "陆离": "https://img/luli",
        "地铁车厢": "https://img/car", "暗色骨簪": "https://img/zan",
    })
    r = await _fns(store, fake)._fn_drama_render_shots(shots_id, rendered_id=full)
    assert r.ok, r.error
    v = fake.videos()
    assert v[0]["image"] == ["https://img/xiaoman", "https://img/car"], "服装没图就用主形象保脸"
    assert v[1]["image"][0] == "https://img/luli"
    assert "退回主形象" in r.content and "小满-旧连帽衫-[前10集]→小满" in r.content
    assert "参考 0 图" not in r.content and "缺" not in r.content
    clips = next(a for a in store.all() if a.creator == "tool:drama_render_shots")
    assert clips.parent_ids == [shots_id, full]
    assert clips.gen_params["refs_resolved"] == 4 and clips.gen_params["refs_missing"] == 0


async def test_包里一个都对不上_花钱之前报错():
    store = AssetStore()
    lib_id, shots_id = _seed(store)
    wrong = _pack(store, lib_id, {"ANNE": "https://img/anne", "COLIN": "https://img/colin"})
    fake = FakeRegistry(store)

    r = await _fns(store, fake)._fn_drama_render_shots(shots_id, rendered_id=wrong)
    assert not r.ok
    assert "一个都对不上" in r.error and "ANNE" in r.error and "小满-旧连帽衫" in r.error
    assert fake.videos() == [], "不能先花钱再说"


async def test_传成资产库id自动换成它的包():
    store = AssetStore()
    lib_id, shots_id = _seed(store)
    pack = _pack(store, lib_id, {
        "小满": "https://img/xiaoman", "陆离": "https://img/luli",
        "地铁车厢": "https://img/car", "暗色骨簪": "https://img/zan",
    })
    fake = FakeRegistry(store)

    r = await _fns(store, fake)._fn_drama_render_shots(shots_id, rendered_id=lib_id)
    assert r.ok, r.error
    assert f"已换成它的参考图包 {pack}" in r.content
    assert fake.videos()[0]["image"][0] == "https://img/xiaoman"

    # 资产库还没渲过图：说清楚该先跑什么
    store2 = AssetStore()
    lib2, shots2 = _seed(store2)
    r2 = await _fns(store2, FakeRegistry(store2))._fn_drama_render_shots(shots2, rendered_id=lib2)
    assert not r2.ok and "还没渲过参考图" in r2.error and "drama_render_assets" in r2.error


async def test_没传rendered_id按血缘自动找最新的包():
    store = AssetStore()
    lib_id, shots_id = _seed(store)
    _pack(store, lib_id, {"小满": "https://img/old"})
    newest = _pack(store, lib_id, {
        "小满": "https://img/new", "陆离": "https://img/luli",
        "地铁车厢": "https://img/car", "暗色骨簪": "https://img/zan",
    })
    fake = FakeRegistry(store)

    r = await _fns(store, fake)._fn_drama_render_shots(shots_id)
    assert r.ok, r.error
    assert f"自动使用这套资产库最新的参考图包 {newest}" in r.content
    assert fake.videos()[0]["image"][0] == "https://img/new"


async def test_完全没有包_照常渲但明确警告():
    store = AssetStore()
    shots_id = store.create(json.dumps(SHOTS, ensure_ascii=False), summary="提示词", creator="t").id
    fake = FakeRegistry(store)
    r = await _fns(store, fake)._fn_drama_render_shots(shots_id)
    assert r.ok, r.error
    assert "没有参考图包" in r.content and "一致性无法保证" in r.content
    assert "image" not in fake.videos()[0]


async def test_参考图顺序_人物先于前序片段与场景道具():
    store = AssetStore()
    lib_id, shots_id = _seed(store)
    pack = _pack(
        store, lib_id,
        {"陆离": "https://img/luli", "小满": "https://img/xiaoman",
         "暗色骨簪": "https://img/zan", "地铁车厢": "https://img/car"},
    )
    fake = FakeRegistry(store)
    r = await _fns(store, fake)._fn_drama_render_shots(shots_id, rendered_id=pack)
    assert r.ok, r.error
    second = fake.videos()[1]
    first_clip = next(a for a in store.all() if a.summary == "[第1集-1场] 1-4").uri
    # 第 2 段参考图：陆离（人物，退回主形象）→ 骨簪（道具）；
    # 前序片段是视频，2026-09-18 起单独走 video_urls（提示词里 @视频1）
    assert second["image"] == ["https://img/luli", "https://img/zan"]
    assert second["video_urls"] == [first_clip]


async def test_不是包也不是资产库的id直接报错():
    store = AssetStore()
    _, shots_id = _seed(store)
    junk = store.create("随便一段文本", summary="x").id
    r = await _fns(store, FakeRegistry(store))._fn_drama_render_shots(shots_id, rendered_id=junk)
    assert not r.ok and "不是 drama_render_assets 的产物" in r.error


# ---------------------------------------------------------------- render_assets 增量


async def test_补跑只生成缺的_主形象复用不换脸():
    store = AssetStore()
    lib_id, _ = _seed(store)
    fake = FakeRegistry(store)
    fns = _fns(store, fake)

    r1 = await fns._fn_drama_render_assets(lib_id, only="characters")
    assert r1.ok and len(fake.images()) == 2
    assert "覆盖：角色 2/2 · 服装 0/2 · 场景 0/1 · 道具 0/1" in r1.content
    assert "退回角色主形象" in r1.content
    first_pack = json.loads(store.content(r1.asset_ref))
    assert first_pack["小满"]["kind"] == "角色"

    r2 = await fns._fn_drama_render_assets(lib_id)  # 补全
    assert r2.ok, r2.error
    gen_names = [a["summary"] for a in fake.images()[2:]]
    expected = [
        "服装·小满-旧连帽衫-[前10集]",
        "服装·陆离-深色风衣-[8/1]",
        "场景·地铁车厢",
        "道具·暗色骨簪",
    ]
    assert sorted(gen_names) == sorted(expected), "主形象不该再生成一遍"
    pack2 = json.loads(store.content(r2.asset_ref))
    assert pack2["小满"] == first_pack["小满"], "复用的主形象是同一张脸"
    assert pack2["小满-旧连帽衫-[前10集]"]["kind"] == "服装"
    assert not [k for k in pack2 if "三视图" in k]
    # 服装的参考图 = 复用的主形象（脸）
    costume_call = next(
        a for a in fake.images() if a["summary"] == "服装·小满-旧连帽衫-[前10集]"
    )
    assert costume_call["image"] == [first_pack["小满"]["url"]]
    assert "复用 2" in r2.content
    assert "覆盖：角色 2/2 · 服装 2/2 · 场景 1/1 · 道具 1/1" in r2.content
    assert store.get(r2.asset_ref).parent_ids == [lib_id, r1.asset_ref]

    r3 = await fns._fn_drama_render_assets(lib_id, reuse=False)
    assert r3.ok and len(fake.images()) == 6 + 6, "reuse=false 全部重生成"


async def test_补跑后渲视频服装精确命中():
    store = AssetStore()
    lib_id, shots_id = _seed(store)
    fake = FakeRegistry(store)
    fns = _fns(store, fake)
    await fns._fn_drama_render_assets(lib_id, only="characters")
    r = await fns._fn_drama_render_assets(lib_id)
    rs = await fns._fn_drama_render_shots(shots_id, rendered_id=r.asset_ref)
    assert rs.ok, rs.error
    # 服装 + 这个角色的主形象（脸）+ 场景/道具：服装全身图里脸太小，脸要单独给（2026-09-23）
    assert "参考 3 图" in rs.content and "退回" not in rs.content and "缺" not in rs.content
    assert "参考图匹配：4 个精确" in rs.content
    pack = json.loads(store.content(r.asset_ref))
    first = fake.videos()[0]
    assert first["image"][:2] == [pack["小满-旧连帽衫-[前10集]"]["url"], pack["小满"]["url"]]
    assert "角色「小满」本人" in first["prompt"], "脸那张要在提示词里点名"

"""2026-09-23 审查的短剧分环节问题（资产库与参考图 / 视频提示词 / 渲染 / 逐集写作）。

  · render_assets(only=…) 出的新包只有这次那几类，盖掉完整包 → 渲视频时人物全没了
  · 换了主形象（重渲 / 用户给图 / 面容审查），按旧脸生成的服装图照样复用 —— 新脸到不了视频
  · 渲视频只传服装全身图，脸只占一小块
  · 主形象被内容护栏拒掉，服装全部连带跳过，没有人知道该怎么办
  · 真实感写法不分年龄（attractive / 毛孔 / 雀斑用在 8 岁女主身上）
  · 全角「（场景名）」不算引用；「(OS)」反倒被当成引用，整批拦下；「朱锦娘」被算成「朱锦」
  · 换了视频模型，单段时长被静默截短
  · 音色锚点开渲时读、结束时整表覆盖，并行两集时丢更新
  · allow_no_refs=true 一键绕过引用门
  · 逐集写作的子代理看不到打回理由
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from aigc_agent.capabilities.subagents import SubAgentResult
from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.drama.format import EpisodeFormat, check_shots
from aigc_agent.domain.drama.models import ShotPrompt, normalize_ref_parens
from aigc_agent.domain.drama.voice import Anchor, dump_anchors
from aigc_agent.domain.functions.drama import (
    REFUSED_STAGE,
    DramaFunctions,
    _mentioned_characters,
    preflight_refs,
)
from aigc_agent.domain.functions.episodes import EpisodeFunctions
from aigc_agent.domain.realism import (
    apply_conflicts,
    is_minor,
    person_image_prompt,
    person_video_prompt,
)
from aigc_agent.harness.tools.provider import ToolResult

LIB = {
    "characters": [
        {
            "baseRoleName": "小满",
            "roleTotalDesc": "女 | 30岁 | 记者",
            "roleCostumeList": [{"costumeName": "小满-旧连帽衫-[1-3]", "costumeDesc": "旧卫衣"}],
        },
        {
            "baseRoleName": "陆离",
            "roleTotalDesc": "男 | 28岁 | 守门人",
            "roleCostumeList": [{"costumeName": "陆离-深色风衣-[1]", "costumeDesc": "风衣"}],
        },
    ],
    "scenes": [{"name": "地铁车厢", "description": "末班车"}],
    "props": [{"name": "暗色骨簪", "description": "骨簪"}],
}


class Registry:
    """假注册表：gen_image / gen_video 落资产；refuse 里的摘要返回内容护栏拒绝。"""

    def __init__(self, store: AssetStore, refuse: tuple[str, ...] = ()) -> None:
        self.store = store
        self.refuse = refuse
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def invoke(self, name: str, args: dict[str, Any]) -> ToolResult:
        self.calls.append((name, args))
        if any(r in str(args.get("summary", "")) for r in self.refuse):
            return ToolResult(ok=False, error="生图失败：触发内容安全策略（minor protection）")
        a = self.store.create("", summary=args.get("summary", name), creator="fake")
        a.uri = f"https://fake/{a.id}"
        self.store.put(a)
        return ToolResult(ok=True, content="ok", asset_ref=a.id)

    def images(self, summary: str = "") -> list[dict[str, Any]]:
        return [a for n, a in self.calls if n == "gen_image" and summary in a["summary"]]


def _lib(store: AssetStore, lib: dict | None = None) -> str:
    return store.create(json.dumps(lib or LIB, ensure_ascii=False), summary="资产库",
                        creator="tool:drama_assets", type_=AssetType.STORYBOARD).id


def _pack(store: AssetStore, pack_id: str) -> dict[str, Any]:
    return json.loads(store.content(pack_id))


# ---------------------------------------------------------------- 参考图包


async def test_only某类出的新包不丢其余几类():
    store = AssetStore()
    lib_id = _lib(store)
    reg = Registry(store)
    fns = DramaFunctions(None, store, registry=reg)
    full = await fns._fn_drama_render_assets(lib_id)
    assert full.ok
    r = await fns._fn_drama_render_assets(lib_id, only="scenes", reuse=False)
    assert r.ok, r.error
    pack = _pack(store, r.asset_ref)
    assert {"小满", "陆离", "小满-旧连帽衫-[1-3]", "地铁车厢", "暗色骨簪"} <= set(pack)
    assert pack["地铁车厢"]["asset"] != _pack(store, full.asset_ref)["地铁车厢"]["asset"]
    assert "沿用上一个包" in r.content


async def test_主形象换了_旧服装图作废重生成():
    store = AssetStore()
    lib_id = _lib(store)
    reg = Registry(store)
    fns = DramaFunctions(None, store, registry=reg)
    first = await fns._fn_drama_render_assets(lib_id)
    p1 = _pack(store, first.asset_ref)
    assert p1["小满-旧连帽衫-[1-3]"]["portrait"] == p1["小满"]["asset"], "服装记着按哪张脸生成"

    # 只重渲主形象（不复用）→ 新脸；服装包里仍是旧脸那张
    await fns._fn_drama_render_assets(lib_id, only="characters", reuse=False)
    n_before = len(reg.images("服装·小满"))
    r = await fns._fn_drama_render_assets(lib_id, only="costumes")
    p3 = _pack(store, r.asset_ref)
    assert len(reg.images("服装·小满")) == n_before + 1, "按新脸重生成服装"
    assert p3["小满-旧连帽衫-[1-3]"]["portrait"] == p3["小满"]["asset"]
    assert p3["小满"]["asset"] != p1["小满"]["asset"]


async def test_用户给了新脸_旧服装图从包里拿掉(tmp_path):
    store = AssetStore()
    lib_id = _lib(store)
    fns = DramaFunctions(None, store, registry=Registry(store))
    await fns._fn_drama_render_assets(lib_id)
    face = store.create("", type_=AssetType.IMAGE, summary="用户给的小满")
    face.uri = "https://host/xiaoman.png"
    store.put(face)

    class Hosting:
        enabled = True

        async def ensure_asset(self, store, a, ttl, force=False):
            return a.uri, ""

    fns.hosting = Hosting()
    r = await fns._fn_drama_use_local_ref("小满", asset_id=face.id, assets_id=lib_id)
    assert r.ok, r.error
    pack = _pack(store, r.asset_ref)
    assert pack["小满"]["asset"] == face.id
    assert "小满-旧连帽衫-[1-3]" not in pack and "陆离-深色风衣-[1]" in pack
    assert "按旧脸生成" in r.content and 'only="costumes"' in r.content


async def test_主形象被拒_停下来问人():
    store = AssetStore()
    lib_id = _lib(store)
    reg = Registry(store, refuse=("角色·小满",))
    fns = DramaFunctions(None, store, registry=reg)
    r = await fns._fn_drama_render_assets(lib_id)
    assert r.suspend and r.suspend_payload["stage"] == REFUSED_STAGE
    assert r.suspend_payload["major"] is True
    q = r.suspend_payload["question"]
    assert "小满" in q and "drama_use_local_ref" in q
    assert "陆离" in _pack(store, r.asset_ref), "别的角色照常进包"


# ---------------------------------------------------------------- 渲视频


def _shots(store: AssetStore, lib_id: str, shots: list[dict]) -> str:
    sb = store.create("[]", summary="分镜", creator="tool:drama_storyboard").id
    return store.create(json.dumps(shots, ensure_ascii=False), summary="提示词",
                        creator="tool:drama_shots", parents=[sb, lib_id],
                        type_=AssetType.STORYBOARD).id


async def test_全角括号引用和OS标注():
    store = AssetStore()
    lib_id = _lib(store)
    reg = Registry(store)
    fns = DramaFunctions(None, store, registry=reg)
    pack = await fns._fn_drama_render_assets(lib_id)
    shots_id = _shots(store, lib_id, [{
        "scene_index": "[第1集-1场]", "video_name": "1-4", "video_duration": "12s",
        "description": "场景设定: （地铁车厢） [夜]。(陆离-深色风衣-[1])(OS)说：“走。”",
    }])
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack.asset_ref)
    assert r.ok, r.error
    call = next(a for n, a in reg.calls if n == "gen_video")
    p = _pack(store, pack.asset_ref)
    assert p["地铁车厢"]["url"] in call["image"], "全角括号里的场景名也算引用"
    assert "(地铁车厢)" in call["prompt"]


def test_标注不当引用_长名字不算短名字():
    lib = SimpleNamespace(
        characters=[SimpleNamespace(name="朱锦"), SimpleNamespace(name="朱锦娘")],
        all_names=lambda: {"朱锦", "朱锦娘", "朱锦娘-红裙-[1]"},
        # 同真实资产库：服装 ID 认得出是哪个角色的
        character_of=lambda r: SimpleNamespace(name="朱锦娘") if r.startswith("朱锦娘") else None,
    )
    assert _mentioned_characters("朱锦娘推门进来", lib) == ["朱锦娘"]
    s = ShotPrompt(scene_index="[第1集-1场]", video_name="1", duration="10s",
                   description="(朱锦娘-红裙-[1])(OS)说：……")
    m = SimpleNamespace(url="u", person=True, key="朱锦娘-红裙-[1]")
    problems = preflight_refs([s], [0], {"朱锦娘-红裙-[1]": m}, lib)
    assert problems == [], problems


def test_全角括号归一只认资产名():
    known = {"地铁车厢", "陆离-深色风衣-[1]"}
    assert normalize_ref_parens("在（地铁车厢）里，陆离（低声）说", known) == \
        "在(地铁车厢)里，陆离（低声）说"


async def test_单段时长超过模型上限_花钱前拦():
    store = AssetStore()
    lib_id = _lib(store)
    reg = Registry(store)
    fns = DramaFunctions(None, store, registry=reg)
    pack = await fns._fn_drama_render_assets(lib_id)
    fns.catalog = SimpleNamespace(
        drama={}, max_concurrency=lambda k: 0,
        get=lambda kind, mid: SimpleNamespace(max_duration=10, max_refs=0),
    )
    shots_id = _shots(store, lib_id, [{
        "scene_index": "[第1集-1场]", "video_name": "1-5", "video_duration": "15s",
        "description": "(陆离-深色风衣-[1]) 走进 (地铁车厢)。",
    }])
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack.asset_ref)
    assert not r.ok and "单段最长 10s" in (r.error or "") and r.meta.get("charged") is False
    assert not [1 for n, _ in reg.calls if n == "gen_video"], "一段都没提交"


def test_音色锚点合并写_不覆盖人工pin():
    store = AssetStore()
    fns = DramaFunctions(None, store, registry=None)
    fns._save_anchors({"陆离": Anchor(character="陆离", asset="as_a", pinned=True)}, [])
    # 这一集开渲之后，并行的另一集写了苏晏的锚点
    fns._save_anchors({
        "陆离": Anchor(character="陆离", asset="as_a", pinned=True),
        "苏晏": Anchor(character="苏晏", asset="as_b"),
    }, [])
    # 这一集结束：新定了小满，也想把陆离换成自己这段
    fns._merge_save_anchors({
        "陆离": Anchor(character="陆离", asset="as_x"),
        "小满": Anchor(character="小满", asset="as_c"),
    }, [])
    latest = fns._load_anchors()
    assert set(latest) == {"陆离", "苏晏", "小满"}, "别的集新定的不能被冲掉"
    assert latest["陆离"].asset == "as_a", "人手动 pin 的不覆盖"
    assert dump_anchors(latest)


# ---------------------------------------------------------------- 真实感分年龄


def test_未成年角色用儿童安全写法():
    assert is_minor("女 | 外表 8 岁（实际元神 1400+ 年） | 萌宝")
    assert is_minor("16岁流浪少女")
    assert not is_minor("男 | 28岁 | 基因分析师")
    assert is_minor("萌宝，圆脸大眼")
    text, notes = person_image_prompt("女 | 8岁 | 圆脸大眼")
    # 成年人那套（attractive / RAW / 可见毛孔 / 补瑕疵锚点）一样都不能有
    for adult_only in ("attractive", "RAW", "可见毛孔", "面部真实质感", "硬光"):
        assert adult_only not in text, adult_only
    assert "儿童" in text and notes
    adult, _ = person_image_prompt("男 | 28岁 | 守门人")
    assert "attractive" in adult
    child_video = person_video_prompt("她蹲下", minor=True)
    assert "age-appropriate" in child_video and "pores" not in child_video


def test_皮肤规则按档位():
    tpl = '禁止项：严禁出现"皱纹 (wrinkles)"、"青筋 (veins)"、"血丝 (bloodshot)"、' \
          '"痣 (moles/freckles)"以及"深陷的眼窝 (deep-set eyes)"。'
    assert "必须**保留毛孔" not in apply_conflicts(tpl, "subtle")
    assert "零星几点极淡" in apply_conflicts(tpl, "subtle")
    assert "少量雀斑" in apply_conflicts(tpl, "natural")


def test_切镜数对不上cuts():
    fmt = EpisodeFormat()
    s = ShotPrompt(scene_index="[第1集-1场]", video_name="1-4", duration="12s",
                   description="[中景]A。[切镜：近景]B。", cuts=[3, 3, 3, 3])
    assert any("[切镜] 数和 cuts 对不上" in p for p in check_shots([s], fmt))
    ok = ShotPrompt(scene_index="[第1集-1场]", video_name="1-4", duration="12s",
                    description="[中景]A。[切镜：近景]B。[切镜：特写]C。[切镜：中景]D。",
                    cuts=[3, 3, 3, 3])
    assert not any("[切镜]" in p for p in check_shots([ok], fmt))


# ---------------------------------------------------------------- 逐集写作


class _Runner:
    def __init__(self) -> None:
        self.tasks: list[str] = []

    async def run(self, defn, task: str = "", **kw) -> SubAgentResult:
        self.tasks.append(task)
        text = "# 第1集：开场\n\n" + "正文正文正文。" * 200 + "\n\n> 🎣 本集钩子：门开了"
        return SubAgentResult(ok=True, text=text, iterations=1, cost=0.01,
                              stop_reason="no_tool_calls")


async def test_逐集写作带上打回理由():
    store = AssetStore()
    store.create("第1集：开场 —— 门开了", type_=AssetType.OUTLINE, summary="分集目录",
                 creator="model")
    runner = _Runner()
    fns = EpisodeFunctions(runner, store)  # type: ignore[arg-type]
    fns.brief_source = lambda topic: "must_not：开头不要喊口号（打回理由）"
    r = await fns.invoke("drama_write_episode", {"episode": 1})
    assert r.ok, r.error
    assert "开头不要喊口号" in runner.tasks[0] and "用户定过的要求" in runner.tasks[0]

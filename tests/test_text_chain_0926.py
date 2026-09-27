"""短剧文本链的纯 bug（2026-09-26 审查清单：J6、J8–J13、I1、I2）。

- J6  面容审查只看本项目的图：没分项目的旧剧同名角色图不算「另一张脸」
- J8  挑分镜修订版：问题少了但丢了台词的修订版不许胜出
- J9  模型把集号标错（第 5 集的剧本标成 episodeIndex 1）：按剧本的「第N集」/ 剧本资产的集号纠正
- J10 drama_write 按「第N集」拆开、一集一份资产（摘要「第N集·剧本」、记 episode）
- J11 裸角色名：先挑这一集能穿的服装；一套能穿的都没有才报缺口（之前默认第一套、一声不吭）
- J12 项目卡核对分镜的台词：用它当初拆的那份剧本，不拿改过的最新剧本比
- J13 项目卡：没标集号的视频提示词 / 片段索引不算「一集」
- I1  想法写成剧本后要给用户看、他确认了再拆（工具说明、入口判断、系统提示词三处一致）
- I2  内容过滤后用克制措辞重写的分镜要核对：丢了台词 / 少了场次就不用这一版，并列出改动
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.drama import bind_costumes, parse_assets
from aigc_agent.domain.drama.card import build_project_card
from aigc_agent.domain.drama.format import HOOK_MARK, EpisodeFormat
from aigc_agent.domain.drama.models import Episode
from aigc_agent.domain.drama.parse import parse_shots
from aigc_agent.domain.functions.drama import DramaFunctions
from aigc_agent.domain.system_prompt import SYSTEM_PROMPT
from tests.test_episode_format import LIB as ONE_CHAR_LIB

# ---------------------------------------------------------------- 公共素材

LINES = ["第一句台词不能丢掉", "第二句台词也要保留", "第三句台词同样重要"]
# 一场、三句对白、够长（拆解类工具不收几个字的空壳）
SCRIPT = (
    "[夜] [内] [地铁车厢]\n"
    + "\n".join(f"陆离：{t}" for t in LINES)
    + "\n"
    + "△ 列车进站，灯光一明一暗，陆离攥紧了手里的车票，指节发白。\n" * 8
)
# 两场、四句对白
TWO_SCENES = SCRIPT + "[日] [外] [站台]\n陆离：第四句在站台上说\n"


def _board(keep: list[str], filler: int, hook: bool = True, secs: float = 2.5) -> str:
    """分镜正文：一场，keep 里的台词各占一个镜头（带引号），再补 filler 个动作镜头。"""
    rows = ["[夜] [内] [地铁车厢]"]
    for i, t in enumerate(keep):
        mark = HOOK_MARK if hook and i == 0 else ""
        rows.append(f"{mark}[近景/手持/{secs:g}s] 陆离开口：“{t}”")
    rows += [f"[中景/固定/{secs:g}s] 陆离第{i}个动作" for i in range(filler)]
    return "\n".join(rows)


def _reply(desc: str, index: int = 1, title: str = "第1集") -> str:
    return json.dumps(
        [{"episodeIndex": index, "episodeTitle": title, "episodeDesc": desc}], ensure_ascii=False
    )


# 6 镜、三句都在、没有高潮点：两条规格问题（总时长太短 + 开场没有高潮点），没丢台词
V1 = _reply(_board(LINES, filler=3, hook=False))
# 96 镜 × 2.5s、有高潮点、规格全过 —— 但丢了第三句
V2_LOST = _reply(_board(LINES[:2], filler=94))
# 96 镜、规格全过、三句都在
V2_FULL = _reply(_board(LINES, filler=93))


class _Seq:
    """按顺序回：(正文, finish_reason)。回完了一直回最后一份。"""

    def __init__(self, *replies: tuple[str, str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(self, role: str, messages: list[dict[str, Any]], **_: Any) -> Any:
        self.calls.append(messages)
        text, fr = self.replies[min(len(self.calls), len(self.replies)) - 1]
        return SimpleNamespace(text=text, finish_reason=fr)


def _fns(gw: Any, store: AssetStore | None = None) -> DramaFunctions:
    return DramaFunctions(gw, AssetStore() if store is None else store, registry=None,
                          catalog=None, fmt=EpisodeFormat())


def _saved_board(store: AssetStore, asset_id: str) -> list[dict[str, Any]]:
    return json.loads(store.content(asset_id))


# ---------------------------------------------------------------- J6 面容审查只看本项目


def test_面容审查只看本项目的图_没分项目的旧剧同名图不算():
    store = AssetStore()
    store.project = ""
    legacy = store.create("", type_=AssetType.IMAGE, summary="角色·陆离·旧剧", creator="model:x")
    store.project = "不渡"
    mine = store.create("", type_=AssetType.IMAGE, summary="角色·陆离", creator="model:x")
    assert legacy.project == "" and mine.project == "不渡"
    assert {a.id for a in store.find(type_=AssetType.IMAGE)} == {legacy.id, mine.id}, (
        "没分项目的旧资产按项目查也看得见 —— 所以面容审查要自己再筛一道"
    )
    lib, err = parse_assets(json.dumps(ONE_CHAR_LIB, ensure_ascii=False))
    assert not err
    cands = DramaFunctions(None, store)._collect_faces(lib, {}, deep=False)
    assert [c.asset_id for c in cands] == [mine.id], "旧剧的同名角色图不当成这部剧的另一张脸"

    store.project = ""  # 没有项目键的老会话：照旧全看
    ids = {c.asset_id for c in DramaFunctions(None, store)._collect_faces(lib, {}, deep=False)}
    assert ids == {legacy.id, mine.id}


# ---------------------------------------------------------------- J8 修订版不许丢台词


async def test_修订版问题少了但丢了台词_不采用():
    gw = _Seq((V1, "stop"), (V2_LOST, "stop"))
    store = AssetStore()
    r = await _fns(gw, store)._fn_drama_storyboard(SCRIPT, ethnicity="asian", language="zh")
    assert r.ok, r.error
    assert len(gw.calls) == 2, "第一版有规格问题：让模型改了一次"
    board = _saved_board(store, r.asset_ref)
    assert board[0]["episodeDesc"] == json.loads(V1)[0]["episodeDesc"], (
        "修订版规格全过、问题从 2 条变 0 条，但丢了「第三句」—— 留第一版"
    )
    assert store.get(r.asset_ref).gen_params["format_problems"] == 2
    assert "6 镜" in r.content


async def test_修订版问题少了也没丢台词_采用():
    gw = _Seq((V1, "stop"), (V2_FULL, "stop"))
    store = AssetStore()
    r = await _fns(gw, store)._fn_drama_storyboard(SCRIPT, ethnicity="asian", language="zh")
    assert r.ok, r.error
    assert _saved_board(store, r.asset_ref)[0]["episodeDesc"] == json.loads(V2_FULL)[0][
        "episodeDesc"
    ]
    assert store.get(r.asset_ref).gen_params["format_problems"] == 0
    assert "96 镜" in r.content


# ---------------------------------------------------------------- J9 集号纠正


def test_集号标错_按剧本的第N集标题纠正_标题跟着改():
    script = "第3集 夜路\n" + SCRIPT + "第4集 天亮\n" + SCRIPT
    eps = [Episode(1, "第1集 夜路", "x"), Episode(2, "第2集", "y")]
    fixed = _fns(None)._fix_episode_index(eps, script)
    assert [(e.index, e.title) for e in fixed] == [(3, "第3集 夜路"), (4, "第4集")]

    right = [Episode(3, "第3集", "x"), Episode(4, "第4集", "y")]
    assert _fns(None)._fix_episode_index(right, script) == right, "标对了原样不动"
    one = [Episode(1, "第1集", "x")]
    assert _fns(None)._fix_episode_index(one, script) == one, "集数对不上（2 集对 1 集）不猜"


def test_剧本没有集标题_按剧本资产的集号纠正():
    store = AssetStore()
    src = store.create(SCRIPT, type_=AssetType.SCRIPT, summary="第7集·剧本",
                       gen_params={"episode": 7})
    fixed = _fns(None, store)._fix_episode_index([Episode(1, "", "x")], SCRIPT, src.id)
    assert [(e.index, e.title) for e in fixed] == [(7, "第7集")]


async def test_拆分镜_模型把第5集标成第1集_存下来的是第5集():
    store = AssetStore()
    src = store.create(SCRIPT, type_=AssetType.SCRIPT, summary="第5集·剧本",
                       gen_params={"episode": 5})
    gw = _Seq((_reply(_board(LINES, filler=93), index=1, title="第1集"), "stop"))
    r = await _fns(gw, store)._fn_drama_storyboard(
        ethnicity="asian", language="zh", script_id=src.id
    )
    assert r.ok, r.error
    board = _saved_board(store, r.asset_ref)
    assert (board[0]["episodeIndex"], board[0]["episodeTitle"]) == (5, "第5集")
    assert "第5集" in r.content and len(gw.calls) == 1


# ---------------------------------------------------------------- J10 按集存剧本


TWO_EPISODES = (
    "第1集 雨夜\n[夜] [内] [地铁车厢]\n陆离：这趟车不该停在这里。\n"
    + "△ 车厢里的灯忽明忽暗，陆离盯着车窗上自己的倒影。\n" * 3
    + "第2集 站台\n[日] [外] [站台]\n小满：你终于来了，我等了很久。\n"
    + "△ 站台上空无一人，广播里一遍遍报着站名。\n" * 3
)


async def test_写剧本_按第N集拆开_一集一份资产():
    store = AssetStore()
    gw = _Seq((TWO_EPISODES, "stop"))
    r = await _fns(gw, store)._fn_drama_write("雨夜地铁里的守门人", episodes=2)
    assert r.ok, r.error
    scripts = sorted(store.find(type_=AssetType.SCRIPT), key=lambda a: a.seq)
    assert [a.summary for a in scripts] == ["第1集·剧本", "第2集·剧本"]
    assert [a.gen_params["episode"] for a in scripts] == [1, 2]
    assert all(a.creator == "tool:drama_write" for a in scripts)
    assert all(a.gen_params["idea"] == "雨夜地铁里的守门人" for a in scripts)
    one, two = (store.content(a.id) for a in scripts)
    assert one.startswith("第1集 雨夜") and "小满" not in one
    assert two.startswith("第2集 站台") and "陆离：这趟车" not in two
    assert r.asset_ref == scripts[0].id
    assert f"按集存好：第1集 {scripts[0].id}、第2集 {scripts[1].id}" in r.content
    assert store.episodes_done() == [1, 2], "项目卡和按集流水认得出集号"


async def test_写剧本_没有集标题_整份存一份():
    store = AssetStore()
    gw = _Seq((SCRIPT, "stop"))
    r = await _fns(gw, store)._fn_drama_write("雨夜地铁里的守门人")
    assert r.ok, r.error
    only = store.find(type_=AssetType.SCRIPT)
    assert len(only) == 1 and only[0].summary == "剧本·1集"
    assert "episode" not in only[0].gen_params
    assert f"资产 {only[0].id}" in r.content


# ---------------------------------------------------------------- J11 裸角色名挑能穿的服装


def _one_role(costumes: list[dict[str, Any]]) -> Any:
    lib, err = parse_assets(json.dumps({
        "characters": [{"baseRoleName": "唐僧", "roleTotalDesc": "男 | 30岁",
                        "roleCostumeList": costumes}],
        "scenes": [{"name": "禅房", "description": "x"}],
        "props": [],
    }, ensure_ascii=False))
    assert not err, err
    return lib


def _bare_shot() -> list[Any]:
    rows = [{"scene_index": "[第3集-1场]", "video_name": "1-3", "video_duration": "12s",
             "description": "场景设定: (禅房)。(唐僧) 合十。"}]
    shots, err = parse_shots(json.dumps(rows, ensure_ascii=False))
    assert not err
    return shots


def test_裸角色名_先挑这一集能穿的服装_不默认第一套():
    lib = _one_role([
        {"costumeName": "唐僧-锦襕袈裟-[11-20]", "costumeDesc": "锦襕袈裟", "episodes": "11-20"},
        {"costumeName": "唐僧-粗布僧衣", "costumeDesc": "粗布僧衣"},  # 没标集数：哪集都能穿
    ])
    shots = _bare_shot()
    changes, warns, _ = bind_costumes(shots, lib)
    assert "(唐僧-粗布僧衣)" in shots[0].description, "第一套是第 11–20 集的戏服，第 3 集不穿"
    assert changes == ["[第3集-1场]：唐僧 → 唐僧-粗布僧衣"]
    assert warns == [], "有能穿的就不报"


def test_裸角色名_这一集一套能穿的都没有_报缺口():
    lib = _one_role([
        {"costumeName": "唐僧-锦襕袈裟-[11-20]", "costumeDesc": "锦襕袈裟", "episodes": "11-20"},
        {"costumeName": "唐僧-破袈裟-[15-20]", "costumeDesc": "破袈裟", "episodes": "15-20"},
    ])
    shots = _bare_shot()
    _, warns, _ = bind_costumes(shots, lib)
    assert "(唐僧-锦襕袈裟-[11-20])" in shots[0].description, "至少换成一套服装 ID，才对得上图"
    assert warns == [
        "唐僧 在「禅房」（[第3集-1场]）这一集没有能穿的服装，"
        "先用 唐僧-锦襕袈裟-[11-20]（它标的是第 11-20 集）"
    ]


# ---------------------------------------------------------------- J12 / J13 项目卡


_PAD = "△ 陆离看着窗外的雨，雨水顺着玻璃往下淌，他很久都没有说话。\n" * 12
OLD_LINES = [f"第{w}句旧剧本里就有的话" for w in "一二三四"]
NEW_LINES = [f"第{w}句改稿以后才加的话" for w in "五六七"]


def _script(store: AssetStore, lines: list[str]) -> Any:
    body = "[夜] [内] [地铁车厢]\n" + "\n".join(f"陆离：{t}" for t in lines) + "\n" + _PAD
    assert len(body) >= 300
    return store.create(body, type_=AssetType.SCRIPT, summary="第1集·剧本",
                        gen_params={"episode": 1}, creator="tool:drama_write")


def _storyboard(store: AssetStore, lines: list[str], parents: list[str]) -> Any:
    desc = "\n".join(
        ["[夜] [内] [地铁车厢]"]
        + [f"[近景/手持/3s] 陆离开口：“{t}”" for t in lines]
        + [f"[中景/固定/3s] 陆离第{i}个动作" for i in range(26)]
    )
    return store.create(_reply(desc), type_=AssetType.STORYBOARD, summary="分镜脚本·1集",
                        creator="tool:drama_storyboard", parents=parents)


def test_项目卡核对分镜台词_用它当初拆的那份剧本():
    store = AssetStore()
    v1 = _script(store, OLD_LINES)
    _storyboard(store, OLD_LINES, parents=[v1.id])
    _script(store, OLD_LINES + NEW_LINES)  # 剧本改过一版，多了三句
    card = build_project_card(store)
    assert card and "分镜比剧本少了台词" not in card, (
        "分镜是按第一版拆的、一句没丢：不能拿改过的剧本判它「少了台词」"
    )

    _storyboard(store, OLD_LINES, parents=[])  # 没记从哪份剧本拆的：退回这一集最新的剧本
    assert "第1集少 3 句" in build_project_card(store)


def test_项目卡_没标集号的提示词和片段索引不算一集():
    store = AssetStore()
    _script(store, OLD_LINES)
    for ep in (1, None):
        gp = {"episode": ep} if ep else {}
        store.create("[]", summary="提示词", creator="tool:drama_shots", gen_params=gp)
        store.create("[]", summary="片段", creator="tool:drama_render_shots",
                     gen_params={**gp, "complete": True})
    store.create("[]", summary="片段·第0集", creator="tool:drama_render_shots",
                 gen_params={"episode": 0})
    card = build_project_card(store)
    assert "视频提示词 1 集" in card and "视频片段 1 集" in card, card


# ---------------------------------------------------------------- I1 写完剧本先给人看


async def test_想法写成剧本之后_要给用户看_确认了再拆():
    fns = _fns(None)
    schema = json.dumps(await fns.get_schema("drama_write"), ensure_ascii=False)
    assert "把剧本给用户看，他确认了再进拆解链" in schema
    assert "minutes" not in schema, "集长只由项目规格定，模型不能自己传"
    r = await fns._fn_drama_intake("一个外卖小哥被富豪家族认回的逆袭故事")
    assert "一句话想法" in r.content
    assert "他确认了再进拆解流程" in r.content and "别拿没确认过的剧本往下拆" in r.content
    assert "给用户看过、他确认了，再往下拆解" in SYSTEM_PROMPT
    for text in (schema, r.content, SYSTEM_PROMPT):
        assert "不再单独等确认" not in text


# ---------------------------------------------------------------- I2 克制措辞重写要核对


async def test_被过滤后重写的分镜丢了台词少了场次_不采用_列出改动():
    store = AssetStore()
    gw = _Seq(("", "content_filter"), (V2_FULL, "stop"))
    r = await _fns(gw, store)._fn_drama_storyboard(TWO_SCENES, ethnicity="asian", language="zh")
    assert not r.ok
    assert len(gw.calls) == 2 and "【措辞】" in gw.calls[1][-1]["content"]
    assert "没有采用" in r.error and "不要自己删台词绕过去" in r.error
    assert "少了 1 句台词" in r.error and "第四句在站台上说" in r.error
    assert "场次从 2 场变成 1 场" in r.error
    assert store.find(creator="tool:drama_storyboard") == [], "改了剧情的那版不落库"


async def test_被过滤后重写的分镜_台词场次都在_照常采用():
    store = AssetStore()
    gw = _Seq(("", "content_filter"), (V2_FULL, "stop"))
    r = await _fns(gw, store)._fn_drama_storyboard(SCRIPT, ethnicity="asian", language="zh")
    assert r.ok, r.error
    assert "内容安全过滤拦了一次" in r.content
    assert len(store.find(creator="tool:drama_storyboard")) == 1

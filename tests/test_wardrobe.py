"""服装按场景分配与绑定（2026-09-18）。

用户发现：不同场景下角色的服装会乱变。根因是资产库一个角色只有一套服装，
分镜里没有可引用的换装 ID，视频模型就按文字随机发挥。现在：
  ② 资产库：每套服装带 scenes / episodes，一个角色×场景组合必须有服装
  ③ 提示词：按镜头所在场景**确定性**绑定服装 ID，不靠模型自觉
  ④ 参考图：主形象 → 各场景服装（参考主形象保脸；三视图版式在服装图内部）
"""

from __future__ import annotations

import asyncio
import json
import time

from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.drama import (
    assets_system,
    bind_costumes,
    normalize,
    parse_assets,
    shots_system,
    unbound_costumes,
)
from aigc_agent.domain.drama.models import episode_ranges
from aigc_agent.domain.drama.parse import parse_shots
from aigc_agent.domain.functions.drama import DramaFunctions
from aigc_agent.domain.media.naming import library_reference_names
from aigc_agent.harness.model.gateway import ModelResponse, Usage
from aigc_agent.harness.tools.provider import ToolResult

LIB = {
    "characters": [
        {
            "baseRoleName": "陆离",
            "roleTotalDesc": "男 | 35岁 | 守门人 [强制白底证件照]",
            "roleCostumeList": [
                {
                    "costumeName": "陆离-深色风衣-[1-3]",
                    "costumeDesc": "深色风衣",
                    "scenes": ["地铁车厢", "修表铺"],
                    "episodes": "1-3",
                },
                {
                    "costumeName": "陆离-白大褂-[2-4]",
                    "costumeDesc": "白大褂",
                    "scenes": "疾控中心实验室",  # 模型偶尔给字符串
                    "episodes": [2, 3, 4],  # 或数组
                },
            ],
        },
        {
            "baseRoleName": "小满",
            "roleTotalDesc": "女 | 12岁 [强制白底证件照]",
            "roleCostumeList": [
                {"costumeName": "小满-旧连帽衫-[全集]", "costumeDesc": "卫衣", "episodes": "全集"}
            ],
        },
        {
            "baseRoleName": "老鬼",
            "roleTotalDesc": "男 | 60岁",
            "roleCostumeList": [{"costumeName": "老鬼-劳保服-[1]", "costumeDesc": "劳保服"}],
        },
    ],
    "scenes": [
        {"name": "地铁车厢", "description": "x"},
        {"name": "修表铺", "description": "y"},
        {"name": "疾控中心实验室", "description": "z"},
    ],
    "props": [{"name": "骨簪", "description": "p"}],
}


def _lib():
    lib, err = parse_assets(json.dumps(LIB, ensure_ascii=False))
    assert not err, err
    return lib


# ---------------------------------------------------------------- 数据模型


def test_解析服装的场景与集数_容忍字符串和数组():
    lib = _lib()
    luli = lib.characters[0]
    assert luli.costumes[0].scenes == ["地铁车厢", "修表铺"] and luli.costumes[0].episodes == "1-3"
    assert luli.costumes[1].scenes == ["疾控中心实验室"] and luli.costumes[1].episodes == "2,3,4"
    assert lib.characters[1].costumes[0].scenes == []


def test_集数范围解析():
    assert episode_ranges("1-3") == [(1, 3)]
    assert episode_ranges("1,5-8") == [(5, 8), (1, 1)]
    assert episode_ranges("前10集") == [(1, 10)]
    assert episode_ranges("全集") == [(1, 9999)]
    assert episode_ranges("[8/1]") == [(1, 8)]
    assert episode_ranges("") == []


def test_按场景选服装_场景优先于集数():
    lib = _lib()
    luli, xiaoman, laogui = lib.characters
    assert luli.costume_for("疾控中心实验室", 1).name == "陆离-白大褂-[2-4]", "场景绑定压过集数"
    assert luli.costume_for("修表铺", 4).name == "陆离-深色风衣-[1-3]"
    assert luli.costume_for("没登记的场景", 2).name == "陆离-深色风衣-[1-3]", "没场景就按集数"
    assert luli.costume_for("没登记的场景", 4).name == "陆离-白大褂-[2-4]"
    assert luli.costume_for("没登记的场景", 9) is None
    assert xiaoman.costume_for("任何场景", 37).name == "小满-旧连帽衫-[全集]"
    assert laogui.costume_for("地铁车厢", 1) is None, "既没标场景也没标集数，不猜"
    assert unbound_costumes(lib) == ["老鬼-劳保服-[1]"]


def test_喂给第三步的摘要带场景与集数标注():
    d = _lib().digest()
    assert '"costumeName": "陆离-深色风衣-[1-3]"   // 用于 地铁车厢、修表铺；第1-3集' in d
    assert "深色风衣" not in d.replace("陆离-深色风衣-[1-3]", ""), "只给名字不给描述"


def test_分配表可读():
    m = _lib().wardrobe_matrix()
    assert "陆离：2 套服装" in m and "陆离-白大褂-[2-4] ← 疾控中心实验室 · 第2,3,4集" in m
    assert "老鬼-劳保服-[1] ← （未标场景）" in m


def test_提示词里有分配协议与选用规则():
    a = assets_system(normalize("chinese", "zh"))
    assert "服装按场景分配协议" in a and '"scenes": ["伦敦深夜巷弄"]' in a
    assert "roleCostumeList 不能为空" in a, "原模板的约束还在"
    s = shots_system(normalize("chinese", "zh"))
    assert "服装按场景选用" in s and "同一场景内所有镜头穿搭一致" in s


# ---------------------------------------------------------------- 绑定


def _shot(scene: str, name: str, desc: str) -> dict:
    return {"scene_index": scene, "video_name": name, "video_duration": "12s", "description": desc}


SHOTS = [
    _shot(
        "[第2集-1场]", "1-3",
        "场景设定: (疾控中心实验室) [日] [内]。(陆离-深色风衣-[1-3]) 坐下。(小满) 站着。",
    ),
    _shot("[第2集-2场]", "4-6", "场景设定: (修表铺) [夜] [内]。(陆离-白大褂-[2-4]) 推门进来。"),
    _shot("[第5集-1场]", "1-2", "(骨簪) 在桌上。(老鬼) 蹲下。"),
    _shot("[第5集-2场]", "3-4", "场景设定: (地铁车厢)。(老鬼-劳保服-[1]) 走过来。"),
]


def test_按场景绑定服装_裸名换成服装_缺口报出来():
    lib = _lib()
    shots, err = parse_shots(json.dumps(SHOTS, ensure_ascii=False))
    assert not err
    changes, warns = bind_costumes(shots, lib)

    assert "(陆离-白大褂-[2-4])" in shots[0].description, "实验室里该穿白大褂"
    assert "(陆离-深色风衣-[1-3])" not in shots[0].description
    assert "(小满-旧连帽衫-[全集])" in shots[0].description, "裸角色名换成服装 ID"
    assert "(陆离-深色风衣-[1-3])" in shots[1].description, "修表铺换回风衣"
    assert "(老鬼-劳保服-[1])" in shots[2].description, "没绑定但只有一套：裸名至少换成它"
    assert shots[3].description.count("(老鬼-劳保服-[1])") == 1, "没绑定的引用原样保留"
    assert changes == [
        "[第2集-1场]：陆离-深色风衣-[1-3] → 陆离-白大褂-[2-4]",
        "[第2集-1场]：小满 → 小满-旧连帽衫-[全集]",
        "[第2集-2场]：陆离-白大褂-[2-4] → 陆离-深色风衣-[1-3]",
        "[第5集-1场]：老鬼 → 老鬼-劳保服-[1]",
    ]
    assert len(warns) == 1 and "老鬼 在「地铁车厢」" in warns[0]

    again, _ = bind_costumes(shots, lib)
    assert again == [], "绑定是幂等的"


# ---------------------------------------------------------------- 第③步集成


class _Gw:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[list[dict]] = []

    async def chat(self, role, messages, tools=None, **kw):
        self.calls.append(messages)
        return ModelResponse(text=self.text, usage=Usage(1, 1))


async def test_drama_shots产物已按场景绑定():
    store = AssetStore()
    lib_id = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库").id
    sb = [
        {
            "episodeIndex": 2,
            "episodeTitle": "第2集",
            "episodeDesc": "[日] [内] [疾控中心实验室]\n[MCU] 陆离 坐下。",
        }
    ]
    sb_id = store.create(json.dumps(sb, ensure_ascii=False), summary="分镜").id
    gw = _Gw(json.dumps(SHOTS[:2], ensure_ascii=False))
    fns = DramaFunctions(gw, store)

    r = await fns._fn_drama_shots(sb_id, lib_id)
    assert r.ok, r.error
    # 三处：实验室里风衣→白大褂、裸名「小满」→服装、修表铺白大褂→风衣
    assert "服装已按场景绑定，改了 3 处引用" in r.content
    assert "陆离-深色风衣-[1-3] → 陆离-白大褂-[2-4]" in r.content
    stored, _ = parse_shots(store.content(r.asset_ref))
    assert "(陆离-白大褂-[2-4])" in stored[0].description
    assert "(小满-旧连帽衫-[全集])" in stored[0].description
    assert store.get(r.asset_ref).gen_params["costume_bindings"] == 3
    # 资产库摘要（带场景标注）进了模型的输入
    user_msg = gw.calls[0][-1]["content"]
    assert "// 用于 疾控中心实验室" in user_msg


async def test_drama_assets输出分配表与缺口(monkeypatch):
    store = AssetStore()
    gw = _Gw(json.dumps(LIB, ensure_ascii=False))
    fns = DramaFunctions(gw, store)
    r = await fns._fn_drama_assets("剧本…", ethnicity="chinese", language="zh")
    assert r.ok, r.error
    assert "陆离：2 套服装" in r.content and "← 疾控中心实验室" in r.content
    assert "1 套服装没标场景/集数（老鬼-劳保服-[1]）" in r.content
    assert "服装按场景分配协议" in gw.calls[0][0]["content"]


# ---------------------------------------------------------------- 第④步：参考图分层


class _Reg:
    def __init__(self, store: AssetStore) -> None:
        self.store = store
        self.calls: list[dict] = []
        self.timeline: dict[str, tuple[float, float]] = {}

    async def invoke(self, name, args):
        t0 = time.perf_counter()
        await asyncio.sleep(0.01)
        self.calls.append(args)
        a = self.store.create("", summary=args["summary"], creator="fake")
        a.uri = f"https://fake/{args['summary']}"
        self.store.put(a)
        self.timeline[args["summary"]] = (t0, time.perf_counter())
        return ToolResult(ok=True, content="ok", asset_ref=a.id)

    def by_summary(self, s: str) -> dict:
        return next(a for a in self.calls if a["summary"] == s)


async def test_服装参考主形象_不再出三视图():
    store = AssetStore()
    lib_id = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库").id
    reg = _Reg(store)
    r = await DramaFunctions(None, store, registry=reg)._fn_drama_render_assets(lib_id)
    assert r.ok, r.error

    assert not [a for a in reg.calls if "三视图" in a["summary"]], "三视图层已经去掉"
    coat = reg.by_summary("服装·陆离-白大褂-[2-4]")
    assert coat["image"] == ["https://fake/角色·陆离"], "服装只参考主形象保脸"
    # 服装图是纯白底单张全身正面照，不是三视图
    assert coat["aspect_ratio"] == "16:9"
    assert "单张全身站姿正面照" in coat["prompt"] and "三视图" not in coat["prompt"]
    assert coat["local_name"] == "参考图-服装-02_陆离-白大褂-[2-4]"
    # 分层：服装等所有第一层（角色/场景/道具）完成
    phase_a_end = max(
        end for k, (_, end) in reg.timeline.items() if not k.startswith("服装")
    )
    assert reg.timeline["服装·陆离-白大褂-[2-4]"][0] >= phase_a_end
    assert "✓ 服装 陆离-白大褂-[2-4] ← 疾控中心实验室" in r.content
    assert "三视图" not in r.content
    assert "覆盖：角色 3/3 · 服装 4/4 · 场景 3/3 · 道具 1/1" in r.content
    pack = json.loads(store.content(r.asset_ref))
    assert not [k for k in pack if "三视图" in k]


async def test_只渲服装会连带缺的主形象():
    store = AssetStore()
    lib_id = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库").id
    reg = _Reg(store)
    r = await DramaFunctions(None, store, registry=reg)._fn_drama_render_assets(
        lib_id, only="costumes"
    )
    assert r.ok
    assert sorted(a["summary"] for a in reg.calls) == sorted(
        ["角色·陆离", "角色·小满", "角色·老鬼",
         "服装·陆离-深色风衣-[1-3]", "服装·陆离-白大褂-[2-4]",
         "服装·小满-旧连帽衫-[全集]", "服装·老鬼-劳保服-[1]"]
    )


def test_参考图文件名不含三视图():
    names = library_reference_names(_lib())
    assert names["陆离"] == "参考图-角色-01_陆离"
    assert names["小满"] == "参考图-角色-02_小满"
    assert not [k for k in names if "三视图" in k]

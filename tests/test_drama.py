"""短剧三段式流水线验收。

盯的是这套流程**存在的理由**：跨镜头、跨集的视觉一致性。
它靠三件事撑着，任何一件塌了整条链就白跑：

  1. 资产描述的**锁定后缀**要被切出来（三视图/白底/8K 那段）。
     混在正文里让人编辑，改坏了出来就不是三视图；丢了则生图没有版式约束。
  2. 第③步只能引用资产库里**真实存在**的 ID。编一个出来看着没毛病，
     但生视频时找不到参考图 —— 人物就会变脸。
  3. 模型的 JSON 经常不合法，且是**结构性**的：规格要求台词带引号，
     而它又活在 JSON 字符串里。解析必须能修，不能整条流程挂掉。
"""

from __future__ import annotations

import pytest

from aigc_agent.domain.drama import (
    assets_system,
    audit_refs,
    normalize,
    parse_assets,
    parse_episodes,
    parse_shots,
    shots_system,
    split_locked,
    storyboard_system,
    strip_cites,
)

_KEEP = normalize("mixed", "keep")
ASSETS_SYSTEM = assets_system(_KEEP)
SHOTS_SYSTEM = shots_system(_KEEP)
STORYBOARD_SYSTEM = storyboard_system(_KEEP)

ASSETS_JSON = """{
  "characters": [
    {"baseRoleName": "Anne",
     "roleTotalDesc": "女 | 23岁 | 高加索裔 | 现代 | 浅绿色瞳孔 | 苗条 [强制白底证件照]",
     "roleCostumeList": [
       {"costumeName": "Anne-酒店员工制服-[全集]",
        "costumeDesc": "深蓝色聚酯纤维。[强制16:9 横版构图，三视图，超写实 8K，写实]"}]},
    {"baseRoleName": "Colin",
     "roleTotalDesc": "男 | 32岁 | 高加索裔 | 现代 | 深蓝色瞳孔 [强制白底证件照]",
     "roleCostumeList": [
       {"costumeName": "Colin-定制黑色西装-[1-3]",
        "costumeDesc": "意大利羊毛哑光黑。[强制16:9 横版构图，三视图，超写实 8K，写实]"}]}
  ],
  "scenes": [{"name": "酒店走廊", "description": "高反射率大理石地面。"}],
  "props": [{"name": "离职纸箱", "description": "粗糙瓦楞纸，边缘毛刺。"}]
}"""


# ---------------------------------------------------------------- 锁定后缀


def test_锁定后缀被切出来():
    """版式/白底那段是**版式约束不是创作**，不该让人看见也不该让人改。"""
    body, locked = split_locked("女 | 23岁 | 浅绿色瞳孔 [强制白底证件照]")
    assert locked == "[强制白底证件照]"
    assert "强制白底" not in body
    assert body.endswith("浅绿色瞳孔")


def test_正文里的方括号不会被误当成后缀():
    """只有**最后**那个方括号是技术约束，正文里的锚点描述要留着。"""
    body, locked = split_locked("[Female] | [25] | [Caucasian] | 冷感 [强制白底证件照]")
    assert locked == "[强制白底证件照]"
    assert "[Female]" in body and "[Caucasian]" in body


def test_没有后缀时不瞎切():
    body, locked = split_locked("潮湿的鹅卵石街道，煤气灯光在积水中倒影。")
    assert locked == ""
    assert body.endswith("倒影。")


def test_生图时后缀要拼回去():
    lib, err = parse_assets(ASSETS_JSON)
    assert not err
    cos = lib.characters[0].costumes[0]
    assert "强制16:9" not in cos.body, "后缀漏进了可编辑正文"
    assert "单张全身站姿正面照" in cos.prompt(), "提交生图时没把后缀拼回去"
    assert "三视图" not in cos.prompt(), "模型抄来的三视图版式没被换成标准后缀"


# ---------------------------------------------------------------- 解析容错


def test_引用标记被清掉():
    """模型会把文档工具的 [cite: 3, 4] 抄进来，它不是内容还会干扰生图。"""
    assert strip_cites("闪电劈落。 [cite: 4] 墙面刷白。 [cite: 5, 6]") == "闪电劈落。 墙面刷白。"


@pytest.mark.parametrize(
    "wrapped",
    [
        '[{{"episodeIndex":1,"episodeTitle":"第1集","episodeDesc":"{d}"}}]',
        '、[{{"episodeIndex":1,"episodeTitle":"第1集","episodeDesc":"{d}"}}]',
        '```json\n[{{"episodeIndex":1,"episodeTitle":"第1集","episodeDesc":"{d}"}}]\n```',
        '{{"episodes":[{{"episodeIndex":1,"episodeTitle":"第1集","episodeDesc":"{d}"}}]}}',
    ],
)
def test_解析容忍模型的各种包装(wrapped):
    text = wrapped.format(d="[夜] [内] [走廊]\\n[MCU] ANNE 停步。")
    eps, err = parse_episodes(text)
    assert not err
    assert eps[0].shot_count == 1


def test_台词引号没转义时能修():
    """**结构性**故障：规格要求台词写成 "..."，而它又是 JSON 字符串的值。"""
    bad = (
        '[{"episodeIndex":1,"episodeTitle":"第1集","episodeDesc":'
        '"[夜] [内] [走廊]\\n[MCU] ANNE 说："Is he blind?!" 她停步。"}]'
    )
    eps, err = parse_episodes(bad)
    assert not err, f"没修好：{err}"
    assert "Is he blind?!" in eps[0].desc


def test_裸换行写进字符串时能修():
    bad = '[{"episodeIndex":1,"episodeTitle":"第1集","episodeDesc":"[夜] [内] [走廊]\n[MCU] x"}]'
    eps, err = parse_episodes(bad)
    assert not err and eps[0].shot_count == 1


def test_合法输入不被误改():
    ok = '[{"episodeIndex":1,"episodeTitle":"第1集","episodeDesc":"[夜] [内] [走廊]\\n[MCU] x"}]'
    eps, err = parse_episodes(ok)
    assert not err and eps[0].desc.count("\n") == 1


def test_标题只有数字时补全():
    eps, _ = parse_episodes('[{"episodeTitle":"1","episodeDesc":"[夜] [内] [走廊]\\n[MCU] x"}]')
    assert eps[0].title == "第1集"


def test_解析失败要说原因而不是静默返回空():
    """一步错步步错：资产库空了，第③步就会编造引用。"""
    _, err = parse_episodes("模型今天不想输出 JSON")
    assert err
    _, err2 = parse_assets("{}")
    assert "角色" in err2


# ---------------------------------------------------------------- 场景与镜头


def test_场景标头和镜头能分开数():
    eps, _ = parse_episodes(
        '[{"episodeIndex":1,"episodeTitle":"第1集","episodeDesc":'
        '"[夜] [内] [走廊]\\n[全景] a\\n[近景] b\\n\\n[日] [内] [后厨]\\n[中景] c"}]'
    )
    e = eps[0]
    assert e.shot_count == 3
    assert e.scenes == ["[夜] [内] [走廊]", "[日] [内] [后厨]"]


# ---------------------------------------------------------------- 引用审计


def test_喂给第三步的摘要只有名字():
    """喂全量描述会撑爆上下文，而且模型会把服装细节抄进镜头 —— 那是第②步的活。"""
    lib, _ = parse_assets(ASSETS_JSON)
    d = lib.digest()
    assert "Anne-酒店员工制服-[全集]" in d
    assert "聚酯纤维" not in d, "把描述也喂进去了"
    assert "三视图" not in d


def test_编造的引用会被抓出来():
    """这是第③步最容易出的错，也是最难发现的：看着没毛病，生视频时变脸。"""
    lib, _ = parse_assets(ASSETS_JSON)
    shots, err = parse_shots(
        '[{"scene_index":"[第1集-1场]","video_name":"1-3","video_duration":"14s",'
        '"description":"(酒店走廊) 里，(Anne-制服-[1]) 推开 (Colin-定制黑色西装-[1-3])。"}]'
    )
    assert not err
    bad = audit_refs(shots, lib)
    assert any("Anne-制服-[1]" in b for b in bad)
    assert not any("Colin-定制黑色西装" in b for b in bad), "真实 ID 被误报了"


def test_切镜标记不算资产引用():
    shots, _ = parse_shots(
        '[{"scene_index":"[第1集-1场]","video_name":"1-3","video_duration":"14s",'
        '"description":"[切镜：近景] (Anne-酒店员工制服-[全集]) (L-Cut Voice-over)说：“x”"}]'
    )
    refs = shots[0].refs()
    assert "Anne-酒店员工制服-[全集]" in refs
    assert not any("L-Cut" in r for r in refs)


def test_前序视频引入被识别():
    """🔴 {第1集-1场} 是延续上一段，要把那段视频当参考图传给模型。"""
    shots, _ = parse_shots(
        '[{"scene_index":"[第1集-2场]","video_name":"9-20","video_duration":"12s",'
        '"description":"延长视频{第1集-1场}余光微弱。(酒店走廊) 中 x。"}]'
    )
    assert shots[0].carries() == ["第1集-1场"]


@pytest.mark.parametrize(("raw", "want"), [("14s", 14), ("12s", 12), ("", 15), ("abc", 15)])
def test_时长解析带兜底(raw, want):
    """🟣 时长不展示给前端，只用于提交模型。模型不给时按 15s 兜底。"""
    shots, _ = parse_shots(
        '[{"scene_index":"[1-1]","video_name":"1","video_duration":"' + raw + '",'
        '"description":"x"}]'
    )
    assert shots[0].seconds == want


# ---------------------------------------------------------------- 提示词完整性


def test_三段提示词都带输出示例():
    """示例定义了字段名和嵌套结构。

    真踩过：转录时把 ASSETS 的示例删了，模型就不知道 roleCostumeList 里
    该放什么，直接给空数组 —— 于是第③步无从引用，编出 11 个假 ID。
    """
    assert '"episodeDesc"' in STORYBOARD_SYSTEM
    assert '"roleCostumeList"' in ASSETS_SYSTEM
    assert '"costumeName"' in ASSETS_SYSTEM and '"costumeDesc"' in ASSETS_SYSTEM
    assert '"scene_index"' in SHOTS_SYSTEM and '"video_duration"' in SHOTS_SYSTEM


def test_资产提示词要求服装不能为空():
    assert "roleCostumeList 不能为空" in ASSETS_SYSTEM


def test_镜头提示词禁止编造ID():
    assert "严禁自己编造" in SHOTS_SYSTEM or "编造" in SHOTS_SYSTEM


# ---------------------------------------------------------------- 模型与版式


def test_视频提示词系统提示带表演节拍规律():
    """来自 awesome-seedance 案例库的几条实测规律要进 drama_shots 的系统提示词，
    管的是 description 怎么写；渲染层另外包的（参考锁定/快切/音色）不在这里重复。"""
    from aigc_agent.domain.drama import shots_system
    from aigc_agent.domain.drama.options import normalize

    p = shots_system(normalize("chinese", "zh"))
    assert "一拍只让一个人说话" in p
    assert "Handheld Low Angle" in p, "机位术语要中英混写"
    assert "hard cut" in p and "淡出" in p
    assert "情绪写行为不写症状" in p
    assert "结束状态" in p
    assert p.index("【表演与台词节拍") < p.index("台词一律是"), "语种那句仍在最后"


def test_生图生视频模型走配置不写死():
    """换模型不该动代码。"""
    from aigc_agent.app import PROJECT_ROOT
    from aigc_agent.domain.generators.catalog import MediaCatalog

    c = MediaCatalog.load(PROJECT_ROOT / "config" / "media_models.yaml")
    assert c.drama.get("image_model"), "没配生图模型"
    assert c.drama.get("video_model"), "没配生视频模型"


def test_配的模型在目录里真实存在():
    """配一个不存在的 id，要等到跑到那一步才炸，太晚。"""
    from aigc_agent.app import PROJECT_ROOT
    from aigc_agent.domain.generators.catalog import MediaCatalog, MediaKind

    c = MediaCatalog.load(PROJECT_ROOT / "config" / "media_models.yaml")
    assert c.choose(MediaKind.IMAGE, c.drama["image_model"], "")[0]
    assert c.choose(MediaKind.VIDEO, c.drama["video_model"], "")[0]


def test_锁定后缀缺失时用标准值兜底():
    """真踩过：模型没吐 [强制白底证件照]，切出来是空的，
    角色图就变成了环境人像而不是证件照。

    这段是**服务端持有的版式约束**，不是模型输出 —— 模型吐不吐都不该影响。

    2026-09-18 起「证件照」三个字在拼提示词时换成「正面半身照，原图直出无美颜」——
    证件照会把生图模型带向均匀布光的修图脸；白底、正面的版式要求保留。
    """
    from aigc_agent.domain.drama.models import Character, Costume

    c = Character(name="X", body="女 | 25岁", locked="")
    assert "强制白底" in c.prompt()
    assert "证件照" not in c.prompt()
    assert "原图直出" in c.prompt()

    cos = Costume(name="X-衣-[1]", body="深蓝色", locked="")
    assert "单张全身站姿正面照" in cos.prompt()
    assert "纯白色背景" in cos.prompt()
    assert "三视图" not in cos.prompt()


def test_模型吐了后缀就用它的():
    """兜底不能覆盖模型的正常输出。"""
    from aigc_agent.domain.drama.models import Character

    c = Character(name="X", body="女", locked="[自定义版式]")
    assert c.prompt().endswith("[自定义版式]")
    assert "证件照" not in c.prompt()


# ---------------------------------------------------------------- 面孔与语言

from aigc_agent.domain.drama import ETHNICITIES, LANGUAGES, ask_text  # noqa: E402


def test_没给选项时不生成而是提问():
    """族裔和语言会贯穿三视图与全部分镜视频，选错等于整条链重跑。

    所以**不猜**：缺任一项就停下来，把选项摆给用户。
    """
    o = normalize("", "")
    assert not o.ready
    assert set(o.missing) == {"ethnicity", "language"}
    assert normalize("asian", "").missing == ["language"]
    assert normalize("", "zh").missing == ["ethnicity"]


def test_提问文本把选项列全():
    t = ask_text()
    for k in ETHNICITIES:
        assert k in t
    for k in LANGUAGES:
        assert k in t
    assert "重跑" in t, "没说清楚为什么必须先定，用户会随便选"


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("亚洲", "asian"), ("亚裔", "asian"), ("asian", "asian"), ("East Asian", "asian"),
        ("中国", "chinese"), ("华人", "chinese"),
        ("欧美", "caucasian"), ("白人", "caucasian"), ("Caucasian", "caucasian"),
        ("黑人", "african"), ("拉丁", "latino"), ("不限", "mixed"),
    ],
)
def test_族裔说法能归一(raw, want):
    assert normalize(raw, "zh").ethnicity == want


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("中文", "zh"), ("zh", "zh"), ("普通话", "zh"),
        ("英文", "en"), ("English", "en"), ("原文", "keep"),
    ],
)
def test_语言说法能归一(raw, want):
    assert normalize("asian", raw).language == want


def test_选亚洲面孔时不再禁止亚裔():
    """原始提示词写死了「禁止出现亚裔」—— 那是为出海片调的，做国内片就是错的。"""
    p = assets_system(normalize("asian", "zh"))
    assert "禁止出现亚裔" not in p
    assert "East Asian" in p


def test_选欧美面孔时锁死欧美():
    p = assets_system(normalize("caucasian", "en"))
    assert "Caucasian" in p
    assert "East Asian" not in p


def test_不限定时让模型按剧本背景判断():
    p = assets_system(normalize("mixed", "keep"))
    assert "剧本" in p and "统一" in p


def test_指定语言时说清楚这是主动要的翻译():
    """否则会和「严禁翻译/二创」那条打架，模型无所适从。"""
    p = storyboard_system(normalize("asian", "en"))
    assert "English" in p
    assert "用户主动指定" in p
    assert "只有台词用" in p, "没说清楚运镜描写仍用中文，模型会整段译掉"


def test_跟原文时不加语言指令():
    assert storyboard_system(normalize("asian", "keep")) == storyboard_system(
        normalize("mixed", "keep")
    )


def test_第三步不再做语言转换():
    """①已经翻译过了，③再译一次会把台词改得对不上。"""
    p = shots_system(normalize("asian", "zh"))
    assert "不要再做语言转换" in p


# ---------------------------------------------------------------- 产线引导


def test_系统提示词要求先分清产线():
    """两条产线流程、配方、成本完全不同，走错等于白跑，
    而且要到出片才看得出来 —— 所以入口处必须先分清。
    """
    from aigc_agent.domain.system_prompt import SYSTEM_PROMPT as P

    assert "短视频" in P and "短剧" in P
    assert "先问" in P, "没要求模糊需求时主动问"
    assert "不要多此一问" in P, "没说明确需求时别啰嗦，会变成每次都问"


def test_四条产线的差别写清楚了():
    """只列名字模型分不出该走哪条，得说清楚各自是什么。
    2026-09-23 从两条扩到四条（加了广告、设计），并且用户 /type 选过就不再问。"""
    from aigc_agent.domain.system_prompt import SYSTEM_PROMPT as P

    assert "热榜" in P and "口播" in P  # 抖音短视频的特征
    assert "角色" in P and "一致性" in P  # 短剧的特征
    assert "product-ad" in P and "身份锁" in P  # 广告的特征
    assert "海报" in P and "make_poster" in P  # 设计的特征
    assert "当前产线" in P and "不要再问" in P, "用户选过产线就别再问"


def test_引导入口列出两条产线():
    from aigc_agent.interfaces.cli.start_cmd import LINES

    assert {v[0] for v in LINES.values()} == {"video", "drama"}
    assert all(v[2] for v in LINES.values()), "有产线没写说明，用户选不出来"

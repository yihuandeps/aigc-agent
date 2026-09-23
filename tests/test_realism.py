"""真实感预设验收 —— 消 AI 味儿。

人一旦看出"这是 AI 生成的"，内容就废了一半。暴露点高度集中在三处：
皮肤太干净、光太柔、成像太完美。这三条必须**贯穿所有会出真人的链路**，
漏一条就前功尽弃 —— 短剧资产图挂了、分镜视频没挂，成片里就会一半真一半假。

⚠️ 这套和短剧资产提示词原本的「明星美学干预 / 去瑕疵化」是**相反**的。
   原文明令禁止 freckles、要求皮肤"像剥壳鸡蛋"、强制柔和蝴蝶光。
   下面几条专门钉住"冲突确实被反转了"，防止后续迭代把老规则改回来。
"""

from __future__ import annotations

import pytest

from aigc_agent.domain.drama import assets_system, normalize, shots_system, storyboard_system
from aigc_agent.domain.realism import (
    LIGHT,
    OPTICS,
    PERSON,
    apply_conflicts,
    image_suffix,
    prompt_rules,
    video_suffix,
)

OPTS = normalize("asian", "zh")


# ---------------------------------------------------------------- 三条要求


@pytest.mark.parametrize(
    ("kw", "where"),
    [
        ("原图模式", PERSON), ("自然纹理", PERSON), ("雀斑", PERSON), ("颗粒", PERSON),
        ("侧光", LIGHT), ("局部光", LIGHT), ("硬光", LIGHT), ("闪光灯", LIGHT),
        ("色差", OPTICS), ("色散", OPTICS), ("红蓝溢出", OPTICS),
    ],
)
def test_三类要求都在(kw, where):
    assert kw in where


def test_明确禁掉磨皮和柔光():
    """只说"要什么"不够，模型的默认行为就是磨皮+柔光，必须显式禁掉。"""
    assert "严禁" in PERSON and "磨皮" in PERSON
    assert "不要" in LIGHT and ("柔光" in LIGHT or "蝴蝶光" in LIGHT)


def test_视频版要求颗粒随帧浮动():
    """静止的颗粒在视频里像脏镜头，不像胶片。"""
    assert "随帧" in video_suffix()
    assert "随帧" not in image_suffix()


# ---------------------------------------------------------------- 冲突反转


def test_不再禁止雀斑():
    """原文写死了 严禁出现"痣 (moles/freckles)" —— 和"加入雀斑"直接打架。"""
    p = assets_system(OPTS)
    assert "moles/freckles" not in p, "老的禁令还在，雀斑出不来"
    assert "雀斑" in p


def test_不再强制柔和蝴蝶光():
    p = assets_system(OPTS)
    assert '强制使用"柔和的蝴蝶光' not in p
    assert "侧光" in p and "硬光" in p


def test_不再要求剥壳鸡蛋皮肤():
    p = assets_system(OPTS)
    assert "剥壳鸡蛋一样" not in p
    assert "毛孔" in p


def test_不再强制高度对称():
    """完全对称的五官本身就是 AI 特征。"""
    p = assets_system(OPTS)
    assert '强制加入"高度对称的五官' not in p
    assert "不要高度对称" in p


def test_不冲突的老规则要留着():
    """族裔锁定、微表情注入、骨骼干预和真实感不冲突，不能一起删掉。"""
    p = assets_system(OPTS)
    assert "族裔锁定协议" in p
    assert "微表情" in p
    assert "物理骨骼干预" in p
    assert "单张全身站姿正面照" in p, "版式约束被误删了"
    assert "三视图" not in p, "资产库提示词里不许再出现三视图"


def test_关掉预设时老规则原样保留():
    """留一条退路：有人就是要棚拍明星脸。"""
    p = assets_system(OPTS, realism=False)
    assert "moles/freckles" in p
    assert '强制使用"柔和的蝴蝶光' in p
    assert "【真实感准则】" not in p


def test_替换是逐条的不是整段重写():
    assert apply_conflicts("无关文本") == "无关文本"


# ---------------------------------------------------------------- 覆盖全链路


def test_短剧三段提示词都挂了():
    for p in (assets_system(OPTS), storyboard_system(OPTS), shots_system(OPTS)):
        assert "【真实感准则】" in p


def test_角色护照分镜也挂了():
    from aigc_agent.domain.storyboard import Character, system_prompt

    p = system_prompt(Character(name="林夏", appearance="短黑发"))
    assert "雀斑" in p and "侧光" in p and "色差" in p


def test_短视频配方也挂了():
    """配方里写而不是代码里写 —— 用户要能直接调。"""
    from aigc_agent.domain.pipeline.recipe import load_recipe

    r = load_recipe("tech-short")
    assert r.shots.get("realism"), "配方里没有真实感段落"
    p = r.shot_prompt("一位工程师站在管道前")
    assert "雀斑" in p and "侧光" in p and "色差" in p


def test_真实感放在镜头描述之后():
    """它是全局约束，放前面会被当成画面主体 —— 模型会生成"一堆雀斑的特写"。"""
    from aigc_agent.domain.pipeline.recipe import load_recipe

    p = load_recipe("tech-short").shot_prompt("一位工程师站在管道前")
    assert p.index("工程师") < p.index("雀斑")


def test_规则段三类齐全():
    r = prompt_rules()
    assert "皮肤" in r and "光线" in r and "成像" in r

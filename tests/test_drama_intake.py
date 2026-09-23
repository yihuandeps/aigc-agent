"""短剧入口判断验收：一句话想法 vs 完整剧本。

**判错的代价不对称，两个方向都很痛：**

  想法误判成剧本 → 拿"一个霸总爱上我"去拆分镜，模型只能硬编，
                    出来的和用户脑子里的完全不是一回事
  剧本误判成想法 → 让用户重写他已经写好的剧本，最直接的冒犯

所以不靠模型猜，用可解释的结构信号打分；信号不够强就如实说"看不准"。
"""

from __future__ import annotations

import json

import pytest

from aigc_agent.domain.drama import ask_script_next, ask_unclear, classify, expand_prompt
from aigc_agent.domain.drama.writing import write_prompt
from aigc_agent.domain.functions.drama import _unwrap_script

SCRIPT = (
    "第1集\n\n[夜] [内] [酒店走廊]\n\n"
    "林夏抱着纸箱走在走廊上，牙关咬紧。\n\n"
    "林夏：干了六年，一句话就没了。\n\n"
    "陆川从电梯里走出来，两人撞在一起。\n\n"
    "陆川：没长眼睛？\n\n"
    "林夏蹲下去捡东西，手在发抖。\n\n"
    "林夏：对不起。"
)


# ---------------------------------------------------------------- 分类


@pytest.mark.parametrize(
    "idea",
    [
        "我想做一部霸总爱上我的短剧",
        "帮我做个都市逆袭题材的短剧，20集",
        "一个女生穿越回十年前",
        "讲的是外卖员捡到一部能看到明天新闻的手机",
    ],
)
def test_一句话被判成想法(idea):
    assert classify(idea).is_idea


def test_完整剧本被判成剧本():
    r = classify(SCRIPT)
    assert r.is_script
    assert any("场景标头" in s for s in r.signals)
    assert any("对白" in s for s in r.signals)


def test_极简但有结构的也算剧本():
    """三行也可能是剧本 —— 靠结构判，不靠长度。"""
    tiny = "第1集\n\n夜。走廊。\n\n林夏：我不去。\n陆川：由不得你。"
    assert classify(tiny).is_script


def test_说明加剧本不会被措辞带偏():
    """「我想做的这个剧本你看下：」开头，但正文是真剧本。

    措辞像需求只在**结构很弱**时才压分，否则这种就会被误判成想法，
    用户会被要求重写他刚发来的剧本。
    """
    assert classify("我想做的这个剧本你看下：\n\n" + SCRIPT).is_script


def test_空输入当想法():
    assert classify("").is_idea
    assert classify("   ").is_idea


def test_中间地带如实说看不准():
    """宁可问一句，也不要判错。"""
    r = classify("第1集\n主角登场\n然后发生了很多事\n最后反转")
    assert r.kind in {"unclear", "idea"}


def test_信号可解释():
    """判断依据要能说给人听，否则用户不知道为什么被要求重写。"""
    r = classify(SCRIPT)
    assert r.signals
    assert "完整剧本" in r.brief()


# ---------------------------------------------------------------- 提问


def test_看不准时把两条路都摆出来():
    t = ask_unclear(SCRIPT)
    assert "剧本" in t and "想法" in t
    assert "确认" in t


def test_拿到剧本时问直接进还是先改():
    t = ask_script_next()
    assert "直接进工程" in t
    assert "扩写" in t and "修改" in t
    assert "很贵" in t or "22 分钟" in t, "没说清楚代价，用户会随便选"


# ---------------------------------------------------------------- 写作提示词


def test_写剧本带方法论():
    p = write_prompt("外卖员捡到能看到明天新闻的手机", episodes=2, minutes=1.5)
    assert "前 3 秒" in p
    assert "信息缺口" in p
    assert "2 集" in p


def test_写剧本要求输出可拆解的格式():
    """写出来的东西下一步要能进 drama_storyboard，格式必须对上。"""
    p = write_prompt("随便")
    assert "[夜] [内]" in p or "场景标头" in p
    assert "角色名：" in p
    assert "不要写运镜和景别" in p, "写了运镜会和分镜师的活重复"


def test_扩写要保留原设定():
    """用户要的是改，不是换 —— 重起炉灶等于把他的活扔了。"""
    p = expand_prompt(SCRIPT)
    assert "保留原剧本的人物" in p
    assert "不要重起炉灶" in p


def test_扩写带具体检查项():
    p = expand_prompt(SCRIPT)
    for kw in ("开篇", "信息缺口", "两分钟", "同一集"):
        assert kw in p


# ---------------------------------------------------------------- JSON 剥壳


@pytest.mark.parametrize("key", ["script", "content", "text", "随便起的名"])
def test_剥掉模型套的json外壳(key):
    """drama 角色配了 response_format: json，写剧本这步被连带影响。

    不剥的话：存进去的是 JSON，下一步拿到的不是剧本；
    而且里面的换行是 \n 转义，结构分类器一行都认不出来。
    """
    raw = json.dumps({key: SCRIPT}, ensure_ascii=False)
    assert _unwrap_script(raw) == SCRIPT


def test_带markdown围栏也能剥():
    raw = "```json\n" + json.dumps({"script": SCRIPT}, ensure_ascii=False) + "\n```"
    assert _unwrap_script(raw) == SCRIPT


def test_纯文本不被动():
    assert _unwrap_script(SCRIPT) == SCRIPT


def test_剥壳后分类才准():
    raw = json.dumps({"script": SCRIPT}, ensure_ascii=False)
    assert not classify(raw).is_script, "没剥壳时本就不该被当成剧本"
    assert classify(_unwrap_script(raw)).is_script

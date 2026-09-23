"""分镜与角色护照验收（移植自 ai-character-passport）。

盯两类东西：

一是**移植时必须改掉的地方**。原项目按即梦的规格写死单镜 10–15 秒，
这边 veo3.1 最长 8 秒 —— 照搬的话模型会按 15 秒的信息量编排画面，
生成出来必然截断。这条最容易在后续迭代里被人"改回去"，所以钉死。

二是**护照的核心价值**：外貌必须逐镜注入。生成模型每次调用是独立的，
第一镜写清楚、后面用"她"指代，第二镜就会换一张脸。
"""

from __future__ import annotations

import json

import pytest

from aigc_agent.domain.storyboard import (
    Character,
    parse_shots,
    render_passport,
    script_messages,
    system_prompt,
)

LIN = Character(
    name="林夏",
    base_prompt="28岁女程序员，沉默寡言",
    appearance="短黑发, 圆框眼镜, 灰色连帽衫",
    style_lighting="冷蓝夜景, 电影感",
)


# ---------- 护照渲染 ----------


def test_自然语言格式带齐四个字段():
    s = render_passport(LIN, "natural")
    for bit in ("林夏", "28岁女程序员", "短黑发", "冷蓝夜景"):
        assert bit in s


def test_标签格式是逗号串且去重():
    c = Character(name="A", appearance="红发, 红发, 皮衣", style_lighting="皮衣")
    tags = render_passport(c, "tags").split(", ")
    assert len(tags) == len(set(tags)), "标签没去重，重复标签会让模型过度加权"
    assert "红发" in tags


def test_标签格式认中文逗号():
    c = Character(name="A", appearance="红发，皮衣；长靴")
    assert set(render_passport(c, "tags").split(", ")) >= {"红发", "皮衣", "长靴"}


def test_json格式可解析():
    d = json.loads(render_passport(LIN, "json"))
    assert d["character"]["name"] == "林夏"
    assert "短黑发" in d["character"]["appearance"]


def test_空护照渲染成空字符串():
    assert render_passport(None, "natural") == ""
    assert render_passport(Character(name=""), "natural") == ""


def test_护照太空要能被识别出来():
    """只有名字的护照注入进去等于没注入。"""
    assert not Character(name="张三").filled
    assert Character(name="张三", appearance="红发").filled


def test_护照可往返序列化():
    assert Character.from_dict(LIN.to_dict()) == LIN


# ---------- 提示词 ----------


def test_外貌必须逐镜注入而不是靠指代():
    """这是护照的全部意义所在。"""
    p = system_prompt(LIN)
    assert "短黑发" in p, "护照没进系统提示词"
    assert "每一镜" in p, "没要求逐镜注入，模型会只在第一镜写"
    assert "指代" in p, "没说清楚为什么不能用代词指代"


def test_没有角色时要求模型自己固定一套外貌():
    p = system_prompt(None)
    assert "固定" in p and "复用" in p


@pytest.mark.parametrize("secs", [3, 5, 8, 12])
def test_单镜时长跟着调用方走(secs):
    """原项目写死 10-15 秒，是按即梦的规格来的。

    这边 veo3.1 最长 8 秒，写大了模型会按更长的信息量编排画面，生成出来
    是截断的。这个数必须由调用方给。
    """
    p = system_prompt(LIN, seconds_each=secs)
    assert f"{secs}s" in p
    assert f"{secs} 秒" in p


def test_没有把原项目的十到十五秒抄进来():
    p = system_prompt(LIN, seconds_each=8)
    assert "10-15" not in p and "10–15" not in p


def test_要求运镜和光影():
    p = system_prompt(LIN)
    assert "push in" in p or "运镜" in p
    assert "lighting" in p or "光影" in p


def test_画面英文旁白中文():
    p = system_prompt(LIN)
    assert "English" in p, "画面提示词要英文：生成模型对英文标签的响应更稳"
    assert "中文旁白" in p, "旁白要中文：是拿去配音的"


def test_旁白要能直接配音而不是场记描述():
    assert "配音" in system_prompt(LIN)


def test_指定镜头数会写进提示词():
    assert "6 个镜头" in system_prompt(LIN, shot_count=6)
    assert "不要硬凑" in system_prompt(LIN, shot_count=0)


def test_无参考图时消息体是纯文本():
    msgs = script_messages("剧本内容", LIN)
    assert isinstance(msgs[-1]["content"], str)
    assert "剧本内容" in msgs[-1]["content"]


def test_有参考图时消息体带图():
    msgs = script_messages("剧本内容", LIN, reference_image="data:image/jpeg;base64,AAA")
    parts = msgs[-1]["content"]
    assert isinstance(parts, list)
    assert any(p.get("type") == "image_url" for p in parts)


# ---------- 解析 ----------

SHOTS_JSON = (
    '[{"shotNumber":1,"duration":"8s",'
    '"prompt":"close up, woman typing","narration":"她还在改"}]'
)


@pytest.mark.parametrize(
    "text",
    [
        SHOTS_JSON,
        f"```json\n{SHOTS_JSON}\n```",
        f"好的，这是分镜：\n{SHOTS_JSON}\n以上。",
        '{"shots": ' + SHOTS_JSON + "}",
        '{"storyboard": ' + SHOTS_JSON + "}",
    ],
)
def test_解析容忍模型的各种包装(text):
    shots, warn = parse_shots(text)
    assert not warn
    assert len(shots) == 1
    assert shots[0].prompt == "close up, woman typing"
    assert shots[0].narration == "她还在改"


def test_没有prompt的镜头被丢掉():
    shots, _ = parse_shots('[{"shotNumber":1,"prompt":""},{"shotNumber":2,"prompt":"a shot"}]')
    assert len(shots) == 1


def test_缺序号时按顺序补():
    shots, _ = parse_shots('[{"prompt":"one"},{"prompt":"two"}]')
    assert [s.number for s in shots] == [1, 2]


def test_解析失败返回警告而不是抛():
    shots, warn = parse_shots("我觉得可以这样拆分镜……")
    assert shots == []
    assert warn


def test_空数组也算失败():
    shots, warn = parse_shots("[]")
    assert shots == [] and warn

"""真实感档位（2026-09-18 下午）—— 用户看成片：雀斑、皱纹太明显，人物太丑，要减到原来的 1/3。

三档：subtle（默认，真实但干净）/ natural（早上那版）/ strong。档位贯穿：
人物图/视频提示词的硬约束、写作规则、校验目标、重生成方向，以及描述里过重瑕疵的压轻。
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from aigc_agent.domain.drama import assets_system, normalize
from aigc_agent.domain.realism import (
    DEFAULT_LEVEL,
    HARD_HEAD,
    PERSON,
    image_suffix,
    norm_level,
    parse_realism_report,
    person_image_prompt,
    person_video_prompt,
    prompt_rules,
    realism_check_messages,
    soften_flaws,
)

# ---------------------------------------------------------------- 档位与措辞


def test_默认档是subtle_不认识的值也回到默认():
    assert DEFAULT_LEVEL == "subtle"
    assert norm_level("") == "subtle" and norm_level("STRONG") == "strong"
    assert norm_level("heavy") == "subtle"


def test_subtle档的措辞是克制的():
    assert "零星几点" in PERSON and "不要明显的皱纹" in PERSON
    assert "不要刻意做旧" in HARD_HEAD and "attractive" in HARD_HEAD
    assert "也一律按轻微处理" in HARD_HEAD  # 描述里写重了也压回去
    # 三档强度单调：subtle 不要求"少量雀斑"，natural 要，strong 要"较多"
    assert "少量雀斑" not in image_suffix("subtle")
    assert "少量雀斑" in image_suffix("natural")
    assert "较多雀斑" in image_suffix("strong")


def test_写作规则按档位_subtle只许一处轻微细节():
    sub = prompt_rules("subtle")
    assert "一处轻微" in sub and "严禁写粗大毛孔" in sub
    nat = prompt_rules("natural")
    assert "至少两处" in nat
    # 资产库提示词跟着档位走
    assert "一处轻微" in assets_system(normalize("asian", "zh"), level="subtle")
    assert "至少两处" in assets_system(normalize("asian", "zh"), level="natural")


def test_过重瑕疵压轻_只在subtle档():
    src = "男 | 40岁 | 毛孔粗大，满脸雀斑，深深的法令纹，眼下明显的黑眼圈，痘坑 | 疲惫"
    out, hits = soften_flaws(src)
    for bad in ("毛孔粗大", "满脸雀斑", "深深的法令纹", "明显的黑眼圈", "痘坑"):
        assert bad not in out, bad
    assert "细看可见的毛孔" in out and "浅浅的法令纹" in out and "淡淡的黑眼圈" in out
    assert len(hits) == 5
    assert "疲惫" in out  # 其余原样
    # natural 档不压：那是用户明确要的力度
    text_nat, _ = person_image_prompt(src, level="natural")
    assert "满脸雀斑" in text_nat
    text_sub, _ = person_image_prompt(src, level="subtle")
    assert "满脸雀斑" not in text_sub


def test_视频提示词也按档位():
    sub = person_video_prompt("陆离推门，毛孔粗大", level="subtle")
    assert "毛孔粗大" not in sub and "no heavy wrinkles" in sub and "随帧" in sub
    nat = person_video_prompt("陆离推门，毛孔粗大", level="natural")
    assert "毛孔粗大" in nat and "skin shows pores, freckles" in nat


# ---------------------------------------------------------------- 校验与重生成方向


def test_校验目标按档位():
    text = realism_check_messages("data:image/png;base64,AA", "subtle")[0]["content"][0]["text"]
    assert "真实但干净" in text and "heavy" in text and "issue_type" in text
    text2 = realism_check_messages("data:image/png;base64,AA", "strong")[0]["content"][0]["text"]
    assert "岁月感" in text2


def test_解析方向_新格式与老格式():
    v = parse_realism_report(
        '{"pass": false, "score": 3, "issue_type": "heavy", "issues": ["雀斑太重"]}'
    )
    assert v.passed is False and v.direction == "heavy"
    v2 = parse_realism_report('{"pass": false, "score": 4, "issues": ["雀斑和皱纹太明显，显老"]}')
    assert v2.direction == "heavy", "老格式没 issue_type，从问题描述猜方向"
    v3 = parse_realism_report('{"pass": false, "score": 4, "issues": ["明显磨皮"]}')
    assert v3.direction == "smooth"
    v4 = parse_realism_report('{"pass": true, "score": 8, "issue_type": "ok", "issues": []}')
    assert v4.passed and v4.direction == ""


def test_重生成按方向换提示词():
    heavy, _ = person_image_prompt("女 | 鼻翼细看有毛孔", retry="heavy")
    assert heavy.startswith("【第二次生成】上一版面部瑕疵过重") and "三分之一" in heavy
    light, _ = person_image_prompt("女 | 鼻翼细看有毛孔", retry="light")
    assert "影棚均匀柔光" in light.split("\n")[0]
    smooth, _ = person_image_prompt("女 | 鼻翼细看有毛孔", retry=True)
    assert "过于光滑" in smooth.split("\n")[0] and "不要加重雀斑和皱纹" in smooth


# ---------------------------------------------------------------- 生图链路：做旧过头 → 往轻里重生成


class _FakeStore:
    def __init__(self) -> None:
        self.assets: dict[str, Any] = {}

    def get(self, aid: str) -> Any:
        return self.assets[aid]


def _make(verdicts: list[dict[str, Any]], level: str = "subtle") -> Any:
    from aigc_agent.domain.functions.drama import DramaFunctions

    store = _FakeStore()
    calls: list[tuple[str, Any]] = []

    async def chat(role: str, messages: list[dict[str, Any]], **_: Any) -> Any:
        calls.append((role, messages))
        return SimpleNamespace(text=json.dumps(verdicts.pop(0), ensure_ascii=False))

    fn = DramaFunctions.__new__(DramaFunctions)
    fn.gateway = SimpleNamespace(chat=chat)
    fn.catalog = SimpleNamespace(
        drama={"realism_gate": "true", "realism_retries": "1", "realism_level": level},
        max_concurrency=lambda kind: 2,
    )
    fn.store = store
    prompts: list[str] = []
    n = {"i": 0}

    async def invoke(name: str, args: dict[str, Any]) -> Any:
        prompts.append(args["prompt"])
        n["i"] += 1
        aid = f"as_{n['i']}"
        store.assets[aid] = SimpleNamespace(uri=f"https://x/{aid}.png", gen_params={})
        return SimpleNamespace(ok=True, asset_ref=aid, error="")

    fn.registry_invoke = invoke
    return fn, calls, prompts


def test_做旧过头_第二次往轻里生成():
    fn, calls, prompts = _make(
        [
            {"pass": False, "score": 3, "issue_type": "heavy", "issues": ["雀斑皱纹太重"]},
            {"pass": True, "score": 8, "issue_type": "ok", "issues": []},
        ]
    )
    report: list[dict[str, Any]] = []
    aid, _, err = asyncio.run(
        fn._gen_image("女 | 鼻翼细看有毛孔", "3:4", "角色·A", person=True, report=report)
    )
    assert err == "" and aid == "as_2"
    assert prompts[0].startswith(HARD_HEAD)
    assert prompts[1].startswith("【第二次生成】上一版面部瑕疵过重")
    assert report[0]["retry_direction"] == "heavy" and report[0]["pass"] is True
    # 校验提示词用的是 subtle 的目标
    assert "真实但干净" in calls[0][1][0]["content"][0]["text"]


def test_档位从配置读_natural用早上那版硬约束():
    ok = {"pass": True, "score": 8, "issue_type": "ok", "issues": []}
    fn, _, prompts = _make([ok], "natural")
    asyncio.run(fn._gen_image("女 | 鼻翼细看有毛孔", "3:4", "角色·A", person=True))
    assert prompts[0].startswith("【硬约束，优先级高于下文所有描述】原图直出的真实照片")
    assert "不要刻意做旧" not in prompts[0]

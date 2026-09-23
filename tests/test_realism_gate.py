"""真实感硬约束与生成后校验 —— 2026-09-18。

实测 7 张角色主形象：提示词里明明写了毛孔雀斑，出来还是磨皮脸。
光把真实感段挂在提示词末尾不够（生图模型当成可选氛围），改成三道保险：
  ① 硬约束放人物提示词开头，明确"忽略下文相反要求"，末尾再复述
  ② 模型写的描述先清洗：磨皮/光滑无瑕/柔光换掉；没写具体瑕疵的补默认锚点
  ③ 生成后视觉模型校验，不合格用更狠的提示词重生成一次；都不过就留分最高的并报「仍不达标」
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from aigc_agent.domain.realism import (
    HARD_HEAD,
    HARD_TAIL_VIDEO,
    RETRY_BOOST,
    ensure_imperfections,
    has_imperfections,
    image_suffix,
    parse_realism_verdict,
    person_image_prompt,
    person_video_prompt,
    realism_check_messages,
    sanitize_beauty,
)

# ---------------------------------------------------------------- ① 硬约束位置


def test_硬约束在人物提示词开头_真实感段在末尾():
    text, _ = person_image_prompt("女 | 25岁 | 鼻翼有雀斑 [强制白底正面半身照]")
    assert text.startswith(HARD_HEAD)
    assert text.index("鼻翼有雀斑") > text.index("忽略")
    assert text.endswith(image_suffix())
    assert "**" not in HARD_HEAD  # 生图模型不认 markdown，星号只会当噪音


def test_硬约束双语且声明优先级():
    for kw in ("毛孔", "雀斑", "硬光", "严禁磨皮", "一律忽略", "pores", "freckles", "Ignore"):
        assert kw in HARD_HEAD, kw


def test_重生成版本加更狠的前缀():
    text, _ = person_image_prompt("女 | 眼下细纹", retry=True)
    assert text.startswith(RETRY_BOOST)
    assert HARD_HEAD in text


# ---------------------------------------------------------------- ② 描述清洗


def test_美化措辞被换掉并记录():
    src = "女 | 25岁 | 皮肤光滑无瑕，柔和的蝴蝶光，高度对称的五官 | 神态倦怠"
    out, hits = sanitize_beauty(src)
    assert "光滑" not in out and "蝴蝶光" not in out and "高度对称" not in out
    assert "毛孔" in out and "硬光" in out and "不对称" in out
    assert hits  # 报告里要能看到替换了什么
    assert "神态倦怠" in out  # 其余原样保留


def test_英文美化词也清洗():
    out, hits = sanitize_beauty("flawless porcelain skin, perfectly symmetrical face")
    assert "flawless" not in out.lower() and "porcelain" not in out.lower()
    assert "asymmetrical" in out
    assert len(hits) >= 2


def test_没写具体瑕疵就补默认锚点():
    assert not has_imperfections("女 | 25岁 | 黑长直 | 冷淡")
    out, added = ensure_imperfections("女 | 25岁 | 黑长直 | 冷淡")
    assert added and "毛孔" in out
    same, added2 = ensure_imperfections("男 | 40岁 | 眼下有细纹")
    assert not added2 and same == "男 | 40岁 | 眼下有细纹"


def test_处理记录写进备注():
    _, notes = person_image_prompt("女 | 皮肤光滑 | 冷淡")
    joined = "".join(notes)
    assert "替换美化措辞" in joined and "光滑" in joined
    _, notes2 = person_image_prompt("女 | 黑长直")
    assert any("补默认锚点" in n for n in notes2)
    _, notes3 = person_image_prompt("女 | 鼻翼细看有毛孔")
    assert notes3 == []
    # subtle 档：写得过重的瑕疵会被压轻并记录（用户看成片说雀斑皱纹太明显）
    text4, notes4 = person_image_prompt("女 | 鼻翼毛孔粗大 | 满脸雀斑 | 深深的法令纹")
    assert any("压轻过重的瑕疵" in n for n in notes4)
    assert "毛孔粗大" not in text4 and "满脸雀斑" not in text4 and "深深的法令纹" not in text4
    assert "细看可见的毛孔" in text4 and "零星几点极淡的雀斑" in text4 and "浅浅的法令纹" in text4


def test_视频提示词描述在前_硬约束尾巴在后():
    text = person_video_prompt("陆离推门进屋，皮肤光滑，柔光")
    # 描述部分被清洗（真实感段自己会以"不要…柔光"的口吻提到柔光，那是禁令，不算）
    assert text.startswith("陆离推门进屋，皮肤有自然纹理与可见毛孔，侧向硬光")
    assert text.endswith(HARD_TAIL_VIDEO)
    assert "随帧" in text  # 视频版真实感段
    assert "皮肤光滑" not in text


# ---------------------------------------------------------------- ③ 校验解析


def test_校验消息是视觉格式():
    msgs = realism_check_messages("data:image/png;base64,AAAA")
    parts = msgs[0]["content"]
    assert parts[0]["type"] == "text" and "JSON" in parts[0]["text"]
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png")


def test_解析结果容忍围栏与前后废话():
    ok, score, issues = parse_realism_verdict(
        '看图结论如下：\n```json\n{"pass": false, "score": 3, "issues": ["明显磨皮"]}\n```'
    )
    assert ok is False and score == 3 and issues == ["明显磨皮"]
    ok2, score2, _ = parse_realism_verdict('{"score": 8, "issues": []}')  # 没给 pass 就按分数
    assert ok2 is True and score2 == 8


def test_解析失败按通过但标出来():
    ok, score, issues = parse_realism_verdict("模型抽风了")
    assert ok is True and score == -1 and issues


# ---------------------------------------------------------------- ④ 生图链路：不合格重生成


class _FakeStore:
    def __init__(self) -> None:
        self.assets: dict[str, Any] = {}

    def get(self, aid: str) -> Any:
        return self.assets[aid]


class _FakeGateway:
    """每次 chat 按队列吐一个校验结论。"""

    def __init__(self, verdicts: list[dict[str, Any]]) -> None:
        self.verdicts = list(verdicts)
        self.calls: list[tuple[str, Any]] = []

    async def chat(self, role: str, messages: list[dict[str, Any]], **_: Any) -> Any:
        self.calls.append((role, messages))
        v = self.verdicts.pop(0)
        return SimpleNamespace(text=json.dumps(v, ensure_ascii=False))


def _make(verdicts: list[dict[str, Any]], gate: str = "true", retries: str = "1") -> Any:
    from aigc_agent.domain.functions.drama import DramaFunctions

    store = _FakeStore()
    gateway = _FakeGateway(verdicts)
    catalog = SimpleNamespace(
        drama={"image_model": "m", "realism_gate": gate, "realism_retries": retries},
        max_concurrency=lambda kind: 2,
    )
    fn = DramaFunctions.__new__(DramaFunctions)
    fn.gateway = gateway
    fn.catalog = catalog
    fn.store = store
    prompts: list[str] = []
    counter = {"n": 0}

    async def invoke(name: str, args: dict[str, Any]) -> Any:
        assert name == "gen_image"
        prompts.append(args["prompt"])
        counter["n"] += 1
        aid = f"as_{counter['n']}"
        store.assets[aid] = SimpleNamespace(uri=f"https://x/{aid}.png", gen_params={})
        return SimpleNamespace(ok=True, asset_ref=aid, error="")

    fn.registry_invoke = invoke
    return fn, gateway, prompts


def test_一次通过不重生成():
    fn, gw, prompts = _make([{"pass": True, "score": 8, "issues": []}])
    report: list[dict[str, Any]] = []
    aid, url, err = asyncio.run(
        fn._gen_image("女 | 鼻翼雀斑", "3:4", "角色·A", person=True, report=report)
    )
    assert err == "" and aid == "as_1" and len(prompts) == 1
    assert prompts[0].startswith(HARD_HEAD)
    assert gw.calls[0][0] == "realism_check"
    assert report[0]["pass"] is True and report[0]["attempts"] == 1
    assert report[0]["asset"] == "as_1"


def test_不合格就用更狠的提示词重生成一次():
    fn, _, prompts = _make(
        [{"pass": False, "score": 3, "issues": ["磨皮"]}, {"pass": True, "score": 7, "issues": []}]
    )
    report: list[dict[str, Any]] = []
    aid, _, err = asyncio.run(
        fn._gen_image("女 | 鼻翼雀斑", "3:4", "角色·A", person=True, report=report)
    )
    assert err == "" and aid == "as_2" and len(prompts) == 2
    assert prompts[1].startswith(RETRY_BOOST)
    assert report[0]["pass"] is True and report[0]["attempts"] == 2
    assert report[0]["asset"] == "as_2"


def test_都不过就留分数最高的并标不达标():
    fn, _, prompts = _make(
        [{"pass": False, "score": 4, "issues": ["磨皮"]}, {"pass": False, "score": 2, "issues": []}]
    )
    report: list[dict[str, Any]] = []
    aid, _, err = asyncio.run(
        fn._gen_image("女 | 鼻翼雀斑", "3:4", "角色·A", person=True, report=report)
    )
    assert err == "" and len(prompts) == 2
    assert aid == "as_1"  # 第一张 4 分比第二张 2 分高
    assert report[0]["pass"] is False and report[0]["asset"] == "as_1"


def test_校验关着就只加提示词不调视觉模型():
    fn, gw, prompts = _make([], gate="false")
    report: list[dict[str, Any]] = []
    aid, _, err = asyncio.run(
        fn._gen_image("女 | 鼻翼雀斑", "3:4", "角色·A", person=True, report=report)
    )
    assert err == "" and aid == "as_1" and not gw.calls
    assert prompts[0].startswith(HARD_HEAD)
    assert report[0]["checked"] is False


def test_没配校验角色时按通过并写备注():
    fn, gw, _ = _make([])

    async def boom(role: str, messages: Any, **_: Any) -> Any:
        raise KeyError(role)

    gw.chat = boom  # type: ignore[assignment]
    report: list[dict[str, Any]] = []
    aid, _, err = asyncio.run(
        fn._gen_image("女 | 鼻翼雀斑", "3:4", "角色·A", person=True, report=report)
    )
    assert err == "" and aid == "as_1"
    assert report[0]["pass"] is True and "realism_check" in report[0]["note"]


def test_场景道具图不挂人物约束():
    fn, gw, prompts = _make([])
    aid, _, err = asyncio.run(fn._gen_image("老宅客厅，木地板", "16:9", "场景·客厅"))
    assert err == "" and aid == "as_1" and not gw.calls
    assert prompts[0] == "老宅客厅，木地板"


# ---------------------------------------------------------------- ⑤ 报告文案


def test_报告汇总一次通过_重生成_仍不达标():
    from aigc_agent.domain.functions.drama import _realism_note

    log = [
        {"name": "角色·A", "notes": [], "checked": True, "pass": True, "attempts": 1},
        {"name": "角色·B", "notes": ["替换美化措辞：光滑"], "checked": True, "pass": True,
         "attempts": 2},
        {"name": "服装·C", "notes": [], "checked": True, "pass": False, "attempts": 2,
         "score": 3, "issues": ["磨皮明显"]},
    ]
    note = _realism_note(log)
    assert "一次通过 1" in note and "重生成后通过 1" in note and "仍不达标 1" in note
    assert "服装·C" in note and "磨皮明显" in note
    assert "清洗/补瑕疵锚点 1 张" in note


def test_校验没开时报告说明原因():
    from aigc_agent.domain.functions.drama import _realism_note

    note = _realism_note([{"name": "角色·A", "notes": [], "checked": False}])
    assert "硬约束" in note and "realism_gate" in note
    assert _realism_note([]) == ""

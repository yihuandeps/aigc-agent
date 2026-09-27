"""一集的规格（2026-09-18 用户定的）：一集 4 分钟；开场 15 秒必须给高潮点。

贯穿写剧本（字数 + 高潮点行）、拆分镜（镜头数 + 【高潮点】标记）、视频提示词
（单段 10–15s、总时长、开场段 hook:true），每步都有确定性检查，不合格让模型改一次。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.drama import normalize, parse_shots, shots_system, storyboard_system
from aigc_agent.domain.drama.format import (
    HOOK_MARK,
    EpisodeFormat,
    check_script,
    check_shots,
    check_storyboard,
    shot_lines,
    shots_rules,
    storyboard_rules,
    writing_rules,
)
from aigc_agent.domain.drama.models import Episode
from aigc_agent.domain.drama.writing import expand_prompt, write_prompt
from aigc_agent.domain.functions.drama import DramaFunctions
from aigc_agent.domain.functions.episodes import build_contract

CONFIG = Path(__file__).resolve().parents[1] / "config" / "drama.yaml"
# 拆解类工具不收几个字的空壳（2026-09-23 缺口 F），测试要给一段像样的剧本
SCRIPT = "场景一 内景 客栈 夜\n" + "林秋推门进来，掌柜抬头看了她一眼，手里的算盘停了。\n" * 12


# ---------------------------------------------------------------- 规格本身


def test_规格派生值_与配置一致():
    fmt = EpisodeFormat.load(CONFIG)
    assert fmt.minutes == 4 and fmt.seconds == 240 and fmt.hook_seconds == 15
    assert fmt.script_chars == 1280 and fmt.script_range == (960, 1664)
    assert fmt.duration_range == (204, 276)
    assert fmt.shot_range == (16, 24) and fmt.hook_shots == 5
    # 2026-09-20 用户定的硬性要求：每个镜头 ≤3 秒
    assert fmt.max_cut_seconds == 3 and fmt.min_cut_seconds == 1
    assert fmt.cuts_per_segment(10) == 4 and fmt.cuts_per_segment(15) == 5
    assert fmt.cut_line_range == (68, 276) and fmt.cut_line_typical == (80, 120)
    assert "4 分钟" in fmt.brief() and "15 秒" in fmt.brief() and "≤3 秒" in fmt.brief()
    assert EpisodeFormat.load(Path("nope.yaml")) == EpisodeFormat()


def test_三段规则文本都带关键数字():
    fmt = EpisodeFormat()
    w = writing_rules(fmt)
    assert "4 分钟" in w and "1280 字" in w and "15 秒" in w and "⚡" in w
    assert "不超过 3 秒" in w, "写剧本就要按可切的短拍写"
    s = storyboard_rules(fmt)
    assert "240 秒" in s and "至少 68 个镜头行" in s and "80–120" in s and HOOK_MARK in s
    assert "每个镜头不超过 3 秒" in s and "[近景/推入/平视/2s]" in s and "前 5 镜" in s
    p = shots_rules(fmt)
    assert "10–15s" in p and "204–276s" in p and '"hook": true' in p
    assert '"cuts"' in p and "至少 4 个镜头" in p and "至少 5 个" in p


def test_规则进了写作_分镜_提示词的提示():
    fmt = EpisodeFormat()
    assert "1280 字" in write_prompt("想法", fmt=fmt)
    assert "约 640 字" in write_prompt("想法", minutes=2, fmt=fmt), "显式给的时长优先"
    assert "⚡" in expand_prompt("剧本", fmt=fmt)
    assert HOOK_MARK in storyboard_system(normalize("asian", "zh"), fmt=fmt)
    assert '"hook": true' in shots_system(normalize("asian", "zh"), fmt=fmt)
    assert HOOK_MARK not in storyboard_system(normalize("asian", "zh"))
    c = build_contract(
        n=3, total=40, entry="第3集 反转", prev_entry="", next_entry="", characters="",
        plan="", prev_tail="", note="", mode="domestic", rules=writing_rules(fmt),
    )
    assert "## 一集的规格（硬性）" in c and c.index("一集的规格") < c.index("输出模板")


# ---------------------------------------------------------------- 确定性检查


def _desc(n_shots: int, hook_at: int | None = 1, secs: float | None = 2.5) -> str:
    """分镜脚本：每个镜头行方括号最后一项标时长（2026-09-20 起，单镜 ≤3 秒）；secs=None 不标。"""
    lines = ["[夜] [内] [地铁车厢]"]
    for i in range(1, n_shots + 1):
        mark = HOOK_MARK if hook_at == i else ""
        tag = f"/{secs:g}s" if secs is not None else ""
        lines.append(f"{mark}[近景/手持{tag}] 陆离第{i}个动作：“台词{i}”")
    return "\n".join(lines)


def test_分镜脚本检查_镜头数与开场高潮点():
    fmt = EpisodeFormat()
    ok = Episode(index=1, title="第1集", desc=_desc(96, hook_at=1))  # 96 × 2.5s = 240s
    assert check_storyboard(ok, fmt) == []
    assert len(shot_lines(ok.desc)) == 96 and ok.shot_count == 96
    few = Episode(index=2, title="第2集", desc=_desc(6, hook_at=1))
    assert any("加起来 15s" in p and "204–276s" in p for p in check_storyboard(few, fmt))
    late = Episode(index=3, title="第3集", desc=_desc(96, hook_at=8))  # 第 8 镜从 17.5s 开始
    assert any("开场 15 秒内没有高潮点" in p for p in check_storyboard(late, fmt))
    edge = Episode(index=3, title="第3集", desc=_desc(96, hook_at=6))  # 第 6 镜从 12.5s 开始
    assert not any("开场" in p for p in check_storyboard(edge, fmt))
    none = Episode(index=4, title="第4集", desc=_desc(96, hook_at=None))
    assert len(check_storyboard(none, fmt)) == 1


def test_分镜脚本检查_单镜时长():
    """2026-09-20 用户定的硬性要求：每个镜头 ≤3 秒；没标时长也不合格。"""
    fmt = EpisodeFormat()
    untimed = Episode(index=1, title="第1集", desc=_desc(96, hook_at=1, secs=None))
    probs = check_storyboard(untimed, fmt)
    assert len(probs) == 1 and "96 个镜头行没标时长" in probs[0] and "2s" in probs[0]
    too_long = Episode(index=2, title="第2集", desc=_desc(80, hook_at=1, secs=3.5))
    probs = check_storyboard(too_long, fmt)
    assert any(
        "80 个镜头超过 3 秒" in p and "最长 3.5s" in p and "拆成多个镜头" in p for p in probs
    )
    assert any("加起来 280s" in p for p in probs)
    mixed = _desc(95, hook_at=1) + "\n[全景/固定/俯拍/5s] 陆离站了很久"
    probs = check_storyboard(Episode(index=3, title="第3集", desc=mixed), fmt)
    assert any("1 个镜头超过 3 秒" in p and "陆离站了很久" in p for p in probs)
    tiny = _desc(96, hook_at=1) + "\n[特写/固定/0.5s] 一闪"
    tiny_ep = Episode(index=4, title="x", desc=tiny)
    assert any("短于 1 秒" in p for p in check_storyboard(tiny_ep, fmt))
    # 时长写法：整数秒 / 小数 / 中文「秒」 / 【高潮点】前缀
    from aigc_agent.domain.drama.format import shot_seconds

    assert shot_seconds("[近景/推入/平视/2s] x") == 2
    assert shot_seconds("【高潮点】[特写/急推/1.5秒] x") == 1.5
    assert shot_seconds("[近景/推入] x") is None
    assert shot_seconds("[日] [内] [地铁]") is None


def _cuts_for(d: int, max_cut: int = 3) -> list[int]:
    out, left = [], d
    while left > 0:
        out.append(min(max_cut, left))
        left -= out[-1]
    return out


def _shots(
    durs: list[int], hook_idx: set[int], ep: int = 1, cuts: Any = "auto"
) -> list[Any]:
    rows = []
    for i, d in enumerate(durs, 1):
        row = {
            "scene_index": f"[第{ep}集-{i}场]", "video_name": f"{i}", "video_duration": f"{d}s",
            "description": f"镜头{i} (陆离)", "hook": i in hook_idx,
        }
        if cuts == "auto":
            row["cuts"] = _cuts_for(d)
        elif cuts:
            row["cuts"] = cuts
        rows.append(row)
    shots, err = parse_shots(json.dumps(rows, ensure_ascii=False))
    assert not err, err
    return shots


def test_提示词检查_段内镜头cuts():
    """2026-09-20：一段 10–15s 是多镜头快切，cuts 每个 ≤3s、加起来 = 段长。"""
    fmt = EpisodeFormat()
    no_cuts = _shots([12] * 20, {1}, cuts=False)
    probs = check_shots(no_cuts, fmt)
    assert any("20 段没给 cuts" in p and "≤ 3s" in p for p in probs)
    assert any("覆盖的镜头太少" in p and "至少 4 镜" in p for p in probs), "video_name 只有 1 镜"
    assert no_cuts[0].shot_count == 1 and no_cuts[0].cuts == []

    rows = [{"scene_index": "[第1集-1场]", "video_name": "9-13", "video_duration": "12s",
             "description": "x (陆离)", "hook": True, "cuts": "3,3,3,3"}]
    s = parse_shots(json.dumps(rows))[0][0]
    assert s.cuts == [3, 3, 3, 3], "cuts 字符串也认"
    assert s.shot_count == 5, "video_name 范围能算镜头数"

    long_cut = _shots([12] * 20, {1}, cuts=[4, 4, 4])
    probs = check_shots(long_cut, fmt)
    assert any("cuts 里有超过 3s 的镜头（最长 4s）" in p for p in probs)
    mismatch = _shots([12] * 20, {1}, cuts=[3, 3])
    assert any("cuts 加起来不等于 video_duration" in p and "6s≠12s" in p
               for p in check_shots(mismatch, fmt))


def test_提示词检查_总时长_单段时长_开场hook():
    fmt = EpisodeFormat()
    good = _shots([12] * 20, {1})  # 240s
    assert check_shots(good, fmt) == [] and good[0].hook and not good[1].hook
    short = _shots([12] * 8, {1})  # 96s
    assert any("总时长 96s" in p and "204–276s" in p for p in check_shots(short, fmt))
    long_seg = _shots([12] * 19 + [30], {1})
    assert any("不在 10–15s" in p for p in check_shots(long_seg, fmt))
    no_hook = _shots([12] * 20, {5})
    assert any("开场 15 秒内" in p and "没有标" in p for p in check_shots(no_hook, fmt))
    # 第一段 10s 没 hook、第二段（起点 10s < 15s）有 hook 也算开场高潮点
    second = _shots([10] + [12] * 19 + [2], {2})
    assert not any("开场" in p for p in check_shots(second, fmt))
    # hook 字段容忍字符串
    rows = [{"scene_index": "[第1集-1场]", "video_name": "1", "video_duration": "12s",
             "description": "x (a)", "hook": "true"}]
    assert parse_shots(json.dumps(rows))[0][0].hook is True


def test_剧本检查_字数与高潮点行():
    fmt = EpisodeFormat()
    text = "# 第1集：开场\n> ⚡ 前15秒高潮点：陆离被推下站台\n" + "正文" * 700
    assert check_script(text, fmt) == []
    short = "# 第1集\n> ⚡ 前15秒高潮点：x\n" + "字" * 300
    assert any("只有" in p and "1280" in p for p in check_script(short, fmt))
    no_hook = "# 第1集\n" + "字" * 1300
    assert any("⚡" in p for p in check_script(no_hook, fmt))
    too_long = "# 第1集\n> ⚡ 前15秒高潮点：x\n" + "字" * 2000
    assert any("超出" in p for p in check_script(too_long, fmt))


# ---------------------------------------------------------------- 不合格让模型改一次


class SeqGateway:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(self, role: str, messages: list[dict[str, Any]], **_: Any) -> Any:
        self.calls.append(messages)
        return SimpleNamespace(text=self.texts.pop(0))


LIB = {
    "characters": [
        {
            "baseRoleName": "陆离",
            "roleTotalDesc": "男 | 35岁",
            "roleCostumeList": [{"costumeName": "陆离-风衣-[全集]", "costumeDesc": "风衣"}],
        }
    ],
    "scenes": [{"name": "地铁车厢", "description": "x"}],
    "props": [],
}


def _lib_asset(store: AssetStore) -> Any:
    return store.create(
        json.dumps(LIB, ensure_ascii=False), summary="资产库", creator="tool:drama_assets"
    )


def _shots_json(durs: list[int], hook_idx: set[int], names: list[str] | None = None) -> str:
    """names：每段覆盖的分镜镜号（如 "1-2"），不给就是第 i 段只覆盖第 i 镜。"""
    rows = [
        {"scene_index": f"[第1集-{i}场]",
         "video_name": names[i - 1] if names else f"{i}", "video_duration": f"{d}s",
         "cuts": _cuts_for(d),
         "description": f"(地铁车厢) (陆离-风衣-[全集]) 镜头{i}", "hook": i in hook_idx}
        for i, d in enumerate(durs, 1)
    ]
    return json.dumps(rows, ensure_ascii=False)


async def test_提示词不合规格_改一次_合格版落资产():
    store = AssetStore()
    # 96 镜 × 2.5s = 240s 的分镜（合规格）；第一版只写了 8 段 96s，改一次后 20 段覆盖全部 96 镜
    sb = store.create(
        json.dumps([{"episodeIndex": 1, "episodeTitle": "第1集", "episodeDesc": _desc(96)}]),
        summary="分镜", creator="tool:drama_storyboard",
    )
    lib = _lib_asset(store)
    spans = [f"{5 * i - 4}-{min(5 * i, 96)}" for i in range(1, 21)]
    gw = SeqGateway([_shots_json([12] * 8, set()), _shots_json([12] * 20, {1}, spans)])
    fns = DramaFunctions(gw, store, registry=None, catalog=None, fmt=EpisodeFormat())
    r = await fns._fn_drama_shots(sb.id, lib.id)
    assert r.ok, r.error
    assert len(gw.calls) == 2 and "硬性问题" in gw.calls[1][-1]["content"]
    assert "总时长 96s" in gw.calls[1][-1]["content"]
    assert "20 段" in r.content and "✓ 规格" in r.content
    saved = json.loads(store.content(r.asset_ref))
    assert len(saved) == 20 and saved[0]["hook"] is True
    assert store.get(r.asset_ref).gen_params["format_problems"] == 0


async def test_改一次仍不合格_如实报出():
    store = AssetStore()
    sb = store.create(
        json.dumps([{"episodeIndex": 1, "episodeTitle": "第1集", "episodeDesc": _desc(18)}]),
        summary="分镜", creator="tool:drama_storyboard",
    )
    lib = _lib_asset(store)
    # 八段覆盖分镜全部 18 个镜头（2026-09-25 起漏镜头会先补写、补不上就不保存），只是总时长不够
    spans = ["1-2", "3-4", "5-6", "7-8", "9-10", "11-12", "13-15", "16-18"]
    gw = SeqGateway([_shots_json([12] * 8, set(), spans), _shots_json([12] * 8, set(), spans)])
    fns = DramaFunctions(gw, store, registry=None, catalog=None, fmt=EpisodeFormat())
    r = await fns._fn_drama_shots(sb.id, lib.id, note="第一段就是推下站台")
    assert r.ok and "⚠ 规格检查未过" in r.content and "总时长 96s" in r.content
    assert "第一段就是推下站台" in gw.calls[0][-1]["content"]
    assert store.get(r.asset_ref).gen_params["format_problems"] == 2


async def test_分镜脚本不合规格_改一次():
    store = AssetStore()
    bad = json.dumps([{"episodeIndex": 1, "episodeTitle": "第1集", "episodeDesc": _desc(6, None)}])
    good = json.dumps([{"episodeIndex": 1, "episodeTitle": "第1集", "episodeDesc": _desc(96, 1)}])
    gw = SeqGateway([bad, good])
    fns = DramaFunctions(gw, store, registry=None, catalog=None, fmt=EpisodeFormat())
    r = await fns._fn_drama_storyboard(SCRIPT, ethnicity="asian", language="zh")
    assert r.ok, r.error
    assert len(gw.calls) == 2 and "96 镜" in r.content and "✓ 规格" in r.content
    assert HOOK_MARK in gw.calls[0][0]["content"]  # system 提示里有规则

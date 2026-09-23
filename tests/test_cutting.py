"""剪辑节奏验收。

用户的要求是硬的：**单个镜头最长不超过 3 秒**。这里盯三件事 ——
不超时、切点是真的（相邻两刀不能是同一段素材的连续帧，否则接起来
看不出切）、以及成片时长不被剪辑改掉。

还有一条容易被忽略的：素材池和成片时长必须解耦。耦在一起的话，
想切 12 刀就得生成 12 段素材，成本和耗时都是三倍。
"""

from __future__ import annotations

import pytest

from aigc_agent.domain.pipeline.cutting import Cut, describe, plan_cuts
from aigc_agent.domain.pipeline.recipe import load_recipe

CASES = [
    ([8.0, 8.0, 8.0, 8.0], 30.0),
    ([8.0] * 6, 30.0),
    ([8.0, 8.0, 8.0, 8.0], 32.0),
    ([8.0, 8.0], 10.0),
    ([6.0, 6.0, 6.0], 30.0),
    ([12.0, 12.0], 45.0),
    ([8.0, 5.0, 8.0, 3.0], 25.0),
]


@pytest.mark.parametrize(("clips", "total"), CASES)
def test_没有一刀超过上限(clips, total):
    for c in plan_cuts(clips, total, max_seconds=3.0, min_seconds=1.5):
        assert c.dur <= 3.0 + 1e-6, f"这一刀 {c.dur}s 超过 3 秒上限"


@pytest.mark.parametrize(("clips", "total"), CASES)
def test_总时长不被剪辑改掉(clips, total):
    cuts = plan_cuts(clips, total, max_seconds=3.0, min_seconds=1.5)
    assert abs(sum(c.dur for c in cuts) - total) < 0.01


@pytest.mark.parametrize(("clips", "total"), CASES)
def test_不会切到素材边界外(clips, total):
    for c in plan_cuts(clips, total, max_seconds=3.0, min_seconds=1.5):
        assert c.end <= clips[c.clip] + 1e-6, f"{c} 超出了素材长度 {clips[c.clip]}s"


@pytest.mark.parametrize(("clips", "total"), CASES)
def test_相邻两刀必须是真切点(clips, total):
    """多素材时轮流取；单素材时窗口不能重叠 —— 否则接起来是连续帧或卡顿。"""
    cuts = plan_cuts(clips, total, max_seconds=3.0, min_seconds=1.5)
    for a, b in zip(cuts, cuts[1:], strict=False):
        if len(clips) > 1:
            assert a.clip != b.clip, f"相邻两刀来自同一段素材：{a} → {b}"
        else:
            assert not (a.start < b.end - 1e-6 and b.start < a.end - 1e-6)


def test_单素材也要切出真切点():
    cuts = plan_cuts([8.0], 20.0, max_seconds=3.0, min_seconds=1.5)
    for a, b in zip(cuts, cuts[1:], strict=False):
        assert not (a.start < b.end - 1e-6 and b.start < a.end - 1e-6), (
            f"单素材相邻窗口重叠，播出来是卡顿不是切点：{a} → {b}"
        )


def test_刀数够密():
    """30 秒 3 秒上限，至少得有 10 刀，否则谈不上快切。"""
    cuts = plan_cuts([8.0] * 4, 30.0, max_seconds=3.0, min_seconds=1.5)
    assert len(cuts) >= 10


def test_单刀时长不整齐划一():
    """全部等长会像幻灯片。"""
    durs = {c.dur for c in plan_cuts([8.0] * 4, 30.0, max_seconds=3.0, min_seconds=1.5)}
    assert len(durs) > 3, "每刀都一样长，节奏会很机械"


def test_上限调小刀数就变多():
    a = plan_cuts([8.0] * 4, 30.0, max_seconds=3.0, min_seconds=1.0)
    b = plan_cuts([8.0] * 4, 30.0, max_seconds=1.5, min_seconds=0.8)
    assert len(b) > len(a)
    assert all(c.dur <= 1.5 + 1e-6 for c in b)


def test_短片不用切():
    assert plan_cuts([8.0, 8.0], 2.5, max_seconds=3.0) == [Cut(0, 0.0, 2.5)]


def test_空输入不崩():
    assert plan_cuts([], 30.0) == []
    assert plan_cuts([8.0], 0.0) == []
    assert "无剪辑表" in describe([])


def test_下限比上限还紧时以上限为准():
    """min 写得离谱不该让硬约束失效。"""
    cuts = plan_cuts([8.0] * 4, 30.0, max_seconds=3.0, min_seconds=9.0)
    assert all(c.dur <= 3.0 + 1e-6 for c in cuts)
    assert abs(sum(c.dur for c in cuts) - 30.0) < 0.01


# ---------- 配方 ----------


def test_配方开了快切且上限不超过三秒():
    r = load_recipe("tech-short")
    assert r.cut_enabled
    assert 0 < r.cut_max <= 3.0


def test_素材池和成片时长是分开的():
    """耦在一起的话，切 12 刀就得生成 12 段，成本三倍。"""
    r = load_recipe("tech-short")
    assert r.source_seconds != r.total_seconds
    assert r.total_seconds == int(r.output["duration"])


def test_文案字数按成片算而不是素材池():
    r = load_recipe("tech-short")
    assert r.script_chars == int(r.total_seconds * 4.5)


def test_命令行能覆盖节奏():
    r = load_recipe("tech-short").override(max_cut=1.8)
    assert r.cut_max == 1.8
    assert load_recipe("tech-short").cut_max != 1.8, "覆盖不该写回原配方"


def test_快切下的duration改的是成片不是素材量():
    r = load_recipe("tech-short")
    o = r.override(duration=45)
    assert o.total_seconds == 45
    assert o.shot_count == r.shot_count, "快切模式下加时长不该多生成素材"


def test_画面时长要能盖住旁白():
    """文案按 30 秒写，TTS 渲染出来可能是 35 秒。

    剪辑表如果只排到 30 秒，混音时会按画面长度把音频截掉，旁白断在半句上。
    所以有配音时以配音长度排刀 —— 这条在 compose_video 里，这里守住规划器
    对任意目标时长都能排满。
    """
    for voice_len in (27.4, 30.0, 35.8):
        cuts = plan_cuts([8.0] * 6, voice_len, max_seconds=3.0, min_seconds=1.5)
        assert abs(sum(c.dur for c in cuts) - voice_len) < 0.01
        assert all(c.dur <= 3.0 + 1e-6 for c in cuts)

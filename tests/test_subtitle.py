"""字幕对齐验收。

这条修的是一个真出过的事故：一条讲氢能的片子，ASR 把"氢"全程听成"芯"
（"所以卡住芯能的"），错字烧进了画面，而这是整条片子的核心词。

根因是设计问题 —— 配音是我们自己的文案合成的，原稿就在手上，它才是
ground truth；ASR 该只负责时间轴。所以这里盯两件事：
**文字必须来自原稿**，**时间轴必须一个字节都不动**。
"""

from __future__ import annotations

from aigc_agent.domain.pipeline.subtitle import align_script, parse_srt, render_srt

# 真实事故现场：ASR 把"氢"听成"芯"
BAD_SRT = """1
00:00:00,000 --> 00:00:03,000
芯为什么造得出却运不起?

2
00:00:03,000 --> 00:00:07,000
这个话题热度已冲到1194.7万

3
00:00:07,000 --> 00:00:14,000
第一,芯是宇宙最轻的元素,同样重量,体积却大得多

4
00:00:14,000 --> 00:00:21,000
所以卡住芯能的,从来不是实验室,而是运输的账本
"""

SCRIPT = (
    "氢为什么造得出却运不起？"
    "这个话题热度已冲到1194.7万。"
    "第一，氢是宇宙最轻的元素，同样重量，体积却大得多。"
    "所以卡住氢能的，从来不是实验室，而是运输的账本。"
)


def test_同音字被原稿纠正():
    fixed, changed = align_script(SCRIPT, BAD_SRT)
    assert "芯" not in fixed, "ASR 的错字还在，等于白修"
    assert "氢" in fixed
    assert changed > 0


def test_时间轴一个字节都不动():
    """时间轴是 ASR 唯一不可替代的贡献，动了就白用它了。"""
    before = [(c.start, c.end) for c in parse_srt(BAD_SRT)]
    after = [(c.start, c.end) for c in parse_srt(align_script(SCRIPT, BAD_SRT)[0])]
    assert before == after


def test_条数不变():
    assert len(parse_srt(align_script(SCRIPT, BAD_SRT)[0])) == len(parse_srt(BAD_SRT))


def test_原稿内容全部落进字幕():
    fixed, _ = align_script(SCRIPT, BAD_SRT)
    joined = "".join(c.text for c in parse_srt(fixed))
    # 标点会在切分时被剥掉，比字符集合
    assert set("氢为什么造得出却运不起") <= set(joined)
    assert set("运输的账本") <= set(joined)


def test_没有空字幕条():
    for c in parse_srt(align_script(SCRIPT, BAD_SRT)[0]):
        assert c.text.strip(), "出现空字幕条，画面上会闪一下空白"


def test_按各段比重分配而不是平均切():
    """第 3 条的 ASR 文字最长，分到的原稿也该最长。"""
    cues = parse_srt(align_script(SCRIPT, BAD_SRT)[0])
    assert len(cues[2].text) > len(cues[1].text)


# ---------- 退路 ----------


def test_srt解析不了就原样返回():
    assert align_script(SCRIPT, "这不是 srt") == ("这不是 srt", 0)


def test_空原稿不动字幕():
    assert align_script("", BAD_SRT) == (BAD_SRT, 0)
    assert align_script("   ", BAD_SRT) == (BAD_SRT, 0)


def test_原稿和字幕一致时不做无谓改动():
    srt, _ = align_script(SCRIPT, BAD_SRT)
    again, changed = align_script(SCRIPT, srt)
    assert changed == 0, "已经对齐过的字幕不该再被改"


def test_无标点长句也能切开():
    srt = """1
00:00:00,000 --> 00:00:02,000
aaa

2
00:00:02,000 --> 00:00:04,000
bbb
"""
    fixed, _ = align_script("这是一段完全没有标点的很长的中文句子用来测试切分", srt)
    cues = parse_srt(fixed)
    assert len(cues) == 2
    assert all(c.text for c in cues)


def test_没有序号行的srt也认():
    srt = "00:00:00,000 --> 00:00:02,000\n芯很轻\n\n00:00:02,000 --> 00:00:04,000\n运不起\n"
    cues = parse_srt(srt)
    assert len(cues) == 2
    assert cues[0].start == "00:00:00,000"


def test_渲染后能被重新解析():
    cues = parse_srt(BAD_SRT)
    assert len(parse_srt(render_srt(cues))) == len(cues)


# ---------- 切点吸附：两个真踩过的坑 ----------

# ASR 的断句（简化自真实案例）
SEG_SRT = """1
00:00:00,000 --> 00:00:04,000
把氫氣冷卻到零下253度變成液體

2
00:00:04,000 --> 00:00:06,000
體積大幅縮小

3
00:00:06,000 --> 00:00:09,000
第二 管道疏氫落地
"""
SEG_SCRIPT = "把氢气冷却到零下253度变成液体，体积大幅缩小。第二，管道输氢落地。"


def test_不把下一句的开头词拖进上一条():
    """坑一：切点向前吸附到了"第二，"后面。

    屏幕上会出现"体积大幅缩小。第二"，下一条才是"管道输氢落地"——读起来是错的。
    """
    cues = parse_srt(align_script(SEG_SCRIPT, SEG_SRT)[0])
    assert not cues[1].text.rstrip().endswith("第二"), f"又把下一句开头拖进来了：{cues[1].text}"


def test_不为了够句号而吞掉整条():
    """坑二：修坑一时矫枉过正 —— 句末标点优先做成了双向。

    往前够句号会跨过中间的逗号，把下一条的内容整段吞掉，
    于是那一条只剩"第二"两个字。
    """
    cues = parse_srt(align_script(SEG_SCRIPT, SEG_SRT)[0])
    for c in cues:
        assert len(c.text) >= 4, f"这条被吞得只剩 {c.text!r}"


def test_切分贴合asr的断句():
    """ASR 的段落边界是有信息量的，对齐后不该偏移。"""
    cues = parse_srt(align_script(SEG_SCRIPT, SEG_SRT)[0])
    assert "液体" in cues[0].text
    assert "体积大幅缩小" in cues[1].text
    assert "管道输氢" in cues[2].text


def test_繁体asr被简体原稿覆盖():
    """ASR 可能输出繁体，而成片是给大陆平台的。"""
    fixed, _ = align_script(SEG_SCRIPT, SEG_SRT)
    assert "氫" not in fixed and "體" not in fixed


def test_同音错字被纠正():
    """ASR 把"输氢"听成"疏氢"——原稿是唯一真相。"""
    fixed, _ = align_script(SEG_SCRIPT, SEG_SRT)
    assert "疏氢" not in fixed and "管道输氢" in fixed

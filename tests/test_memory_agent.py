"""M8.2 Memory Agent 验收 —— 长期记忆的生成与召回。

**这是 P1 之前真正缺的一块。** 在此之前的实际行为是：滑窗涨到 15 轮、
淘汰 5 轮，那 5 轮**直接丢弃** —— `on_evict` 钩子留着没人挂，
`memory_extract` / `memory_recall` 两个角色配了没人调，长期记忆永远是空的。

盯三件事：
  1. 提取要**带 polarity** —— 纯关键词丢掉"要还是不要"，
     「别写成硬广」只存 [开头,硬广] 的话，下次召回分不清方向
  2. 提取要**过滤** —— 记忆会被反复注入上下文，噪音会挤掉真正有用的
  3. 召回要**能跨中文分词** —— 存「短视频开头」查「帮我写个开头」，
     精确匹配一条都召不回来（实测过，全空）
"""

from __future__ import annotations

import json

import pytest

from aigc_agent.capabilities.memory.agent import (
    MAX_PER_BATCH,
    extract_prompt,
    parse_memories,
    terms_of,
)
from aigc_agent.capabilities.memory.store import Category, MemoryStore, Polarity, Source


def _raw(*items: dict) -> str:
    return json.dumps(list(items), ensure_ascii=False)


# ---------------------------------------------------------------- 提取


def test_提取出的记忆带极性():
    """「别写成硬广」和「要写成硬广」存下来必须能分辨。"""
    m = parse_memories(
        _raw({
            "content": "开头不要写成硬广腔调",
            "terms": ["开头", "硬广"],
            "polarity": "negative",
            "category": "constraint",
            "quote": "开头太硬广了，别这么写",
        }),
        "turns:0-4",
    )
    assert len(m) == 1
    assert all(k.polarity is Polarity.NEGATIVE for k in m[0].keywords)
    assert m[0].keywords[0].origin_quote


def test_约束类权重更高():
    """约束是最高价值的记忆，召回排序要排在前面。"""
    c = parse_memories(_raw({"content": "时长 30 秒内", "terms": ["时长"],
                             "category": "constraint"}), "x")[0]
    t = parse_memories(_raw({"content": "在做科技选题", "terms": ["科技"],
                             "category": "topic"}), "x")[0]
    assert c.weight > t.weight


def test_从对话提的一律算推测():
    """**不得自动晋升到账号层** —— 模型把"这次这么说"当成"一直都这样"
    是最常见的记忆污染。
    """
    m = parse_memories(_raw({"content": "用户喜欢口语化", "terms": ["口语"]}), "x")[0]
    assert m.source is Source.INFERRED


def test_没有关键词的被丢掉():
    """召不回来的记忆是死数据，存了只占地方。"""
    assert parse_memories(_raw({"content": "某个事实", "terms": []}), "x") == []


def test_内容太短的被丢掉():
    assert parse_memories(_raw({"content": "好", "terms": ["好"]}), "x") == []


def test_条数有上限():
    """提太多等于没提 —— 召回时全是噪音。"""
    items = [{"content": f"事实{i}", "terms": [f"词{i}"]} for i in range(20)]
    assert len(parse_memories(_raw(*items), "x")) <= MAX_PER_BATCH


def test_解析失败返回空而不是抛():
    """记忆提取失败不该让用户那一轮对话跟着失败。"""
    assert parse_memories("模型今天不想输出 JSON", "x") == []
    assert parse_memories("", "x") == []


def test_非法极性和类别退回默认():
    m = parse_memories(_raw({"content": "某个事实", "terms": ["词"],
                             "polarity": "很负面", "category": "瞎写的"}), "x")[0]
    assert m.keywords[0].polarity is Polarity.NEUTRAL
    assert m.keywords[0].category is Category.TOPIC


def test_提取提示词强调过滤和极性():
    p = extract_prompt("一段对话")
    assert "只抽真正会影响后续工作的" in p
    assert "不能省" in p and "polarity" in p
    assert "噪音" in p, "没说清楚为什么要少抽，模型会塞一堆"


# ---------------------------------------------------------------- 召回


def _store(tmp_path):
    s = MemoryStore(tmp_path)
    for m in parse_memories(
        _raw(
            {"content": "科技短视频开头不要写成硬广腔调", "terms": ["短视频开头", "硬广"],
             "polarity": "negative", "category": "constraint"},
            {"content": "账号短视频时长控制在 30 秒以内", "terms": ["短视频时长", "完播率"],
             "polarity": "positive", "category": "constraint"},
        ),
        "turns:0-4",
    ):
        s.put(m)
    return s


def test_中文查询能跨过分词(tmp_path):
    """倒排索引是精确匹配，存「短视频开头」查「帮我写个开头」一条都召不回来。

    做法是反过来拿索引里已有的词去查询串里找 —— 词表有界，扫一遍很便宜，
    不用引入分词器，也不用每轮加一次模型调用。
    """
    s = _store(tmp_path)
    terms = s.match_terms("帮我写个开头")
    assert terms, "中文查询一个词都没匹配上"
    assert s.recall(terms), "匹配到词却召不回记忆"


def test_无关查询不误召回(tmp_path):
    """召回错的比召不回更糟 —— 会往上下文里塞无关约束。"""
    s = _store(tmp_path)
    assert not s.recall(s.match_terms("今天天气怎么样"))


def test_精确词优先(tmp_path):
    s = _store(tmp_path)
    assert "硬广" in s.match_terms("这段是不是太硬广了")


def test_短查询不触发全表扫描(tmp_path):
    s = _store(tmp_path)
    assert s.match_terms("好") == []
    assert s.match_terms("") == []


@pytest.mark.parametrize(
    ("text", "want"),
    [("开头 别写 硬广", ["开头", "别写", "硬广"]), ("a, b, cc", ["cc"]), ("", [])],
)
def test_朴素切词仍是退路(text, want):
    """英文和带标点的输入靠它，不能因为加了索引匹配就删掉。"""
    assert terms_of(text) == want


# ---------------------------------------------------------------- 接线


def test_淘汰钩子被挂上了():
    """这是 P1 的缺口所在：钩子留着但没人挂，淘汰的轮次直接丢弃。"""
    import inspect

    from aigc_agent.app import Agent

    src = inspect.getsource(Agent.create)
    assert "on_evict" in src, "淘汰钩子没挂，长期记忆永远写不进去"
    assert "mem_agent.submit" in src


def test_每轮都会召回():
    import inspect

    from aigc_agent.app import Agent

    src = inspect.getsource(Agent.chat)
    assert "recall" in src, "召回没接进主循环，存了也用不上"
    assert "recalled=" in src


def test_轮次能转成可读文本():
    """记忆提取要看到完整的一问一答，只给 user_text 会丢掉助手做了什么。"""
    from aigc_agent.harness.context.window import Turn

    t = Turn(index=0, messages=[
        {"role": "user", "content": "开头别写成硬广"},
        {"role": "assistant", "content": "明白"},
    ])
    assert "用户：开头别写成硬广" in t.transcript
    assert "助手：明白" in t.transcript

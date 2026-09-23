"""2026-09-23 审查的记忆治理问题：一次性事实永久有效、写入不去重、命中数不落盘。"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

from aigc_agent.capabilities.memory.agent import MemoryAgent, parse_memories
from aigc_agent.capabilities.memory.brief import build_brief, transient_ttl
from aigc_agent.capabilities.memory.store import (
    Category,
    Keyword,
    Memory,
    MemoryStore,
    Polarity,
    Source,
)


def _raw(*items: dict) -> str:
    return json.dumps(list(items), ensure_ascii=False)


def test_一次性事实设过期():
    assert transient_ttl("seedance 账户余额不足（HTTP 402）") is not None
    assert transient_ttl("第1集剧本以 as_0c8964a3f7 为准") is not None
    assert transient_ttl("开头不要写成硬广") is None
    mems = parse_memories(_raw(
        {"content": "seedance 余额不足（402），第3场没生成", "terms": ["seedance"],
         "polarity": "negative", "category": "constraint"},
        {"content": "开头不要写成硬广腔", "terms": ["开头"], "polarity": "negative",
         "category": "constraint"},
    ), "turns:1-2", "p1")
    assert mems[0].valid_until is not None and mems[0].valid_until < time.time() + 3 * 86400
    assert mems[1].valid_until is None


def test_老数据里的一次性事实按创建时间算过期():
    s = MemoryStore()
    old = Memory(content="seedance 余额不足（HTTP 402）", project_id="p1",
                 source=Source.INFERRED, created_at=time.time() - 5 * 86400,
                 keywords=[Keyword(term="余额", polarity=Polarity.NEUTRAL,
                                   category=Category.PREFERENCE)])
    s.put(old)
    b = build_brief(s, "p1")
    assert not any("余额" in x for x in b.should + b.must + b.must_not)


async def test_写入去重_给旧的加分量():
    s = MemoryStore()
    agent = MemoryAgent(None, s, project_id="p1")
    first = _raw({"content": "科技短视频开头不要写成硬广式的震惊腔调", "terms": ["硬广"],
                  "polarity": "negative", "category": "constraint"})
    second = _raw({"content": "科技短视频开头不要写成硬广式震惊腔调", "terms": ["硬广"],
                   "polarity": "negative", "category": "constraint"})
    replies = iter([first, second])

    async def chat(role, messages, **kw):
        return SimpleNamespace(text=next(replies))

    agent.gateway = SimpleNamespace(chat=chat)
    assert await agent.extract_now("第一段对话") == 1
    assert await agent.extract_now("第二段对话") == 0, "同一件事不再存一条"
    only = s.all(project_id="p1")
    assert len(only) == 1 and only[0].weight > 1.5 and only[0].hit_count == 1


def test_命中数落盘(tmp_path):
    s = MemoryStore(tmp_path)
    m = s.put(Memory(content="开头不要硬广", project_id="p1",
                     keywords=[Keyword(term="硬广", polarity=Polarity.NEGATIVE,
                                       category=Category.CONSTRAINT)]))
    s.recall(["硬广"], project_id="p1")
    again = MemoryStore(tmp_path)
    assert again.get(m.id).hit_count == 1, "重启后命中数还在"

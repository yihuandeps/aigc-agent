"""会话现场快照 —— 同名 session 重启后接着上次聊。

跨会话会丢的只有滑窗原文（资产与长期记忆本来就落盘）。
快照只存轮次、不存 pin；装回按窗口上限截断。
"""

from __future__ import annotations

import pytest

from aigc_agent.capabilities.memory.session import SessionSnapshot
from aigc_agent.harness.context.window import ShortTermMemory, WindowPolicy


def _memory_with_turns(n: int, **kw) -> ShortTermMemory:
    m = ShortTermMemory(**kw)
    for i in range(n):
        t = m.new_turn()
        t.messages.append({"role": "user", "content": f"第{i + 1}句"})
        t.messages.append({"role": "assistant", "content": f"回{i + 1}"})
        t.tokens = 10
    return m


def test_落盘后装回接着上次聊(tmp_path):
    store = SessionSnapshot(tmp_path, "demo")
    store.save(_memory_with_turns(3))

    fresh = ShortTermMemory()
    assert store.load_into(fresh) == 3
    assert [t.user_text for t in fresh.turns] == ["第1句", "第2句", "第3句"]
    assert fresh.token_estimate == 30
    # 轮次编号接续旧会话，新开的轮不与装回的撞车
    assert fresh.new_turn().index == 3


def test_没有快照或快照坏了就是全新会话(tmp_path):
    assert SessionSnapshot(tmp_path, "nobody").load_into(ShortTermMemory()) == 0

    bad = SessionSnapshot(tmp_path, "bad")
    bad.root.mkdir(parents=True, exist_ok=True)
    bad.path.write_text("{不是 JSON", encoding="utf-8")
    assert bad.load_into(ShortTermMemory()) == 0


def test_装回时按窗口上限截断(tmp_path):
    store = SessionSnapshot(tmp_path, "demo")
    store.save(_memory_with_turns(8))

    # 窗口策略可能在两次会话之间被调小过
    fresh = ShortTermMemory(policy=WindowPolicy(window_turns=5, evict_at=7))
    assert store.load_into(fresh) == 5
    assert fresh.turns[0].user_text == "第4句"
    assert fresh.new_turn().index == 8


def test_pin不随快照复活(tmp_path):
    """Memory Brief 每轮重 pin、rollback 说明是一次性的 —— 旧 pin 复活只会带回过期约束。"""
    store = SessionSnapshot(tmp_path, "demo")
    m = _memory_with_turns(1)
    m.pin("rollback", "已回退到 xx", position="pre_input")
    store.save(m)

    fresh = ShortTermMemory()
    assert store.load_into(fresh) == 1
    assert fresh.pins == {}


def test_session名里的文件名非法字符清洗(tmp_path):
    store = SessionSnapshot(tmp_path, 'a/b:c*d?"<>|')
    store.save(_memory_with_turns(1))
    assert "/" not in store.path.name and ":" not in store.path.name
    assert store.load_into(ShortTermMemory()) == 1


def test_产物目录随快照记住(tmp_path):
    store = SessionSnapshot(tmp_path, "demo")
    store.save(_memory_with_turns(1))
    store.set_output_dir("E:/产物")

    fresh = SessionSnapshot(tmp_path, "demo")
    assert fresh.output_dir == "E:/产物"
    # 之后每轮的 save 是整体重写 —— 不带 output_dir 的话就丢了
    fresh.save(_memory_with_turns(2))
    assert SessionSnapshot(tmp_path, "demo").output_dir == "E:/产物"


def test_旧快照没有output_dir字段兼容(tmp_path):
    import json

    (tmp_path / "legacy.json").write_text(
        json.dumps({"format": 1, "name": "legacy", "next_index": 0, "turns": []}),
        encoding="utf-8",
    )
    assert SessionSnapshot(tmp_path, "legacy").output_dir == ""


def test_写失败不上抛(tmp_path):
    """快照是锦上添花，落盘失败不该打断对话。"""
    store = SessionSnapshot(tmp_path / "blocked", "demo")
    store.root.mkdir(parents=True)
    store.path.mkdir()  # 同名目录挡在前面，write_text 会 OSError
    store.save(_memory_with_turns(1))  # 不抛


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

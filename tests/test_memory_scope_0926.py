"""2026-09-26 用户拍板（docs/待拍板-2026-09-26.md 一·7，方案 A + B + C）：记忆分级、跨剧隔离。

真实现场：蜘蛛精剧的「控制在十二集」「视频模型切换…不允许换」「已充值用 seedance 2.0 fast」
在《不渡》里每轮都 pin；问「用 drama_render_shots 渲第 3 集」会召回「需逐镜 gen_video 传
image_urls」（推测），和系统提示词正好相反。库里 56 条记忆有 48 条没有项目键。

  A. 决策类环节（换模型、面容、素材、出片确认…）的打回不存避雷；内容类打回挂项目和环节，
     默认 30 天过期
  B. 没有项目键的旧记忆在有项目的会话里隔离（不召回、不 pin），等人认领
  C. 对话里 /remember（人亲口定的规则，每轮 pin）和 /forget（列出来、确认后作废）
另外两条纯 bug：打回理由带着 a/r/j 前缀入库；召回把「推测」的否定条目标成「本项目已被打回」。
"""

from __future__ import annotations

import contextlib
import time
from types import SimpleNamespace
from typing import Any

from aigc_agent.capabilities.memory.agent import MemoryAgent
from aigc_agent.capabilities.memory.brief import build_brief
from aigc_agent.capabilities.memory.recorder import RejectionRecorder, is_decision_stage
from aigc_agent.capabilities.memory.store import (
    Category,
    Keyword,
    Layer,
    Memory,
    MemoryStore,
    Polarity,
    Source,
    in_scope,
)
from aigc_agent.harness.events.bus import Event, EventType


def _decided(node: str, reason: str, decision: str = "revise") -> Event:
    return Event(type=EventType.CHECKPOINT_DECIDED,
                 data={"node": node, "target_node": node, "decision": decision, "reason": reason})


def _legacy(content: str, term: str, source: Source = Source.HUMAN) -> Memory:
    """没有项目键的旧记忆（迁移前的样子）。"""
    return Memory(
        content=content,
        keywords=[Keyword(term=term, polarity=Polarity.NEGATIVE, category=Category.CONSTRAINT)],
        source=source,
        project_id="",
    )


# ---------------------------------------------------------------- A. 决策类不存、内容类过期


def test_决策类环节的打回不存避雷_内容类挂项目和环节_30天过期():
    store = MemoryStore()
    rec = RejectionRecorder(store, project_id="不渡")
    for node in ("视频模型切换", "生图模型切换", "角色面容冲突", "素材", "出片确认",
                 "参考图·主形象被拒"):
        rec._on_event(_decided(node, "不允许换"))
    assert not store.all() and len(rec.skipped) == 6, "一次性的操作决定，不是内容偏好"
    # 文案审核 / 成片审核的打回多半是内容意见（「口播太硬」）：照常存
    rec._on_event(_decided("文案审核", "口播太硬，像念报告"))
    assert [x.stage for x in store.all()] == ["文案审核"]

    rec._on_event(_decided("剧本·创作方案", "节奏再快一点，控制在十二集"))
    m = next(x for x in store.all() if x.stage == "剧本·创作方案")
    assert m.project_id == "不渡" and m.source is Source.HUMAN
    assert m.valid_until and 29 * 86400 < m.valid_until - time.time() <= 30 * 86400
    assert is_decision_stage("视频模型切换") and not is_decision_stage("剧本第3集")


def test_打回理由过期后不再进简报():
    store = MemoryStore()
    m = store.record_rejection("开头太平了", node_id="正文", run_id="", project_id="p1")
    assert any("太平" in x for x in build_brief(store, "p1").must_not)
    m.valid_until = time.time() - 1
    store.put(m)
    assert not build_brief(store, "p1").must_not, "过期就不 pin 了"
    keep = store.record_rejection("开头太平了", node_id="正文", run_id="", project_id="p1",
                                  ttl_days=None)
    assert keep.valid_until is None, "ttl_days=None：永久（旧行为可选）"


# ---------------------------------------------------------------- B. 没有项目键的旧记忆隔离


def test_没有项目键的旧记忆_有项目的会话里不召回不pin_账号层照旧():
    store = MemoryStore()
    old = store.put(_legacy("「剧本·创作方案」曾被打回（revise）：控制在十二集", "剧本"))
    store.put(Memory(layer=Layer.ACCOUNT, content="这个号不用感叹号", source=Source.HUMAN,
                     keywords=[Keyword(term="感叹号", polarity=Polarity.NEGATIVE,
                                       category=Category.CONSTRAINT)]))
    b = build_brief(store, "不渡")
    assert not any("十二集" in x for x in b.must_not), "别的剧的规矩不 pin 进这部剧"
    assert any("感叹号" in x for x in b.must_not), "账号层全局照样生效"
    assert not store.recall(["剧本"], project_id="不渡")
    assert store.recall(["剧本"]), "没有项目上下文（脚本、测试）照旧看得见"
    assert not in_scope(old, "不渡") and in_scope(old, "")
    assert [m.id for m in store.unclaimed()] == [old.id]

    store.claim(old.id, "不渡")
    assert any("十二集" in x for x in build_brief(store, "不渡").must_not), "认领后才生效"
    assert not store.unclaimed()


def test_认领账号层记忆要报错_认领要给项目键():
    import pytest

    store = MemoryStore()
    acc = store.put(Memory(layer=Layer.ACCOUNT, content="x", source=Source.HUMAN))
    old = store.put(_legacy("y", "y"))
    with pytest.raises(ValueError):
        store.claim(acc.id, "p1")
    with pytest.raises(ValueError):
        store.claim(old.id, "")


async def test_记忆提取不跟被隔离的旧记忆去重():
    """用户在新剧里又说了一遍：之前只给那条被隔离的旧记忆加分量，新剧里还是没有。"""

    class _Gw:
        async def chat(self, role: str, messages: list[dict[str, Any]], **_: Any) -> Any:
            return SimpleNamespace(text='[{"content": "开头不要写成硬广腔调", "terms": ["硬广"], '
                                        '"polarity": "negative", "category": "constraint"}]')

    store = MemoryStore()
    old = store.put(_legacy("开头不要写成硬广腔调", "硬广", source=Source.INFERRED))
    agent = MemoryAgent(_Gw(), store, project_id="不渡")
    assert await agent.extract_now("用户：开头别写硬广") == 1
    mine = [m for m in store.all() if m.project_id == "不渡"]
    assert len(mine) == 1 and store.get(old.id).weight == 1.0


# ---------------------------------------------------------------- C. /remember /forget


def test_remember_人亲口定的规则_这个项目每轮pin():
    store = MemoryStore()
    ban = store.remember("开头别写成硬广", project_id="不渡")
    store.remember("每集片尾都要有剩余花瓣计数", project_id="不渡")
    assert ban.source is Source.HUMAN and ban.valid_until is None
    b = build_brief(store, "不渡")
    assert "开头别写成硬广" in b.must_not and "每集片尾都要有剩余花瓣计数" in b.must
    assert "开头别写成硬广" in b.pin_text()
    assert build_brief(store, "西游记").empty, "只对这个项目"
    g = store.remember("所有视频都不要出现水印", everywhere=True)
    assert g.layer is Layer.ACCOUNT
    assert any("水印" in x for x in build_brief(store, "西游记").must_not), "全局的哪部剧都带"


def test_forget按关键词或id找_只找这个项目看得到的():
    store = MemoryStore()
    a = store.remember("开头别写成硬广", project_id="不渡")
    store.remember("开头别写成硬广", project_id="西游记")
    old = store.put(_legacy("「视频模型切换」曾被打回（revise）：r 不允许换", "视频模型切换"))
    assert [m.id for m in store.find(a.id, "不渡")] == [a.id]
    assert [m.project_id for m in store.find("硬广", "不渡")] == ["不渡"], "别的剧的不列"
    assert [m.id for m in store.find("不允许换", "不渡")] == [old.id], "待认领的旧记忆也能找到"


class _Hub:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.stopped_by_user = False

    async def ask(self, prompt: str = "") -> str:
        return self.answers.pop(0)

    async def watch(self, coro: Any) -> Any:
        return await coro


async def test_对话里_remember_forget():
    from aigc_agent.interfaces.cli.main import _forget, _remember

    store = MemoryStore()
    agent = SimpleNamespace(mem_agent=SimpleNamespace(store=store, project_id="不渡"))
    await _remember(agent, "开头别写成硬广")
    await _remember(agent, "全局 所有视频都不要出现水印")
    kinds = {(m.content, m.layer, m.project_id) for m in store.all()}
    assert ("开头别写成硬广", Layer.PROJECT, "不渡") in kinds
    assert ("所有视频都不要出现水印", Layer.ACCOUNT, "") in kinds

    old = store.put(_legacy("控制在十二集", "剧本"))
    await _remember(agent, old.id)
    assert store.get(old.id).project_id == "不渡", "/remember mem_xxx = 认领到这个项目"

    await _forget(agent, _Hub(["n"]), "硬广")
    assert any(m.content == "开头别写成硬广" for m in store.all()), "没确认就不动"
    await _forget(agent, _Hub(["y"]), "硬广")
    assert not any(m.content == "开头别写成硬广" for m in store.all())


# ---------------------------------------------------------------- 纯 bug


async def test_追问理由时又敲了决定字母_只存后面的理由():
    from aigc_agent.interfaces.cli.main import _decide_review

    got: list[tuple[str, str]] = []

    async def resume_turn(decision: str, reason: str = "", decided_by: str = "human") -> str:
        got.append((decision, reason))
        return "ok"

    agent = SimpleNamespace(
        loop=SimpleNamespace(
            pending_review={"stage": "剧本", "question": "看一下", "assets": []},
            resume_turn=resume_turn,
        ),
        assets=None,
    )
    board = SimpleNamespace(running=contextlib.nullcontext)
    await _decide_review(agent, board, _Hub(["r", "r 不允许换"]))  # type: ignore[arg-type]
    assert got == [("revise", "不允许换")], "之前存成了「r 不允许换」"


def test_推测的否定条目不标成本项目已打回():
    store = MemoryStore()
    guess = store.put(Memory(
        content="需逐镜 gen_video 传 image_urls", source=Source.INFERRED,
        keywords=[Keyword(term="gen_video", polarity=Polarity.NEGATIVE,
                          category=Category.CONSTRAINT)],
    ))
    rej = store.record_rejection("开头太平了", node_id="正文", run_id="")
    text = store.render_brief([guess, rej])
    must_not = text.split("### 参考")[0]
    assert "开头太平了" in must_not and "gen_video" not in must_not
    assert "避免：需逐镜 gen_video 传 image_urls（推测）" in text

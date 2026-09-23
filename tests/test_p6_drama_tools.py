"""2026-09-17 复盘后的领域侧改动验收：资产查找、合规分型、写作子代理、项目卡。

真实日志里发生过的事：
  · list_assets 把 612 份资产全吐、被截到前几十条，模型以为其余不存在，
    转头用 search_library 逐集找了 34 次，还编出不存在的 id（as_d6191a0a80）
  · 广告法极限词打在剧本台词上，「最强」「绝对」「第一」全判 block，
    模型花了三个会话重写 60 集去删词
  · 60 集在对话里逐集写，一次「继续」只写 1–2 集，上下文撑到 25 万
  · 35 条记忆召回命中 0 次 —— 输入全是「继续」
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aigc_agent.capabilities.subagents import SubAgentResult
from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.compliance import ComplianceChecker, ComplianceRules
from aigc_agent.domain.drama.card import CARD_PIN, build_project_card
from aigc_agent.domain.functions.compliance import ComplianceFunctions
from aigc_agent.domain.functions.content import ContentFunctions
from aigc_agent.domain.functions.drama import DramaFunctions
from aigc_agent.domain.functions.episodes import (
    EpisodeFunctions,
    build_contract,
    episode_title,
    outline_entry,
)
from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.model.gateway import ModelResponse, ToolCall, Usage
from aigc_agent.harness.permission.gate import PermissionGate
from aigc_agent.harness.tools.dispatcher import ToolDispatcher
from aigc_agent.harness.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
RULES = ComplianceRules.load(ROOT / "config" / "compliance.yaml")


def _episode(store: AssetStore, n: int, title: str = "标题", creator: str = "model") -> Any:
    return store.create(
        f"# 第{n}集：{title}\n\n正文…\n\n> 🎣 本集钩子：第{n}集的悬念",
        type_=AssetType.SCRIPT,
        summary=f"第{n}集·{title}",
        creator=creator,
        gen_params={"episode": n},
    )


async def _content_registry(store: AssetStore):
    bus = EventBus()
    reg = ToolRegistry(bus)
    reg.register(ContentFunctions(store))
    await reg.refresh()
    return reg, bus


# ---------------------------------------------------------------- 资产查找


def test_资产不存在时给相近的id():
    store = AssetStore()
    real = store.create("x", summary="真的")
    fake = real.id[:8] + "zz"  # 前几位对、后面编 —— 超长上下文下的典型幻觉
    with pytest.raises(KeyError) as e:
        store.get(fake)
    assert real.id in str(e.value) and "相近" in str(e.value)
    with pytest.raises(KeyError) as e2:
        store.get("as_zzzzzzzzzz")
    assert "相近" not in str(e2.value), "毫不相干的 id 不要瞎猜"


def test_find按类型集号创建者关键词过滤():
    store = AssetStore()
    e1 = _episode(store, 1)
    e2a = _episode(store, 2)
    e2b = _episode(store, 2, title="修订")  # 同一集新版本
    store.create("大纲", type_=AssetType.OUTLINE, summary="分集目录", creator="model")
    shots = store.create("[]", summary="提示词·第2集", creator="tool:drama_shots",
                         gen_params={"episode": 2})

    assert [a.id for a in store.find(type_=AssetType.SCRIPT, episode=2)] == [e2b.id, e2a.id]
    assert store.find(creator="tool:")[0].id == shots.id
    assert store.find(contains="修订")[0].id == e2b.id
    assert store.find(type_=AssetType.SCRIPT, newest_first=False)[0].id == e1.id
    assert store.episodes_done() == [1, 2]
    found = store.episode_assets(2)
    assert found["剧本"].id == e2b.id and found["视频提示词"].id == shots.id


async def test_list_assets默认最新30条并可翻页():
    store = AssetStore()
    for i in range(1, 46):
        _episode(store, i)
    reg, _ = await _content_registry(store)

    r = await reg.invoke("list_assets", {})
    assert r.ok and "匹配 45 份" in r.content and "第 1–30 条" in r.content
    assert "第45集" in r.content and "第1集·" not in r.content, "默认最新在前"
    assert "offset=30" in r.content, "要告诉模型怎么翻页"

    r2 = await reg.invoke("list_assets", {"offset": 30, "limit": 30})
    assert "第 31–45 条" in r2.content and "第1集·" in r2.content

    r3 = await reg.invoke("list_assets", {"type": "script", "episode": 7})
    assert "匹配 1 份" in r3.content and "第7集" in r3.content

    r4 = await reg.invoke("list_assets", {"type": "outline"})
    assert "没有匹配" in r4.content and "换个过滤条件" in r4.content

    assert not (await reg.invoke("list_assets", {"type": "不存在"})).ok


async def test_list_assets截断上限单独放宽():
    """调度器默认 4000 字截断对列表类工具太小 —— 60 条就截了，模型以为剩下的不存在。"""
    store = AssetStore()
    for i in range(1, 121):
        _episode(store, i, title="一个比较长的集标题用来撑字数")
    reg, bus = await _content_registry(store)
    disp = ToolDispatcher(reg, PermissionGate(bus), bus)
    out = await disp.run([ToolCall(id="c1", name="list_assets", arguments='{"limit": 120}')])
    result = out[0][1]
    assert result.ok and len(result.content) > 4000
    assert not result.truncated


async def test_截断时告诉模型截了多少():
    store = AssetStore()
    store.create("x" * 9000, summary="长文")
    reg, bus = await _content_registry(store)
    disp = ToolDispatcher(reg, PermissionGate(bus), bus)
    aid = store.all()[0].id
    call = ToolCall(id="c1", name="read_asset", arguments=json.dumps({"asset_id": aid}))
    out = await disp.run([call])
    result = out[0][1]
    assert result.truncated and "结果已截断：共 9000 字" in result.content


async def test_find_episode():
    store = AssetStore()
    _episode(store, 1)
    _episode(store, 3)
    reg, _ = await _content_registry(store)
    r = await reg.invoke("find_episode", {"episode": 3})
    assert r.ok and "第 3 集" in r.content and "剧本" in r.content and r.asset_ref
    r2 = await reg.invoke("find_episode", {"episode": 2})
    assert "还没有任何产物" in r2.content and "1, 3" in r2.content


async def test_request_review带major进payload():
    store = AssetStore()
    a = store.create("x", summary="a")
    reg, _ = await _content_registry(store)
    r = await reg.invoke(
        "request_review", {"asset_ids": [a.id], "question": "q", "stage": "第3集", "major": True}
    )
    assert r.suspend and r.suspend_payload["major"] is True
    r2 = await reg.invoke("request_review", {"asset_ids": [a.id], "question": "q"})
    assert "major" not in r2.suspend_payload


# ---------------------------------------------------------------- 合规分型


def test_剧本台词不过广告法极限词():
    checker = ComplianceChecker(RULES)
    text = "陆离：你就是最强的守门人，绝对没有第一个人能挡住它。"
    script = checker.check(text, asset_type="script")
    assert not [f for f in script.findings if f.rule == "absolute_claims"]
    assert not [f for f in script.findings if f.rule == "aigc_label"], "剧本是中间产物，不查标识"
    assert not [f for f in script.findings if f.rule == "facts"]

    copy = checker.check(text, asset_type="text")
    assert [f for f in copy.findings if f.rule == "absolute_claims"], "发布文案照查"
    untyped = checker.check(text)
    assert [f for f in untyped.findings if f.rule == "absolute_claims"], "没说类型就全查"


def test_敏感话题对剧本照样提示():
    r = ComplianceChecker(RULES).check("剧情涉及赌博与毒品。", asset_type="script")
    assert [f for f in r.findings if f.rule == "sensitive_topics"]


def test_规则摘要标出生效范围():
    assert "只查 text/outline" in RULES.summary()


async def test_批量机审返回汇总且每份落报告():
    store = AssetStore()
    bus = EventBus()
    ok1 = store.create("本内容由 AI 生成。干净的文案。", creator="model", summary="干净")
    bad = store.create("本内容由 AI 生成。全球首发最强产品。", creator="model", summary="极限词")
    reg = ToolRegistry(bus)
    reg.register(ComplianceFunctions(ComplianceChecker(RULES), store, bus))
    await reg.refresh()

    r = await reg.invoke(
        "check_compliance",
        {"asset_ids": [ok1.id, bad.id, "as_ffffffffff"], "platform": "douyin"},
    )
    assert r.ok
    assert "批量机审（douyin）3 份" in r.content
    assert f"✓ {ok1.id}" in r.content and f"✗ {bad.id}" in r.content
    assert "as_ffffffffff" in r.content and "不存在" in r.content
    reports = store.find(type_=AssetType.REPORT)
    assert len(reports) == 2
    assert {tuple(x.parent_ids) for x in reports} == {(ok1.id,), (bad.id,)}
    ev = [e for e in bus.history if e.type is EventType.COMPLIANCE_CHECKED]
    assert len(ev) == 2

    single = await reg.invoke("check_compliance", {"asset_id": bad.id})
    assert single.ok and "合规机审" in single.content and "出现「最强」" in single.content
    assert not (await reg.invoke("check_compliance", {})).ok


# ---------------------------------------------------------------- 写作子代理


class _Runner:
    """假的子代理运行器：记录契约，按剧本返回。"""

    def __init__(self, texts: list[str] | None = None, fail: bool = False) -> None:
        self.tasks: list[str] = []
        self.texts = list(texts or [])
        self.fail = fail

    async def run(self, defn, task: str, context: str = "") -> SubAgentResult:
        self.tasks.append(task)
        if self.fail:
            return SubAgentResult(ok=False, error="模拟失败")
        n = len(self.tasks)
        # 默认文本满足一集规格（≥960 字、有开场高潮点行），否则会触发扩写重跑
        text = self.texts.pop(0) if self.texts else (
            f"# 第{n}集：自动标题\n> ⚡ 前15秒高潮点：开场即高潮\n\n## 场次一\n\n△ （全景）"
            + "正文" * 500
            + f"\n\n> 🎣 本集钩子：第{n}集的悬念"
        )
        return SubAgentResult(
            ok=True, text=text, iterations=1, cost=0.01, stop_reason="no_tool_calls"
        )


OUTLINE = """第1集：隧道里的呼吸声 —— 主角发现异象 🔥
第2集：工牌 —— 父亲的旧档案 💰
第3集：门 —— 归墟之门初现
"""


def _seed(store: AssetStore) -> tuple[Any, Any, Any]:
    plan = store.create("三幕结构，7 个付费卡点", type_=AssetType.OUTLINE, summary="创作方案")
    chars = store.create("陆离：守门人，沉默。苏晏：记者。", type_=AssetType.OUTLINE,
                         summary="角色档案")
    outline = store.create(OUTLINE, type_=AssetType.OUTLINE, summary="分集目录")
    return plan, chars, outline


def test_目录条目抽取与标题解析():
    entry, prev, nxt = outline_entry(OUTLINE, 2)
    assert entry.startswith("第2集") and "💰" in entry
    assert prev.startswith("第1集") and nxt.startswith("第3集")
    assert outline_entry(OUTLINE, 9) == ("", "", "")
    assert episode_title("# 第2集：工牌\n\n正文", 2) == "工牌"
    assert episode_title("# Episode 2: The Badge\n", 2) == "The Badge"
    assert episode_title("没有标题", 2) == ""


def test_契约只带这一集需要的东西():
    c = build_contract(
        n=2, total=60, entry="第2集：工牌", prev_entry="第1集：呼吸声", next_entry="第3集：门",
        characters="陆离：守门人", plan="三幕", prev_tail="……门缝里透出蓝光。", note="别太血腥",
        mode="domestic",
    )
    assert "写第 2 集（共 60 集）" in c and "第2集：工牌" in c
    assert "陆离：守门人" in c and "蓝光" in c and "别太血腥" in c
    assert "# 第{N}集" in c and "Episode" not in c
    assert "Episode {N}" in build_contract(
        n=1, total=0, entry="", prev_entry="", next_entry="", characters="", plan="",
        prev_tail="", note="", mode="overseas",
    )


async def _episodes_registry(store: AssetStore, runner: _Runner):
    bus = EventBus()
    reg = ToolRegistry(bus)
    reg.register(EpisodeFunctions(runner, store, bus))  # type: ignore[arg-type]
    await reg.refresh()
    return reg, bus


async def test_写一集_落资产并回传钩子():
    store = AssetStore()
    plan, chars, outline = _seed(store)
    runner = _Runner()
    reg, _ = await _episodes_registry(store, runner)

    r = await reg.invoke("drama_write_episode", {"episode": 1, "total": 60})
    assert r.ok, r.error
    a = store.get(r.asset_ref)
    assert a.type is AssetType.SCRIPT and a.gen_params["episode"] == 1
    assert a.summary == "第1集·自动标题" and a.parent_ids == [outline.id]
    assert a.gen_cost == 0.01
    assert "钩子" in r.content and "正文正文正文" not in r.content, "只收 id 和钩子，不收全文"
    task = runner.tasks[0]
    assert "第1集：隧道里的呼吸声" in task and "陆离：守门人" in task and "三幕" in task
    assert "写第 1 集（共 60 集）" in task


async def test_写下一集会带上上一集结尾():
    store = AssetStore()
    _seed(store)
    _episode(store, 1, title="呼吸声")
    runner = _Runner()
    reg, _ = await _episodes_registry(store, runner)
    r = await reg.invoke("drama_write_episode", {"episode": 2})
    assert r.ok
    assert "第1集的悬念" in runner.tasks[0], "上一集的结尾要进契约，前情提要才接得上"
    assert store.get(r.asset_ref).parent_ids[1] == store.episode_assets(1)["剧本"].id


async def test_已有的集默认不重写():
    store = AssetStore()
    _seed(store)
    old = _episode(store, 1)
    runner = _Runner()
    reg, _ = await _episodes_registry(store, runner)
    r = await reg.invoke("drama_write_episode", {"episode": 1})
    assert r.ok and r.asset_ref == old.id and runner.tasks == []
    r2 = await reg.invoke("drama_write_episode", {"episode": 1, "overwrite": True})
    assert r2.ok and r2.asset_ref != old.id


async def test_没有分集目录就报错指路():
    store = AssetStore()
    reg, _ = await _episodes_registry(store, _Runner())
    r = await reg.invoke("drama_write_episode", {"episode": 1})
    assert not r.ok and "分集目录" in r.error


async def test_太短的产出不入库():
    store = AssetStore()
    _seed(store)
    reg, _ = await _episodes_registry(store, _Runner(texts=["太短"]))
    r = await reg.invoke("drama_write_episode", {"episode": 1})
    assert not r.ok and "太短" in r.error
    assert store.episodes_done() == []


async def test_批量连写_跳过已有_失败即停():
    store = AssetStore()
    _seed(store)
    _episode(store, 2, title="已有")
    runner = _Runner()
    reg, bus = await _episodes_registry(store, runner)

    r = await reg.invoke("drama_write_episodes", {"from_episode": 1, "to_episode": 3})
    assert r.ok, r.error
    assert store.episodes_done() == [1, 2, 3]
    assert "第 2 集已有，跳过" in r.content and "写了 2 集" in r.content
    assert len(runner.tasks) == 2
    assert "已有" in runner.tasks[1] or "第2集的悬念" in runner.tasks[1], "第 3 集要承接第 2 集"
    progress = [e for e in bus.history if e.type is EventType.BATCH_PROGRESS]
    assert progress and progress[-1].data["done"] == 3 and progress[-1].data["total"] == 3

    failing = _Runner(fail=True)
    reg2, _ = await _episodes_registry(store, failing)
    r2 = await reg2.invoke("drama_write_episodes", {"from_episode": 4, "to_episode": 6})
    assert not r2.ok and "停在第 4 集" in r2.content
    assert store.episodes_done() == [1, 2, 3]

    assert not (await reg.invoke("drama_write_episodes", {"from_episode": 1, "to_episode": 30})).ok


# ---------------------------------------------------------------- 项目卡


def test_项目卡从资产与状态文件算出来(tmp_path):
    store = AssetStore()
    assert build_project_card(store, tmp_path) == "", "没有短剧痕迹就不 pin"

    plan, chars, outline = _seed(store)
    for n in (1, 2, 3, 5):
        _episode(store, n)
    store.create("[]", summary="提示词", creator="tool:drama_shots", gen_params={"episode": 1})
    (tmp_path / ".drama-state.json").write_text(
        json.dumps({"dramaTitle": "山海守门人", "totalEpisodes": 60, "ethnicity": "chinese",
                    "language": "zh", "currentStep": "episode", "genre": ["悬疑", "科幻"]}),
        encoding="utf-8",
    )
    card = build_project_card(store, tmp_path)
    assert "《山海守门人》" in card and "共 60 集" in card and "悬疑/科幻" in card
    assert f"分集目录 {outline.id}" in card and f"角色档案 {chars.id}" in card
    assert "第 1–3, 5 集（4/60）" in card and "缺：4, 6–60" in card
    assert store.episode_assets(5)["剧本"].id in card
    assert "视频提示词 1 集" in card
    assert "find_episode" in card


def test_项目卡没有状态文件也能用():
    store = AssetStore()
    _seed(store)
    _episode(store, 1)
    card = build_project_card(store, None)
    assert "第 1 集" in card and "共" not in card.split("\n")[1]


# ---------------------------------------------------------------- Agent 装配层（不连网）


async def test_项目卡与召回查询接进Agent(tmp_path, monkeypatch):
    """Agent.create 走真实装配（不发请求）；prepare_turn 后 pre_input 位有项目卡，
    召回查询在输入很短时带上上一轮助手文本。"""
    from aigc_agent.app import Agent

    monkeypatch.setenv("KIMI_API_KEY", "sk-test")
    monkeypatch.setenv("APIMART_API_KEY", "sk-test")
    agent = Agent.create(session_id="p6-test")
    try:
        agent.assets._items.clear()  # noqa: SLF001 — 隔离：不看磁盘上已有的资产
        # 也不往真实资产库写：否则每跑一次全量测试，workspace/assets 就多一份
        # 「第1集·标题」，script_of(1) 会把它当成第 1 集最新版（2026-09-22 实测）
        agent.assets.root = None
        agent.assets.mirror = None
        assert agent.assembler.token_budget > 100_000
        assert agent.assembler.calibration > 1.0
        provider = agent.registry._providers["episodes"]  # noqa: SLF001
        assert "drama_write_episodes" in {m.name for m in await provider.list_tools()}

        _seed(agent.assets)
        _episode(agent.assets, 1)
        await agent.prepare_turn("继续")
        assert CARD_PIN in agent.memory.pins
        assert agent.memory.pins[CARD_PIN].position == "pre_input"

        t = agent.memory.new_turn()
        t.messages = [
            {"role": "user", "content": "写"},
            {"role": "assistant", "content": "《山海守门人》第 1 集已保存"},
        ]
        q = agent._recall_query("继续")  # noqa: SLF001
        assert "山海守门人" in q
        assert agent._recall_query("请把第二集的开头改成夜戏") == "请把第二集的开头改成夜戏"  # noqa: SLF001
    finally:
        await agent.aclose()


# ---------------------------------------------------------------- drama_prose 角色


class _RoleGw:
    def __init__(self, known: set[str]) -> None:
        self.known = known
        self.roles: list[str] = []

    async def chat(self, role, messages, tools=None, **kw):
        if role not in self.known:
            raise KeyError(f"未知角色 {role!r}")
        self.roles.append(role)
        text = "第1集\n\n[夜] [内] [走廊]\n\n陆离：谁在那儿？\n" * 5
        return ModelResponse(text=text, usage=Usage(1, 1))


async def test_写剧本走散文角色_没配时退回drama():
    store = AssetStore()
    gw = _RoleGw({"drama", "drama_prose"})
    fn = DramaFunctions(gw, store)
    r = await fn.invoke("drama_write", {"idea": "山海经"})
    assert r.ok and gw.roles == ["drama_prose"]

    legacy = _RoleGw({"drama"})
    fn2 = DramaFunctions(legacy, store)
    r2 = await fn2.invoke("drama_write", {"idea": "山海经"})
    assert r2.ok and legacy.roles == ["drama"]

"""2026-09-17 复盘后的上下文控制验收。

真实日志里发生过的事，每条对应一个测试：
  · 一轮 12 次迭代把上下文推到 26 万 token，撞上 k3-256k 上限报 401，整轮作废
    → 轮内压缩：旧迭代的工具参数/结果折叠成存根；撞上限压缩后重试一次
  · 估算比真实 token 低四成，18 万的熔断实际 25 万才触发
    → 用真实 usage 校准估算；预检超预算先剔历史轮
  · 重启装回 10 轮 = 22 万 token 冷启动，一次 ¥4.6
    → 快照恢复按 token 上限截
  · 金额超限后 50 多轮每轮只跑一次迭代就停，人说"继续"也没用
    → 有询问器就问一次追加多少，人点头就抬上限继续
  · 同一次迭代提了 9 次 request_review，只有第一个挂起，其余被当成功回填
    → 多余的按失败回填
  · /auto 靠 stage 关键词判大小节点，skill 得叮嘱「stage 必须填剧本」
    → request_review 显式 major 参数优先
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from aigc_agent.capabilities.memory.session import SessionSnapshot
from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.functions.content import ContentFunctions
from aigc_agent.harness.context.assembler import ContextAssembler
from aigc_agent.harness.context.compaction import (
    HARD,
    CompactionPolicy,
    find_fold_marks,
    fold_arguments,
    fold_message,
    fold_turn_messages,
    is_only_fold_mark,
)
from aigc_agent.harness.context.window import ShortTermMemory, Turn, WindowPolicy
from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.execution.loop import LoopRuntime, StopReason
from aigc_agent.harness.model.budget import CostGuard
from aigc_agent.harness.model.gateway import (
    ModelResponse,
    ToolCall,
    Usage,
    classify_model_error,
    estimate_tokens,
)
from aigc_agent.harness.permission.gate import PermissionGate
from aigc_agent.harness.tools.builtin import builtin
from aigc_agent.harness.tools.dispatcher import ToolDispatcher
from aigc_agent.harness.tools.registry import ToolRegistry

_ID = "as_0123456789"
_BIG = "第三集正文。" * 400  # 2400 字


def _tc(cid: str, name: str, args: str = "{}") -> ToolCall:
    return ToolCall(id=cid, name=name, arguments=args)


def _iteration(n: int, *, big: bool = True) -> list[dict[str, Any]]:
    """一次迭代 = 助手带工具调用 + 工具结果。"""
    payload = {"content": _BIG if big else "短", "kind": "script", "episode": n}
    args = json.dumps(payload, ensure_ascii=False)
    fn = {"name": "save_draft", "arguments": args}
    call = {"id": f"c{n}", "type": "function", "function": fn}
    result = f"已存为 {_ID}（script v{n}）：第{n}集\n" + ("正文回显" * 300 if big else "")
    return [
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": f"c{n}", "content": result},
    ]


# ---------------------------------------------------------------- 折叠规则


def test_参数折叠只动长字段且仍是合法JSON():
    raw = json.dumps({"content": _BIG, "kind": "script", "episode": 3, "summary": "第3集·标题"})
    folded = json.loads(fold_arguments(raw))
    assert folded["kind"] == "script" and folded["episode"] == 3
    assert folded["summary"] == "第3集·标题", "短字段要留着，模型才知道那次调用干了什么"
    assert "已折叠" in folded["content"] and len(folded["content"]) < 40


def test_非JSON参数折叠后也是合法JSON():
    assert json.loads(fold_arguments("{坏掉的"))["_folded"]


# ---------------------------------------------------------------- 折叠标记被抄回参数
# 2026-09-21：主模型照着历史里折叠后的样子写参数，12 次 save_draft 的 content 全是
# 「<7xxx 字已折叠>」，12 集分镜存成 12 个 11 字的占位符，工具还报成功。


def test_认得出本文件产出的每一种折叠标记():
    folded = json.loads(fold_arguments(json.dumps({"content": _BIG, "kind": "script"})))
    assert find_fold_marks(folded) == ["content"]
    assert find_fold_marks(json.loads(fold_arguments("{坏掉的"))) == ["_folded"]

    tool_msg = {"role": "tool", "tool_call_id": "c1", "content": f"已存为 {_ID}。" + "x" * 2000}
    tail = fold_message(tool_msg, CompactionPolicy())[0]["content"]
    assert find_fold_marks({"text": "前文" + tail}) == ["text"], "折叠后的工具结果被抄进正文"

    nested = {"shots": [{"desc": "正常"}, {"desc": "<812 字已折叠>"}]}
    assert find_fold_marks(nested) == ["shots[1].desc"]
    # 模型自己编的变体（9-21 drama_assets 真实收到过）
    assert find_fold_marks({"script": "<29000字剧本全文>"}) == ["script"]


def test_夹在正文中间的占位符也算():
    """2026-09-22 补的洞：整串检查只挡得住"整个值就是占位符"。模型拼长正文时会把
    半截真内容和占位符缝在一起，那种放行下去存的是缺了一大块的稿子，还不一定报错。"""
    spliced = "第1集正文……\n<4939 字已折叠>\n第3集正文……"
    assert find_fold_marks({"content": spliced}) == ["content"]
    assert find_fold_marks({"shots": [{"desc": "前半" + "<812 字已折叠>"}]}) == ["shots[0].desc"]


def test_整段是不是占位符():
    assert is_only_fold_mark("<4939 字已折叠>")
    assert is_only_fold_mark("  <29000字剧本全文>  ")
    assert not is_only_fold_mark("第1集正文…<4939 字已折叠>…"), "夹着真内容的不算"
    assert not is_only_fold_mark("正常的一整集剧本")
    assert not is_only_fold_mark("")
    assert not is_only_fold_mark(None)


def test_正常内容不误报():
    assert find_fold_marks({"content": _BIG, "episode": 3, "tags": ["a", "b"]}) == []
    # 正文里提到「已折叠」、或是源码里的 f-string 模板，都不是标记
    prose = "他把信折叠好。上下文里 12 字已折叠的部分要 read_asset 取回"
    src = 'tail = f"…（工具结果 {len(content)} 字已折叠"'
    assert find_fold_marks({"a": prose, "b": src, "c": "<abc 字已折叠>"}) == []


async def test_调度器拒收折叠占位符且不落盘():
    store = AssetStore()
    bus = EventBus()
    registry = ToolRegistry(bus)
    registry.register(ContentFunctions(store))
    await registry.refresh()
    disp = ToolDispatcher(registry, PermissionGate(bus), bus)

    bad = {"content": "<7351 字已折叠>", "kind": "script", "episode": 1, "summary": "分镜·第1集"}
    out = await disp.run([_tc("c1", "save_draft", json.dumps(bad, ensure_ascii=False))])
    r = out[0][1]
    assert not r.ok and "content" in r.error and "read_asset" in r.error
    assert store.all() == [], "占位符不能当正文存盘"

    good = {**bad, "content": _BIG}
    out = await disp.run([_tc("c2", "save_draft", json.dumps(good, ensure_ascii=False))])
    assert out[0][1].ok and len(store.all()) == 1


def test_工具结果折叠保留开头与产物id():
    m = {"role": "tool", "tool_call_id": "c1", "content": f"已存为 {_ID}。" + "x" * 2000}
    fm, changed = fold_message(m, CompactionPolicy())
    assert changed == 1
    assert fm["content"].startswith("已存为 " + _ID)
    assert _ID in fm["content"] and "read_asset" in fm["content"]
    assert m["content"].endswith("x"), "原消息不能被改"


def test_失败结果多留一点():
    m = {"role": "tool", "tool_call_id": "c1", "content": "[工具执行失败] " + "原因" * 600}
    fm, _ = fold_message(m, CompactionPolicy())
    assert fm["content"].startswith("[工具执行失败]") and len(fm["content"]) >= 300


def test_用户消息永远不折():
    msgs = [{"role": "user", "content": "u" * 5000}] + _iteration(1)
    out, n = fold_turn_messages(msgs, keep_tail_iterations=0, policy=CompactionPolicy())
    assert out[0]["content"] == "u" * 5000
    assert n == 2  # 参数 + 结果各折一处


def test_保留最近N次迭代原文():
    msgs = [{"role": "user", "content": "写"}] + _iteration(1) + _iteration(2) + _iteration(3)
    out, n = fold_turn_messages(msgs, keep_tail_iterations=1, policy=CompactionPolicy())
    # 最后一次迭代（c3）原样；c1/c2 折叠
    last_args = out[-2]["tool_calls"][0]["function"]["arguments"]
    assert _BIG in last_args
    first_args = out[1]["tool_calls"][0]["function"]["arguments"]
    assert "已折叠" in first_args
    assert n == 4


def test_折叠是确定性的():
    msgs = [{"role": "user", "content": "写"}] + _iteration(1) + _iteration(2)
    a, _ = fold_turn_messages(msgs, keep_tail_iterations=0, policy=CompactionPolicy())
    b, _ = fold_turn_messages(msgs, keep_tail_iterations=0, policy=CompactionPolicy())
    assert a == b, "同一段消息折出来必须一样，否则历史轮前缀不稳，缓存被折叠本身打穿"


# ---------------------------------------------------------------- 装配层


async def _assembled(asm: ContextAssembler, mem: ShortTermMemory):
    return await asm.assemble(mem, mem.turns[-1])


async def test_小轮不折_大历史轮折叠():
    bus = EventBus()
    mem = ShortTermMemory()
    small = mem.new_turn()
    small.messages = [{"role": "user", "content": "小"}] + _iteration(1, big=False)
    small.tokens = 50
    big = mem.new_turn()
    big.messages = [{"role": "user", "content": "大"}] + _iteration(2) + _iteration(3)
    big.tokens = estimate_tokens(big.messages)
    cur = mem.new_turn()
    cur.messages = [{"role": "user", "content": "当前"}]

    asm = ContextAssembler(bus, compaction=CompactionPolicy(history_turn_tokens=1_000))
    msgs = await _assembled(asm, mem)
    joined = json.dumps(msgs, ensure_ascii=False)
    assert "短" in joined, "小轮原样保留"
    assert _BIG not in joined, "大历史轮的正文该折掉 —— 它在资产库里"
    assert _ID in joined, "折叠后还得看得到产物 id"
    assert asm.last_folded == 4
    assert any(e.type is EventType.CONTEXT_COMPACTED for e in bus.history)


async def test_当前轮涨大后只留最近迭代():
    bus = EventBus()
    mem = ShortTermMemory()
    cur = mem.new_turn()
    cur.messages = [{"role": "user", "content": "写"}]
    cur.messages += _iteration(1) + _iteration(2) + _iteration(3)
    asm = ContextAssembler(
        bus, compaction=CompactionPolicy(current_turn_tokens=500, keep_recent_iterations=1)
    )
    msgs = await _assembled(asm, mem)
    args = [
        tc["function"]["arguments"]
        for m in msgs
        if m.get("role") == "assistant"
        for tc in m.get("tool_calls") or []
    ]
    assert "已折叠" in args[0] and "已折叠" in args[1]
    assert _BIG in args[2], "最近一次迭代必须原文"


async def test_未到阈值的当前轮一字不动():
    bus = EventBus()
    mem = ShortTermMemory()
    cur = mem.new_turn()
    cur.messages = [{"role": "user", "content": "写"}] + _iteration(1)
    msgs = await _assembled(ContextAssembler(bus), mem)
    assert msgs[-1] == cur.messages[-1] and msgs[-2] == cur.messages[-2]


async def test_估算按真实usage校准():
    bus = EventBus()
    mem = ShortTermMemory()
    cur = mem.new_turn()
    cur.messages = [{"role": "user", "content": "中文内容" * 100}]
    asm = ContextAssembler(bus)
    await _assembled(asm, mem)
    raw = asm.last_estimate
    asm.observe(int(raw * 1.5))  # 真实值比估算高一半
    assert asm.calibration > 1.2
    assert asm.calibrated(raw) > raw
    asm.observe(0)  # 没有 usage 不动
    assert asm.calibration > 1.2


async def test_历史轮折不折的决定不随校准漂移():
    """阈值附近的轮次来回翻转会让前缀不稳定，缓存白白击穿。"""
    bus = EventBus()
    mem = ShortTermMemory()
    old = mem.new_turn()
    old.messages = [{"role": "user", "content": "旧"}] + _iteration(1)
    old.tokens = 900
    mem.new_turn().messages = [{"role": "user", "content": "当前"}]
    asm = ContextAssembler(bus, compaction=CompactionPolicy(history_turn_tokens=1_000))
    first = await _assembled(asm, mem)
    asm.calibration = 2.0  # 现在 900×2 > 1000，但决定已经记住了
    second = await _assembled(asm, mem)
    assert first == second


async def test_应急档折掉一切只留最后一次迭代():
    bus = EventBus()
    mem = ShortTermMemory()
    old = mem.new_turn()
    old.messages = [{"role": "user", "content": "旧"}] + _iteration(1, big=False)
    old.tokens = 10
    cur = mem.new_turn()
    cur.messages = [{"role": "user", "content": "写"}] + _iteration(2) + _iteration(3)
    asm = ContextAssembler(bus)
    asm.shrink = True
    msgs = await _assembled(asm, mem)
    joined = json.dumps(msgs, ensure_ascii=False)
    assert joined.count(_BIG) == 1, "只有最后一次迭代留原文"
    assert HARD.keep_recent_iterations == 1


# ---------------------------------------------------------------- Loop 预检与错误路径


class _Gw:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(self, role, messages, tools=None, **kw):
        self.calls.append(messages)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return item


async def _loop(gw: _Gw, *, token_budget: int = 0, guard=None, budget_asker=None, memory=None):
    bus = EventBus()
    registry = ToolRegistry(bus)
    registry.register(builtin)
    registry.register(ContentFunctions(AssetStore()))
    await registry.refresh()
    mem = memory or ShortTermMemory()
    loop = LoopRuntime(
        gateway=gw,  # type: ignore[arg-type]
        registry=registry,
        dispatcher=ToolDispatcher(registry, PermissionGate(bus, guard=guard), bus),
        assembler=ContextAssembler(bus, token_budget=token_budget),
        memory=mem,
        bus=bus,
        guard=guard,
        budget_asker=budget_asker,
    )
    return bus, mem, loop


async def test_预检超预算先剔历史轮再发():
    gw = _Gw([ModelResponse(text="好", usage=Usage(50, 5))])
    mem = ShortTermMemory()
    for i in range(4):
        t = mem.new_turn()
        t.messages = [{"role": "user", "content": f"第{i}轮" + "填充" * 400}]
        t.tokens = 900
    evicted: list[list[Turn]] = []
    mem.on_evict = evicted.append
    bus, mem, loop = await _loop(gw, token_budget=2_000, memory=mem)

    r = await loop.run_turn("当前")
    assert r.stop_reason is StopReason.NO_TOOL_CALLS
    sent = json.dumps(gw.calls[0], ensure_ascii=False)
    assert "第0轮" not in sent and "第1轮" not in sent, "最老的轮先剔"
    assert "当前" in sent
    assert evicted and evicted[0][0].index == 0, "剔掉的轮要交给记忆提取"
    ev = [e for e in bus.history if e.type is EventType.WINDOW_EVICT]
    assert ev and ev[0].data.get("reason") == "token_budget"


async def test_撞上下文上限压缩后重试一次():
    class _Overflow(Exception):
        status_code = 401

    gw = _Gw(
        [
            _Overflow("Error code: 401 - k3-256k supports only 256K context."),
            ModelResponse(text="重试成功", usage=Usage(10, 5)),
        ]
    )
    bus, mem, loop = await _loop(gw)
    r = await loop.run_turn("继续")
    assert r.stop_reason is StopReason.NO_TOOL_CALLS and "重试成功" in r.text
    assert len(gw.calls) == 2
    assert loop.assembler.shrink is True, "本轮剩下的迭代继续用应急档"
    warnings = [e.data.get("message", "") for e in bus.history if e.type is EventType.WARNING]
    assert any("压缩" in w for w in warnings)


async def test_撞两次就交回给人():
    class _Overflow(Exception):
        status_code = 401

    err = _Overflow("k3-256k supports only 256K context.")
    gw = _Gw([err, err, err])
    _, _, loop = await _loop(gw)
    r = await loop.run_turn("继续")
    assert r.stop_reason is StopReason.ERROR
    assert "上下文上限" in r.text
    assert len(gw.calls) == 2


async def test_额度用尽给人能看懂的提示():
    class _Denied(Exception):
        status_code = 403

    gw = _Gw([_Denied("You've reached your 5-hour usage limit.")])
    _, _, loop = await _loop(gw)
    r = await loop.run_turn("继续")
    assert r.stop_reason is StopReason.ERROR
    assert "额度" in r.text and "继续" in r.text


def test_错误分类():
    class _E(Exception):
        def __init__(self, msg, status=None):
            super().__init__(msg)
            self.status_code = status

    kind = classify_model_error
    assert kind(_E("k3-256k supports only 256K context.", 401))[0] == "context_overflow"
    assert kind(_E("maximum context length is 262144 tokens", 400))[0] == "context_overflow"
    assert classify_model_error(_E("You've reached your 5-hour usage limit", 403))[0] == "quota"
    assert classify_model_error(_E("insufficient balance", 402))[0] == "quota"
    assert classify_model_error(_E("connection reset", 500))[0] == "other"


async def test_新轮清掉应急档():
    class _Overflow(Exception):
        status_code = 401

    gw = _Gw([_Overflow("supports only 256K context"), ModelResponse(text="ok", usage=Usage(1, 1))])
    _, _, loop = await _loop(gw)
    await loop.run_turn("一")
    assert loop.assembler.shrink is True
    gw.script = [ModelResponse(text="ok", usage=Usage(1, 1))]
    await loop.run_turn("二")
    assert loop.assembler.shrink is False


async def test_真实usage写回窗口():
    gw = _Gw([ModelResponse(text="好", usage=Usage(1234, 5))])
    _, mem, loop = await _loop(gw)
    await loop.run_turn("嗨")
    assert mem.observed_prompt_tokens == 1234


# ---------------------------------------------------------------- 金额护栏出口


class _GwLook:
    def __init__(self) -> None:
        self.n = 0

    async def chat(self, role, messages, tools=None, **kw):
        self.n += 1
        if self.n <= 2:
            return ModelResponse(
                tool_calls=[ToolCall(id=f"c{self.n}", name="now", arguments="{}")],
                usage=Usage(5, 5),
            )
        return ModelResponse(text="完成", usage=Usage(5, 5))


async def test_金额超限时问人_追加后继续():
    asked: list[str] = []

    async def ask(reason: str) -> float:
        asked.append(reason)
        return 50.0

    guard = CostGuard(money_limit=1.0)
    bus = EventBus()
    guard.attach(bus)
    registry = ToolRegistry(bus)
    registry.register(builtin)
    await registry.refresh()
    loop = LoopRuntime(
        gateway=_GwLook(),  # type: ignore[arg-type]
        registry=registry,
        dispatcher=ToolDispatcher(registry, PermissionGate(bus, guard=guard), bus),
        assembler=ContextAssembler(bus),
        memory=ShortTermMemory(),
        bus=bus,
        guard=guard,
        budget_asker=ask,
    )
    await bus.emit(EventType.COST, modality="video", cost=1.5)

    r = await loop.run_turn("继续")
    assert r.stop_reason is StopReason.NO_TOOL_CALLS, "人追加了额度就该跑完"
    assert len(asked) == 1 and "上限" in asked[0], "只问一次，之后上限已抬"
    assert guard.extra_money == 50.0
    assert guard.check().ok


async def test_人不追加就停并带出口提示():
    async def no(reason: str) -> float:
        return 0.0

    guard = CostGuard(money_limit=1.0)
    bus = EventBus()
    guard.attach(bus)
    registry = ToolRegistry(bus)
    registry.register(builtin)
    await registry.refresh()
    loop = LoopRuntime(
        gateway=_GwLook(),  # type: ignore[arg-type]
        registry=registry,
        dispatcher=ToolDispatcher(registry, PermissionGate(bus, guard=guard), bus),
        assembler=ContextAssembler(bus),
        memory=ShortTermMemory(),
        bus=bus,
        guard=guard,
        budget_asker=no,
        budget_hint="要继续：/budget allow 50",
    )
    await bus.emit(EventType.COST, modality="video", cost=1.5)
    r = await loop.run_turn("继续")
    assert r.stop_reason is StopReason.BUDGET_EXCEEDED
    assert "/budget allow" in r.text


def test_临时追加抬三级上限_reset清零():
    g = CostGuard(money_limit=1.0)
    g.record("text", 1.2)
    assert not g.check().ok
    g.allow_more_money(5)
    assert g.check().ok
    assert "临时追加 ¥5" in g.brief()
    g.reset()
    assert g.extra_money == 0.0


# ---------------------------------------------------------------- 人审：多余的请求与 major


async def test_同一迭代第二个人审按失败回填():
    bus = EventBus()
    assets = AssetStore()
    a = assets.create("x", summary="a")
    b = assets.create("y", summary="b")
    registry = ToolRegistry(bus)
    registry.register(ContentFunctions(assets))
    await registry.refresh()
    gw = _Gw(
        [
            ModelResponse(
                tool_calls=[
                    _tc("c1", "request_review", json.dumps({"asset_ids": [a.id], "question": "1"})),
                    _tc("c2", "request_review", json.dumps({"asset_ids": [b.id], "question": "2"})),
                ],
                usage=Usage(1, 1),
            ),
            ModelResponse(text="ok", usage=Usage(1, 1)),
        ]
    )
    loop = LoopRuntime(
        gateway=gw,  # type: ignore[arg-type]
        registry=registry,
        dispatcher=ToolDispatcher(registry, PermissionGate(bus), bus),
        assembler=ContextAssembler(bus),
        memory=ShortTermMemory(),
        bus=bus,
    )
    r = await loop.run_turn("审")
    assert r.stop_reason is StopReason.AWAITING_REVIEW
    assert loop.pending_review["call_id"] == "c1"
    extra = [m for m in r.turn.messages if m.get("tool_call_id") == "c2"]
    assert extra and "只能提一次人审" in extra[0]["content"]
    r2 = await loop.resume_turn("adopt")
    assert r2.stop_reason is StopReason.NO_TOOL_CALLS


@pytest.mark.parametrize(
    ("stage", "major", "suspended"),
    [
        ("第3集", True, True),  # 显式 major 压过关键词
        ("剧本", False, False),  # 显式 minor 压过关键词
        ("剧本", None, True),  # 没给就按关键词
        ("第3集", None, False),
    ],
)
async def test_auto模式显式major优先(stage, major, suspended):
    bus = EventBus()
    assets = AssetStore()
    a = assets.create("x", summary="a")
    registry = ToolRegistry(bus)
    registry.register(ContentFunctions(assets))
    await registry.refresh()
    args: dict[str, Any] = {"asset_ids": [a.id], "question": "q", "stage": stage}
    if major is not None:
        args["major"] = major
    gw = _Gw(
        [
            ModelResponse(
                tool_calls=[_tc("c1", "request_review", json.dumps(args))], usage=Usage(1, 1)
            ),
            ModelResponse(text="继续", usage=Usage(1, 1)),
        ]
    )
    loop = LoopRuntime(
        gateway=gw,  # type: ignore[arg-type]
        registry=registry,
        dispatcher=ToolDispatcher(registry, PermissionGate(bus), bus),
        assembler=ContextAssembler(bus),
        memory=ShortTermMemory(),
        bus=bus,
    )
    loop.auto_review = True
    r = await loop.run_turn("干活")
    assert (r.stop_reason is StopReason.AWAITING_REVIEW) is suspended


# ---------------------------------------------------------------- 快照恢复


def test_快照恢复按token上限从新往旧截(tmp_path):
    m = ShortTermMemory()
    for i in range(6):
        t = m.new_turn()
        t.messages = [
            {"role": "user", "content": f"第{i}句"},
            {"role": "assistant", "content": "回"},
        ]
        t.tokens = 10_000
    SessionSnapshot(tmp_path, "s").save(m)

    fresh = ShortTermMemory(policy=WindowPolicy(window_turns=10))
    n = SessionSnapshot(tmp_path, "s").load_into(fresh, max_tokens=25_000)
    assert n == 2, "2 轮 = 2 万，再加一轮就 3 万超了"
    assert [t.user_text for t in fresh.turns] == ["第4句", "第5句"]
    assert fresh.new_turn().index == 6

    fresh = ShortTermMemory()
    assert SessionSnapshot(tmp_path, "s").load_into(fresh, max_tokens=1) == 1, "至少留一轮"

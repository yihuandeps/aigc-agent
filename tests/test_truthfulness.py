"""缺口 F（2026-09-23 审查）：模型读得全，汇报得真。

真实日志里发生过的事：
  · 回复被内容安全过滤拦下（finish_reason=content_filter），当成正常完成，空回复就那么过去了
  · 长工具参数被输出上限截断，工具只报「JSON 不合法」，模型原样重试到撞迭代上限
  · 工具报错之后模型仍说「已完成」—— 人看不到哪几步没成
  · 拆分镜收到 11 字的「测试」照样调模型，118 秒后编出另一部剧的 15 段分镜
  · read_asset / 参考文档被截到 4000 字，截断提示又指回同一个工具，后半截永远读不到
  · 工具报错里的方括号被 rich 当成标记，[/] 直接把 chat 带崩
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

from aigc_agent.capabilities.skill_hub import SkillHub
from aigc_agent.capabilities.skill_hub.functions import MAX_REF_CHARS, SkillFunctions
from aigc_agent.capabilities.subagents import SubAgentDef, SubAgentRunner
from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions.content import READ_PAGE, ContentFunctions, page_of
from aigc_agent.domain.functions.drama import (
    DramaFunctions,
    _finish_problem,
    _script_problem,
)
from aigc_agent.harness.context.assembler import ContextAssembler
from aigc_agent.harness.context.window import ShortTermMemory, WindowPolicy
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.execution.loop import LoopRuntime, StopReason
from aigc_agent.harness.model.gateway import ModelResponse, ToolCall
from aigc_agent.harness.permission.gate import PermissionGate
from aigc_agent.harness.tools.builtin import builtin
from aigc_agent.harness.tools.dispatcher import ToolDispatcher
from aigc_agent.harness.tools.registry import ToolRegistry
from aigc_agent.interfaces.cli import main as cli

_SCRIPT = (
    "【第1集】\n场景一 内景 客栈 夜\n"
    + "林秋推门进来，雨水顺着斗笠往下淌。掌柜抬头看了她一眼，手里的算盘停了。\n" * 12
)


class ScriptedGateway:
    def __init__(self, script: list[ModelResponse]) -> None:
        self.script = script
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(self, role, messages, tools=None, **kw) -> ModelResponse:
        self.calls.append(messages)
        return self.script[min(len(self.calls) - 1, len(self.script) - 1)]


def _tc(cid: str, name: str, args: str = "{}") -> ToolCall:
    return ToolCall(id=cid, name=name, arguments=args)


async def _loop(script: list[ModelResponse]) -> LoopRuntime:
    bus = EventBus(session_id="test")
    registry = ToolRegistry(bus)
    registry.register(builtin)
    await registry.refresh()
    return LoopRuntime(
        gateway=ScriptedGateway(script),  # type: ignore[arg-type]
        registry=registry,
        dispatcher=ToolDispatcher(registry, PermissionGate(bus), bus, timeout=10),
        assembler=ContextAssembler(bus),
        memory=ShortTermMemory(policy=WindowPolicy()),
        bus=bus,
    )


# ---------------------------------------------------------------- 主循环：finish_reason


async def test_内容过滤不当成正常完成():
    loop = await _loop(
        [ModelResponse(text="", tool_calls=[_tc("c1", "now")], finish_reason="content_filter")]
    )
    r = await loop.run_turn("写一段")
    assert r.stop_reason is StopReason.CONTENT_FILTER
    assert "内容安全过滤" in r.text
    # 被拦的半截回复带着工具调用，进历史就是「有调用没结果」，下一次请求直接 400
    assert not any(m.get("tool_calls") for m in r.turn.messages)


async def test_正文被截断要说出来():
    loop = await _loop([ModelResponse(text="第一段……第二", finish_reason="length")])
    r = await loop.run_turn("写")
    assert r.stop_reason is StopReason.NO_TOOL_CALLS
    assert r.text.startswith("第一段") and "被截断" in r.text


async def test_参数被截断时告诉模型真正的原因():
    loop = await _loop(
        [
            ModelResponse(tool_calls=[_tc("c1", "calc", '{"expression": "1+')],
                          finish_reason="length"),
            ModelResponse(text="好"),
        ]
    )
    r = await loop.run_turn("算")
    tool_msg = next(m for m in r.turn.messages if m["role"] == "tool")
    assert "不是合法 JSON" in tool_msg["content"]
    assert "超过了模型单次输出上限" in tool_msg["content"] and "fs_write" in tool_msg["content"]


async def test_失败的工具调用汇总给人看():
    loop = await _loop(
        [
            ModelResponse(tool_calls=[_tc("c1", "calc", '{"expression": "1/0"}'),
                                      _tc("c2", "now")]),
            ModelResponse(text="全部完成了"),  # 模型嘴上说完成
        ]
    )
    r = await loop.run_turn("算")
    assert r.text == "全部完成了"
    assert len(r.tool_failures) == 1 and r.tool_failures[0].startswith("calc：")

    # 下一轮重新计数
    loop.gateway.script = [ModelResponse(text="好")]  # type: ignore[attr-defined]
    loop.gateway.calls.clear()  # type: ignore[attr-defined]
    r2 = await loop.run_turn("再来")
    assert r2.tool_failures == []


def test_cli把失败列出来且不解析方括号(monkeypatch: pytest.MonkeyPatch):
    out = Console(record=True, width=120)
    monkeypatch.setattr(cli, "console", out)
    result = SimpleNamespace(
        text="已完成 [/] [/budget allow 50]",
        iterations=2,
        stop_reason=StopReason.NO_TOOL_CALLS,
        cost=None,
        tool_failures=["drama_render_shots：[bold]第2段[/] 超时"],
    )
    cli._print_result(result)  # type: ignore[arg-type]
    text = out.export_text()
    assert "[/budget allow 50]" in text, "模型回复原样显示"
    assert "1 次工具调用失败" in text and "[bold]第2段[/]" in text


def test_brief已转义():
    assert cli._brief("[/] 参数") == "\\[/] 参数"
    Console(file=None).render_str(f"[dim]{cli._brief('[/red]x')}[/]")  # 不抛 MarkupError


# ---------------------------------------------------------------- 子代理


async def test_子代理跑满轮次或被过滤不算产出():
    ok = SubAgentRunner._collect(
        SubAgentDef(name="w", system_prompt="x"), "半截", 5, None, StopReason.MAX_ITERATIONS
    )
    assert not ok.ok and "5 轮" in (ok.error or "")
    ok2 = SubAgentRunner._collect(
        SubAgentDef(name="w", system_prompt="x"), "", 1, None, StopReason.CONTENT_FILTER
    )
    assert not ok2.ok and "内容安全过滤" in (ok2.error or "")


# ---------------------------------------------------------------- 拆解入参与截断


def test_finish_reason分类():
    assert "截断" in _finish_problem(SimpleNamespace(finish_reason="length"), "分镜脚本")
    assert "内容安全过滤" in _finish_problem(
        SimpleNamespace(finish_reason="content_filter"), "分镜脚本"
    )
    assert _finish_problem(SimpleNamespace(finish_reason="stop"), "x") == ""
    assert _finish_problem(SimpleNamespace(), "x") == ""


def test_剧本入参像不像剧本():
    assert "没有剧本内容" in _script_problem("")
    assert "只有 2 个字" in _script_problem("测试")
    assert "折叠" in _script_problem("第1集\n<12345 字已折叠>\n" + "正文" * 200)
    assert "占位" in _script_problem("第1集 同上\n" + "对白。" * 100)
    assert _script_problem(_SCRIPT) == ""


class _NoCallGateway:
    def __init__(self, resp: ModelResponse | None = None) -> None:
        self.resp = resp
        self.calls = 0

    async def chat(self, role, messages, tools=None, **kw):
        self.calls += 1
        assert self.resp is not None, "入参不合格就不该调模型"
        return self.resp


async def test_拆分镜不拿空壳去调模型():
    store = AssetStore()
    gw = _NoCallGateway()
    fns = DramaFunctions(gw, store)
    r = await fns.invoke(
        "drama_storyboard", {"script": "测试", "ethnicity": "chinese", "language": "zh"}
    )
    assert not r.ok and r.meta.get("charged") is False and gw.calls == 0

    r2 = await fns.invoke(
        "drama_storyboard", {"script_id": "as_nope", "ethnicity": "chinese", "language": "zh"}
    )
    assert not r2.ok and "取不到" in (r2.error or "") and gw.calls == 0


async def test_拆分镜认剧本id且截断时不存():
    store = AssetStore()
    sid = store.create(_SCRIPT, type_=AssetType.SCRIPT, summary="第1集", creator="model").id
    gw = _NoCallGateway(ModelResponse(text='[{"episode": 1, "sce', finish_reason="length"))
    fns = DramaFunctions(gw, store)
    before = len(store.find(type_=AssetType.STORYBOARD))
    r = await fns.invoke(
        "drama_storyboard", {"script_id": sid, "ethnicity": "chinese", "language": "zh"}
    )
    assert gw.calls == 1
    assert not r.ok and "截断" in (r.error or "") and "不要原样重试" in (r.error or "")
    assert len(store.find(type_=AssetType.STORYBOARD)) == before, "半截的分镜不落库"


# ---------------------------------------------------------------- 存稿拦空壳


async def test_存稿拒收占位空壳():
    store = AssetStore()
    reg = ToolRegistry(EventBus())
    reg.register(ContentFunctions(store))
    await reg.refresh()
    r = await reg.invoke(
        "save_draft",
        {"content": "第3集剧本（同上，此处省略）", "kind": "script", "summary": "第3集"},
    )
    assert not r.ok and "不是正文" in (r.error or "")
    # 长正文里偶尔出现「同上」不误伤
    long_text = "他说：同上。" + "正文内容。" * 80
    r2 = await reg.invoke("save_draft", {"content": long_text, "kind": "script", "summary": "x"})
    assert r2.ok


def test_revise也拦折叠占位符():
    store = AssetStore()
    base = store.create("原稿" * 50, type_=AssetType.SCRIPT, summary="v1")
    with pytest.raises(ValueError, match="占位符"):
        store.revise(base.id, "<3210 字已折叠>", summary="v2")


# ---------------------------------------------------------------- 分页读


def test_page_of():
    text = "字" * (READ_PAGE * 2 + 10)
    p1 = page_of(text, 0, 'read_asset(asset_id="a"')
    assert f"offset={READ_PAGE}" in p1 and "还有" in p1
    p3 = page_of(text, READ_PAGE * 2, 'read_asset(asset_id="a"')
    assert "[读完了]" in p3 and "还有" not in p3
    assert page_of("短", 0, "x") == "短", "一页读得完就原样给"
    assert "超过末尾" in page_of("短", 99, "x"), "offset 越界不抛，说清楚"


async def test_read_asset能读全长资产():
    store = AssetStore()
    body = "".join(f"第{i}行内容。\n" for i in range(4000))
    aid = store.create(body, type_=AssetType.SCRIPT, summary="长剧本").id
    bus = EventBus()
    reg = ToolRegistry(bus)
    reg.register(ContentFunctions(store))
    await reg.refresh()
    disp = ToolDispatcher(reg, PermissionGate(bus), bus, timeout=10)

    got, offset = "", 0
    for i in range(10):
        args = f'{{"asset_id": "{aid}", "offset": {offset}}}'
        ((_, r),) = await disp.run([_tc(f"c{i}", "read_asset", args)])
        assert r.ok and "结果已截断" not in r.content, "一页不能再被调度器截"
        page = r.content
        if page.startswith("[第"):
            page = page.split("\n", 1)[1]
        if "\n\n[还有" in page:
            page, tail = page.rsplit("\n\n[还有", 1)
            got += page
            offset = int(tail.split("offset=")[1].split(")")[0])
            continue
        got += page.removesuffix("\n\n[读完了]")
        break
    assert got == body


def _big_skill(tmp: Path) -> Path:
    d = tmp / "skills" / "big"
    (d / "references").mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\nname: big\ndescription: 大参考\nscope: content_type\napplies_to: [短剧]\n"
        "stage: []\npriority: 50\nversion: 1.0.0\nowner: 测试\nstatus: active\n---\n\n"
        "# big\n\n正文\n",
        encoding="utf-8",
    )
    (d / "references" / "long.md").write_text(
        "甲" * MAX_REF_CHARS + "乙" * 500 + "结尾标记", encoding="utf-8"
    )
    return tmp / "skills"


async def test_参考文档分页(tmp_path: Path):
    hub = SkillHub(_big_skill(tmp_path))
    hub.load()
    bus = EventBus()
    reg = ToolRegistry(bus)
    reg.register(SkillFunctions(hub))
    await reg.refresh()
    disp = ToolDispatcher(reg, PermissionGate(bus), bus, timeout=10)

    ((_, r1),) = await disp.run(
        [_tc("c1", "load_skill_reference", '{"skill": "big", "name": "long"}')]
    )
    assert "结果已截断" not in r1.content and f"offset={MAX_REF_CHARS}" in r1.content
    assert "结尾标记" not in r1.content
    (_, r2), = await disp.run(
        [_tc("c2", "load_skill_reference",
             f'{{"skill": "big", "name": "long", "offset": {MAX_REF_CHARS}}}')]
    )
    assert "结尾标记" in r2.content and "还有" not in r2.content


# ---------------------------------------------------------------- 人审面板


def test_人审候选显示媒体的本地文件(tmp_path: Path):
    store = AssetStore()
    png = tmp_path / "a.png"
    png.write_bytes(b"\x89PNG")
    img = store.create("", type_=AssetType.IMAGE, summary="主角定妆", creator="model:x",
                       gen_params={"local": str(png)})
    img.uri = "https://cdn/x.png"
    store.put(img)
    body, local = cli._candidate_view(SimpleNamespace(assets=store), img)  # type: ignore[arg-type]
    assert local == png and str(png) in body and "https://cdn/x.png" in body

    txt = store.create("字" * 3000, type_=AssetType.SCRIPT, summary="剧本")
    body2, local2 = cli._candidate_view(SimpleNamespace(assets=store), txt)  # type: ignore[arg-type]
    assert local2 is None and "共 3000 字" in body2 and len(body2) < 1700

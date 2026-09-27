"""2026-09-26 用户拍板的三条（docs/待拍板-2026-09-26.md 一·1–3）。

  1. 集长：加 /length auto（集长跟剧本走：不查总时长，只查台词念不念得完、单镜 ≤3 秒、开场
     高潮点）；集长只剩一个来源 —— 项目规格（/length，或人采纳的创作方案里写的每集时长），
     drama_write 不再收模型自己传的 minutes。
  2. 剧本自带的上屏字（【字幕：灵山】【片尾字幕：剩余花瓣 9】）：分镜 / 提示词这一步抽成叠字
     清单（screen_text），生成的画面里不写字，拼完成片后用 overlay_text 叠。
  3. 花钱的确认：渲参考图、渲每一集之前整批报价（含质检重生成的最坏值），确认一次；批内
     超出额度不再逐个问，中途被问到答 N 整批停。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aigc_agent.capabilities.memory.session import SessionSnapshot
from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.drama.cards import card_loss, find_cards, missing_cards, strip_cards
from aigc_agent.domain.drama.format import (
    HOOK_MARK,
    EpisodeFormat,
    card_lines,
    check_script,
    check_shots,
    check_storyboard,
    place_cards,
    plan_length,
    shots_rules,
    storyboard_problems,
    storyboard_rules,
)
from aigc_agent.domain.drama.models import Episode, ShotPrompt, as_dict
from aigc_agent.domain.drama.parse import parse_shots
from aigc_agent.domain.functions.drama import DramaFunctions, _length_hint, _no_text_name
from aigc_agent.domain.pipeline.episode_pipeline import EpisodePipeline
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.model.budget import CostGuard
from aigc_agent.harness.permission.gate import BatchPass, PermissionGate, batch_scope
from aigc_agent.harness.tools.provider import PermissionLevel, ToolMeta
from tests.test_review_0924 import _ChunkGateway

AUTO = EpisodeFormat(follow_script=True)


def _board(n: int = 200, secs: float = 3.0, line: str = "“一句台词”") -> Episode:
    rows = ["[夜] [内] [禅房]"]
    for i in range(1, n + 1):
        mark = HOOK_MARK if i == 1 else ""
        rows.append(f"{mark}[近景/固定/{secs:g}s] 唐僧第{i}个动作：{line}")
    return Episode(1, "第1集", "\n".join(rows))


# ================================================================ 1. 集长跟剧本走


def test_跟剧本走_不查总时长_只查台词念不念得完():
    ep = _board(200)  # 600 秒，离 4 分钟、20 分钟都很远
    assert not check_storyboard(ep, AUTO), "跟剧本走：总时长不是问题"
    twenty = check_storyboard(ep, EpisodeFormat(minutes=20))
    assert any("应在" in p for p in twenty), "数字集长照旧是目标值（对照）"
    long_line = "“" + "字" * 40 + "”"
    rushed = check_storyboard(_board(30, 3.0, long_line), AUTO)  # 每镜 3 秒念 40 字
    assert any("台词被压快" in p and "至少放到" in p for p in rushed)


def test_跟剧本走_规则文本和检查都不按集长凑():
    sb = storyboard_rules(AUTO)
    assert "跟着剧本走" in sb and "≈ 240 秒" not in sb
    assert "【字幕：…】" in sb, "分镜规则要求照抄上屏字标记"
    sh = shots_rules(AUTO)
    assert "镜头时长的合计" in sh and "画面里不许出现任何文字" in sh
    shots = [ShotPrompt("[第1集-1场]", "1-5", "15s", "x", hook=True, cuts=[3] * 5)]
    assert not any("总时长" in p for p in check_shots(shots, AUTO))
    assert any("总时长" in p for p in check_shots(shots, EpisodeFormat()))
    assert check_script("短", AUTO) == [check_script("短", EpisodeFormat())[-1]], "只查高潮点行"
    assert _length_hint([_board(200)], AUTO) == ""
    assert "/length auto" in _length_hint([_board(200)], EpisodeFormat())


def test_创作方案里写的每集时长():
    assert plan_length("共 40 集，每集 8 分钟，5 集一批") == (8.0, False)
    assert plan_length("每集时长：约六分钟") == (6.0, False)
    assert plan_length("单集跟剧本走，不凑时长") == (0.0, True)
    assert plan_length("每集改成 6 分钟吧") == (6.0, False)
    assert plan_length("8 分钟一集，共 12 集") == (8.0, False)
    assert plan_length("每集 45 分钟") == (0.0, False), "1–30 分钟以外不认"
    assert plan_length("随便聊聊") == (0.0, False)
    assert plan_length("第一集 3 分钟的预告，5 集一批") == (0.0, False), "「第一集」不是每集时长"


def test_跟剧本走存进项目快照_重启沿用(tmp_path: Path):
    snap = SessionSnapshot(tmp_path, "s")
    snap.set_episode_minutes(0, auto=True)
    again = SessionSnapshot(tmp_path, "s")
    assert again.episode_auto is True and again.episode_minutes == 0.0
    again.set_episode_minutes(8)
    assert SessionSnapshot(tmp_path, "s").episode_auto is False


def test_人采纳的创作方案里写的每集时长_就是项目集长():
    from aigc_agent.app import Agent
    from aigc_agent.harness.events.bus import Event, EventType

    calls: list[tuple[Any, bool]] = []

    class _Stub:
        episode_fmt = EpisodeFormat()

        def set_episode_minutes(self, minutes: Any, persist: bool = True,
                                auto: bool = False) -> None:
            calls.append((minutes, auto))

        def _length_note(self, what: str) -> None: ...

    stub = _Stub()

    def run(question: str, decision: str = "adopt", by: str = "human", reason: str = "",
            node: str = "剧本") -> None:
        Agent._on_plan_review(stub, Event(type=EventType.CHECKPOINT_REACHED,  # type: ignore[arg-type]
                                          data={"stage": node, "question": question}))
        Agent._on_plan_review(stub, Event(type=EventType.CHECKPOINT_DECIDED,  # type: ignore[arg-type]
                                          data={"node": node, "decision": decision,
                                                "decided_by": by, "reason": reason}))

    run("共 20 集，每集 8 分钟，5 集一批")
    assert calls == [(8.0, False)]
    run("共 20 集，每集 8 分钟", reason="每集改成 6 分钟")
    assert calls[-1] == (6.0, False), "采纳附言里另写了时长，以附言为准"
    run("共 20 集，单集跟剧本走")
    assert calls[-1] == (None, True)
    n = len(calls)
    run("共 20 集，每集 8 分钟", decision="revise")
    run("共 20 集，每集 8 分钟", by="auto")
    run("第 3 集，每集 8 分钟", node="剧本第3集")
    assert len(calls) == n, "打回、/auto 自动采纳、别的环节都不改集长"


async def test_drama_write不收模型自己传的集长():
    seen: list[str] = []

    class _Gw:
        async def chat(self, role: str, messages: list[dict[str, Any]], **_: Any) -> Any:
            seen.append(messages[-1]["content"])
            body = "第1集\n\n[夜] [内] [禅房]\n\n唐僧：施主请回。\n\n" * 5
            return SimpleNamespace(text=body, finish_reason="stop")

    fns = DramaFunctions(_Gw(), AssetStore(), fmt=EpisodeFormat(minutes=8))
    assert "minutes" not in fns._specs["drama_write"].parameters["properties"]
    r = await fns._fn_drama_write("取经路上的一场误会", minutes=20)
    assert r.ok and "每集约 8 分钟" in seen[0] and "20" not in seen[0].split("每集约")[1][:4]
    assert "没按 minutes=20 写" in r.content


# ================================================================ 2. 上屏字抽出来后期叠


SCRIPT = """第1集

[日] [外] [灵山]
【字幕：灵山】
金蝉子：弟子不服。

[夜] [内] [长安]
【字幕：贞观十三年，唐僧奉旨西行】
唐僧：阿弥陀佛。
【片尾字幕：剩余花瓣 9】
"""


def test_认字幕卡_剥掉_查分镜丢没丢():
    cards = find_cards(SCRIPT)
    assert [c.kind for c in cards] == ["字幕", "字幕", "片尾字幕"] and cards[2].ending
    board = "[远景/固定/3s] 两人对视，【片尾字幕：剩余花瓣 9】定格。"
    assert [c.text for c in missing_cards(SCRIPT, board)] == ["灵山", "贞观十三年，唐僧奉旨西行"]
    assert "少了 2 张" in card_loss("第1集", SCRIPT, board)
    # 模型把「【片尾字幕：剩余花瓣 9】」改写成「【剩余花瓣 9】」：按文字也要剥掉
    text = strip_cards("字迹浮现：【剩余花瓣 9】。【字幕：灵山】远山", ["剩余花瓣 9"])
    assert "剩余花瓣" not in text and "灵山" not in text and "：。" not in text


def test_字幕卡按镜号定位_排进段内时间线():
    desc = "\n".join([
        "[日] [外] [灵山]",
        "【字幕：灵山】",  # 单独成行：归下一个镜头
        "[远景/推入/3s] 云海翻涌",  # 1
        "[近景/固定/2s] 金蝉子抬头：“弟子不服。”",  # 2
        "[全景/拉远/2.5s] 两人对视，【片尾字幕：剩余花瓣 9】定格",  # 3
    ])
    cards = card_lines(desc)
    assert [c.text for c in cards[1]] == ["灵山"] and cards[3][0].ending
    seg = ShotPrompt("[第1集-1场]", "1-3", "10s", "x", cuts=[4, 3, 3])
    lost = place_cards([seg], cards, {1: 3.0, 2: 2.0, 3: 2.5})
    assert not lost
    first, last = seg.screen_text
    assert first["text"] == "灵山" and first["at"] == 0 and first["dur"] == 4
    assert last["at"] == 7 and last["end"] is True and last["dur"] >= 2.5, "片尾卡按 cuts 定位"
    assert place_cards([seg], {9: cards[1]}, {}) == cards[1], "没有段覆盖的镜头：报出来"
    back = parse_shots(json.dumps([as_dict(seg)], ensure_ascii=False))[0][0]
    assert back.screen_text[1]["end"] is True, "存进提示词资产再读回来不丢"


def test_分镜丢了字幕卡_算分镜问题():
    ep = Episode(1, "第1集", "[日] [外] [灵山]\n【高潮点】[远景/推入/3s] 云海\n")
    probs = storyboard_problems([ep], SCRIPT, EpisodeFormat())
    assert any("上屏字" in p and "【字幕：灵山】" in p for p in probs)


class _CardGateway(_ChunkGateway):
    """视频提示词照 _ChunkGateway 写；补放字幕卡的那次小调用（只有一条 user 消息）回镜号。"""

    def __init__(self) -> None:
        super().__init__()
        self.card_prompts: list[str] = []

    async def chat(self, role: str, messages: list[dict[str, Any]], **kw: Any) -> Any:
        if len(messages) == 1:
            self.card_prompts.append(messages[0]["content"])
            out = {"cards": [{"text": "灵山", "shot": 1}]}
            return SimpleNamespace(text=json.dumps(out, ensure_ascii=False), finish_reason="stop")
        return await super().chat(role, messages, **kw)


async def test_drama_shots_字不进画面_排进时间线_分镜丢的按剧本补回():
    from tests.test_episode_format import LIB

    store = AssetStore()
    script = store.create(SCRIPT, type_=AssetType.SCRIPT, summary="第1集剧本")
    rows = ["[日] [外] [灵山]"]
    for i in range(1, 11):
        tail = "，【片尾字幕：剩余花瓣 9】定格" if i == 10 else ""
        mark = HOOK_MARK if i == 1 else ""
        rows.append(f"{mark}[近景/固定/2s] 第{i}个动作：“台词{i}”{tail}")
    sb = store.create(
        json.dumps([{"episodeIndex": 1, "episodeTitle": "第1集", "episodeDesc": "\n".join(rows)}],
                   ensure_ascii=False),
        summary="分镜", creator="tool:drama_storyboard", parents=[script.id],
    )
    lib = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库",
                       creator="tool:drama_assets")
    gw = _CardGateway()
    fns = DramaFunctions(gw, store, registry=None, catalog=None, fmt=EpisodeFormat())
    r = await fns._fn_drama_shots(sb.id, lib.id)
    assert r.ok, r.error
    sent = "".join(m[1]["content"] for m in gw.calls if len(m) > 1)
    assert "剩余花瓣" not in sent, "字卡不给写提示词的模型看，免得写进画面"
    assert gw.card_prompts and "【字幕：灵山】" in gw.card_prompts[0]
    saved = json.loads(store.content(r.asset_ref))
    texts = {x["text"]: x for s in saved for x in s.get("screen_text", [])}
    assert set(texts) == {"灵山", "剩余花瓣 9"}, "分镜里的 + 按剧本补回的"
    assert texts["剩余花瓣 9"]["end"] is True
    assert store.get(r.asset_ref).gen_params["screen_text"] == 2
    assert "上屏字 2 张不进画面" in r.content and "按剧本补回" in r.content
    # 「贞观十三年」那张：小调用没给镜号，如实报出来
    assert "没排进时间线" in r.content and "贞观十三年" in r.content


async def test_旧规则出的提示词_有字幕卡就不许渲():
    store = AssetStore()
    script = store.create(SCRIPT, type_=AssetType.SCRIPT, summary="第1集剧本")
    desc = "[日] [外] [灵山]\n【高潮点】[远景/推入/3s] 云海翻涌"
    sb = store.create(
        json.dumps([{"episodeIndex": 1, "episodeTitle": "第1集", "episodeDesc": desc}],
                   ensure_ascii=False),
        summary="分镜", creator="tool:drama_storyboard", parents=[script.id],
    )
    shots = [ShotPrompt("[第1集-1场]", "1", "3s", "云海翻涌，字迹浮现【灵山】", cuts=[3])]
    old = store.create(json.dumps([as_dict(s) for s in shots], ensure_ascii=False),
                       summary="提示词", creator="tool:drama_shots", parents=[sb.id])
    fns = DramaFunctions(None, store)
    why = fns._coverage_gate(old.id, shots)
    assert "旧规则" in why and "drama_shots" in why
    new = store.create(json.dumps([as_dict(s) for s in shots], ensure_ascii=False),
                       summary="提示词", creator="tool:drama_shots", parents=[sb.id],
                       gen_params={"screen_text": 1})
    assert fns._coverage_gate(new.id, shots) == ""


async def test_成片叠字_按各段实际位置换算绝对时间():
    fns = DramaFunctions(None, AssetStore())
    a = ShotPrompt("[第1集-1场]", "1-5", "10s", "x",
                   screen_text=[{"text": "灵山", "at": 1.0, "dur": 2.5}])
    b = ShotPrompt("[第1集-2场]", "6-10", "12s", "y",
                   screen_text=[{"text": "剩余花瓣 9", "at": 9.0, "dur": 2.5, "end": True}])
    items = await fns._overlay_items([a, b], {0: {"asset": "as_x"}, 1: {"asset": "as_y"}})
    assert items[0] == {"text": "灵山", "start": 1.0, "end": 3.5, "position": "lower_third",
                        "size": "l"}
    assert items[1]["start"] == 19.0 and items[1]["end"] == 22.0, "片尾卡叠到这一集结束"
    assert items[1]["position"] == "center"
    assert _no_text_name("第01集.mp4") == "第01集_无字.mp4"


# ================================================================ 3. 整批报价


def _gate_with(answers: list[bool], guard: CostGuard) -> tuple[PermissionGate, list[str]]:
    asked: list[str] = []
    it = iter(answers)

    async def asker(meta: ToolMeta, args: Any) -> bool:
        asked.append(meta.summary)
        return next(it)

    return PermissionGate(EventBus(), asker=asker, guard=guard), asked


_VIDEO = ToolMeta(name="gen_video", summary="x", permission=PermissionLevel.COMPUTE,
                  cost_kind="video")


async def test_整批确认过_超额度不再逐个问_单子用完才问_答N整批停():
    guard = CostGuard(call_limits={"video": 1})
    gate, asked = _gate_with([False, False], guard)
    bp = BatchPass(label="第3集 4 段", units={"video": 3})
    with batch_scope(bp):
        results = [(await gate.check(_VIDEO, {}))[0] for _ in range(3)]
        assert results == [True, True, True] and not asked, "单子范围内不问"
        ok, why = await gate.check(_VIDEO, {})  # 第 4 段：比最坏情况还多 → 问，答 N
        assert not ok and len(asked) == 1 and "整批停下" in asked[0]
        assert bp.stopped
        ok, why = await gate.check(_VIDEO, {})
        assert not ok and "已经被你叫停" in why and len(asked) == 1, "叫停后一个都不发、也不再问"
    assert guard.usage.calls["video"] == 3, "批内的调用照样记账"
    ok, _ = await gate.check(_VIDEO, {})
    assert not ok and len(asked) == 2, "出了这一批，超额度照常逐次问（额度没被这一批永久抬高）"


async def test_批内被问到答y_这一批剩下的都不再问():
    guard = CostGuard(call_limits={"video": 0})
    gate, asked = _gate_with([True], guard)
    bp = BatchPass(label="第3集", units={"video": 1})
    with batch_scope(bp):
        assert (await gate.check(_VIDEO, {}))[0]
        assert (await gate.check(_VIDEO, {}))[0] and len(asked) == 1
        assert (await gate.check(_VIDEO, {}))[0] and len(asked) == 1


async def test_整批报价_写清最坏情况和额度_拒了就一个都不发():
    guard = CostGuard(call_limits={"video": 10}, seconds_limit=100)
    gate, asked = _gate_with([True, False], guard)
    fns = DramaFunctions(None, AssetStore(), registry=SimpleNamespace(gate=gate))
    worst = (12, 180.0, None)
    bp, deny = await fns._quote("drama_render_shots", "第3集 4 段", "video",
                                "第 3 集要新渲 4 段视频", worst, None, "每段最多再生成 2 次", {})
    assert bp is not None and not deny and bp.units == {"video": 12} and bp.seconds == 180
    text = asked[0]
    assert "最多 12 段 / 180 秒" in text and "没填单价" in text
    assert "额度还剩：10 段、100 秒" in text and "确认即为这一批放行" in text
    bp, deny = await fns._quote("drama_render_shots", "第3集 4 段", "video", "…", worst,
                                None, "", {})
    assert bp is None and "没有确认" in deny and "不要换个方式绕过去" in deny
    none, _ = await DramaFunctions(None, AssetStore())._quote(
        "x", "x", "video", "x", worst, None, "", {}
    )
    assert none is None, "没有询问入口（脚本 / 测试）：不开整批，照常逐次过闸门"


class _QuotingRegistry:
    """test_drama_render 的假注册表 + 一个会报价的闸门；记下每次生成调用时在不在整批里。"""

    def __init__(self, store: AssetStore, answer: bool = True) -> None:
        from tests.test_drama_render import FakeRegistry

        self.inner = FakeRegistry(store, delay=0.0)
        self.quotes: list[str] = []
        self.in_batch: list[bool] = []
        outer = self

        class _Gate:
            guard = CostGuard(call_limits={"image": 3, "video": 1})

            async def confirm_batch(self, tool: str, text: str, args: Any) -> bool:
                outer.quotes.append(text)
                return answer

        self.gate = _Gate()

    async def invoke(self, name: str, args: dict[str, Any]) -> Any:
        from aigc_agent.harness.permission.gate import current_batch

        if name != "compose_video":
            self.in_batch.append(current_batch() is not None)
        return await self.inner.invoke(name, args)


async def test_渲参考图_先整批报价_生成都在这一批里_拒了一张都不生成():
    from tests.test_drama_render import _lib_asset

    store = AssetStore()
    reg = _QuotingRegistry(store)
    r = await DramaFunctions(None, store, registry=reg)._fn_drama_render_assets(_lib_asset(store))
    assert r.ok, r.error
    assert len(reg.quotes) == 1 and "参考图要新生成 5 张" in reg.quotes[0]
    assert "额度还剩：3 张" in reg.quotes[0] and "确认即为这一批放行" in reg.quotes[0]
    assert reg.in_batch and all(reg.in_batch), "每次生图都在这一批的放行范围里"

    store2 = AssetStore()
    reg2 = _QuotingRegistry(store2, answer=False)
    r2 = await DramaFunctions(None, store2, registry=reg2)._fn_drama_render_assets(
        _lib_asset(store2)
    )
    assert not r2.ok and "没有确认" in (r2.error or "") and not reg2.in_batch
    assert r2.meta.get("charged") is False


async def test_渲一集_先整批报价():
    from tests.test_drama_render import SHOTS, _shots_asset

    store = AssetStore()
    reg = _QuotingRegistry(store)
    fns = DramaFunctions(None, store, registry=reg)
    r = await fns._fn_drama_render_shots(_shots_asset(store, SHOTS), episode=1)
    assert r.ok, r.error
    assert len(reg.quotes) == 1 and "第 1 集要新渲 2 段视频" in reg.quotes[0]
    assert "共约 16 秒" in reg.quotes[0] and all(reg.in_batch)


def test_额度还剩多少():
    g = CostGuard(call_limits={"image": 80}, seconds_limit=600, money_limit=200)
    g.record_call("image", n=30)
    room = g.room("image")
    assert room["calls"] == 50 and room["seconds"] == 600 and room["money"] == 200


async def test_流水线_有人可问时交给整批报价_不再按额度拦派发(tmp_path: Path):
    guard = CostGuard(call_limits={"video": 0})
    p = EpisodePipeline(SimpleNamespace(), AssetStore(), EventBus(), SimpleNamespace(),
                        guard=guard)
    assert not await p._budget_ok("video", 10, 100.0, "渲第 1 集")
    p.quotes = True
    assert await p._budget_ok("video", 10, 100.0, "渲第 1 集")

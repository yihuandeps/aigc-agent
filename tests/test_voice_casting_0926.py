"""定音（2026-09-26 用户定的第 10 条，选 B）：主角先各渲一段独白，人听过、采纳了才固定成音色锚点。

之前锚点是渲第 1 集时自动挑的段；/auto 两集并行渲，第 1、2 集的唐僧会是两种声音，第 3 集起
跟着后完成的那集走。现在：

- drama_voice_casting：台词最多的几个主角（或点名的几个）各渲一段约 5 秒的独白 —— TA 的主形象
  当参考、TA 自己的一句台词（挑 5 秒念得完的）、音色卡锁在提示词里；渲之前整批报价
- 渲完停下来请人听（大节点，/auto 也停）；人采纳才固定，采纳附言里点名的角色不固定；
  打回、/auto 自动采纳、别的节点的决定都不固定
- 没过质检（⛔）的独白不进待定
- 人定的锚点：渲染自动定的不覆盖它；定音采纳可以覆盖旧的人工锚点；渲染时就用它
"""

from __future__ import annotations

import json
import time
from typing import Any

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.drama.voice import ANCHORS_CREATOR, Anchor, parse_anchors
from aigc_agent.domain.functions.drama import CASTING_STAGE, DramaFunctions, _casting_line
from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.model.budget import CostGuard
from aigc_agent.harness.permission.gate import current_batch
from aigc_agent.harness.tools.provider import PermissionLevel, ToolResult
from tests.test_gate_chain import _fns as _gate_fns
from tests.test_voice import SHOTS, FakeRegistry, _seed

# 第 1 集的分镜：陆离 3 句、小满 2 句、老鬼一句都没有
ROWS = [
    "[夜] [内] [地铁车厢]",
    "[近景/固定/2s] 陆离压低声音说：“别回头。”",
    "[近景/固定/2s] 小满一愣：“为什么？”",
    "[中景/固定/3s] 陆离自语：“数到三的时候，我们一起往外跑。”",
    "[近景/固定/2s] 陆离：“三。”",
    "[近景/固定/3s] 小满喊：“等等我，我的鞋带散开了！”",
    "[全景/固定/2s] 老鬼在角落里打盹。",
]
LULI_LINE = "数到三的时候，我们一起往外跑。"
XM_LINE = "等等我，我的鞋带散开了！"


def _board(store: AssetStore, rows: list[str] = ROWS) -> str:
    return store.create(
        json.dumps(
            [{"episodeIndex": 1, "episodeTitle": "第1集", "episodeDesc": "\n".join(rows)}],
            ensure_ascii=False,
        ),
        summary="分镜·第1集",
        creator="tool:drama_storyboard",
    ).id


def _setup(fake: Any = None) -> tuple[AssetStore, DramaFunctions, Any, str, str, str]:
    """资产库（test_voice 的陆离 / 小满 / 老鬼，带音色卡）+ 参考图包 + 提示词 + 分镜。"""
    store = AssetStore()
    lib_id, shots_id, pack_id = _seed(store)
    _board(store)
    fake = fake(store) if fake is not None else FakeRegistry(store)
    fns = DramaFunctions(None, store, registry=fake, catalog=None)
    return store, fns, fake, lib_id, shots_id, pack_id


def _table(store: AssetStore) -> dict[str, Anchor]:
    tables = store.find(creator=ANCHORS_CREATOR)
    return parse_anchors(store.content(tables[0].id)) if tables else {}


async def _decide(fns: DramaFunctions, decision: str = "adopt", reason: str = "",
                  decided_by: str = "human", node: str = CASTING_STAGE) -> None:
    """走真的事件总线：app.py 就是这样把 drama.on_event 挂上去的。"""
    bus = EventBus()
    bus.subscribe(fns.on_event)
    await bus.emit(
        EventType.CHECKPOINT_DECIDED, turn=1, node=node, target_node=node,
        decision=decision, reason=reason, decided_by=decided_by, candidates=[],
    )


def _clip_of(store: AssetStore, name: str) -> Any:
    return next(a for a in store.find(type_=AssetType.VIDEO) if a.summary == f"定音·{name}")


# ---------------------------------------------------------------- 工具与挑台词


async def test_定音工具在目录里_要资产库id_算花钱的():
    _, fns, _, _, _, _ = _setup()
    metas = {m.name: m for m in await fns.list_tools()}
    assert "drama_voice_casting" in metas
    assert metas["drama_voice_casting"].permission is PermissionLevel.COMPUTE
    fn = (await fns.get_schema("drama_voice_casting"))["function"]
    assert fn["parameters"]["required"] == ["assets_id"]
    assert "渲第一集之前做一次" in fn["description"]


def test_定音用哪句台词_五秒念得完里挑最长_都太长挑最短():
    long = "这一句实在是太长了，五秒钟怎么念都念不完，" * 3
    assert _casting_line(["别回头。", LULI_LINE, "三。"]) == LULI_LINE, "太短的听不出音色"
    assert _casting_line(["数到三的时候跑。", LULI_LINE, XM_LINE]) == LULI_LINE, "念得完的挑最长"
    assert _casting_line([long, long + "还有"]) == long, "都念不完：挑最短的"
    assert _casting_line([]) == ""


# ---------------------------------------------------------------- 渲独白、停下来等人听


async def test_台词最多的主角各渲一段独白_停下来等人听_人没听之前不固定():
    store, fns, fake, lib_id, _, _ = _setup()
    r = await fns.invoke("drama_voice_casting", {"assets_id": lib_id})
    assert r.ok and r.suspend, r.error
    payload = r.suspend_payload
    assert payload["stage"] == CASTING_STAGE and payload["major"] is True, "大节点：/auto 也停"

    videos = {a["summary"]: a for a in fake.videos()}
    assert set(videos) == {"定音·陆离", "定音·小满"}, "老鬼没台词，不渲"
    luli = videos["定音·陆离"]
    assert luli["duration"] == 5 and luli["image"] == ["https://img/luli"], "TA 的主形象当参考"
    assert LULI_LINE in luli["prompt"] and "【定音独白】(陆离)" in luli["prompt"]
    assert "【音色锁定】(陆离)：男声" in luli["prompt"], "音色卡锁进提示词"
    assert luli["tags"] == {"casting": "陆离", "library": lib_id}
    assert XM_LINE in videos["定音·小满"]["prompt"]

    luli_clip, xm_clip = _clip_of(store, "陆离"), _clip_of(store, "小满")
    assert sorted(payload["assets"]) == sorted([luli_clip.id, xm_clip.id])
    assert f"♪ 陆离：「{LULI_LINE}」" in r.content and "声音对就采纳" in r.content
    assert _table(store) == {}, "人没听之前一个都不固定"


async def test_点名的角色_服装名也认_没台词的说清楚():
    store, fns, fake, lib_id, _, _ = _setup()
    r = await fns._fn_drama_voice_casting(
        lib_id, characters=["小满-连帽衫-[全集]", "老鬼", "路人甲"]
    )
    assert r.suspend, r.error
    assert [a["summary"] for a in fake.videos()] == ["定音·小满"], "只渲点名的、渲得了的"
    assert "· 老鬼：分镜里没找到 TA 的台词" in r.content
    assert set(fns._pending_casting) == {"小满"}  # noqa: SLF001


async def test_没渲参考图_没台词_包里没主形象_都在花钱之前拦下():
    store, fns, fake, lib_id, _, _ = _setup()
    bare_lib = store.create(store.content(lib_id), summary="资产库·没渲图", creator="t").id
    r = await fns._fn_drama_voice_casting(bare_lib)
    assert not r.ok and "还没渲参考图" in r.error and r.meta.get("charged") is False

    r = await fns._fn_drama_voice_casting(lib_id, characters=["老鬼"])
    assert not r.ok and "没有能定音的角色" in r.error and "老鬼：分镜里没找到 TA 的台词" in r.error
    assert r.meta.get("charged") is False

    no_xm = store.create(
        json.dumps({"陆离": {"asset": "x", "url": "https://img/luli", "kind": "角色"}}),
        summary="参考图包·缺小满", creator="tool:drama_render_assets", parents=[lib_id],
    )
    assert no_xm.id
    r = await fns._fn_drama_voice_casting(lib_id, characters=["小满"])
    assert not r.ok and "小满：参考图包里没有主形象" in r.error

    r = await fns._fn_drama_voice_casting("as_nope")
    assert not r.ok and "没有资产库" in r.error
    assert fake.videos() == [], "一段都没渲"


class _QuotingRegistry(FakeRegistry):
    """test_voice 的假注册表 + 会报价的闸门；记下每次生成在不在整批里。"""

    answer = True

    def __init__(self, store: AssetStore) -> None:
        super().__init__(store)
        self.quotes: list[tuple[str, str]] = []
        self.in_batch: list[bool] = []
        outer = self

        class _Gate:
            guard = CostGuard(call_limits={"video": 10})

            async def confirm_batch(self, tool: str, text: str, args: Any) -> bool:
                outer.quotes.append((tool, text))
                return outer.answer

        self.gate = _Gate()

    async def invoke(self, name: str, args: dict) -> ToolResult:
        self.in_batch.append(current_batch() is not None)
        return await super().invoke(name, args)


async def test_定音先整批报价_拒了一段都不渲():
    store, fns, fake, lib_id, _, _ = _setup(_QuotingRegistry)
    r = await fns._fn_drama_voice_casting(lib_id)
    assert r.suspend, r.error
    tool, text = fake.quotes[0]
    assert tool == "drama_voice_casting"
    assert "定音：给 2 个主角各渲一段 5 秒的独白（陆离、小满）" in text
    assert "额度还剩：10 段" in text
    assert fake.in_batch and all(fake.in_batch), "生成都在这一批的放行范围里"

    store2, fns2, fake2, lib2, _, _ = _setup(_QuotingRegistry)
    fake2.answer = False
    r2 = await fns2._fn_drama_voice_casting(lib2)
    assert not r2.ok and "没有确认" in r2.error and r2.meta.get("charged") is False
    assert fake2.videos() == [] and fns2._pending_casting == {}  # noqa: SLF001


# ---------------------------------------------------------------- 人采纳才固定


async def test_人采纳才固定_附言点名的角色不固定():
    store, fns, _, lib_id, _, _ = _setup()
    await fns._fn_drama_voice_casting(lib_id)
    luli_clip = _clip_of(store, "陆离")

    await _decide(fns, "adopt", reason="小满的声音太尖了，重来")
    table = _table(store)
    assert set(table) == {"陆离"}, "附言里点名的小满不固定"
    a = table["陆离"]
    assert a.asset == luli_clip.id and a.pinned and a.solo and a.scene == "定音"
    assert fns._pending_casting == {}  # noqa: SLF001

    listed = await fns._fn_drama_voice_anchors("list")
    assert "陆离" in listed.content and "人工指定" in listed.content


async def test_打回_auto自动采纳_别的节点的决定_都不固定():
    store, fns, _, lib_id, _, _ = _setup()
    await fns._fn_drama_voice_casting(lib_id)
    await _decide(fns, "revise", reason="声音都不对")
    assert _table(store) == {} and fns._pending_casting == {}  # noqa: SLF001

    await fns._fn_drama_voice_casting(lib_id)
    await _decide(fns, "adopt", decided_by="auto")
    assert _table(store) == {}, "/auto 自动采纳不算人听过"

    await fns._fn_drama_voice_casting(lib_id)
    bus = EventBus()
    bus.subscribe(fns.on_event)
    await bus.emit(EventType.CHECKPOINT_REACHED, stage=CASTING_STAGE, question="?")
    await _decide(fns, "adopt", node="剧本")
    assert _table(store) == {} and set(fns._pending_casting) == {"陆离", "小满"}, (  # noqa: SLF001
        "别的节点的决定、挂起事件都不动待定的定音"
    )
    await _decide(fns, "adopt")
    assert set(_table(store)) == {"陆离", "小满"}


async def test_没过质检的独白不进待定_都没过就不停下来问():
    store = AssetStore()
    fns, reg, _ = _gate_fns(store, {"定音·小满": [{"sub": True}, {"sub": True}]})
    lib_id, _, _ = _seed(store)
    _board(store)
    r = await fns._fn_drama_voice_casting(lib_id)
    assert r.suspend, r.error
    assert "⛔ 小满" in r.content and "♪ 陆离" in r.content
    assert set(fns._pending_casting) == {"陆离"}, "带字的那段不能当声音基准"  # noqa: SLF001
    assert r.suspend_payload["assets"] == [fns._pending_casting["陆离"].asset]  # noqa: SLF001

    store2 = AssetStore()
    both = {"定音·小满": [{"sub": True}] * 2, "定音·陆离": [{"sub": True}] * 2}
    fns2, _, _ = _gate_fns(store2, both)
    lib2, _, _ = _seed(store2)
    _board(store2)
    r2 = await fns2._fn_drama_voice_casting(lib2)
    assert not r2.ok and not r2.suspend and "定音一段都没渲成（或都没过质检）" in r2.error


# ---------------------------------------------------------------- 人定的锚点


async def test_定音采纳覆盖旧的人工锚点_渲染自动定的不覆盖人定的():
    store, fns, _, lib_id, _, _ = _setup()
    fns._save_anchors({"陆离": Anchor(character="陆离", asset="as_human", pinned=True)}, [])
    fns._merge_save_anchors({"陆离": Anchor(character="陆离", asset="as_auto")}, [])
    assert _table(store)["陆离"].asset == "as_human", "渲染自动定的不覆盖人定的"

    await fns._fn_drama_voice_casting(lib_id)
    await _decide(fns, "adopt")
    table = _table(store)
    assert table["陆离"].asset == _clip_of(store, "陆离").id, "定音本身就是人定的：覆盖旧的人工锚点"
    assert table["小满"].asset == _clip_of(store, "小满").id


async def test_定音的锚点_渲染时就用它_不再自动重定():
    store, fns, fake, lib_id, shots_id, pack_id = _setup()
    await fns._fn_drama_voice_casting(lib_id)
    await _decide(fns, "adopt")
    luli_url, xm_url = _clip_of(store, "陆离").uri, _clip_of(store, "小满").uri
    tables = len(store.find(creator=ANCHORS_CREATOR))

    fake.calls.clear()
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.ok, r.error
    by = {a["summary"]: a for a in fake.videos()}
    duo = set(by["[第1集-1场] 1-4"]["video_urls"])
    assert duo == {luli_url, xm_url}, "对话段对着定音的两段声音"
    assert by["[第1集-2场] 5-6"]["video_urls"] == [luli_url]
    assert xm_url in by["[第1集-3场] 7-8"]["video_urls"]
    assert "锚点沿用 2 人" in r.content and "本次新定" not in r.content
    assert len(store.find(creator=ANCHORS_CREATOR)) == tables, "锚点没变就不重写锚点表"
    assert SHOTS  # test_voice 的三段：对话、陆离独白、小满独白（带前序）


def _expire_casting(store: AssetStore) -> None:
    for name in ("陆离", "小满"):
        clip = _clip_of(store, name)
        clip.created_at = time.time() - 30 * 3600  # 生成链接过了有效期、又没配托管
        store.put(clip)


async def test_定音的锚点链接过期_渲染也不改人定的锚点表():
    store, fns, _, lib_id, shots_id, pack_id = _setup()
    await fns._fn_drama_voice_casting(lib_id)
    await _decide(fns, "adopt")
    luli_id = _clip_of(store, "陆离").id
    _expire_casting(store)

    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.ok, r.error
    table = _table(store)
    assert table["陆离"].asset == luli_id and table["陆离"].pinned, "人定的锚点，渲染不替换"


async def test_定音的锚点链接过期_结果不能说已经接替():
    """人定的锚点没被替换（上一条），结果里就不能告诉人「旧锚点链接过期已接替」。"""
    store, fns, _, lib_id, shots_id, pack_id = _setup()
    await fns._fn_drama_voice_casting(lib_id)
    await _decide(fns, "adopt")
    _expire_casting(store)

    tables = len(store.find(creator=ANCHORS_CREATOR))
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.ok, r.error
    assert "已接替" not in r.content, r.content
    assert "你定音的 陆离、小满 锚点链接过期了" in r.content and "你定的锚点没动" in r.content
    assert "本次新定" not in r.content, "这一集的临时锚点不算新定"
    assert len(store.find(creator=ANCHORS_CREATOR)) == tables, "锚点没变：不再存一份一样的表"


async def test_定音的锚点链接过期_报价时先说():
    store, fns, fake, lib_id, shots_id, pack_id = _setup(_QuotingRegistry)
    await fns._fn_drama_voice_casting(lib_id)
    await _decide(fns, "adopt")
    _expire_casting(store)
    fake.quotes.clear()

    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.ok, r.error
    tool, text = fake.quotes[0]
    assert tool == "drama_render_shots"
    assert "你定音的 陆离、小满 锚点链接过期了" in text and "会和你定的不一样" in text

"""按集流水的两个停点、/auto go、/auto stop、在跑环节的额度预留（2026-09-26 用户定的第 4 条）。

- 参考图渲完停一次（看脸），/auto go 放行后只渲第 1 集；第 1 集渲完再停一次（看片），
  放行后其余各集并行
- 放行按项目记盘：重启（新管线实例）不用再点；换个项目要重新放行
- 参考图换过（新的包）要再看一次
- /auto stop 当场取消在跑的环节（排队等信号量的也算），不记失败；/auto on 重派
- 在跑的渲染还没记账：下一集派发前把它们要花的算上（之前两集各查都够、合起来超）
- 等人处理的事（停点 / 失败）能拿出来放进模型上下文
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.output import OutputPrefs
from aigc_agent.domain.pipeline.episode_pipeline import EpisodePipeline
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.model.budget import Verdict

from .test_episode_pipeline import FakeRegistry, _episode, _settle, _write_state


def _pipe(
    store: AssetStore, bus: EventBus, fake: FakeRegistry, root: Path, *, total: int = 3,
    gates: bool = True, gates_path: Path | None = None, guard=None, renders: int = 2,
) -> EpisodePipeline:
    store.bus = bus
    _write_state(root, total)
    pipe = EpisodePipeline(
        fake, store, bus, OutputPrefs(root), guard=guard, enabled=True,
        max_parallel_renders=renders,
    )
    pipe.gates = gates
    pipe.gates_path = gates_path
    notes: list[str] = []
    pipe.note = notes.append
    pipe.notes = notes  # type: ignore[attr-defined]
    pipe.attach()
    return pipe


def _rendered(fake: FakeRegistry) -> list[int]:
    return sorted(a["episode"] for a in fake.args_of("drama_render_shots"))


async def _until(cond, seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if cond():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("条件没在限期内成立")


class SlowRegistry(FakeRegistry):
    """渲染卡住，直到测试放行（模拟一集要渲几十分钟）。"""

    def __init__(self, store: AssetStore):
        super().__init__(store)
        self.release = asyncio.Event()
        self.started: list[int] = []

    async def invoke(self, name, args):
        if name == "drama_render_shots":
            self.started.append(args["episode"])
            await self.release.wait()
        return await super().invoke(name, args)


class LimitGuard:
    """视频最多 limit 段（按这次要花的量查；不记账 —— 在跑的没记账正是要测的情形）。"""

    def __init__(self, limit: int):
        self.limit = limit
        self.asked: list[int] = []

    def check(self, kind: str = "", units: int = 1, **_: object) -> Verdict:
        if kind != "video":
            return Verdict(True)
        self.asked.append(units)
        ok = units <= self.limit
        return Verdict(ok, "" if ok else f"视频要 {units} 段，上限 {self.limit}")


# ---------------------------------------------------------------- 两个停点


async def test_参考图渲完先停_放行后只渲第1集_看片后再放行其余并行(tmp_path):
    store, bus = AssetStore(), EventBus()
    fake = FakeRegistry(store)
    pipe = _pipe(store, bus, fake, tmp_path)
    try:
        for n in range(1, 4):
            await _episode(store, n)
        await _settle(pipe)
        assert fake.count("drama_render_assets") == 1
        assert _rendered(fake) == [], "参考图渲完先停，一集都不渲"
        assert "参考图渲完了" in pipe.holding and "/auto go" in pipe.holding
        assert sum("参考图渲完了" in n for n in pipe.notes) == 1, "提示只打一次"
        assert "参考图渲完了" in pipe.notices()

        pipe.kick()  # 再踢也不渲、不重复提示
        await _settle(pipe)
        assert _rendered(fake) == []
        assert sum("参考图渲完了" in n for n in pipe.notes) == 1

        assert pipe.pass_gate() == "开始渲第 1 集"
        await _settle(pipe)
        assert _rendered(fake) == [1], "放行后第 1 集单独渲"
        assert "第 1 集渲完了" in pipe.holding

        assert pipe.pass_gate() == "其余各集开始自动并行渲"
        await _settle(pipe)
        assert _rendered(fake) == [1, 2, 3]
        assert pipe.holding == ""
        assert pipe.pass_gate() == "", "没停着的时候 /auto go 什么都不做"
    finally:
        await pipe.aclose()


async def test_放行按项目记盘_重启不用再点_换项目要重新放行(tmp_path):
    gates = tmp_path / "gates.json"
    root = tmp_path / "out"
    root.mkdir()
    store, bus = AssetStore(), EventBus()
    store.project = "proj-a"
    fake = FakeRegistry(store)
    pipe = _pipe(store, bus, fake, root, gates_path=gates)
    try:
        for n in range(1, 4):
            await _episode(store, n)
        await _settle(pipe)
        pipe.pass_gate()
        await _settle(pipe)
        pipe.pass_gate()
        await _settle(pipe)
        assert _rendered(fake) == [1, 2, 3]
    finally:
        await pipe.aclose()
    assert gates.exists()

    # 重启：同一个库、同一个项目，新管线实例。删掉一集的成片 → 直接补渲，不再停
    for a in store.find(newest_first=True):
        if a.creator == "tool:drama_render_shots" and a.gen_params.get("episode") == 3:
            store.set_status(a.id, "rejected")
            break
    fake2 = FakeRegistry(store)
    bus2 = EventBus()
    pipe2 = _pipe(store, bus2, fake2, root, gates_path=gates)
    try:
        pipe2.kick()
        await _settle(pipe2)
        assert _rendered(fake2) == [3]
        assert pipe2.holding == ""
    finally:
        await pipe2.aclose()

    # 别的项目：放行记录不通用
    pipe3 = EpisodePipeline(fake2, store, bus2, OutputPrefs(root))
    pipe3.gates, pipe3.gates_path = True, gates
    store.project = "proj-b"
    assert pipe3._passed() == {}  # noqa: SLF001
    store.project = "proj-a"
    assert set(pipe3._passed()) == {"refs", "first"}  # noqa: SLF001


async def test_参考图换过要再看一次(tmp_path):
    store, bus = AssetStore(), EventBus()
    fake = FakeRegistry(store)
    pipe = _pipe(store, bus, fake, tmp_path)
    try:
        for n in range(1, 4):
            await _episode(store, n)
        await _settle(pipe)
        pipe.pass_gate()
        await _settle(pipe)
        pipe.pass_gate()
        await _settle(pipe)
        assert _rendered(fake) == [1, 2, 3]

        # 人换了一张脸：新的参考图包
        store.create("{}", summary="参考图·换脸", creator="tool:drama_render_assets")
        await asyncio.sleep(0)
        await _settle(pipe)
        assert "参考图换过了" in pipe.holding
        assert "接着渲各集" in pipe.holding
        assert pipe.pass_gate() == "接着渲各集"
    finally:
        await pipe.aclose()


async def test_停点只在有人值守时开_默认不拦(tmp_path):
    store, bus = AssetStore(), EventBus()
    fake = FakeRegistry(store)
    pipe = _pipe(store, bus, fake, tmp_path, gates=False)
    try:
        for n in range(1, 4):
            await _episode(store, n)
        await _settle(pipe)
        assert _rendered(fake) == [1, 2, 3]
        assert pipe.holding == ""
    finally:
        await pipe.aclose()


# ---------------------------------------------------------------- /auto stop


async def test_auto_stop当场取消在跑和排队的渲染_不记失败_on后重派(tmp_path):
    store, bus = AssetStore(), EventBus()
    fake = SlowRegistry(store)
    pipe = _pipe(store, bus, fake, tmp_path, gates=False, renders=1)
    try:
        for n in range(1, 4):
            await _episode(store, n)
        await _until(lambda: fake.started == [1])
        await asyncio.sleep(0.05)
        assert pipe.busy
        assert {k for k in pipe._inflight if k.startswith("render:")} == {  # noqa: SLF001
            "render:1", "render:2", "render:3"
        }, "第 2、3 集在排队等信号量"

        n = pipe.stop()
        assert n == 3
        await _until(lambda: not pipe.busy)
        assert pipe.enabled is False
        assert pipe.failed == [], "人叫停的不算失败"
        assert pipe._reserved == {}  # noqa: SLF001
        assert fake.started == [1], "排队的没开始就取消了"

        fake.release.set()
        pipe.enabled = True
        pipe.on_enable()
        await _settle(pipe)
        assert sorted(fake.started) == [1, 1, 2, 3], "再开之后三集都重派"
    finally:
        fake.release.set()
        await pipe.aclose()


# ---------------------------------------------------------------- 额度预留


async def test_在跑的渲染还没记账_下一集派发前把它算上(tmp_path):
    store, bus = AssetStore(), EventBus()
    fake = SlowRegistry(store)
    guard = LimitGuard(limit=1)
    pipe = _pipe(store, bus, fake, tmp_path, total=2, gates=False, guard=guard)
    try:
        for n in range(1, 3):
            await _episode(store, n)
        await _until(lambda: fake.started == [1])
        await asyncio.sleep(0.1)
        assert fake.started == [1], "第 1 集在渲、还没记账：第 2 集（1+1 段）超了，不派"
        assert 2 in guard.asked
        assert any("预留" in n for n in pipe.notes)
        assert "额度不够" in pipe.notices()

        fake.release.set()  # 第 1 集渲完，预留释放
        await _settle(pipe)
        assert sorted(fake.started) == [1, 2]
    finally:
        fake.release.set()
        await pipe.aclose()


# ---------------------------------------------------------------- 等人处理的事


async def test_失败的环节进等人处理清单_retry后清掉(tmp_path):
    store, bus = AssetStore(), EventBus()
    fake = FakeRegistry(store)
    fake.fail["drama_render_shots"] = 1
    pipe = _pipe(store, bus, fake, tmp_path, total=1, gates=False)
    try:
        await _episode(store, 1)
        await _settle(pipe)
        text = pipe.notices()
        assert "第 1 集渲染失败" in text
        assert text.startswith("按集流水（/auto）等你处理的事")

        pipe.enabled = False
        assert "/auto 已关" in pipe.notices(), "关了也留着：人问起来模型答得上"

        pipe.enabled = True
        assert pipe.retry() == ["render:1"]
        await _settle(pipe)
        assert pipe.notices() == ""
    finally:
        await pipe.aclose()


# ---------------------------------------------------------------- 没人可问时的预检口径


async def test_预检按最坏情况算_扣掉过了质检的段():
    from tests.test_gate_chain import _fns as _gate_fns
    from tests.test_gate_chain import _seed

    store = AssetStore()
    fns, _, _ = _gate_fns(store, {})
    shots_id, pack_id = _seed(store)
    k = 1 + fns._worst_video_regen()  # noqa: SLF001
    assert k > 1, "质检会重生成：预检要把它算进去"
    assert fns.render_need(shots_id, 1) == (2 * k, 24.0 * k)

    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, episode=1)  # noqa: SLF001
    assert r.ok and r.meta["complete"]
    assert fns.render_need(shots_id, 1) == (0, 0.0), "都过了质检：重跑只拼成片，不花钱"

    pipe = EpisodePipeline(FakeRegistry(store), store, EventBus(), OutputPrefs(Path(".")))
    pipe.render_estimate = fns.render_need
    assert pipe._render_need(1, shots_id) == (0, 0.0)  # noqa: SLF001

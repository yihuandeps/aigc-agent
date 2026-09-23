"""按集流水并行管线（EpisodePipeline）验收。

- 每一集的剧本一到就单独拆这一集的分镜（传 script_id；2026-09-23 前是 5 集并成一次，
  必然超出模型单次输出上限被截断）
- 全剧剧本齐 → 资产库 → 参考图 顺序派发（各一次，跨集复用）
- 参考图就绪 → 逐集提示词 → 逐集渲染（第N集）
- enabled=False 一个都不派；on_enable() 扫存量补派
- 预算超支暂停渲染派发（文本环节不受影响），清零后自动续派
- 重复事件/重复 reconcile 幂等，不重复派发
- .drama-state.json 缺失或缺字段 → 不猜不派，打提示；补齐后续派
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions.content import ContentFunctions
from aigc_agent.domain.output import OutputPrefs
from aigc_agent.domain.pipeline.episode_pipeline import EpisodePipeline
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.model.budget import Verdict
from aigc_agent.harness.tools.provider import ToolResult


class FakeRegistry:
    """模拟 drama 工具：记录调用，并按真工具的约定落产出资产。

    落资产会发 ASSET_CREATED → 反过来触发管线 reconcile ——
    和真实链路一样，测试的就是这个事件驱动循环。
    """

    def __init__(self, store: AssetStore):
        self.store = store
        self.calls: list[tuple[str, dict]] = []
        self.fail: dict[str, int] = {}  # 工具名 → 还要失败几次
        self.spec = ""  # drama_shots 产物上的规格戳（真工具打 EpisodeFormat.stamp）

    def count(self, name: str) -> int:
        return sum(1 for n, _ in self.calls if n == name)

    def args_of(self, name: str) -> list[dict]:
        return [a for n, a in self.calls if n == name]

    async def invoke(self, name, args):
        self.calls.append((name, args))
        if self.fail.get(name, 0) > 0:
            self.fail[name] -= 1
            return ToolResult(ok=False, error="模拟失败")
        if name == "drama_storyboard":
            # 从剧本资产认集号，产出覆盖这些集的分镜资产（真工具的资产格式）
            script = self.store.content(args["script_id"]) if args.get("script_id") else (
                args["script"]
            )
            eps = sorted({int(m) for m in re.findall(r"第(\d+)集", script)})
            content = json.dumps(
                [
                    {
                        "episodeIndex": e,
                        "episodeTitle": f"第{e}集",
                        "episodeDesc": f"[第{e}集-1场] 镜头",
                    }
                    for e in eps
                ],
                ensure_ascii=False,
            )
            a = self.store.create(
                content,
                type_=AssetType.STORYBOARD,
                summary="分镜",
                creator="tool:drama_storyboard",
            )
            return ToolResult(ok=True, asset_ref=a.id)
        if name == "drama_assets":
            a = self.store.create("{}", summary="资产库", creator="tool:drama_assets")
            return ToolResult(ok=True, asset_ref=a.id)
        if name == "drama_render_assets":
            a = self.store.create("{}", summary="参考图", creator="tool:drama_render_assets")
            return ToolResult(ok=True, asset_ref=a.id)
        if name == "drama_shots":
            gp = {"episode": args["episode"]}
            if self.spec:
                gp["spec"] = self.spec
            a = self.store.create(
                "[]",
                summary=f"提示词·第{args['episode']}集",
                creator="tool:drama_shots",
                gen_params=gp,
            )
            return ToolResult(ok=True, asset_ref=a.id)
        if name == "drama_render_shots":
            a = self.store.create(
                "[]",
                summary=f"片段·第{args['episode']}集",
                creator="tool:drama_render_shots",
                gen_params={"episode": args["episode"]},
            )
            return ToolResult(ok=True, asset_ref=a.id)
        raise AssertionError(f"意外调用 {name}")


class FakeGuard:
    def __init__(self, over: bool = False):
        self.over = over

    def check(self, kind: str = "", **_: object) -> Verdict:
        # 真的 CostGuard.check 还收 units / money / seconds（2026-09-23 按这一步要花多少查）
        return Verdict(not self.over, "预算超了" if self.over else "")


def _write_state(root: Path, total: int) -> None:
    (root / ".drama-state.json").write_text(
        json.dumps({"ethnicity": "chinese", "language": "zh", "totalEpisodes": total}),
        encoding="utf-8",
    )


def _build(
    store: AssetStore,
    bus: EventBus,
    fake: FakeRegistry,
    root: Path,
    *,
    total: int = 12,
    enabled: bool = True,
    guard=None,
    spec: str = "",
) -> EpisodePipeline:
    store.bus = bus
    _write_state(root, total)
    pipe = EpisodePipeline(
        fake, store, bus, OutputPrefs(root), guard=guard, enabled=enabled, spec=spec
    )
    notes: list[str] = []
    pipe.note = notes.append
    pipe.attach()
    pipe.notes = notes  # type: ignore[attr-defined]  # 测试断言用
    return pipe


async def _episode(store: AssetStore, n: int) -> None:
    store.create(
        f"第{n}集 剧本内容",
        type_=AssetType.SCRIPT,
        summary=f"第{n}集·测试",
        creator="model",
        gen_params={"episode": n},
    )
    await asyncio.sleep(0)  # 让 fire-and-forget 的资产事件发出来


async def _settle(pipe: EpisodePipeline, seconds: float = 5.0) -> None:
    """等管线静下来：队列清空 + 没有在派任务，连续安静 0.1s。"""
    deadline = time.monotonic() + seconds
    quiet = 0.0
    while time.monotonic() < deadline:
        await asyncio.sleep(0.02)
        if pipe._queue.empty() and not pipe._inflight:
            quiet += 0.02
            if quiet >= 0.1:
                return
        else:
            quiet = 0.0
    raise AssertionError("管线没在限期内静下来")


# ---------------------------------------------------------------- 批次分镜


async def test_每集剧本一到就单独拆这一集的分镜(tmp_path):
    store, bus = AssetStore(), EventBus()
    fake = FakeRegistry(store)
    pipe = _build(store, bus, fake, tmp_path, total=12)
    try:
        for n in range(1, 4):
            await _episode(store, n)
        await _settle(pipe)
        calls = fake.args_of("drama_storyboard")
        assert len(calls) == 3, "一集一次，不再等凑满 5 集"
        assert all("script_id" in a and "script" not in a for a in calls), "传资产 id 不塞全文"
        got = sorted(int(store.get(a["script_id"]).gen_params["episode"]) for a in calls)
        assert got == [1, 2, 3]

        await _episode(store, 4)
        await _settle(pipe)
        assert fake.count("drama_storyboard") == 4
    finally:
        await pipe.aclose()


# ---------------------------------------------------------------- 逐级接续


async def test_全剧齐后资产库参考图提示词渲染逐级接续(tmp_path):
    store, bus = AssetStore(), EventBus()
    fake = FakeRegistry(store)
    pipe = _build(store, bus, fake, tmp_path, total=3)
    try:
        for n in range(1, 4):
            await _episode(store, n)
        await _settle(pipe)

        assert fake.count("drama_storyboard") == 3  # 一集一次
        assert fake.count("drama_assets") == 1
        assert len(fake.args_of("drama_assets")[0]["script_ids"]) == 3  # 全剧按集号传 id
        assert fake.count("drama_render_assets") == 1
        # 逐集提示词，episode 参数正确
        assert {a["episode"] for a in fake.args_of("drama_shots")} == {1, 2, 3}
        # 逐集渲染，带参考图资产 id
        renders = fake.args_of("drama_render_shots")
        assert {a["episode"] for a in renders} == {1, 2, 3}
        assert all(a["rendered_id"] for a in renders)
        assert all(a["shots_id"] for a in renders)
    finally:
        await pipe.aclose()


# ---------------------------------------------------------------- 开关


async def test_关闭时不派发_on后扫存量补派(tmp_path):
    store, bus = AssetStore(), EventBus()
    fake = FakeRegistry(store)
    pipe = _build(store, bus, fake, tmp_path, total=3, enabled=False)
    try:
        for n in range(1, 4):
            await _episode(store, n)
        await _settle(pipe)
        assert fake.calls == []  # 事件照收，一个都不派

        pipe.enabled = True
        pipe.on_enable()
        await _settle(pipe)
        assert fake.count("drama_storyboard") == 3
        assert fake.count("drama_render_shots") == 3
    finally:
        await pipe.aclose()


# ---------------------------------------------------------------- 预算刹车


async def test_预算超支暂停渲染_清零后续派(tmp_path):
    store, bus = AssetStore(), EventBus()
    fake = FakeRegistry(store)
    guard = FakeGuard(over=True)
    pipe = _build(store, bus, fake, tmp_path, total=3, guard=guard)
    try:
        for n in range(1, 4):
            await _episode(store, n)
        await _settle(pipe)

        # 文本环节（分镜/资产库/提示词）不受预算闸影响，花钱的渲染类被刹住
        assert fake.count("drama_storyboard") == 3
        assert fake.count("drama_assets") == 1
        assert fake.count("drama_render_assets") == 0
        assert fake.count("drama_shots") == 3
        assert fake.count("drama_render_shots") == 0
        assert any("预算" in n for n in pipe.notes)

        guard.over = False  # 等于 /budget reset
        pipe.kick()
        await _settle(pipe)
        assert fake.count("drama_render_assets") == 1
        assert fake.count("drama_render_shots") == 3
    finally:
        await pipe.aclose()


# ---------------------------------------------------------------- 幂等


async def test_重复事件与重复reconcile不重复派发(tmp_path):
    store, bus = AssetStore(), EventBus()
    fake = FakeRegistry(store)
    pipe = _build(store, bus, fake, tmp_path, total=3)
    try:
        for n in range(1, 4):
            await _episode(store, n)
        await _settle(pipe)
        before = list(fake.calls)

        for _ in range(3):
            pipe.kick()
        await _settle(pipe)
        assert fake.calls == before  # 一个都没多派
    finally:
        await pipe.aclose()


# ---------------------------------------------------------------- 状态文件


async def test_状态文件缺失不猜不派_补齐后续派(tmp_path):
    store, bus = AssetStore(), EventBus()
    fake = FakeRegistry(store)
    store.bus = bus
    pipe = EpisodePipeline(fake, store, bus, OutputPrefs(tmp_path), enabled=True)
    notes: list[str] = []
    pipe.note = notes.append
    pipe.attach()
    try:
        await _episode(store, 1)
        await _settle(pipe)
        assert fake.calls == []
        assert sum("ethnicity" in n for n in notes) == 1  # 提示只打一次

        _write_state(tmp_path, 1)
        pipe.kick()  # 模拟补齐后的下一条资产事件
        await _settle(pipe)
        assert fake.count("drama_storyboard") == 1
    finally:
        await pipe.aclose()


# ---------------------------------------------------------------- save_draft


async def test_save_draft带episode落gen_params():
    store = AssetStore()
    fns = ContentFunctions(store)
    r = await fns.invoke(
        "save_draft",
        {"content": "第三集内容", "kind": "script", "summary": "第3集·测试", "episode": 3},
    )
    assert r.ok
    assert store.get(r.asset_ref).gen_params["episode"] == 3
    # 不传 episode 时不落标记（流水线不把它当剧集）
    r2 = await fns.invoke("save_draft", {"content": "普通稿", "kind": "copy"})
    assert "episode" not in store.get(r2.asset_ref).gen_params


# ---------------------------------------------------------------- 失败重派 / 旧规格 / 事件筛选


async def test_无关资产事件不触发reconcile(tmp_path):
    store, bus = AssetStore(), EventBus()
    fake = FakeRegistry(store)
    pipe = _build(store, bus, fake, tmp_path, total=3)
    try:
        await _settle(pipe)
        ran = 0
        orig = pipe._reconcile

        async def counting() -> None:
            nonlocal ran
            ran += 1
            await orig()

        pipe._reconcile = counting  # type: ignore[method-assign]
        # 渲染时每段视频 / 每张图都会落库（一段 put 2–3 次）：这些不改变流水线视图
        for i in range(5):
            a = store.create("", type_=AssetType.VIDEO, summary=f"片段{i}",
                             creator="tool:gen_video")
            store.put(a)
        await asyncio.sleep(0.05)
        await _settle(pipe)
        assert ran == 0, "媒体资产落库也跑了全量 reconcile"
        await _episode(store, 1)
        await _settle(pipe)
        assert ran >= 1 and fake.count("drama_storyboard") == 1
    finally:
        await pipe.aclose()


async def test_文本环节失败自动再试一次(tmp_path):
    store, bus = AssetStore(), EventBus()
    fake = FakeRegistry(store)
    fake.fail["drama_storyboard"] = 1
    pipe = _build(store, bus, fake, tmp_path, total=12)
    pipe.retry_delay = 0.05
    try:
        await _episode(store, 1)
        await _settle(pipe)
        # 冷却过了会自己踢一次（没有别的事件来踢）
        for _ in range(150):
            if fake.count("drama_storyboard") >= 2:
                break
            await asyncio.sleep(0.02)
        await _settle(pipe)
        assert fake.count("drama_storyboard") == 2, "偶发失败的文本环节应该自动再试一次"
        assert not pipe.failed
        assert any("自动再试" in n for n in pipe.notes)
    finally:
        await pipe.aclose()


async def test_失败后输入变了自动重派_不变就等人(tmp_path):
    store, bus = AssetStore(), EventBus()
    fake = FakeRegistry(store)
    fake.fail["drama_storyboard"] = 5
    pipe = _build(store, bus, fake, tmp_path, total=12)
    pipe.text_retries = 0  # 不自动再试，直接看失败标记
    try:
        await _episode(store, 1)
        await _settle(pipe)
        assert fake.count("drama_storyboard") == 1 and pipe.failed == ["storyboard:1"]
        pipe.kick()
        await _settle(pipe)
        assert fake.count("drama_storyboard") == 1, "输入没变，不该反复派"

        fake.fail.clear()
        await _episode(store, 1)  # 人改了第 1 集剧本：新版本 → 新的输入
        await _settle(pipe)
        assert fake.count("drama_storyboard") == 2 and not pipe.failed
    finally:
        await pipe.aclose()


async def test_auto_retry重派失败的渲染(tmp_path):
    store, bus = AssetStore(), EventBus()
    fake = FakeRegistry(store)
    fake.fail["drama_render_shots"] = 1
    pipe = _build(store, bus, fake, tmp_path, total=1)
    try:
        await _episode(store, 1)
        await _settle(pipe)
        assert fake.count("drama_render_shots") == 1 and pipe.failed == ["render:1"]
        pipe.kick()
        await _settle(pipe)
        assert fake.count("drama_render_shots") == 1, "花钱的环节不自动重试"

        assert pipe.retry() == ["render:1"]
        await _settle(pipe)
        assert fake.count("drama_render_shots") == 2 and not pipe.failed
    finally:
        await pipe.aclose()


def _prepopulate(store: AssetStore, shots_spec: str) -> str:
    """一集已经走到「有提示词、有参考图、还没渲」的存量（store 还没接总线，不发事件）。"""
    store.create("第1集 剧本", type_=AssetType.SCRIPT, summary="第1集", creator="model",
                 gen_params={"episode": 1})
    sb = json.dumps([{"episodeIndex": 1, "episodeTitle": "第1集",
                      "episodeDesc": "[第1集-1场] 镜头"}], ensure_ascii=False)
    store.create(sb, type_=AssetType.STORYBOARD, summary="分镜", creator="tool:drama_storyboard")
    store.create("{}", summary="资产库", creator="tool:drama_assets")
    store.create("{}", summary="参考图", creator="tool:drama_render_assets")
    gp = {"episode": 1, **({"spec": shots_spec} if shots_spec else {})}
    old = store.create("[]", summary="提示词·第1集", creator="tool:drama_shots", gen_params=gp)
    return old.id


async def test_旧规格的提示词先按现行规格重出再渲(tmp_path):
    store, bus = AssetStore(), EventBus()
    old = _prepopulate(store, "S1")
    fake = FakeRegistry(store)
    fake.spec = "S2"
    pipe = _build(store, bus, fake, tmp_path, total=1, spec="S2")
    try:
        pipe.kick()
        await _settle(pipe)
        assert fake.count("drama_shots") == 1, "旧规格的提示词要重出"
        renders = fake.args_of("drama_render_shots")
        assert len(renders) == 1 and renders[0]["shots_id"] != old, "不许拿旧规格的去花钱"
        assert any("旧规格" in n for n in pipe.notes)
    finally:
        await pipe.aclose()


async def test_规格戳一致不重出_重出后仍旧也只重出一次(tmp_path):
    store, bus = AssetStore(), EventBus()
    _prepopulate(store, "S2")
    fake = FakeRegistry(store)
    pipe = _build(store, bus, fake, tmp_path, total=1, spec="S2")
    try:
        pipe.kick()
        await _settle(pipe)
        assert fake.count("drama_shots") == 0 and fake.count("drama_render_shots") == 1
    finally:
        await pipe.aclose()

    store2, bus2 = AssetStore(), EventBus()
    _prepopulate(store2, "")  # 旧版产物没有规格戳
    fake2 = FakeRegistry(store2)  # 工具产物也不打戳：重出一次就照着渲，不能死循环
    pipe2 = _build(store2, bus2, fake2, tmp_path, total=1, spec="S2")
    try:
        pipe2.kick()
        await _settle(pipe2)
        assert fake2.count("drama_shots") == 1 and fake2.count("drama_render_shots") == 1
    finally:
        await pipe2.aclose()

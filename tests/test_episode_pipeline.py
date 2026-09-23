"""按集流水并行管线（EpisodePipeline）验收。

- 剧本满 5 集才派分镜；最后一批不满 5 也派
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

    def count(self, name: str) -> int:
        return sum(1 for n, _ in self.calls if n == name)

    def args_of(self, name: str) -> list[dict]:
        return [a for n, a in self.calls if n == name]

    async def invoke(self, name, args):
        self.calls.append((name, args))
        if name == "drama_storyboard":
            # 从剧本原文认集号，产出覆盖这些集的分镜资产（真工具的资产格式）
            eps = sorted({int(m) for m in re.findall(r"第(\d+)集", args["script"])})
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
            a = self.store.create(
                "[]",
                summary=f"提示词·第{args['episode']}集",
                creator="tool:drama_shots",
                gen_params={"episode": args["episode"]},
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

    def check(self, kind: str = "") -> Verdict:
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
) -> EpisodePipeline:
    store.bus = bus
    _write_state(root, total)
    pipe = EpisodePipeline(fake, store, bus, OutputPrefs(root), guard=guard, enabled=enabled)
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


async def test_满5集才派分镜_末批不满也派(tmp_path):
    store, bus = AssetStore(), EventBus()
    fake = FakeRegistry(store)
    pipe = _build(store, bus, fake, tmp_path, total=12)
    try:
        for n in range(1, 5):
            await _episode(store, n)
        await _settle(pipe)
        assert fake.count("drama_storyboard") == 0  # 4 集不派

        await _episode(store, 5)
        await _settle(pipe)
        assert fake.count("drama_storyboard") == 1  # 第 1-5 集一批
        script = fake.args_of("drama_storyboard")[0]["script"]
        assert "第1集" in script and "第5集" in script

        for n in range(6, 10):
            await _episode(store, n)
        await _settle(pipe)
        assert fake.count("drama_storyboard") == 1  # 第二批还差一集

        await _episode(store, 10)
        await _settle(pipe)
        assert fake.count("drama_storyboard") == 2

        await _episode(store, 11)
        await _episode(store, 12)
        await _settle(pipe)
        assert fake.count("drama_storyboard") == 3  # 末批只有 2 集也派
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

        assert fake.count("drama_storyboard") == 1  # 不满 5 的唯一一批
        assert fake.count("drama_assets") == 1
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
        assert fake.count("drama_storyboard") == 1
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
        assert fake.count("drama_storyboard") == 1
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

"""批量生成并发（2026-09-19）。

用户实测：19 段视频跑了 2.3 小时、零重叠。根因不是并发写错了，而是模型一次只调一个
gen_video —— 一次工具调用就是一个迭代，必然串行；并发逻辑当时只在 drama_render_shots 里，
手搓链路享受不到。这里把并发下沉到 gen_videos / gen_images，并钉住：真的并发、受上限约束、
单个失败不拖垮整批、顺序与入参一致。
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions.media import MediaFunctions
from aigc_agent.domain.generators.catalog import Concurrency, MediaCatalog
from aigc_agent.domain.output import OutputPrefs
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.model.media import MediaGateway, MediaKind, MediaTask, TaskStatus

CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "media_models.yaml"


class SlowProvider:
    """每次生成耗时固定，记录并发峰值与每次调用的提示词。"""

    name = "fake"

    def __init__(self, delay: float = 0.15, fail_on: set[str] | None = None) -> None:
        self.delay = delay
        self.fail_on = fail_on or set()
        self.active = 0
        self.max_active = 0
        self.prompts: list[str] = []
        self.params: list[dict[str, Any]] = []

    async def submit(self, kind: MediaKind, model: str, prompt: str, **params: Any) -> MediaTask:
        self.prompts.append(prompt)
        self.params.append(params)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.active -= 1
        if any(k in prompt for k in self.fail_on):
            return MediaTask(
                task_id="", kind=kind, model=model, status=TaskStatus.FAILED, error="模拟失败"
            )
        ext = "mp4" if kind is MediaKind.VIDEO else "png"
        return MediaTask(
            task_id="t", kind=kind, model=model, status=TaskStatus.SUCCEEDED,
            urls=[f"https://fake/{len(self.prompts)}.{ext}"],
        )

    async def poll(self, task: MediaTask) -> MediaTask:
        return task

    async def close(self) -> None:
        return None


async def _no_download(url: str, dest: Path) -> bool:
    return False


def _fns(tmp_path: Path, provider: SlowProvider, image: int = 6, video: int = 8) -> MediaFunctions:
    catalog = MediaCatalog.load(CATALOG_PATH)
    catalog.concurrency = Concurrency(image=image, video=video)
    gw = MediaGateway(
        {catalog.provider: provider}, EventBus(), poll_interval=0.01, max_poll_interval=0.02
    )
    return MediaFunctions(
        gw, catalog, AssetStore(tmp_path / "assets"),
        prefs=OutputPrefs(tmp_path / "out", downloader=_no_download),
    )


def _jobs(n: int, prefix: str = "镜头") -> list[dict[str, Any]]:
    return [{"prompt": f"{prefix}{i}", "summary": f"{prefix} {i}"} for i in range(1, n + 1)]


# ---------------------------------------------------------------- 真的并发


async def test_批量视频真的并发_不是一段段跑(tmp_path: Path):
    p = SlowProvider(delay=0.2)
    fns = _fns(tmp_path, p, video=8)
    t0 = time.perf_counter()
    r = await fns.invoke("gen_videos", {"jobs": _jobs(8), "model": "seedance-2.0"})
    dt = time.perf_counter() - t0
    assert r.ok, r.error
    assert p.max_active == 8, f"应该 8 段同时在跑，实际峰值 {p.max_active}"
    assert dt < 0.2 * 4, f"并发跑完应该远快于串行，实际 {dt:.2f}s"
    assert "成功 8，失败 0" in r.content


async def test_并发受配置上限约束(tmp_path: Path):
    p = SlowProvider(delay=0.1)
    fns = _fns(tmp_path, p, video=3)
    r = await fns.invoke("gen_videos", {"jobs": _jobs(9), "model": "seedance-2.0"})
    assert r.ok and p.max_active == 3, f"上限 3，实际峰值 {p.max_active}"
    assert len(p.prompts) == 9


async def test_批量图片同样并发(tmp_path: Path):
    p = SlowProvider(delay=0.1)
    fns = _fns(tmp_path, p, image=6)
    r = await fns.invoke("gen_images", {"jobs": _jobs(6, "参考图"), "model": "gpt-image-2"})
    assert r.ok and p.max_active == 6
    assert "批量生成图片 6 个" in r.content


# ---------------------------------------------------------------- 结果与容错


async def test_返回顺序与入参一致_可直接喂给拼接(tmp_path: Path):
    p = SlowProvider(delay=0.02)
    fns = _fns(tmp_path, p, video=2)  # 上限小于任务数，完成顺序会乱
    r = await fns.invoke("gen_videos", {"jobs": _jobs(5), "model": "seedance-2.0"})
    assert r.ok, r.error
    ids = r.content.rsplit("按顺序的资产 id：", 1)[1].split()
    assert len(ids) == 5
    store = fns.store
    assert [store.get(i).summary for i in ids] == [f"镜头 {i}" for i in range(1, 6)]


async def test_单个失败不拖垮整批_失败项说清楚(tmp_path: Path):
    p = SlowProvider(delay=0.02, fail_on={"镜头3"})
    fns = _fns(tmp_path, p, video=4)
    r = await fns.invoke("gen_videos", {"jobs": _jobs(5), "model": "seedance-2.0"})
    assert r.ok, "一个失败不算整批失败"
    assert "成功 4，失败 1" in r.content and "✗ 镜头 3" in r.content
    ids = r.content.rsplit("按顺序的资产 id：", 1)[1].split()
    assert len(ids) == 4


async def test_全部失败才算失败(tmp_path: Path):
    p = SlowProvider(delay=0.02, fail_on={"镜头"})
    fns = _fns(tmp_path, p, video=4)
    r = await fns.invoke("gen_videos", {"jobs": _jobs(3), "model": "seedance-2.0"})
    assert not r.ok and "成功 0，失败 3" in r.error


async def test_空任务与超量都拦住(tmp_path: Path):
    fns = _fns(tmp_path, SlowProvider())
    assert not (await fns.invoke("gen_videos", {"jobs": []})).ok
    r = await fns.invoke("gen_videos", {"jobs": _jobs(41)})
    assert not r.ok and "最多 40 个" in r.error


# ---------------------------------------------------------------- 参数合并


async def test_公共参数与单项覆盖(tmp_path: Path):
    p = SlowProvider(delay=0.01)
    fns = _fns(tmp_path, p, video=4)
    jobs = [
        {"prompt": "第一段", "summary": "s1"},
        {"prompt": "第二段", "summary": "s2", "duration": 5, "image": ["https://i/1"]},
    ]
    r = await fns.invoke(
        "gen_videos",
        {"jobs": jobs, "model": "seedance-2.0", "aspect_ratio": "9:16",
         "resolution": "720p", "duration": 12},
    )
    assert r.ok, r.error
    by_prompt = {pr: pa for pr, pa in zip(p.prompts, p.params, strict=True)}
    first = next(pa for pr, pa in by_prompt.items() if "第一段" in pr)
    second = next(pa for pr, pa in by_prompt.items() if "第二段" in pr)
    assert first["duration"] == 12 and second["duration"] == 5, "单项覆盖公共时长"
    assert first["aspect_ratio"] == "9:16" and second["resolution"] == "720p"
    assert second["image"] == ["https://i/1"] and "image" not in first
    # 禁字幕的最外层禁令批量时同样生效
    assert all("严禁" in pr and "字幕" in pr for pr in p.prompts)


async def test_批量图片的资产类型与参考图(tmp_path: Path):
    p = SlowProvider(delay=0.01)
    fns = _fns(tmp_path, p, image=4)
    jobs = [{"prompt": "主形象", "summary": "角色·陆离", "image": ["https://ref/1"]}]
    r = await fns.invoke("gen_images", {"jobs": jobs, "model": "gpt-image-2"})
    assert r.ok, r.error
    ids = r.content.rsplit("按顺序的资产 id：", 1)[1].split()
    a = fns.store.get(ids[0])
    assert a.type is AssetType.IMAGE and a.summary == "角色·陆离"
    assert p.params[0]["image"] == ["https://ref/1"]


# ---------------------------------------------------------------- 预算按真实数量算


async def test_批量按段数计次_不是按调用次数(tmp_path: Path):
    """一次 gen_videos 生成 N 段要记 N 次，否则预算护栏形同虚设。"""
    from aigc_agent.harness.model.budget import CostGuard
    from aigc_agent.harness.permission.gate import PermissionGate
    from aigc_agent.harness.tools.registry import ToolRegistry

    bus = EventBus()
    guard = CostGuard(call_limits={"video": 10})
    registry = ToolRegistry(bus)
    registry.register(_fns(tmp_path, SlowProvider(delay=0.01), video=4))
    await registry.refresh()

    meta = registry.meta("gen_videos")
    assert meta is not None and meta.cost_units_arg == "jobs"

    ok, _ = await PermissionGate(bus, guard=guard).check(meta, {"jobs": _jobs(6)})
    assert ok and guard.usage.calls["video"] == 6, "6 段要记 6 次"

    # 再来 6 段就超过 10 次上限，必须拦下（没有询问器时直接拒）
    ok2, why = await PermissionGate(bus, guard=guard).check(meta, {"jobs": _jobs(6)})
    assert not ok2 and "这次要 6 个" in why and "上限 10 次" in why
    assert guard.usage.calls["video"] == 6, "被拦下就不该记账"

    # 单段工具仍然按 1 次算
    single = registry.meta("gen_video")
    assert single is not None and single.cost_units_arg == ""
    ok3, _ = await PermissionGate(bus, guard=guard).check(single, {"prompt": "x"})
    assert ok3 and guard.usage.calls["video"] == 7

"""媒体生成验收 —— 异步任务模型 + 自由选型。

异步任务是架构点名「最容易漏、后期改起来最痛」的一处，重点验它。
不打真实 API（花钱且慢），用 FakeMediaProvider 控制轮询次数与失败点。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions.media import MediaFunctions
from aigc_agent.domain.generators.catalog import MediaCatalog
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.model.media import (
    FakeMediaProvider,
    MediaGateway,
    MediaKind,
    MediaTask,
    TaskStatus,
    extract_urls,
    normalize_status,
)
from aigc_agent.harness.tools.registry import ToolRegistry

CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "media_models.yaml"


def _gateway(provider: FakeMediaProvider, **kw) -> MediaGateway:
    return MediaGateway(
        {"apimart": provider}, EventBus(), poll_interval=0.01, max_poll_interval=0.02, **kw
    )


async def _funcs(provider: FakeMediaProvider):
    store = AssetStore()
    catalog = MediaCatalog.load(CATALOG_PATH)
    fns = MediaFunctions(_gateway(provider), catalog, store)
    bus = EventBus()
    registry = ToolRegistry(bus)
    registry.register(fns)
    await registry.refresh()
    return registry, store, catalog


# ---------------------------------------------------------------- 响应解析


def test_各家状态字符串归一():
    for raw in ("completed", "success", "SUCCEEDED", "Finished"):
        assert normalize_status(raw) is TaskStatus.SUCCEEDED
    for raw in ("pending", "queued", "in_queue"):
        assert normalize_status(raw) is TaskStatus.SUBMITTED
    for raw in ("processing", "in_progress", "generating"):
        assert normalize_status(raw) is TaskStatus.RUNNING
    for raw in ("failed", "error", "cancelled"):
        assert normalize_status(raw) is TaskStatus.FAILED
    # 认不出来的当作还在跑，不误判成失败
    assert normalize_status("某种没见过的状态") is TaskStatus.RUNNING
    assert normalize_status(None) is TaskStatus.RUNNING


def test_从任意结构里捞出媒体URL():
    """各家返回结构差异很大，与其逐个写解析器不如认 URL 形状。"""
    assert extract_urls({"data": [{"url": "https://a/1.png"}]}) == ["https://a/1.png"]
    assert extract_urls({"output": {"video_url": "https://b/v.mp4"}}) == ["https://b/v.mp4"]
    assert extract_urls({"result": {"urls": ["https://c/1", "https://c/2"]}}) == [
        "https://c/1",
        "https://c/2",
    ]
    # 去重保序、不误伤非 URL
    assert extract_urls({"a": "https://x/1", "b": "https://x/1", "c": "not a url"}) == [
        "https://x/1"
    ]


# ---------------------------------------------------------------- 异步任务


async def test_提交后轮询直到完成():
    p = FakeMediaProvider(polls_needed=3, urls=["https://x/out.png"])
    task = await _gateway(p).generate("apimart", MediaKind.IMAGE, "m", "一只猫")

    assert task.ok
    assert task.status is TaskStatus.SUCCEEDED
    assert task.urls == ["https://x/out.png"]
    assert task.polls == 3, "该轮询几次就几次，不能提前认为完成"


async def test_同步返回的模型不进轮询():
    """部分图像模型直接返回结果，不该白等一个轮询周期。"""
    p = FakeMediaProvider(sync=True, urls=["https://x/direct.png"])
    task = await _gateway(p).generate("apimart", MediaKind.IMAGE, "m", "猫")
    assert task.ok and task.polls == 0


async def test_提交失败立刻返回不轮询():
    p = FakeMediaProvider(fail_at="submit")
    task = await _gateway(p).generate("apimart", MediaKind.IMAGE, "m", "猫")
    assert not task.ok
    assert task.status is TaskStatus.FAILED
    assert task.polls == 0
    assert "模拟提交失败" in task.error


async def test_生成中途失败被捕获():
    p = FakeMediaProvider(fail_at="poll")
    task = await _gateway(p).generate("apimart", MediaKind.IMAGE, "m", "猫")
    assert task.status is TaskStatus.FAILED and "模拟生成失败" in task.error


async def test_超时会停下来而不是永远等():
    """视频动辄几分钟，但不能无上限等下去。"""
    p = FakeMediaProvider(polls_needed=10_000)
    task = await _gateway(p).generate(
        "apimart", MediaKind.VIDEO, "m", "海边日落", max_wait_s=0.08
    )
    assert task.status is TaskStatus.TIMEOUT
    assert "超时" in task.error
    assert task.polls > 0


# ---------------------------------------------------------------- 排队/运行双计时


class QueuedProvider:
    """前 queue_polls 次轮询都还在服务端排队（SUBMITTED），然后开跑、成功。

    模拟「提交即返 task_id，但服务端队列拥堵」的场景 —— 等依赖、等前序
    任务时任务就挂在这个状态。
    """

    name = "fake"

    def __init__(self, queue_polls: int = 3, run_polls: int = 2):
        self.queue_polls = queue_polls
        self.run_polls = run_polls

    async def submit(self, kind, model, prompt, **kw):
        return MediaTask(task_id="t_q", kind=kind, model=model)

    async def poll(self, task):
        task.polls += 1
        if task.polls <= self.queue_polls:
            task.status = TaskStatus.SUBMITTED
        elif task.polls <= self.queue_polls + self.run_polls:
            task.status = TaskStatus.RUNNING
        else:
            task.status = TaskStatus.SUCCEEDED
            task.urls = ["https://example.com/out.mp4"]
        return task

    async def close(self):
        return None


async def test_排队时间不吃生成预算():
    """生成超时的计时从任务真正开跑（状态变 running）才起算。
    排队 0.1s 已超过 0.05s 的生成预算 —— 旧逻辑会误报超时，新逻辑不该。"""
    p = QueuedProvider(queue_polls=8, run_polls=2)
    task = await _gateway(p, max_queue_s=10).generate(
        "apimart", MediaKind.VIDEO, "m", "海边日落", max_wait_s=0.05
    )
    assert task.status is TaskStatus.SUCCEEDED
    assert task.polls >= 10


async def test_排队watchdog超时_报排队而不是生成():
    """排队也不是无限等：watchdog 超了按「排队超时」报，和生成超时区分开。"""
    p = QueuedProvider(queue_polls=100_000)
    task = await _gateway(p, max_queue_s=0.05).generate(
        "apimart", MediaKind.VIDEO, "m", "海边日落", max_wait_s=30
    )
    assert task.status is TaskStatus.TIMEOUT
    assert "排队超时" in task.error


async def test_开跑后超预算_报生成超时():
    p = FakeMediaProvider(polls_needed=100_000)  # 第一次 poll 就是 RUNNING
    task = await _gateway(p).generate("apimart", MediaKind.VIDEO, "m", "海边日落", max_wait_s=0.05)
    assert task.status is TaskStatus.TIMEOUT
    assert "生成超时" in task.error


async def test_轮询间隔会退避():
    p = FakeMediaProvider(polls_needed=5)
    gw = MediaGateway(
        {"apimart": p}, EventBus(), poll_interval=0.01, poll_backoff=2.0, max_poll_interval=0.05
    )
    task = await gw.generate("apimart", MediaKind.VIDEO, "m", "x")
    assert task.ok  # 退避不影响最终完成


async def test_未配置的provider给出可读错误():
    gw = _gateway(FakeMediaProvider())
    task = await gw.generate("不存在的网关", MediaKind.IMAGE, "m", "x")
    assert task.status is TaskStatus.FAILED and "未配置 provider" in task.error


# ---------------------------------------------------------------- 模型目录


def test_目录从yaml加载且覆盖图与视频():
    c = MediaCatalog.load(CATALOG_PATH)
    assert c.provider == "apimart"
    assert {m.id for m in c.image} >= {"seedream-4-5", "gpt-image-2", "qwen-image-2.0"}
    assert {m.id for m in c.video} >= {"veo3.1-quality", "veo3.1-fast", "sora-2"}


def test_三档选型各取默认():
    c = MediaCatalog.load(CATALOG_PATH)
    assert c.choose(MediaKind.IMAGE, prefer="quality")[0] == "seedream-4-5"
    assert c.choose(MediaKind.IMAGE, prefer="fast")[0] == "qwen-image-2.0"
    assert c.choose(MediaKind.VIDEO, prefer="quality")[0] == "veo3.1-quality"
    assert c.choose(MediaKind.VIDEO, prefer="fast")[0] == "veo3.1-lite"


def test_显式指定模型优先():
    c = MediaCatalog.load(CATALOG_PATH)
    picked, why = c.choose(MediaKind.VIDEO, model="sora-2", prefer="fast")
    assert picked == "sora-2" and "指定" in why


def test_未知模型直接报错不静默替换():
    """悄悄换成别的模型会让人对着结果百思不得其解。"""
    c = MediaCatalog.load(CATALOG_PATH)
    picked, why = c.choose(MediaKind.IMAGE, model="midjourney-v9")
    assert picked == ""
    assert "未知模型" in why and "seedream-4-5" in why  # 列出可用的


def test_目录渲染进上下文足够紧凑():
    c = MediaCatalog.load(CATALOG_PATH)
    text = c.render(MediaKind.VIDEO)
    assert "veo3.1-quality" in text and "成本5/5" in text
    assert len(text) < 1400, "目录要紧凑，它是常驻上下文的"


# ---------------------------------------------------------------- functions


async def test_媒体function注册且权限正确():
    registry, _, _ = await _funcs(FakeMediaProvider())
    perms = {m.name: m.permission.value for m in registry.catalog()}
    assert set(perms) == {
        "list_media_models", "gen_image", "gen_video", "gen_images", "gen_videos",
        "media_tasks", "media_recover",  # 任务台账：查看 / 取回没拿到结果的任务（2026-09-23）
    }
    assert perms["list_media_models"] == "L-read"
    assert perms["media_tasks"] == "L-read"
    assert perms["media_recover"] == "L-write"  # 取回已付费的任务，不产生新费用
    assert perms["gen_image"] == "L-compute"  # 花钱
    assert perms["gen_video"] == "L-compute"
    # 批量版本同样花钱，权限不能松（2026-09-19 加，解决逐段调用串行）
    assert perms["gen_images"] == "L-compute" and perms["gen_videos"] == "L-compute"


async def test_列目录给模型看():
    registry, _, _ = await _funcs(FakeMediaProvider())
    r = await registry.invoke("list_media_models", {"kind": "all"})
    assert r.ok and "图像模型" in r.content and "视频模型" in r.content
    assert "擅长" not in r.content  # 用 strengths 原文，不啰嗦


async def test_生图产物落成资产并带血缘():
    p = FakeMediaProvider(urls=["https://x/a.png"])
    registry, store, _ = await _funcs(p)
    script = store.create("分镜脚本", summary="脚本")

    r = await registry.invoke(
        "gen_image",
        {"prompt": "露营帐篷夜景", "prefer": "fast", "parent_id": script.id, "summary": "封面"},
    )
    assert r.ok, r.error
    asset = store.get(r.asset_ref)
    assert asset.type is AssetType.IMAGE
    assert asset.uri == "https://x/a.png"
    assert asset.parent_ids == [script.id]  # 血缘挂在脚本下面
    assert asset.creator == "model:qwen-image-2.0"  # fast 档
    assert asset.gen_params["prompt"] == "露营帐篷夜景"


async def test_多候选各自成资产():
    """关键节点给人多个选择，别只给一个。"""
    p = FakeMediaProvider(urls=[f"https://x/{i}.png" for i in range(3)])
    registry, store, _ = await _funcs(p)
    r = await registry.invoke("gen_image", {"prompt": "封面", "n": 3})
    assert r.ok
    assert len(store.all()) == 3
    assert all(a.uri for a in store.all())


async def test_视频时长被模型上限裁剪():
    p = FakeMediaProvider(urls=["https://x/v.mp4"])
    registry, _, _ = await _funcs(p)
    await registry.invoke(
        "gen_video", {"prompt": "海边", "model": "veo3.1-fast", "duration": 30}
    )
    assert p.submitted[0]["duration"] == 8  # veo3.1 上限 8s


async def test_生成失败给出可诊断的错误():
    p = FakeMediaProvider(fail_at="poll")
    registry, store, _ = await _funcs(p)
    r = await registry.invoke("gen_image", {"prompt": "x"})
    assert not r.ok
    assert "生成失败" in r.error and "轮询" in r.error
    assert len(store) == 0  # 失败不产生垃圾资产


async def test_指定未知模型时不生成():
    p = FakeMediaProvider()
    registry, store, _ = await _funcs(p)
    r = await registry.invoke("gen_image", {"prompt": "x", "model": "不存在"})
    assert not r.ok and "未知模型" in r.error
    assert p.submitted == [], "没确认模型就不该发出请求"


async def test_候选数有上限():
    p = FakeMediaProvider(urls=["https://x/a.png"])
    registry, _, _ = await _funcs(p)
    await registry.invoke("gen_image", {"prompt": "x", "n": 99})
    assert p.submitted[0]["n"] == 4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

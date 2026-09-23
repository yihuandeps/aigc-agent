"""短剧批量渲染的并发与依赖排队。

- render_assets：角色/场景/道具无依赖 → 并发；服装等自己角色的主形象 → 排队
- render_shots：无「前序引入」的镜头并发；带 {第X集-Y场} 的按依赖排队；
  成片拼接按镜头原序，不按完成顺序
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.functions.drama import DramaFunctions
from aigc_agent.domain.generators.catalog import Concurrency, MediaCatalog
from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.tools.provider import ToolResult


class FakeRegistry:
    """按调用名造假产物，记录并发峰值与每个调用的起止时间。"""

    def __init__(
        self, store: AssetStore, delay: float = 0.05, delays: dict[str, float] | None = None
    ):
        self.store = store
        self.delay = delay
        self.delays = delays or {}  # 按 summary 区分耗时，用来打乱完成顺序
        self.active = 0
        self.max_active = 0
        self.timeline: dict[str, tuple[float, float]] = {}
        self.calls: dict[str, dict] = {}
        self.compose_clips: list[str] = []

    async def invoke(self, name, args):
        key = args.get("summary", name)
        self.calls[key] = args
        if name == "compose_video":
            self.compose_clips = list(args["clips"])
            a = self.store.create("mp4", summary="成片", creator="fake")
            a.uri = "https://fake/composed.mp4"
            self.store.put(a)
            return ToolResult(ok=True, content="拼好了", asset_ref=a.id)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        t0 = time.perf_counter()
        await asyncio.sleep(self.delays.get(key, self.delay))
        a = self.store.create("", summary=key, creator="fake")
        a.uri = f"https://fake/{a.id}"
        self.store.put(a)
        self.active -= 1
        self.timeline[key] = (t0, time.perf_counter())
        return ToolResult(ok=True, content="ok", asset_ref=a.id)


def _fns(store: AssetStore, fake: FakeRegistry, catalog: MediaCatalog | None = None):
    return DramaFunctions(None, store, registry=fake, catalog=catalog)


def _lib_asset(store: AssetStore) -> str:
    lib = {
        "characters": [
            {
                "baseRoleName": "安妮",
                "roleTotalDesc": "25岁酒店员工",
                "roleCostumeList": [
                    {"costumeName": "安妮-制服-[全集]", "costumeDesc": "红色制服"}
                ],
            },
            {"baseRoleName": "老王", "roleTotalDesc": "50岁经理", "roleCostumeList": []},
        ],
        "scenes": [{"name": "酒店走廊", "description": "顶奢酒店"}],
        "props": [{"name": "行李箱", "description": "银色"}],
    }
    return store.create(json.dumps(lib, ensure_ascii=False), summary="资产库", creator="t").id


def _shots_asset(store: AssetStore, shots: list[dict]) -> str:
    return store.create(
        json.dumps(shots, ensure_ascii=False), summary="提示词", creator="t"
    ).id


# ---------------------------------------------------------------- render_assets


async def test_资产生图_无依赖并发_服装等主形象():
    store = AssetStore()
    fake = FakeRegistry(store)
    r = await _fns(store, fake)._fn_drama_render_assets(_lib_asset(store))

    assert r.ok
    # 第一层 4 个无依赖任务（2 角色 + 场景 + 道具）并发跑满
    assert fake.max_active == 4
    # 严格分批：服装不仅要等自己角色的主形象，要等**所有**无依赖图都完成
    phase_a_end = max(end for k, (_, end) in fake.timeline.items() if not k.startswith("服装"))
    assert fake.timeline["服装·安妮-制服-[全集]"][0] >= phase_a_end
    # 服装的参考图就是主形象的 url
    char_asset_url = fake.timeline and fake.calls["服装·安妮-制服-[全集]"]["image"][0]
    assert char_asset_url.startswith("https://fake/as_")
    assert "✓ 角色 安妮" in r.content and "✓ 服装 安妮-制服-[全集]" in r.content


async def test_资产生图_并发上限生效():
    store = AssetStore()
    fake = FakeRegistry(store)
    catalog = MediaCatalog(concurrency=Concurrency(image=2))
    r = await _fns(store, fake, catalog)._fn_drama_render_assets(_lib_asset(store))

    assert r.ok
    assert fake.max_active == 2  # 4 个第一层任务被信号量压在 2


async def test_资产生图_主形象失败则服装跳过():
    store = AssetStore()

    class FailA(FakeRegistry):
        async def invoke(self, name, args):
            if args.get("summary") == "角色·安妮":
                return ToolResult(ok=False, error="模拟生图失败")
            return await super().invoke(name, args)

    fake = FailA(store)
    r = await _fns(store, fake)._fn_drama_render_assets(_lib_asset(store))

    assert r.ok  # 老王/场景/道具成了，不算全灭
    assert "角色 安妮：模拟生图失败" in r.content
    assert "服装 安妮-制服-[全集]：缺角色 安妮 的主形象" in r.content
    assert "服装·安妮-制服-[全集]" not in fake.timeline  # 根本没去生成


# ---------------------------------------------------------------- render_shots

SHOTS = [
    {
        "scene_index": "[第1集-1场]",
        "video_name": "1-2",
        "video_duration": "8s",
        "description": "开场 (安妮-制服-[全集])",
    },
    {
        "scene_index": "[第1集-2场]",
        "video_name": "3-4",
        "video_duration": "8s",
        "description": "接上场的尾帧 {第1集-1场} (安妮-制服-[全集])",
    },
    {
        "scene_index": "[第2集-1场]",
        "video_name": "1-2",
        "video_duration": "8s",
        "description": "新集开场 (老王)",
    },
]


async def test_分镜视频_引入排队_其余并发_成片按原序():
    store = AssetStore()
    # 第1集-1场故意放慢：若没并发，整体顺序会是 0 → 1 → 2
    fake = FakeRegistry(store, delays={"[第1集-1场] 1-2": 0.12}, delay=0.03)
    r = await _fns(store, fake)._fn_drama_render_shots(_shots_asset(store, SHOTS))

    assert r.ok, r.error
    s0 = fake.timeline["[第1集-1场] 1-2"]
    s1 = fake.timeline["[第1集-2场] 3-4"]
    s2 = fake.timeline["[第2集-1场] 1-2"]

    # 无引入的第2集与慢的第1集-1场并发：它开始时慢的还没跑完
    assert s2[0] < s0[1]
    # 带 {第1集-1场} 引入的镜头排队：等被引入段完成才开始
    assert s1[0] >= s0[1]
    # 引入段的视频 url 进了它的参考视频列表（2026-09-18 起视频参考走 video_urls，
    # 提示词里 @视频1 对应它；参考图列表只放图）
    s0_url = fake.calls["[第1集-1场] 1-2"]  # 生成时返回的 url 落在资产上
    a0 = next(a for a in store.all() if a.summary == "[第1集-1场] 1-2")
    assert a0.uri in fake.calls["[第1集-2场] 3-4"]["video_urls"]
    assert "image" not in fake.calls["[第1集-2场] 3-4"]
    assert s0_url  # 占位，url 断言在上面

    # 成片拼接按镜头原序，与完成顺序无关（第2集先完成也不能排到前面）
    ids = [next(a for a in store.all() if a.summary == k).id
           for k in ("[第1集-1场] 1-2", "[第1集-2场] 3-4", "[第2集-1场] 1-2")]
    assert fake.compose_clips == ids


async def test_分镜视频_严格分批_依赖层等所有无依赖完成():
    """用户定的顺序约束：所有无前置依赖的镜头完全生成完后，需要前置依赖的才开始。"""
    store = AssetStore()
    fake = FakeRegistry(store, delays={"[第1集-1场] 1-2": 0.10}, delay=0.02)
    r = await _fns(store, fake)._fn_drama_render_shots(_shots_asset(store, SHOTS))

    assert r.ok, r.error
    s0 = fake.timeline["[第1集-1场] 1-2"]  # 无依赖，慢
    s1 = fake.timeline["[第1集-2场] 3-4"]  # 依赖 s0
    s2 = fake.timeline["[第2集-1场] 1-2"]  # 无依赖，快，但与 s1 无关
    # 依赖镜头不只等它引入的那一段，要等**所有**无依赖镜头完成 —— 包括无关的 s2
    assert s1[0] >= s0[1] and s1[0] >= s2[1]


async def test_分镜视频_引入指向不存在场景也排到依赖层():
    """声明了引入就是「需要前置依赖」的内容 —— 指不到已产出场景也不能混进第一批。"""
    store = AssetStore()
    fake = FakeRegistry(store, delays={"[第1集-1场] 1-2": 0.10}, delay=0.02)
    shots = [
        SHOTS[0],  # 无依赖（慢）
        {
            "scene_index": "[第9集-1场]",
            "video_name": "1-2",
            "video_duration": "8s",
            "description": "引入 {第9集-0场} (老王)",  # 指向不存在的场景
        },
    ]
    r = await _fns(store, fake)._fn_drama_render_shots(_shots_asset(store, shots))

    assert r.ok, r.error
    assert fake.timeline["[第9集-1场] 1-2"][0] >= fake.timeline["[第1集-1场] 1-2"][1]


async def test_分镜视频_并发上限生效():
    store = AssetStore()
    fake = FakeRegistry(store)
    catalog = MediaCatalog(concurrency=Concurrency(video=1))
    shots = [SHOTS[0], SHOTS[2]]  # 两段互相独立
    r = await _fns(store, fake, catalog)._fn_drama_render_shots(_shots_asset(store, shots))

    assert r.ok, r.error
    assert fake.max_active == 1  # 上限 1 = 退化成串行


# ---------------------------------------------------------------- 按集渲染（流水线）


async def test_分镜视频_按集过滤只渲染该集():
    """按集流水：episode=2 只渲染第2集的镜头，成片默认 第2集.mp4。"""
    store = AssetStore()
    fake = FakeRegistry(store, delay=0.01)
    r = await _fns(store, fake)._fn_drama_render_shots(_shots_asset(store, SHOTS), episode=2)

    assert r.ok, r.error
    assert list(fake.timeline) == ["[第2集-1场] 1-2"]  # 第1集的镜头没来
    assert fake.calls["compose_video"]["filename"] == "第02集.mp4"  # 补零，60 集才排得对
    clips = next(a for a in store.all() if a.creator == "tool:drama_render_shots")
    assert clips.gen_params["episode"] == 2  # 产物按集标记，管线靠它认


async def test_分镜视频_按集过滤_集号前缀不串集():
    """episode=1 不能误匹配「第12集-」——前缀必须带「集-」后缀。"""
    store = AssetStore()
    fake = FakeRegistry(store, delay=0.01)
    shots = SHOTS[:2] + [
        {
            "scene_index": "[第12集-1场]",
            "video_name": "1-2",
            "video_duration": "8s",
            "description": "别的集 (老王)",
        }
    ]
    r = await _fns(store, fake)._fn_drama_render_shots(_shots_asset(store, shots), episode=1)

    assert r.ok, r.error
    assert sorted(fake.timeline) == ["[第1集-1场] 1-2", "[第1集-2场] 3-4"]
    assert fake.calls["compose_video"]["filename"] == "第01集.mp4"


async def test_分镜视频_按集过滤_没这集就报错():
    store = AssetStore()
    fake = FakeRegistry(store, delay=0.01)
    r = await _fns(store, fake)._fn_drama_render_shots(_shots_asset(store, SHOTS), episode=9)

    assert not r.ok and "第 9 集" in r.error
    assert fake.timeline == {}  # 一段都没渲


# ---------------------------------------------------------------- 进度事件


async def test_渲染资产发批量进度事件():
    bus = EventBus()
    store = AssetStore()
    fake = FakeRegistry(store)
    fns = DramaFunctions(None, store, registry=fake, bus=bus)
    await fns._fn_drama_render_assets(_lib_asset(store))

    evs = [e for e in bus.history if e.type is EventType.BATCH_PROGRESS]
    # 2 角色 + 1 场景 + 1 道具 + 1 服装 = 5
    assert evs[0].data["done"] == 0 and evs[0].data["total"] == 5
    assert [e.data["done"] for e in evs] == list(range(6))  # 0→5 单调递增
    assert all(e.data["stage"] == "渲染参考图" for e in evs)


async def test_渲染分镜发批量进度事件():
    bus = EventBus()
    store = AssetStore()
    fake = FakeRegistry(store)
    fns = DramaFunctions(None, store, registry=fake, bus=bus)
    await fns._fn_drama_render_shots(_shots_asset(store, SHOTS))

    evs = [e for e in bus.history if e.type is EventType.BATCH_PROGRESS]
    assert evs[0].data["done"] == 0 and evs[0].data["total"] == 3
    assert evs[-1].data["done"] == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

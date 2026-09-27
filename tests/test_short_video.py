"""抖音短视频分支（2026-09-18 用户定的路径）。

关键词 + 风格 → RPA 热点 → 归纳成新内容（简报）→ 实拍素材（问人 / 联网）→ 出片 → 看片。
这里验：简报解析与时长决策、素材问答的解析、以及整条工具链用假网关/假注册表跑通。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions.short_video import ShortVideoFunctions
from aigc_agent.domain.pipeline.recipe import load_recipe
from aigc_agent.domain.pipeline.short_video import (
    Brief,
    Shot,
    brief_prompt,
    materials_question,
    parse_brief,
    parse_material_reply,
)
from aigc_agent.harness.events.bus import Event, EventType
from aigc_agent.harness.tools.provider import ToolResult

RECIPES = Path(__file__).resolve().parents[1] / "config" / "recipes"


def _plan(**over: Any) -> dict[str, Any]:
    base = {
        "angle": "从一条热搜看芯片产能",
        "title": "AI 芯片为什么不够用",
        "summary": "三条热搜指向同一件事：产能被抢光了。",
        "key_facts": [{"fact": "热搜第 1 条热度 120 万", "source": "抖音热榜#1"}],
        "hook": "你买不到的不是显卡，是产能。",
        "duration_seconds": 30,
        "why_duration": "三个信息点",
        "script": "你买不到的不是显卡，是产能。" * 4,
        "shots": [
            {"desc": "晶圆厂机械臂", "seconds": 5, "source": "generate"},
            {"desc": "真实的芯片产品特写", "seconds": 5, "source": "real",
             "real_need": "一块真实的 GPU 板卡实拍", "search_terms": ["gpu card close up"]},
            {"desc": "数据中心机柜灯光", "seconds": 5, "source": "generate"},
        ],
        "risks": ["产能数字来自热搜标题，未经核实"],
    }
    base.update(over)
    return base


# ---------------------------------------------------------------- 简报解析


def test_解析简报_时长夹在风格范围内_口播过长就上调():
    b, warns = parse_brief(json.dumps(_plan(duration_seconds=80)), "AI 芯片", "tech-short", 20, 45)
    assert b is not None and b.duration == 45 and any("超出风格范围" in w for w in warns)
    long_script = "字" * 200  # 200 字 ≈ 45s
    b2, warns2 = parse_brief(
        json.dumps(_plan(duration_seconds=20, script=long_script)), "k", "tech-short", 20, 45
    )
    assert b2 is not None and b2.duration == 45 and any("调到 45s" in w for w in warns2)
    b3, warns3 = parse_brief(json.dumps(_plan(duration_seconds=0)), "k", "tech-short", 20, 45)
    assert b3 is not None and 20 <= b3.duration <= 45 and any("按口播字数" in w for w in warns3)
    assert parse_brief("不是 json", "k", "s", 20, 45)[0] is None
    assert parse_brief(json.dumps({"shots": []}), "k", "s", 20, 45)[0] is None


def test_镜头归一_来源别名与秒数均分():
    plan = _plan(shots=[
        {"desc": "a", "source": "stock"}, {"desc": "b", "source": "AI"}, {"desc": "c"},
    ], duration_seconds=30)
    b, _ = parse_brief(json.dumps(plan), "k", "tech-short", 20, 45)
    assert [s.source for s in b.shots] == ["real", "generate", "generate"]
    assert all(s.seconds == 10.0 for s in b.shots)
    assert b.material_needs()[0][0] == 1
    back = Brief.from_dict(b.as_dict())
    assert back.shots[0].source == "real" and back.duration == 30


def test_提示词带热点材料与风格范围():
    p = brief_prompt("AI 芯片", "资讯快切", "讲一件事", "1. 热搜A 120万", 20, 45, notes="要幽默")
    for kw in ("热搜A", "20–45 秒", "source=real", "search_terms", "要幽默", "不要编"):
        assert kw in p, kw
    assert "没有抓到任何热点数据" in brief_prompt("k", "s", "d", "", 20, 45)


def test_素材问法与回复解析():
    b, _ = parse_brief(json.dumps(_plan()), "k", "tech-short", 20, 45)
    q = materials_question(b)
    assert "第2镜" in q and "gpu card close up" in q and "联网找" in q and "生成" in q
    got = parse_material_reply("第2镜 E:\\素材\\gpu.mp4", [2])
    assert got == {2: ("file", "E:\\素材\\gpu.mp4")}
    got = parse_material_reply("2: 联网找；3、生成", [2, 3])
    assert got == {2: ("online", ""), 3: ("generate", "")}
    # 目录名里带"生成""联网"也得认成路径
    got = parse_material_reply("第2镜 E:\\AI生成\\联网找的\\gpu.mp4", [2])
    assert got == {2: ("file", "E:\\AI生成\\联网找的\\gpu.mp4")}
    assert parse_material_reply("都联网找", [2, 3]) == {2: ("online", ""), 3: ("online", "")}
    assert parse_material_reply("全部生成吧", [2]) == {2: ("generate", "")}
    assert parse_material_reply("嗯", [2]) == {}
    assert parse_material_reply("第9镜 生成", [2]) == {}


def test_风格信息来自配方():
    r = load_recipe("tech-short", RECIPES)
    assert r.style_label == "资讯快切" and r.duration_bounds == (20, 45)
    assert "AI 生成" in r.style_desc
    assert "颗粒" in r.shot_prompt("晶圆厂", "subtle"), "realism: auto 用全局档位"
    assert "满脸雀斑" not in r.shot_prompt("晶圆厂", "subtle")
    th = load_recipe("talking-head", RECIPES)
    assert th.style.get("footage") == "real" and th.duration_bounds == (30, 60)


# ---------------------------------------------------------------- 工具链


class Gateway:
    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan = plan
        self.calls: list[tuple[str, str]] = []

    async def chat(self, role: str, messages: list[dict[str, Any]], **_: Any) -> Any:
        self.calls.append((role, messages[-1]["content"]))
        if role == "voice_select":
            return SimpleNamespace(
                text='{"voice": "Chinese (Mandarin)_News_Anchor", "speed": 1.0, "why": "资讯"}'
            )
        return SimpleNamespace(text=json.dumps(self.plan, ensure_ascii=False))


class Registry:
    def __init__(self, store: AssetStore, fail_once: dict[str, str] | None = None) -> None:
        self.store = store
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail_once = dict(fail_once or {})

    def of(self, name: str) -> list[dict[str, Any]]:
        return [a for n, a in self.calls if n == name]

    async def invoke(self, name: str, args: dict[str, Any]) -> ToolResult:
        self.calls.append((name, dict(args)))
        key = args.get("summary", "")
        if name == "gen_video":
            if key in self.fail_once:
                return ToolResult(ok=False, error=self.fail_once.pop(key))
            a = self.store.create("", type_=AssetType.VIDEO, summary=key, creator="model:x")
            a.uri = f"https://fake/{a.id}.mp4"
            self.store.put(a)
            return ToolResult(ok=True, content="ok", asset_ref=a.id)
        if name == "tts":
            a = self.store.create("", type_=AssetType.AUDIO, summary=key, creator="model:tts")
            a.uri = f"https://fake/{a.id}.mp3"
            self.store.put(a)
            return ToolResult(ok=True, content="ok", asset_ref=a.id)
        if name == "transcribe":
            srt = "1\n00:00:00,000 --> 00:00:04,000\n你买不到的不是显卡是产能\n"
            a = self.store.create(srt, type_=AssetType.SUBTITLE, summary="字幕")
            return ToolResult(ok=True, content="ok", asset_ref=a.id)
        if name == "compose_video":
            a = self.store.create("", type_=AssetType.VIDEO, summary="成片", creator="tool:compose")
            a.uri = "E:/out/成片.mp4"
            self.store.put(a)
            return ToolResult(ok=True, content="成片已导出：E:/out/成片.mp4", asset_ref=a.id)
        return ToolResult(ok=False, error=f"unknown {name}")


def _catalog() -> Any:
    return SimpleNamespace(
        drama={"realism_level": "subtle", "video_retries": "1"},
        max_concurrency=lambda kind: 0,
        speech_voices=[SimpleNamespace(name="Chinese (Mandarin)_News_Anchor", note="女声新闻")],
        speech_default_voice="Chinese (Mandarin)_News_Anchor",
    )


def _fns(
    store: AssetStore, reg: Registry, plan: dict[str, Any] | None = None
) -> ShortVideoFunctions:
    return ShortVideoFunctions(
        Gateway(plan or _plan()), store, registry=reg, catalog=_catalog(), recipes_dir=RECIPES
    )


async def test_风格列表():
    store = AssetStore()
    r = await _fns(store, Registry(store)).invoke("list_video_styles", {})
    assert r.ok and "tech-short" in r.content and "资讯快切" in r.content and "20–45s" in r.content


async def test_简报_用热点资产归纳_标出实拍需求():
    store = AssetStore()
    hot = store.create(
        "1. 热搜A 热度 120 万\n2. 热搜B",
        summary="抖音热榜（RPA）·2条",
        creator="tool:douyin_hot_rpa",
    )
    fns = _fns(store, Registry(store))
    args = {"keyword": "AI 芯片", "style": "tech-short", "sources": [hot.id, "as_nope"]}
    r = await fns.invoke("short_video_brief", args)
    assert r.ok, r.error
    role, prompt = fns.gateway.calls[0]  # type: ignore[attr-defined]
    assert role == "short_video_planner" and "热搜A" in prompt and "资讯快切" in prompt
    assert "需要实拍/外部素材的镜头：1 个" in r.content and "request_materials" in r.content
    assert "as_nope" in r.content  # 不存在的来源要报出来
    brief = store.get(r.asset_ref)
    assert brief.type is AssetType.OUTLINE and brief.parent_ids == [hot.id]
    assert brief.gen_params["real_shots"] == 1


async def test_要素材_挂起且auto模式也停():
    store = AssetStore()
    fns = _fns(store, Registry(store))
    bid = (await fns.invoke("short_video_brief", {"keyword": "k", "style": "tech-short"})).asset_ref
    r = await fns.invoke("request_materials", {"brief_id": bid})
    assert r.ok and r.suspend and r.suspend_payload["major"] is True
    assert r.suspend_payload["stage"] == "素材" and r.suspend_payload["assets"] == [bid]
    assert r.suspend_payload["materials"][0]["shot"] == 2
    assert "第2镜" in r.suspend_payload["question"]
    r2 = await fns.invoke("request_materials", {"brief_id": bid, "shots": [1]})
    assert r2.ok and not r2.suspend and "没有需要实拍" in r2.content


async def test_读回复_文件登记_改生成_联网给搜索词(tmp_path: Path):
    store = AssetStore()
    fns = _fns(store, Registry(store))
    bid = (await fns.invoke("short_video_brief", {"keyword": "k", "style": "tech-short"})).asset_ref
    clip = tmp_path / "gpu.mp4"
    await asyncio.to_thread(clip.write_bytes, b"mp4")
    r = await fns.invoke("resolve_materials", {"brief_id": bid, "reply": f"第2镜 {clip}"})
    assert r.ok, r.error
    new = json.loads(store.content(r.asset_ref))
    mat = new["materials"]["2"]
    assert store.get(mat).type is AssetType.VIDEO and "已登记" in r.content
    r = await fns.invoke("resolve_materials", {"brief_id": r.asset_ref, "reply": "第2镜 联网找"})
    assert r.ok and "gpu card close up" in r.content and "stock_media_search" in r.content
    r = await fns.invoke("resolve_materials", {"brief_id": r.asset_ref, "reply": "第2镜 生成"})
    assert r.ok and json.loads(store.content(r.asset_ref))["shots"][1]["source"] == "generate"
    bad = await fns.invoke("resolve_materials", {"brief_id": bid, "reply": "嗯"})
    assert not bad.ok and "没读懂" in bad.error


async def test_出片_素材直用_缺的并发生成_配音字幕合成_重跑复用(tmp_path: Path):
    store = AssetStore()
    # 提交时连不上服务端（请求没送到）才原样重提；轮询失败不重提（任务已在服务端计费）
    reg = Registry(
        store, fail_once={"AI 芯片为什么不够用·第1镜": "提交失败（连不上服务端）：ConnectError: x"}
    )
    fns = _fns(store, reg)
    made = await fns.invoke("short_video_brief", {"keyword": "AI 芯片", "style": "tech-short"})
    clip = tmp_path / "gpu.mp4"
    await asyncio.to_thread(clip.write_bytes, b"mp4")
    reply = {"brief_id": made.asset_ref, "reply": f"第2镜 {clip}"}
    bid = (await fns.invoke("resolve_materials", reply)).asset_ref

    # 要新生成镜头：先停下来报镜头数和秒数，一分钱没花（2026-09-23 审查）
    ask = await fns.invoke("short_video_produce", {"brief_id": bid})
    assert ask.suspend and ask.suspend_payload["major"] is True and not reg.of("gen_video")
    assert "要新生成 2 段视频" in ask.suspend_payload["question"]
    # 人在确认单上采纳（总线 CHECKPOINT_DECIDED）之后，confirm=true 才生效（2026-09-26）
    fns.on_event(Event(type=EventType.CHECKPOINT_DECIDED, data={
        "node": "出片确认", "decision": "adopt", "decided_by": "human", "candidates": [bid],
    }))
    r = await fns.invoke("short_video_produce", {"brief_id": bid, "confirm": True})
    assert r.ok, r.error
    gens = reg.of("gen_video")
    assert [g["summary"] for g in gens] == [
        "AI 芯片为什么不够用·第1镜", "AI 芯片为什么不够用·第1镜", "AI 芯片为什么不够用·第3镜"
    ] or sorted(g["summary"] for g in gens) == sorted([
        "AI 芯片为什么不够用·第1镜", "AI 芯片为什么不够用·第1镜", "AI 芯片为什么不够用·第3镜"
    ])
    assert all("颗粒" in g["prompt"] and "冷蓝科技色调" in g["prompt"] for g in gens)
    assert gens[0]["local_name"].endswith("第01镜")
    assert reg.of("tts")[0]["voice"] == "Chinese (Mandarin)_News_Anchor"
    assert reg.of("transcribe")[0]["format"] == "srt"
    compose = reg.of("compose_video")[0]
    assert len(compose["clips"]) == 3 and compose["max_cut_seconds"] == 3.0
    assert compose["total_seconds"] == 30 and compose["audio_id"] and compose["subtitle_id"]
    material = json.loads(store.content(bid))["materials"]["2"]
    assert compose["clips"][1] == material, "第 2 镜用的是用户给的素材"
    assert "字幕：按原稿校正" in r.content and "view_video" in r.content
    record = next(a for a in store.all() if a.creator == "tool:short_video_produce")
    assert record.parent_ids == [bid]

    # 重跑：全部复用，不再生成
    reg.calls.clear()
    r2 = await fns.invoke("short_video_produce", {"brief_id": bid, "no_voiceover": True})
    assert r2.ok and not reg.of("gen_video") and not reg.of("tts")
    assert len(reg.of("compose_video")[0]["clips"]) == 3


async def test_没给实拍素材_改用生成并标出_图片素材做成镜头(tmp_path: Path):
    import aigc_agent.domain.functions.short_video as mod

    store = AssetStore()
    reg = Registry(store)
    fns = _fns(store, reg)
    bid = (await fns.invoke("short_video_brief", {"keyword": "k", "style": "tech-short"})).asset_ref
    fns.approve(bid)  # 人在终端点了头（2026-09-26：confirm=true 只认人的确认）
    r = await fns.invoke("short_video_produce", {"brief_id": bid, "confirm": True})
    assert r.ok and "第2镜需要实拍素材但没提供" in r.content
    assert len(reg.of("gen_video")) == 3

    img = tmp_path / "board.png"
    await asyncio.to_thread(img.write_bytes, b"png")
    photo = store.create(
        "", type_=AssetType.IMAGE, summary="板卡照片", gen_params={"local": str(img)}
    )
    photo.uri = str(img)
    store.put(photo)

    async def fake_still(
        image: Path, out: Path, seconds: float, size: Any = None
    ) -> tuple[bool, str]:
        def write() -> None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"clip")

        await asyncio.to_thread(write)
        return True, ""

    orig = mod.ffmpeg.still_to_clip
    mod.ffmpeg.still_to_clip = fake_still  # type: ignore[assignment]
    try:
        reg.calls.clear()
        fns.approve(bid)
        r = await fns.invoke(
            "short_video_produce",
            {"brief_id": bid, "materials": {"2": photo.id}, "reuse": False, "confirm": True},
        )
    finally:
        mod.ffmpeg.still_to_clip = orig  # type: ignore[assignment]
    assert r.ok, r.error
    compose = reg.of("compose_video")[0]
    made = store.get(compose["clips"][1])
    assert made.type is AssetType.VIDEO and made.parent_ids == [photo.id]
    assert await asyncio.to_thread(Path(made.gen_params["local"]).exists)


def test_权限分级():
    store = AssetStore()
    fns = _fns(store, Registry(store))
    metas = {s.name: s.permission.value for s in fns._specs.values()}
    assert metas["list_video_styles"] == "L-read"
    assert metas["short_video_brief"] == "L-compute" and metas["short_video_produce"] == "L-compute"
    assert metas["request_materials"] == "L-write" and metas["resolve_materials"] == "L-write"
    assert isinstance(Shot("x"), Shot)

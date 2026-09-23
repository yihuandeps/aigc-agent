"""M14 成片链路验收 —— 真 ffmpeg，假素材。

短视频链路里生成那一半（gen_video / tts / transcribe）都有假 provider 盯着，
但最后一公里 —— 下载、拼接、混音、烧字幕、导出 —— 之前**没有任何测试**。
磁盘上有十几次真实成片记录说明它能跑，但那不是护栏：改一处 ffmpeg 参数
没人知道会不会把音轨丢掉（这事真发生过，见 media/ffmpeg.py concat 的注释）。

素材用 ffmpeg 的 lavfi 源现场合成（纯色画面 + 正弦音），不依赖任何模型和网络。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions.video_edit import VideoEditFunctions, _safe_name
from aigc_agent.domain.media import ffmpeg
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.execution.trace import ExecutionTrace
from aigc_agent.harness.tools.registry import ToolRegistry

pytestmark = pytest.mark.skipif(not ffmpeg.have_ffmpeg(), reason="需要 ffmpeg/ffprobe")

SRT = """1
00:00:00,000 --> 00:00:01,500
第一句字幕

2
00:00:01,500 --> 00:00:03,500
第二句字幕
"""


async def _clip(path: Path, seconds: float, color: str) -> None:
    code, err = await ffmpeg.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=240x426:d={seconds}:r=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ])
    assert code == 0, err


async def _tone(path: Path, seconds: float) -> None:
    code, err = await ffmpeg.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-c:a", "libmp3lame", str(path),
    ])
    assert code == 0, err


async def _setup(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    await _clip(src / "a.mp4", 2.0, "red")
    await _clip(src / "b.mp4", 2.0, "blue")
    await _tone(src / "voice.mp3", 3.0)

    store = AssetStore(tmp_path / "assets")

    def blob(path: Path, type_: AssetType, mime: str):
        a = store.create("", type_=type_, summary=path.name, creator="test")
        a.uri, a.mime = str(path), mime
        return store.put(a)

    clips = [blob(src / "a.mp4", AssetType.VIDEO, "video/mp4"),
             blob(src / "b.mp4", AssetType.VIDEO, "video/mp4")]
    voice = blob(src / "voice.mp3", AssetType.AUDIO, "audio/mpeg")
    sub = store.create(SRT, type_=AssetType.SUBTITLE, summary="字幕")

    bus = EventBus()
    trace = ExecutionTrace()
    trace.attach(bus)
    registry = ToolRegistry(bus)
    registry.register(VideoEditFunctions(store, tmp_path / "ws"))
    await registry.refresh()
    return registry, store, trace, clips, voice, sub


async def test_探测媒体信息(tmp_path: Path):
    registry, _, _, clips, voice, _ = await _setup(tmp_path)
    r = await registry.invoke("probe_media", {"asset_ids": [c.id for c in clips] + [voice.id]})
    assert r.ok, r.error
    assert "240x426" in r.content and "有音轨" in r.content and "无音轨" in r.content
    assert "合计时长 7." in r.content  # 2 + 2 + 3


async def test_顺序拼接_配音_字幕_导出并带血缘(tmp_path: Path):
    registry, store, trace, clips, voice, sub = await _setup(tmp_path)
    out_dir = tmp_path / "out"
    r = await registry.invoke(
        "compose_video",
        {
            "clips": [c.id for c in clips],
            "audio_id": voice.id,
            "subtitle_id": sub.id,
            "out_dir": str(out_dir),
            "filename": "测试_成片.mp4",
            "keep_whole": True,  # 整段顺序拼（不传切点时默认按 ≤3s 快切）
        },
    )
    assert r.ok, r.error
    final = out_dir / "测试_成片.mp4"
    assert final.exists() and final.stat().st_size > 0
    assert "烧字幕" in r.content and "字幕失败" not in r.content

    info = await ffmpeg.probe(final)
    assert abs(info.duration - 4.0) < 0.35, f"画面 2+2 秒，配音按画面截断：{info.duration}"
    assert info.has_audio, "配音丢了 —— 这是之前真踩过的坑"
    assert (info.width, info.height) == (240, 426)

    asset = store.get(r.asset_ref)
    assert asset.type is AssetType.VIDEO
    assert asset.parent_ids == [c.id for c in clips] + [voice.id, sub.id], "血缘挂全"
    assert asset.creator == "tool:compose_video"

    # 执行痕迹按资产依赖连边：成片这次调用消费了四份资产
    node = trace.producer_of(asset.id)
    assert node is not None and set(node.inputs) >= {clips[0].id, voice.id, sub.id}


async def test_没传切点默认按全局上限快切_短剧片段不切(tmp_path: Path):
    """2026-09-23 审查：max_cut_seconds 省略就整段拼 —— 配方关了快切的片子一镜 8 秒。"""
    registry, store, _, clips, _, _ = await _setup(tmp_path)
    r = await registry.invoke(
        "compose_video",
        {"clips": [c.id for c in clips], "out_dir": str(tmp_path / "o1"), "filename": "a.mp4"},
    )
    assert r.ok, r.error
    assert "没传切点" in r.content and "刀" in r.content
    # 短剧片段（带 shots_id 标签）生成时已按时间线硬切过，拼接不能再切
    for c in clips:
        c.gen_params["tags"] = {"shots_id": "as_x", "scene": "[第1集-1场]"}
        store.put(c)
    r2 = await registry.invoke(
        "compose_video",
        {"clips": [c.id for c in clips], "out_dir": str(tmp_path / "o2"), "filename": "b.mp4"},
    )
    assert r2.ok, r2.error
    assert store.get(r2.asset_ref).gen_params["steps"] == ["拼接 2 段"], "整段拼，没有再切"


async def test_同名导出不覆盖(tmp_path: Path):
    registry, _, _, clips, _, _ = await _setup(tmp_path)
    args = {"clips": [clips[0].id], "out_dir": str(tmp_path / "out"), "filename": "同名.mp4",
            "keep_whole": True}
    first = await registry.invoke("compose_video", args)
    second = await registry.invoke("compose_video", args)
    assert first.ok and second.ok
    assert (tmp_path / "out" / "同名.mp4").exists() and (tmp_path / "out" / "同名-v2.mp4").exists()
    assert second.content != first.content


async def test_口播出镜_原声做音轨_BROLL穿插_画幅固定(tmp_path: Path):
    """2026-09-23 审查：口播出镜配方成片没声音没字幕，出镜画面被切碎轮转。"""
    registry, store, _, clips, _, _ = await _setup(tmp_path)
    talk = tmp_path / "src" / "talk.mp4"
    code, err = await ffmpeg.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=green:s=426x240:d=5:r=30",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=5", "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(talk),
    ])
    assert code == 0, err
    a = store.create("", type_=AssetType.VIDEO, summary="出镜", creator="human:import")
    a.uri = str(talk)
    store.put(a)
    r = await registry.invoke(
        "compose_video",
        {"clips": [c.id for c in clips], "aroll_id": a.id, "out_dir": str(tmp_path / "out"),
         "filename": "talk.mp4", "max_cut_seconds": 1.5, "min_cut_seconds": 0.8,
         "aspect_ratio": "9:16"},
    )
    assert r.ok, r.error
    assert "出镜" in r.content and "B-roll" in r.content
    info = await ffmpeg.probe(tmp_path / "out" / "talk.mp4")
    assert info.has_audio, "出镜人的原声就是音轨"
    assert abs(info.duration - 5.0) < 0.4, info.duration
    assert (info.width, info.height) == (1080, 1920), "按画幅定画布，横屏素材裁切铺满"


async def test_快切模式镜头不超上限且总长对齐(tmp_path: Path):
    registry, _, _, clips, _, _ = await _setup(tmp_path)
    r = await registry.invoke(
        "compose_video",
        {
            "clips": [c.id for c in clips],
            "out_dir": str(tmp_path / "out"),
            "filename": "cut.mp4",
            "max_cut_seconds": 1.0,
            "min_cut_seconds": 0.5,
            "total_seconds": 3.0,
        },
    )
    assert r.ok, r.error
    assert "刀" in r.content, "快切模式要报剪辑表"
    info = await ffmpeg.probe(tmp_path / "out" / "cut.mp4")
    assert abs(info.duration - 3.0) < 0.35


async def test_缺失素材给出可读错误(tmp_path: Path):
    registry, _, _, _, _, _ = await _setup(tmp_path)
    r = await registry.invoke("compose_video", {"clips": ["as_0000000000"]})
    assert not r.ok and "as_0000000000" in (r.error or "")


async def test_文件名清洗防目录穿越():
    assert _safe_name("../../evil.mp4") == "evil.mp4"
    assert _safe_name("科技热点_30s") == "科技热点_30s.mp4"
    assert _safe_name("") == ""

"""每个镜头不超过 3 秒（2026-09-20，用户定的硬性要求：不管一集多长，单镜必须 ≤3 秒）。

四层：写剧本按短拍写（writing_rules）→ 分镜每行标时长 + 检查（test_episode_format.py）
→ 提示词每段 cuts + 检查（同上）→ 渲染：提示词最外层包快切硬约束与时间线，生成后 ffmpeg
场景切换检测量最长镜头，超了重生成一次，还超标出来交人复核。配方短视频的快切上限也不超过它。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.drama.format import EpisodeFormat, global_max_cut
from aigc_agent.domain.functions.drama import DramaFunctions
from aigc_agent.domain.media import ffmpeg
from aigc_agent.domain.media.fast_cut import (
    FAST_CUT_MARK,
    FAST_CUT_RETRY_MARK,
    fast_cut_prompt,
    fast_cut_retry,
    longest_shot,
    merge_close,
    parse_showinfo,
    timeline,
)
from aigc_agent.domain.pipeline.recipe import RECIPES_DIR, Recipe
from aigc_agent.harness.tools.provider import ToolResult

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

# ---------------------------------------------------------------- 提示词


def test_快切约束包在最外层_带时间线_幂等():
    p = fast_cut_prompt("【参考锁定】…\n陆离推门进屋", 12, [3, 3, 3, 3], 3)
    assert p.startswith(FAST_CUT_MARK)
    assert p.index("每个镜头不超过 3 秒") < p.index("参考锁定")
    assert "至少 4 个镜头" in p and "0–3s 镜头1 / 3–6s 镜头2 / 6–9s 镜头3 / 9–12s 镜头4" in p
    assert "NO shot longer than 3 seconds" in p and "hard cut" in p.lower()
    assert p.rstrip().endswith("hard cut to the next shot.")
    assert fast_cut_prompt(p, 12, [3, 3, 3, 3], 3) == p
    # 没给 cuts：按上限算至少几镜，没有时间线
    q = fast_cut_prompt("正文", 10, [], 3)
    assert "至少 4 个镜头" in q and "时间线" not in q
    assert timeline([2.5, 2.5], 5) == "0–2.5s 镜头1 / 2.5–5s 镜头2" and timeline([]) == ""


def test_重生成时在约束后面插一句上一版镜头太长_幂等():
    base = fast_cut_prompt("正文", 12, [3, 3, 3, 3], 3)
    r = fast_cut_retry(base, 5.2, 3)
    assert r.startswith(FAST_CUT_MARK)
    assert r.index(FAST_CUT_RETRY_MARK) < r.index("正文") and "长达 5.2 秒" in r
    assert fast_cut_retry(r, 5.2, 3) == r
    assert fast_cut_retry("裸提示词", 4.0, 3).startswith(FAST_CUT_RETRY_MARK)


# ---------------------------------------------------------------- 最长镜头


def test_场景切换点算最长镜头():
    stderr = (
        "[Parsed_showinfo_1 @ 0] n:   0 pts:  75000 pts_time:3.0     pos: 1\n"
        "[Parsed_showinfo_1 @ 0] n:   1 pts:  76000 pts_time:3.04    pos: 2\n"
        "[Parsed_showinfo_1 @ 0] n:   2 pts: 200000 pts_time:8.0     pos: 3\n"
    )
    assert parse_showinfo(stderr) == [3.0, 3.04, 8.0]
    assert merge_close([3.0, 3.04, 8.0]) == [3.0, 8.0], "甩镜触发的连续帧合并成一次切换"
    assert longest_shot(12.0, [3.0, 3.04, 8.0]) == 5.0  # 8→12 是最长的一段
    assert longest_shot(12.0, [3, 6, 9]) == 3.0
    assert longest_shot(12.0, []) == 12.0, "没切换 = 一镜到底"
    assert longest_shot(0, [1]) == 0.0
    assert longest_shot(10.0, [-1, 0, 10, 11, 2.5]) == 7.5, "越界的切换点不算"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="本机没有 ffmpeg")
async def test_真实ffmpeg_硬切视频量得出最长镜头(tmp_path: Path):
    """红 2s → 蓝 2s → 绿 3s 三段硬切拼成 7s：最长镜头应为 3s；单色 7s 一镜到底应为 7s。"""
    out = tmp_path / "cuts.mp4"
    code, err = await ffmpeg.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=red:s=160x90:r=25:d=2",
            "-f", "lavfi", "-i", "color=c=blue:s=160x90:r=25:d=2",
            "-f", "lavfi", "-i", "color=c=green:s=160x90:r=25:d=3",
            "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map", "[v]", "-pix_fmt", "yuv420p", str(out),
        ]
    )
    assert code == 0 and out.exists(), err
    times, why = await ffmpeg.scene_cuts(out, 0.3)
    assert not why and len(times) == 2, (times, why)
    assert abs(times[0] - 2.0) < 0.2 and abs(times[1] - 4.0) < 0.2
    info = await ffmpeg.probe(out)
    assert abs(longest_shot(info.duration, times) - 3.0) < 0.2

    one = tmp_path / "one.mp4"
    code, err = await ffmpeg.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=160x90:r=25:d=7",
         "-pix_fmt", "yuv420p", str(one)]
    )
    assert code == 0, err
    times, why = await ffmpeg.scene_cuts(one, 0.3)
    assert not why and times == []
    assert abs(longest_shot((await ffmpeg.probe(one)).duration, times) - 7.0) < 0.2


# ---------------------------------------------------------------- 镜头门（渲染）


class Registry:
    def __init__(self, store: AssetStore, tmp: Path) -> None:
        self.store = store
        self.tmp = tmp
        self.calls: list[dict] = []

    async def invoke(self, name: str, args: dict) -> ToolResult:
        if name == "compose_video":
            a = self.store.create("mp4", summary="成片", creator="fake")
            a.uri = "https://fake/composed.mp4"
            self.store.put(a)
            return ToolResult(ok=True, content="拼好了", asset_ref=a.id)
        self.calls.append(dict(args))
        a = self.store.create("", type_=AssetType.VIDEO, summary=args["summary"], creator="model:x")
        local = self.tmp / f"{a.id}.mp4"
        local.write_bytes(b"fake")
        a.uri = f"https://fake/{a.id}.mp4"
        a.gen_params["local"] = str(local)
        self.store.put(a)
        return ToolResult(ok=True, content="ok", asset_ref=a.id)


SHOTS = [
    {"scene_index": "[第1集-1场]", "video_name": "1-4", "video_duration": "12s",
     "cuts": [3, 3, 3, 3], "description": "开场 (安妮)"},
    {"scene_index": "[第1集-2场]", "video_name": "5-8", "video_duration": "12s",
     "cuts": [3, 3, 3, 3], "description": "收尾 (安妮)"},
]


def _fns(store: AssetStore, tmp: Path, longest: dict[str, list[float]], retries: str = "1"):
    """longest：summary → 每次检查量到的最长镜头序列。"""
    reg = Registry(store, tmp)
    fns = DramaFunctions(None, store, registry=reg, catalog=None, fmt=EpisodeFormat())
    fns.catalog = SimpleNamespace(
        drama={"cut_gate": "true", "cut_retries": retries, "cut_tolerance": "0.5"},
        max_concurrency=lambda k: 0,
    )
    checked: list[str] = []

    async def check(asset_id: str, threshold: float) -> tuple[float | None, str]:
        summary = store.get(asset_id).summary
        checked.append(summary)
        seq = longest.get(summary)
        if seq is None:
            return 2.5, ""
        return seq.pop(0), ""

    fns._check_cuts = check  # type: ignore[method-assign]
    return fns, reg, checked


def _shots(store: AssetStore) -> str:
    return store.create(json.dumps(SHOTS, ensure_ascii=False), summary="提示词", creator="t").id


async def test_渲染提示词带快切约束与时间线(tmp_path: Path):
    store = AssetStore()
    fns, reg, _ = _fns(store, tmp_path, {})
    r = await fns._fn_drama_render_shots(_shots(store))
    assert r.ok, r.error
    p = reg.calls[0]["prompt"]
    assert FAST_CUT_MARK in p and "0–3s 镜头1 / 3–6s 镜头2 / 6–9s 镜头3 / 9–12s 镜头4" in p
    assert p.index(FAST_CUT_MARK) < p.index("开场 (安妮)")
    assert "12s/4镜" in r.content


async def test_镜头太长_用更硬的提示词重生成_第二版达标(tmp_path: Path):
    store = AssetStore()
    fns, reg, checked = _fns(store, tmp_path, {"[第1集-1场] 1-4": [5.0, 2.8]})
    r = await fns._fn_drama_render_shots(_shots(store))
    assert r.ok, r.error
    first = [c for c in reg.calls if c["summary"] == "[第1集-1场] 1-4"]
    assert len(first) == 2 and FAST_CUT_RETRY_MARK in first[1]["prompt"]
    assert "长达 5.0 秒" in first[1]["prompt"] and FAST_CUT_RETRY_MARK not in first[0]["prompt"]
    assert "有镜头长 5.0s，超过 3s，已重生成" in r.content
    assert "重生成后镜头达标（最长 2.8s）" in r.content
    assert checked.count("[第1集-1场] 1-4") == 2 and checked.count("[第1集-2场] 5-8") == 1
    render = next(a for a in store.all() if a.creator == "tool:drama_render_shots")
    kept = json.loads(store.content(render.id))
    clips = [a for a in store.all() if a.summary == "[第1集-1场] 1-4"]
    assert kept[0]["asset"] == clips[1].id, "成片用的是第二版"


async def test_重生成后仍超时长_标出来交人复核(tmp_path: Path):
    store = AssetStore()
    fns, reg, _ = _fns(store, tmp_path, {"[第1集-2场] 5-8": [4.0, 3.9]})
    r = await fns._fn_drama_render_shots(_shots(store))
    assert r.ok, r.error
    assert sum(1 for c in reg.calls if c["summary"] == "[第1集-2场] 5-8") == 2
    assert "仍有超过 3 秒的镜头（最长 3.9s）" in r.content
    assert "用户要求每镜 ≤3 秒" in r.content
    assert "[第1集-2场] 5-8" in r.content.split("用户要求每镜")[1]


async def test_容差内算合格_只检查不重生成(tmp_path: Path):
    store = AssetStore()
    fns, reg, _ = _fns(store, tmp_path, {"[第1集-1场] 1-4": [3.4]})
    r = await fns._fn_drama_render_shots(_shots(store))
    assert r.ok and "超过" not in r.content, "3.4s ≤ 3 + 0.5 容差"

    store2 = AssetStore()
    fns2, reg2, _ = _fns(store2, tmp_path / "b", {"[第1集-1场] 1-4": [6.0]}, retries="0")
    (tmp_path / "b").mkdir()
    r2 = await fns2._fn_drama_render_shots(_shots(store2))
    assert r2.ok and sum(1 for c in reg2.calls if c["summary"] == "[第1集-1场] 1-4") == 1
    assert "仍有超过 3 秒的镜头（最长 6.0s）" in r2.content


async def test_关掉镜头门或没本地副本就不查(tmp_path: Path):
    store = AssetStore()
    fns, reg, checked = _fns(store, tmp_path, {"[第1集-1场] 1-4": [9.0]})
    fns.catalog.drama["cut_gate"] = "false"
    r = await fns._fn_drama_render_shots(_shots(store))
    assert r.ok and not checked and "超过" not in r.content

    # 真实 _check_cuts：没有本地副本 → 静默跳过，不记备注
    store2 = AssetStore()
    fns2 = DramaFunctions(None, store2, registry=Registry(store2, tmp_path), catalog=None)
    a = store2.create("", type_=AssetType.VIDEO, summary="远端", creator="model:x")
    a.uri = "https://fake/x.mp4"
    store2.put(a)
    assert await fns2._check_cuts(a.id, 0.3) == (None, "")
    assert fns2._cut_cfg() == (True, 1, 0.3, 0.5), "catalog=None 时按默认值开着"


# ---------------------------------------------------------------- 全局上限：配方短视频也不超过


def test_配方快切上限不超过全局单镜上限():
    assert global_max_cut(CONFIG_DIR) == 3.0
    r = Recipe(name="x", cut={"enabled": True, "max_seconds": 4})
    assert r.cut_max == 3.0, "配方写 4 也按 3 切"
    assert Recipe(name="y", cut={"enabled": True, "max_seconds": 2}).cut_max == 2.0
    # 关了快切也守全局上限（2026-09-23 审查：之前返回 0，合成时整段拼、一镜 8 秒）
    assert Recipe(name="z", cut={"enabled": False, "max_seconds": 4}).cut_max == 3.0
    assert Recipe(name="w", cut={"enabled": True}).cut_max == 3.0, "没写上限就用全局的"
    for p in sorted(RECIPES_DIR.glob("*.yaml")):
        data: dict[str, Any] = __import__("yaml").safe_load(p.read_text(encoding="utf-8")) or {}
        cut = data.get("cut") or {}
        if cut.get("enabled"):
            r = Recipe(name=p.stem, cut=cut)
            assert r.cut_max <= 3.0, p.name

# ruff: noqa: ASYNC109, ASYNC240
# 本模块的文件操作都紧邻 ffmpeg 子进程调用（写入/读取/stat），
# 单次、小量、必然与子进程串行，丢线程池反而增加复杂度而无收益。
# timeout 参数是这层的显式契约，不走 cancel scope。
"""M14 媒体处理 —— 确定性操作，不过模型。

这层全是确定性的：下载、拼接、混音、烧字幕、抽封面。
**不让模型直接写 ffmpeg 命令行**，而是封成参数化操作 ——
模型负责决策（"这几段按这个顺序拼"），这里负责执行。

为什么必须有这层：视频模型单次最长 8–12 秒，30 秒成片必须分段生成后拼接。
没有它，前面所有生成能力都只能产出零件，交付不了成品。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import httpx2 as httpx

from ...harness.model.media import default_proxy

DOWNLOAD_TIMEOUT = 300.0
FFMPEG_TIMEOUT = 900.0
_PTS_TIME = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")  # showinfo 每帧一行


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@dataclass
class Probe:
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = False

    @property
    def brief(self) -> str:
        return f"{self.duration:.1f}s · {self.width}x{self.height} · {self.fps:.0f}fps" + (
            " · 有音轨" if self.has_audio else " · 无音轨"
        )


async def run(cmd: list[str], timeout: float = FFMPEG_TIMEOUT) -> tuple[int, str]:
    """跑外部命令。超时/崩溃归一成 (code, stderr)，不抛出去打断 loop。"""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    try:
        _, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, f"超时（>{timeout:.0f}s）"
    return proc.returncode or 0, err.decode("utf-8", errors="replace")


async def download(url: str, target: Path) -> tuple[bool, str]:
    """把远端媒体抓到本地。

    生成接口返回的是**临时外链**，能存多久没保证。要归档、要喂给 ffmpeg，
    都得先落盘。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        async with httpx.AsyncClient(
            timeout=DOWNLOAD_TIMEOUT, proxy=default_proxy(), follow_redirects=True
        ) as c:
            resp = await c.get(url)
            if resp.status_code >= 400:
                return False, f"HTTP {resp.status_code}"
            target.write_bytes(resp.content)
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    if target.stat().st_size == 0:
        return False, "下载到 0 字节"
    return True, ""


async def probe(path: Path) -> Probe:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    try:
        data = json.loads(out.decode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return Probe()

    p = Probe()
    p.duration = float((data.get("format") or {}).get("duration") or 0)
    for st in data.get("streams") or []:
        if st.get("codec_type") == "video" and not p.width:
            p.width = int(st.get("width") or 0)
            p.height = int(st.get("height") or 0)
            rate = str(st.get("r_frame_rate") or "0/1")
            num, _, den = rate.partition("/")
            p.fps = float(num) / float(den or 1) if float(den or 1) else 0.0
        elif st.get("codec_type") == "audio":
            p.has_audio = True
    return p


async def concat(
    clips: list[Path], out: Path, target_size: tuple[int, int] | None = None
) -> tuple[bool, str]:
    """按顺序拼接多段视频。

    **统一重编码而不是 concat demuxer 直拷**：分段是不同次生成的，
    分辨率/帧率/编码参数不保证一致，直拷会花屏或只出第一段。
    慢一点，但可靠。
    """
    if not clips:
        return False, "没有片段"
    out.parent.mkdir(parents=True, exist_ok=True)

    w, h = target_size or (0, 0)
    if not w:
        first = await probe(clips[0])
        w, h = first.width or 1080, first.height or 1920

    cmd: list[str] = ["ffmpeg", "-y"]
    for c in clips:
        cmd += ["-i", str(c)]

    # 每段先缩放补边到统一尺寸，再拼接。补边而不是裁剪，避免切掉主体。
    # 有没有音轨要逐段查。**混着来是常态**：配过音的段有，没台词的段没有。
    # 之前这里写死 a=0 只拼视频，结果每段辛苦混好的配音在拼接时被整个丢掉，
    # 成片"无音轨" —— 前面全部成功，最后一步静默清零，很难归因。
    has_audio = [bool((await probe(c)).has_audio) for c in clips]
    keep_audio = any(has_audio)

    parts = []
    for i in range(len(clips)):
        parts.append(
            f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[v{i}]"
        )

    streams = ""
    if keep_audio:
        # 没音轨的段补等长静音，否则 concat 的音视频段数对不上会直接报错
        silent = len(clips)
        for i, ok_a in enumerate(has_audio):
            if ok_a:
                parts.append(f"[{i}:a]aresample=44100,asetpts=PTS-STARTPTS[a{i}]")
            else:
                d = (await probe(clips[i])).duration or 1.0
                cmd += ["-f", "lavfi", "-t", f"{d:.3f}", "-i", "anullsrc=r=44100:cl=stereo"]
                parts.append(f"[{silent}:a]aresample=44100,asetpts=PTS-STARTPTS[a{i}]")
                silent += 1
        streams = "".join(f"[v{i}][a{i}]" for i in range(len(clips)))
        filt = ";".join(parts) + f";{streams}concat=n={len(clips)}:v=1:a=1[outv][outa]"
    else:
        streams = "".join(f"[v{i}]" for i in range(len(clips)))
        filt = ";".join(parts) + f";{streams}concat=n={len(clips)}:v=1:a=0[outv]"

    cmd += ["-filter_complex", filt, "-map", "[outv]"]
    if keep_audio:
        cmd += ["-map", "[outa]", "-c:a", "aac", "-b:a", "192k"]
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(out),
    ]
    code, err = await run(cmd)
    if code != 0 or not out.exists():
        return False, _tail(err)
    return True, ""


async def concat_cuts(
    sources: list[Path],
    cuts: list[tuple[int, float, float]],
    out: Path,
    target_size: tuple[int, int] | None = None,
) -> tuple[bool, str]:
    """按剪辑表拼接：同一段素材可以被切成多刀、在成片里出现多次。

    cuts 是 (素材下标, 起点秒, 时长秒)。

    **每一刀单独 -i 一次同一个文件**，而不是用 split 滤镜分流：滤镜图里一个
    输入 pad 只能连一次，要复用就得 split，写起来啰嗦且漏一条就整条图报错。
    重复 -i 让 ffmpeg 自己去处理，素材只有几秒，多解码几次可以忽略。
    """
    if not cuts:
        return False, "剪辑表为空"
    if any(c[0] >= len(sources) for c in cuts):
        return False, "剪辑表引用了不存在的素材"
    out.parent.mkdir(parents=True, exist_ok=True)

    w, h = target_size or (0, 0)
    if not w:
        first = await probe(sources[0])
        w, h = first.width or 1080, first.height or 1920

    cmd: list[str] = ["ffmpeg", "-y"]
    for idx, _, _ in cuts:
        cmd += ["-i", str(sources[idx])]

    parts = []
    for i, (_, start, dur) in enumerate(cuts):
        parts.append(
            f"[{i}:v]trim=start={start:.3f}:duration={dur:.3f},setpts=PTS-STARTPTS,"
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[v{i}]"
        )
    streams = "".join(f"[v{i}]" for i in range(len(cuts)))
    filt = ";".join(parts) + f";{streams}concat=n={len(cuts)}:v=1:a=0[outv]"

    cmd += [
        "-filter_complex",
        filt,
        "-map",
        "[outv]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(out),
    ]
    code, err = await run(cmd)
    if code != 0 or not out.exists():
        return False, _tail(err)
    return True, ""


async def mux_audio(video: Path, audio: Path, out: Path, fit: str = "pad") -> tuple[bool, str]:
    """给视频配上音轨。

    fit:
      pad  音频短于视频时补静音，长于则截断（保视频完整，默认）
      trim 以音频长度为准裁剪视频
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
    ]
    cmd += ["-shortest"] if fit == "trim" else ["-af", "apad"]
    if fit != "trim":
        cmd += ["-t", str((await probe(video)).duration or 30)]
    cmd += [str(out)]
    code, err = await run(cmd)
    if code != 0 or not out.exists():
        return False, _tail(err)
    return True, ""


async def burn_subtitle(video: Path, srt: Path, out: Path, font_size: int = 16) -> tuple[bool, str]:
    """烧录字幕。

    Windows 下 ffmpeg 的 subtitles 滤镜对路径极挑剔（盘符冒号、反斜杠、
    中文目录都会让它解析失败），所以把字幕复制到临时目录用相对路径调用。
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    work = srt.parent
    tmp_srt = work / "_sub.srt"
    shutil.copyfile(srt, tmp_srt)

    style = (
        f"FontSize={font_size},PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H90000000,BorderStyle=3,Outline=1,MarginV=40"
    )
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-i",
        str(video.resolve()),
        "-vf",
        f"subtitles=_sub.srt:force_style='{style}'",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-c:a",
        "copy",
        str(out.resolve()),
        cwd=str(work),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, err = await asyncio.wait_for(proc.communicate(), timeout=FFMPEG_TIMEOUT)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return False, "烧字幕超时"
    finally:
        tmp_srt.unlink(missing_ok=True)

    if proc.returncode != 0 or not out.exists():
        return False, _tail(err.decode("utf-8", errors="replace"))
    return True, ""


async def extract_frames(video: Path, out_dir: Path, count: int = 12) -> list[Path]:
    """按时间均匀抽 count 张关键帧，用于给视觉模型看参考视频。

    **均匀抽而不是按场景抽**：这里要的是"这段片子的节奏和构图长什么样"，
    场景检测会把长镜头整段跳过、把快切段落抽出一大把，时间分布就歪了。

    抽出来的是 jpg 且缩到宽 384、质量压到 7 —— 模型侧用的是 detail=low，
    本来就会降采样，实测 512/q4 一帧要 145KB base64，十几帧就是 2MB 的
    请求体，白白拖长首字节时间。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    info = await probe(video)
    dur = info.duration or 0.0
    if dur <= 0:
        return []

    paths: list[Path] = []
    for i in range(max(1, count)):
        # 避开首尾：第一帧常是黑场，最后一帧常是淡出
        t = dur * (i + 0.5) / max(1, count)
        f = out_dir / f"frame{i:02d}.jpg"
        code, _ = await run(
            ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(video),
             "-frames:v", "1", "-vf", "scale=384:-2", "-q:v", "7", str(f)]
        )
        if code == 0 and f.exists():
            paths.append(f)
    return paths


async def scene_cuts(video: Path, threshold: float = 0.3) -> tuple[list[float], str]:
    """场景切换时刻（秒）：相邻两帧差异超过 threshold 的位置。

    给「每个镜头 ≤3 秒」的镜头门用：切换点把片子分成若干镜头，量最长的那个。
    阈值 0.3 左右：硬切通常 0.5 以上，同一镜头里的运镜一般不到 0.2。
    返回 (切换时刻列表, 失败原因)。
    """
    code, err = await run(
        [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(video),
            "-vf", f"select='gt(scene,{threshold:g})',showinfo",
            "-an", "-f", "null", "-",
        ]
    )
    if code != 0:
        return [], _tail(err)
    times = sorted({float(t) for t in _PTS_TIME.findall(err)})
    return times, ""


async def silence(seconds: float, out: Path) -> bool:
    """生成一段静音。用来在台词之间留呼吸，以及把音轨补到片段长度。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    code, _ = await run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", f"{max(0.01, seconds):.3f}", "-c:a", "libmp3lame", "-q:a", "4", str(out),
    ])
    return code == 0 and out.exists()


async def concat_audio(parts: list[Path], out: Path) -> tuple[bool, str]:
    """顺序拼接音频。统一重编码，理由同 concat()：各段来源不同，参数不保证一致。"""
    if not parts:
        return False, "没有音频片段"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = ["ffmpeg", "-y"]
    for p in parts:
        cmd += ["-i", str(p)]
    streams = "".join(f"[{i}:a]" for i in range(len(parts)))
    cmd += [
        "-filter_complex", f"{streams}concat=n={len(parts)}:v=0:a=1[outa]",
        "-map", "[outa]", "-c:a", "libmp3lame", "-q:a", "4", str(out),
    ]
    code, err = await run(cmd)
    if code != 0 or not out.exists():
        return False, _tail(err)
    return True, ""


async def fit_audio(
    audio: Path, seconds: float, out: Path, lead: float = 0.3
) -> tuple[bool, str]:
    """把音轨调整到正好 seconds 长。

    短了补静音；长了**加速**而不是截断 —— 截断会把台词切在半句上，
    那是观众一眼能听出来的破绽。加速上限 1.25 倍，再快就失真了，
    超过就只能如实截断并让调用方知道。

    lead 是**前置留白占空余时长的比例**。把静音全堆在尾部的话，
    一段 14 秒的镜头里台词 2 秒说完、后面 12 秒死寂，听起来像
    "一开口就说完然后发呆"。留一点前摇，台词落在动作上而不是抢拍。
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    info = await probe(audio)
    cur = info.duration or 0.0
    if cur <= 0:
        return False, "读不到音频时长"

    if cur <= seconds + 0.05:
        slack = max(0.0, seconds - cur)
        before = round(slack * max(0.0, min(1.0, lead)), 3)
        code, err = await run([
            "ffmpeg", "-y", "-i", str(audio),
            "-af", f"adelay={int(before * 1000)}:all=1,apad=whole_dur={seconds:.3f}",
            "-t", f"{seconds:.3f}", "-c:a", "libmp3lame", "-q:a", "4", str(out),
        ])
        ok = code == 0 and out.exists()
        return ok, ("" if ok else _tail(err))

    ratio = cur / seconds
    note = ""
    if ratio > 1.25:
        ratio = 1.25
        note = f"台词比画面长 {cur - seconds:.1f}s，加速到上限后仍会截断"
    code, err = await run([
        "ffmpeg", "-y", "-i", str(audio),
        "-af", f"atempo={ratio:.4f}", "-t", f"{seconds:.3f}",
        "-c:a", "libmp3lame", "-q:a", "4", str(out),
    ])
    if code != 0 or not out.exists():
        return False, _tail(err)
    return True, note


async def still_to_clip(
    image: Path, out: Path, seconds: float, size: tuple[int, int] = (720, 1280)
) -> tuple[bool, str]:
    """一张图 → 一段静止镜头（居中裁满画幅）。用户给的照片当素材时走这里。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},format=yuv420p"
    code, err = await run(
        [
            "ffmpeg", "-y", "-loop", "1", "-i", str(image), "-t", f"{max(0.5, seconds):.2f}",
            "-vf", vf, "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-an", str(out),
        ]
    )
    if code != 0 or not out.exists():
        return False, _tail(err)
    return True, ""


async def grab_cover(video: Path, out: Path, at: float = 0.5) -> tuple[bool, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    code, err = await run(
        ["ffmpeg", "-y", "-ss", str(at), "-i", str(video), "-vframes", "1", "-q:v", "2", str(out)]
    )
    if code != 0 or not out.exists():
        return False, _tail(err)
    return True, ""


def _tail(err: str, n: int = 400) -> str:
    lines = [ln for ln in err.splitlines() if ln.strip()]
    return "\n".join(lines[-6:])[:n] if lines else "（无错误输出）"

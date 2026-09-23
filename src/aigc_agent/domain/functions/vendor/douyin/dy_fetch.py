#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dy_fetch.py — 抖音爆款拆解：素材获取 + 双轨抽帧 + 音频分析（Windows / macOS / Linux 通用）

用法：
  python dy_fetch.py "<抖音分享口令 / 短链 / 视频页链接 / aweme_id>" [--out DIR] [--transcribe] [--cookies cookies.txt]
  python dy_fetch.py <本地视频.mp4> [--out DIR] [--transcribe]

数据来源（按顺序尝试，拿到一个就停）：
  1. 本地视频文件                → 直接进入抽帧
  2. TikHub API（环境变量 TIKHUB_API_KEY）→ 公开数据 + Top 评论 + 无水印下载
  3. yt-dlp + 你自己导出的 cookies.txt（--cookies 或环境变量 DY_COOKIES）→ 下载 + 基础数据
  4. 都没有 → 只解析出 video_id，并打印人工补充清单

产出（--out 目录，默认 ./dy_analysis/<video_id>/）：
  meta.json       视频元信息 + 公开数据 + 评论（拿到多少写多少，缺的标 null）
  video.mp4       视频文件
  frames/         first.jpg / last.jpg / uniform_*.jpg / scene_*.jpg + index.json（每帧时间戳）
  audio.json      音量 / 静音段 / 口播占比估计
  audio16k.wav    供转写使用
  transcript.*    （--transcribe 时，调用 dy_transcribe.py）
  summary.md      给分析用的人读摘要 —— 先读这个文件
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
MOBILE_UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
             "(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1")
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")

# 抖音是国内服务，绕过系统代理直连
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def log(msg):
    print(f"[dy_fetch] {msg}", flush=True)


def http_get(url, headers=None, timeout=30):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": MOBILE_UA})
    return _opener.open(req, timeout=timeout)


# ---------------------------------------------------------------- 1. 解析输入
ID_PATTERNS = [
    r"douyin\.com/(?:video|note)/(\d{15,20})",
    r"iesdouyin\.com/share/(?:video|note|slides)/(\d{15,20})",
    r"modal_id=(\d{15,20})",
]


def extract_ref(text):
    """返回 ('file', path) / ('url', url) / ('id', aweme_id) / (None, None)"""
    t = text.strip().strip('"').strip("'")
    if os.path.isfile(t):
        return "file", t
    for pat in ID_PATTERNS:
        m = re.search(pat, t)
        if m:
            return "id", m.group(1)
    m = re.search(r"https?://v\.douyin\.com/[A-Za-z0-9_\-]+/?", t)
    if m:
        return "url", m.group(0)
    m = re.search(r"https?://[^\s\"']+", t)
    if m:
        return "url", m.group(0)
    if re.fullmatch(r"\d{15,20}", t):
        return "id", t
    return None, None


def resolve_short_link(url):
    """跟随短链跳转，从最终 URL 里取 aweme_id（纯 HTTP 重定向，不做任何签名）"""
    try:
        resp = http_get(url, timeout=20)
        final = resp.geturl()
    except urllib.error.HTTPError as e:
        final = e.geturl() or url
    log(f"短链最终落点: {final[:120]}")
    if "webcast" in final or "/live" in final:
        raise SystemExit("这是直播间链接，不是视频，无法拆解。")
    for pat in ID_PATTERNS:
        m = re.search(pat, final)
        if m:
            return m.group(1)
    raise SystemExit(f"没能从跳转结果里识别出视频 ID：{final}")


# ---------------------------------------------------------------- 2. TikHub
def tikhub_fetch(aweme_id, key):
    base = "https://api.tikhub.io/api/v1/douyin"
    hdr = {"Authorization": f"Bearer {key}", "User-Agent": "dy_fetch/1.0", "Accept": "application/json"}

    def call(path):
        req = urllib.request.Request(base + path, headers=hdr)
        try:
            with urllib.request.urlopen(req, timeout=40) as r:   # TikHub 是海外服务，走系统代理
                return json.loads(r.read().decode("utf-8", "ignore"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")[:300]
            return {"code": e.code, "error": body}
        except Exception as e:
            return {"code": -1, "error": str(e)}

    log("TikHub: 拉取视频详情 …")
    # [本地补丁 2026-09-11] 上游写的是 fetch_one_video_v3，实测返回 400；
    # 正确端点名是 fetch_one_video（即使路径在 /app/v3/ 下）。
    # 从上游同步时注意保留这一行。
    d = call(f"/app/v3/fetch_one_video?aweme_id={aweme_id}")
    if not d:
        d = call(f"/web/fetch_one_video?aweme_id={aweme_id}")
    if d.get("code") != 200:
        log(f"TikHub 视频详情失败: code={d.get('code')} {str(d.get('error') or d.get('detail') or '')[:200]}")
        return None
    data = d.get("data") or {}
    det = data.get("aweme_detail") or (data.get("aweme_details") or [None])[0] or (data if data.get("aweme_id") else None)
    if not det:
        log("TikHub 返回里没有 aweme_detail")
        return None

    st = det.get("statistics") or {}
    video = det.get("video") or {}
    play_urls = []
    candidates = [video.get("play_addr")]
    if video.get("bit_rate"):
        candidates.append((video["bit_rate"][0] or {}).get("play_addr"))
    candidates += [video.get("play_addr_h264"), video.get("download_addr")]
    for src in candidates:
        if src and src.get("url_list"):
            play_urls.extend(src["url_list"])
    author = det.get("author") or {}
    meta = {
        "source": "tikhub",
        "video_id": det.get("aweme_id") or aweme_id,
        "desc": det.get("desc"),
        "create_time": det.get("create_time"),
        "author": {"nickname": author.get("nickname"), "uid": author.get("uid"), "sec_uid": author.get("sec_uid"),
                   "follower_count": author.get("follower_count"), "signature": author.get("signature")},
        "stats": {"digg_count": st.get("digg_count"), "comment_count": st.get("comment_count"),
                  "collect_count": st.get("collect_count"), "share_count": st.get("share_count"),
                  "play_count": st.get("play_count")},
        "duration_ms": video.get("duration") or det.get("duration"),
        "hashtags": [x.get("hashtag_name") for x in (det.get("text_extra") or []) if x.get("hashtag_name")],
        "music": {"title": (det.get("music") or {}).get("title"), "author": (det.get("music") or {}).get("author")},
        "aweme_type": det.get("aweme_type"),
        "images": [(im.get("url_list") or [None])[0] for im in (det.get("images") or [])],
        "play_urls": play_urls,
        "comments": [],
    }

    log("TikHub: 拉取评论（APP 接口，失败自动切 WEB）…")
    c = call(f"/app/v3/fetch_video_comments?aweme_id={aweme_id}&cursor=0&count=20")
    if c.get("code") != 200 or not (c.get("data") or {}).get("comments"):
        c = call(f"/web/fetch_video_comments?aweme_id={aweme_id}&cursor=0&count=20")
    comments = (c.get("data") or {}).get("comments") or []
    for cm in comments:
        meta["comments"].append({
            "text": cm.get("text"),
            "digg_count": cm.get("digg_count"),
            "reply_count": cm.get("reply_comment_total"),
            "user": (cm.get("user") or {}).get("nickname"),
            "create_time": cm.get("create_time"),
        })
    meta["comments"].sort(key=lambda x: -(x.get("digg_count") or 0))
    log(f"TikHub: 拿到 {len(meta['comments'])} 条评论")
    return meta


# ---------------------------------------------------------------- 3. yt-dlp（用户自己的 cookies）
def ytdlp_fetch(aweme_id, cookies, out_dir):
    url = f"https://www.douyin.com/video/{aweme_id}"
    base = [sys.executable, "-m", "yt_dlp", "--no-warnings", "--no-playlist", "--cookies", cookies]
    log("yt-dlp: 读取视频信息 …")
    p = subprocess.run(base + ["-j", url], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if not lines:
        log(f"yt-dlp 失败: {p.stderr.strip()[:300]}")
        return None
    info = json.loads(lines[0])
    meta = {
        "source": "yt-dlp",
        "video_id": info.get("id") or aweme_id,
        "desc": info.get("title") or info.get("description"),
        "create_time": info.get("timestamp"),
        "author": {"nickname": info.get("uploader"), "uid": info.get("uploader_id"), "sec_uid": None,
                   "follower_count": None, "signature": None},
        "stats": {"digg_count": info.get("like_count"), "comment_count": info.get("comment_count"),
                  "collect_count": None, "share_count": info.get("repost_count"), "play_count": info.get("view_count")},
        "duration_ms": int(info["duration"] * 1000) if info.get("duration") else None,
        "hashtags": info.get("tags") or [],
        "music": {"title": info.get("track"), "author": info.get("artist")},
        "aweme_type": None, "images": [], "play_urls": [], "comments": [],
    }
    log("yt-dlp: 下载视频 …")
    target = out_dir / "video.mp4"
    p = subprocess.run(base + ["-f", "bv*+ba/b", "--merge-output-format", "mp4", "-o", str(target), url],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    if not target.exists():
        log(f"yt-dlp 下载失败: {p.stderr.strip()[:300]}")
    return meta


# ---------------------------------------------------------------- 4. 下载
def download(urls, target):
    for u in urls:
        try:
            log(f"下载: {u[:100]} …")
            req = urllib.request.Request(u, headers={"User-Agent": MOBILE_UA, "Referer": "https://www.douyin.com/"})
            with urllib.request.urlopen(req, timeout=60) as r, open(target, "wb") as f:
                shutil.copyfileobj(r, f)
            if target.stat().st_size > 50 * 1024:
                log(f"下载完成 {target.stat().st_size / 1024 / 1024:.1f} MB")
                return True
            log("文件太小，换下一个地址")
        except Exception as e:
            log(f"下载失败: {e}")
    return False


# ---------------------------------------------------------------- 5. ffprobe / 抽帧 / 音频
def run(cmd, timeout=600):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def probe(video):
    p = run([FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(video)])
    if p.returncode != 0:
        raise SystemExit(f"ffprobe 失败: {p.stderr[:300]}")
    j = json.loads(p.stdout)
    v = next((s for s in j["streams"] if s.get("codec_type") == "video"), {})
    a = next((s for s in j["streams"] if s.get("codec_type") == "audio"), None)
    fr = v.get("r_frame_rate", "0/1")
    try:
        num, den = fr.split("/")
        fps = round(float(num) / float(den), 2)
    except Exception:
        fps = None
    return {
        "duration": float(j["format"].get("duration", 0) or 0),
        "width": v.get("width"), "height": v.get("height"), "fps": fps,
        "video_codec": v.get("codec_name"), "bit_rate": int(j["format"].get("bit_rate", 0) or 0),
        "has_audio": a is not None, "audio_codec": a.get("codec_name") if a else None,
    }


def pick_fps(duration):
    """SKILL.md 抽帧密度铁律"""
    if duration <= 10:
        return 4
    if duration <= 30:
        return 2
    if duration <= 120:
        return 1
    return 0.5


def extract_frames(video, frames_dir, duration):
    frames_dir.mkdir(parents=True, exist_ok=True)
    fps = pick_fps(duration)
    scale = "scale='min(720,iw)':-2"
    index = {"fps": fps, "scene_threshold": 0.15, "first": None, "last": None, "uniform": [], "scene": []}

    log("抽帧: 第 0 帧 / 最后一帧 …")
    run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video), "-frames:v", "1", "-vf", scale,
         "-q:v", "2", str(frames_dir / "first.jpg")])
    run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-sseof", "-0.3", "-i", str(video), "-update", "1",
         "-frames:v", "1", "-vf", scale, "-q:v", "2", str(frames_dir / "last.jpg")])
    if not (frames_dir / "last.jpg").exists():
        run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{max(duration - 0.2, 0):.2f}", "-i", str(video),
             "-update", "1", "-frames:v", "1", "-vf", scale, "-q:v", "2", str(frames_dir / "last.jpg")])
    index["first"] = "first.jpg" if (frames_dir / "first.jpg").exists() else None
    index["last"] = "last.jpg" if (frames_dir / "last.jpg").exists() else None

    log(f"抽帧: 轨道 A 均匀采样 {fps} fps …")
    run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video), "-vf", f"fps={fps},{scale}", "-q:v", "3",
         str(frames_dir / "uniform_%04d.jpg")])
    for i, f in enumerate(sorted(frames_dir.glob("uniform_*.jpg"))):
        index["uniform"].append({"file": f.name, "t": round(i / fps, 3)})

    log("抽帧: 轨道 B 场景切换检测（阈值 0.15）…")
    p = run([FFMPEG, "-y", "-hide_banner", "-loglevel", "info", "-i", str(video),
             "-vf", f"select='gt(scene,0.15)',showinfo,{scale}", "-fps_mode", "vfr", "-q:v", "3",
             str(frames_dir / "scene_%04d.jpg")])
    times = [float(x) for x in re.findall(r"pts_time:\s*([0-9.]+)", p.stderr)]
    for i, f in enumerate(sorted(frames_dir.glob("scene_*.jpg"))):
        index["scene"].append({"file": f.name, "t": round(times[i], 3) if i < len(times) else None})
    (frames_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"抽帧完成: 均匀 {len(index['uniform'])} 帧, 场景切换 {len(index['scene'])} 帧")
    return index


def analyze_audio(video, out_dir, duration, has_audio):
    res = {"has_audio": has_audio, "mean_volume_db": None, "max_volume_db": None, "silences": [],
           "silence_total_s": 0.0, "speech_ratio_est": None, "wav16k": None}
    if not has_audio:
        log("视频没有音轨，跳过音频分析")
        return res
    log("音频: 音量 / 静音段 …")
    p = run([FFMPEG, "-hide_banner", "-i", str(video), "-af", "volumedetect", "-f", "null", "-"])
    m = re.search(r"mean_volume:\s*(-?[0-9.]+)", p.stderr)
    res["mean_volume_db"] = float(m.group(1)) if m else None
    m = re.search(r"max_volume:\s*(-?[0-9.]+)", p.stderr)
    res["max_volume_db"] = float(m.group(1)) if m else None
    p = run([FFMPEG, "-hide_banner", "-i", str(video), "-af", "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-"])
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[0-9.]+)", p.stderr)]
    ends = re.findall(r"silence_end:\s*(-?[0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)", p.stderr)
    for i, s in enumerate(starts):
        if i < len(ends):
            e, d = float(ends[i][0]), float(ends[i][1])
        else:
            e, d = duration, max(duration - s, 0)
        res["silences"].append({"start": round(s, 2), "end": round(e, 2), "dur": round(d, 2)})
    res["silence_total_s"] = round(sum(x["dur"] for x in res["silences"]), 2)
    if duration > 0:
        res["speech_ratio_est"] = round(max(0.0, 1 - res["silence_total_s"] / duration), 2)
    wav = out_dir / "audio16k.wav"
    run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(wav)])
    res["wav16k"] = wav.name if wav.exists() else None
    (out_dir / "audio.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


# ---------------------------------------------------------------- 6. 摘要
def ratio(a, b):
    if not a or not b:
        return "—"
    return f"1:{b / a:.0f}" if b >= a else f"{a / b:.0f}:1"


def fmt_num(n):
    if n is None:
        return "未知"
    n = int(n)
    return f"{n / 10000:.1f}w" if n >= 10000 else str(n)


def write_summary(out_dir, meta, tech, frames, audio, transcript_txt):
    st = meta.get("stats") or {}
    digg, cmt, col, sh = st.get("digg_count"), st.get("comment_count"), st.get("collect_count"), st.get("share_count")
    a = meta.get("author") or {}
    lines = []
    lines.append(f"# 素材包 · {meta.get('video_id') or '(本地文件)'}\n")
    lines.append(f"- 来源: {meta.get('source')}")
    lines.append(f"- 标题/文案: {meta.get('desc') or '未知（请人工补充）'}")
    lines.append(f"- 作者: {a.get('nickname') or '未知'}  粉丝: {fmt_num(a.get('follower_count'))}")
    if meta.get("hashtags"):
        lines.append(f"- 话题: {' '.join('#' + h for h in meta['hashtags'])}")
    if (meta.get("music") or {}).get("title"):
        lines.append(f"- BGM: {meta['music']['title']} — {meta['music'].get('author') or ''}")
    if meta.get("create_time"):
        lines.append(f"- 发布时间: {time.strftime('%Y-%m-%d %H:%M', time.localtime(int(meta['create_time'])))}")
    if meta.get("page_url"):
        lines.append(f"- 视频页: {meta['page_url']}")
    lines.append("")
    lines.append("## 公开数据")
    lines.append("| 点赞 | 评论 | 收藏 | 转发 | 播放 |")
    lines.append("|---|---|---|---|---|")
    play = fmt_num(st.get("play_count")) if st.get("play_count") else "不公开"
    lines.append(f"| {fmt_num(digg)} | {fmt_num(cmt)} | {fmt_num(col)} | {fmt_num(sh)} | {play} |")
    lines.append("")
    lines.append(f"- 评赞比 {ratio(cmt, digg)} · 藏赞比 {ratio(col, digg)} · 转赞比 {ratio(sh, digg)}")
    if digg and cmt:
        lines.append(f"- 互动结构: 评论/点赞 = {cmt / digg:.3f}, 收藏/点赞 = {(col or 0) / digg:.3f}, 转发/点赞 = {(sh or 0) / digg:.3f}")
    lines.append("")
    lines.append("## 视频技术参数")
    if tech:
        lines.append(f"- 时长 {tech['duration']:.1f}s · {tech['width']}x{tech['height']} · {tech['fps']} fps · "
                     f"{tech['video_codec']} · 音轨: {'有' if tech['has_audio'] else '无'}")
    else:
        lines.append("- 未拿到视频文件，无法抽帧。请按下方清单补充。")
    lines.append("")
    if frames:
        lines.append("## 抽帧（先读 first.jpg 和 last.jpg，再按时间顺序读 uniform，scene 用来数节奏）")
        lines.append(f"- 目录: frames/  · 均匀采样 {frames['fps']} fps 共 {len(frames['uniform'])} 帧 · "
                     f"场景切换 {len(frames['scene'])} 次")
        if frames["scene"]:
            ts = [str(s["t"]) for s in frames["scene"] if s["t"] is not None]
            lines.append(f"- 场景切换时间点(s): {', '.join(ts[:60])}{' …' if len(ts) > 60 else ''}")
            if tech and tech["duration"] > 0:
                lines.append(f"- 平均每 {tech['duration'] / max(len(frames['scene']), 1):.1f}s 切一次镜头")
        lines.append("")
    if audio and audio.get("has_audio"):
        lines.append("## 音频")
        lines.append(f"- 平均音量 {audio['mean_volume_db']} dB · 峰值 {audio['max_volume_db']} dB")
        lines.append(f"- 静音段 {len(audio['silences'])} 段，共 {audio['silence_total_s']}s · "
                     f"有声占比估计 {audio['speech_ratio_est']}")
        if audio["silences"]:
            lines.append("- 静音区间: " + "; ".join(f"{s['start']}–{s['end']}" for s in audio["silences"][:20]))
        lines.append("")
    if transcript_txt and transcript_txt.exists():
        lines.append("## 口播逐字稿（whisper，同音字需与画面字幕交叉验证）")
        lines.append("```")
        lines.append(transcript_txt.read_text(encoding="utf-8").strip()[:4000])
        lines.append("```")
        lines.append("")
    if meta.get("comments"):
        lines.append(f"## 高赞评论 Top {min(len(meta['comments']), 20)}")
        for c in meta["comments"][:20]:
            lines.append(f"- [{fmt_num(c.get('digg_count'))}赞 / {c.get('reply_count') or 0}回复] {c.get('text')}")
        lines.append("")
    missing = []
    if not meta.get("desc"):
        missing.append("标题/文案")
    if digg is None:
        missing.append("点赞、评论、收藏、转发四个数字")
    if not meta.get("comments"):
        missing.append("高赞评论 Top 5–10（归因校验的关键，务必补）")
    if not (meta.get("author") or {}).get("follower_count"):
        missing.append("账号粉丝量、人设标签")
    if not tech:
        missing.append("视频文件（抖音 App 内「保存到本地」后把 mp4 路径给我）或关键帧截图")
    if missing:
        lines.append("## ⚠️ 仍需人工补充")
        for m in missing:
            lines.append(f"- {m}")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", help="分享口令 / 链接 / aweme_id / 本地视频路径")
    ap.add_argument("--out", help="输出目录（默认 ./dy_analysis/<video_id>）")
    ap.add_argument("--cookies", default=os.environ.get("DY_COOKIES"), help="Netscape 格式 cookies.txt（yt-dlp 用）")
    ap.add_argument("--transcribe", action="store_true", help="抽帧后调用 dy_transcribe.py 做口播转写")
    ap.add_argument("--model", default="large-v3", help="whisper 模型（large-v3 / medium / small）")
    ap.add_argument("--no-frames", action="store_true")
    args = ap.parse_args()

    kind, val = extract_ref(args.ref)
    if kind is None:
        raise SystemExit("没识别出链接、视频 ID 或本地文件。请直接粘贴抖音分享口令（含 v.douyin.com 链接）。")

    meta = None
    video = None
    if kind == "file":
        video = Path(val)
        vid = re.sub(r"[^\w\-]+", "_", video.stem)[:40]
        out_dir = Path(args.out or f"dy_analysis/{vid}")
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {"source": "local-file", "video_id": None, "desc": None, "author": {}, "stats": {}, "comments": [],
                "hashtags": [], "music": {}, "local_file": str(video.resolve())}
    else:
        aweme_id = resolve_short_link(val) if kind == "url" else val
        log(f"视频 ID: {aweme_id}")
        out_dir = Path(args.out or f"dy_analysis/{aweme_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        key = os.environ.get("TIKHUB_API_KEY")
        if key:
            meta = tikhub_fetch(aweme_id, key)
        else:
            log("未配置 TIKHUB_API_KEY，跳过 TikHub")
        if meta and meta.get("play_urls"):
            target = out_dir / "video.mp4"
            if download(meta["play_urls"], target):
                video = target
        if meta is None and args.cookies and os.path.isfile(args.cookies):
            meta = ytdlp_fetch(aweme_id, args.cookies, out_dir)
            if (out_dir / "video.mp4").exists():
                video = out_dir / "video.mp4"
        elif meta is None:
            log("未提供 cookies.txt，跳过 yt-dlp")
        if meta is None:
            meta = {"source": "id-only", "video_id": aweme_id, "desc": None, "author": {}, "stats": {}, "comments": [],
                    "hashtags": [], "music": {},
                    "page_url": f"https://www.douyin.com/video/{aweme_id}"}
            log("⚠️ 只解析出了视频 ID，没有数据源可用。要么配置 TIKHUB_API_KEY，要么给 cookies.txt，"
                "要么把视频保存到本地再传路径。")

    tech = frames = audio = None
    transcript_txt = None
    if video and video.exists():
        tech = probe(video)
        meta["video_file"] = str(video.resolve())
        meta["tech"] = tech
        if not args.no_frames:
            frames = extract_frames(video, out_dir / "frames", tech["duration"])
        audio = analyze_audio(video, out_dir, tech["duration"], tech["has_audio"])
        meta["audio"] = audio
        if args.transcribe and audio.get("wav16k"):
            tr = HERE / "dy_transcribe.py"
            log("转写: 调用 dy_transcribe.py …")
            subprocess.run([sys.executable, str(tr), str(out_dir / audio["wav16k"]), "--out", str(out_dir),
                            "--model", args.model], text=True, encoding="utf-8", errors="replace")
            transcript_txt = out_dir / "transcript.txt"
    elif meta.get("images"):
        log("这是图文帖子（非视频），下载图片 …")
        (out_dir / "frames").mkdir(exist_ok=True)
        for i, u in enumerate(meta["images"]):
            if u:
                download([u], out_dir / "frames" / f"image_{i + 1:02d}.jpg")

    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(out_dir, meta, tech, frames, audio, transcript_txt)
    log(f"完成 → {out_dir.resolve()}")
    print()
    print((out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

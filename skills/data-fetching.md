---
name: data-fetching
description: 用户给了抖音分享链接或口令却没给完整素材时，按这里把视频、文案、转写、互动数据拿全再分析
scope: content_type
applies_to: [抖音]
stage: [选题]
priority: 40
version: 0.1.0
owner: 运营
# draft：流程依赖项目外的脚本（~/.claude/skills/... 下的 dy_fetch.py），Agent 读不到那个目录，
# 加载了只会让模型去跑一个不存在的命令。要启用先把脚本搬进项目，再改成 active。
# （2026-09-23 审查：之前这篇没有 frontmatter，被静默忽略，没人知道它不生效）
status: draft
---

# 数据获取流水线

本文件记录如何从一个抖音分享链接 / 短链 / 口令出发，自动拿到分析所需的全部数据。
读这个文件的时机：用户给了链接但没给完整素材，并且环境里有 `sharelink` 或 `ffmpeg` 或 TikHub API key。

## 0. Windows 本机快速路径（优先用这个）

本机没有 `sharelink` / `whisper-cli`，下面第 1–4 节的命令已经全部封装进两个脚本，直接跑即可：

```powershell
python "$env:USERPROFILE\.claude\skills\douyin-viral-analyzer\scripts\dy_fetch.py" "<分享口令 / 链接 / aweme_id / 本地mp4>" --out <目录> --transcribe
```

| 本文小节 | 脚本里对应的实现 |
| --- | --- |
| 1. 分享链接 → video_id | `dy_fetch.py` `resolve_short_link()`：跟随 `v.douyin.com` 跳转取 ID，不做任何签名 |
| 2. 下载视频 + 元信息 | 有 `TIKHUB_API_KEY` → TikHub `fetch_one_video_v3` 拿 `play_addr` 直链下载；否则 `--cookies cookies.txt` → yt-dlp；否则请用户给本地 mp4 |
| 3. 抽帧 | 均匀轨 `fps=` 按时长自动选（≤10s 4fps / ≤30s 2fps / ≤120s 1fps / 更长 0.5fps）+ 场景轨 `select='gt(scene,0.15)'`，始终产出 `first.jpg` `last.jpg`，帧宽缩到 720 省 token；时间戳写在 `frames/index.json` |
| 4.1 音频性质 | `volumedetect` + `silencedetect=noise=-30dB:d=0.5` → `audio.json`，同时产出 16k 单声道 `audio16k.wav` |
| 4.2 口播逐字稿 | `dy_transcribe.py`：faster-whisper `large-v3`（CPU int8），输出 `transcript.srt/.txt/.json`；国内默认走 `hf-mirror.com` 下模型 |
| 5. TikHub 数据补全 | `dy_fetch.py` `tikhub_fetch()`：详情 + 评论（APP 接口失败自动切 WEB），评论按赞数排序写进 `summary.md` |

单独转写一个已有的音频/视频：

```powershell
python "...\scripts\dy_transcribe.py" <audio.wav 或 video.mp4> --out <目录> [--model medium]   # 想快一点可换 medium
```

**不要做：** 伪造 `a_bogus` / `X-Bogus` 签名、伪造浏览器指纹、读取浏览器 Cookie 数据库来绕过抖音网页接口的保护。本机安装时已验证这条路会被安全策略拦下，也不应该做。

## 完整流水线一览

```
分享口令/短链
    │
    ├─ sharelink parse -json  ──→  video_id + 无水印 play_url
    │
    ├─ curl 下载 mp4
    │     │
    │     ├─ ffprobe          ──→  时长 / 帧率 / 分辨率
    │     ├─ ffmpeg 抽帧       ──→  画面分析（场景切换 + 均匀采样）
    │     └─ ffmpeg 音频统计   ──→  BGM 节奏 / 是否有口播
    │
    └─ TikHub API（用 video_id）
          │
          ├─ fetch_one_video_v3  ──→  标题/作者/统计/hashtags/BGM元信息
          └─ fetch_video_comments(web)  ──→  Top 20 热门评论
```

每个环节都可能失败，下面分别说怎么用、怎么兜底。

---

## 1. 分享链接 → video_id

用户给的"分享口令"形如：
```
8.74 :9pm odA:/ M@J.Iv 02/18 "八秒让你爱上非遗" # 非遗 # 转场 ...
https://v.douyin.com/PRhsfcq7pIk/
```

如果环境里有 `sharelink` CLI（用户的 macOS 上通常 `/opt/homebrew/bin/sharelink`，也有 `dy` shell 函数包装）：

```bash
sharelink parse -json 'https://v.douyin.com/PRhsfcq7pIk/'
# → {"play_url":"https://aweme.snssdk.com/...","video_id":"7604093452372472795"}
```

如果没有 `sharelink`：
- 抖音短链 `v.douyin.com/<code>/` 跳转两次后最终落到 `www.douyin.com/video/<aweme_id>`，aweme_id 即 video_id
- 用 `curl -sI -L 'https://v.douyin.com/<code>/' | grep -i location` 可以拿到，但抖音 Web 页面要登录，所以仅能拿 ID，拿不到内容
- 拿不到的就停下来问用户

## 2. 下载视频 + 元信息

```bash
mkdir -p /tmp/dy-analysis
cd /tmp/dy-analysis
curl -sL -o video.mp4 "<play_url>"
ffprobe -v error -show_entries format=duration,bit_rate \
  -show_entries stream=width,height,r_frame_rate,codec_name \
  -of default video.mp4
```

关键判断：
- 时长 ≤ 10s 的视频 → 用密集抽帧（每 0.4s 一帧）
- 时长 10–30s → 每 0.7s 一帧
- 时长 > 30s → 用场景切换检测为主

## 3. 抽帧

### 🚨 密度铁律

**宁愿密集慢，不可丢失关键信息。** 漏一帧 = 归因偏差一次。读一张帧的成本远低于错一次归因的成本。

**两条不可破的规则：**
1. **场景切换 + 均匀采样必须双轨并行**，不能用一个代替另一个
2. **永远第一时间读第 0 帧和最后一帧**，无论抽帧策略是什么

### 抽帧命令（按时长选 fps）

```bash
mkdir -p frames scene_frames

# 决定 fps（按视频时长）
# - ≤ 10s:    fps=4      （每 0.25s 一帧）
# - 10-30s:   fps=2      （每 0.5s 一帧）
# - 30-120s:  fps=1      （每 1s 一帧）   ← 不要再用 1/4！
# - > 120s:   fps=1/2    （每 2s 一帧）

# 轨道 A：均匀采样 — 清晰静态画面，识别内容/人物/字幕/表情
ffmpeg -hide_banner -loglevel error -i video.mp4 -vf "fps=1" frames/f_%03d.jpg

# 轨道 B：场景切换检测 — 阈值用 0.15（宁可多抓不要漏抓）
ffmpeg -hide_banner -loglevel error -i video.mp4 \
  -vf "select='gt(scene,0.15)',showinfo" -vsync vfr scene_frames/s_%03d.jpg
```

### 为什么两套都要

- **场景切换帧** = 捕的就是切换瞬间，大概率运动模糊 → 用来数节奏密度、判断剪辑速度
- **均匀采样帧** = 相对静止画面 → 用来辨认具体内容（人物、字幕、品类、表情、文字、道具）

历史教训：
- **教训一**：只读场景帧，差点错过 8 秒非遗视频开头那个安静的"素人在空舞台"钩子 — 那一帧是整条视频归因的核心
- **教训二**：分析 115 秒五四视频时用 `fps=1/4`（每 4 秒一帧）只拿到 29 帧，1m55s 的微纪录片**很可能漏掉了关键瞬间**（旁白卡点、人物特写、字幕浮现），后期无法验证

### 真的资源紧张时怎么减

如果环境真的有限额（比如 cowork），减的顺序：
1. **保留**：第 0 帧、最后一帧、所有场景切换帧 — 这些是底线
2. **可减**：均匀采样的中段密度 — 比如 60s 视频从 1fps 减到 0.5 fps
3. **不可减**：场景切换的阈值 — 不要把 0.15 调到 0.3 来减少帧数

## 4. 音频分析

### 4.1 音频性质判断（ffmpeg，无需模型）

```bash
# 总体音量 / 动态范围
ffmpeg -hide_banner -i video.mp4 -af "volumedetect" -vn -f null - 2>&1 | grep -E "mean|max"

# 静音段检测（判断有无口播）
ffmpeg -hide_banner -i video.mp4 -af "silencedetect=n=-30dB:d=0.2" -vn -f null - 2>&1 | grep silence
```

解读：
- `max_volume ≈ 0 dB` 且 `mean_volume > -15 dB` → 极限压限，典型短视频"听感最大化"母带
- 无静音段 → 纯 BGM 驱动，无口播留白
- 有 ≥ 0.3s 的静音段 → 大概率有口播或刻意停顿
- 音频在前 0.1s 急速从 -45dB 冲到 -10dB → 典型 "drop in" 入场，常和大转场卡点联动

### 4.2 口播逐字稿（whisper-cli + large-v3）

**判断标准**：
- 若上一步 `silencedetect` 显示有 ≥ 0.3s 静音段 → 几乎肯定有口播/旁白 → **必须转写**
- 若是纯 BGM 无口播 → 可以跳过转写省时间
- 若不确定 → **转写**（成本几十秒，但漏掉口播 = 漏掉一半归因证据）

**命令：**

```bash
# Step 1: 抽 16kHz 单声道 WAV（whisper.cpp 的标准输入）
ffmpeg -hide_banner -loglevel error -y -i video.mp4 \
  -ar 16000 -ac 1 -c:a pcm_s16le audio.wav

# Step 2: 转写（中文，输出带时间戳的 srt + 纯文本 txt）
whisper-cli \
  -m ~/.cache/whisper/ggml-large-v3.bin \
  -l zh \
  -osrt -otxt \
  -of transcript \
  audio.wav

# 产物：
# - transcript.srt （带时间戳，用于对齐画面）
# - transcript.txt （纯文本，用于直接阅读）
```

**常用参数：**
- `-l zh`：中文（必加，否则会被识别成日韩等近邻语言）
- `-l auto`：自动检测语言（用于多语言混合）
- `-osrt`：输出 SRT 时间戳字幕
- `-otxt`：输出纯文本
- `-of <prefix>`：输出文件前缀
- `-t <N>`：线程数，默认 4，M 系列芯片可设 8 加速
- `-pp`：打印处理过程（调试用）

**性能参考**（Apple Silicon Metal）：
- 1.5 分钟 720p 视频音频 → 转写约 10–20 秒
- 15 分钟视频音频 → 转写约 1–2 分钟

**何时降级到 medium 模型：**
- 仅当 large-v3 不可用 / 磁盘紧张时
- medium 中文准确率明显下降，可能漏识专有名词、误识口音
- 紧急场景：`-m ~/.cache/whisper/ggml-medium.bin`

### ⚠️ 同音字风险与画面字幕交叉验证

large-v3 在中文古语 / 口号 / 历史专有名词上偶尔会**同音字错识**。实测案例（五四视频）：
- "外**争**国权 内**除**国**贼**"（原文）→ 转写成 "外**政**主权 内**出**国**粹**"（3 处错）
- "腰杆能挺得更**直**" → 转写成 "腰杆能挺得更**深**"
- "哪有什么**局**外人" → 转写成 "哪有什么**居**外人"

**应对：** 抖音爆款视频几乎都有**画面字幕/花字**（中文创作者习惯）。把均匀采样帧里有字幕的帧单独读一遍，**用画面字幕修正口播转写的同音字错误**。这是双向校验，不是择一信任。

写报告时，对于已经验证的金句台词照搬原文；对于不确定的，标注 "（疑似：xxxx）" 保留两种可能。

### 4.3 转写产出怎么用进报告

| 转写产出 | 写入报告哪一节 |
| --- | --- |
| 第 0–3 秒的台词原文 | 节 1 钩子拆解（"第 1 句台词"字段） |
| 全篇按 SRT 切段的旁白 | 节 2 脚本结构与节奏（每段时间轴的内容描述） |
| 关键金句/反问/排比 | 节 3 情绪曲线 + 节 7 爆款归因 |
| BGM 是否盖过人声（音量曲线 + 转写完整度） | 节 4 视听元素 |
| 结尾 CTA 原话 | 节 8.3 拍摄/剪辑要点的"必须保留"清单 |

## 5. TikHub API 数据补全

**API key 已配置**：`$TIKHUB_API_KEY` 在 `~/.claude/settings.json` 的 env 段。所有 curl 直接 `-H "Authorization: Bearer $TIKHUB_API_KEY"` 即可，不需要问用户。如果某次发现 env 里没有（比如别人的机器），让用户提供并加入 settings.json。

用这套组合：

**每条视频 0.002 credit（≈ $0.002）**，足够拿全所有分析所需的公开数据。

### 视频详情（一次调用拿全）

```bash
curl -sS -H "Authorization: Bearer $TIKHUB_API_KEY" \
  "https://api.tikhub.io/api/v1/douyin/app/v3/fetch_one_video_v3?aweme_id=$AWEME_ID" \
  -o video_detail.json
```

为什么 V3：V1/V2 对版权受限内容会返回空（reason=8），V3 无此限制。同价 0.001 credit。

返回的 JSON 路径：
- `data.aweme_detail.desc` — 标题/文案
- `data.aweme_detail.statistics.digg_count` / `comment_count` / `collect_count` / `share_count` / `download_count`
  - **注意**：`play_count` 几乎总是返回 0，抖音 API 限制。要播放数得另用 `fetch_video_statistics`（仍是 0.001 credit，但也常常拿不准）
- `data.aweme_detail.author.nickname` / `signature` / `total_favorited` / `aweme_count` / `uid`
  - **注意**：`follower_count` 经常返回 0，需要另调 user info 接口才能拿粉丝量
- `data.aweme_detail.text_extra[].hashtag_name` — hashtags 列表
- `data.aweme_detail.music.title` / `author` / `is_original` — BGM 信息
- `data.aweme_detail.create_time` — Unix 时间戳

### 评论（注意 APP 经常 400，要兜底 WEB）

```bash
# 首选 APP 接口
curl -sS -H "Authorization: Bearer $TIKHUB_API_KEY" \
  "https://api.tikhub.io/api/v1/douyin/app/v3/fetch_video_comments?aweme_id=$AWEME_ID&cursor=0&count=20" \
  -o comments.json

# 如果返回 code != 200 或 detail.code == 400，立刻切到 web 版本
curl -sS -H "Authorization: Bearer $TIKHUB_API_KEY" \
  "https://api.tikhub.io/api/v1/douyin/web/fetch_video_comments?aweme_id=$AWEME_ID&cursor=0&count=20" \
  -o comments.json
```

返回 JSON 路径：`data.comments[]`，每条有：
- `text` — 评论内容
- `digg_count` — 点赞数（**按这个排序才是真热门**）
- `reply_comment_total` — 回复数
- `user.nickname` / `user.uid`

按 `digg_count` 倒序取 Top 5–10 即可。

### 选型对比表

同价（0.001 credit）但字段差异巨大，记住这个表，省得每次现查：

| 接口 | 字段完整性 | 用途 |
|---|---|---|
| `fetch_one_video_v3` | ⭐⭐⭐⭐⭐ 全字段 | **默认选这个** |
| `fetch_one_video_v2` / `v1` | ⭐⭐⭐⭐ 缺版权受限 | v3 失败再试 |
| `fetch_video_statistics` | ⭐⭐ 只有 4 个数 | 缺评论数和收藏数，**不要单独用** |
| `fetch_video_comments`(app) | ⭐⭐⭐ 经常 400 | 首选但要兜底 |
| `fetch_video_comments`(web) | ⭐⭐⭐⭐ 稳定 | APP 失败时的兜底 |

### 其他可选

- 想看作者整体数据：`fetch_user_post_videos` + 自己计算近 30 天爆款率
- 想拉评论的回复：`fetch_video_comment_replies`（按需，单独 0.001 credit/页）
- 想做评论词云：抖音自带 `billboard/fetch_hot_comment_word_list`（如果可用）

### 星图（Xingtu）接口 — 拿播放量和横向对标（贵 20 倍但值）

抖音常规接口 `play_count` 几乎总返回 0。要拿真实播放数和账号级 KPI，必须走星图接口。

**两步走（共 0.021 credit ≈ $0.021）：**

```bash
# Step 1: UID → kolId （0.001 credit）
curl -sS -H "Authorization: Bearer $TIKHUB_API_KEY" \
  "https://api.tikhub.io/api/v1/douyin/xingtu/get_xingtu_kolid_by_uid?uid=$UID" -o kolid.json
# 取出 data.id 作为 kolId

# Step 2: KOL 视频表现 （0.02 credit）
curl -sS -H "Authorization: Bearer $TIKHUB_API_KEY" \
  "https://api.tikhub.io/api/v1/douyin/xingtu/kol_video_performance_v1?kolId=$KOL_ID&onlyAssign=false" -o perf.json
```

**返回的两大块金矿：**

1. **`data.data_description`**：账号级 KPI（互动率、中位播放、完播率），每个都有 `compare_avg`（vs 同类）和 `compare_author`（自己作品分位），**这是判断该账号是否真"头部"的硬指标**
2. **`data.latest_item_info[]`**：最近 ~15 条非商业作品，每条带 `play` / `like` / `comment` / `share` 真实数据，可以用来**对比目标视频在该账号矩阵里的真实分位**

**关键用途：避免"外部看是爆款，账号内是常态"的误判**

我曾经把一条 96 万赞的视频归因为"内容设计精妙带来的爆款"。但拉了星图发现该账号中位播放就是 516 万，这条只是"中等偏上"。真正的差异化在转发比 — 比该账号均值高 1.8 倍。**没有星图的横向对标，这个洞察出不来。**

**注意限制：**
- 只对**在星图注册过的 KOL** 生效（很多非商业账号没注册，会返回空）
- `latest_item_info` 只返回最近 15-20 条，老视频可能不在内；要老视频的播放数，还是只能用 `fetch_video_statistics` 单条抓（也常常拿不准）
- `tag` 字段给的是星图官方分类（比如"运动健身"、"美妆"），有时和创作者自我认知不同，但**算法对该账号的归类基本看这个**

**何时不调星图：**
- 已经能从其他来源大致判断账号规模（比如用户自己说"这是个百万粉博主"），就不用花这 0.02
- 视频太老（>3 个月），星图返回不到，浪费钱
- 账号明显是素人/小号（< 10 万粉），大概率没在星图

---

## 6. 错误兜底总则

| 失败场景 | 处理 |
|---|---|
| sharelink 没装 | Windows 用 `scripts/dy_fetch.py`（自动解析 ID）；仍拿不到视频就让用户在抖音 App「保存到本地」后给 mp4 |
| play_url 下载 403/404 | 让用户用抖音 APP "保存到相册" 后上传文件 |
| TikHub 接口 401 | API key 失效/未提供，让用户检查 |
| TikHub APP 接口 400 | 立刻切 WEB 版本（已经成案例） |
| play_count = 0 | 抖音 API 限制，**不要谎称视频没人看**，标注"播放数 API 不返回" |
| follower_count = 0 | 同上，需要另调用户接口 |
| 评论数 > 1000 但 API 只返回 20 | 翻页（cursor）拿更多，但 Top 20 按 digg_count 排序通常已够 |

---

## 7. 数据 → 报告的写入对应

| 数据来源 | 写入报告哪一节 |
|---|---|
| `sharelink` + `ffmpeg` 抽帧 | 节 1 钩子拆解 / 节 2 脚本结构 / 节 3 情绪曲线 / 节 4 视听元素 |
| `ffmpeg` 音频统计 | 节 4 视听元素的 BGM 子项 |
| `fetch_one_video_v3.statistics` | 节 5 公开数据比例分析 |
| `fetch_one_video_v3.author` | 节 0 元信息 + 节 7 爆款归因（账号势能因素） |
| `fetch_one_video_v3.create_time` + hashtags | 节 0 元信息 + 节 7 归因（时机因素） |
| `fetch_video_comments` Top 20 | 节 6 评论区洞察（**最重要的归因校验数据**） |

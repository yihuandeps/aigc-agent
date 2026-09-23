# vendor 本地补丁记录

这些脚本原样拷自上游 `rainwell-douyin-viral-analyzer`，**只在必要时打补丁**，
每条都记在这里，便于将来从上游同步时保留。

## dy_fetch.py

### 2026-09-11 · 视频详情端点名错误
- 上游：`/app/v3/fetch_one_video_v3`
- 实测：返回 HTTP 400（"Request failed. Please retry."）
- 修正：`/app/v3/fetch_one_video`，并加 `/web/fetch_one_video` 作为回退
- 验证：两个端点都实测 200，取到作者「天守垣」

评论端点 `fetch_video_comments` 上游写法正确，未改。

## 没有移植的文件

- `dy_transcribe.py` —— agent 已有 `transcribe` function（whisper-1，可直出 srt），
  留两套转写会让血缘分叉

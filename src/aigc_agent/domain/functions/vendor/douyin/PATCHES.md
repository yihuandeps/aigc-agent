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

## render_pdf.py

### 2026-10-06 · 样式文件放到脚本找的位置（脚本本身未改）
- 脚本按「脚本目录的上一级 / assets / report.css」找样式（照搬上游 skill 的目录布局），
  找不到就静默用空样式
- 之前样式文件放在仓库根的 `assets/report.css`，脚本读不到，PDF 报告一直没有样式（2026-09-29 审查）
- 处理：挪到 `vendor/assets/report.css`；以后从上游同步时，把上游的 `assets/report.css` 放到这里

## 没有移植的文件

- `dy_transcribe.py` —— agent 已有 `transcribe` function（whisper-1，可直出 srt），
  留两套转写会让血缘分叉

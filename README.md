# AIGC 内容创作 Agent

内部提效的半自动内容生产 harness。**Agentic 为主**：生产能力做成 function，模型自己决定调哪个、
什么时候请人拍板；图作为可选的标准化流程保留。覆盖短视频 / 图文 / 文案 / 图像四种形态。

- 架构与模块划分：[ARCHITECTURE.md](ARCHITECTURE.md)
- 技术栈决策：[TECH_STACK.md](TECH_STACK.md)

**当前进度**

| 阶段 | 状态 | 内容 |
|---|---|---|
| P0 | ✅ | Loop Runtime · Model Gateway · Context · Tool Runtime · Permission · Events |
| P1 | ✅ | **Content Functions · `request_review` 挂起/恢复** · Asset 血缘 · **Memory Store + Memory Agent（M8.2 关键词提取/召回）** · Router · Graph Runtime + 回退边（模式 B）· CLI 人审 |
| P2 | ✅ | **执行痕迹 → DAG + `/rollback` 单步重跑** · 多模态 function（生图 / 生视频 / TTS / 转写 / 成片）· **MCP Hub 接入第一个真实 server** · **Cost Guard 接线** · `cache.mode` 实测定为 `auto_prefix` |
| P3 | ✅ | **M10 子代理运行时** · **M8.2 Memory Brief（四区 + 冲突暴露 + 子代理整理）** · **M7 Skill Hub 目录进上下文 + 热加载 + 版本回滚** · **capability_budget 两 Hub 共管** · **M9 统一检索** |
| P4 | ✅ | **M15 机审（只出清单不改字）· M16 待发布包（人上传，无 publish）· M17 数据回流（提炼成 DATA 来源的账号层记忆）** |
| P5 | ✅ | **M10 并行扇出与差异汇总 + 图 `parallel`/`join`** · **Cost Guard 项目级/日级预算（台账跨会话）** · **M6 事件流落盘 + 会话回放 / 成本看板 / 痕迹重建** |

> 文本模型走 **Kimi Code**（`api.kimi.com/coding/v1`，订阅额度制，不按 token 开账单）。
> 它和 platform.moonshot.cn 是两条产品线，端点不通用 —— 见 `config/models.yaml` 顶部的校准记录。
> `agent doctor` 会逐个 provider 核对 model id 是否在各自端点可用。

---

## 快速开始

```bash
# 1. 环境（需要 Python 3.11+，本机 3.9 是系统默认，别用它）
py -3.11 -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"

# 2. 填密钥
cp .env.example .env      # 然后编辑 .env

# 3. 自检
./.venv/Scripts/python.exe -m aigc_agent.interfaces.cli.main doctor
```

## 命令

```bash
# 自检：配置解析 / 密钥 / 工具注册 / MCP server / 各 provider 连通性与模型 id 核对
python -m aigc_agent.interfaces.cli.main doctor
python -m aigc_agent.interfaces.cli.main doctor --cache   # 再实测一次前缀缓存是否命中

# 列服务端实际可用的模型 id
python -m aigc_agent.interfaces.cli.main models

# 交互式对话（-v 看上下文装配与模型请求细节）
python -m aigc_agent.interfaces.cli.main chat -v

# MCP server：状态 / 权限审计 / 手动调一次
python -m aigc_agent.interfaces.cli.main mcp status
python -m aigc_agent.interfaces.cli.main mcp audit
python -m aigc_agent.interfaces.cli.main mcp call material_lib__search_materials '{"query": "日落"}'

# 方法论 skill：列表 / 查看 / 版本记录 / 一键回滚 / 体检
python -m aigc_agent.interfaces.cli.main skill list
python -m aigc_agent.interfaces.cli.main skill history drama-script
python -m aigc_agent.interfaces.cli.main skill rollback drama-script
python -m aigc_agent.interfaces.cli.main skill check

# 记忆库：列表 / 简报 / 晋升到账号层 / 作废错记忆
python -m aigc_agent.interfaces.cli.main memory list
python -m aigc_agent.interfaces.cli.main memory brief --topic 露营
python -m aigc_agent.interfaces.cli.main memory promote mem_xxxxxxxxxx
python -m aigc_agent.interfaces.cli.main memory forget mem_xxxxxxxxxx

# 待发布包：列表 / 上传清单 / 人上传后记回链接
python -m aigc_agent.interfaces.cli.main release list
python -m aigc_agent.interfaces.cli.main release show as_xxxxxxxxxx
python -m aigc_agent.interfaces.cli.main release published as_xxxxxxxxxx --url https://…

# 数据回流：人工录入 / 列表 / 复盘（满 3 个作品后自动提炼进账号层记忆）
python -m aigc_agent.interfaces.cli.main analytics record as_xxxxxxxxxx --views 1200 --likes 80 --completion 0.42
python -m aigc_agent.interfaces.cli.main analytics review --platform douyin

# 会话回放 / 成本看板 / 痕迹重建（事件流按会话落盘在 workspace/logs/sessions/）
python -m aigc_agent.interfaces.cli.main sessions list
python -m aigc_agent.interfaces.cli.main sessions replay <session> --tail 40
python -m aigc_agent.interfaces.cli.main sessions cost <session>
python -m aigc_agent.interfaces.cli.main sessions trace <session> --mermaid

# 一条命令出短视频（配方驱动，见 config/recipes/）
python -m aigc_agent.interfaces.cli.main video make "AI 芯片" --dry-run
```

`agent chat --session <项目名>`：记忆、项目级预算、**对话现场**都按它归档；不指定就是 `default`。
同名 session 重启会恢复上次对话窗口接着聊（快照在 `workspace/memory/sessions/`，只存轮次不存 pin）；
换个名字就是全新会话。

**产物目录**：新 session 开工前会问一次「产物存到哪个文件夹」（回车 = `workspace/output/<session>/`，
可填任意路径如 `E:\我的短剧`）。生成的文本（`texts/*.md`）、图片（`images/`）、视频（`videos/`）、
成片导出（`exports/`）都落这里；`/out` 查看，`/out <路径>` 随时改，同名 session 重启沿用。

chat 内可用：`/exit` 退出 · `/stat` 看窗口、成本、缓存命中与能力区占用 · `/tools` 工具列表 ·
`/trace` 执行痕迹 · `/mermaid` 流程图 · `/rollback <资产id>` 回退到某版本重来 ·
`/budget` 预算用量 · `/budget reset` 清零 · **`/brief` 当前记忆简报 · `/skills` 已加载的 skill**

模型调 `request_review` 请人审时会暂停，当场决策后它才继续：
**`a`** 采纳（可带补充要求：`a 控制在60集`）· **`r`** 打回重写 · **`j`** 方向不对退回（r/j 必填理由）

长剧批量出稿嫌逐条确认麻烦：**`/auto on`** 后小节点人审（如某一集单集）自动采纳、不再打断你，
**大节点（剧本/视频生成/图片生成）仍会停下来等你拍板一次**；每条决策仍会逐条打印（标 `auto 自动`）。
随时 **Ctrl+C** 叫停，**`/auto off`** 回到逐条确认。
开的那一刻就自动跑起来：挂着的人审当场按采纳结案、撞迭代上限停下的轮次当场续跑，
不用再发任何话触发（在人审的决策口直接输 `/auto on` 也行）。
大小节点靠 `request_review` 的 `stage` 区分：含「剧本/视频生成/图片生成」关键词的是大节点，
「第N集/单集」一律算小节点；名单在 `LoopRuntime` 的 `major_stages` 参数里可调。
L-external 不可逆操作与预算超限仍然会问你。

**`/auto on` 同时打开按集流水并行**（EpisodePipeline）：剧本每满 5 集自动拆这批分镜；
全剧写完自动拆资产库、渲参考图；之后逐集自动出视频提示词、逐集渲染出 `第N集.mp4` ——
写剧本和工程链并行跑，环节间靠资产事件实时交接，不用等全剧写完再开工。
族裔/语言/总集数从 `<产物目录>/.drama-state.json` 读，缺了会提示你补一句（不猜）。
渲染派发前过预算闸，超支自动暂停。

每轮执行期间底部有**实时进度窗**：当前阶段、批量任务进度条（如 `渲染分镜视频 45/360`）、
正在运行的工具、迭代数、用时与本轮成本；问人（权限/预算/人审）时自动暂停。

## 检查

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q     # 全量验收测试（含真 ffmpeg、真 MCP stdio）
./.venv/Scripts/python.exe -m ruff check src/ tests/
./.venv/Scripts/lint-imports.exe                   # 分层依赖约束
```

> ⚠️ 分层检查必须用 `lint-imports`，**不要用 `python -m importlinter.cli`**
> ——后者是错误入口，会静默 no-op 并返回 0，看着像通过其实什么都没检查。

---

## 2026-09-17 运行优化（按真实日志复盘）

两天的短剧会话文本花费 ¥126，主因是逐集写作在主循环里把上下文推到 25 万 token 撞上模型上限。
这轮改动：

| 问题 | 改动 | 在哪 |
|---|---|---|
| 一轮 12 次迭代把上下文推到 26 万，401 整轮作废 | **轮内压缩**：历史大轮与当前轮旧迭代的工具参数/结果折叠成存根（正文都在资产库）；撞上限压缩后重试一次 | `harness/context/compaction.py`、`assembler.py`、`loop.py` |
| 估算比真实低四成，18 万熔断实际 25 万才触发 | 每次按真实 usage **校准估算**；发请求前**预检**，超预算先剔历史轮；熔断降到 12 万 | `assembler.observe()`、`loop._assemble()` |
| 重启装回 10 轮 = 22 万 token 冷启动，一次 ¥4.6 | 快照恢复按 token 上限截（默认 3 万） | `memory/session.py` |
| 60 集在对话里逐集写，一次「继续」只写 1–2 集 | **写作交给子代理**：`drama_write_episodes` 一批 5 集，每集约 1 万上下文，落资产后主 Agent 只收 id 和钩子 | `domain/functions/episodes.py`、skill `/episode` |
| `list_assets` 全吐被截断 → 模型编 id | 带过滤/分页；新增 `find_episode(N)`；报错附相近 id；列表类工具截断上限单独放宽 | `functions/content.py`、`assets/store.py` |
| 广告法极限词打在剧本台词上，模型花几十元重写 60 集 | 规则按资产类型生效（`applies_to`），剧本不过极限词/数字出处/AIGC 标识；`check_compliance` 支持批量 | `config/compliance.yaml`、`domain/compliance` |
| 金额超限后每轮只跑 1 次迭代且无出口 | 撞上限当场问追加多少；`/budget allow 50`；停机文案写明出口 | `loop.py`、`cli/main.py` |
| 记忆召回命中 0（输入全是「继续」） | 召回查询带上上一轮助手文本；打回理由按 session 归档；**项目卡**每轮 pin（剧名/集数/进度/各产物最新 id） | `app.py`、`domain/drama/card.py` |
| 同一迭代提 9 次 request_review 只挂 1 个，其余"假成功" | 多余的按失败回填；`request_review` 加显式 `major` 参数 | `loop.py`、`functions/content.py` |
| 24 万 token 请求 90–290 秒超时 | 超时按请求大小放宽；额度用尽/超时/上下文超限各给可读提示 | `model/gateway.py` |
| 写剧本被 drama 角色的 JSON 约束连带 | 新角色 `drama_prose` / `episode_writer` | `config/models.yaml` |
| 资产库一个角色只有一套服装，几十个场景一套穿到底，分镜里没有换装 ID，视频里服装按文字乱变 | 第②步**服装按场景分配**（每套带 `scenes` / `episodes`，返回分配表与缺口）；第③步按镜头所在场景**确定性绑定**服装 ID（同场景一致、换场景才换装，裸角色名也换成服装 ID）；第④步参考图按**主形象 → 各场景服装**生成（服装图参考主形象保脸，纯白底单张全身，资产库里不再出现三视图） | `domain/drama/wardrobe.py`、`models.py`、`prompts.py`、`functions/drama.py` |
| 参考图包只渲了角色（`only=characters`），分镜引用服装名，逐镜匹配全空，4 段视频「参考 0 张」静默渲完 | 服装名退回该角色主形象（保脸）；包和分镜一个都对不上**先报错不花钱**；不传 `rendered_id` 按血缘自动找、传成资产库 id 自动换算；参考图顺序人物 → 前序片段 → 场景道具；`drama_render_assets` 增量复用（补跑只生成缺的，主形象不换脸）并报覆盖率 | `domain/functions/drama.py` |
| 角色图的脸没听真实感提示词：提示词里写了毛孔雀斑，出来还是磨皮脸（真实感段挂在「证件照」后面的末尾，被当成可选氛围） | 三道保险：① 硬约束段（中英双语、祈使句、"忽略下文相反要求"）放人物提示词**开头**，末尾再复述，「证件照」改成「正面半身照，原图直出无美颜」；② 模型写的描述先清洗（磨皮/光滑无瑕/柔光/高度对称 → 真实感说法），没写具体瑕疵的补默认锚点，第②步规则要求每个角色写≥2 处具体瑕疵；③ 生成后 `realism_check` 角色看图打分（毛孔/斑点/硬光），不合格用更狠的提示词重生成（`media_models.yaml` drama.`realism_gate` / `realism_retries`），都不过留分最高的并在结果里标「仍不达标」。视频提示词同样清洗 + 尾部硬约束 | `domain/realism.py`、`domain/functions/drama.py`、`config/models.yaml`、`config/media_models.yaml` |
| 集数一多同一角色的声音会漂：seedance 逐段发声，每段各自"发明"一个声音，几十段下来第 8 集已不是第 1 集那个人 | 两道锁：① 第②步资产库每个角色一张**音色卡**（`voice`：性别 \| 年龄听感 \| 音高质地 \| 语速节奏 \| 口音习惯，`VOICE_RULES` 追加在原 prompt 后），渲染时按「台词归它前面最近的角色」判出说话人，把说话角色的音色卡锁进提示词开头；② **音色锚点**：每个角色第一段成功片段（优先独白段）存成锚点表资产（`tool:drama_voice_anchors`，跨集沿用），之后他开口的每段都把锚点片段作为参考视频传入（`video_urls`，提示词写「声音必须与 @视频N 中该角色一致，只取声音」）；锚点镜头先渲、其他有该角色开口的镜头排在其后（同一集内也不各是各的）；链接过期（`ref_url_ttl_hours`）用新片段接替并写回；`drama_voice_anchors list/pin/clear` 查看或人工指定。前序片段也改走 `video_urls`（`ref_fields` 可配），`gen_video` 新增 `video_urls`/`audio_urls` | `domain/drama/voice.py`、`domain/functions/drama.py`、`domain/functions/media.py`、`domain/generators/catalog.py`、`config/media_models.yaml` |
| 视频经常生成失败（真实日志：一集 4 段挂 3 段）：① 拼完第 1 集后 `compose_video` 把片段 `uri` 换成本地路径，下一轮锚点传给模型是 `E:\…\as_xxx.mp4` → `HTTP 400 Invalid format for video_urls[0]`；② 轮询 2 次 ConnectError 就把任务判死，服务端还在跑、照样计费；③ 每次失败都要整集重生成 | ① 下载本地副本**不改 uri**（本地路径记 `gen_params["local"]`，与 gen_image/gen_video 同口径），参考视频只认 http(s) 链接，本地路径/过期链接一律不传；② 轮询按用户口径**每 3 秒一次、最多 200 次**（`polling.max_polls`，`backoff 1.0`），网络错误连续 10 次才判失败（`max_transient`），提交阶段网络错误重试 2 次（`submit_retries`）；③ `drama_render_shots` 默认 `reuse=true`：只重生成失败/缺的段，成功的复用后按原序拼接；单段网络类失败自动原样重试（`drama.video_retries`，400 与超时不重试）。实测也证实 APIMart 的 seedance-2.0 端点认 `video_urls` 字段（它校验了格式） | `domain/functions/video_edit.py`、`harness/model/media.py`、`domain/functions/drama.py`、`config/media_models.yaml` |
| 真实感矫枉过正：成片里雀斑、皱纹太明显，人物太丑（用户要减到原来的 1/3） | 真实感分档 `drama.realism_level`：**subtle**（默认，真实但干净：细看有毛孔和纹理，雀斑至多零星几点极淡，不要明显皱纹）/ natural（早上那版）/ strong。档位贯穿人物图与视频提示词的硬约束、①②③步写作规则（subtle 只许写一处轻微细节，严禁粗大毛孔/明显皱纹/大片斑）、校验目标；subtle 档还会把描述里写重的瑕疵压回去（`soften_flaws`：毛孔粗大→细看可见的毛孔、满脸雀斑→零星几点极淡、深深的法令纹→浅浅的），老资产库不用重生成也能吃到。校验结果带方向（smooth/heavy/light），重生成按方向改：做旧过头就往轻里生 | `domain/realism.py`、`domain/drama/prompts.py`、`domain/functions/drama.py`、`config/media_models.yaml` |
| 视频画面里不许出现字幕/文字（用户定的**最高优先级**约束；模型会把台词烧成字幕条，后期去不掉） | 两道保险：① 所有 `gen_video` 统一在提示词最外层包禁令（开头一段最高优先级 + 末尾复述，中英双语；`allow_text=true` 才放开），短剧/配方/手动调用都逃不掉；② 短剧渲染的**字幕门**：每段生成后抽 4 帧给视觉模型看（`realism_check` 角色），发现字幕/水印就插「上一版出了字幕」用更硬的提示词重生成一次，仍有就在结果里标「仍有字幕/文字」交人复核（`drama.subtitle_gate` / `subtitle_retries`）。检查做不了（没本地副本、没 ffmpeg、输出不可解析）只记备注不拦生成 | `domain/media/no_text.py`、`domain/functions/media.py`、`domain/functions/drama.py`、`config/media_models.yaml` |
| 模型够不着用户电脑上的文件：素材、剧本 txt、剪辑软件导出的片段都要人手动搬进资产库 | 本地文件系统工具 `fs_*`（12 个）：`fs_roots` 看边界、`fs_list`/`fs_info`/`fs_read`（UTF-8/GBK 自动识别、大文件分段）/`fs_search`（关键词或正则）只读；`fs_write`（新建/覆盖/追加，覆盖前备份）/`fs_mkdir`/`fs_move`/`fs_copy`/`fs_delete`（移到 `workspace/trash/`，从不真删）可回滚故为 L-write；`fs_import` 把本地文件登记成资产（媒体不复制、文本入库）、`fs_export` 把资产写成本地文件。边界在 `config/filesystem.yaml`：`roots` 白名单（项目/workspace/产物目录永远在）+ `deny`（.env、*.pem、.venv、.git、系统目录永远不碰）。阻塞 I/O 全在线程里跑 | `domain/functions/files.py`、`config/filesystem.yaml`、`app.py` |
| 模型看不见图片和视频的内容：渲出来的片段有没有变脸、用户素材是什么、参考图对不对，全得人来看 | `view_image`（一张图 → 视觉模型描述主体/人物/场景/光线/画面文字/生成瑕疵，或答问题；大图 ffmpeg 缩到 1600 宽再传）、`view_video`（按时间均匀抽帧带时间戳 → 按时间顺序描述发生了什么、人物是否一致、字幕/文字、穿帮；默认 8 帧最多 24，`transcribe=true` 抽音轨走 `transcribe` 一起给模型）。来源统一接受本地路径（同文件系统边界）/ 资产 id / 链接；角色 `vision`（kimi_k3）在 models.yaml | `domain/functions/vision.py`、`config/models.yaml`、`app.py` |
| 抖音短视频分支按用户定的路径重做：关键词 + 确认风格 → **RPA 抓抖音/小红书热点** → 分析归纳成新内容 → 据此定做法与时长 → 需要实拍的镜头 **① 向用户要 ② 联网找可用素材** → 出片 | 全部做成聊天里可调的工具 + skill `douyin-short`：`list_video_styles`（配方 = 风格，新增 `talking-head` 口播出镜、`story-mix` 情绪混剪，配方加 `style` 段定时长范围与画面来源）；热点走已有 RPA 工具（`douyin_hot_rpa` / `xhs_collect` / `browse_and_copy`，接口 `douyin_hot_list` 兜底）；`short_video_brief`（角色 `short_video_planner`：交叉比对热点归纳新内容，事实标出处，时长在风格范围内按内容密度定并按口播字数校正，分镜标 generate/real）；`request_materials`（挂起问人，/auto 也停）+ `resolve_materials`（读回复：路径→登记素材，"生成"→改标记，"联网找"→给搜索词；目录名含"生成/联网"也先按路径认）；`stock_media_search` / `fetch_stock_media`（只接可商用的 Pexels / Pixabay，授权信息随资产保存）+ `fetch_media_url`（授权 unknown 提醒）+ `web_fetch`；`short_video_produce`（素材直用、图片做成静止镜头、缺的镜头并发生成 + 网络重试 + 重跑复用、配音自动挑音色、字幕原稿回贴、按风格快切合成）。配方 `realism: auto` 走全局真实感档位 | `domain/pipeline/short_video.py`、`domain/functions/short_video.py`、`domain/functions/materials.py`、`domain/pipeline/recipe.py`、`config/recipes/*.yaml`、`skills/douyin-short/SKILL.md`、`config/models.yaml` |
| 一轮跑起来（渲视频十几分钟）终端只剩进度窗，想说句话只能 Ctrl+C | 输入枢纽 `InputHub`：stdin 由一根常驻线程读，所有问人的提示（权限确认 / 金额护栏 / 人审决策 / 产物目录）统一走它。一轮在跑时打字 → **排队**，本轮结束后按序自动发送；`/stop`（/pause）立即停掉当前一轮；`/now <消息>` 停掉当前并把这条插到队首立即发送；`/queue` 看排队、`/queue clear` 清空。正在问权限时打 `/stop` 也能停。停掉的一轮里没执行完的调用由 Loop 在下一轮开始时补记为「已中断」（原有 `_repair_interrupted_calls`），已提交给远端的生成任务取不回来 | `interfaces/cli/inputhub.py`、`interfaces/cli/main.py` |
| 一集的新定义（用户 2026-09-18）：**一集成片 4 分钟；短视频平台要高节奏，开场 15 秒必须给高潮点** | 规格集中在 `config/drama.yaml`（`EpisodeFormat`：4 分钟、±15%、320 字/分钟、开场 15s、单段 10–15s），三步共用、每步有确定性检查，不合格让模型改一次再存：① 写剧本：字数目标 ≈1280（960–1664），标题下一行「> ⚡ 前15秒高潮点：…」，写作子代理契约带规格，写完 `check_script` 太短/缺高潮点行就带问题扩写一次；② 拆分镜：每集 16–24 镜行、前两镜行首标「【高潮点】」，`check_storyboard` 不过让模型改一次；③ 提示词：单段 10–15s、一集 204–276s、开场 15s 内的段 `"hook": true`，`check_shots` 不过改一次；`drama_shots` 加 `note`；结果里打「✓ 规格达标」或「⚠ 规格检查未过」列出问题。`drama_write` 的 minutes 默认按规格 | `domain/drama/format.py`、`config/drama.yaml`、`domain/functions/drama.py`、`domain/functions/episodes.py`、`domain/drama/prompts.py`、`domain/drama/writing.py`、`skills/drama-script/SKILL.md` |
| 参考图传了人物照样漂（用户 2026-09-18）：三个原因叠加 —— 参考图只"传了"没在提示词里点名；生成接口返回的图片链接约 24h 失效，之后渲染表面「参考 N 图」实际零参考；没有任何生成后校验 | 三道：① 提示词开头加**参考锁定段**，@图片N 逐张点名"这是谁、要一致什么"（`domain/drama/identity.py::reference_block`）；参考图同时按 `image` 和 `image_urls` 两个字段送（`ref_fields.image_alias`），防止字段名不对造成零参考；② **人物一致性门**（LOOP）：每段视频/每张带参考的人物图生成后，把参考图（本地副本 data URL，不依赖会过期的链接）和抽帧一起给视觉模型逐人打 0–10 分，低于 `identity_pass_score` 就把差异写进提示词重生成（`identity_retries`），都没过留分最高的并在结果里标「仍与参考图不符（人物漂移）」；③ 渲染前检查参考图链接年龄，超过 `ref_url_ttl_hours` 提示先 `drama_refresh_refs`：用本地副本让生图模型逐像素复刻出新链接（不换脸），生成新的参考图包 | `domain/drama/identity.py`、`domain/functions/drama.py`、`domain/functions/media.py`、`domain/generators/catalog.py`、`config/media_models.yaml` |
| 用户的素材一定在本地，但生成接口只收公网链接（seedance 原话：Only http/https URL or asset:// private asset URL），本地路径、base64 都不行；生成结果自带的链接约 24h 失效 | 本地素材托管 `config/hosting.yaml`：`imghost`（**没有对象存储时用这个** —— preset 选站：imgbb / sm.ms 免费注册拿一个 key，catbox / litterbox 连注册都不用；litterbox 72h 自动删，过期渲染前会自动重传）/ `command`（跑用户自己的上传命令，取 stdout 最后一行链接）/ `s3`（OSS / COS / R2 / MinIO，纯 httpx 的 SigV4 PUT，密钥走 .env）/ `http_put`（自建）四选一，默认 none。参考素材不是 http/https 时 `gen_image` / `gen_video` **提交前就拒**，并指路 `host_file`，不让人白等一段视频的时间。`Hosting.ensure_asset` 统一入口：链接新鲜直接用，否则拿本地副本上传并写回资产（`gen_params.hosted`，托管链接不按 24h 过期）。接入点：`host_file` / `host_asset` / `hosting_status` 工具；短剧 `drama_use_local_ref` 把本地图放进参考图包（主形象/服装/场景/道具，之后 `drama_render_assets(reuse)` 沿用不再生成、渲视频自动引用并做一致性校验）；渲染前 `_rehost_pack` 自动把过期/本地的参考图重新上传；`drama_refresh_refs` 优先重新上传（图不变），上传不了才退回复刻 | `domain/media/hosting.py`、`domain/functions/hosting.py`、`domain/functions/drama.py`、`config/hosting.yaml`、`app.py` |
| 代理抖一下整轮白做（用户 2026-09-19 实测）：`APIConnectionError` 重试 3 次共等 1.5 秒就放弃，然后 `StopReason.ERROR` 把整轮判死 —— 6 次迭代、¥1 的活（加载 skill、读资产、写好的集）全部要重来 | 分两层扛：① **网关**：连接类错误单独给更深的重试预算（`connect_max_attempts` 6 次，指数退避 + `max_delay_ms` 封顶 + `jitter` 抖动，约 16 秒容忍），状态码类照旧 3 次；`classify_model_error` 新增 `connection` 类别。② **循环**：`LoopResult` 带 `error_kind` 与 `resumable`，网络类失败由 CLI 在**同一轮**里等 20/40/60 秒自动 `continue_turn` 重试 3 次（本轮进度不丢，`/stop` 可叫停）；仍失败就停下并提示 `/retry` —— 新命令，修好网络后在同一轮接着跑，不用重新发话 | `harness/model/config.py`、`harness/model/gateway.py`、`harness/execution/loop.py`、`interfaces/cli/main.py`、`config/models.yaml` |
| 视频"没有并发"（用户 2026-09-19：19 段跑了 2.3 小时、零重叠） | 并发本来只存在于 `drama_render_shots` 里；那一轮模型没走短剧链，而是**自己一次一个地调 `gen_video`**，一次工具调用就是一个迭代，必然串行（图片同理，20 次也是串行）。把并发下沉到媒体层：新增 `gen_videos` / `gen_images` 批量工具，一次提交多个任务、按 `concurrency` 上限并发、单个失败不拖垮整批、返回顺序与入参一致可直接喂 `compose_video`；单个版的说明改成"两个以上一律用批量版"。配套修预算：`ToolSpec.cost_units_arg` 让闸门按数组长度计次（否则 20 段只记 1 次，护栏形同虚设），`CostGuard.check(units=)` 要求一次性放得下。另外把短剧链里音色锚点镜头之间的相互依赖去掉，层数不再随角色数增长，最多"锚点层 + 其余全并发" | `domain/functions/media.py`、`harness/tools/provider.py`、`harness/permission/gate.py`、`harness/model/budget.py`、`domain/functions/drama.py` |

| 模型看不见用户电脑上已经生成好的素材（用户 2026-09-20 实测：`E:\内容测试` 里 texts/ 有第 6–37 集剧本、videos/ 有 ep1–ep10 分段、exports/ 有 1–10 集成片，模型却答「剧本只写到第 5 集，6–10 集没有剧本」，让用户去导出） | 两层根因、三处修法。① **资产库认集号太死**：只认 type=script + gen_params.episode；模型自己 `save_draft(kind="outline")` 存的整集、合规修订版（`revise` 还会把 gen_params 丢掉）全不算数。现在 `AssetStore.episode_of` 对文本类资产按摘要里的「第N集」认集；`script_of` type=script 优先，退到摘要带集号、够长度、不是目录/报告/文案的大纲/文本资产；`episodes_done` / `find(episode=)` / 项目卡 / `find_episode` / `drama_write_episodes` 的"已有跳过"全部跟着变；`revise` 带上 gen_params。② **产物目录从没被看过**：新增 `domain/local_materials.py`（只扫产物目录 + `config/filesystem.yaml` 的 `material_dirs`，按文件名认集号 第6集 / ep6_ / EP06 / S1E06、类型、文件名里的 as_ id；ttl + 目录 mtime 签名缓存，上限 4000 个文件，265 个文件约 80ms），每轮把「产物目录在哪、各子目录多少文件、覆盖哪些集」pin 进 pre_input；`find_episode(N)` 连这一集的本地文件（相对路径，可直接给 fs_read / view_image / view_video）一起列，资产库没记录也不再说"没有"；新工具 `find_materials(episode / kind / query / folder)`；系统提示词加一段「用户说已有素材先看本地」。③ **docx 读不了**：`fs_read` / `fs_search` / `fs_import` 认 docx / pptx / xlsx（纯标准库抽文字，`domain/documents.py`），.doc / .pdf 明确提示另存为 docx | `domain/assets/store.py`、`domain/local_materials.py`、`domain/documents.py`、`domain/functions/content.py`、`domain/functions/files.py`、`domain/drama/card.py`、`app.py`、`harness/context/assembler.py`、`config/filesystem.yaml`、`tests/test_local_materials.py` |

| **每一个镜头必须控制在 3 秒以内**（用户 2026-09-20 定的硬性要求，不管一集多长）；之前的规格是「单镜 10–15s = 一段生成视频」，一集只有 16–24 个镜头，一个镜头停十几秒 | 单镜上限进 `config/drama.yaml` `cut.max_seconds`（3），四层硬约束：① 写剧本：规则里加「动作写成可切的短拍」；② 拆分镜：每个镜头行的方括号最后一项写时长 `[近景/推入/平视/2s]`，`check_storyboard` 查没标时长 / 超 3 秒 / 短于 1 秒 / 时长加起来不在 204–276s / 前 15 秒（≈前 5 镜）没高潮点，一集 ≥68 个镜头行（一般 80–120）；③ 提示词：一段 10–15s 是多镜头快切，每段带 `cuts` 字段（各镜头秒数、每个 ≤3、加起来 = video_duration），`check_shots` 查没给 cuts / 有超 3s / 加起来对不上 / video_name 覆盖的镜头数不够（10s 至少 4 镜、15s 至少 5 镜）；②③不合格照旧让模型改一次；④ 渲染：每段提示词最外层包「多镜头快切·硬性」+ 按 cuts 排的时间线（中英双语，`domain/media/fast_cut.py`），生成后**镜头门**用 ffmpeg 场景切换检测（`ffmpeg.scene_cuts`，纯本地不花模型钱）量出最长镜头，超过 3s + 0.5s 容差就用更硬的提示词重生成一次，还超就标「仍有超过 3 秒的镜头」交人复核（`drama.cut_gate` / `cut_retries` / `cut_scene_threshold` / `cut_tolerance`）。配方短视频链的快切上限 `Recipe.cut_max` 也不会超过全局值（talking-head 写的 4 按 3 切）。真实 ffmpeg 验证：红蓝绿三段硬切 7s 片子量出切换点 2.0/4.0、最长 3s；单色 7s 一镜到底量出 7s。**老的 1–10 集分镜/提示词是旧规格**，要按 3 秒切得重跑 drama_storyboard → drama_shots → 渲染 | `config/drama.yaml`、`config/media_models.yaml`、`domain/drama/format.py`、`domain/drama/models.py`、`domain/drama/parse.py`、`domain/drama/prompts.py`、`domain/media/fast_cut.py`、`domain/media/ffmpeg.py`、`domain/functions/drama.py`、`domain/pipeline/recipe.py`、`skills/drama-script/SKILL.md`、`tests/test_cut_rule.py` |
| 视频提示词的写法一直靠自己摸索；社区已有从 463 条人工核对的 Seedance 案例里蒸馏出的 25 个模板（[LearnPrompt/awesome-seedance](https://github.com/LearnPrompt/awesome-seedance)，CC BY 4.0），用户要本地化并融进 Agent（2026-09-23） | 两层：① **skill `seedance-prompting`**（三级披露，主文档 4.5K 字 + 7 篇中文参考：6 个类别各一篇 + 短剧最常用 6 类模板的高热案例提示词原文），由 `scripts/build_seedance_skill.py` 从 `skills/seedance-prompting/data/`（上游两份模板 JSON 合并 + overrides，案例每模板只留热度前 5、原文截 1800 字，`SOURCE.md` 记出处与许可）**生成**，上游更新重跑即可；主文档写明和本流水线怎么对接 —— 参考锁定 / 快切时间线 / 音色 / 真实感都由渲染层包，模板只管 description 怎么写。② **短剧链直接吸收**：`identity.reference_block` 补「不继承」声明（参考图只取长相和穿着，不搬白背景、站姿、正面构图、原始光线 —— 案例库的结论是锁不锁得住差别就在这句）；`prompts.SHOTS_SEEDANCE_RULES` 进 `drama_shots` 系统提示词：一拍一人说话、≤3s 镜头台词 ≤8 词、反应写因果链不写表情清单、情绪写行为不写症状、每段收结束状态、台词不放品牌名数字、机位术语中英混写、结尾硬切不淡出、反转写成画面 | `skills/seedance-prompting/`、`scripts/build_seedance_skill.py`、`domain/drama/identity.py`、`domain/drama/prompts.py`、`tests/test_identity_gate.py`、`tests/test_drama.py` |
| 抖音短视频只有三种风格配方（资讯快切 / 口播出镜 / 情绪混剪），案例库里最富的两类写法没接进来（2026-09-23 用户要的） | 两份新配方，都从 awesome-seedance 模板派生、文件头注明来源与许可：**`ugc-vlog` 手持vlog**（← handheld-ugc-vlog，92 条案例）—— 真实感是用「毛病」买来的：`shots.style` 就是相机缺陷清单（指名 iPhone 随手拍、手抖、对焦来回找、曝光呼吸、no cinematic emulation、无稳定器、不推大特写），`prompt_hint` 要求台词焊在动作里、每句 ≤8 词、身体状态单向递进，footage mixed、15–45s、接热榜选「一件小事」；**`product-ad` 产品广告**（← product-commercial-shotlist，27 条）—— `shots.style` 先写死广告美学词（premium / luxury advertising aesthetic、anamorphic、体积光），`prompt_hint` 要求每个微距指名拍什么、一拍一种物理效果、收尾留干净英雄帧给后期上字、避开印刷标签、产品图当 @图片1 身份锁；footage generate、8–20s、不抓热榜（`grounding.enabled: false`）、`realism: ""`（无人不挂皮肤真实感）、口播与字幕默认关、video_tier quality。两份是两个极端，测试钉住「各自的味道只出现在各自的 prompt 里」（混写 = 塑料感）。`list_video_styles` 自动列出，douyin-short skill 的风格清单同步加上 | `config/recipes/ugc-vlog.yaml`、`config/recipes/product-ad.yaml`、`skills/douyin-short/SKILL.md`、`tests/test_recipes_seedance.py` |
| Agent 只认两条产线（短视频 / 短剧），seedance 模板库进来后其实还能做广告和海报，用户要一个「标签」自己选这次做哪类内容（2026-09-23） | **产线标签**：`domain/lines.py` 定四条 —— 短剧 / 抖音短视频 / 广告 / 设计，每条带 skill 预筛用的 content_type（短视频 / 图像）和一段路由指引（该走哪些工具、哪些 skill、先做什么）。选定后三件事：`CapabilityAllocator.content_type` 按它筛 skill 目录（选设计时短剧/抖音/seedance 三个 skill 不进模型视野）、指引以 `LINE_PIN` pin 进 pre_input（模型不再问「做哪种内容」）、写进会话快照 `content_line` 重启沿用。CLI：开工时列菜单选一次（回车 = 不限定）、`/type`（别名 `/line`）随时切换、`/type off` 回到不限定、面板与 `/stat` 显示当前产线、命令菜单可搜；系统提示词的产线段从两条扩成四条并注明「有当前产线就别再问」。**设计产线的两块基础一起补上**：① 海报渲染器从 E:igc-agent 移植（`domain/media/poster.py` + `functions/poster.py`，`make_poster`：生图出不带字的底图 → 本地叠标题/要点/品牌名，三模板 clean/bold/card、五种比例、自动 AIGC 角标、长标题缩字换行、找不到中文字体不静默；Pillow 进核心依赖）；② `skills/poster-prompt-handbook.md` 从占位改成正式版（两步原则、比例与用途表、底图 prompt 六块、风格锁定、文字区预留、检查清单、禁止项）。`agent new` 的引导入口仍是两条，聊天里的产线标签是主入口 | `domain/lines.py`、`app.py`、`capabilities/memory/session.py`、`harness/context/assembler.py`、`interfaces/cli/main.py`、`interfaces/cli/promptbox.py`、`domain/media/poster.py`、`domain/functions/poster.py`、`skills/poster-prompt-handbook.md`、`tests/test_content_lines.py`、`tests/test_poster.py` |

| 引用失败的镜头还是生成了、成片后半段人物全变脸（用户 2026-09-20 实测第 11 集）。真实日志：20 段里 4 段渲染失败 —— 两段是 5 张参考图按 `image` + `image_urls` 双送 = 10 个，撞上 seedance "at most 9 reference images" 的 HTTP 400；一段内容策略拦截；一段轮询超时。模型随后把这 4 段改写成英文提示词、用 gen_videos **不带任何参考图**补生成，再拼成"完整版" | 用户定的规则：**没有引用成功的镜头不许生成，要在生成前拦住**。三道：① `drama_render_shots` 花钱之前的**引用门**（只查这次要生成的段）：引用了包里没有的名字、描述里出现的角色（台词里提到的不算）没带参考图、有资产库却没渲参考图、链接超过 TTL、人物参考图超过模型参考位、`ref_probe` 对每个参考图链接发 HEAD 发现 4xx —— 任一不过**整批不发起**，报出每段该修什么（`preflight_refs` / `_ref_gate_error`）；网络不通只提示不拦。② 媒体层：目录给视频模型加 `max_refs`（seedance 9），`gen_video` 提交前算 图 + 视频 + 音频 超上限直接拒；双送会超上限时只送实测有效的 `image`；渲染时人物必保、位子不够先省道具再省场景。**绕过拦截**：`MediaFunctions.ref_guard`（装配层挂 `DramaFunctions.reference_guard`）—— 提示词/摘要带集号镜号或提到资产库里的角色/服装、又没带参考图的 gen_video / gen_videos 一律拦下并指回 drama_render_shots；无人物空镜传 `allow_no_refs=true` 才放行。③ 某几段失败**不成片**，说清缺哪几段、重跑 `drama_render_shots(reuse=true)` 只补失败的段。系统提示词与 skill 也写明。用真实第 11 集数据演练：引用门 0 问题、11 个参考图链接全部可访问；那两段 5 张图的段现在会以 6 个参考提交（不再 400） | `domain/functions/drama.py`、`domain/functions/media.py`、`domain/generators/catalog.py`、`config/media_models.yaml`、`app.py`、`harness/context/assembler.py`、`skills/drama-script/SKILL.md`、`tests/test_ref_gate.py` |

| 视频生成悄悄换了模型（用户 2026-09-20：切换视频模型之前一定要与用户沟通，换到不想用的模型质量和体验都很差）。真实日志：短剧用的是用户指定的 seedance-2.0，模型补段时 gen_videos 的 model 留空、prefer=quality，自动选型换成 veo3.1-quality，第 101/102 轮共 12 段都是它生成的，用户没同意过 | **视频模型锁**，规则落在媒体层不靠模型自觉：`MediaFunctions.video_lock`（会话快照记住的 > `media_models.yaml` drama.video_model 用户指定的 > 第一次用的模型）；`gen_video` / `gen_videos` 的 model 留空一律用锁定的，不再按 prefer 自动选型；传了不同的模型 → 工具返回 suspend（环节「视频模型切换」，major=true，/auto 也停）挂起主循环问用户「要把视频模型从 A 换成 B 吗？a 同意 / r、j 不换」，一段都不生成；人采纳后媒体层收总线 CHECKPOINT_DECIDED 事件才换锁并写回会话快照（`SessionSnapshot.video_model`，重启沿用），模型再调一次即可；打回不换。短剧链 `DramaFunctions.video_model` 跟着锁走（用户同意换了整条链一起换）；批量生成任一项要换模型整批先问。启动面板显示当前锁定的视频模型；系统提示词写明不要为省钱/绕过失败悄悄换模型。主循环级测试：挂起 → 采纳 → 换锁 → 再调才生成；打回 → 留空继续用锁定的；/auto 下照样停 | `domain/functions/media.py`、`domain/functions/drama.py`、`capabilities/memory/session.py`、`app.py`、`interfaces/cli/main.py`、`harness/context/assembler.py`、`tests/test_video_model_lock.py` |

`/stat` 里能看到上下文的校准估算、真实 token 与折叠情况。

**产物文件命名（2026-09-18）**：图片/视频不再叫 `as_xxx`，生成时就按顺序号落盘，后期在剪辑软件里按名字排就是顺序：
`第01集-03_2场_镜9-18.mp4`（集-集内序号_场次_镜头范围）· `参考图-角色-01_陆离.png` / `参考图-服装-02_…`（类别-序号_名字）· 配方短视频 `主题_第03镜.mp4` · 整集 `第01集.mp4`。
重渲同一镜头加 `-v2`，不覆盖旧版。已经生成的用 `agent assets rename --session <名>`（或 chat 里 `/rename`，`/rename dry` 只看计划）按同一套规则补改，只动仍叫 `as_…` 的文件，
资产 id 与血缘不变；`agent assets manifest` 出一份「文件 ↔ 资产 id ↔ 说明」清单（改名时会自动写到产物目录的 `产物清单.md`）。规则在 `domain/media/naming.py`。

---

## 已知环境问题（Windows + 中文路径）

项目路径含中文，在 Windows 上会持续制造编码摩擦。已处理的三处：

| 问题 | 现象 | 已做的处理 |
|---|---|---|
| editable 安装失效 | `.pth` 用 UTF-8 写入，`site.py` 用 locale(cp936) 解码 → 路径解析成乱码，包 import 不到 | `.venv/Lib/site-packages/_editable_impl_aigc_agent.pth` 改成纯 ASCII 的运行时计算：`sys.prefix` 的父目录 + `src` |
| 控制台输出崩溃 | stdout 是 GBK，编不了 `✓ ✗ →` 等符号，rich 直接抛 UnicodeEncodeError | CLI 入口最早处 `sys.stdout.reconfigure(encoding="utf-8")` |
| MCP stdio 子进程 | 配置里写绝对路径换机就断；子进程继承 GBK 控制台 | `mcp_servers.yaml` 用内置变量 `${PYTHON}` / `${PROJECT_ROOT}`，相对 `cwd` 按项目根解析；server 自己把 stderr 切成 UTF-8，且**不往 stdout 打印**（那是协议通道） |

**注意**：重新执行 `pip install -e .` 会覆盖掉那个 `.pth` 补丁，需要重打。

> 资产路径落库、ffmpeg 的 subtitles 滤镜（见 `media/ffmpeg.py`）也各自绕过一次。
> **建议在项目变大前把目录换成纯 ASCII 路径**（如 `E:\aigc-agent`），一次性消掉整类问题。

---

## 目录

```
config/          模型 / MCP / 能力预算 / 机审规则 / 平台规格
  recipes/       短视频配方（改 yaml 就能调效果，不用改代码）
skills/          方法论 markdown（运营可直接编辑，存盘即生效）
src/aigc_agent/
  harness/       L0 内核 —— 不含任何 AIGC 概念
  capabilities/  L1 能力层
    skill_hub/         M7  三级披露 · 热加载 · 版本记录与回滚
    capability_budget/ M7+M20 共管的能力预算分配器
    mcp_hub/           M20 外部工具中枢
    memory/            M8  Store（倒排索引）· Agent（提取 / 召回 / Brief）· brief（四区契约）
    subagents/         M10 子代理运行时（最小能力）
    retrieval/         M9  统一检索（记忆源、工具源；资产源在 domain）
  domain/        L2 AIGC 业务
    functions/   M21 内容能力面：存稿/人审、生图/生视频、TTS/转写、成片、热点、分镜、短剧、检索、机审、打包、数据回流
    compliance/  M15 机审器（规则在 config/compliance.yaml，只出清单）
    distribution/ M16 打包器（平台规格在 config/platforms.yaml，产待发布包）
    analytics/   M17 指标存储 · 复盘 · 提炼进记忆（MetricsSource 留给平台 API）
    pipeline/    配方 + 剪辑节奏规划 + 字幕对齐 + 音色自动选择 + 图节点执行器
    storyboard/  分镜与角色护照（移植自 ai-character-passport）
    drama/       短剧三段式：分镜脚本 / 资产库 / seedance 提示词
    rpa/         浏览器采集
    media/       ffmpeg 封装（M14）
  interfaces/    L3 入口
    cli/         agent 命令
    mcp_servers/ 自带的 MCP server（独立进程，不 import 本包其余模块）
tests/
```

依赖方向严格单向向下，由 `lint-imports` 强制。

### 外部工具的移植原则

已移植三个外部工具。做法一致：**只取方法论和提示词工程，不取它的运行时**。

| 来源 | 取了什么 | 丢了什么 |
|---|---|---|
| `视频拆解/rainwell-douyin-viral-analyzer` | 归因铁律、脱敏要求 → `skills/douyin-viral-analyzer.md` | 环境相关的具体接口 |
| `剧本测试/短剧剧本创作-skill` | 创作方法论 → `skills/drama-script/`（三级披露） | — |
| `剧本分镜/ai-character-passport` | 角色护照数据模型、分镜提示词、三种输出格式 → `domain/storyboard/` | Next.js UI、zustand、IndexedDB（这边有 AssetStore）、浏览器 canvas 抽帧（这边有 ffmpeg） |
| 用户自带的短剧三段式 system prompt | 三段提示词**逐字保留** → `domain/drama/prompts.py` | — |

移植时**必须核对硬约束**，照搬会坏。例如 ai-character-passport 按即梦的规格
写死单镜 10–15 秒，而这边 veo3.1 最长 8 秒 —— 不改的话模型会按更长的信息量
编排画面，生成出来是截断的。这条有回归测试钉着。

移植 system prompt 时**不要省略输出示例**。踩过：转录短剧资产提示词时把末尾的
「标准 JSON 输出示例」换成了一句概括，模型随即不知道 `roleCostumeList` 里该放
什么，直接给空数组 —— 下一步无从引用，编出 11 个假资产 ID。示例才是定义字段名
和嵌套结构的地方，散文说明替代不了。

---

## P1 已达成的验收标准

**打回能正确回退并重跑** —— 这是 P1 存在的理由，不是「能生成一篇文案」。
两种执行模式各验一遍，**共用同一套 Asset 与记忆底座**。

```bash
./.venv/Scripts/python.exe -m pytest tests/ -v
```

### Agentic 模式（主）—— `test_p2_agentic.py`

- 模型主动调 `request_review` → **主循环真的挂起**
- 人的决策作为该次调用的返回值填回，模型据此继续
- 打回不填理由被拒，且拒绝后状态不破坏、可重来
- 改写版本挂在原版下面，**血缘不断**
- 请人审不存在的资产被挡下，提示先 `save_draft`
- 人审前后仍算**同一问一答**，滑窗计数正确

### 图模式（备）—— `test_p1_graph.py`

- `revise` 打回 → 沿回退边回 `draft` 重写，**大纲原样保留**
- `reject` 打回 → 回 `outline`，大纲一并作废重来
- 回退时**恢复快照**而非叠加状态（快照数正确截断）
- 打回**不填理由直接拒绝**（半自动模式下不记原因，第二次会重犯）
- 反复打回 3 轮仍正确；节点数护栏触发时挂起交给人，不静默降级
- 资产血缘可回溯，revise 产出新版本而非覆盖

### 记忆子系统（M8）—— `test_memory_agent.py`

- 淘汰的轮次 → 关键词提取 → 长期记忆（`MemoryAgent`，独立上下文 + 异步队列）
- 提取带 **polarity**：「别写成硬广」和「要写成硬广」存下来能分辨
- 提取会**过滤**：记忆要被反复注入上下文，噪音会挤掉真正有用的
- 从对话提的一律 `INFERRED`，**不得自动晋升到账号层**
- 召回走倒排索引，**不调模型** —— 它在每轮的快路径上
- 中文靠**反向匹配**跨过分词：拿索引里已有的词去查询串里找

**已知边界**：词法召回跨不过语义鸿沟 —— 存「时长」查「多长」召不回来。
要跨过去得上向量检索或每轮加一次模型调用，后者会让每轮都变慢，暂不做。

---

## P2 已达成的验收标准

P2 的三条验收（ARCHITECTURE.md §7）：**agentic 模式下也能看到进度图并单步重跑；
短视频链路跑通；接入至少 1 个 MCP server 且工具走两级披露。** 外加两处收尾：
Cost Guard 接线、`cache.mode` 定案。

### 执行痕迹 → DAG + 单步重跑 —— `test_p2_trace.py` · `test_p2_rollback.py`

- 按资产依赖**自动连边**，跑完得到一张 DAG（`to_mermaid()` 可直接渲染）
- `/rollback <资产id>`：以某份资产为新起点，其后的调用**只标记作废、不删除**，
  资产也不删；一段说明 pin 进下一轮的 pre_input 位，模型看一次就撤
- 配方流水线（`agent video`）直接调 `registry.invoke()` 的路径**也进痕迹**了 ——
  之前只有 Dispatcher 那条路发事件，跑完一条短视频 `/trace` 里一个节点都没有

### 多模态 function —— `test_p2_media.py` · `test_p2_audio.py` · `test_p2_compose.py`

- 图像/视频走 **异步任务模型**（提交 → 轮询 → 取回，带退避与超时）；TTS 同步二进制、
  ASR multipart —— 三种调用形状各归各的网关，都落同一套 Asset
- 模型按目录（`media_models.yaml`）**自己选型**；未知模型 id 直接报错不静默替换
- 成片链（下载 → 拼接 → 混音 → 烧字幕 → 导出）用 **真 ffmpeg + lavfi 合成的假素材**
  跑通：时长对齐、音轨不丢、血缘挂全、快切镜头不超上限

### MCP Hub 接入第一个真实 server —— `test_p2_mcp.py` · `test_p2_mcp_live.py`

- 自带的 `material_lib`（本地素材库，stdio，`interfaces/mcp_servers/`）通过 `config/mcp_servers.yaml`
  白名单接入；`${PYTHON}` / `${PROJECT_ROOT}` 内置变量，换机不用改绝对路径
- 走**官方 SDK 真客户端**（mcp 2.x）：命名空间 `material_lib__xxx`、描述包裹标注来源、
  默认 L-external + 只读工具显式降级、删除保持 L-external 且**无人可问时拒绝**
- 两级披露：外部工具默认只上目录（标 `*`），`load_tool_schema` 展开后才进请求体
- 中文文件名在 stdio 上往返正确；越界路径被 server 拒绝并把原因透传回来

### Cost Guard 接线 —— `test_p2_cost_guard.py`

**之前 `budget.py` 写好了但没接任何地方** —— 没有一处 import 它，防线等于不存在。
现在两套口径同时拦：

- **次数**：媒体调用（生图 / 生视频 / TTS / 转写）在 **权限闸门** 处计次，超限先问人，
  人点头**只放行这一次**（加临时额度，不改配置）；无人可问就拒。次数口径不依赖单价
- **金额**：文本花费从 COST 事件按参考价累计；`media_models.yaml` 里填了 `price` 的媒体
  调用也进账并写进资产的 `gen_cost`。累计超过 `per_task_limit` 时 Loop **挂起交给人**
- Kimi Code 端点是订阅额度制，`models.yaml` 里的 pricing 是 platform.kimi.com 的**参考价**，
  用来估算额度消耗，不是账单

### `cache.mode` 定案

`agent doctor --cache` 实测：同一段 3,598 token 前缀连发两次，第二次 **100% 命中** ——
Kimi 是自动前缀缓存，M3 的批量驱逐（涨到 15 轮一次性剔回 10 轮）设计直接生效。
`/stat` 里能看到累计命中率。

---

## P3 已达成的验收标准

P3 的四条验收（ARCHITECTURE.md §7）：**节点启动前能拿到 Memory Brief；打回理由不再重犯；
运营改 markdown 即可影响输出；能力区占用受控在 30K 内。**

### M10 子代理运行时 —— `test_p3_subagent.py`

最小能力：单个子代理拉起 + 独立上下文 + 结构化回传。并行扇出留在 P5。

- `SubAgentDef` 定契约：工具子集、允许的副作用等级、输出 schema、角色名、是否无状态
- 注册表的**裁剪视图**（`ToolRegistry.scoped()`）：子代理只看得到、只调得了被允许的工具
- **L-external 永远不给** —— 子代理背后没有人，不可逆动作没人确认；也不能请人审
- 输出按 schema 校验：顶层类型、必填字段、字段类型；坏输出回 `ok=False` 带原文
- 默认无状态：每次都是新上下文。M8.2 的 `memory_consolidate` 是第一个用例

### Memory Brief —— `test_p3_memory_brief.py`

- **规则版不调模型**：四区（must / must_not / should / refs）+ conflicts 靠 layer / source /
  polarity / category 就能判；每轮对话前、每个节点启动前都跑
- **避雷不做相关性过滤** —— 主题无关也带上打回理由，漏一条就是重犯一次
- **推测不进硬约束**：`inferred` 只能进 should 并带「（推测）」标记；晋升到账号层只能由人
  （`agent memory promote`）或数据回流触发，模型不能
- **矛盾暴露不裁决**：同一个词既有要求又有禁止 → 写进 conflicts 交给人；
  同一句话被提取成两种极性不算矛盾，近似重复只留一条
- **消费点**：图模式进节点契约 + pin 在节点内 Loop 的 pre_input 位；agentic 模式每轮
  `prepare_turn()` 把 must / must_not pin 进主循环 —— 打回理由不再靠关键词碰运气召回
- 完整模式 `consolidate()` 走子代理做去重合并、标冲突；**模型删了避雷不采信**，
  坏输出退回规则版

### Skill Hub + 能力预算 —— `test_p3_capability_budget.py`

**之前 skill 目录从没进过上下文**（`catalog_digest` 没有任何调用方），模型不知道有哪些方法论
可加载；正文作为工具返回值进对话历史，会随滑窗被剔，预算超了也没人能卸。现在：

- 目录 pin 进系统区常驻；正文由分配器 pin 进系统区，`load_skill` 只返回一句确认
- **热加载**：目录指纹（路径 + mtime + 大小）变了就重新解析，下一轮生效；已激活的正文跟着换
- **版本记录 + 一键回滚**：每个版本按内容哈希存进 `workspace/skills_history/`，
  `agent skill history / rollback`；`agent skill check` 体检
- **两 Hub 共管一个池子**（`config/capability_budget.yaml`）：skill 目录 + 工具目录 +
  内置 schema + 激活正文 + 展开的外部 schema 五项合计；超预算按
  `drop_lowest_priority_skill → collapse_expanded_tools → shrink_skill_digest` 降级，
  合规类（priority ≥ 100）永不挤出；达 90% 告警
- 真实装配下能力区约 **7K / 30K**；`/skills` 与 `/stat` 里能看

### 统一检索 —— `test_p3_retrieval.py`

- `search_library(query, kind)` 一个入口扇出到：历史内容（Asset）、记忆（品牌资料 / 打回理由）、
  素材库（`material_lib` MCP 工具，走同一个注册表和闸门）
- 每条命中带来源与**版权状态**（generated / human / unknown…）；素材库默认 unknown，
  function 描述里明写「unknown 的不要直接进成片」
- 单个来源挂了不影响其余；多模态查询（以图搜图、按情绪搜 BGM）留了 kind 口子，等索引再接

## P4 已达成的验收标准

P4 的验收（ARCHITECTURE.md §7）：**完整闭环；数据回流能提炼进账号层记忆。**
两个开放问题按默认值定案：发布**只产待发布包**由人上传（§8 #3）；数据**先人工录入**，
`MetricsSource` 留给平台 API（M17 原话）。

### 合规机审 M15 —— `test_p4_compliance.py`

- 规则是数据：`config/compliance.yaml` 分六类（广告法极限词 / 医疗功效 / 金融承诺 /
  敏感话题 / 诱导互动 / 数字缺出处）+ AIGC 显式标识 + 素材版权 + 账号层禁忌，运营改 yaml 存盘即生效
- **只出风险清单 + 定位 + 建议改法，不改一个字**：block 拦打包、warn 交人判断，每条带行号与片段
- AIGC 标识两件套：显式标识在正文里查（缺了 block），隐式标识由发布包 manifest 写入
- 报告落成 `report` 资产，血缘挂在被审资产与所用素材下
- `skills/compliance-redlines.md` 从占位转为生效（priority 100，永不被预算挤出）：机器版在 yaml，判断原则在 skill

### 待发布包 M16 —— `test_p4_distribution.py`

- `config/platforms.yaml` 平台规格（起步值，以平台当前规则为准）；打包器不接平台 API，没有 `publish` function
- 一个包 = `workspace/releases/<平台>_<标题>_<时间>/`：`content.md` 正文原样、`upload.md` 上传清单、
  `manifest.json`（平台、血缘、机审结果、AIGC 隐式标识、发布状态）
- 格式适配的边界：话题超上限保留前 N 个并注明；**标题超长、正文超长、类型不支持列为问题不截断**；
  机审有 block 拒绝打包，除非人给 `override_reason` 并记进 manifest
- 人上传后 `mark_published` 记回链接，`agent release list/show/published`
- 图模式：`copy.yaml` 的 `finalize` 节点用 `$draft` 引用槽位调 `build_release_package`，
  工具自己落的资产直接接到出槽位，不再包一层

### 数据回流 M17 —— `test_p4_analytics.py`

- `record_metrics` 人工录入（JSONL 追加写，一个包可多次采集，复盘取最新）
- 复盘按账号中位数倍率判强弱：播放 ≥1.5× 且完播不低于中位数为强、播放 ≤0.5× 且完播不高于中位数为弱，样本 <3 不判定
- **提炼进账号层记忆**：强作品 → DATA 来源、positive 偏好；弱作品 → neutral「慎重复制」；
  同一个包只提炼一次。DATA 来源可以进账号层而 INFERRED 不行，这是「晋升」硬规则的另一面
- 完整闭环有测试钉着：存稿 → 机审 → 打包 → 记回链接 → 录 3 条数据 → 账号层出现 DATA 记忆 →
  `build_brief().should` 里能看到 → `search_library` 查得到

## P5 已达成的验收标准

P5 的验收（ARCHITECTURE.md §7）：**并行提速、成本可控、可复盘。**

### 并行扇出 M10 —— `test_p5_fanout.py` · `test_p5_graph_parallel.py`

- `SubAgentRunner.run_many`：同一定义、多个任务并发，信号量限并发；结果按序回传，单个失败不影响其余。
  5 个候选的耗时接近 1 个，不是 5 个之和
- **汇总保留差异点**：每个候选独有的片段（不分词、不调模型），供人对比，不替人挑
- `fan_out_candidates`：一次给 2–8 个角度，每版落资产、挂血缘，返回 id 与差异点，下一步 `request_review`
- 图运行时补上 `parallel` / `join`：分支之间并发、汇聚后继续；分支里不能有 human / parallel / terminal。
  **回退按图依赖作废**：打回正文只重跑那条分支，配图分支保留（旧的整体恢复快照会把兄弟分支一起抹掉）

### 项目级 / 日级预算 —— `test_p5_cost_ledger.py`

- `CostLedger`：JSONL 台账，每次模型调用、每次媒体生成记一行，按天、按项目、按类别求和，跨会话累计
- 三级金额（单任务 / 单项目 / 单日）+ 两级媒体次数（单任务 / 单日），都在闸门前拦；没单价时次数照样拦
- 超限仍是挂起问人，人点头只放行这一次；`/budget` 与 `agent doctor` 能看到今日与项目累计

### 可观测 M6 —— `test_p5_observability.py`

- `EventLog`：事件流按会话落盘 JSONL，大字段截断、流式 delta 不落、写失败不上抛（可观测不能成为主链路的故障点）
- `agent sessions list / replay / cost / trace`：同一份文件三种看法。痕迹重建把事件重新喂给 `ExecutionTrace`，
  得到的 DAG 与在线一致；成本看板按角色、媒体、工具算钱与耗时

## 下一步

六个阶段都完成了。后面按需要长，不再按阶段：

- **运营侧待填**：`config/compliance.yaml` 违禁词按平台最新规则补全；`config/platforms.yaml` 上限核对；
  `media_models.yaml` 的 `price` 填上金额口径才对媒体调用生效；`skills/compliance-redlines.md` 的 owner 指定
- **Web 人审台**（TECH_STACK 的 P2/P3 入口演进）：FastAPI 推事件流 + React Flow 图视图，CLI 里的决策逻辑不用改
- **图文 / 海报两张图**（M11 骨架已支持 parallel）、平台 API 拉数据（M17 的 `MetricsSource`）、
  向量召回（M8.1 已留口子，等词法召回成为瓶颈）
- **迁到纯 ASCII 路径**：中文路径的编码摩擦已经绕过四次，项目再大就该一次性消掉

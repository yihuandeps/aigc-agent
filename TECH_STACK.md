# 技术栈决策

> 结论：**Python 3.11+ 后端，CLI 先行，Web 人审台后置。**

---

## 1. 为什么是 Python

决定性因素是**多模态处理**。这个项目要覆盖短视频（剪辑/TTS/字幕/转码/封面）、图像（生成/精修/抠图合成）、音频（配音/BGM 匹配），M13 + M14 两个模块的工作量占了领域层的一大半。

| 维度 | Python | TypeScript |
|---|---|---|
| 媒体处理生态 | **压倒性优势**：Pillow / librosa / 帧分析 / 音画对齐 | 基本只能 shell out |
| 生成模型 SDK | 各家 Python SDK 最全最新 | 常滞后或缺失 |
| MCP SDK | 官方支持 | 官方支持（打平） |
| Kimi 接入 | OpenAI 兼容，`AsyncOpenAI` 直接用 | 同样可用（打平） |
| Schema 定义 | pydantic v2，够用 | zod，略强 |
| Web 人审台 | 要独立前端 | **同栈优势** |
| 并发（工具并发/节点扇出） | asyncio + TaskGroup | 原生 |

TS 唯一的实质优势在人审台同栈。但**人审台是 P1 后期才需要的东西**，而且 M18 本来就规划了 CLI / Web / IM 三种入口——后端只提供 API，前端独立，这不是妥协，是原本就该有的分层。

**会推翻这个结论的情况只有一个**：你或团队主力是 TS、Python 只是勉强能写。那时 TS 的熟练度收益大于 Python 的生态收益，告诉我，我出 TS 版方案。

---

## 2. 选型清单

### 核心（P0 就要）

| 用途 | 选择 | 说明 |
|---|---|---|
| 语言 | **Python 3.11+** | 3.11 起 `asyncio.TaskGroup` 和异常组好用，并发代码干净很多 |
| 模型调用 | **`openai` (AsyncOpenAI)** | Kimi 兼容 OpenAI 协议，直接指 `base_url`。省掉自写流式解析和重试 |
| Schema | **pydantic v2** | 这个项目 schema 极多（Asset / GraphNode / Memory / Brief / ToolProvider），是刚需不是可选 |
| 配置 | **PyYAML + pydantic-settings** | yaml 读配置和图定义，pydantic 做校验，`.env` 注入密钥 |
| 并发 | **asyncio**（标准库） | 单轮内工具并发、`parallel` 节点扇出、压缩路径异步 |
| 日志/事件 | **structlog** | M6 要结构化事件流，不是文本日志 |
| CLI | **typer + rich** | M18 的调试入口。rich 渲染流式输出和图状态很省事 |

### P1 加

| 用途 | 选择 | 说明 |
|---|---|---|
| 持久化 | **SQLite（`aiosqlite`）** | Memory 项目层、GraphState 快照、事件流。单机内部工具，够用 |
| MCP | **`mcp`（官方 Python SDK）** | M20，stdio + SSE 两种传输都覆盖。⚠️ 装的是 **2.x**：`FastMCP` 改名 `MCPServer`（`mcp.server.mcpserver`），字段改成 snake_case（`input_schema` / `is_error`），预期内的工具错误要抛 `ToolError` 才能把原文透传给客户端。客户端两种字段名都认 |

### P2 加

| 用途 | 选择 | 说明 |
|---|---|---|
| 媒体处理 | **ffmpeg**（外部二进制）+ subprocess 封装 | M14。不用 moviepy——它对长视频内存占用不可控 |
| 图像处理 | **Pillow** | 精修、抠图合成、封面裁剪 |
| Web API | **FastAPI** | 人审台后端，SSE 推事件流 |

### P3+ 视情况

| 用途 | 选择 | 触发条件 |
|---|---|---|
| 前端 | **React + Vite + React Flow** | 做图可视化时。React Flow 是画 Graph 最成熟的库 |
| 向量检索 | 到时再选 | **仅当关键词倒排的召回准确率成为瓶颈**。见架构 M8.1 |
| 音频分析 | librosa | 做 BGM 节奏匹配时 |

---

## 3. ⚠️ 明确不引入的东西

这几个看着相关，但会伤害这个项目：

| 不用 | 原因 |
|---|---|
| **LangChain / LangGraph** | 我们是在手搓 harness。引入它等于把 M1/M3/M4 的设计权交出去，而这三个模块的设计恰恰是本项目的核心资产。它的 Graph 抽象也和我们定的节点/边语义（回退边、human 节点、快照）对不齐，适配成本高于自写 |
| **Celery / RQ** | asyncio 的后台任务足够。压缩路径就是个 `asyncio.Queue` + 一个消费协程 |
| **Postgres / Redis** | 单机内部工具，SQLite + 内存队列够跑很久。等真有多人并发再换，那时数据量也清楚了 |
| **向量库（P3 前）** | 架构 M8.1 已定：起步用关键词倒排索引，够用再上向量 |
| **moviepy** | 长视频内存占用不可控。直接封 ffmpeg 命令行，可控且快 |

**原则：先用标准库和最小依赖跑通，等瓶颈真出现再引入。** 这个项目的复杂度在设计不在库。

---

## 4. 项目布局

`src` layout，包名 `aigc_agent`，目录与 `ARCHITECTURE.md` §5 一一对应。

```
E:\手搓Agent\
├── pyproject.toml
├── .env / .env.example / .gitignore
├── ARCHITECTURE.md
├── TECH_STACK.md
│
├── config/                      # 配置（已建）
│   ├── models.yaml
│   ├── mcp_servers.yaml
│   └── capability_budget.yaml
│
├── skills/                      # 方法论 markdown（已建，运营可直接改）
│
├── src/aigc_agent/
│   ├── harness/                 # L0 内核 —— 不含任何 AIGC 概念
│   │   ├── execution/           # M1
│   │   │   ├── router.py
│   │   │   ├── loop.py
│   │   │   └── graph/           # runtime / state / snapshot / guards
│   │   ├── model/               # M2  providers/ + roles 解析
│   │   ├── context/             # M3  window_policy / budget / assembler
│   │   ├── tools/               # M4  registry / provider.py / dispatcher
│   │   ├── permission/          # M5
│   │   └── events/              # M6
│   │
│   ├── capabilities/            # L1
│   │   ├── skill_hub/           # M7
│   │   ├── mcp_hub/             # M20
│   │   ├── capability_budget/   # M7+M20 共管
│   │   ├── memory/              # M8
│   │   │   ├── store/           #   M8.1
│   │   │   └── agent/           #   M8.2
│   │   ├── retrieval/           # M9
│   │   └── subagents/           # M10
│   │
│   ├── domain/                  # L2 —— AIGC 业务
│   │   ├── pipeline/
│   │   │   ├── graphs/          # M11 图定义（yaml，非代码）
│   │   │   └── subgraphs/
│   │   ├── assets/              # M12
│   │   ├── generators/          # M13
│   │   ├── media/               # M14
│   │   ├── compliance/          # M15
│   │   ├── distribution/        # M16
│   │   └── analytics/           # M17
│   │
│   └── interfaces/              # L3
│       ├── cli/                 # M18  P0 起就有
│       └── api/                 # M18  P2 起，FastAPI
│
├── workspace/                   # 运行时产物（gitignored）
│   ├── assets/
│   ├── projects/
│   └── logs/
│
└── tests/
```

---

## 5. 入口演进

不要一开始就背上前端负担：

| 阶段 | 入口 | 能做什么 |
|---|---|---|
| **P0–P1** | CLI（typer + rich） | 跑 Loop、跑图、在终端里看节点状态、打回重跑 |
| **P2** | + FastAPI + 极简 Web 页 | 人审台：候选并排对比、diff、填打回理由 |
| **P3+** | + React Flow 图视图 | 可视化图运行状态、点节点回退（M19 的核心体验） |

**P1 的验收（打回能否正确回退重跑）在 CLI 里就能完成。** 别为了做界面推迟核心验证。

---

## 6. 几条编码约定

- **全异步**：所有 I/O 走 `async`。Loop 内工具并发、`parallel` 节点扇出、后台压缩都依赖它，混用同步会很痛。
- **schema 集中定义**：`Asset` / `GraphNode` / `GraphEdge` / `GraphState` / `Memory` / `Keyword` / `MemoryBrief` / `Checkpoint` 全部 pydantic 模型，放在各模块的 `models.py`，跨层引用只引用模型不引用实现。
- **配置即数据**：图定义、Skill、MCP server 都是 yaml/markdown，不是 Python。改流程不用改代码。
- **L0 不 import L1/L2**：加一条 CI 检查（`import-linter` 或自写脚本），这是架构里唯一的硬约束，靠自觉守不住。
- **角色而非模型 id**：调模型时传 `role="memory_extract"`，不传模型名。见 `config/models.yaml`。

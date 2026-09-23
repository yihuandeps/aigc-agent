"""装配层 —— 把 L0 各模块接成一个可用的 Agent。

刻意放在包根而不是 harness 里：harness 各模块之间只通过构造函数依赖，
谁都不知道完整的装配长什么样。装配是应用的事，不是内核的事。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from .capabilities.capability_budget import BudgetConfig, CapabilityAllocator
from .capabilities.memory.agent import MemoryAgent
from .capabilities.memory.recorder import RejectionRecorder
from .capabilities.memory.session import SessionSnapshot
from .capabilities.memory.store import MemoryStore
from .capabilities.retrieval import MemorySource, RetrievalHub, ToolSource
from .capabilities.skill_hub import SkillHub
from .capabilities.skill_hub.functions import SkillFunctions
from .capabilities.subagents import SubAgentRunner
from .domain.analytics import Feedback, MetricsStore, Reviewer
from .domain.assets.store import AssetStatus, AssetStore
from .domain.compliance import ComplianceChecker, ComplianceRules
from .domain.distribution import Packager, PlatformCatalog
from .domain.drama.card import CARD_PIN, build_project_card
from .domain.drama.format import EpisodeFormat
from .domain.functions.analytics import AnalyticsFunctions
from .domain.functions.audio import AudioFunctions
from .domain.functions.compliance import ComplianceFunctions
from .domain.functions.content import ContentFunctions
from .domain.functions.distribution import DistributionFunctions
from .domain.functions.douyin import DouyinFunctions
from .domain.functions.drama import DramaFunctions
from .domain.functions.episodes import EpisodeFunctions
from .domain.functions.fanout import FanOutFunctions
from .domain.functions.files import FileFunctions, FsPolicy
from .domain.functions.hosting import HostingFunctions
from .domain.functions.hotspot import HotspotFunctions
from .domain.functions.materials import MaterialFunctions
from .domain.functions.media import MediaFunctions
from .domain.functions.poster import PosterFunctions
from .domain.functions.retrieval import AssetSource, RetrievalFunctions
from .domain.functions.rpa import RpaFunctions
from .domain.functions.short_video import ShortVideoFunctions
from .domain.functions.storyboard import StoryboardFunctions
from .domain.functions.video_edit import VideoEditFunctions
from .domain.functions.vision import VisionFunctions
from .domain.generators.catalog import MediaCatalog
from .domain.lines import LINE_PIN, get_line, line_block, line_providers
from .domain.local_materials import LOCAL_PIN, LocalMaterials
from .domain.media.hosting import Hosting, HostingConfig
from .domain.output import OutputPrefs, default_root
from .domain.pipeline.episode_pipeline import EpisodePipeline
from .domain.project import normalize_root, project_key, project_title
from .domain.system_prompt import MAJOR_STAGES, MINOR_STAGE, SYSTEM_PROMPT
from .envdetect import PROJECT_ROOT, detect, workspace_root
from .harness.context.assembler import ContextAssembler
from .harness.context.window import ShortTermMemory, WindowPolicy
from .harness.events.bus import EventBus, EventType
from .harness.events.log import EventLog
from .harness.execution.loop import LoopResult, LoopRuntime
from .harness.execution.trace import ExecutionTrace
from .harness.model.audio import (
    ApiMartAudioProvider,
    AudioGateway,
    MiniMaxSpeechProvider,
)
from .harness.model.budget import CostGuard
from .harness.model.config import ModelsConfig
from .harness.model.gateway import ModelGateway
from .harness.model.ledger import CostLedger
from .harness.model.media import ApiMartProvider, MediaGateway
from .harness.model.task_ledger import MediaTaskLedger
from .harness.permission.gate import Asker, PermissionGate
from .harness.tools.builtin import builtin
from .harness.tools.disclosure import DisclosureProvider
from .harness.tools.dispatcher import ToolDispatcher
from .harness.tools.registry import ToolRegistry

ROLLBACK_PIN = "rollback"
BRIEF_PIN = "memory_brief"
MATERIAL_SEARCH_TOOL = "material_lib__search_materials"


def load_dotenv(path: Path | None = None) -> list[str]:
    """加载 .env 并自动适配本机环境（代理等）。见 envdetect。"""
    return detect().dotenv_loaded


class Agent:
    """主对话 Loop + 全部 function Provider + MCP Hub 的装配体。

    装配点都在 create() 里，其余不动。
    """

    def __init__(
        self,
        bus: EventBus,
        config: ModelsConfig,
        gateway: ModelGateway,
        registry: ToolRegistry,
        dispatcher: ToolDispatcher,
        assembler: ContextAssembler,
        memory: ShortTermMemory,
        loop: LoopRuntime,
        trace: ExecutionTrace,
        assets: AssetStore,
        memories: MemoryStore,
        skills: SkillHub,
        catalog: MediaCatalog | None = None,
        mem_agent: MemoryAgent | None = None,
        guard: CostGuard | None = None,
        *,
        allocator: CapabilityAllocator | None = None,
        subagents: SubAgentRunner | None = None,
        retrieval: RetrievalHub | None = None,
    ) -> None:
        self.bus = bus
        self.config = config
        self.gateway = gateway
        self.registry = registry
        self.dispatcher = dispatcher
        self.assembler = assembler
        self.memory = memory
        self.loop = loop
        self.trace = trace
        self.assets = assets
        self.memories = memories
        self.mem_agent = mem_agent
        self.skills = skills
        # 媒体目录挂出来：按内容自动选音色要拿到当前 TTS provider 的音色表
        self.catalog = catalog
        self.guard = guard
        self.allocator = allocator  # M7+M20 共管的能力预算
        self.subagents = subagents  # M10 子代理运行器
        self.retrieval = retrieval  # M9 统一检索
        self.media_gw: Any = None
        self.audio_gw: Any = None
        self.checker: Any = None  # M15 机审
        self.packager: Any = None  # M16 打包
        self.metrics: Any = None  # M17 数据
        self.ledger: Any = None  # 成本台账（跨会话）
        self.session_store: Any = None  # 会话现场快照（create() 里装）
        self.output_prefs: Any = None  # 产物目录偏好（create() 里装）
        self.local_materials: Any = None  # 产物目录 / 素材目录索引（create() 里装，每轮 pin 摘要）
        self.media_fns: Any = None  # 媒体工具（视频模型锁在它身上；create() 里装）
        self.pipeline: Any = None  # 按集流水管线（create() 里装，/auto on 打开）
        self.content_line: str = ""  # 用户选的产线标签（/type；create() 里从会话快照恢复）
        self.workspace: Path | None = None  # 运行时产物目录（create() 里定）
        self.config_dir: Path = PROJECT_ROOT / "config"
        self.mcp = None  # type: ignore[assignment]  # setup() 里按需装配
        # 当前项目键（产物目录派生，见 domain/project.py）。资产库查询、台账、记忆共用它
        self.project: str = ""
        self.recorder: Any = None  # 打回理由落库（项目维度跟着换）
        self.memory_source: Any = None  # 统一检索里的记忆来源（项目维度跟着换）

    @classmethod
    def create(
        cls,
        config_dir: Path | None = None,
        asker: Asker | None = None,
        session_id: str = "",
        role: str = "main_agent",
        auto_approve: bool = False,
        workspace: Path | None = None,
    ) -> Agent:
        """auto_approve=True 时自动放行 L-external。

        workspace：运行时产物目录（资产库/台账/记忆/日志）。不传就用 envdetect.workspace_root()
        —— 环境变量 AIGC_WORKSPACE 或项目下的 workspace/；测试进程必须指到临时目录。

        **只给脚本/批处理用**，交互场景永远传 asker 让人确认。
        没有 asker 又不自动放行时，L-external 会被拒绝 —— 这是对的
        （不可逆动作不该静默执行），但脚本里会表现为「工具全挂」，
        所以需要这个显式开关，而不是让人去猜为什么调不通。
        """
        load_dotenv()
        config_dir = config_dir or PROJECT_ROOT / "config"
        config = ModelsConfig.load(config_dir / "models.yaml")

        workspace = Path(workspace) if workspace else workspace_root()
        # 产物目录 = 项目（2026-09-23 审查缺口 A）。2026-09-22 用户定：默认就是**他打开 Agent
        # 的那个文件夹**；会话记住的（/out 改过的）由 CLI 开工时核对。在项目目录里启动时
        # 回落到 workspace/output/<session>。
        out_root = default_root(workspace, session_id or "", PROJECT_ROOT)
        if not session_id:
            # 没指定会话名：一个文件夹一个会话。之前一律 default —— 换任何文件夹启动都装回
            # default 的轮次、模型锁和产物目录（当时是 E:\西游记）
            session_id = _session_for(out_root, workspace)
        project = project_key(out_root)
        bus = EventBus(session_id=session_id)
        # M6：事件流按会话落盘 —— 会话回放、成本看板、痕迹重建都从这份文件来
        EventLog(workspace / "logs" / "sessions", bus.session_id).attach(bus)
        # 执行痕迹：订阅事件总线自动建 DAG，不侵入 Loop
        trace = ExecutionTrace()
        trace.attach(bus)
        gateway = ModelGateway(config, bus)

        # Cost Guard：文本花费从 COST 事件记，媒体调用在权限闸门处计次。
        # 不接这两根线，budget.py 就只是个文件 —— P2 收尾前它正是这个状态。
        # 台账让项目级 / 日级预算跨会话累计（P5）。
        # 单项目上限按**项目键**累计（之前是会话名 default —— 一个永不清零的终身上限）
        ledger = CostLedger(workspace / "costs" / "ledger.jsonl")
        guard = CostGuard.from_config(
            config.cost_guard,
            ledger=ledger,
            project_id=project,
            session_id=bus.session_id,
        )
        guard.attach(bus)

        assets = AssetStore(workspace / "assets")
        assets.project = project  # 新资产打上项目键；查询默认只看本项目
        assets.bind_loop()  # 线程里建的资产（fs_import）也能把落库事件投回主循环
        memories = MemoryStore(workspace / "memory")

        # 产物目录：生成内容（文本/图片/视频）落盘的用户文件夹，/out 随时改（= 换项目）
        output_prefs = OutputPrefs(out_root)
        assets.mirror = output_prefs  # 文本类资产自动镜像一份可读 .md

        # M7：方法论。history_dir 记版本，运营改坏了能一键回滚
        skills = SkillHub(PROJECT_ROOT / "skills", history_dir=workspace / "skills_history")
        skills.load()

        catalog = MediaCatalog.load(config_dir / "media_models.yaml")
        media_gw, audio_gw = _media_gateways(config, catalog, bus, workspace=workspace)

        registry = ToolRegistry(bus)
        memory = ShortTermMemory(policy=WindowPolicy())

        # M7 + M20 共管的能力预算：skill 目录与正文挂在主循环的系统区 pin 上，
        # 展开的外部工具 schema 由注册表管，分配器决定超了卸谁。
        # 接它之前 skill 目录**根本没进过上下文** —— 模型不知道有哪些方法论可加载。
        allocator = CapabilityAllocator(
            BudgetConfig.load(config_dir / "capability_budget.yaml"), registry, skills, memory, bus
        )
        allocator.attach(bus)

        registry.register(builtin)
        registry.register(DisclosureProvider(registry))  # load_tool_schema 元工具
        registry.register(SkillFunctions(skills, on_load=allocator.activate_skill))  # M7
        # 本地素材索引（2026-09-20）：产物目录 + filesystem.yaml material_dirs 里有什么、按集归组。
        # find_episode / find_materials 查它，每轮摘要 pin 进 pre_input —— 用户目录里摆着 6–37 集
        # 剧本、模型却说「只写到第 5 集」，就是因为它从没看过磁盘。
        fs_policy = FsPolicy.load(config_dir / "filesystem.yaml")
        local = LocalMaterials(output_prefs, extra_dirs=fs_policy.material_dirs)
        registry.register(ContentFunctions(assets, local=local))  # M21：存稿 / 请人审
        # 本地文件系统：看/读/写/搬用户电脑上的文件，边界在 config/filesystem.yaml
        files = FileFunctions(
            assets,
            workspace,
            output_prefs,
            fs_policy,
            project_root=PROJECT_ROOT,
            local=local,
        )
        registry.register(files)
        # 本地素材托管（config/hosting.yaml）：本地图/视频上传到用户自己的存储拿公网链接，
        # 生成接口才能拿它当参考；过期的生成链接也靠它用本地副本刷新
        hosting = Hosting(HostingConfig.load(config_dir / "hosting.yaml"))
        registry.register(HostingFunctions(assets, hosting, files))
        # 看图 / 看视频：图片直接给视觉模型，视频抽帧（可选转写音轨）；路径边界同上
        registry.register(VisionFunctions(gateway, assets, files, registry))
        # 抖音分支（2026-09-18）：关键词+风格 → RPA 热点 → 归纳成新内容 → 素材（问人 / 联网）→ 出片
        registry.register(MaterialFunctions(assets, workspace))
        short_video = ShortVideoFunctions(
            gateway, assets, registry, catalog, bus=bus, output=output_prefs, files=files,
            hosting=hosting,
        )
        registry.register(short_video)
        # 生图 / 生视频（prefs = 产物目录，生成成功自动落本地副本）
        media_fns = MediaFunctions(media_gw, catalog, assets, prefs=output_prefs)
        registry.register(media_fns)
        # 出片档位跟会话锁对账：锁着的模型才是真正出片的那个（确认单按它报）
        short_video.video_lock_source = lambda: media_fns.video_lock
        registry.register(AudioFunctions(audio_gw, catalog, assets))  # TTS / 转写
        registry.register(DouyinFunctions(assets, workspace))  # 抖音素材
        registry.register(VideoEditFunctions(assets, workspace, prefs=output_prefs))  # M14
        # 海报 / 图文卡片：本地叠字，不花钱（2026-09-23 从 E:igc-agent 移植，设计产线用）
        registry.register(PosterFunctions(assets))
        # 热点：API 优先（抖音），RPA 兜底（小红书 API 需付费）
        registry.register(HotspotFunctions(assets, os.environ.get('TIKHUB_API_KEY', '')))
        registry.register(RpaFunctions(assets, workspace))
        # 分镜与角色护照（移植自 ai-character-passport）：
        # 拆分镜靠文本模型，所以要把文本网关传进去
        registry.register(StoryboardFunctions(gateway, assets, workspace))
        # 短剧三段式流水线：剧本 → 分镜脚本 / 资产库 → seedance 视频提示词
        # 一集的规格（4 分钟 / 开场 15 秒高潮点）：写、拆、提示词三步共用，见 config/drama.yaml
        episode_fmt = EpisodeFormat.load(config_dir / "drama.yaml")
        drama = DramaFunctions(
            gateway, assets, registry, catalog, bus=bus, fmt=episode_fmt, hosting=hosting
        )
        drama.files = files  # drama_use_local_ref 登记本地图走同一套文件边界
        registry.register(drama)
        # 参考图门（2026-09-20 用户定的规则）：短剧镜头没带参考图不许直接用 gen_video 生成 ——
        # 真实事故：4 段渲染失败后模型自己改写用 gen_videos 无参考补生成，成片后半段人物全变脸
        media_fns.ref_guard = drama.reference_guard
        # 模型锁（用户定的规则：换模型之前必须先问用户）：短剧链跟着锁走，
        # 用户同意换了就整条链一起换（视频 2026-09-20 定、生图 2026-09-22 定）
        drama.video_lock_source = lambda: media_fns.video_lock
        drama.image_lock_source = lambda: media_fns.image_lock
        # 按集流水并行管线：/auto on 时剧本每满一批就自动接续分镜→提示词→渲染。
        # 吃 ASSET_CREATED 事件，默认关闭，CLI /auto on 打开（drama 链的手动模式不变）。
        assets.bus = bus
        pipeline = EpisodePipeline(
            registry, assets, bus, output_prefs, guard=guard, spec=episode_fmt.stamp
        )
        pipeline.attach()
        # M9：一个入口查历史内容 / 记忆 / 素材库。素材库来源走注册表里的 MCP 工具，
        # server 没连上就当没有这个来源
        memory_source = MemorySource(memories, project_id=project)
        retrieval = RetrievalHub(
            [
                AssetSource(assets),
                memory_source,
                ToolSource(registry, MATERIAL_SEARCH_TOOL, name="material_lib"),
            ]
        )
        registry.register(RetrievalFunctions(retrieval))
        # P4 闭环：机审（只出清单）→ 待发布包（人上传）→ 数据回流（提炼进账号层记忆）
        rules = ComplianceRules.load(config_dir / "compliance.yaml")
        checker = ComplianceChecker(rules, memories)
        packager = Packager(
            workspace / "releases", PlatformCatalog.load(config_dir / "platforms.yaml"), assets
        )
        metrics = MetricsStore(workspace / "analytics")
        registry.register(ComplianceFunctions(checker, assets, bus))
        registry.register(DistributionFunctions(packager, checker, assets, bus))
        registry.register(
            AnalyticsFunctions(
                metrics, Reviewer(metrics), Feedback(memories), packager, assets, bus
            )
        )

        if asker is None and auto_approve:

            async def _auto(meta: Any, args: dict[str, Any]) -> bool:  # noqa: ARG001
                return True

            asker = _auto
        # 闸门同时挂权限策略与预算护栏：L-compute 是"预算内放行，超限询问"
        gate = PermissionGate(bus, asker=asker, guard=guard)
        registry.gate = gate  # 公开 invoke() 从此过闸门
        dispatcher = ToolDispatcher(registry, gate, bus)

        # M10：子代理运行器。M8.2 的完整模式是它的第一个用例；
        # fan_out_candidates 用它并行出候选（P5）
        subagents = SubAgentRunner(gateway, registry, bus, guard=guard)
        registry.register(FanOutFunctions(subagents, assets))
        # 逐集写剧本的子代理：写作从主循环搬出来，主 Agent 只收 id 和结尾钩子
        episode_fns = EpisodeFunctions(subagents, assets, bus, fmt=episode_fmt)
        registry.register(episode_fns)
        # M8.2 记忆代理：淘汰的轮次 → 关键词提取 → 长期记忆。
        # 独立上下文 + 异步执行，主循环只入队不等它。
        # 记忆按项目键归档（缺口 A）：之前按会话名，而 CLI 不带 --session 时会话名是空串 ——
        # 库里 48 条记忆全是全局的，蜘蛛精剧那句「控制在十二集」对所有剧都生效
        mem_agent = MemoryAgent(gateway, memories, project_id=project, runner=subagents)
        # 逐集写作的子代理也要守 Memory Brief（打回理由之前对它写的剧本不生效）
        episode_fns.brief_source = lambda topic: mem_agent.brief(topic=topic).pin_text()
        # 打回理由自动落库（M8），同样按项目
        recorder = RejectionRecorder(memories, project_id=project)
        recorder.attach(bus)

        # 滑窗淘汰的轮次转成长期记忆。不挂这个钩子的话，
        # 超出 10 轮的对话就是**直接丢弃** —— 长期记忆永远写不进去。
        memory.on_evict = lambda turns: mem_agent.submit(
            "\n\n".join(t.transcript for t in turns),
            origin_ref=f"turns:{turns[0].index}-{turns[-1].index}",
        )

        # 会话现场快照：同名 session 重启后接着上次聊。资产与长期记忆本来就
        # 落盘，会丢的只有滑窗原文。每个 LOOP_END 存一次，读写失败都不打断对话。
        # （订阅放在 loop 建好之后：挂起中的人审要一起存）
        session_store = SessionSnapshot(workspace / "memory" / "sessions", session_id or "default")

        # 模型锁：会话记住的 > 短剧配置里用户指定的（media_models.yaml drama.video_model /
        # image_model）。留空的 gen_video / gen_image 一律用它；要换的请求挂起问人，
        # 人采纳（总线 CHECKPOINT_DECIDED）才换锁，换了写回会话快照，重启沿用。
        media_fns.video_lock = session_store.video_model or str(
            (catalog.drama or {}).get("video_model") or ""
        )
        media_fns.on_video_lock = session_store.set_video_model
        media_fns.image_lock = session_store.image_model or str(
            (catalog.drama or {}).get("image_model") or ""
        )
        media_fns.on_image_lock = session_store.set_image_model
        bus.subscribe(media_fns.on_event)

        # 上下文预算 = 模型窗口 − 输出额度 − 余量。校准系数 1.4 是实测值
        # （Kimi 对「JSON 里的中文」低估四成），之后每次调用按真实 usage 自动修正。
        provider, _ = config.text.resolve(role)
        token_budget = max(0, provider.context_window - provider.max_output_tokens - 8_192)
        assembler = ContextAssembler(
            bus, system_prompt=SYSTEM_PROMPT, token_budget=token_budget, calibration=1.4
        )
        loop = LoopRuntime(
            gateway=gateway,
            registry=registry,
            dispatcher=dispatcher,
            assembler=assembler,
            memory=memory,
            bus=bus,
            role=role,
            guard=guard,
            major_stages=MAJOR_STAGES,
            minor_stage=MINOR_STAGE,
            budget_hint=(
                "要继续：/budget allow 50 临时追加 50 元（数字可改）；"
                "/budget set 金额 300 视频秒 900 改本次开工的额度；/budget 看各级用量。"
                "（/budget reset 只清零本次开工的用量，台账里的单日 / 单项目累计不清）"
            ),
        )

        def _snapshot(ev: Any) -> None:
            if ev.type is EventType.LOOP_END:
                session_store.save(
                    memory,
                    pending_review=loop.pending_review,
                    active_skills=list(getattr(allocator, "active", []) or []),
                )

        bus.subscribe(_snapshot)
        agent = cls(
            bus, config, gateway, registry, dispatcher, assembler, memory, loop, trace,
            assets, memories, skills, catalog, mem_agent, guard,
            allocator=allocator, subagents=subagents, retrieval=retrieval,
        )
        agent.media_gw, agent.audio_gw = media_gw, audio_gw
        agent.checker, agent.packager, agent.metrics = checker, packager, metrics
        agent.ledger = ledger
        agent.session_store = session_store
        agent.set_content_line(session_store.content_line, persist=False)
        agent.output_prefs = output_prefs
        agent.local_materials = local
        agent.media_fns = media_fns
        agent.pipeline = pipeline
        agent.workspace = workspace
        agent.config_dir = config_dir
        agent.project = project
        agent.recorder = recorder
        agent.memory_source = memory_source
        return agent

    # ---------- 项目（缺口 A：产物目录 = 项目） ----------

    @property
    def project_title(self) -> str:
        """给人看的项目名：剧名（.drama-state.json）优先，否则文件夹名。"""
        return project_title(getattr(self.output_prefs, "root", None))

    def switch_project(self, root: Path | str) -> str:
        """换产物目录 = 换项目：资产库查询范围、台账的单项目累计、记忆一起换。返回新项目键。

        流水线还有活在跑时不换 —— 它们产出的资产会被打上新项目的键（抛 RuntimeError，
        CLI 报给人：等跑完或 /auto off 之后再换）。
        """
        root = Path(root)
        busy = getattr(self.pipeline, "busy", False)
        if busy and normalize_root(root) != normalize_root(self.output_prefs.root):
            raise RuntimeError("流水线还有任务在跑：等它们跑完（或 /auto off）再换产物目录")
        key = project_key(root)
        if self.output_prefs is not None:
            self.output_prefs.root = root
        self.project = key
        self.assets.project = key
        if self.guard is not None:
            self.guard.project_id = key
        for holder in (self.mem_agent, self.recorder, self.memory_source):
            if holder is not None:
                holder.project_id = key
        if self.pipeline is not None:
            self.pipeline.reset()
        if self.session_store is not None:
            self.session_store.set_output_dir(str(root))
        return key

    async def setup(self, mcp: bool = True) -> None:
        """连 MCP server 并注册进同一个工具注册表，然后重建目录。

        MCP 工具不走独立通路 —— 权限、成本记账、日志都和内置工具共用一套。
        """
        if mcp:
            from .capabilities.mcp_hub.hub import McpHub

            path = self.config_dir / "mcp_servers.yaml"
            if path.exists():
                self.mcp = McpHub.from_file(path, self.bus)
                await self.mcp.connect_all()
                self.mcp.register_into(self.registry)
        await self.registry.refresh()
        if self.allocator is not None:
            # 把 skill 目录挂进系统区并核一次预算
            await self.allocator.refresh()

    async def aclose(self) -> None:
        """关掉所有 HTTP 连接。

        不关的话，进程退出时 httpx 的异步生成器被 GC 强行关闭，
        会刷一屏 "generator didn't stop after athrow()" 堆栈 ——
        无害但会把真正的输出淹掉，排查时很烦。
        """
        # 先停流水线：它的后台任务正在用 gateway 跑渲染/生成，
        # 不关就直接 cancel 在 await 半路的调用，堆栈同上。
        if self.pipeline is not None:
            try:
                await self.pipeline.aclose()
            except Exception:  # noqa: BLE001
                pass
        # 记忆队列跑完再关网关：对话结尾往往正是用户给明确要求的地方（「下次别这么写」），
        # 之前退出时不等，最后几轮的记忆直接丢（2026-09-23 审查）
        if self.mem_agent is not None:
            try:
                await asyncio.wait_for(self.mem_agent.close(), timeout=60)
            except Exception:  # noqa: BLE001
                pass
        for gw in (
            getattr(self, "gateway", None),
            getattr(self, "media_gw", None),
            getattr(self, "audio_gw", None),
        ):
            if gw is not None:
                try:
                    await gw.close()
                except Exception:  # noqa: BLE001
                    pass
        if self.mcp is not None:
            try:
                await self.mcp.close_all()
            except Exception:  # noqa: BLE001
                pass

    # ---------- 会话恢复（2026-09-23 审查后补的三样） ----------

    async def restore_session(self) -> int:
        """装回上次的轮次；没结案的人审挂回去；上次加载过的 skill 重新激活。返回装回几轮。

        之前只装轮次：挂起的人审只在内存里，重启后被悄悄补成「已中断」；已加载的 skill
        没了，历史里却写着「正文已常驻系统区」，模型以为方法论还在。
        """
        store = self.session_store
        if store is None:
            return 0
        n = store.load_into(self.memory)
        pr = store.pending_review
        if pr and any(t.index == pr.get("turn_index") for t in self.memory.turns):
            self.loop.pending_review = dict(pr)
        if self.allocator is not None:
            for name in store.active_skills:
                skill = self.skills.get(name)
                if skill is None:
                    continue  # 某篇 skill 被删了 / 改名了，跳过
                try:
                    await self.allocator.activate_skill(skill)
                except Exception:  # noqa: BLE001
                    continue
        return n

    # ---------- 开工额度（用户规则：每次开工前确认上限 —— 金额 / 次数 / 视频秒数） ----------

    def budget_defaults(self) -> dict[str, Any]:
        """这次开工拿来问的默认额度：上次确认过的优先，其次配置里的。"""
        base: dict[str, Any] = dict(self.guard.limits()) if self.guard is not None else {}
        saved = getattr(self.session_store, "budget", None) or {}
        base.update({k: v for k, v in saved.items() if v is not None})
        return base

    def apply_budget(self, limits: dict[str, Any]) -> None:
        """应用人确认过的额度，并记进会话快照（下次开工当默认值）。"""
        if self.guard is None:
            return
        self.guard.set_limits(
            money=limits.get("money"),
            video_calls=limits.get("video_calls"),
            video_seconds=limits.get("video_seconds"),
            image_calls=limits.get("image_calls"),
        )
        if self.session_store is not None:
            self.session_store.set_budget(self.guard.limits())

    @property
    def media_priced(self) -> bool:
        """媒体目录有没有填单价（没有 = 金额上限看不见视频，只能靠段数 / 秒数拦）。"""
        return bool(getattr(self.catalog, "priced", False))

    async def prepare_turn(self, user_input: str = "") -> None:
        """每轮开始前的两件事：

        1. skill 热加载 + 能力预算核算（运营改完 markdown 下一轮就生效）
        2. Memory Brief 的 must / must_not pin 进 pre_input 位 —— 打回理由不靠关键词
           碰运气召回，而是常驻在注意力最强的位置。这就是「不再重犯」的实现。
        """
        if self.allocator is not None:
            await self.allocator.refresh()
        self._pin_line()
        if self.mem_agent is not None:
            text = self.mem_agent.brief(topic=user_input).pin_text()
            if text:
                self.memory.pin(BRIEF_PIN, text, position="pre_input")
            else:
                self.memory.unpin(BRIEF_PIN)
        # 3. 短剧项目卡：剧名/集数/写到哪/各产物最新 id，从资产库算出来，
        #    约 500 token。关键词记忆对「继续」这种输入召不回任何东西，
        #    这些结构化事实直接摆在眼前才可靠。
        root = getattr(self.output_prefs, "root", None) if self.output_prefs else None
        card = build_project_card(self.assets, root)
        if card:
            self.memory.pin(CARD_PIN, card, position="pre_input")
        else:
            self.memory.unpin(CARD_PIN)
        # 4. 本地素材摘要：产物目录 / 素材目录里有什么、覆盖哪些集。目录扫描是磁盘 IO，丢线程。
        summary = ""
        if self.local_materials is not None:
            try:
                summary = await asyncio.to_thread(self.local_materials.summary)
            except OSError:
                summary = ""
        if summary:
            self.memory.pin(LOCAL_PIN, summary, position="pre_input")
        else:
            self.memory.unpin(LOCAL_PIN)

    # ---------- 产线标签（2026-09-23 用户要的：自己选这次做哪类内容） ----------

    def set_content_line(self, key: str, persist: bool = True) -> None:
        """选定 / 清除产线。三件事：skill 预筛的 content_type、路由指引 pin、会话快照。"""
        line = get_line(key)
        self.content_line = line.key if line else ""
        # 工具也按产线收窄：别的产线的工具只上目录、要用时 load_tool_schema 展开
        registry = getattr(self, "registry", None)
        if registry is not None and hasattr(registry, "set_focus"):
            registry.set_focus(line_providers(line) if line else None)
        if self.allocator is not None:
            # 直接改字段而不是 refresh(content_type=...)：refresh 对空串不覆盖，清不掉
            self.allocator.content_type = line.content_type if line else ""
        self._pin_line()
        if persist and self.session_store is not None:
            self.session_store.set_content_line(self.content_line)

    def _pin_line(self) -> None:
        line = get_line(self.content_line)
        if line:
            self.memory.pin(LINE_PIN, line_block(line), position="pre_input")
        else:
            self.memory.unpin(LINE_PIN)

    def _recall_query(self, user_input: str) -> str:
        """召回查询串。用户只说「继续」时用上一轮助手的话补上主题，
        否则倒排索引一个词都对不上 —— 实测 35 条记忆命中 0 次。"""
        q = user_input.strip()
        if len(q) >= 12 or not self.memory.turns:
            return q
        last = self.memory.turns[-1].assistant_text
        return f"{q} {last[:300]}" if last else q

    async def chat(self, user_input: str) -> LoopResult:
        # 每轮先按输入召回相关记忆。走倒排索引不调模型 ——
        # 这在快路径上，加一次模型调用会让每轮都变慢。
        recalled = self.mem_agent.recall(self._recall_query(user_input)) if self.mem_agent else ""
        await self.prepare_turn(user_input)
        try:
            return await self.loop.run_turn(user_input, recalled=recalled)
        finally:
            # 回退说明只需要模型看一次；留着它会每轮都以为刚回退过
            self.memory.unpin(ROLLBACK_PIN)

    async def rollback(self, asset_id: str) -> dict[str, Any]:
        """agentic 模式下的「单步重跑」。见 rollback_to()。"""
        return await rollback_to(
            asset_id, trace=self.trace, memory=self.memory, assets=self.assets, bus=self.bus
        )


async def rollback_to(
    asset_id: str,
    *,
    trace: ExecutionTrace,
    memory: ShortTermMemory,
    assets: AssetStore,
    bus: EventBus,
) -> dict[str, Any]:
    """agentic 模式下的「单步重跑」：选一份资产做新起点，它之后的调用作废。

    图模式的回退是"沿回退边回到某个节点"；这里没有节点，回到的是**某个资产版本**。
    三件事：
      1. 痕迹里把下游调用标成作废 —— 只标记不删除，走过的弯路复盘要用
      2. 资产也不删 —— 血缘要能回答"当时为什么走到那一步"
      3. 把一段说明 pin 进下一轮的 pre_input 位 —— 真正让模型"从这里重来"的是它，
         不是前两步。位置选 pre_input 是因为硬约束要贴着当前输入放（M3）。
    """
    plan = trace.apply_rollback(asset_id)
    if not plan.get("ok"):
        return plan

    try:
        head = assets.get(asset_id).brief()
    except KeyError:
        head = asset_id
    lost = list(plan.get("lost_assets") or [])
    # 作废的产物打状态（缺口 A）：之前只 pin 一轮提醒，下一轮起按「最新」取资产照样取到它们
    for aid in lost:
        try:
            assets.set_status(aid, AssetStatus.VOID, note=f"/rollback 回到 {asset_id}")
        except KeyError:
            pass
    note = f"已回退到 {head}，以它为新的出发点继续。"
    if lost:
        note += f"以下资产已作废，不要再引用：{', '.join(lost)}。"
    note += f"改写请用 save_draft 并填 parent_id={asset_id}，保持血缘。"
    memory.pin(ROLLBACK_PIN, note, position="pre_input")

    await bus.emit(
        EventType.TRACE_ROLLBACK,
        asset=asset_id,
        restart_from=plan.get("restart_from"),
        superseded=plan.get("supersede", []),
        lost_assets=lost,
    )
    return {**plan, "note": note}


def _session_for(out_root: Path, workspace: Path) -> str:
    """没指定会话名时的会话名：产物目录的项目键（一个文件夹一个会话）；
    回落目录（在 Agent 项目里启动）仍叫 default。"""
    if normalize_root(out_root) == normalize_root(workspace / "output" / "default"):
        return "default"
    return project_key(out_root)


def _media_gateways(
    config: ModelsConfig, catalog: MediaCatalog, bus: EventBus, workspace: Path | None = None
) -> tuple[MediaGateway, AudioGateway]:
    """图像/视频与语音的网关。

    两者调用形状不同（异步任务 vs 同步二进制/multipart），所以是两个网关，
    但共用同一份 provider 配置和同一个模型目录。

    workspace 给了就挂任务台账（media_tasks.jsonl）：提交成功就记 task_id，
    中途放弃等待的任务事后能取回，同一份请求重来时先取回、不重新付费（2026-09-23 审查）。
    """
    raw = (config.raw.get("image") or {}).get("providers") or {}
    spec = raw.get(catalog.provider) or {}
    base = spec.get("base_url", "https://api.apimart.ai/v1")
    key = spec.get("api_key", "")

    media = MediaGateway(
        {catalog.provider: ApiMartProvider(base, key, timeout=spec.get("timeout_seconds", 60))},
        bus,
        poll_interval=catalog.polling.interval,
        poll_backoff=catalog.polling.backoff,
        max_poll_interval=catalog.polling.max_interval,
        max_queue_s=catalog.polling.max_queue,
        max_polls=catalog.polling.max_polls,
        max_transient=catalog.polling.max_transient,
        submit_retries=catalog.polling.submit_retries,
        ledger=MediaTaskLedger(workspace / "media_tasks.jsonl") if workspace else None,
        # 按模态共享的并发上限：两批渲染同时跑也不会把服务商的并发打爆
        concurrency={
            "image": catalog.max_concurrency("image"),
            "video": catalog.max_concurrency("video"),
        },
    )
    providers: dict[str, Any] = {catalog.provider: ApiMartAudioProvider(base, key)}

    # TTS 单独走 MiniMax：APIMart 上没有任何中文原生 TTS，只有 OpenAI 那套
    # 英文音色库，中文口播听着假是先天的。转写仍留在原 provider（whisper）。
    # 没配 key 也照样注册 —— 让它在调用时给出"怎么配"的明确报错，
    # 好过启动时静默退回、用户以为已经换成中文音色了。
    if catalog.speech_provider == "minimax":
        mm = (config.raw.get("image") or {}).get("providers", {}).get("minimax") or {}
        providers["minimax"] = MiniMaxSpeechProvider(
            mm.get("base_url", "https://api.minimaxi.com"),
            mm.get("api_key", "") or os.environ.get("MINIMAX_API_KEY", ""),
            mm.get("group_id", "") or os.environ.get("MINIMAX_GROUP_ID", ""),
            timeout=mm.get("timeout_seconds", 120),
        )

    audio = AudioGateway(providers, bus)
    return media, audio

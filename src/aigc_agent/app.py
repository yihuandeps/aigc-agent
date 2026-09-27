"""装配层 —— 把 L0 各模块接成一个可用的 Agent。

刻意放在包根而不是 harness 里：harness 各模块之间只通过构造函数依赖，
谁都不知道完整的装配长什么样。装配是应用的事，不是内核的事。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from .capabilities.capability_budget import BudgetConfig, CapabilityAllocator
from .capabilities.memory.agent import MemoryAgent
from .capabilities.memory.recorder import RejectionRecorder
from .capabilities.memory.session import SessionLock, open_session
from .capabilities.memory.store import MemoryStore
from .capabilities.retrieval import MemorySource, RetrievalHub, ToolSource
from .capabilities.skill_hub import SkillHub
from .capabilities.skill_hub.functions import SkillFunctions
from .capabilities.subagents import SubAgentRunner
from .domain.analytics import Feedback, MetricsStore, Reviewer
from .domain.aspect import DEFAULT_ASPECT, aspect_label, orientation_of, parse_aspect
from .domain.assets.store import AssetStatus, AssetStore
from .domain.compliance import ComplianceChecker, ComplianceRules
from .domain.distribution import Packager, PlatformCatalog
from .domain.drama.card import CARD_PIN, build_project_card
from .domain.drama.format import DEFAULT_FORMAT, EpisodeFormat, plan_length
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
ASPECT_PIN = "project_aspect"  # 项目画幅（/ratio 设过才 pin）
PIPELINE_PIN = "pipeline_notices"  # 按集流水等人处理的事（停点 / 失败 / 额度刹车）
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
        # 占着这份会话的锁；second_window = 这个文件夹已经有窗口在用，本窗口另起了会话
        self.session_lock: SessionLock | None = None
        self.second_window = False
        self.output_prefs: Any = None  # 产物目录偏好（create() 里装）
        self.local_materials: Any = None  # 产物目录 / 素材目录索引（create() 里装，每轮 pin 摘要）
        self.media_fns: Any = None  # 媒体工具（视频模型锁在它身上；create() 里装）
        self.pipeline: Any = None  # 按集流水管线（create() 里装，/auto on 打开）
        self.content_line: str = ""  # 用户选的产线标签（/type；create() 里从会话快照恢复）
        # 一集的规格：base 是 config/drama.yaml 的默认，episode_fmt 是这个项目实际用的（/length）
        self.base_episode_fmt: EpisodeFormat = DEFAULT_FORMAT
        self.episode_fmt: EpisodeFormat = DEFAULT_FORMAT
        self.drama_fns: Any = None  # 短剧三段式（按项目的集长要套到它身上）
        self.episode_fns: Any = None  # 逐集写剧本（同上）
        # 视频画幅（/ratio）：aspect_ratio 是生效的值，aspect_custom = 用户给这个项目设过
        self.aspect_ratio: str = DEFAULT_ASPECT
        self.aspect_custom = False
        self.short_video_fns: Any = None  # 短视频出片（项目画幅套到它身上）
        self.material_fns: Any = None  # 素材站搜索（按项目画幅挑横竖）
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
        material_fns = MaterialFunctions(assets, workspace)
        registry.register(material_fns)
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
        # 重渲一段之前按段指纹找上次没等到的任务取回（含质检重生成的那次，2026-09-26）
        drama.task_ledger = getattr(media_gw, "ledger", None)
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
        # 两个停点（参考图看脸、第 1 集看片）的放行记录按项目键存盘，重启不用再点
        pipeline.gates_path = workspace / "pipeline_gates.json"
        # 没人可问时派发前的预算预检：按最坏情况（质检重生成）算、扣掉过了质检的段
        pipeline.render_estimate = drama.render_need
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
        # 渲参考图 / 渲一集之前整批报价问人（2026-09-26 用户定的）：有人可问时流水线不再按额度
        # 拦派发，由报价说清楚、人确认一次
        pipeline.quotes = asker is not None

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
        # 同一份会话只给一个窗口：别的窗口占着就另起「名字~2」（设置从主会话抄，/auto 不开）
        session_store, session_lock, second_window = open_session(
            workspace / "memory" / "sessions", session_id or "default"
        )

        # 模型锁（_apply_locks，agent 装好之后）：这个项目人定过的 > 按产线的默认。留空的
        # gen_video / gen_image 一律用它；要换的请求挂起问人，人采纳（总线 CHECKPOINT_DECIDED）
        # 才换锁，换了写回会话快照，重启沿用。
        bus.subscribe(media_fns.on_event)
        # 出片确认只认人的采纳（2026-09-26）：confirm=true 要人在确认单上点过头才生效
        bus.subscribe(short_video.on_event)
        # 定音：人采纳了主角的独白段，才固定成音色锚点（2026-09-26）
        bus.subscribe(drama.on_event)

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

        agent = cls(
            bus, config, gateway, registry, dispatcher, assembler, memory, loop, trace,
            assets, memories, skills, catalog, mem_agent, guard,
            allocator=allocator, subagents=subagents, retrieval=retrieval,
        )
        agent.media_gw, agent.audio_gw = media_gw, audio_gw
        agent.checker, agent.packager, agent.metrics = checker, packager, metrics
        agent.ledger = ledger
        agent.session_store = session_store
        agent.session_lock, agent.second_window = session_lock, second_window
        agent.set_content_line(session_store.content_line, persist=False)
        agent.drama_fns, agent.episode_fns = drama, episode_fns
        agent.base_episode_fmt = episode_fmt
        agent.set_episode_minutes(
            session_store.episode_minutes or None, persist=False, auto=session_store.episode_auto
        )
        agent.output_prefs = output_prefs
        agent.local_materials = local
        agent.media_fns = media_fns
        agent.pipeline = pipeline
        agent.workspace = workspace
        agent.config_dir = config_dir
        agent.project = project
        agent.recorder = recorder
        agent.memory_source = memory_source
        agent.short_video_fns, agent.material_fns = short_video, material_fns
        agent.set_aspect_ratio(session_store.aspect_ratio or None, persist=False)
        agent.set_cut_block(session_store.cut_block, persist=False)
        agent._apply_locks()
        # 每个 LOOP_END 存一次快照。存的是 agent.session_store —— /out 换会话之后存到新的那份
        bus.subscribe(agent._save_snapshot)
        # 创作方案（剧本大节点）人采纳时，方案里写的每集时长就是这个项目的集长（2026-09-26）
        bus.subscribe(agent._on_plan_review)
        return agent

    def _save_snapshot(self, ev: Any) -> None:
        if ev.type is EventType.LOOP_END and self.session_store is not None:
            self._park_session()

    def _park_session(self) -> None:
        """把当前窗口、挂起的人审、加载过的 skill 存进当前会话快照。"""
        if self.session_store is None:
            return
        self.session_store.save(
            self.memory,
            pending_review=self.loop.pending_review,
            active_skills=list(getattr(self.allocator, "active", []) or []),
        )

    # ---------- 模型锁的默认值按产线取（2026-09-26 用户定的） ----------

    def _default_lock(self, kind: str) -> str:
        """没人定过锁时的默认：短剧 / 不限定用短剧配置里指定的模型（media_models.yaml drama）；
        抖音 / 广告 / 设计不锁，交给配方的档位选 —— 之前默认锁对所有产线生效，新开广告文件夹，
        product-ad 的 quality 档被锁成 seedance-2.0，海报也跟着用 gpt-image-2。"""
        if self.content_line and self.content_line != "drama":
            return ""
        return str((getattr(self.catalog, "drama", None) or {}).get(f"{kind}_model") or "")

    def _apply_locks(self) -> None:
        """模型锁 = 这个项目人定过的（会话快照）> 按产线的默认。换锁回写到当前会话快照。"""
        mf = self.media_fns
        if mf is None:
            return
        store = self.session_store
        mf.video_lock = (getattr(store, "video_model", "") or "") or self._default_lock("video")
        mf.image_lock = (getattr(store, "image_model", "") or "") or self._default_lock("image")
        if store is not None:
            mf.on_video_lock = store.set_video_model
            mf.on_image_lock = store.set_image_model

    # ---------- 人采纳的创作方案里写的每集时长 → 项目集长（2026-09-26） ----------
    # 之前集长有三个来源：/length、人审里问的「每集时长」（答了不生效）、drama_write 的 minutes
    # （模型能自己传）。现在只有项目规格一个来源；人审问题里写明的时长，人采纳了就写进去 ——
    # 人看过、点了头，才算数（/auto 自动采纳的不算；采纳附言里另写了时长以附言为准）。

    def _on_plan_review(self, ev: Any) -> None:
        data = getattr(ev, "data", None) or {}
        if ev.type is EventType.CHECKPOINT_REACHED:
            self._plan_review = dict(data)
            return
        if ev.type is not EventType.CHECKPOINT_DECIDED:
            return
        node = str(data.get("node") or "")
        if node != "剧本" and "方案" not in node:
            return
        if data.get("decision") != "adopt" or str(data.get("decided_by") or "") == "auto":
            return
        asked = getattr(self, "_plan_review", None) or {}
        question = str(asked.get("question") or "") if asked.get("stage") == node else ""
        for text in (str(data.get("reason") or ""), question):
            minutes, auto = plan_length(text)
            if auto:
                if not self.episode_fmt.follow_script:
                    self.set_episode_minutes(None, auto=True)
                    self._length_note("集长跟剧本走（你采纳的方案里写的）")
                return
            if minutes:
                if self.episode_fmt.follow_script or minutes != self.episode_fmt.minutes:
                    self.set_episode_minutes(minutes)
                    self._length_note(f"每集 {minutes:g} 分钟（你采纳的方案里写的）")
                return

    def _length_note(self, what: str) -> None:
        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().create_task(
                self.bus.emit(
                    EventType.WARNING, message=f"这个项目的集长设为{what}，/length 可改"
                )
            )

    # ---------- 项目（缺口 A：产物目录 = 项目） ----------

    @property
    def project_title(self) -> str:
        """给人看的项目名：剧名（.drama-state.json）优先，否则文件夹名。"""
        return project_title(getattr(self.output_prefs, "root", None))

    def switch_project(self, root: Path | str, session: bool = False) -> str:
        """换产物目录 = 换项目：资产库查询范围、台账的单项目累计、记忆一起换。返回新项目键。

        session=True（/out 用）连会话一起换（2026-09-26 用户定的）：对话窗口、集长、画幅、产线、
        模型锁都换成那个项目自己的（它文件夹的会话快照），当前的先存回本项目。之前 /out 只换
        资产和记忆 —— 在《不渡》里 /out 到西游记，西游记按《不渡》的 20 分钟和模型锁跑，窗口里
        还是《不渡》的对话，提示却说「本项目记住」。

        流水线还有活在跑时不换 —— 它们产出的资产会被打上新项目的键（抛 RuntimeError，
        CLI 报给人：等跑完或 /auto stop 之后再换）。
        """
        # 先 resolve：相对路径 / junction 写法派生出的项目键和直接在该目录启动的不一样，
        # 而且相对串会被存进快照，换个目录启动就解析成另一个项目（2026-09-24 审查）
        root = Path(root).expanduser().resolve()
        busy = getattr(self.pipeline, "busy", False)
        if busy and normalize_root(root) != normalize_root(self.output_prefs.root):
            raise RuntimeError("流水线还有任务在跑：等它们跑完（或 /auto stop）再换产物目录")
        if session:
            # 先存：窗口里这些轮次属于旧项目。记忆提取按入队时的项目键记，排着的不会串到新项目
            self._park_session()
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
        if session:
            self._enter_session(root)
        elif self.session_store is not None:
            self.session_store.set_output_dir(str(root))
        return key

    def _enter_session(self, root: Path) -> None:
        """换到 root 这个文件夹的会话：装回它的窗口、挂起的人审、项目设置（集长 / 画幅 / 产线 /
        模型锁）。和从这个文件夹启动 Agent 看到的是同一份。"""
        old = self.session_store
        if old is None or self.workspace is None:
            return
        name = _session_for(root, self.workspace)
        if name == old.name or (self.second_window and old.name.startswith(f"{name}~")):
            old.set_output_dir(str(root))
            return
        # 那个项目有别的窗口在用：另起「名字~2」，和开第二个窗口一样
        new, lock, second = open_session(old.root, name)
        if self.session_lock is not None:
            self.session_lock.release()
        self.session_lock, self.second_window = lock, second
        self.memory.turns = []
        self.memory._next_index = 0  # noqa: SLF001
        self.memory.observed_prompt_tokens = 0
        self.memory.unpin(ROLLBACK_PIN)  # 回退说明是旧项目的
        self.loop.pending_review = None
        self.loop._recalled = ""  # noqa: SLF001
        self.session_store = new
        new.load_into(self.memory)
        pr = new.pending_review
        if pr and any(t.index == pr.get("turn_index") for t in self.memory.turns):
            self.loop.pending_review = dict(pr)
        new.set_output_dir(str(root))
        if self.media_fns is not None:
            # 旧项目里挂着等人拍板的换模型请求、「只这一次」的放行，不带到新项目
            self.media_fns._pending_switch.clear()  # noqa: SLF001
            self.media_fns._once.clear()  # noqa: SLF001
        self.set_content_line(new.content_line, persist=False)
        self.set_episode_minutes(new.episode_minutes or None, persist=False, auto=new.episode_auto)
        self.set_aspect_ratio(new.aspect_ratio or None, persist=False)
        self.set_cut_block(new.cut_block, persist=False)
        self._apply_locks()

    async def restore_skills(self) -> list[str]:
        """换会话之后：卸下旧会话加载的 skill，装上这个会话上次加载过的。返回装上的。"""
        if self.allocator is None or self.session_store is None:
            return []
        wanted = list(self.session_store.active_skills)
        for name in list(getattr(self.allocator, "active", {}) or {}):
            if name not in wanted:
                self.allocator.deactivate_skill(name)
        loaded: list[str] = []
        for name in wanted:
            skill = self.skills.get(name)
            if skill is None:
                continue
            try:
                await self.allocator.activate_skill(skill)
            except Exception:  # noqa: BLE001
                continue
            loaded.append(name)
        return loaded

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
        if self.session_lock is not None:
            self.session_lock.release()  # 进程退出也会自动放；测试里同一进程开多个 Agent 要显式放

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
        self._pin_aspect()
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
        card = build_project_card(self.assets, root, self.episode_fmt)
        if card:
            card += "\n" + self.episode_spec_line()
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
        # 5. 按集流水等人处理的事：停在停点、失败了要人定、额度刹车。之前只打在控制台，
        #    人问「怎么不动了」模型不知道，还会自己去重跑（2026-09-26）
        notices = ""
        if self.pipeline is not None:
            with contextlib.suppress(Exception):
                notices = str(self.pipeline.notices() or "")
        if notices:
            self.memory.pin(PIPELINE_PIN, notices, position="pre_input")
        else:
            self.memory.unpin(PIPELINE_PIN)

    # ---------- 一集的时长（2026-09-25 用户要的：按项目放宽） ----------

    def set_episode_minutes(
        self, minutes: float | None, persist: bool = True, auto: bool = False
    ) -> EpisodeFormat:
        """这个项目一集几分钟。None / 0 = 用 config/drama.yaml 的默认；auto=True = 集长跟剧本走
        （/length auto，2026-09-26 用户定的：不查总时长，只查台词念不念得完、单镜 ≤3 秒、开场
        高潮点 —— 之前集长只能是一个目标值，重拆一集会被要求加戏或压戏凑时长）。

        只由人改（/length，或人采纳的创作方案里写的每集时长）：集长直接决定每集要生成多少段
        视频（花多少钱），不能让模型为了让规格检查通过自己调。写剧本、拆分镜、写视频提示词
        三步读的都是这里套好的规格 —— 这是集长唯一的来源。
        """
        base = self.base_episode_fmt
        if auto:
            fmt = replace(base, follow_script=True)
        else:
            fmt = replace(base, minutes=float(minutes)) if minutes else base
        for holder in (self.drama_fns, self.episode_fns):
            if holder is not None:
                holder.fmt = fmt
        self.episode_fmt = fmt
        if persist and self.session_store is not None:
            self.session_store.set_episode_minutes(float(minutes or 0.0), auto=auto)
        return fmt

    # ---------- 视频画幅（2026-09-25 用户要的：能出 16:9，比例按项目个性化选） ----------

    def set_aspect_ratio(self, ratio: str | None, persist: bool = True) -> str:
        """这个项目的视频画幅。None / 空 = 默认（短剧竖屏 9:16、短视频按配方）。返回生效的画幅。

        短剧渲染、短视频出片、gen_video 没传比例时、素材站搜索方向、图片转镜头的尺寸都跟着它。
        """
        r = parse_aspect(ratio or "")
        self.aspect_custom = bool(r)
        self.aspect_ratio = r or DEFAULT_ASPECT
        if self.drama_fns is not None:
            self.drama_fns.aspect_ratio = self.aspect_ratio
        if self.short_video_fns is not None:
            self.short_video_fns.aspect_ratio = r
        if self.media_fns is not None:
            self.media_fns.default_aspect = r
        if self.material_fns is not None:
            self.material_fns.default_orientation = orientation_of(r) if r else ""
        self._pin_aspect()
        if persist and self.session_store is not None:
            self.session_store.set_aspect_ratio(r)
        return self.aspect_ratio

    # ---------- 镜头超 3 秒拦不拦成片（2026-09-27 用户定的：先只标，第 1 集看过再定） ----------

    @property
    def cut_block(self) -> bool:
        return bool(getattr(self.drama_fns, "cut_block", False))

    def set_cut_block(self, on: bool, persist: bool = True) -> None:
        """True = 镜头超 3 秒自动重生成、仍超不进成片（⛔）；False = 只标 ⚠。只由人改（/cut）。"""
        if self.drama_fns is not None:
            self.drama_fns.cut_block = bool(on)
        if persist and self.session_store is not None:
            self.session_store.set_cut_block(bool(on))

    def _pin_aspect(self) -> None:
        if not self.aspect_custom:
            self.memory.unpin(ASPECT_PIN)
            return
        self.memory.pin(
            ASPECT_PIN,
            f"## 本项目视频画幅：{aspect_label(self.aspect_ratio)}（用户用 /ratio 设的）\n"
            "短剧渲染、短视频出片、gen_video 没传比例时都按它；不要自己改成别的比例，"
            "用户要换就提醒他 /ratio。",
            position="pre_input",
        )

    def episode_spec_line(self) -> str:
        """项目卡里的「规格」一行：这个项目一集几分钟、分镜总长要落在哪个区间。"""
        fmt = self.episode_fmt
        if fmt.follow_script:
            return (
                f"规格：{fmt.brief()}；画幅 {aspect_label(self.aspect_ratio)}（本项目用户用 "
                "/length auto 设的：分镜总长按台词念完 + 必要的动作定，不按集长凑）。"
                "集长只有用户能改，不要为了让检查通过去删台词、加戏或自己调集长"
            )
        lo, hi = fmt.duration_range
        custom = fmt.minutes != self.base_episode_fmt.minutes
        return (
            f"规格：{fmt.brief()}；分镜总长 {lo}–{hi} 秒；画幅 {aspect_label(self.aspect_ratio)}"
            + ("（本项目用户用 /length 设的）" if custom else "（默认值，用户可用 /length 改）")
            + "。集长只有用户能改，不要为了让检查通过去删台词或自己调集长"
        )

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
        self._apply_locks()  # 没人定过锁的：默认锁跟着产线换

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

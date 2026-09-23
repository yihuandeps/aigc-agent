"""短剧流水线 functions —— 剧本 → 分镜脚本 / 资产库 → 视频提示词。

三步是**有依赖顺序**的：第三步要同时拿到前两步的产物，因为它的核心工作
就是把镜头里的"ANNE"换成 `(Anne-酒店员工制服-[全集])` 这种带资产 ID 的
引用，让每一镜的人物、服装、场景都指向同一张参考图 ——
这是整套流程存在的理由：**跨镜头、跨集的视觉一致性**。

产物都落成资产（STORYBOARD 类型），下一步按 id 取，不用人复制粘贴。
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx2 as httpx

from ...envdetect import workspace_root
from ...harness.events.bus import EventBus, EventType
from ...harness.model.media import MediaKind, default_proxy
from ...harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
    ToolSpec,
)
from ..assets.store import AssetStatus, AssetStore, AssetType, local_copy
from ..drama import (
    ask_script_next,
    ask_text,
    ask_unclear,
    assets_system,
    audit_refs,
    bind_costumes,
    classify,
    expand_prompt,
    normalize,
    parse_assets,
    parse_episodes,
    parse_shots,
    shots_prompt,
    shots_system,
    storyboard_system,
    unbound_costumes,
    write_prompt,
)
from ..drama.faces import (
    FaceAudit,
    FaceCandidate,
    audit_lines,
    conflict_question,
    decide,
    group_candidates,
    matches_character,
    pick_baseline,
)
from ..drama.format import DEFAULT_FORMAT, EpisodeFormat, check_shots, check_storyboard
from ..drama.identity import (
    identity_check_messages,
    identity_retry_prompt,
    parse_identity_verdict,
    reference_block,
)
from ..drama.models import as_dict, normalize_ref_parens
from ..drama.voice import (
    ANCHORS_CREATOR,
    Anchor,
    AnchorPlan,
    anchors_table,
    dump_anchors,
    parse_anchors,
    plan_anchors,
    speakers_of,
    voice_block,
)
from ..media import ffmpeg
from ..media.fast_cut import fast_cut_prompt, fast_cut_retry, longest_shot
from ..media.naming import (
    clip_name,
    episode_export_name,
    library_reference_names,
    parse_scene,
    unique_path,
)
from ..media.no_text import no_text_retry, parse_subtitle_verdict, subtitle_check_messages
from ..realism import (
    REALISM_ROLE,
    is_minor,
    norm_level,
    parse_realism_report,
    person_image_prompt,
    person_video_prompt,
    realism_check_messages,
)
from .content import STUB_HINTS
from .media import retryable_failure

# 人审挂起的环节名：同一个角色查出多张互不一致的脸、又定不了谁是准的时候，
# 按它在总线上认决策（2026-09-22 用户定：Agent 自动挑，歧义才问）
FACE_CONFLICT_STAGE = "角色面容冲突"

_ROLE = "drama"
# 写剧本 / 扩写是散文输出。drama 角色配了 response_format: json，写作被连带后
# 吐 {"script": "..."} 还得剥壳 —— 单独一个角色，没配时退回 drama。
_PROSE_ROLE = "drama_prose"


async def _run_parallel(items: list[Any], fn: Any, limit: int) -> list[Any]:
    """并发跑一批**互相无依赖**的任务，结果顺序与输入一致。limit <= 0 不限。

    依赖不在这里表达 —— 调用方先分层/分批，只有同层（无依赖）才进这个池子。
    """
    if limit > 0:
        sem = asyncio.Semaphore(limit)

        async def one(it: Any) -> Any:
            async with sem:
                return await fn(it)
    else:

        async def one(it: Any) -> Any:
            return await fn(it)

    return await asyncio.gather(*(one(it) for it in items))


class DramaFunctions:
    name = "drama"
    namespaced = False
    disclosure = "full"

    def __init__(
        self,
        gateway: Any,
        store: AssetStore,
        registry: Any = None,
        catalog: Any = None,
        bus: EventBus | None = None,
        fmt: EpisodeFormat | None = None,
        hosting: Any = None,
    ) -> None:
        self.gateway = gateway
        self.store = store
        # 生图/生视频用哪个模型走配置，不写死 —— 见 media_models.yaml 的 drama 段
        self.catalog = catalog
        # 一集的规格：4 分钟、开场 15 秒高潮点（config/drama.yaml）。写、拆、提示词三步都按它查
        self.fmt = fmt or DEFAULT_FORMAT
        # 本地素材托管（config/hosting.yaml）：用户的本地图当参考、过期链接重新上传都靠它
        self.hosting = hosting
        self.files: Any = None  # FileFunctions：drama_use_local_ref 登记本地图（app.py 里装）
        # 渲染要调 gen_image / gen_video，走同一个注册表 ——
        # 权限、成本记账、事件日志才不会分叉出第二套。
        self.registry = registry
        # 批量渲染的进度汇报（BATCH_PROGRESS）发到这里，CLI 进度窗订阅它
        self.bus = bus
        self._specs: dict[str, ToolSpec] = {}
        self._build()

    async def _progress(self, stage: str, done: int, total: int, item: str = "") -> None:
        """批量渲染进度汇报。没接 bus 就静默（测试/脚本环境）。"""
        if self.bus is not None:
            await self.bus.emit(
                EventType.BATCH_PROGRESS, stage=stage, done=done, total=total, item=item
            )

    def _build(self) -> None:
        self._specs["drama_intake"] = ToolSpec(
            name="drama_intake",
            summary="判断用户给的是一句话想法还是完整剧本，并指出下一步",
            permission=PermissionLevel.READ,
            description=(
                "**进短剧流程的第一步。** 先分清输入是什么："
                "一句话想法 → 得先调 drama_write 写成剧本；"
                "完整剧本 → 问用户直接进工程还是先扩写修改。"
                "判错的代价不对称：想法当剧本会拿一句话去拆分镜（模型只能硬编），"
                "剧本当想法会让用户重写他已经写好的东西。所以按结构信号判，"
                "看不准就如实说。"
            ),
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "用户给的原始输入"}},
                "required": ["text"],
            },
        )

        self._specs["drama_write"] = ToolSpec(
            name="drama_write",
            summary="一句话想法 → 短剧剧本",
            permission=PermissionLevel.COMPUTE,
            description=(
                "按短剧方法论写剧本：开篇前 3 秒定生死、结尾断在信息缺口、"
                "爽点同集兑现、反派要有可理解的动机。"
                "写完简报集数与字数，然后**直接进拆解链**"
                "（drama_storyboard → drama_assets）——方向在创作方案阶段已拍板，"
                "不再单独停下来等确认。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "idea": {"type": "string", "description": "一句话想法或题材方向"},
                    "episodes": {"type": "integer", "description": "写几集，默认 1"},
                    "minutes": {
                        "type": "number",
                        "description": "每集几分钟，省略按一集规格（config/drama.yaml，默认 4）",
                    },
                },
                "required": ["idea"],
            },
        )

        self._specs["drama_expand"] = ToolSpec(
            name="drama_expand",
            summary="已有剧本 → 按方法论扩写修改",
            permission=PermissionLevel.COMPUTE,
            description=(
                "用户选了「先扩写/修改」时用。按短剧方法论过一遍：补钩子、"
                "调节奏曲线、加爽点、查合规。"
                "**保留原人物、设定和主线**，是加固不是重写 —— 用户要的是改，不是换。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "script": {"type": "string", "description": "剧本原文"},
                    "script_id": {"type": "string", "description": "或者给剧本资产 id"},
                    "note": {"type": "string", "description": "具体想改什么，可省略"},
                },
            },
        )

        self._specs["drama_storyboard"] = ToolSpec(
            name="drama_storyboard",
            summary="剧本 → 分集分镜脚本（逐字保留台词）",
            permission=PermissionLevel.COMPUTE,
            # 一次 gemini 调用 56–105s，规格不合格还要改一次（_revise_once），
            # 默认 120s 必超 —— 2026-09-21 两次被误杀，第一次调用的钱白花
            timeout=600,
            description=(
                "第①步。把短剧剧本拆成分集的镜头序列：场景标头 + 逐镜的"
                "景别/运镜/视觉动作 + **100% 保留的原文台词**。\n"
                "**一次拆一集**（传这一集的 script_id）：一集 80–120 个镜头行，几集并成一次"
                "会超出模型单次输出上限被截断。产物存为资产，第③步要用它的 id。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "script_id": {
                        "type": "string",
                        "description": "剧本资产 id（**优先用它**：一集或几集的 script 资产，"
                        "find_episode / list_assets 查得到）—— 长剧本别整段塞进参数",
                    },
                    "script": {
                        "type": "string",
                        "description": "剧本原文（没有资产 id 时才用；不能是占位符或摘要）",
                    },
                    "ethnicity": {
                        "type": "string",
                        "enum": ["asian", "chinese", "caucasian", "african", "latino", "mixed"],
                        "description": "画面里的面孔族裔。**用户没明确说就先问他**，不要替他选",
                    },
                    "language": {
                        "type": "string",
                        "enum": ["zh", "en", "keep"],
                        "description": "台词语言。**用户没明确说就先问他**，不要替他选",
                    },
                    "note": {"type": "string", "description": "额外要求，可省略"},
                },
                "required": ["ethnicity", "language"],
            },
        )

        self._specs["drama_assets"] = ToolSpec(
            name="drama_assets",
            summary="剧本 → 资产库（角色 / 按场景分配的服装 / 场景 / 道具，可直接生图）",
            permission=PermissionLevel.COMPUTE,
            # 一次 gemini 调用就要 90s 上下，默认 120s 没余量，连接重试一抖就超
            timeout=600,
            description=(
                "第②步。抽出角色主形象提示词、**按场景分配的服装**（每套带 scenes / episodes：\n"
                "哪些场景、哪几集穿它；一个角色×场景组合必须有对应服装）、场景、道具。\n"
                "角色和服装描述带一段**锁定后缀**（纯白底/单张全身/16:9/8K），"
                "不透传给人看、也不让人改，生图时由服务端原样拼回去。\n"
                "返回里有服装-场景分配表，缺口会标 ⚠。产物存为资产，第③步要用它的 id。\n"
                "全剧的剧本传 script_ids（按集号顺序的剧本资产 id 列表），不要把几万字塞进参数。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "script_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "全剧各集的剧本资产 id（按集号顺序）—— **优先用它**",
                    },
                    "script_id": {
                        "type": "string",
                        "description": "剧本资产 id（**优先用它**：一集或几集的 script 资产，"
                        "find_episode / list_assets 查得到）—— 长剧本别整段塞进参数",
                    },
                    "script": {
                        "type": "string",
                        "description": "剧本原文（没有资产 id 时才用；不能是占位符或摘要）",
                    },
                    "ethnicity": {
                        "type": "string",
                        "enum": ["asian", "chinese", "caucasian", "african", "latino", "mixed"],
                        "description": "画面里的面孔族裔。**用户没明确说就先问他**，不要替他选",
                    },
                    "language": {
                        "type": "string",
                        "enum": ["zh", "en", "keep"],
                        "description": "台词语言。**用户没明确说就先问他**，不要替他选",
                    },
                    "note": {"type": "string", "description": "额外要求，可省略"},
                },
                "required": ["ethnicity", "language"],
            },
        )

        self._specs["drama_shots"] = ToolSpec(
            name="drama_shots",
            summary="分镜脚本 + 资产库 → seedance 视频提示词（带资产 ID 绑定）",
            permission=PermissionLevel.COMPUTE,
            # 一次 gemini 调用 38–118s，规格不合格还要改一次，默认 120s 必超 ——
            # 2026-09-22 连着四次被误杀，连只有 8 镜的单场都没跑完
            timeout=600,
            description=(
                "第③步。**必须先跑完①和②**。把镜头描述里的角色名换成资产 ID"
                "（如 (Anne-酒店员工制服-[1-3])），场景道具同理。\n"
                "服装按镜头所在场景**自动绑定**（确定性校正，不靠模型自觉）：同一场景穿搭一致，"
                "换场景才换装；资产库没配的角色×场景会报缺口。\n"
                "会自动审计引用：模型编一个资产库里没有的 ID 出来时会报出来 ——"
                "那种错看着没毛病，但生视频时找不到参考图，人物就会变脸。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "storyboard_id": {"type": "string", "description": "第①步的资产 id"},
                    "assets_id": {"type": "string", "description": "第②步的资产 id"},
                    "note": {"type": "string", "description": "额外要求（返工时写清楚要改什么）"},
                    "episode": {
                        "type": "integer",
                        "description": "只处理第几集。省略则全部（长剧本建议分集跑）",
                    },
                },
                "required": ["storyboard_id", "assets_id"],
            },
        )

        self._specs["drama_render_assets"] = ToolSpec(
            name="drama_render_assets",
            summary="给资产库批量生图（角色主形象 / 各场景服装 / 场景 / 道具）",
            permission=PermissionLevel.COMPUTE,
            # 一批几十张图，并发跑也要几分钟；默认 120s 会误杀
            timeout=3600,
            description=(
                "第②步的产物 → 参考图，四类两层：角色主形象（脸）→ 各场景服装"
                "（参考主形象保脸，一套一张纯白底全身服装图）；场景 / 道具无依赖。"
                "没有主形象就出不了服装图 —— 这是同一个角色换装后还是同一张脸的唯一手段。"
                "同一层里无依赖的图**并发**生成，"
                "并发上限走 media_models.yaml 的 concurrency.image。"
                "版式参数固定（角色 3:4、其余 16:9、720P），不由模型决定。"
                "**增量**：同一资产库再跑一次只补上次没生成的（默认 reuse），不重复花钱；"
                "分镜引用的是**服装名和场景名**，只渲 characters 的话渲视频时服装会退回"
                "角色主形象、场景无参考 —— 返回里有覆盖率，缺的补跑一次即可。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "assets_id": {"type": "string", "description": "第②步的资产 id"},
                    "only": {
                        "type": "string",
                        "enum": ["all", "characters", "costumes", "scenes", "props"],
                        "description": "只渲染某一类，默认 all（costumes 会连带缺的主形象一起渲）",
                    },
                    "reuse": {
                        "type": "boolean",
                        "description": "复用这套资产库上次已生成的图，只补缺的（默认 true）",
                    },
                },
                "required": ["assets_id"],
            },
        )

        self._specs["drama_render_shots"] = ToolSpec(
            name="drama_render_shots",
            summary="把分镜提示词批量生成视频（seedance）",
            permission=PermissionLevel.COMPUTE,
            # 一部剧几百段视频，按依赖分层并发也要以小时计；默认 120s 会误杀
            timeout=7200,
            description=(
                "第③步的产物 → 视频片段。每段会自动带上它引用到的资产图"
                "（角色/服装/场景/道具）作为参考，以及前序片段（{第X集-Y场}）。"
                "服装引用在参考图包里没有时退回该角色的主形象（保脸）。"
                "**渲染前引用门**（用户规则：没有引用成功的镜头不许生成）：逐段核对引用的参考图都在包里、"
                "描述里出现的角色都带了参考图、链接没过期且可访问、参考数不超模型上限，"
                "任一不过**整批不发起、不花钱**，报出该修什么。某几段生成失败**不成片**，"
                "修好原因后重跑本工具（reuse=true 只补失败的段）；不要用 gen_video 自己补。"
                "说话角色的音色卡会锁进提示词，并把该角色的音色锚点片段作为参考视频传入"
                "（跨集沿用，防止声音漂移；锚点见 drama_voice_anchors）。"
                "无引入依赖的镜头**并发**生成，带引入的按依赖排队"
                "（上限见 media_models.yaml 的 concurrency.video）。"
                "9:16 / 720p 固定。**很慢很贵**，建议先用 limit 跑一两段看方向。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "shots_id": {"type": "string", "description": "第③步的资产 id"},
                    "rendered_id": {
                        "type": "string",
                        "description": (
                            "drama_render_assets 的产物 id（参考图包）。省略则自动找这套"
                            "资产库最新的包；误传成资产库 id 也会自动换成它的包"
                        ),
                    },
                    "limit": {"type": "integer", "description": "只跑前几段，0=全部"},
                    "episode": {
                        "type": "integer",
                        "description": "只渲染第几集（按「第N集-」前缀过滤镜头），0=全部。"
                        "按集流水时传它，成片默认命名为 第N集.mp4",
                    },
                    "compose": {
                        "type": "boolean",
                        "description": "跑完直接拼成整集，默认 true",
                    },
                    "out_dir": {"type": "string", "description": "成片导出目录"},
                    "filename": {"type": "string", "description": "成片文件名"},
                    "reuse": {
                        "type": "boolean",
                        "description": "复用这份提示词上次已成功、过了质检门、参考图包没换的片段，"
                        "只重生成失败/缺的（默认 true；要全部重生成传 false）",
                    },
                    "redo": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "reuse 时强制重渲这几段（写场次或镜头，如 "
                        "\"第1集-2场 镜5-9\" 或 \"镜5-9\"）—— 人复核后不满意的段用它",
                    },
                    "accept": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "上次被质检门标 ⛔（仍有字幕 / 人物漂移 / 字幕没查成）"
                        "没进成片的段，**用户亲自看过说可以**才写进来放行（写法同 redo）。"
                        "不要替用户决定",
                    },
                },
                "required": ["shots_id"],
            },
        )

        self._specs["drama_refresh_refs"] = ToolSpec(
            name="drama_refresh_refs",
            summary="参考图链接过期时，把本地副本重新上传图床换新链接（图不变、不花生成的钱）",
            permission=PermissionLevel.WRITE,
            description=(
                "生成接口返回的图片链接约 24 小时失效，之后渲视频时模型拿不到参考图 —— "
                "表面上「参考 N 图」，实际零参考，人物必漂。这个工具把参考图的本地副本重新上传"
                "到素材托管（config/hosting.yaml 配的图床），换成新链接并生成新的参考图包。"
                "没配托管就刷不了（生成接口只收公网链接）。"
                "渲染前看到「链接超过 N 小时」的提示就先跑它。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "rendered_id": {
                        "type": "string",
                        "description": "参考图包 id，省略用最新的包",
                    },
                    "only_stale": {
                        "type": "boolean",
                        "description": "只刷新链接可能过期的（默认 true）；false = 全部刷新",
                    },
                },
            },
        )

        self._specs["drama_use_local_ref"] = ToolSpec(
            name="drama_use_local_ref",
            summary="把用户本地的图放进参考图包，当角色主形象 / 服装 / 场景 / 道具的参考",
            permission=PermissionLevel.WRITE,
            description=(
                "用户自己有素材（角色定妆照、产品图、实景照）时用：登记本地图、上传到用户配置的"
                "存储拿到公网链接（config/hosting.yaml），写进这套资产库最新的参考图包。之后 "
                "drama_render_assets(reuse=true) 沿用它不再生成该项，drama_render_shots 引用到"
                "这个名字时自动带上并做人物一致性校验。name 必须是资产库里的名字"
                "（角色名 / 服装名 / 场景名 / 道具名）。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "资产库里的名字，如 陆离 / 陆离-风衣-[全集]",
                    },
                    "path": {"type": "string", "description": "本地图片路径"},
                    "asset_id": {"type": "string", "description": "或已登记的图片资产 id"},
                    "assets_id": {"type": "string", "description": "资产库 id，省略用最新的"},
                    "kind": {
                        "type": "string",
                        "enum": ["auto", "角色", "服装", "场景", "道具"],
                        "description": "默认按名字在资产库里自动判断",
                    },
                },
                "required": ["name"],
            },
        )

        self._specs["drama_audit_faces"] = ToolSpec(
            name="drama_audit_faces",
            summary="查同一个角色有没有多张不同的脸，只留一张，其余移进回收目录",
            permission=PermissionLevel.WRITE,
            timeout=1200,  # 一个角色一次视觉比对，角色多了要几分钟
            description=(
                "生成过程里同一个角色常会攒出好几张脸（换了模型、重生成、手动补图），"
                "后面渲视频引用到哪张全凭运气，成片就会变脸。这个工具把散在参考图包、"
                "资产库、产物目录里的同名角色图归堆，用视觉模型逐张和基准比对，"
                "**只保留一张**：你自己钉过的 > 参考图包在用的 > 产物里最新的。\n"
                "不合格的**不删**，只移进 workspace/trash/ 并在资产上标 superseded_by，随时找得回。"
                "有角色定不了谁是准的（几张脸各自成群、又没人钉过）会**暂停问你**。\n"
                "默认 drama_render_assets 跑完自动查一遍"
                "（media_models.yaml drama.face_audit 可关）。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "assets_id": {"type": "string", "description": "资产库 id，省略用最新的"},
                    "character": {"type": "string", "description": "只查某个角色，省略查全部"},
                    "apply": {
                        "type": "boolean",
                        "description": "true（默认）执行清理；false 只出报告不动任何文件",
                    },
                    "deep": {
                        "type": "boolean",
                        "description": "连服装图一起复核（默认 false：服装图生成时已过一道"
                        "一致性门，再查一遍是重复花钱）",
                    },
                },
            },
        )

        self._specs["drama_voice_anchors"] = ToolSpec(
            name="drama_voice_anchors",
            summary="查看 / 指定 / 清除角色的音色锚点片段",
            permission=PermissionLevel.WRITE,
            description=(
                "音色锚点 = 每个角色的声音基准片段：之后他开口的每一段视频都把这段当参考视频"
                "传给模型，声音才不会一集一个样。渲染时自动定（优先独白段），"
                "用户觉得某段的声音更对时用 pin 指定那段；clear 清掉后下次渲染重定。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "pin", "clear"]},
                    "character": {"type": "string", "description": "角色名（pin/clear 必填）"},
                    "clip_id": {"type": "string", "description": "pin 用：视频片段资产 id"},
                },
                "required": ["action"],
            },
        )

    # ---------- ToolProvider ----------

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        if self.registry is None:
            return ProviderHealth(ok=True, detail="拆解可用；渲染不可用（没接注册表）")
        return ProviderHealth(ok=True, detail="短剧流水线就绪")

    @property
    def image_model(self) -> str:
        """渲染用的生图模型：本会话锁定的 > 配置 > 内置默认。见 video_model 的说明。"""
        cfg = getattr(self.catalog, "drama", {}) or {}
        return self._locked("image_lock_source") or cfg.get("image_model") or IMAGE_MODEL

    @property
    def video_model(self) -> str:
        """渲染用的视频模型：本会话锁定的（用户指定或同意切换后的）> 配置 > 内置默认。

        用户定的规则：换模型之前必须先问用户（视频 2026-09-20、生图 2026-09-22）。
        锁在媒体层（MediaFunctions.video_lock / image_lock），装配层把它接进来
        （video_lock_source / image_lock_source）；短剧链跟着锁走，用户同意换了
        就整条链一起换。
        """
        cfg = getattr(self.catalog, "drama", {}) or {}
        return self._locked("video_lock_source") or cfg.get("video_model") or VIDEO_MODEL

    def _locked(self, source_attr: str) -> str:
        """问装配层要当前锁定的模型。没接线或取不到就当没锁。"""
        source = getattr(self, source_attr, None)
        if not callable(source):
            return ""
        try:
            return str(source() or "")
        except Exception:  # noqa: BLE001
            return ""

    def _max_concurrency(self, kind: str) -> int:
        """批量渲染并发上限（"image" / "video"）。0 = 不限；没配 catalog 时不限。"""
        if self.catalog is None:
            return 0
        return self.catalog.max_concurrency(kind)

    # ---------- 音色锚点 ----------

    def _voice_cfg(self) -> dict[str, Any]:
        """media_models.yaml drama 段：voice_anchor / ref_url_ttl_hours / max_video_refs。"""
        cfg = getattr(self.catalog, "drama", {}) or {}

        def num(key: str, default: float) -> float:
            try:
                return float(cfg.get(key, default))
            except (TypeError, ValueError):
                return default

        enabled = str(cfg.get("voice_anchor", "true")).strip().lower() in ("1", "true", "yes", "on")
        return {
            "enabled": enabled,
            "ttl_h": num("ref_url_ttl_hours", 20.0),
            "max_videos": max(0, int(num("max_video_refs", 3))),
        }

    def _load_anchors(self) -> dict[str, Anchor]:
        items = self.store.find(creator=ANCHORS_CREATOR)
        if not items:
            return {}
        try:
            return parse_anchors(self.store.content(items[0].id))
        except KeyError:
            return {}

    def _save_anchors(self, anchors: dict[str, Anchor], parents: list[str]) -> str:
        prev = self.store.find(creator=ANCHORS_CREATOR)
        asset = self.store.create(
            dump_anchors(anchors),
            type_=AssetType.STORYBOARD,
            summary=f"音色锚点·{len(anchors)}人",
            parents=([prev[0].id] if prev else []) + [p for p in parents if p],
            creator=ANCHORS_CREATOR,
            gen_params={"count": len(anchors)},
        )
        return asset.id

    def _merge_save_anchors(self, new_anchors: dict[str, Anchor], parents: list[str]) -> str:
        """把这次新定的锚点并进**最新的**锚点表再存。

        渲染这段时间里别的集可能写过锚点表（/auto 两集并行）：之前开渲时读、结束时整表覆盖，
        后结束的那集把先结束那集新定的锚点冲掉（2026-09-23 审查）。人手动 pin 的不覆盖。
        """
        latest = self._load_anchors()
        for name, a in new_anchors.items():
            old = latest.get(name)
            if old is not None and old.pinned:
                continue
            latest[name] = a
        return self._save_anchors(latest, parents)

    def _anchor_url(self, anchor: Anchor, vcfg: dict[str, Any]) -> tuple[str, str]:
        """锚点片段现在还能不能当参考：返回 (url, 不能用的原因)。

        生成结果的链接会过期（APIMart 文档：24h），过期的链接传给模型只会失败，
        所以到点就当没有 —— 本次用新片段接替，并写回锚点表。
        """
        return self._clip_ref_url(anchor.asset, vcfg)

    async def _rehost_anchors(self, anchors: dict[str, Anchor], vcfg: dict[str, Any]) -> list[str]:
        """锚点片段的链接过期了、本地还有副本：配了托管就重新上传换新链接（声音一点不变）。

        2026-09-23 审查：之前过期（20 小时）就重定一个新锚点，而新锚点那段生成时不带旧声音 ——
        同一个角色的声音基准每天断一次。参考图早就走托管刷新了，锚点一直没走。
        """
        if self.hosting is None or not getattr(self.hosting, "enabled", False):
            return []
        notes: list[str] = []
        for name, a in anchors.items():
            url, why = self._anchor_url(a, vcfg)
            if url or not why or "不存在" in why:
                continue
            try:
                clip = self.store.get(a.asset)
            except KeyError:
                continue
            if local_copy(clip) is None:
                continue
            new_url, err = await self.hosting.ensure_asset(
                self.store, clip, vcfg["ttl_h"], force=True
            )
            notes.append(
                f"音色锚点 {name}：链接过期，已用本地副本重新托管"
                if new_url and not err
                else f"⚠ 音色锚点 {name}：重新托管失败（{err}），这次会重新定锚点"
            )
        return notes

    def _clip_ref_url(self, asset_id: str, vcfg: dict[str, Any]) -> tuple[str, str]:
        """一段视频能不能当参考视频传给模型：返回 (url, 不能用的原因)。

        只认远端 http(s) 链接 —— 本地路径接口直接 400（2026-09-18 实测：拼完第 1 集后
        片段 uri 曾被换成本地路径，锚点传进去就是 E:\\…\\as_xxx.mp4）。
        """
        try:
            asset = self.store.get(asset_id)
        except KeyError:
            return "", "片段资产不存在"
        url = asset.uri or ""
        if not url:
            return "", "片段没有可用链接"
        if not url.startswith(("http://", "https://")):
            return "", "片段链接是本地路径，接口只收 http(s) 链接"
        ttl = vcfg["ttl_h"]
        expired = ttl > 0 and time.time() - asset.created_at > ttl * 3600
        if expired and not self._hosted_fresh(asset):
            return "", f"链接超过 {ttl:g} 小时，可能已过期"
        return url, ""

    def _hosted_fresh(self, asset: Any) -> bool:
        """托管到用户自己存储的链接（gen_params.hosted）不按生成链接的 24h 过期算。"""
        gp = asset.gen_params if isinstance(asset.gen_params, dict) else {}
        hosted = gp.get("hosted")
        if not isinstance(hosted, dict) or hosted.get("url") != (asset.uri or ""):
            return False
        if self.hosting is None:
            return True
        return bool(self.hosting.is_fresh(asset, 0))

    async def _rehost_pack(
        self, pack_id: str, images: dict[str, dict[str, str]]
    ) -> tuple[dict[str, dict[str, str]], list[str]]:
        """参考图链接过期或是本地路径时，配了托管就用本地副本重新上传换新链接（图不变）。"""
        if self.hosting is None or not self.hosting.enabled or not images:
            return images, []
        ttl = self._voice_cfg()["ttl_h"]
        out = {k: dict(v) for k, v in images.items()}
        changed: list[str] = []
        failed: list[str] = []
        for key, entry in out.items():
            try:
                a = self.store.get(str(entry.get("asset") or ""))
            except KeyError:
                continue
            if self.hosting.is_fresh(a, ttl):
                if entry.get("url") != a.uri and a.uri:
                    entry["url"] = a.uri
                    changed.append(key)
                continue
            url, err = await self.hosting.ensure_asset(self.store, a, ttl)
            if url and not err and url != entry.get("url"):
                entry["url"] = url
                changed.append(key)
            elif err:
                failed.append(f"{key}：{err}")
        notes: list[str] = []
        if changed:
            try:
                parents = [pack_id] + list(self.store.get(pack_id).parent_ids[:1])
            except KeyError:
                parents = [pack_id]
            new = self.store.create(
                json.dumps(out, ensure_ascii=False, indent=2),
                type_=AssetType.STORYBOARD,
                summary=f"资产参考图·重新托管·{len(out)}张",
                parents=parents,
                creator="tool:drama_render_assets",
                gen_params={
                    "model": self.image_model,
                    "count": len(out),
                    "rehosted": len(changed),
                },
            )
            notes.append(
                f"↑ {len(changed)} 张参考图链接过期或是本地文件，已用本地副本重新托管"
                f"（{', '.join(changed[:6])}），新参考图包 {new.id}"
            )
        if failed:
            notes.append("⚠ 重新托管失败：" + "；".join(failed[:4]))
        return out, notes

    def _clip_playable(self, asset_id: str) -> bool:
        """片段还能不能拼进成片：本地副本在、或还有远端链接（拼接时下载）。"""
        try:
            a = self.store.get(asset_id)
        except KeyError:
            return False
        if local_copy(a) is not None:
            return True
        return (a.uri or "").startswith(("http://", "https://"))

    def _previous_clips(
        self,
        shots_id: str,
        episode: int,
        pack_id: str = "",
        redo: list[str] | None = None,
        accept: list[str] | None = None,
    ) -> dict[tuple[str, str], str]:
        """这份提示词此前渲过、**过了质检门**的片段：(场次, 镜头范围) → 片段资产 id，最新的优先。

        重跑只补失败/缺的段 —— 之前一段网络错误就要整集重生成，成功的段也跟着再付一遍钱。

        2026-09-23 审查后改的三处：
          · 不再只认整批跑完才写的片段索引：每段视频生成时就带着标签（所属提示词、场次、
            镜头、参考图包），渲到一半被 /stop、超时、崩溃，已付费的段下次照样能复用
          · 参考图包换了（换脸、补了服装）就不复用旧片段 —— 否则新脸永远到不了视频
          · redo 里点名的段强制重渲（之前结果里让人「去掉问题段」，却没有参数能做到）
        """
        cand: dict[tuple[str, str], tuple[int, str]] = {}

        def offer(key: tuple[str, str], seq: int, cid: str) -> None:
            if not cid or not self._clip_playable(cid):
                return
            if key not in cand or seq > cand[key][0]:
                cand[key] = (seq, cid)

        # ① 每段自带标签的片段（新口径）
        for a in self.store.find(type_=AssetType.VIDEO):
            tags = (a.gen_params or {}).get("tags") or {}
            if not isinstance(tags, dict) or tags.get("shots_id") != shots_id:
                continue
            if pack_id and tags.get("pack") and tags.get("pack") != pack_id:
                continue
            key = (str(tags.get("scene") or ""), str(tags.get("name") or ""))
            if not tags.get("accepted"):
                # 没过质检门（带字幕 / 变脸）的段不复用 —— 除非用户看过后点名放行；
                # 被换掉的废片（rejected）永远不复用
                if tags.get("rejected") or not _label_hit(key, accept):
                    continue
                tags["accepted"] = True
                tags["accepted_by"] = "human"
                a.gen_params["tags"] = tags
                self.store.put(a)
            offer(key, a.seq, a.id)

        # ② 老口径：整批跑完落的片段索引
        for a in self.store.find(creator="tool:drama_render_shots"):
            if not a.parent_ids or a.parent_ids[0] != shots_id:
                continue
            if pack_id and len(a.parent_ids) > 1 and a.parent_ids[1] != pack_id:
                continue
            ep = int(a.gen_params.get("episode") or 0)
            if episode and ep not in (0, episode):
                continue
            try:
                rows = json.loads(self.store.content(a.id))
            except (KeyError, json.JSONDecodeError):
                continue
            for row in rows if isinstance(rows, list) else []:
                if row.get("flagged"):
                    continue  # 带着问题标记进索引的段（仍有字幕 / 仍漂移）不复用
                key = (str(row.get("scene") or ""), str(row.get("name") or ""))
                offer(key, a.seq, str(row.get("asset") or ""))

        out = {k: v[1] for k, v in cand.items()}
        for key in list(out):
            if _label_hit(key, redo):
                out.pop(key, None)
        return out

    def _video_retries(self) -> int:
        cfg = getattr(self.catalog, "drama", {}) or {}
        try:
            return max(0, int(cfg.get("video_retries", 1)))
        except (TypeError, ValueError):
            return 1

    # ---------- 字幕门：画面里不许有字幕/文字 ----------

    def _subtitle_cfg(self) -> tuple[bool, int]:
        """(开没开, 发现字幕后允许重生成几次)。没有文本网关（脚本/测试）就关。"""
        if self.gateway is None:
            return False, 0
        cfg = getattr(self.catalog, "drama", {}) or {}
        on = str(cfg.get("subtitle_gate", "true")).strip().lower() in ("1", "true", "yes", "on")
        try:
            n = max(0, int(cfg.get("subtitle_retries", 1)))
        except (TypeError, ValueError):
            n = 1
        return on, n

    async def _invoke_video(self, args: dict[str, Any], retries: int) -> Any:
        """gen_video + 「请求没送到」类失败原样重试。

        只看媒体层给的结构化标记（retryable_failure）：连不上、被 429 拒收才重提；
        轮询失败 / 超时 / 5xx 一律不重提 —— 那时任务多半还在服务端跑、已经计费，
        之前按错误文本里的「网络」「HTTP 5」重提，一个镜头付两份钱（2026-09-23 审查）。
        真要再来一次，媒体网关会先去任务台账取回同一份请求没交付的那个任务。
        """
        r: Any = None
        for attempt in range(retries + 1):
            r = await self.registry_invoke("gen_video", args)
            if r.ok or attempt >= retries or not retryable_failure(r):
                break
            await asyncio.sleep(3)
        return r

    def _gate_cfg(self) -> dict[str, Any]:
        """质检门的公共配置（media_models.yaml drama.*）。

        gate_block      哪些门没过就不许进成片（逗号分隔：subtitle / identity / cut）。
                        默认 subtitle,identity —— 字幕是用户定的最高优先级；人物漂移是
                        用户最在意的一致性问题；镜头超 3 秒只标出来。
        gate_max_regen  一段视频因质检最多重生成几次（所有门合计，每次合并各门的修正）。
                        没配时取各门 *_retries 里最大的那个。
        check_frames    抽几帧给视觉模型看（字幕 / 一致性）。
        """
        cfg = getattr(self.catalog, "drama", {}) or {}
        block = str(cfg.get("gate_block", "subtitle,identity"))
        per_gate: list[int] = []
        for key in ("subtitle_retries", "identity_retries", "cut_retries"):
            try:
                per_gate.append(max(0, int(cfg.get(key, 1))))
            except (TypeError, ValueError):
                per_gate.append(1)
        try:
            regen = max(0, int(cfg["gate_max_regen"]))
        except (KeyError, TypeError, ValueError):
            regen = max(per_gate) if per_gate else 1
        try:
            frames = max(2, min(12, int(cfg.get("check_frames", 6))))
        except (TypeError, ValueError):
            frames = 6
        return {
            "block": {x.strip() for x in block.split(",") if x.strip()},
            "max_regen": regen,
            "frames": frames,
        }

    async def _gen_clip(
        self,
        args: dict[str, Any],
        retries: int,
        sub_gate: bool,
        sub_retries: int,
        id_refs: list[tuple[str, str]] | None = None,
    ) -> tuple[Any, list[str]]:
        """生成一段视频并过质检门。返回 (最终采用的工具结果, 给人看的备注)。

        2026-09-23 审查后重写成**一个循环**（之前是三道门串着各跑各的）：
          · 每一版都**全量**过门：镜头门（ffmpeg，免费）→ 字幕门 → 人物一致性门。
            之前后面的门重生成之后不回查前面的门 —— 为了消字幕重生成的那版可能变了脸，
            照样放行。
          · 几道门同时不过，修正**合并**进同一次重生成（之前每道门拿原始提示词重来，
            前一道门的修正丢了）；总次数封顶（gate_max_regen），各门另有自己的上限。
          · 字幕是用户定的最高优先级：**查不成也算不过**（没本地副本、抽不出帧、判读
            不出结果）—— 之前这些情况只记一句备注就放行，最高优先级的规则只剩提示词一道。
          · 到上限还没过：留问题最少的那一版（一致性分高的优先），备注里标 ⛔ ——
            ⛔ 的段不进成片，人看过后用 accept 放行或 redo 重渲。
          · 被换掉的版本挪进 废弃/ 子目录，主文件名留给最终采用的那版（之前废片占着
            主文件名，按文件名剪片会剪进带字幕 / 变脸的版本）。
        """
        notes: list[str] = []
        r = await self._invoke_video(args, retries)
        if not r.ok:
            return r, notes
        gc = self._gate_cfg()
        cut_on, cut_n, thr, tol = self._cut_cfg()
        id_on, id_n, pass_score = self._identity_cfg()
        id_on = id_on and bool(id_refs)
        limit = float(self.fmt.max_cut_seconds)
        budget = {"subtitle": sub_retries, "identity": id_n, "cut": cut_n}
        base = args["prompt"]
        prompt = base
        # 每一版：(结果, 排序键, 这一版的问题)
        versions: list[tuple[Any, tuple[int, int, int, int], dict[str, Any]]] = []
        regen = 0
        while True:
            issues: dict[str, Any] = {}
            cut_len: float | None = None
            if cut_on:
                longest, why = await self._check_cuts(r.asset_ref, thr)
                if why:
                    notes.append(why)
                elif longest is not None:
                    cut_len = longest
                    if longest > limit + tol:
                        issues["cut"] = longest
            if sub_gate:
                found, why = await self._check_subtitles(r.asset_ref)
                if why:
                    issues["subtitle_unchecked"] = why
                elif found:
                    issues["subtitle"] = True
            score = 10
            if id_on:
                v = await self._check_identity(
                    r.asset_ref, id_refs or [], is_video=True, pass_score=pass_score
                )
                if v.note:
                    notes.append(v.note)
                elif not v.passed:
                    issues["identity"] = v
                    score = int(v.score)
                else:
                    score = int(v.score)
                    notes.append(f"人物一致 {v.score}/10" + ("（重生成后）" if regen else ""))
            blocking = sum(1 for g in issues if _gate_blocks(g, gc["block"]))
            # 质检门判定发事件：复盘时看得到每一版过没过、卡在哪道门（之前只在备注里）
            if self.bus is not None:
                await self.bus.emit(
                    EventType.GATE_VERDICT,
                    clip=str(args.get("summary") or ""),
                    version=regen + 1,
                    passed=not issues,
                    blocking=blocking,
                    gate=",".join(issues) or "all",
                    detail={
                        k: (round(v, 2) if isinstance(v, float) else str(v)[:80])
                        for k, v in issues.items()
                    },
                    asset=r.asset_ref,
                )
            # 排序：拦成片的问题少 → 问题总数少 → 一致性分高 → 越新越好
            versions.append((r, (blocking, len(issues), -score, -len(versions)), issues))
            if regen and not any(k in issues for k in ("subtitle", "subtitle_unchecked")) and (
                versions[-2][2].get("subtitle")
            ):
                notes.append("重生成后无字幕")
            if regen and "cut" not in issues and "cut" in versions[-2][2] and cut_len is not None:
                notes.append(f"重生成后镜头达标（最长 {cut_len:.1f}s）")
            if not issues:
                break
            # 还能针对哪些问题重生成（「查不成」重生成也查不成，不算）
            fixable = [g for g in ("subtitle", "cut", "identity") if g in issues and budget[g] > 0]
            if not fixable or regen >= gc["max_regen"]:
                break
            for g in fixable:
                budget[g] -= 1
            if "subtitle" in fixable:
                notes.append("画面出现字幕，已重生成")
                prompt = no_text_retry(prompt)
            if "cut" in fixable:
                notes.append(f"有镜头长 {issues['cut']:.1f}s，超过 {limit:g}s，已重生成")
                prompt = fast_cut_retry(prompt, issues["cut"], limit)
            if "identity" in fixable:
                v = issues["identity"]
                why = "；".join(v.issues)[:120] or "与参考图不符"
                notes.append(f"人物与参考图不符（{v.score}/10：{why}），已重生成")
                prompt = identity_retry_prompt(prompt, v.issues)
            r2 = await self._invoke_video({**args, "prompt": prompt}, retries)
            regen += 1
            if not r2.ok:
                notes.append(f"重生成失败：{(r2.error or '')[:60]}，保留已有的版本")
                break
            r = r2

        # 选问题最少的一版（一致性分高的优先）；其余挪进 废弃/，主文件名留给它
        chosen, _, final = min(versions, key=lambda x: x[1])
        for other, _, why in versions:
            if other is not chosen and other.asset_ref:
                self._retire_version(other.asset_ref, "、".join(why) or "被新版本替换")
        if len(versions) > 1 and chosen.asset_ref:
            self._reclaim_name(chosen.asset_ref, versions[0][0].asset_ref)

        if "subtitle" in final:
            mark = "⛔ " if _gate_blocks("subtitle", gc["block"]) else "⚠ "
            notes.append(mark + "仍有字幕/文字")
        if "subtitle_unchecked" in final:
            notes.append(
                ("⛔ " if _gate_blocks("subtitle_unchecked", gc["block"]) else "⚠ ")
                + f"字幕没查成（{final['subtitle_unchecked']}）"
            )
        if "cut" in final:
            notes.append(
                ("⛔ " if _gate_blocks("cut", gc["block"]) else "⚠ ")
                + f"仍有超过 {limit:g} 秒的镜头（最长 {final['cut']:.1f}s）"
            )
        if "identity" in final:
            v = final["identity"]
            why = "；".join(v.issues)[:120] or "与参考图不符"
            notes.append(
                ("⛔ " if _gate_blocks("identity", gc["block"]) else "⚠ ")
                + f"仍与参考图不符（{v.score}/10：{why}），保留分最高的一版"
            )
        return chosen, notes

    def _retire_version(self, asset_id: str, reason: str) -> None:
        """被质检门换掉的那一版：标签记下原因（不复用、不进成片），本地文件挪进 废弃/。"""
        try:
            a = self.store.get(asset_id)
        except KeyError:
            return
        tags = dict((a.gen_params or {}).get("tags") or {})
        tags["accepted"] = False
        tags["rejected"] = reason[:120]
        a.gen_params["tags"] = tags
        a.status = AssetStatus.REJECTED
        a.status_note = f"质检门：{reason[:120]}"
        local = str(a.gen_params.get("local") or "")
        if local and Path(local).exists():
            p = Path(local)
            try:
                dest_dir = p.parent / "废弃"
                dest_dir.mkdir(exist_ok=True)
                dest = unique_path(dest_dir, p.name)
                shutil.move(str(p), str(dest))
                a.gen_params["local"] = str(dest)
            except OSError:
                pass
        self.store.put(a)

    def _reclaim_name(self, asset_id: str, first_id: str | None) -> None:
        """最终采用的版本拿回第一版的文件名（重渲时它被命名成 -v2，第一版挪走后名字空出来了）。"""
        try:
            a = self.store.get(asset_id)
            first = self.store.get(first_id) if first_id else None
        except KeyError:
            return
        mine = Path(str(a.gen_params.get("local") or ""))
        if first is None or not mine.exists():
            return
        want = Path(str(first.gen_params.get("local") or "")).name
        if not want or first.id == a.id:
            return
        target = mine.parent / want
        if target.exists() or target == mine:
            return
        # 第一版原来在同一个目录（挪进 废弃/ 之前）：名字空出来了才拿
        try:
            mine.rename(target)
            a.gen_params["local"] = str(target)
            self.store.put(a)
        except OSError:
            pass

    # ---------- 镜头门：每个镜头不超过 3 秒（2026-09-20 用户定的硬性要求） ----------

    def _cut_cfg(self) -> tuple[bool, int, float, float]:
        """(开没开, 超时长后允许重生成几次, 场景切换阈值, 容差秒)。上限本身在 self.fmt。"""
        cfg = getattr(self.catalog, "drama", {}) or {}
        on = str(cfg.get("cut_gate", "true")).strip().lower() in ("1", "true", "yes", "on")
        try:
            n = max(0, int(cfg.get("cut_retries", 1)))
        except (TypeError, ValueError):
            n = 1
        try:
            thr = min(0.95, max(0.05, float(cfg.get("cut_scene_threshold", 0.3))))
        except (TypeError, ValueError):
            thr = 0.3
        try:
            tol = max(0.0, float(cfg.get("cut_tolerance", 0.5)))
        except (TypeError, ValueError):
            tol = 0.5
        return on, n, thr, tol

    async def _check_cuts(self, asset_id: str, threshold: float) -> tuple[float | None, str]:
        """量这段视频里最长的镜头有几秒。返回 (秒数, 检查不了的原因)；没本地副本静默跳过。"""
        local = self._local_video(asset_id)
        if local is None:
            return None, ""
        if not ffmpeg.have_ffmpeg():
            return None, "没有 ffmpeg，跳过镜头时长检查"
        info = await ffmpeg.probe(local)
        if info.duration <= 0:
            return None, "读不到片段时长，跳过镜头时长检查"
        times, err = await ffmpeg.scene_cuts(local, threshold)
        if err:
            return None, f"镜头时长检查失败：{err[:60]}"
        return longest_shot(info.duration, times), ""

    # ---------- 引用门：没有引用成功的镜头不许生成（2026-09-20 用户定的规则） ----------

    def _known_names(self) -> list[str]:
        """最近几套资产库里的角色名与服装 ID ——
        gen_video 的参考图门靠它认出「这是短剧人物镜头」。"""
        libs = self.store.find(creator="tool:drama_assets")[:3]
        key = tuple(a.id for a in libs)
        cached = getattr(self, "_names_cache", None)
        if cached and cached[0] == key:
            return cached[1]
        names: list[str] = []
        for a in libs:
            try:
                lib, err = parse_assets(self.store.content(a.id))
            except KeyError:
                continue
            if err:
                continue
            for c in lib.characters:
                names.append(c.name)
                names += [cos.name for cos in c.costumes]
        names = [n for n in dict.fromkeys(names) if len(n) >= 2]
        self._names_cache = (key, names)
        return names

    def reference_guard(self, prompt: str, summary: str = "") -> str:
        """gen_video / gen_videos 没带参考图时的拦截原因；空串 = 放行。

        真实事故（2026-09-20）：4 段渲染失败（参考图超上限 400 / 内容策略 / 超时）后，模型自己把
        提示词改写成英文、用 gen_videos 无参考生成补上，再拼成片 —— 后半段人物全变脸。
        规则：提示词/摘要带集号镜号，或提到资产库里的角色/服装，就是短剧人物镜头，必须走
        drama_render_shots 或显式带参考图。
        """
        text = f"{summary or ''}\n{prompt or ''}"
        reasons: list[str] = []
        if re.search(r"第\s*\d+\s*集", text) or re.search(r"(?<![A-Za-z])镜\s*\d+", text):
            reasons.append("提示词/摘要里带集号或镜号，是短剧分镜")
        hit = [n for n in self._known_names() if n in text]
        if hit:
            reasons.append(f"提到了资产库里的角色/服装：{'、'.join(hit[:4])}")
        if not reasons:
            return ""
        how = (
            "无参考生成的人物必然变脸，用户明确禁止。正确做法：失败/缺的段用 "
            "drama_render_shots(shots_id=…, rendered_id=…, reuse=true) 补渲，它会自动带参考图、"
            "锁音色并做一致性校验；要改镜头内容先改分镜提示词资产（重跑 drama_shots 或 "
            "save_draft 新版本）再渲染。"
        )
        if hit:
            # 点了角色名就不是空镜：allow_no_refs 也放不过（⛔ 开头 = 硬拦，见 media.gen_video）。
            # 2026-09-23 审查：之前一个 allow_no_refs=true 就能绕过整道门，拦截文案还在教模型这么传
            return f"⛔ 已拦截：提示词里有角色，不能无参考生成（{'；'.join(reasons)}）—— " + how
        return (
            f"已拦截：没有参考图不能直接生成短剧镜头（{'；'.join(reasons)}）—— " + how
            + "只有确定是没有任何人物的空镜，才可以传 allow_no_refs=true。"
        )

    def _ref_probe_on(self) -> bool:
        cfg = getattr(self.catalog, "drama", {}) or {}
        return str(cfg.get("ref_probe", "false")).strip().lower() in ("1", "true", "yes", "on")

    def _max_refs(self) -> int:
        """视频模型的参考素材上限（media_models.yaml max_refs）。0 = 不限 / 不知道。"""
        get = getattr(self.catalog, "get", None)
        if not callable(get):
            return 0
        try:
            spec = get(MediaKind.VIDEO, self.video_model)
        except Exception:  # noqa: BLE001
            return 0
        return int(getattr(spec, "max_refs", 0) or 0)

    def _max_duration(self) -> int:
        """锁定的视频模型单段最长几秒（media_models.yaml max_duration）。0 = 不限 / 不知道。"""
        get = getattr(self.catalog, "get", None)
        if not callable(get):
            return 0
        try:
            spec = get(MediaKind.VIDEO, self.video_model)
        except Exception:  # noqa: BLE001
            return 0
        return int(getattr(spec, "max_duration", 0) or 0)

    async def _probe_urls(self, urls: list[str]) -> dict[str, str]:
        """参考图链接现在还能不能访问：url → 失败原因（"" = 可访问）。

        生成链接约 24h 失效，按时间猜不准；花钱之前发一次 HEAD 最稳。4xx 判失效；
        网络不通只算"没验上"（原因以 ? 开头），由调用方决定提示还是拦。
        """
        out: dict[str, str] = {}
        targets = [u for u in dict.fromkeys(urls) if u.startswith(("http://", "https://"))]
        if not targets:
            return out
        try:
            async with httpx.AsyncClient(
                proxy=default_proxy(), timeout=8, follow_redirects=True
            ) as client:
                sem = asyncio.Semaphore(6)

                async def one(u: str) -> None:
                    async with sem:
                        try:
                            resp = await client.head(u)
                            if resp.status_code in (403, 405, 501):
                                resp = await client.get(u, headers={"Range": "bytes=0-0"})
                            out[u] = (
                                "" if resp.status_code < 400 else f"HTTP {resp.status_code}"
                            )
                        except Exception as e:  # noqa: BLE001
                            out[u] = f"?{type(e).__name__}"

                await asyncio.gather(*(one(u) for u in targets))
        except Exception as e:  # noqa: BLE001
            return {u: f"?{type(e).__name__}" for u in targets}
        return out

    # ---------- 人物一致性门：生成结果与参考图是不是同一个人 ----------

    def _identity_cfg(self) -> tuple[bool, int, int]:
        """(开没开, 不合格后允许重生成几次, 通过分)。没有文本网关就关。"""
        if self.gateway is None:
            return False, 0, 7
        cfg = getattr(self.catalog, "drama", {}) or {}
        on = str(cfg.get("identity_gate", "true")).strip().lower() in ("1", "true", "yes", "on")
        try:
            n = max(0, int(cfg.get("identity_retries", 1)))
        except (TypeError, ValueError):
            n = 1
        try:
            score = max(1, min(10, int(cfg.get("identity_pass_score", 7))))
        except (TypeError, ValueError):
            score = 7
        return on, n, score

    def _asset_by_uri(self, url: str) -> Any:
        for a in self.store.all():
            if a.uri == url:
                return a
        return None

    def _ref_payload(self, ref: str) -> str:
        """参考图 → 给视觉模型看的 data URL：优先本地副本（远端链接会过期），没有就原样给链接。"""
        if ref.startswith("as_"):
            return self._image_payload(ref, "")
        a = self._asset_by_uri(ref)
        if a is not None:
            return self._image_payload(a.id, ref)
        return ref

    async def _check_identity(
        self, target: str, refs: list[tuple[str, str]], is_video: bool, pass_score: int
    ) -> Any:
        """把参考图和生成结果一起给视觉模型，判是不是同一个人。做不了就 note 说明，按通过处理。"""
        from ..drama.identity import IdentityVerdict

        if self.gateway is None:
            return IdentityVerdict(note="没有文本网关，跳过一致性校验")
        ref_parts = [(label, self._ref_payload(u)) for label, u in refs]
        ref_parts = [(lb, u) for lb, u in ref_parts if u]
        if not ref_parts:
            return IdentityVerdict(note="参考图拿不到（没本地副本也没链接），跳过一致性校验")
        if is_video:
            local = self._local_video(target)
            if local is None:
                return IdentityVerdict(note="片段没有本地副本，跳过一致性校验")
            frames = await self._frames_of(local)
            if not frames:
                return IdentityVerdict(note="抽不出画面帧（检查 ffmpeg），跳过一致性校验")
            targets = ["data:image/jpeg;base64," + base64.b64encode(b).decode() for b in frames]
        else:
            payload = self._image_payload(target, "")
            if not payload:
                return IdentityVerdict(note="生成图没有本地副本，跳过一致性校验")
            targets = [payload]
        try:
            resp = await self.gateway.chat(
                REALISM_ROLE, identity_check_messages(ref_parts, targets, is_video)
            )
        except KeyError:
            return IdentityVerdict(note=f"models.yaml 没配 {REALISM_ROLE} 角色，跳过一致性校验")
        except Exception as e:  # noqa: BLE001
            return IdentityVerdict(note=f"一致性校验调用失败：{type(e).__name__}")
        return parse_identity_verdict(resp.text, pass_score)

    def _stale_refs(
        self, resolved: dict[str, Any], images: dict[str, dict[str, str]], vcfg: dict[str, Any]
    ) -> list[str]:
        """参考图链接可能已过期的引用名（生成结果链接 24h 失效，过期后模型拿不到参考图）。"""
        ttl = vcfg["ttl_h"]
        if ttl <= 0:
            return []
        out: list[str] = []
        for m in resolved.values():
            if m is None:
                continue
            entry = images.get(m.key) or {}
            try:
                a = self.store.get(str(entry.get("asset") or ""))
            except KeyError:
                continue
            if self._hosted_fresh(a):
                continue
            remote = (a.uri or "").startswith(("http://", "https://"))
            if remote and time.time() - a.created_at > ttl * 3600 and m.key not in out:
                out.append(m.key)
        return out

    def _mark_clip(self, asset_id: str, notes: list[str]) -> bool:
        """一段过完质检门之后在它的标签上记结论：accepted 才能被下次复用、进成片。

        带着「仍有字幕」（用户定的最高优先级）或检查没做成的段不算 accepted。
        返回是否 accepted。
        """
        accepted = not any(_blocking_note(n) for n in notes)
        try:
            a = self.store.get(asset_id)
        except KeyError:
            return accepted
        tags = dict((a.gen_params or {}).get("tags") or {})
        tags["accepted"] = accepted
        if notes:
            tags["notes"] = list(notes)[-6:]
        a.gen_params["tags"] = tags
        self.store.put(a)
        return accepted

    def _local_video(self, asset_id: str) -> Path | None:
        try:
            a = self.store.get(asset_id)
        except KeyError:
            return None
        return local_copy(a)

    async def _frames_of(self, video: Path, count: int = 4) -> list[bytes]:
        """抽几帧 jpg 给视觉模型看。抽不出来返回空。

        宽 512（之前 384）：字幕条上的小字在 384 宽的低清帧里经常认不出来。"""
        tmp = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="subcheck_"))
        try:
            paths = await ffmpeg.extract_frames(video, tmp, count=count, width=512)
            return await asyncio.to_thread(lambda: [p.read_bytes() for p in paths])
        except Exception:  # noqa: BLE001
            return []
        finally:
            await asyncio.to_thread(shutil.rmtree, tmp, True)

    async def _check_subtitles(self, asset_id: str) -> tuple[bool, str]:
        """看几帧画面里有没有字幕/文字。返回 (发现了, 检查不了的原因)。"""
        local = self._local_video(asset_id)
        if local is None:
            return False, "片段没有本地副本，跳过字幕检查"
        frames = await self._frames_of(local, self._gate_cfg()["frames"])
        if not frames:
            return False, "抽不出画面帧（检查 ffmpeg），跳过字幕检查"
        urls = ["data:image/jpeg;base64," + base64.b64encode(b).decode() for b in frames]
        note = ""
        for _ in range(2):  # 判读输出不是 JSON / 缺字段：再问一次，还不行才算查不成
            try:
                resp = await self.gateway.chat(REALISM_ROLE, subtitle_check_messages(urls))
            except KeyError:
                return False, f"models.yaml 没配 {REALISM_ROLE} 角色，跳过字幕检查"
            except Exception as e:  # noqa: BLE001
                return False, f"字幕检查调用失败：{type(e).__name__}"
            found, note = parse_subtitle_verdict(resp.text)
            if found is not None:
                return found, ""
        return False, note or "字幕检查没给出结论"

    def _library_for(self, shots_id: str, pack_id: str) -> Any:
        """渲染时找回资产库（音色卡在里面）：分镜提示词的 parents[1]，或参考图包的 parents[0]。"""
        cands: list[str] = []
        try:
            cands += self.store.get(shots_id).parent_ids[1:]
        except KeyError:
            pass
        if pack_id:
            try:
                cands += self.store.get(pack_id).parent_ids[:1]
            except KeyError:
                pass
        for aid in cands:
            try:
                lib, err = parse_assets(self.store.content(aid))
            except KeyError:
                continue
            if not err and lib.characters:
                return lib
        return None

    async def _fn_drama_refresh_refs(
        self, rendered_id: str = "", only_stale: bool = True
    ) -> ToolResult:
        if self.registry is None:
            return ToolResult(ok=False, error="没接工具注册表，生不了图")
        if rendered_id:
            try:
                pack_asset = self.store.get(rendered_id)
                images = _load_pack(self.store.content(rendered_id))
            except KeyError as e:
                return ToolResult(ok=False, error=f"取不到参考图包：{e}")
        else:
            packs = self.store.find(creator="tool:drama_render_assets")
            if not packs:
                return ToolResult(ok=False, error="还没有参考图包（先 drama_render_assets）")
            pack_asset = packs[0]
            images = _load_pack(self.store.content(pack_asset.id))
        if not images:
            return ToolResult(ok=False, error=f"{pack_asset.id} 不是参考图包")

        ttl = self._voice_cfg()["ttl_h"]
        new_images: dict[str, dict[str, str]] = {}
        lines: list[str] = []
        refreshed = 0
        for key, entry in images.items():
            new_images[key] = dict(entry)
            try:
                a = self.store.get(str(entry.get("asset") or ""))
            except KeyError:
                lines.append(f"  ✗ {key}：原资产不存在")
                continue
            stale = (
                not self._hosted_fresh(a) and ttl > 0 and time.time() - a.created_at > ttl * 3600
            )
            if only_stale and not stale:
                continue
            # 只走托管：把本地副本重新上传 —— 链接换新，图一个像素都不变。
            # 2026-09-23 审查后去掉了「托管不了就让生图模型复刻一张」的退路：生成接口只收
            # 公网链接，本地图的 data URL 在提交前就会被拒（这条路其实从没走通过）；
            # 真走通了也是重新生成一张不过任何质检门的图，脸可能变。
            if self.hosting is None or not self.hosting.enabled:
                lines.append(
                    f"  ✗ {key}：没配素材托管，刷不了 —— 生成接口只收公网链接，"
                    "先按 hosting_status 配好图床（config/hosting.yaml）"
                )
                continue
            if local_copy(a) is None:
                lines.append(f"  ✗ {key}：没有本地副本，刷不了（只能重新渲这张参考图）")
                continue
            url, err = await self.hosting.ensure_asset(self.store, a, ttl, force=True)
            if url and not err:
                new_images[key].update({"url": url})
                refreshed += 1
                lines.append(f"  ✓ {key} 重新托管 → {url[:70]}")
                continue
            lines.append(f"  ✗ {key}：重新托管失败（{err}）")
        if not refreshed:
            detail = "\n".join(lines) if lines else "  没有需要刷新的（链接都还新鲜）"
            return ToolResult(ok=False, error="一张都没刷新：\n" + detail)
        asset = self.store.create(
            json.dumps(new_images, ensure_ascii=False, indent=2),
            type_=AssetType.STORYBOARD,
            summary=f"资产参考图·刷新·{len(new_images)}张",
            parents=[pack_asset.id] + list(pack_asset.parent_ids[:1]),
            creator="tool:drama_render_assets",
            gen_params={
                "model": self.image_model,
                "count": len(new_images),
                "refreshed": refreshed,
            },
        )
        return ToolResult(
            content=f"刷新了 {refreshed} 张参考图的链接：\n" + "\n".join(lines)
            + f"\n\n新参考图包 {asset.id}（渲视频时用它）",
            asset_ref=asset.id,
        )

    async def _fn_drama_use_local_ref(
        self,
        name: str,
        path: str = "",
        asset_id: str = "",
        assets_id: str = "",
        kind: str = "auto",
    ) -> ToolResult:
        if not path and not asset_id:
            return ToolResult(ok=False, error="给 path（本地图片）或 asset_id（已登记的图片资产）")
        if self.hosting is None or not self.hosting.enabled:
            return ToolResult(
                ok=False,
                error="没有配置素材托管：生成接口只收公网链接，本地图当不了参考。"
                "在 config/hosting.yaml 配 imghost / command / s3 / http_put 之一后重启"
                "（hosting_status 可查）。",
            )
        lib_id = assets_id
        if not lib_id:
            libs = self.store.find(creator="tool:drama_assets")
            if not libs:
                return ToolResult(ok=False, error="还没有资产库（先 drama_assets）")
            lib_id = libs[0].id
        try:
            lib, err = parse_assets(self.store.content(lib_id))
        except KeyError as e:
            return ToolResult(ok=False, error=f"取不到资产库：{e}")
        if err:
            return ToolResult(ok=False, error=f"资产库读不出来：{err}")
        name = name.strip()
        if kind == "auto":
            if any(c.name == name for c in lib.characters):
                kind = "角色"
            elif any(cos.name == name for c in lib.characters for cos in c.costumes):
                kind = "服装"
            elif name in lib.scene_names():
                kind = "场景"
            elif any(p.name == name for p in lib.props):
                kind = "道具"
            else:
                names = ", ".join(sorted(lib.all_names()))[:400]
                return ToolResult(
                    ok=False,
                    error=f"资产库里没有「{name}」。可用的名字：{names}",
                )
        if asset_id:
            try:
                a = self.store.get(asset_id)
            except KeyError:
                return ToolResult(ok=False, error=f"没有资产 {asset_id}")
        else:
            if self.files is None:
                return ToolResult(ok=False, error="没接文件工具，登记不了本地文件（传 asset_id）")
            args = {"path": path, "summary": f"{kind}·{name}·用户素材"}
            r = await self.files.invoke("fs_import", args)
            if not r.ok:
                return r
            a = self.store.get(r.asset_ref or "")
        if a.type is not AssetType.IMAGE:
            return ToolResult(ok=False, error=f"{a.id} 是 {a.type.value}，参考图要图片")
        url, err = await self.hosting.ensure_asset(self.store, a, 0)
        if not url:
            return ToolResult(ok=False, error=f"上传失败：{err}")
        prev_id, prev = self._previous_pack(lib_id)
        images = {k: dict(v) for k, v in prev.items()}
        images[name] = {"asset": a.id, "url": url, "kind": kind, "source": "user"}
        dropped: list[str] = []
        if kind == "角色":
            # 换了脸：旧服装图是按旧脸生成的，留着渲出来就是两张脸（2026-09-23 审查：换脸不传播）
            char = next((c for c in lib.characters if c.name == name), None)
            for cos in (char.costumes if char else []):
                if images.pop(cos.name, None) is not None:
                    dropped.append(cos.name)
        new = self.store.create(
            json.dumps(images, ensure_ascii=False, indent=2),
            type_=AssetType.STORYBOARD,
            summary=f"资产参考图·{len(images)}张",
            parents=[lib_id] + ([prev_id] if prev_id else []),
            creator="tool:drama_render_assets",
            gen_params={"model": self.image_model, "count": len(images), "user_refs": 1},
        )
        redo = ""
        if dropped:
            redo = (
                f"\n该角色有 {len(dropped)} 套服装图是按旧脸生成的，已从包里拿掉（渲视频时先退回"
                f"主形象）：{'、'.join(dropped)}。跑 drama_render_assets(assets_id=\"{lib_id}\", "
                "only=\"costumes\", reuse=true) 按新脸补生成。"
            )
        return ToolResult(
            content=(
                f"已把 {path or asset_id} 作为「{name}」的{kind}参考放进参考图包 {new.id}。"
                f"之后 drama_render_assets(reuse=true) 沿用它不再生成该项；"
                f"drama_render_shots 引用 ({name}) 时自动带上并做人物一致性校验。{redo}"
            ),
            asset_ref=new.id,
        )

    # ---------- 面容审查：同一个角色只留一张脸（2026-09-22 用户提的） ----------

    def _face_cfg(self) -> tuple[bool, int]:
        """(渲完参考图后自动查?, 判同一人的及格分)。见 media_models.yaml drama 段。"""
        cfg = getattr(self.catalog, "drama", {}) or {}
        on = str(cfg.get("face_audit", "true")).strip().lower() not in ("false", "0", "off", "no")
        try:
            score = int(cfg.get("identity_pass_score", 7))
        except (TypeError, ValueError):
            score = 7
        return on, score

    def _collect_faces(
        self, lib: Any, pack: dict[str, dict[str, str]], deep: bool = False
    ) -> list[FaceCandidate]:
        """把"疑似某个角色"的图归拢：参考图包在用的 + 资产库里摘要带角色名的图片。

        摘要匹配是故意放宽的 —— 换模型试出来的图摘要五花八门（「阿蛛·模型试探·qwen」），
        按"摘要里出现角色名"才捞得到；同名更长的角色靠 matches_character 排除。

        默认**只看主形象层**：服装图在生成时已经过了一道人物一致性门（_identity_gate_image），
        再查一遍是重复花视觉模型的钱。deep=true 才把服装图也拉进来复核。
        效果是：干净的工作区里每个角色只有 1 个候选 → 一次视觉调用都不发；
        只有真的攒出了多张脸才开始花钱。
        """
        names = [c.name for c in lib.characters]
        want_kinds = ("角色", "服装") if deep else ("角色",)
        seen: set[str] = set()
        out: list[FaceCandidate] = []
        for key, entry in pack.items():
            aid = str(entry.get("asset") or "")
            kind = str(entry.get("kind") or "")
            if not aid or kind not in want_kinds:
                continue
            who = next((n for n in names if matches_character(key, n, names)), "")
            if not who:
                continue
            seen.add(aid)
            out.append(FaceCandidate(
                character=who, asset_id=aid, summary=key,
                url=str(entry.get("url") or ""), local=self._local_of(aid),
                source="user" if entry.get("source") == "user" else "pack",
                pack_key=key, kind=kind, seq=self._seq_of(aid),
            ))
        # 散落在资产库里的同名图。不 deep 时要把服装图也挡在外面 —— 它们不在上面的
        # 包循环里被收，却会从"摘要含角色名"这条路溜进来，deep 开关就白设了
        costumes = {cos.name for c in lib.characters for cos in c.costumes}
        # 场景 / 道具图不是人物图：「场景·陆离的公寓」摘要里也有角色名，之前被当成候选，
        # 判不一致后场景条目的 url 被换成了人像（2026-09-23 审查）
        places = set(lib.scene_names()) | {p.name for p in lib.props}
        for a in self.store.find(type_=AssetType.IMAGE, newest_first=False):
            if a.id in seen:
                continue
            s = a.summary or ""
            if s.startswith(("场景·", "道具·")) or any(n and n in s for n in places):
                continue
            if not deep and (s.startswith("服装·") or any(n and n in s for n in costumes)):
                continue
            who = next((n for n in names if matches_character(s, n, names)), "")
            if not who:
                continue
            out.append(FaceCandidate(
                character=who, asset_id=a.id, summary=s,
                url=str(a.uri or ""), local=self._local_of(a.id),
                source="asset", kind="", seq=a.seq,
            ))
        return out

    def _seq_of(self, asset_id: str) -> int:
        try:
            return self.store.get(asset_id).seq
        except KeyError:
            return 0

    def _local_of(self, asset_id: str) -> str:
        try:
            a = self.store.get(asset_id)
        except KeyError:
            return ""
        p = str((a.gen_params or {}).get("local") or "")
        return p if p and Path(p).exists() else ""

    async def _audit_one(self, who: str, cands: list[FaceCandidate], pass_score: int) -> FaceAudit:
        """一个角色：定基准 → 逐张比对 → 分组裁决。只花 N-1 次视觉调用。"""
        base, why = pick_baseline(cands)
        if base is None:
            return FaceAudit(character=who, skipped="没有候选")
        if len(cands) == 1:
            return FaceAudit(character=who, keeper=base, why=why)
        if not base.local and not base.url:
            return FaceAudit(character=who, keeper=base, why=why,
                             skipped="基准图既没有本地副本也没有链接，比不了")
        refs = [(f"角色「{who}」本人", base.local or base.url)]
        verdicts: dict[str, tuple[bool, int, list[str]]] = {}
        notes: list[str] = []
        for c in cands:
            if c.asset_id == base.asset_id:
                continue
            v = await self._check_identity(c.asset_id, refs, is_video=False, pass_score=pass_score)
            if getattr(v, "note", ""):
                notes.append(f"{c.asset_id}：{v.note}")
                continue  # 比不了的按"没判过"处理，不冤枉它
            verdicts[c.asset_id] = (bool(v.passed), int(v.score), list(v.issues))
        a = decide(who, cands, verdicts, base, why)
        if notes and not verdicts:
            a.skipped = "；".join(notes[:2])
        return a

    def _retire(self, audits: list[FaceAudit]) -> tuple[list[str], list[str]]:
        """把弃用的图移进 trash 并在资产上留记号。从不真删。返回 (动作, 失败)。"""
        done: list[str] = []
        failed: list[str] = []
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for a in audits:
            if a.ambiguous or not a.drift or a.keeper is None:
                continue
            for c in a.drift:
                try:
                    asset = self.store.get(c.asset_id)
                except KeyError:
                    continue
                asset.gen_params["superseded_by"] = a.keeper.asset_id
                asset.gen_params["retired_at"] = stamp
                # 打状态：之后按项目查图（面容审查、找参考）都不会再取到它（之前 superseded_by
                # 写了没有任何读取点）
                asset.status = AssetStatus.SUPERSEDED
                asset.status_note = f"面容审查：{a.character} 以 {a.keeper.asset_id} 为准"
                self.store.put(asset)
                if not c.local:
                    done.append(f"{a.character} {c.asset_id}（只标记，没有本地文件）")
                    continue
                src = Path(c.local)
                dest = self._trash_root() / f"faces-{stamp}" / a.character / src.name
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dest))
                    asset.gen_params["local"] = str(dest)
                    self.store.put(asset)
                    done.append(f"{a.character} {src.name} → {dest.parent}")
                except OSError as e:
                    failed.append(f"{a.character} {src.name}：{e}")
        return done, failed

    def _trash_root(self) -> Path:
        """回收目录跟 FileFunctions 用同一个（workspace/trash），没接文件工具就退到 workspace。"""
        t = getattr(self.files, "trash", None)
        return Path(t) if t else workspace_root() / "trash"

    async def _fn_drama_audit_faces(
        self, assets_id: str = "", character: str = "", apply: bool = True, deep: bool = False
    ) -> ToolResult:
        lib_id = assets_id
        if not lib_id:
            libs = self.store.find(creator="tool:drama_assets")
            if not libs:
                return ToolResult(ok=False, error="还没有资产库（先 drama_assets）")
            lib_id = libs[0].id
        try:
            lib, err = parse_assets(self.store.content(lib_id))
        except KeyError as e:
            return ToolResult(ok=False, error=f"取不到资产库：{e}")
        if err:
            return ToolResult(ok=False, error=f"资产库读不出来：{err}")
        _, pass_score = self._face_cfg()
        pack_id, pack = self._previous_pack(lib_id)
        names = [c.name for c in lib.characters]
        if character and character not in names:
            who = "、".join(names)
            return ToolResult(ok=False, error=f"资产库里没有角色「{character}」：{who}")
        want = [character] if character else names
        grouped = group_candidates(self._collect_faces(lib, pack, deep), want)
        if not grouped:
            return ToolResult(content="没有找到任何角色图，没什么可查的。")

        audits: list[FaceAudit] = []
        for who in names:
            if who in grouped:
                audits.append(await self._audit_one(who, grouped[who], pass_score))

        head = f"面容审查（资产库 {lib_id}，{len(grouped)} 个角色）：\n"
        body = "\n".join(audit_lines(audits))
        bad = [a for a in audits if a.ambiguous]
        if bad:
            return ToolResult(
                content=head + body + "\n\n有定不了的角色，已暂停等你拍板。",
                suspend=True,
                suspend_payload={
                    "question": conflict_question(audits),
                    "stage": FACE_CONFLICT_STAGE,
                    "target": FACE_CONFLICT_STAGE,
                    "assets": [
                        c.asset_id for a in bad
                        for c in ([a.keeper] if a.keeper else []) + a.drift
                    ],
                    "major": True,  # /auto 也停：留哪张脸只能人定
                },
            )
        conflicted = [a for a in audits if a.conflicted]
        if not conflicted:
            return ToolResult(content=head + body + "\n\n没有冲突，不用清理。")
        if not apply:
            return ToolResult(content=head + body + "\n\n（apply=false，只出报告没动文件）")
        done, failed = self._retire(audits)
        tail = "\n\n已清理（移进回收目录，可找回）：\n  " + "\n  ".join(done) if done else ""
        if failed:
            tail += "\n\n没能移动：\n  " + "\n  ".join(failed)
        new_pack, dropped = self._repin_pack(lib_id, pack_id, pack, audits, lib)
        if new_pack:
            tail += f"\n\n参考图包已更新为 {new_pack}（冲突项换成保留的那张）"
        if dropped:
            tail += (
                f"\n主形象换了，按旧脸生成的 {len(dropped)} 套服装图已从包里拿掉："
                f"{'、'.join(dropped)}。"
                "跑 drama_render_assets(only=\"costumes\", reuse=true) 补生成"
            )
        return ToolResult(content=head + body + tail, asset_ref=new_pack or None)

    def _repin_pack(
        self,
        lib_id: str,
        pack_id: str,
        pack: dict[str, dict[str, str]],
        audits: list[FaceAudit],
        lib: Any = None,
    ) -> tuple[str, list[str]]:
        """参考图包里若指向被弃用的图，换成保留的那张。没有要换的就不建新版。
        返回 (新包 id, 因为主形象换了而拿掉的服装)。"""
        if not pack:
            return "", []
        retired = {c.asset_id: a for a in audits if not a.ambiguous for c in a.drift}
        images = {k: dict(v) for k, v in pack.items()}
        changed = False
        swapped: dict[str, str] = {}  # 换了主形象的角色 → 新主形象资产
        for key, entry in images.items():
            a = retired.get(str(entry.get("asset") or ""))
            if a is None or a.keeper is None:
                continue
            entry["asset"] = a.keeper.asset_id
            entry["url"] = a.keeper.url
            entry["repinned_from"] = key
            if str(entry.get("kind") or "") == "角色":
                swapped[key] = a.keeper.asset_id
            changed = True
        if not changed:
            return "", []
        # 主形象换了：按旧脸生成的服装图作废（记着 portrait 且就是新脸的留下）
        owner = {cos.name: c.name for c in lib.characters for cos in c.costumes} if lib else {}
        dropped: list[str] = []
        for key in list(images):
            who = owner.get(key, "")
            if who in swapped and str(images[key].get("portrait") or "") != swapped[who]:
                images.pop(key)
                dropped.append(key)
        asset = self.store.create(
            json.dumps(images, ensure_ascii=False, indent=2),
            type_=AssetType.STORYBOARD,
            summary=f"资产参考图·{len(images)}张",
            parents=[lib_id] + ([pack_id] if pack_id else []),
            creator="tool:drama_render_assets",
            gen_params={"model": self.image_model, "count": len(images), "face_audit": 1},
        )
        return asset.id, dropped

    async def _fn_drama_voice_anchors(
        self, action: str = "list", character: str = "", clip_id: str = ""
    ) -> ToolResult:
        anchors = self._load_anchors()
        vcfg = self._voice_cfg()
        if action == "list":
            state = "开" if vcfg["enabled"] else "关（只锁音色卡文字，不传参考视频）"
            lines = [f"音色锚点（参考视频模式：{state}，链接有效期 {vcfg['ttl_h']:g}h）："]
            lines.append(anchors_table(anchors))
            for a in anchors.values():
                _, why = self._anchor_url(a, vcfg)
                if why:
                    lines.append(f"  ⚠ {a.character}：{why}，下次渲染会用新片段接替")
            return ToolResult(content="\n".join(lines))
        if not character:
            return ToolResult(ok=False, error="pin / clear 要给 character")
        if action == "clear":
            if character == "all":
                anchors.clear()
            elif anchors.pop(character, None) is None:
                return ToolResult(ok=False, error=f"{character} 没有锚点")
            aid = self._save_anchors(anchors, [])
            return ToolResult(content=f"已清除，下次渲染重新定锚点。锚点表 {aid}", asset_ref=aid)
        if action == "pin":
            if not clip_id:
                return ToolResult(ok=False, error="pin 要给 clip_id（视频片段资产 id）")
            try:
                clip = self.store.get(clip_id)
            except KeyError:
                near = self.store.nearest(clip_id) if hasattr(self.store, "nearest") else ""
                hint = f"；相近的有 {near}" if near else ""
                return ToolResult(ok=False, error=f"没有资产 {clip_id}{hint}")
            if clip.type is not AssetType.VIDEO and not (clip.uri or "").endswith(".mp4"):
                return ToolResult(ok=False, error=f"{clip_id} 不是视频片段（{clip.type.value}）")
            anchors[character] = Anchor(
                character=character,
                asset=clip_id,
                scene=str(clip.summary or "").split(" ")[0],
                clip=" ".join(str(clip.summary or "").split(" ")[1:]),
                pinned=True,
            )
            aid = self._save_anchors(anchors, [clip_id])
            return ToolResult(
                content=f"已把 {character} 的声音锚在 {clip_id}（{clip.summary}）。锚点表 {aid}",
                asset_ref=aid,
            )
        return ToolResult(ok=False, error=f"未知 action {action!r}，用 list / pin / clear")

    async def registry_invoke(self, name: str, args: dict[str, Any]) -> Any:
        if self.registry is None:
            return ToolResult(ok=False, error=f"没接注册表，调不了 {name}")
        return await self.registry.invoke(name, args)

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            r = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
            r = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        r.duration_ms = int((time.perf_counter() - started) * 1000)
        return r


    # ---------- 入口判断：想法 还是 剧本 ----------

    async def _fn_drama_intake(self, text: str) -> ToolResult:
        """判断用户给的是一句话想法还是完整剧本，并指出下一步。

        判错的代价不对称：想法当剧本 → 拿一句话去拆分镜，模型只能硬编；
        剧本当想法 → 让用户重写他已经写好的东西。所以用结构信号判，
        看不准就如实说，不猜。
        """
        r = classify(text)
        if r.kind == "idea":
            return ToolResult(
                content=(
                    f"判定：**一句话想法**（{'、'.join(r.signals)}）\n\n"
                    "短剧要先有剧本才能往下走。下一步：\n"
                    "  调 drama_write 把这个想法写成剧本（按短剧方法论："
                    "开篇够狠、结尾断在信息缺口、爽点同集兑现）。\n"
                    "  写完简报后直接进拆解流程（drama_storyboard + drama_assets），"
                    "不再单独等确认。"
                )
            )
        if r.kind == "script":
            return ToolResult(
                content=(
                    f"判定：**完整剧本**（{'、'.join(r.signals)}）\n\n"
                    + ask_script_next().replace("[bold]", "").replace("[/]", "")
                    + "\n\n选 1 → 直接 drama_storyboard + drama_assets；"
                    "选 2 → 先调 drama_expand。"
                )
            )
        return ToolResult(content=f"判定：**看不准**（得分 {r.score}）\n\n{ask_unclear(text)}")

    # ---------- 写剧本 / 扩写 ----------

    async def _fn_drama_write(
        self, idea: str, episodes: int = 1, minutes: float = 0
    ) -> ToolResult:
        if not idea.strip():
            return ToolResult(ok=False, error="idea 是空的")
        resp = await self._prose(
            [{"role": "user", "content": write_prompt(idea, episodes, minutes, fmt=self.fmt)}]
        )
        stop = _finish_problem(resp, "剧本")
        if stop:
            return ToolResult(ok=False, error=stop)
        return self._save_script(resp.text, f"剧本·{episodes}集", ["idea"], idea)

    async def _prose(self, messages: list[dict[str, Any]]) -> Any:
        """散文类调用走 drama_prose；角色没配（旧配置）就退回 drama。"""
        try:
            return await self.gateway.chat(_PROSE_ROLE, messages)
        except KeyError:
            return await self.gateway.chat(_ROLE, messages)

    async def _fn_drama_expand(self, script: str = "", script_id: str = "", note: str = "") -> (
        ToolResult
    ):
        if script_id and not script:
            try:
                script = self.store.content(script_id)
            except KeyError as e:
                return ToolResult(ok=False, error=f"取不到剧本资产：{e}")
        if not script.strip():
            return ToolResult(ok=False, error="没有剧本内容")
        resp = await self._prose(
            [{"role": "user", "content": expand_prompt(script, note, fmt=self.fmt)}]
        )
        stop = _finish_problem(resp, "扩写")
        if stop:
            return ToolResult(ok=False, error=stop)
        return self._save_script(
            resp.text, "剧本·扩写修改", [script_id] if script_id else [], ""
        )

    def _script_input(self, script: str, ids: list[str]) -> tuple[str, str]:
        """拆解类工具的剧本入参：资产 id 优先（按顺序拼），否则用原文。返回 (剧本, 错误)。

        2026-09-23 审查：之前只收全文 —— 模型把整部剧塞进工具参数（25 万上下文那次事故的
        入口），或者抄回折叠占位符、写个「测试」，工具照样调 gemini，凭空编出另一部剧的
        分镜（真实日志：11 字的「测试」→ 118 秒后返回「15 段·共 210s」）。
        """
        parts: list[str] = []
        for aid in ids:
            try:
                parts.append(self.store.content(aid))
            except KeyError as e:
                return "", f"取不到剧本资产：{e}"
        text = "\n\n".join(parts) if parts else (script or "")
        problem = _script_problem(text)
        return (text, "") if not problem else ("", problem)

    def _save_script(
        self, text: str, summary: str, parents: list[str], idea: str
    ) -> ToolResult:
        body = _unwrap_script(text)
        if len(body) < 50:
            return ToolResult(ok=False, error=f"模型没写出像样的剧本：{body[:120]}")

        r = classify(body)
        asset = self.store.create(
            body,
            type_=AssetType.SCRIPT,
            summary=summary,
            parents=[p for p in parents if p and p != "idea"],
            creator="tool:drama_write",
            gen_params={"idea": idea} if idea else {},
        )
        # 自查：写出来的东西得真像剧本，否则下一步拆分镜会出洋相
        warn = ""
        if not r.is_script:
            warn = (
                f"\n\n⚠ 自查：写出来的内容结构上不像剧本（{r.brief()}）。"
                "可能缺场景标头或对白格式，进拆解前先看一眼。"
            )
        return ToolResult(
            content=f"{body}\n\n---\n{summary} · {len(body)} 字 · 资产 {asset.id}{warn}",
            asset_ref=asset.id,
        )

    # ---------- ① 分镜脚本 ----------

    async def _fn_drama_storyboard(
        self,
        script: str = "",
        ethnicity: str = "",
        language: str = "",
        note: str = "",
        script_id: str = "",
    ) -> ToolResult:
        script, err = self._script_input(script, [script_id] if script_id else [])
        if err:
            return ToolResult(ok=False, error=err, meta={"charged": False})
        opts = normalize(ethnicity, language)
        if not opts.ready:
            # 不猜。族裔和语言会贯穿角色形象和全部分镜视频，选错等于整条链重跑。
            return ToolResult(ok=False, error=ask_text())
        user = script if not note else f"{script}\n\n【额外要求】{note}"
        messages = [
            {
                "role": "system",
                "content": storyboard_system(opts, level=self._realism_level(), fmt=self.fmt),
            },
            {"role": "user", "content": user},
        ]
        resp = await self.gateway.chat(_ROLE, messages)
        stop = _finish_problem(resp, "分镜脚本")
        if stop:
            return ToolResult(ok=False, error=stop)
        eps, err = parse_episodes(resp.text)
        if err:
            return ToolResult(ok=False, error=f"分镜解析失败：{err}")

        # 规格检查（一集 4 分钟的镜头数、开场 15 秒高潮点）：不合格让模型改一次
        problems = [p for e in eps for p in check_storyboard(e, self.fmt)]
        if problems:
            text2 = await self._revise_once(_ROLE, messages, resp.text, problems)
            eps2, err2 = parse_episodes(text2)
            if not err2:
                problems2 = [p for e in eps2 for p in check_storyboard(e, self.fmt)]
                if len(problems2) < len(problems):
                    eps, problems = eps2, problems2

        asset = self.store.create(
            json.dumps([as_dict(e) for e in eps], ensure_ascii=False, indent=2),
            type_=AssetType.STORYBOARD,
            summary=f"分镜脚本·{len(eps)}集",
            parents=[script_id] if script_id else [],
            creator="tool:drama_storyboard",
            gen_params={
                "episodes": len(eps),
                "ethnicity": opts.ethnicity,
                "language": opts.language,
                "format_problems": len(problems),
            },
        )
        body = "\n".join(
            f"{e.title}　{e.shot_count} 镜 / {len(e.scenes)} 场" for e in eps
        )
        return ToolResult(
            content=f"拆出 {len(eps)} 集：\n{body}{_format_note(problems, self.fmt)}"
            f"\n\n资产 {asset.id}（第③步要用）",
            asset_ref=asset.id,
        )

    async def _revise_once(
        self, role: str, messages: list[dict[str, Any]], first: str, problems: list[str]
    ) -> str:
        """规格不合格时让模型改一次：上一版 + 问题清单一起发回去，要求重出完整 JSON。"""
        follow = [
            *messages,
            {"role": "assistant", "content": first},
            {
                "role": "user",
                "content": "上一版有以下硬性问题，逐条修正后**重新输出完整 JSON**"
                "（其它内容保持不变，格式不变）：\n- " + "\n- ".join(problems),
            },
        ]
        resp = await self.gateway.chat(role, follow)
        if _finish_problem(resp, "修订"):
            return first  # 改的那版被截断 / 被过滤：保留第一版，别拿半截的替换它
        return resp.text

    # ---------- ② 资产库 ----------

    async def _fn_drama_assets(
        self,
        script: str = "",
        ethnicity: str = "",
        language: str = "",
        note: str = "",
        script_id: str = "",
        script_ids: list[str] | None = None,
    ) -> ToolResult:
        ids = list(script_ids or []) + ([script_id] if script_id else [])
        script, err = self._script_input(script, ids)
        if err:
            return ToolResult(ok=False, error=err, meta={"charged": False})
        opts = normalize(ethnicity, language)
        if not opts.ready:
            return ToolResult(ok=False, error=ask_text())
        user = script if not note else f"{script}\n\n【额外要求】{note}"
        resp = await self.gateway.chat(
            _ROLE,
            [
                {"role": "system", "content": assets_system(opts, level=self._realism_level())},
                {"role": "user", "content": user},
            ],
        )
        stop = _finish_problem(resp, "资产库")
        if stop:
            return ToolResult(ok=False, error=stop)
        lib, err = parse_assets(resp.text)
        if err:
            return ToolResult(ok=False, error=f"资产库解析失败：{err}")

        payload = {
            "characters": [as_dict(c) for c in lib.characters],
            "scenes": [as_dict(s) for s in lib.scenes],
            "props": [as_dict(p) for p in lib.props],
        }
        asset = self.store.create(
            json.dumps(payload, ensure_ascii=False, indent=2),
            type_=AssetType.STORYBOARD,
            summary=f"资产库·{lib.counts}",
            creator="tool:drama_assets",
            gen_params={
                "characters": len(lib.characters),
                "ethnicity": opts.ethnicity,
                "language": opts.language,
            },
        )
        # 服装-场景分配表给人看：哪个角色在哪些场景穿哪套，一眼能看出缺口
        lines = [f"角色 {len(lib.characters)} 个 · {lib.counts}", lib.wardrobe_matrix()]
        # 音色卡：渲染时逐段锁进提示词，缺了的角色声音会漂
        lines.append("音色卡：")
        lines += [f"  {c.name}：{c.voice or '（缺）'}" for c in lib.characters]
        unvoiced = [c.name for c in lib.characters if not c.voice]
        if unvoiced:
            lines.append(
                f"⚠ {len(unvoiced)} 个角色没有音色卡（{', '.join(unvoiced[:5])}）—— "
                "多集下来声音会漂；重跑 drama_assets 或人工补 voice 字段"
            )
        if lib.scenes:
            lines.append(f"场景：{'、'.join(s.name for s in lib.scenes)}")
        if lib.props:
            lines.append(f"道具：{'、'.join(p.name for p in lib.props)}")
        loose = unbound_costumes(lib)
        if loose:
            lines.append(
                f"⚠ {len(loose)} 套服装没标场景/集数（{', '.join(loose[:5])}）—— "
                "第③步只能拿它兜底，换装时机说不清；重跑 drama_assets 或人工补 scenes"
            )
        return ToolResult(
            content="\n".join(lines) + f"\n\n资产 {asset.id}（第③步要用）",
            asset_ref=asset.id,
        )

    def _inherit_options(self, *asset_ids: str) -> Any:
        """从前序资产里取回族裔/语言选择。"""
        for aid in asset_ids:
            try:
                gp = self.store.get(aid).gen_params or {}
            except KeyError:
                continue
            if gp.get("ethnicity") or gp.get("language"):
                return normalize(str(gp.get("ethnicity") or ""), str(gp.get("language") or ""))
        return normalize("", "")

    # ---------- ③ 视频提示词 ----------

    async def _fn_drama_shots(
        self, storyboard_id: str, assets_id: str, episode: int = 0, note: str = ""
    ) -> ToolResult:
        try:
            sb_raw = self.store.content(storyboard_id)
            as_raw = self.store.content(assets_id)
        except KeyError as e:
            return ToolResult(ok=False, error=f"取不到资产：{e}")

        lib, err = parse_assets(as_raw)
        if err:
            return ToolResult(ok=False, error=f"资产库读不出来：{err}")
        eps, err = parse_episodes(sb_raw)
        if err:
            return ToolResult(ok=False, error=f"分镜脚本读不出来：{err}")

        if episode:
            eps = [e for e in eps if e.index == episode]
            if not eps:
                return ToolResult(ok=False, error=f"分镜里没有第 {episode} 集")

        # 语言选择从第①步继承。重新问一遍是多余的，而且用户答得不一样时
        # 会把已经翻译好的台词又译回去 —— 台词必须在整条链上保持一致。
        opts = self._inherit_options(storyboard_id, assets_id)

        sb_text = "\n\n".join(f"{e.title}\n{e.desc}" for e in eps)
        user = shots_prompt(lib.digest(), sb_text)
        if note:
            user += f"\n\n【额外要求】{note}"
        messages = [
            {
                "role": "system",
                "content": shots_system(opts, level=self._realism_level(), fmt=self.fmt),
            },
            {"role": "user", "content": user},
        ]
        resp = await self.gateway.chat(_ROLE, messages)
        stop = _finish_problem(resp, "视频提示词")
        if stop:
            return ToolResult(ok=False, error=stop)
        shots, err = parse_shots(resp.text)
        if err:
            return ToolResult(ok=False, error=f"镜头提示词解析失败：{err}")

        # 规格检查（单段 10–15s、一集 ≈240s、开场 15s 内有 hook）：不合格让模型改一次
        problems = check_shots(shots, self.fmt)
        if problems:
            text2 = await self._revise_once(_ROLE, messages, resp.text, problems)
            shots2, err2 = parse_shots(text2)
            if not err2:
                problems2 = check_shots(shots2, self.fmt)
                if len(problems2) < len(problems):
                    shots, problems = shots2, problems2

        # 全角括号里的资产名换成半角：提示词示例曾写成全角，模型照抄，场景引用和按场景换装
        # 就全部失效（2026-09-23 审查）
        known = lib.all_names()
        for s in shots:
            s.description = normalize_ref_parens(s.description, known)
        # 服装按场景绑定（确定性，不靠模型自觉）：镜头所在场景有绑定服装就换成它的 ID，
        # 同一场景内穿搭一致、换场景才换装；裸角色名也换成服装 ID 让参考图对得上
        bindings, wardrobe_warns = bind_costumes(shots, lib)
        bad = audit_refs(shots, lib)
        gp: dict[str, Any] = {
            "shots": len(shots),
            "bad_refs": len(bad),
            "costume_bindings": len(bindings),
            "wardrobe_gaps": len(wardrobe_warns),
            "format_problems": len(problems),
        }
        if episode:
            gp["episode"] = episode  # 按集流水的产物标记：pipeline 按它认出「第N集提示词」
        asset = self.store.create(
            json.dumps([as_dict(s) for s in shots], ensure_ascii=False, indent=2),
            type_=AssetType.STORYBOARD,
            summary=f"视频提示词·{len(shots)}段" + (f"·第{episode}集" if episode else ""),
            parents=[storyboard_id, assets_id],
            creator="tool:drama_shots",
            gen_params=gp,
        )
        lines = []
        for s in shots:
            carry = f"　引入 {'/'.join(s.carries())}" if s.carries() else ""
            cuts = f"/{len(s.cuts)}镜" if s.cuts else ""
            lines.append(
                f"{s.scene_index} {s.video_name} {s.seconds}s{cuts}　"
                f"引用 {len(s.refs())} 个资产{carry}"
            )
        warn = ""
        if bindings:
            shown = bindings[:10]
            more = f"\n  …还有 {len(bindings) - 10} 处" if len(bindings) > 10 else ""
            warn += (
                f"\n\n服装已按场景绑定，改了 {len(bindings)} 处引用：\n  "
                + "\n  ".join(shown) + more
            )
        if wardrobe_warns:
            warn += (
                "\n\n⚠ 服装分配缺口（资产库没给这些角色×场景配服装，沿用了原引用；"
                "要换装就重跑 drama_assets 或人工补 scenes）：\n  " + "\n  ".join(wardrobe_warns)
            )
        if bad:
            warn += (
                "\n\n⚠ 这些引用在资产库里找不到，生视频时会没有参考图（人物会变脸）：\n  "
                + "\n  ".join(bad)
            )
        total = sum(s.seconds for s in shots)
        warn += _format_note(problems, self.fmt)
        return ToolResult(
            content=f"{len(shots)} 段 · 共 {total}s\n" + "\n".join(lines) + warn +
            f"\n\n资产 {asset.id}",
            asset_ref=asset.id,
        )

    # ---------- 渲染：资产生图 ----------

    async def _gen_image(
        self,
        prompt: str,
        ratio: str,
        summary: str,
        ref: str | list[str] = "",
        person: bool = False,
        local_name: str = "",
        report: list[dict[str, Any]] | None = None,
        minor: bool | None = None,
    ) -> tuple[str, str, str]:
        """出一张图，返回 (资产 id, 图片 url, 错误)。ref 是参考图 url（可多张）。

        人物图（person=True）走三道真实感保险：提示词硬约束在前、描述清洗、
        生成后视觉校验不合格自动重生成（次数见 media_models.yaml drama.realism_retries）。
        report 给了就把每张图的校验记录追加进去（调用方汇总给人看）。
        场景道具不挂这些 —— 那些不需要"皮肤纹理"，硬挂反而会干扰材质描述。
        """
        args: dict[str, Any] = {
            "model": self.image_model,
            "aspect_ratio": ratio,
            "summary": summary,
        }
        if local_name:
            args["local_name"] = local_name  # 产物目录里的文件名：类别-序号_名字
        refs = [ref] if isinstance(ref, str) else list(ref)
        refs = [u for u in refs if u]
        if refs:
            # 实测 2026-09-12：参考图走 image=[url]。
            # 对照实验证过确实生效 —— 同一句"参考图里的物体放到沙滩上"，
            # 带参考图出的是苹果，不带出的是相机。
            args["image"] = refs
        if not person:
            args["prompt"] = prompt
            return await self._gen_once(args)

        level = self._realism_level()
        # 未成年角色：儿童安全写法（服装图的描述里没有年龄，由调用方按角色传进来）
        minor = is_minor(prompt) if minor is None else minor
        text, notes = person_image_prompt(prompt, level=level, minor=minor)
        aid, url, err = await self._gen_once({**args, "prompt": text})
        if err:
            return aid, url, err
        entry: dict[str, Any] = {"name": summary, "notes": notes, "attempts": 1, "asset": aid}
        if report is not None:
            report.append(entry)
        aid, url = await self._realism_gate(args, prompt, level, entry, aid, url, minor=minor)
        # 有参考图的人物图（服装）再过一道人物一致性：和主形象是不是同一个人
        if refs:
            aid, url = await self._identity_gate_image(args, text, refs, entry, aid, url)
        entry["asset"] = aid
        return aid, url, ""

    async def _realism_gate(
        self,
        args: dict[str, Any],
        prompt: str,
        level: str,
        entry: dict[str, Any],
        aid: str,
        url: str,
        minor: bool = False,
    ) -> tuple[str, str]:
        """真实感门：不合格按方向重生成，都没过留分数最高的。返回最终 (资产, url)。"""
        retries = self._realism_retries()
        if retries < 0:  # 校验关着（没配 / 没网关）
            entry["checked"] = False
            return aid, url
        best = (aid, url)
        best_score = -1
        for attempt in range(retries + 1):
            passed, score, issues, note, direction = await self._check_realism(
                aid, url, level, minor=minor
            )
            entry.update({"checked": True, "pass": passed, "score": score, "issues": issues})
            if note:
                entry["note"] = note
            if passed:
                return aid, url
            if score > best_score:
                best, best_score = (aid, url), score
            if attempt >= retries:
                break
            # 不合格：按校验指出的方向换提示词再生成一张（磨皮了→要质感；做旧过头→压轻；
            # 光太柔→要层次）。同样的参考图与版式；文件名自动 -v2
            entry["attempts"] = attempt + 2
            entry["retry_direction"] = direction or "smooth"
            text, _ = person_image_prompt(
                prompt, retry=direction or "smooth", level=level, minor=minor
            )
            aid2, url2, err2 = await self._gen_once({**args, "prompt": text})
            if err2:
                entry["note"] = f"重生成失败：{err2}"
                break
            aid, url = aid2, url2
        # 都没过：留分数最高的那张，报告里标「仍不达标」
        entry["pass"] = False
        return best

    async def _identity_gate_image(
        self,
        args: dict[str, Any],
        text: str,
        refs: list[str],
        entry: dict[str, Any],
        aid: str,
        url: str,
    ) -> tuple[str, str]:
        """人物一致性门（图）：和参考图不是同一个人就把差异写进提示词重生成一次。"""
        on, retries, pass_score = self._identity_cfg()
        if not on:
            return aid, url
        labels = [(f"参考图{i}", u) for i, u in enumerate(refs, 1)]
        best, best_score = (aid, url), -1
        attempts = 1
        for attempt in range(retries + 1):
            v = await self._check_identity(aid, labels, is_video=False, pass_score=pass_score)
            entry["identity"] = {
                "pass": v.passed, "score": v.score, "issues": v.issues, "attempts": attempts,
            }
            if v.note:
                entry["identity"]["note"] = v.note
            if v.passed:
                return aid, url
            if v.score > best_score:
                best, best_score = (aid, url), v.score
            if attempt >= retries:
                break
            attempts += 1
            again = {**args, "prompt": identity_retry_prompt(text, v.issues)}
            aid2, url2, err2 = await self._gen_once(again)
            if err2:
                entry["identity"]["note"] = f"重生成失败：{err2}"
                break
            aid, url = aid2, url2
        entry["identity"]["pass"] = False
        return best

    async def _gen_once(self, args: dict[str, Any]) -> tuple[str, str, str]:
        r = await self.registry_invoke("gen_image", args)
        if not r.ok:
            return "", "", r.error or "生图失败"
        try:
            uri = self.store.get(r.asset_ref).uri or ""
        except KeyError:
            uri = ""
        return r.asset_ref, uri, ""

    # ---------- 真实感校验 ----------

    def _realism_retries(self) -> int:
        """校验开关：-1 = 关；否则是不合格时允许重生成的次数。

        没有文本网关（脚本/测试）或配置里没开就关。默认关：多一次视觉调用 + 可能的重生成
        都是钱，要用户在 media_models.yaml 的 drama 段显式打开。
        """
        if self.gateway is None or self.catalog is None:
            return -1
        cfg = getattr(self.catalog, "drama", {}) or {}
        if str(cfg.get("realism_gate", "")).strip().lower() not in ("1", "true", "yes", "on"):
            return -1
        try:
            return max(0, int(cfg.get("realism_retries", 1)))
        except (TypeError, ValueError):
            return 1

    def _image_payload(self, asset_id: str, url: str) -> str:
        """给视觉模型看的图：本地副本转 data URL（不依赖对方能不能拉外链），没有就给 url。"""
        try:
            local = local_copy(self.store.get(asset_id))
        except KeyError:
            local = None
        if local:
            try:
                data = Path(str(local)).read_bytes()
                mime = "image/png" if str(local).lower().endswith(".png") else "image/jpeg"
                return f"data:{mime};base64," + base64.b64encode(data).decode()
            except OSError:
                pass
        return url

    def _realism_level(self) -> str:
        """真实感档位：subtle（默认，真实但干净）/ natural / strong。

        见 media_models.yaml 的 drama.realism_level。
        """
        cfg = getattr(self.catalog, "drama", {}) or {}
        return norm_level(cfg.get("realism_level"))

    async def _check_realism(
        self, asset_id: str, url: str, level: str = "", minor: bool = False
    ) -> tuple[bool, int, list[str], str, str]:
        """视觉模型判一张人物图的皮肤质感是否达标。返回 (通过, 分数, 问题, 备注, 不合格方向)。

        不合格方向：smooth 磨皮了 / heavy 做旧过头 / light 光太柔 —— 决定重生成往哪边改。
        校验本身失败（角色没配、调用异常、输出不是 JSON）按通过处理但写备注 ——
        校验是保险，不能把生图链路一起挂掉。
        """
        image = self._image_payload(asset_id, url)
        if not image:
            return True, -1, [], "没有可校验的图片", ""
        try:
            resp = await self.gateway.chat(
                REALISM_ROLE, realism_check_messages(image, level, minor=minor)
            )
        except KeyError:
            return True, -1, [], f"models.yaml 没配 {REALISM_ROLE} 角色，跳过校验", ""
        except Exception as e:  # noqa: BLE001
            return True, -1, [], f"校验调用失败：{type(e).__name__}: {e}", ""
        v = parse_realism_report(resp.text)
        note = "；".join(i for i in v.issues if "不是" in i and "JSON" in i)
        return v.passed, v.score, v.issues, note, v.direction

    def _previous_pack(self, assets_id: str) -> tuple[str, dict[str, dict[str, str]]]:
        """这套资产库最新的参考图包 (id, images)。没有就 ("", {})。"""
        for a in self.store.find(creator="tool:drama_render_assets"):
            if assets_id in a.parent_ids:
                images = _load_pack(self.store.content(a.id))
                if images:
                    return a.id, images
        return "", {}

    async def _fn_drama_render_assets(
        self, assets_id: str, only: str = "all", reuse: bool = True
    ) -> ToolResult:
        try:
            raw = self.store.content(assets_id)
        except KeyError as e:
            return ToolResult(ok=False, error=f"取不到资产库：{e}")
        lib, err = parse_assets(raw)
        if err:
            return ToolResult(ok=False, error=f"资产库读不出来：{err}")

        # 四类图、两层依赖（2026-09-22 去掉了三视图层）：
        #   主形象（3:4 证件照，脸）→ 各场景服装（参考主形象保脸，纯白底单张全身）
        #   场景 / 道具 无依赖，和主形象同层
        want = {
            "all": {"characters", "costumes", "scenes", "props"},
            "characters": {"characters"},
            "costumes": {"characters", "costumes"},  # 服装依赖主形象
            "scenes": {"scenes"},
            "props": {"props"},
        }.get(only, {"characters", "costumes", "scenes", "props"})
        # 本地文件名：类别-序号_名字，按资产库原序编号，后期按名字就能排
        names = library_reference_names(lib)

        # 增量：上次已经生成过的图直接复用，只补缺的。
        # 实测模型先 only=characters 省钱，之后再补服装/场景 —— 不复用的话主形象
        # 会再花一遍钱，而且新主形象和旧的不是同一张脸。
        # 上一个包不管 reuse 与否都读：这次没要求生成的类别要原样带进新包（2026-09-23 审查：
        # 之前 only=scenes 出的新包里只有场景，下游默认取最新的包，渲视频时人物全没了、被引用门
        # 拦下；再跑 all 又换脸又重复付费）。reuse 只管要生成的这几类能不能沿用旧图
        prev_id, prev = self._previous_pack(assets_id)
        images: dict[str, dict[str, str]] = {}
        reused: list[str] = []
        failed: list[str] = []
        refused: list[str] = []  # 主形象被服务商内容护栏拒掉的角色
        gate_log: list[dict[str, Any]] = []  # 每张人物图的真实感处理与校验记录
        limit = self._max_concurrency("image")

        def take(name: str) -> bool:
            if reuse and name in prev and prev[name].get("url"):
                images[name] = dict(prev[name])
                reused.append(name)
                return True
            return False

        # 并发 + 依赖排队（用户 2026-09-17 定的顺序约束）：
        # **所有无前置依赖的图（角色/场景/道具）全部生成完成后**，
        # 需要前置依赖的（服装要等自己角色的主形象当参考图保脸）才开始。
        # 层内并发，层间严格等齐；信号量只压同层任务，依赖永远排队。
        # 元素：(类别, 名字, 提示词, 比例, 是否人物)
        phase_a: list[tuple[str, str, str, str, bool]] = []
        if "characters" in want:
            for c in lib.characters:
                if not take(c.name):
                    phase_a.append(("角色", c.name, c.prompt(), "3:4", True))
        if "scenes" in want:
            for n in lib.scenes:
                if not take(n.name):
                    phase_a.append(("场景", n.name, n.prompt(), "16:9", False))
        if "props" in want:
            for n in lib.props:
                if not take(n.name):
                    phase_a.append(("道具", n.name, n.prompt(), "16:9", False))
        planned_costumes: list[Any] = []
        # 这次要重生成主形象的角色：它的旧服装图是按旧脸生成的，不能再沿用（换脸要传到服装）
        face_changed: set[str] = set()
        if "costumes" in want:
            for c in lib.characters:
                if "characters" in want and c.name not in reused:
                    face_changed.add(c.name)
                face_id = str((prev.get(c.name) or {}).get("asset") or "")
                for cos in c.costumes:
                    made_from = str((prev.get(cos.name) or {}).get("portrait") or "")
                    stale = c.name in face_changed or bool(made_from and made_from != face_id)
                    if stale or not take(cos.name):
                        planned_costumes.append((c, cos))

        # 进度窗的总数：第一层 + 计划的服装数（缺主形象被跳过的也算进去，
        # 跳过时同样报进度 —— 总数在一开始就报出去了，不能半路缩水）
        total = len(phase_a) + len(planned_costumes)
        done = 0

        async def gen_tracked(label: str, name: str, *args: Any, **kwargs: Any) -> Any:
            nonlocal done
            r = await self._gen_image(*args, report=gate_log, **kwargs)
            done += 1
            await self._progress("渲染参考图", done, total, f"{label}·{name}")
            return r

        await self._progress("渲染参考图", 0, total)
        res_a = await _run_parallel(
            phase_a,
            lambda p: gen_tracked(
                p[0], p[1], p[2], p[3], f"{p[0]}·{p[1]}",
                person=p[4], local_name=names.get(p[1], ""),
            ),
            limit,
        )
        for (label, name, *_), (aid, url, e) in zip(phase_a, res_a, strict=True):
            if e:
                failed.append(f"{label} {name}：{e[:70]}")
                if label == "角色" and _refused(e):
                    refused.append(name)
                continue
            # kind 写进包里：渲视频时按它排参考图的优先级（人物先于场景道具）
            images[name] = {"asset": aid, "url": url, "kind": label}

        # 第二层：各场景服装。**没有主形象就不生成** —— 生了也是另一张脸，
        # 那正是整套流程要消灭的问题，不如直接报出来。
        # 参考 = 主形象（脸）；服装图是纯白底单张全身正面照，一套服装一张。
        # 元素：(服装名, 提示词, 参考图 url 列表, 主形象资产 id, 是否未成年)
        phase_b: list[tuple[str, str, list[str], str, bool]] = []
        if planned_costumes:
            for c, cos in planned_costumes:
                face = images.get(c.name) or (prev.get(c.name) if reuse else None) or {}
                portrait = str(face.get("url") or "")
                if not portrait:
                    failed.append(f"服装 {cos.name}：缺角色 {c.name} 的主形象，跳过")
                    done += 1  # 跳过也计入进度，总数不缩水
                    await self._progress("渲染参考图", done, total, f"服装·{cos.name}")
                    continue
                phase_b.append((
                    cos.name, cos.prompt(), [portrait], str(face.get("asset") or ""),
                    is_minor(c.body),
                ))
            res_b = await _run_parallel(
                phase_b,
                lambda p: gen_tracked(
                    "服装", p[0], p[1], "16:9", f"服装·{p[0]}",
                    ref=p[2], person=True, local_name=names.get(p[0], ""), minor=p[4],
                ),
                limit,
            )
            for (name, _p, _r, face_id, _m), (aid, url, e) in zip(phase_b, res_b, strict=True):
                if e:
                    failed.append(f"服装 {name}：{e[:70]}")
                    continue
                # 记下按哪张脸生成的：主形象换了，这张就作废（下次 render_assets 会重生成）
                images[name] = {"asset": aid, "url": url, "kind": "服装", "portrait": face_id}

        # 输出按资产库原序，不按完成顺序（并发后两者不一样）
        lines: list[str] = []
        for c in lib.characters:
            if c.name in images:
                lines.append(f"  ✓ 角色 {c.name}" + ("（复用）" if c.name in reused else ""))
            for cos in c.costumes:
                if cos.name in images:
                    where = f" ← {'、'.join(cos.scenes)}" if cos.scenes else ""
                    lines.append(
                        f"  ✓ 服装 {cos.name}{where}"
                        + ("（复用）" if cos.name in reused else "")
                    )
        for label, bucket in (("场景", lib.scenes), ("道具", lib.props)):
            for n in bucket:
                if n.name in images:
                    lines.append(f"  ✓ {label} {n.name}" + ("（复用）" if n.name in reused else ""))

        if not images:
            return ToolResult(ok=False, error="一张图都没生成：\n" + "\n".join(failed))

        # 新包 = 上一个包 + 这次生成 / 沿用的。主形象这次换了、服装却没重生成成功的：旧服装图
        # 是旧脸，不能留 —— 去掉后渲视频时退回主形象（新脸）
        merged = {k: dict(v) for k, v in prev.items()}
        for c in lib.characters:
            if c.name in face_changed and c.name in images:
                for cos in c.costumes:
                    if cos.name not in images and merged.pop(cos.name, None) is not None:
                        failed.append(
                            f"服装 {cos.name}：主形象换了、新服装图没生成出来，旧的（旧脸）已从包里"
                            "去掉，渲视频时先退回主形象"
                        )
        merged.update(images)
        kept = [k for k in merged if k not in images]
        if kept:
            lines.append(f"  · 其余 {len(kept)} 张沿用上一个参考图包（这次没要求生成这几类）")

        # 把真实感校验结果记进包里：哪张图过了、几次才过（按资产 id 对上）
        by_asset = {e["asset"]: e for e in gate_log if e.get("checked") and e.get("asset")}
        for entry in merged.values():
            rec = by_asset.get(entry.get("asset", ""))
            if rec:
                entry["realism"] = {
                    "pass": bool(rec.get("pass")),
                    "score": rec.get("score", -1),
                    "attempts": rec.get("attempts", 1),
                }

        asset = self.store.create(
            json.dumps(merged, ensure_ascii=False, indent=2),
            type_=AssetType.STORYBOARD,
            summary=f"资产参考图·{len(merged)}张",
            parents=[assets_id] + ([prev_id] if prev_id else []),
            creator="tool:drama_render_assets",
            gen_params={
                "model": self.image_model,
                "count": len(merged),
                "only": only,
                "reused": len(reused),
                "kept": len(kept),
                "realism_checked": len(by_asset),
            },
        )

        fresh = len(images) - len(reused)
        head = (
            f"参考图包共 {len(merged)} 张（本次新生成 {fresh}，复用 {len(reused)}"
            + (f"，沿用上一个包 {len(kept)}" if kept else "")
            + "）：\n"
        )
        warn = ("\n\n未生成：\n  " + "\n  ".join(failed)) if failed else ""
        if refused:
            # 主形象被内容护栏拒掉：没有降级路径（换脸就不是这个角色了），停下来问人
            return ToolResult(
                content=head + "\n".join(lines) + warn + f"\n\n资产 {asset.id}",
                asset_ref=asset.id,
                suspend=True,
                suspend_payload={
                    "question": _refused_question(refused),
                    "stage": REFUSED_STAGE,
                    "target": REFUSED_STAGE,
                    "assets": [asset.id],
                    "major": True,
                },
            )
        # 面容审查（2026-09-22 用户定：渲完参考图后自动跑）——
        # 这一步是"最早能拦住脸漂移"的位置：再往后就是渲视频，带着错的脸花的是大钱。
        # 有定不了的角色会挂起问人，那时整个渲染结果也一并带出去，不会白渲。
        face = ""
        if self._face_cfg()[0]:
            r = await self._fn_drama_audit_faces(assets_id=assets_id)
            if r.suspend:
                r.content = head + "\n".join(lines) + warn + "\n\n" + r.content
                return r
            face = "\n\n" + (r.content or r.error or "")
        return ToolResult(
            content=head + "\n".join(lines) + warn
            + "\n\n" + _coverage_note(lib, merged)
            + _realism_note(gate_log)
            + face
            + f"\n\n资产 {asset.id}（生视频时要用）",
            asset_ref=asset.id,
        )

    # ---------- 渲染：分镜生视频 ----------

    async def _fn_drama_render_shots(
        self,
        shots_id: str,
        rendered_id: str = "",
        limit: int = 0,
        compose: bool = True,
        out_dir: str = "",
        filename: str = "",
        episode: int = 0,
        reuse: bool = True,
        redo: list[str] | None = None,
        accept: list[str] | None = None,
    ) -> ToolResult:
        try:
            raw = self.store.content(shots_id)
        except KeyError as e:
            return ToolResult(ok=False, error=f"取不到分镜提示词：{e}")
        shots, err = parse_shots(raw)
        if err:
            return ToolResult(ok=False, error=f"分镜提示词读不出来：{err}")

        # 集内顺序号按完整列表算（limit / episode 过滤不影响编号），落盘文件名用它：
        # 第01集-03_2场_镜9-18.mp4 —— 后期在剪辑软件里按名字排就是播放顺序
        seq_of: dict[int, int] = {}
        counters: dict[int, int] = {}
        for s in shots:
            ep_no, _ = parse_scene(s.scene_index)
            counters[ep_no] = counters.get(ep_no, 0) + 1
            seq_of[id(s)] = counters[ep_no]

        if episode:
            # 按集流水：只渲染本集镜头。「第N集-」带后缀，不会误匹配「第N1集」。
            # 跨集的 {前序} 引入解析不到（上一集镜头不在本批），会退化为无引入 ——
            # 分层渲染照常，只是那一段少了前序视频参考。
            shots = [
                s for s in shots if s.scene_index.strip("[]").startswith(f"第{episode}集-")
            ]
            if not shots:
                return ToolResult(ok=False, error=f"提示词里没有第 {episode} 集的镜头")

        if limit > 0:
            shots = shots[:limit]

        # ---- 参考图包：定位 + 校验 + 引用预匹配（花钱之前做完）----
        pack_id, images, pack_notes, err = self._locate_pack(rendered_id, shots_id)
        if err:
            return ToolResult(ok=False, error=err)
        # 参考图链接过期/本地：配了托管就先重新上传（图不变、链接换新），不然模型拿不到参考
        images, host_notes = await self._rehost_pack(pack_id, images)
        pack_notes += host_notes
        lib = self._library_for(shots_id, pack_id)
        # 全角括号里的资产名换成半角（老提示词照抄过全角示例）
        known = set(images) | (lib.all_names() if lib is not None else set())
        for s in shots:
            s.description = normalize_ref_parens(s.description, known)
        # 「(OS)」「(画外音)」这类标注不是资产引用，不参与匹配、也不算「包里没有」
        all_refs = [
            r for r in dict.fromkeys(r for s in shots for r in s.refs())
            if r in images or _looks_like_asset(r, lib)
        ]
        resolved = {r: _resolve_ref(r, images) for r in all_refs}
        if images and all_refs and not any(resolved.values()):
            return ToolResult(
                ok=False,
                error=(
                    f"分镜引用了 {len(all_refs)} 个资产名，参考图包里一个都对不上，"
                    "渲出来人物必然变脸，没有发起生成。\n"
                    f"  分镜引用：{', '.join(all_refs[:8])}\n"
                    f"  包里只有：{', '.join(list(images)[:8])}\n"
                    "多半是 rendered_id 传错了（要传 drama_render_assets 的产物），"
                    "或分镜与资产库不是同一套。"
                ),
            )
        # 增量重跑：上次已成功的段直接复用，只生成失败/缺的 —— 引用门只查这次要生成的段
        prev_clips = (
            self._previous_clips(shots_id, episode, pack_id, redo, accept) if reuse else {}
        )
        todo = [
            i for i, s in enumerate(shots)
            if not prev_clips.get((s.scene_index, s.video_name), "")
        ]
        assets_hint = ""
        if pack_id:
            try:
                assets_hint = (self.store.get(pack_id).parent_ids or [""])[0]
            except KeyError:
                assets_hint = ""
        if not images:
            if lib is not None and any(shots[i].refs() for i in todo):
                # 有资产库、分镜引用了人物/场景，却没渲过参考图：不许无参考生成
                try:
                    parents = self.store.get(shots_id).parent_ids
                except KeyError:
                    parents = []
                lib_id = assets_hint or (parents[1] if len(parents) > 1 else "")
                return ToolResult(
                    ok=False,
                    error=_ref_gate_error(
                        ["没有参考图包，分镜引用的人物/场景一个都拿不到参考图"], lib_id
                    ),
                )
            pack_notes.append(
                "⚠ 没有参考图包：全部镜头无参考渲染，人物一致性无法保证。"
                "先 drama_render_assets 再传 rendered_id。"
            )
        exact = [r for r, m in resolved.items() if m and m.key == r]
        fallback = [f"{r}→{m.key}" for r, m in resolved.items() if m and m.key != r]
        missing = [r for r, m in resolved.items() if m is None]
        max_refs = self._max_refs()
        if images:
            note = f"参考图匹配：{len(exact)} 个精确"
            if fallback:
                note += f" · {len(fallback)} 个服装退回主形象（{', '.join(fallback[:6])}）"
            pack_notes.append(note)
            # ---- 引用门（2026-09-20 用户定的规则）：没有引用成功的镜头不许生成，
            #      花钱之前整批拦 ----
            vcfg0 = self._voice_cfg()
            max_people = max(0, max_refs - vcfg0["max_videos"]) if max_refs else 0
            problems = preflight_refs(shots, todo, resolved, lib, max_people)
            stale = self._stale_refs(resolved, images, vcfg0)
            if stale:
                problems.append(
                    f"{len(stale)} 张参考图的链接超过 {vcfg0['ttl_h']:g} 小时"
                    f"（{', '.join(stale[:6])}），生成接口拿不到参考图"
                )
            if not problems and self._ref_probe_on():
                urls = list(dict.fromkeys(
                    resolved[r].url for i in todo for r in shots[i].refs()
                    if resolved.get(r) is not None
                ))
                probe = await self._probe_urls(urls)
                name_of = {m.url: m.key for m in resolved.values() if m is not None}
                dead = [name_of.get(u, u) for u, e in probe.items() if e and not e[:1] == "?"]
                unverified = [name_of.get(u, u) for u, e in probe.items() if e[:1] == "?"]
                if dead:
                    problems.append(
                        f"{len(dead)} 张参考图的链接已失效（{', '.join(dead[:6])}）"
                    )
                if unverified:
                    pack_notes.append(
                        f"⚠ {len(unverified)} 张参考图链接没验证上（网络不通），照常渲染："
                        f"{', '.join(unverified[:4])}"
                    )
            if problems:
                return ToolResult(ok=False, error=_ref_gate_error(problems, assets_hint))
            if missing:
                pack_notes.append(f"（复用的段引用了包里没有的 {', '.join(missing[:4])}，未重渲）")

        # ---- 单段时长超过锁定模型的上限：花钱之前拦 ----
        # 2026-09-23 审查：换了视频模型后时长被媒体层静默截到新模型上限，按 15s 排的镜头时间线
        # 尾巴被砍、一集总时长只剩约 2/3，全程没有提示
        cap = self._max_duration()
        over = [i for i in todo if cap and (shots[i].seconds or FALLBACK_SECONDS) > cap]
        if over:
            shown = ", ".join(
                f"{shots[i].scene_index} {shots[i].video_name}（{shots[i].seconds}s）"
                for i in over[:4]
            )
            return ToolResult(
                ok=False,
                meta={"charged": False},
                error=(
                    f"当前视频模型 {self.video_model} 单段最长 {cap}s，这次要渲的 {len(over)} 段"
                    f"超过它（{shown}）—— 渲出来会被截短，段内镜头时间线和一集总时长都对不上，"
                    "没有发起生成。换回支持这个时长的模型（会先问用户），或重跑 drama_shots 按"
                    f" ≤{cap}s 一段重新规划提示词"
                ),
            )

        # ---- 音色锁定（花钱之前算好）：谁在说话、沿用哪些锚点、本次要新定哪些 ----
        # 每段视频各自发声，几十段下来同一角色的声音必然漂。两道锁：说话角色的音色卡
        # 锁进提示词；该角色的锚点片段（第一段独白）当参考视频（@视频N 取音色），跨集沿用。
        lib_scene_names = lib.scene_names() if lib else set()
        # 服装 → 角色（渲视频时给服装配上角色的脸）；未成年角色（这段用儿童安全的真实感尾巴）
        owner_of = {cos.name: c.name for c in lib.characters for cos in c.costumes} if lib else {}
        minors = {c.name for c in lib.characters if is_minor(c.body)} if lib else set()
        speakers_by_shot = [speakers_of(s, lib) if lib else {} for s in shots]
        vcfg = self._voice_cfg()
        anchors = self._load_anchors() if lib else {}
        if anchors and vcfg["enabled"]:
            pack_notes += await self._rehost_anchors(anchors, vcfg)
        usable = {n for n, a in anchors.items() if not self._anchor_url(a, vcfg)[1]}
        plan = plan_anchors(speakers_by_shot, lib, anchors, usable) if lib else AnchorPlan()
        if not vcfg["enabled"]:
            plan.births.clear()  # 关了：不传参考视频，音色卡文字照锁
            plan.ready.clear()
        anchor_urls: dict[str, str] = {
            n: self._anchor_url(a, vcfg)[0] for n, a in plan.ready.items()
        }
        plan.spoken = list(dict.fromkeys(n for sp in speakers_by_shot for n in sp))
        new_anchors: dict[str, Anchor] = {}
        no_anchor: list[str] = []  # 有说话角色却没带上锚点的段（给人看）

        # ---- 增量重跑：prev_clips 在引用门之前就算好了（只查这次要生成的段） ----
        reused: list[int] = []
        retries = self._video_retries()
        sub_gate, sub_retries = self._subtitle_cfg()
        notes_of: dict[int, list[str]] = {}  # 每段的字幕检查备注

        # 并发 + 依赖排队（用户 2026-09-17 定的顺序约束）：
        # **所有无前置依赖的镜头（第 0 层）全部生成完成后**，需要前置依赖的才开始。
        # 「{第X集-Y场}」前序引入 = 依赖那个场景当时最新的一段
        # （沿用旧串行语义：指向列表里排在自己前面、同场景的最后一段）。
        # 层内并发，层间严格等齐。上限走配置的 concurrency.video。
        scene_prev: dict[str, int] = {}  # 场景 → 目前最后一段的下标
        level_of = [0] * len(shots)
        carry_dep: list[list[int]] = [[] for _ in shots]
        for i, s in enumerate(shots):
            carry_dep[i] = [j for c in s.carries() if (j := scene_prev.get(c)) is not None]
            if carry_dep[i]:
                level_of[i] = 1 + max(level_of[j] for j in carry_dep[i])
            elif s.carries():
                # 声明了引入但指不到已产出的场景（写错或指向后面的场景）：
                # 仍算「需要前置依赖」的内容，排到依赖层，不混进第一批
                level_of[i] = 1
            scene_prev[s.scene_index.strip("[]")] = i

        # 锚点镜头先渲，其他有这些角色开口的镜头排在它之后 —— 否则同一集里第一批并发
        # 出来的几段声音就各是各的。锚点依赖可能指向靠后的镜头（独白段在后面），
        # 和前序引入一起可能成环；成环就放弃锚点依赖，照原顺序渲（那几段只锁音色卡文字）。
        #
        # **锚点镜头之间互不依赖**（2026-09-19）：否则几个角色的锚点会串成一条链，
        # 层数随角色数增长，并发被压没。这样排下来最多就是「锚点层 + 其余全并发」两层。
        anchor_shots = set(plan.births.values())
        anchor_dep = [
            (
                []
                if i in anchor_shots
                else [plan.births[n] for n in sp if n in plan.births and plan.births[n] != i]
            )
            for i, sp in enumerate(speakers_by_shot)
        ]
        levels = list(level_of)
        for _ in range(len(shots) + 1):
            changed = False
            for i in range(len(shots)):
                want = max([levels[i]] + [levels[j] + 1 for j in carry_dep[i] + anchor_dep[i]])
                if want > levels[i]:
                    levels[i], changed = want, True
            if not changed:
                level_of = levels
                break
        else:
            pack_notes.append("⚠ 音色锚点依赖与前序引入成环，本次按原顺序渲染（锚点尽力而为）")

        layers: list[list[int]] = []
        for i, lv in enumerate(level_of):
            while len(layers) <= lv:
                layers.append([])
            layers[lv].append(i)

        done: dict[int, dict[str, Any]] = {}
        n_refs: dict[int, tuple[int, int]] = {}
        failed: list[str] = []
        blocked: list[str] = []  # 生成出来了但被质检门判 ⛔（不进成片，等人 accept 或 redo）
        by_scene: dict[str, str] = {}  # 场次 → 已成功的视频 url，供 {前序} 引入
        vlimit = self._max_concurrency("video")
        vmax = vcfg["max_videos"]
        done_v = 0

        await self._progress("渲染分镜视频", 0, len(shots))
        for layer in layers:

            async def gen(i: int) -> tuple[int, str, tuple[int, int], str, bool, list[str]]:
                nonlocal done_v
                s = shots[i]
                prev_id = prev_clips.get((s.scene_index, s.video_name), "")
                if prev_id:
                    done_v += 1
                    label = f"{s.scene_index} {s.video_name}（复用）"
                    await self._progress("渲染分镜视频", done_v, len(shots), label)
                    return i, prev_id, (0, 0), "", True, [], False
                # 引用到的资产图当参考：人物/服装优先（脸最要紧），再场景道具 ——
                # 接口若截断参考图数，先丢的是最不影响一致性的
                people: list[str] = []
                faces: list[str] = []  # 服装对应角色的主形象（脸）：可选参考，位子够才带
                others: list[str] = []
                label_of: dict[str, tuple[str, str]] = {}  # url → (类别, 名字)
                id_refs: list[tuple[str, str]] = []  # 一致性校验用：(角色名, url)
                for r in s.refs():
                    m = resolved.get(r)
                    if not m:
                        continue
                    (people if m.person else others).append(m.url)
                    kind = str((images.get(m.key) or {}).get("kind") or "")
                    if not kind:
                        kind = "角色" if m.person else ("场景" if r in lib_scene_names else "道具")
                    key = m.key.removesuffix("·三视图")
                    label_of.setdefault(m.url, (kind, key))
                    if m.person and kind in ("角色", "服装"):
                        id_refs.append((key, m.url))
                    if m.person and kind == "服装":
                        # 服装全身图里脸只占一小块：同时带上这个角色的主形象当脸的参考
                        # （2026-09-23 审查：之前只传服装图，「参考图传了照样漂」的主因之一）
                        who = owner_of.get(key, "")
                        fu = str((images.get(who) or {}).get("url") or "") if who else ""
                        if fu and fu != m.url:
                            faces.append(fu)
                            label_of.setdefault(fu, ("角色", who))
                refs = list(dict.fromkeys(people))
                face_refs = [u for u in dict.fromkeys(faces) if u not in refs]
                refs += face_refs
                refs += [u for u in dict.fromkeys(others) if u not in refs]
                ref_labels = [label_of[u] for u in refs if u in label_of]
                # 一致性校验优先拿脸比（主形象比全身服装图清楚得多）
                id_refs = list(dict.fromkeys(
                    [(label_of[u][1], u) for u in face_refs] + id_refs
                ))[:3]
                # 参考视频（@视频N 按顺序编号）：🔴 前序片段在前（画面延续），
                # 再是说话角色的音色锚点（台词多的优先），总数不超过接口上限
                videos: list[str] = []
                carry_tokens: list[tuple[str, int]] = []
                for carry in s.carries():
                    u = by_scene.get(carry)
                    if u and u not in videos and len(videos) < vmax:
                        videos.append(u)
                        carry_tokens.append((carry, len(videos)))
                sp = speakers_by_shot[i]
                anchor_tokens: dict[str, int] = {}
                for name in sorted(sp, key=lambda n: -sp[n]):
                    u = anchor_urls.get(name)
                    if not u:
                        continue
                    if u in videos:
                        anchor_tokens[name] = videos.index(u) + 1
                    elif len(videos) < vmax:
                        videos.append(u)
                        anchor_tokens[name] = len(videos)
                if sp and len(anchor_tokens) < len(sp):
                    lost = [n for n in sp if n not in anchor_tokens]
                    no_anchor.append(f"{s.scene_index} {s.video_name}（{'、'.join(lost)}）")
                # 参考位上限（seedance 9 个，含参考视频；媒体层双送参考图时会自动只送一份）：
                # 人物必保，位子不够先省道具、再省场景。人物本身就超位的在引用门已经拦下。
                trim_notes: list[str] = []
                if max_refs and len(refs) + len(videos) > max_refs:
                    budget = max(0, max_refs - len(videos))
                    people_set = set(people)
                    keep = [u for u in refs if u in people_set]
                    rest = [u for u in refs if u not in people_set]
                    # 位子不够时先保脸（主形象），再场景，最后道具
                    rank = {"角色": 0, "场景": 1}
                    rest.sort(key=lambda u: rank.get(label_of.get(u, ("", ""))[0], 2))
                    room = max(0, budget - len(keep))
                    dropped = rest[room:]
                    refs = keep + rest[:room]
                    ref_labels = [label_of[u] for u in refs if u in label_of]
                    if dropped:
                        gone = "、".join(
                            f"{label_of[u][0]}「{label_of[u][1]}」"
                            for u in dropped
                            if u in label_of
                        )
                        trim_notes.append(f"参考位不够（上限 {max_refs}），省略 {gone}")
                block = voice_block(list(sp), lib, anchor_tokens, carry_tokens) if lib else ""
                # 参考锁定：@图片N 逐张点名"这是谁、要一致什么"——只传图不点名，模型常常不用
                locks = "\n".join(b for b in (reference_block(ref_labels), block) if b)
                core = f"{locks}\n{s.description}" if locks else s.description
                # 每个镜头 ≤3 秒（2026-09-20 硬性要求）：快切约束 + 按 cuts 排的时间线包在最外层
                core = fast_cut_prompt(
                    core, s.seconds or FALLBACK_SECONDS, s.cuts, self.fmt.max_cut_seconds
                )
                args: dict[str, Any] = {
                    # 真实感：画面描述在前，真实感段与硬约束尾巴在后（视频模型先要听懂发生了什么）
                    "prompt": person_video_prompt(
                        core,
                        level=self._realism_level(),
                        minor=bool(minors & (
                            _covered_characters(s, resolved, lib)
                            | set(_mentioned_characters(s.description, lib))
                        )),
                    ),
                    "model": self.video_model,
                    "aspect_ratio": "9:16",
                    "resolution": "720p",
                    "duration": s.seconds or FALLBACK_SECONDS,
                    "summary": f"{s.scene_index} {s.video_name}",
                    "local_name": clip_name(s.scene_index, s.video_name, seq_of[id(s)]),
                }
                if refs:
                    args["image"] = refs
                if videos:
                    args["video_urls"] = videos
                # 片段自带标签（2026-09-23）：渲到一半被打断，已付费的段下次凭它复用；
                # 参考图包也记上，换了包（换脸）就不复用旧片段
                args["tags"] = {
                    "shots_id": shots_id,
                    "scene": s.scene_index,
                    "name": s.video_name,
                    "pack": pack_id,
                    "episode": episode,
                }
                # 网络类失败自动重试 + 人物一致性门 + 字幕门
                r, clip_notes = await self._gen_clip(
                    args, retries, sub_gate, sub_retries, id_refs=id_refs
                )
                blocked = False
                if r.ok and r.asset_ref:
                    blocked = not self._mark_clip(r.asset_ref, clip_notes)
                done_v += 1
                await self._progress(
                    "渲染分镜视频", done_v, len(shots), f"{s.scene_index} {s.video_name}"
                )
                counts = (len(refs), len(videos))
                err = "" if r.ok else (r.error or "")
                return (
                    i, r.asset_ref if r.ok else "", counts, err, False,
                    trim_notes + clip_notes, blocked,
                )

            # 层内按下标升序结算：by_scene 的「最新一段」语义与旧串行完全一致
            results = await _run_parallel(layer, gen, vlimit)
            for i, aid, counts, err, was_reused, clip_notes, was_blocked in results:
                s = shots[i]
                tag = f"{s.scene_index} {s.video_name}"
                if clip_notes:
                    notes_of[i] = clip_notes
                if err:
                    failed.append(f"{tag}：{err[:90]}")
                    continue
                if was_blocked:
                    blocked.append(tag)
                # 只有远端且未过期的链接能当后面镜头的参考（前序引入 / 音色锚点）
                url, _ = self._clip_ref_url(aid, vcfg)
                by_scene[s.scene_index.strip("[]")] = url
                done[i] = {"scene": s.scene_index, "name": s.video_name, "asset": aid}
                if was_blocked:
                    done[i]["flagged"] = True  # 被质检门拦下：进索引备查，不进成片、不复用
                n_refs[i] = counts
                if was_reused:
                    reused.append(i)
                # 这一段是某些角色的锚点镜头：从现在起他们的声音以它为准
                for name, idx in plan.births.items():
                    if idx == i and url:
                        anchor_urls[name] = url
                        old = anchors.get(name)
                        new_anchors[name] = Anchor(
                            character=name,
                            asset=aid,
                            scene=s.scene_index,
                            clip=s.video_name,
                            lines=speakers_by_shot[i].get(name, 0),
                            solo=len(speakers_by_shot[i]) == 1,
                            rolled_from=old.asset if old else "",
                        )

        if not done:
            return ToolResult(ok=False, error="一段视频都没生成：\n" + "\n".join(failed))

        anchors_id = ""
        if new_anchors:
            anchors_id = self._merge_save_anchors(new_anchors, [pack_id, shots_id])

        # 拼接按镜头原序 —— 并发后完成顺序是乱的，不能按完成顺序拼
        ordered = [done[i] for i in sorted(done)]
        rgp: dict[str, Any] = {
            "model": self.video_model,
            "count": len(ordered),
            "refs_resolved": sum(1 for m in resolved.values() if m),
            "refs_missing": len(missing),
            "voice_locked": sum(1 for sp in speakers_by_shot if sp),
            "anchors_born": len(new_anchors),
            # 这一集能不能成片：没有失败段、也没有被质检门拦下的段。按集流水靠它判断
            # 「这一集渲完了」—— 之前有失败段也落索引，流水线就当渲完了（2026-09-23 审查）
            "complete": not failed and not blocked,
            "failed": len(failed),
            "blocked": len(blocked),
        }
        if episode:
            rgp["episode"] = episode  # 按集流水：pipeline 按它认出「第N集片段已渲」
        asset = self.store.create(
            json.dumps(ordered, ensure_ascii=False, indent=2),
            type_=AssetType.STORYBOARD,
            summary=f"分镜视频·{len(ordered)}段" + (f"·第{episode}集" if episode else ""),
            parents=[shots_id] + ([pack_id] if pack_id else []),
            creator="tool:drama_render_shots",
            gen_params=rgp,
        )
        lines = []
        reused_set = set(reused)
        for i in sorted(done):
            s = shots[i]
            if i in reused_set:
                lines.append(
                    f"  ↺ {s.scene_index} {s.video_name} {s.seconds}s　复用上次成功片段 "
                    f"{done[i]['asset']}"
                )
                continue
            lost = [r for r in s.refs() if resolved.get(r) is None and _looks_like_asset(r, lib)]
            n_img, n_vid = n_refs[i]
            note = f"　参考 {n_img} 图" + (f" + {n_vid} 视频" if n_vid else "")
            if lost and images:
                note += f"（缺 {', '.join(lost[:3])}）"
            if speakers_by_shot[i]:
                note += f"　音色 {len(speakers_by_shot[i])} 人"
            if notes_of.get(i):
                note += "　检查：" + "；".join(notes_of[i])
            fname = clip_name(s.scene_index, s.video_name, seq_of[id(s)])
            cuts = f"/{len(s.cuts)}镜" if s.cuts else ""
            lines.append(
                f"  ✓ {s.scene_index} {s.video_name} {s.seconds}s{cuts}{note} → {fname}.mp4"
            )
        warn = ("\n\n失败：\n  " + "\n  ".join(failed)) if failed else ""
        if failed:
            warn += "\n  （再跑一次 drama_render_shots 只会补这些失败的段，成功的直接复用）"
        drifted = [
            f"{shots[i].scene_index} {shots[i].video_name}"
            for i in sorted(done)
            if any("仍与参考图不符" in n for n in notes_of.get(i, []))
        ]
        if drifted:
            warn += (
                "\n\n⚠ 这些段重生成后人物仍与参考图不符（人物漂移），需人工复核："
                f"{'；'.join(drifted)}"
            )
        still_text = [
            f"{shots[i].scene_index} {shots[i].video_name}"
            for i in sorted(done)
            if any("仍有字幕" in n for n in notes_of.get(i, []))
        ]
        if still_text:
            warn += (
                "\n\n⚠ 这些段重生成后画面里仍检测到字幕/文字，需人工复核："
                f"{'；'.join(still_text)}"
            )
        too_long = [
            f"{shots[i].scene_index} {shots[i].video_name}"
            for i in sorted(done)
            if any("仍有超过" in n for n in notes_of.get(i, []))
        ]
        if too_long:
            warn += (
                f"\n\n⚠ 这些段重生成后仍有超过 {self.fmt.max_cut_seconds:g} 秒的镜头"
                "（用户要求每镜 ≤3 秒），需人工复核或单独重渲："
                f"{'；'.join(too_long)}"
            )
        notes = ("\n\n" + "\n".join(pack_notes)) if pack_notes else ""
        notes += _voice_note(lib, plan, new_anchors, no_anchor, anchors_id, vcfg)
        fresh = len(ordered) - len(reused)
        head = f"生成 {len(ordered)} 段视频"
        if reused:
            head += f"（复用 {len(reused)}，新生成 {fresh}）"
        head += "：\n" + "\n".join(lines) + notes + warn

        state = {"complete": not failed and not blocked, "failed": len(failed),
                 "blocked": len(blocked)}
        if failed or blocked:
            # 缺段不成片（2026-09-20）：拼一个"看起来完整"的成片会让人以为渲完了 —— 真实事故里
            # 模型接着自己无参考补段、再拼成片。被质检门判 ⛔ 的段同样不进成片（2026-09-23：
            # 之前只提一句「需人工复核」照样拼，带字幕 / 变脸的段进了成片）。
            clips = " ".join(d["asset"] for d in ordered)
            why: list[str] = []
            if failed:
                why.append(
                    f"未成片：{len(failed)} 段没生成出来，这一集不完整，没有拼接 —— "
                    "修好失败原因后再跑一次 drama_render_shots(reuse=true)，只会补这些段"
                )
            if blocked:
                why.append(
                    f"未成片：{len(blocked)} 段没过质检门（{'；'.join(blocked[:6])}），"
                    "没有拼接 —— 请用户看一下这几段（本地文件在 videos/，被换掉的版本在 "
                    "videos/废弃/）：用户说可以就 drama_render_shots(reuse=true, accept=[…]) "
                    "放行，不行就 redo=[…] 重渲。不要替用户决定放行"
                )
            return ToolResult(
                content=(
                    f"{head}\n\n⚠ "
                    + "\n⚠ ".join(why)
                    + "\n不要用 gen_video / gen_videos 自己补（没有参考图，人物会变脸）。\n"
                    f"片段资产：{clips}\n资产 {asset.id}"
                ),
                asset_ref=asset.id,
                meta=state,
            )
        if not compose:
            clips = " ".join(d["asset"] for d in ordered)
            return ToolResult(
                content=f"{head}\n\n片段资产：{clips}\n资产 {asset.id}", asset_ref=asset.id,
                meta=state,
            )

        # 片段**自带音轨** —— seedance 原生出声，台词和环境音都在里面。
        # 拼接会保留它，所以**不要再单独做 TTS 配音**：那等于用合成语音
        # 盖掉模型生成的原声，口型也对不上。
        res = await self.registry_invoke(
            "compose_video",
            {
                "clips": [d["asset"] for d in ordered],
                "out_dir": out_dir,
                "filename": filename or (episode_export_name(episode) if episode else "短剧.mp4"),
                # 片段生成时已按时间线硬切过（每镜 ≤3s），拼接时不能再切，否则台词被剪断
                "keep_whole": True,
            },
        )
        if not res.ok:
            return ToolResult(
                content=f"{head}\n\n⚠ 拼接失败：{res.error}", asset_ref=asset.id,
                meta={**state, "complete": False},
            )
        return ToolResult(
            content=f"{head}\n\n{res.content}", asset_ref=res.asset_ref, meta=state
        )


    # ---------- 参考图包定位与引用匹配 ----------

    def _locate_pack(
        self, rendered_id: str, shots_id: str
    ) -> tuple[str, dict[str, dict[str, str]], list[str], str]:
        """找到本次渲染用的参考图包。返回 (包 id, images, 给人看的说明, 错误)。

        实测 2026-09-18 的失败：分镜引用 (小满-旧连帽衫-[前10集])，包里只有 only=characters
        渲出来的「小满」，逐镜匹配全空 —— 4 段视频「参考 0 张」静默渲完，钱花了脸没保住。
        所以这里把三件事做在花钱之前：包传错了换算、没传自动找、对不上直接报。
        不用实例状态：按集流水会并发调渲染，同一个对象上跑多份。
        """
        notes: list[str] = []
        if rendered_id:
            try:
                content = self.store.content(rendered_id)
            except KeyError as e:
                return "", {}, notes, f"取不到参考图包：{e}"
            images = _load_pack(content)
            if images:
                return rendered_id, images, notes, ""
            # 传成资产库了：换成它的包
            lib, err = parse_assets(content)
            if not err and lib.characters:
                pid, images = self._previous_pack(rendered_id)
                if images:
                    notes.append(f"rendered_id {rendered_id} 是资产库，已换成它的参考图包 {pid}")
                    return pid, images, notes, ""
                return "", {}, notes, (
                    f"{rendered_id} 是资产库不是参考图包，而且这套库还没渲过参考图。"
                    f'先 drama_render_assets(assets_id="{rendered_id}")，再把返回的 id 传进来。'
                )
            return "", {}, notes, (
                f"{rendered_id} 不是 drama_render_assets 的产物（里面没有图片 url）"
            )

        # 没传：按分镜提示词的血缘（parents = [分镜脚本, 资产库]）找这套库最新的包
        try:
            parents = self.store.get(shots_id).parent_ids
        except KeyError:
            parents = []
        for pid_lib in parents[1:]:
            pid, images = self._previous_pack(pid_lib)
            if images:
                notes.append(f"未传 rendered_id，自动使用这套资产库最新的参考图包 {pid}")
                return pid, images, notes, ""
        return "", {}, notes, ""


class _Ref:
    """一次引用的匹配结果：包里的键、url、是不是人物（角色/服装）。"""

    __slots__ = ("key", "url", "person")

    def __init__(self, key: str, url: str, person: bool) -> None:
        self.key, self.url, self.person = key, url, person


def _load_pack(content: str) -> dict[str, dict[str, str]]:
    """参考图包 = {名字: {asset, url, kind?}}。不是这个形状就返回空。"""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {
        k: v for k, v in data.items() if isinstance(v, dict) and isinstance(v.get("url"), str)
    }
    return out if out else {}


def _is_person(entry: dict[str, str], key: str, ref: str) -> bool:
    """包里带 kind 就看 kind（新包）；老包没有 kind，按名字形状猜：
    服装名带 -[集]，退回主形象的也是人物。"""
    kind = entry.get("kind")
    if kind:
        # 三视图是 2026-09-22 之前的包才有的，留在这里只为老包还能渲
        return kind in ("角色", "服装", "三视图")
    return "-[" in ref or key != ref


def _resolve_ref(name: str, images: dict[str, dict[str, str]]) -> _Ref | None:
    """镜头里的引用名 → 包里的一张图。

      精确命中 → 用它
      服装名（角色-服装-[集]）包里没有 → 退回该角色的主形象：脸是一致性的核心，
                                        服装可以靠提示词描述补
      都没有 → None
    """
    if name in images:
        return _Ref(name, images[name]["url"], _is_person(images[name], name, name))
    squeezed = name.replace(" ", "").lower()
    for k, v in images.items():
        if k.replace(" ", "").lower() == squeezed:
            return _Ref(k, v["url"], _is_person(v, k, name))
    if "-" in name:
        base = name.split("-", 1)[0].strip()
        if base and base in images:
            return _Ref(base, images[base]["url"], True)
    return None


# ---------------------------------------------------------------- 引用门（渲染前，花钱之前）

_QUOTED = re.compile(r"[“\"「『‘]([^”\"」』’]*)[”\"」』’]")


def _strip_dialogue(text: str) -> str:
    """去掉台词（引号里的内容）：角色只在别人的台词里被提到，不算出镜。"""
    return _QUOTED.sub(" ", text or "")


def _mentioned_characters(desc: str, lib: Any) -> list[str]:
    """描述里（台词之外）出现的角色名。"""
    if lib is None:
        return []
    body = _strip_dialogue(desc)
    names = [c.name for c in lib.characters if c.name]
    # 「朱锦娘」里有「朱锦」：长名字出现时短名字不算（之前按子串，同框只有朱锦娘也判朱锦没带参考）
    return [n for n in names if matches_character(body, n, names)]


def _looks_like_asset(ref: str, lib: Any) -> bool:
    """括号里的是不是资产引用：服装 ID 形状（-[…]）或含资产库里的名字。
    「(OS)」「(画外音)」这类标注不是 —— 之前一律当引用，包里找不到就整批拦下（2026-09-23 审查）。"""
    if "-[" in ref or lib is None:
        return True
    return any(n and n in ref for n in lib.all_names())


def _covered_characters(shot: Any, resolved: dict[str, Any], lib: Any) -> set[str]:
    """这一段已经带了参考图的角色。"""
    out: set[str] = set()
    for r in shot.refs():
        m = resolved.get(r)
        if m is None or not m.person:
            continue
        c = lib.character_of(r) if lib is not None else None
        out.add(c.name if c is not None else m.key.removesuffix("·三视图"))
    return out


def preflight_refs(
    shots: list[Any], todo: list[int], resolved: dict[str, Any], lib: Any, max_people: int = 0
) -> list[str]:
    """渲染前的引用门（2026-09-20 用户定的规则：没有引用成功的镜头不许生成）。

    只查这次要生成的段（todo）。三种阻断：引用了包里没有的名字；描述里出现的角色没带参考图；
    一段人物参考图多到超过模型参考位。返回问题列表，空 = 放行。
    """
    problems: list[str] = []
    for i in todo:
        s = shots[i]
        tag = f"{s.scene_index} {s.video_name}"
        missing = [r for r in s.refs() if resolved.get(r) is None and _looks_like_asset(r, lib)]
        if missing:
            problems.append(f"{tag}：引用了参考图包里没有的「{'」「'.join(missing)}」")
        mentioned = _mentioned_characters(s.description, lib)
        covered = _covered_characters(s, resolved, lib)
        bare = [n for n in mentioned if n not in covered]
        if bare:
            problems.append(
                f"{tag}：提到了 {'、'.join(bare)} 却没有引用其角色/服装参考图"
                "（描述里要写成 (角色名) 或 (服装ID)）"
            )
        people_urls = {
            resolved[r].url for r in s.refs()
            if resolved.get(r) is not None and resolved[r].person
        }
        if max_people and len(people_urls) > max_people:
            problems.append(
                f"{tag}：人物参考图 {len(people_urls)} 张超过模型参考位（最多 {max_people}），"
                "拆镜或减少同框人物"
            )
    return problems


def _ref_gate_error(problems: list[str], assets_id: str = "") -> str:
    lib = f'assets_id="{assets_id}", ' if assets_id else ""
    return (
        "渲染前引用检查未过，没有发起任何生成（用户规则：没有引用成功的镜头不许生成，"
        "不能生成完再说）：\n  - " + "\n  - ".join(problems) + "\n修法：缺参考图 → "
        f"drama_render_assets({lib}reuse=true) 补齐后重跑本工具；引用名写错 / 角色没引用 → "
        "改分镜提示词（重跑 drama_shots 或 save_draft 新版本）；链接过期或失效 → "
        "drama_refresh_refs 后重跑；人物太多超参考位 → 拆镜。"
        "不要用 gen_video / gen_videos 绕过去补段。"
    )


def _format_note(problems: list[str], fmt: EpisodeFormat) -> str:
    """一集规格（4 分钟 / 开场 15 秒高潮点）的检查结果，给人看。"""
    if not problems:
        return f"\n\n✓ 规格：{fmt.brief()} —— 达标"
    return (
        f"\n\n⚠ 规格检查未过（{fmt.brief()}；已让模型改过一次仍未达标，"
        "可带 note 重跑或人工调）：\n  " + "\n  ".join(problems)
    )


def _finish_problem(resp: Any, what: str) -> str:
    """文本模型这次输出被截断 / 被内容过滤了没有。没问题返回空串。

    2026-09-23 审查：之前一律不看 finish_reason —— 截断和内容过滤都只报「JSON 解析失败」，
    模型当成偶发故障原样重试（真实事故：第 1 集真剧本拆分镜时 gemini 返回
    content_filter，输出 1205 字被截，工具只说解析失败）。
    """
    fr = str(getattr(resp, "finish_reason", "") or "").strip().lower()
    if fr in ("length", "max_tokens"):
        return (
            f"{what}被截断：输出超过了模型单次输出上限（finish_reason={fr}），内容不完整、"
            "没有保存。把输入拆小再来（一次一集），不要原样重试"
        )
    if fr in ("content_filter", "content-filter", "safety", "sensitive", "blocked"):
        return (
            f"{what}被模型服务商的内容安全过滤拦下（finish_reason={fr}），没有产出。"
            "这不是网络问题，原样重试多半还会被拦 —— 检查里面有没有未成年人涉险、暴力、"
            "身体细节描写这类内容，调整措辞后再试，或者如实告诉用户"
        )
    return ""


def _script_problem(text: str) -> str:
    """拆解类工具收到的剧本像不像真剧本。像就返回空串。"""
    from ...harness.context.compaction import find_fold_marks

    s = (text or "").strip()
    if not s:
        return "没有剧本内容：传 script_id（剧本资产 id）或完整的剧本原文"
    if find_fold_marks({"script": s}):
        return (
            "剧本里有上下文折叠留下的占位标记（形如「<N 字已折叠>」），不是真实内容 —— "
            "传剧本资产 id（script_id），别把历史里被折叠的正文抄回来"
        )
    if len(s) < 200:
        return (
            f"剧本只有 {len(s)} 个字，不像完整剧本：拆出来的分镜只能是模型凭空编的。"
            "传剧本资产 id（script_id / script_ids），或把完整正文传进来"
        )
    # 不到 800 字又写着占位字样的，不可能是完整剧本
    if len(s) < 800 and any(h in s for h in STUB_HINTS):
        return (
            "剧本里写着「此处省略 / 略 / 同上」之类的占位字样，不是完整正文 —— "
            "传剧本资产 id（script_id），或把完整正文传进来"
        )
    return ""


REFUSED_STAGE = "参考图·主形象被拒"
# 服务商内容护栏拒绝的常见说法（APIMart / 各家生图接口的报错原文）
_REFUSAL_WORDS = (
    "内容安全", "内容审核", "内容策略", "护栏", "违规", "不合规", "敏感", "未成年",
    "policy", "safety", "sensitive", "moderation", "refus", "prohibited", "not allowed",
    "nsfw", "minor",
)


def _refused(err: str) -> bool:
    low = (err or "").lower()
    return any(w in low for w in _REFUSAL_WORDS)


def _refused_question(names: list[str]) -> str:
    who = "、".join(names)
    return (
        f"这些角色的主形象被生图服务的内容护栏拒了：{who}（儿童角色最常见）。没有主形象，"
        "它们的服装图也生成不了，渲视频时会被引用门拦下。怎么处理？在 a 后面写你的选择：\n"
        "① 给一张本地照片当主形象（写上路径，我用 drama_use_local_ref 登记）\n"
        "② 换一家生图模型再试（写上想换哪家，换之前我会再确认）\n"
        "③ 改写这个角色的外貌描述（写上怎么改）后重渲主形象\n"
        "r = 先不管这几个角色，接着做别的"
    )


def _label_hit(key: tuple[str, str], wants: list[str] | None) -> bool:
    """(场次, 镜头范围) 是否被 redo / accept 列表点到：写全称、只写场次或只写镜头都认。"""
    label = f"{key[0]} {key[1]}"
    for want in wants or []:
        w = str(want or "").strip()
        if w and (w in label or w == key[1] or w == key[0].strip("[]")):
            return True
    return False


def _gate_blocks(gate: str, block: set[str]) -> bool:
    """这道门没过是不是就不许进成片。subtitle_unchecked 跟着 subtitle 走。"""
    return (gate.removesuffix("_unchecked") if gate.endswith("_unchecked") else gate) in block


def _blocking_note(note: str) -> bool:
    """这条质检结论是否意味着这段不能进成片 / 不能被复用。

    ⛔ 开头的是质检门判的「不通过」（哪些门会判 ⛔ 由 drama.gate_block 定，默认字幕与
    人物一致性）。2026-09-23 审查：之前「仍有字幕」只在结果里提一句，照样拼进成片。
    """
    return note.startswith("⛔")


def _voice_note(
    lib: Any,
    plan: AnchorPlan,
    born: dict[str, Anchor],
    no_anchor: list[str],
    anchors_id: str,
    vcfg: dict[str, Any],
) -> str:
    """音色锁定汇总：锁了几段、锚点沿用/新定/过期重定、缺音色卡的角色、没带上锚点的段。"""
    if lib is None:
        return (
            "\n\n⚠ 找不到资产库（分镜提示词与参考图包都没指回它），本次没有锁音色："
            "同一角色的声音会漂。分镜提示词请用 drama_shots 生成。"
        )
    if not plan.spoken:
        return ""
    parts = [f"\n\n音色锁定：{len(plan.spoken)} 个说话角色已锁音色卡"]
    if not vcfg["enabled"]:
        parts.append("；参考视频模式已关（media_models.yaml drama.voice_anchor）")
    if plan.ready:
        parts.append(f"；锚点沿用 {len(plan.ready)} 人（{', '.join(plan.ready)}）")
    if born:
        desc = "、".join(
            f"{a.character} ← {a.scene} 镜{a.clip}" + ("·独白段" if a.solo else "")
            for a in born.values()
        )
        parts.append(f"；本次新定 {len(born)} 人：{desc}")
    replaced = [n for n in plan.stale if n in born]
    kept = [n for n in plan.stale if n not in born]
    if replaced:
        parts.append(f"；{len(replaced)} 人的旧锚点链接过期已接替（{', '.join(replaced)}）")
    if kept:
        parts.append(
            f"；{len(kept)} 人的旧锚点链接过期、本次没有新片段可接替（{', '.join(kept)}），"
            "这些段只锁了音色卡文字；要重定锚点用 reuse=false 重生成或 drama_voice_anchors pin"
        )
    if anchors_id:
        parts.append(f"；锚点表 {anchors_id}（drama_voice_anchors 可查看/指定）")
    if plan.unvoiced:
        parts.append(
            f"\n  ⚠ 没有音色卡的角色：{', '.join(plan.unvoiced)} —— 只能靠锚点片段兜底；"
            "重跑 drama_assets 补音色卡"
        )
    if no_anchor:
        shown = no_anchor[:6]
        more = f" …还有 {len(no_anchor) - 6} 段" if len(no_anchor) > 6 else ""
        parts.append(
            f"\n  ⚠ 未带锚点参考的段（锚点镜头本身、锚点生成失败或参考位不够）："
            f"{'；'.join(shown)}{more}"
        )
    return "".join(parts)


def _realism_note(gate_log: list[dict[str, Any]]) -> str:
    """人物图的真实感处理汇总：清洗/补锚点了几处，校验通过几张、重生成几张、仍不达标几张。"""
    if not gate_log:
        return ""
    cleaned = sum(1 for e in gate_log if e.get("notes"))
    checked = [e for e in gate_log if e.get("checked")]
    lines = [
        f"\n\n真实感：{len(gate_log)} 张人物图提示词已加硬约束（清洗/补瑕疵锚点 {cleaned} 张）"
    ]
    if not checked:
        lines.append("；未做生成后校验（media_models.yaml drama.realism_gate 没开或没有文本网关）")
        return "".join(lines)
    ident = [e for e in gate_log if isinstance(e.get("identity"), dict)]
    if ident:
        ok = [e for e in ident if e["identity"].get("pass")]
        bad = [e for e in ident if not e["identity"].get("pass")]
        lines.append(f"；人物一致性：{len(ok)} 张与主形象一致")
        if bad:
            lines.append(f"，{len(bad)} 张仍不像同一个人：")
            for e in bad:
                why = "；".join(e["identity"].get("issues") or [])[:80]
                lines.append(f"\n  ⚠ {e['name']}（{e['identity'].get('score', -1)}/10）{why}")
    first = [e for e in checked if e.get("pass") and e.get("attempts", 1) == 1]
    retried_ok = [e for e in checked if e.get("pass") and e.get("attempts", 1) > 1]
    failed = [e for e in checked if not e.get("pass")]
    lines.append(
        f"；校验：一次通过 {len(first)} · 重生成后通过 {len(retried_ok)} · 仍不达标 {len(failed)}"
    )
    for e in failed:
        issues = "；".join(e.get("issues") or [])[:120]
        lines.append(f"\n  ⚠ {e['name']}（{e.get('score', -1)} 分）{issues}")
    notes = [f"{e['name']}：{e['note']}" for e in checked if e.get("note")]
    if notes:
        lines.append("\n  校验备注：" + "；".join(notes[:4]))
    return "".join(lines)


def _coverage_note(lib: Any, images: dict[str, Any]) -> str:
    """参考图包对资产库的覆盖率。分镜按服装名/场景名引用，缺哪类一眼能看出来。"""
    costumes = [cos.name for c in lib.characters for cos in c.costumes]
    parts = [
        f"角色 {sum(1 for c in lib.characters if c.name in images)}/{len(lib.characters)}",
        f"服装 {sum(1 for n in costumes if n in images)}/{len(costumes)}",
        f"场景 {sum(1 for s in lib.scenes if s.name in images)}/{len(lib.scenes)}",
        f"道具 {sum(1 for p in lib.props if p.name in images)}/{len(lib.props)}",
    ]
    note = "覆盖：" + " · ".join(parts)
    if costumes and not any(n in images for n in costumes):
        note += "。分镜引用的是服装名，服装一张没有时渲视频会退回角色主形象（保脸不保服装）"
    if lib.scenes and not any(s.name in images for s in lib.scenes):
        note += "；场景没有参考图"
    return note


# ==================================================================== 渲染
#
# 用户定的固定参数（不让模型自由发挥，这些是版式约束不是创作）：
#   角色主形象   3:4   720P  banana2
#   角色分集形象 16:9  720P  banana2  + 必须带主形象作参考图
#   场景 / 道具  16:9  720P  banana2
#   分镜视频     9:16  720p  seedance-2.0  + 引用资产图 + 可选前序视频

# 配置没给时的兜底。正常走 media_models.yaml 的 drama 段。
IMAGE_MODEL = "gpt-image-2"
VIDEO_MODEL = "seedance-2.0"
FALLBACK_SECONDS = 15  # 模型没给时长时的兜底，用户规格里定的


def _unwrap_script(text: str) -> str:
    """剥掉模型可能套的 JSON 外壳。

    drama 角色配了 response_format: json（三段拆解都要严格 JSON），
    写剧本这步就被连带影响，吐出 {"script": "第1集\n\n[日]..."} 这种。
    直接存下去的话，下一步拿到的是 JSON 而不是剧本，
    而且里面的换行是 \n 转义 —— 结构分类器一行都认不出来。
    """
    raw = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if m:
        raw = m.group(1).strip()
    if not raw.startswith("{"):
        return raw
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(data, dict):
        return raw
    for key in ("script", "content", "text", "episodeDesc", "剧本"):
        v = data.get(key)
        if isinstance(v, str) and len(v) > 30:
            return v.strip()
    # 只有一个字符串字段时也认
    vals = [v for v in data.values() if isinstance(v, str) and len(v) > 30]
    return vals[0].strip() if len(vals) == 1 else raw

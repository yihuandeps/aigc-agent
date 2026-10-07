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
import contextlib
import hashlib
import json
import math
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx2 as httpx

from ...envdetect import workspace_root
from ...harness.events.bus import EventBus, EventType
from ...harness.model.media import MediaKind, default_proxy
from ...harness.permission.gate import BatchPass, batch_scope
from ...harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
    ToolSpec,
)
from ..aspect import DEFAULT_ASPECT, parse_aspect, unsupported_note
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
from ..drama.card import latest_storyboards
from ..drama.cards import Card, card_key, find_cards, missing_cards, strip_cards
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
from ..drama.format import (
    DEFAULT_FORMAT,
    DIALOGUE_SHARE_MAX,
    EpisodeFormat,
    card_lines,
    check_shots,
    compression_problems,
    coverage_gaps,
    covered_numbers,
    dialogue_rushed,
    dialogue_timing,
    format_ranges,
    has_flash,
    missing_dialogue,
    number_ranges,
    only_lines,
    opening_note,
    place_cards,
    plan_chunks,
    prompt_problems,
    quoted_lost,
    scene_blocks,
    scene_count,
    shot_lines,
    speech_seconds,
    split_script,
    storyboard_problems,
)
from ..drama.identity import (
    identity_check_messages,
    identity_retry_prompt,
    parse_identity_verdict,
    reference_block,
)
from ..drama.models import as_dict, fix_costume_refs, normalize_ref_parens
from ..drama.refpack import (
    library_chain,
    library_of_shots,
    load_pack,
    pack_identity,
    pack_library,
    same_library,
)
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
    safe_name,
    unique_path,
)
from ..media.no_text import no_text_retry, parse_subtitle_verdict, subtitle_check_messages
from ..media.vision_input import image_data_url
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
from .media import keep_blob, retryable_failure

# 人审挂起的环节名：同一个角色查出多张互不一致的脸、又定不了谁是准的时候，
# 按它在总线上认决策（2026-09-22 用户定：Agent 自动挑，歧义才问）
FACE_CONFLICT_STAGE = "角色面容冲突"
# 定音（2026-09-26）：主角的独白段渲好了，等人听过再固定成音色锚点
CASTING_STAGE = "定音"
# 人定的音色锚点在产物目录里永久存一份（2026-09-27 用户要的）：<产物目录>/音色锚点/<角色>.mp4
ANCHOR_DIR = "音色锚点"
# 质检「没查成」的备注（视觉服务不通 / 没本地副本）：按没过算，重跑时先补查、过了复用不重付
_SUB_UNCHECKED = "字幕没查成"
_ID_UNCHECKED = "人物一致没查成"

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


# drama_render_assets 的 only → 要生成哪几类（服装依赖主形象，要服装就连带主形象）
_RENDER_ONLY: dict[str, set[str]] = {
    "all": {"characters", "costumes", "scenes", "props"},
    "characters": {"characters"},
    "costumes": {"characters", "costumes"},
    "scenes": {"scenes"},
    "props": {"props"},
}
# 渲参考图的超时按张数给（2026-09-29 审查 1.4）：9-21 实测 58 张 1843 秒、约 32 秒一张，
# 每张按 60 秒给，留出质检重生成的余量；不少于工具声明的 1 小时
_RENDER_ASSETS_PER_IMAGE_S = 60.0


def _batch_key(messages: list[dict[str, Any]]) -> str:
    """一批的输入指纹：发给模型的完整消息一样，上次出好的结果就能沿用。"""
    raw = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


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
        # 视频画幅（2026-09-25 用户要的：能出 16:9）：项目设置（/ratio），app.py 里套；默认竖屏
        self.aspect_ratio = DEFAULT_ASPECT
        # 本地素材托管（config/hosting.yaml）：用户的本地图当参考、过期链接重新上传都靠它
        self.hosting = hosting
        self.files: Any = None  # FileFunctions：drama_use_local_ref 登记本地图（app.py 里装）
        # 渲染要调 gen_image / gen_video，走同一个注册表 ——
        # 权限、成本记账、事件日志才不会分叉出第二套。
        self.registry = registry
        # 批量渲染的进度汇报（BATCH_PROGRESS）发到这里，CLI 进度窗订阅它
        self.bus = bus
        # 媒体任务台账（app.py 接 MediaGateway.ledger）：重渲一段之前先按段指纹找上次没等到的
        # 任务（含质检重生成的那次）取回，不再白付（2026-09-26）
        self.task_ledger: Any = None
        # 定音渲好、等人听的独白段：角色 → 锚点（人采纳后 on_event 固定）
        self._pending_casting: dict[str, Anchor] = {}
        # 镜头超 3 秒拦不拦成片（项目设置，/cut，2026-09-27 用户定的：先只标 ⚠，第 1 集看过
        # 检测准不准再定）。True = 超了自动重生成、仍超 ⛔ 不进成片
        self.cut_block = False
        # 做到一半的进度（分批出的提示词、渲好的参考图）：(类别, 键) → 内容，见 _scratch_*
        self._scratch: dict[tuple[str, str], Any] = {}
        self._specs: dict[str, ToolSpec] = {}
        self._build()

    async def _progress(self, stage: str, done: int, total: int, item: str = "") -> None:
        """批量渲染进度汇报。没接 bus 就静默（测试/脚本环境）。"""
        if self.bus is not None:
            await self.bus.emit(
                EventType.BATCH_PROGRESS, stage=stage, done=done, total=total, item=item
            )

    # ---------- 做到一半的进度：下次接着用，不重付（2026-09-29 审查 1.4、2.4） ----------
    # 分批出提示词一批失败、渲参考图撞上超时 / 被 /stop，之前已经付过钱的那部分全丢，重跑全部
    # 重付。这里按 (类别, 键) 记下做完的部分：内存一份；资产库有目录时再落一份 JSON，换了会话
    # 也接得上。不进资产库：半成品进了库，流水线会当成「参考图渲完了」去渲视频。

    def _scratch_file(self, kind: str, key: str) -> Path | None:
        root = getattr(self.store, "root", None)
        return Path(root) / "progress" / kind / f"{key}.json" if root else None

    def _scratch_load(self, kind: str, key: str) -> Any:
        if (kind, key) in self._scratch:
            return self._scratch[(kind, key)]
        p = self._scratch_file(kind, key)
        if p is None:
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        self._scratch[(kind, key)] = data
        return data

    def _scratch_save(self, kind: str, key: str, data: Any) -> None:
        self._scratch[(kind, key)] = data
        p = self._scratch_file(kind, key)
        if p is None:
            return
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(p)
        except OSError:
            pass  # 落不了盘只是换了会话接不上，这次照常

    def _scratch_drop(self, kind: str, key: str) -> None:
        self._scratch.pop((kind, key), None)
        p = self._scratch_file(kind, key)
        if p is not None:
            with contextlib.suppress(OSError):
                p.unlink(missing_ok=True)

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
                "写完**把剧本给用户看，他确认了再进拆解链**（drama_storyboard → drama_assets）"
                "—— 一句话想法写出来的剧本和他想的不一定是一回事，往下拆、往下渲都很贵。"
                "按集存好，一集一份资产。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "idea": {"type": "string", "description": "一句话想法或题材方向"},
                    "episodes": {"type": "integer", "description": "写几集，默认 1"},
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
                "全剧的剧本传 script_ids（按集号顺序的剧本资产 id 列表），不要把几万字塞进参数。\n"
                "**一部剧一份资产库**：全剧各集一次传完。失败了原样再调一次（输出坏了工具会先自己修，"
                "修不好自动重发），不要拆成几段分别生成，也不要手工合并资产库 JSON。\n"
                "**渲过参考图就定稿了**：之后再调（续写了新集、补了角色）只补新集里新出现的角色 / "
                "服装 / 场景 / 道具，已有的原样沿用、参考图也沿用，不会全员换脸。"
                "真要整份重做才传 rebuild=true（会先问用户）。"
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
                    "rebuild": {
                        "type": "boolean",
                        "description": "资产库已定稿时整份重做（重新设计全部角色 / 服装 / 场景，"
                        "描述变了的参考图全要重生成）。只在用户明确要重做时传；会当场问用户",
                    },
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
                "那种错看着没毛病，但生视频时找不到参考图，人物就会变脸。服装 ID 的集数范围被写短了"
                "（-[1] 写成库里没有的样子）会自动换回库里的全名，不用在 note 里列全名。\n"
                "**分镜的每个镜头都会写进提示词**：一集分镜太长会自动按场分批生成、漏掉的镜头自动补写"
                "一次，补不上就不保存；总时长跟着分镜走（不会为了凑一集的标准时长合并或删镜头）。"
                "不要自己手动把分镜切成几份再调。"
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
                "同一层里无依赖的图**并发**生成。"
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
                "无引入依赖的镜头**并发**生成，带引入的按依赖排队。"
                "画幅按项目设置（用户用 /ratio 选，默认竖屏 9:16），只这一次不同就传 aspect_ratio；"
                "720p。**很慢很贵**，建议先用 limit 跑一两段看方向。"
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
                    "aspect_ratio": {
                        "type": "string",
                        "description": "画幅，如 16:9 横屏 / 9:16 竖屏。留空用项目设置（/ratio），"
                        "只在用户要求这一次不同时传",
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
                        "也可以写片段资产 id：redo 换掉的旧版本（在 videos/废弃/ 里）新版更差时，"
                        "用户说还是要旧的，就写旧片段的 id 把它找回来。不要替用户决定",
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
                "到素材托管（用户配的图床），换成新链接并生成新的参考图包。"
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
                "存储拿到公网链接，写进这套资产库最新的参考图包。之后 "
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
                "默认 drama_render_assets 跑完自动查一遍。"
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

        self._specs["drama_voice_casting"] = ToolSpec(
            name="drama_voice_casting",
            summary="定音：主角各渲一段独白（TA 的参考图 + TA 的一句台词），人听过后定成音色锚点",
            permission=PermissionLevel.COMPUTE,
            timeout=3600,
            description=(
                "**渲第一集之前做一次**：每个主角先渲一段约 5 秒的独白，"
                "用户听过、采纳了就固定成这个角色的音色锚点 —— 之后每一集他开口都和这段对齐。"
                "不做的话，锚点是渲第 1 集时自动挑的段，并行渲的两集还可能各定各的。"
                "characters 省略 = 台词最多的前几个角色（max_characters，默认 6）。"
                "渲之前会报一次价；渲完停下来请用户听：采纳就固定（采纳附言里点名的角色除外），"
                "打回就都不固定。某几个要重来，再调一次只写那几个角色。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "assets_id": {
                        "type": "string", "description": "资产库 id（角色和音色卡在里面）",
                    },
                    "characters": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "只给这几个角色定音；省略 = 台词最多的前几个",
                    },
                    "max_characters": {
                        "type": "integer", "description": "省略 characters 时取几个，默认 6",
                    },
                },
                "required": ["assets_id"],
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

    def _merge_save_anchors(
        self, new_anchors: dict[str, Anchor], parents: list[str], override_pinned: bool = False
    ) -> str:
        """把这次新定的锚点并进**最新的**锚点表再存。

        渲染这段时间里别的集可能写过锚点表（/auto 两集并行）：之前开渲时读、结束时整表覆盖，
        后结束的那集把先结束那集新定的锚点冲掉（2026-09-23 审查）。人手动 pin 的不覆盖 ——
        除非这次本身就是人定的（定音采纳，override_pinned）。
        """
        latest = self._load_anchors()
        changed = False
        for name, a in new_anchors.items():
            old = latest.get(name)
            if old is not None and old.pinned and not override_pinned:
                continue
            latest[name] = a
            changed = True
        # 一个都没换（全是人定的）：不再存一份内容一样的锚点表
        return self._save_anchors(latest, parents) if changed else ""

    # ---------- 人定的音色锚点：本地永久存一份（2026-09-27 用户要的） ----------
    # 生成链接约 24 小时失效；产物目录里的 videos/ 是用户的，会被整理掉。人听过、定下的声音
    # 得有一份不会丢的本地文件：链接过期后从它重新上传（配了托管的话），重新定音也不用重渲。

    def _output_root(self) -> Path | None:
        root = getattr(self.files, "output_root", None) if self.files is not None else None
        return Path(root) if root else None

    def _anchor_home(self, name: str) -> Path | None:
        """人定的锚点在产物目录里存哪：<产物目录>/音色锚点/<角色>.mp4。没接产物目录返回 None。"""
        root = self._output_root()
        return root / ANCHOR_DIR / f"{safe_name(name)}.mp4" if root is not None else None

    def _keep_anchor_files(self, anchors: dict[str, Anchor]) -> list[str]:
        """把人定的锚点片段存进 <产物目录>/音色锚点/，Agent 自己在资产库 blobs/ 再留一份（产物目录
        被整理掉也还在）。片段资产的本地副本改指到这里。同一个角色换了一段（重新定音），旧文件挪进
        音色锚点/旧版/，不删。返回给人看的几行（存到哪 / 哪个没存成）。"""
        rows: list[str] = []
        for name, a in anchors.items():
            try:
                clip = self.store.get(a.asset)
            except KeyError:
                rows.append(f"⚠ {name}：片段 {a.asset} 不在资产库里，没存成")
                continue
            src = local_copy(clip)
            if src is None:
                rows.append(
                    f"⚠ {name}：本地没有这段的副本，没存成 —— 链接还没过期就先 "
                    f'fetch_asset_file(asset_ids=["{clip.id}"]) 下载下来再定一次'
                )
                continue
            dest = self._anchor_home(name)
            if dest is not None:
                dest = dest.with_suffix(src.suffix or ".mp4")
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    same = dest.exists() and dest.resolve() == src.resolve()
                    if dest.exists() and not same:
                        old = dest.parent / "旧版"
                        old.mkdir(exist_ok=True)
                        stamp = time.strftime("%Y%m%d-%H%M%S")
                        backup = unique_path(old, f"{dest.stem}-{stamp}{dest.suffix}")
                        shutil.move(str(dest), str(backup))
                    if not same:
                        shutil.copy2(src, dest)
                except OSError as e:
                    rows.append(f"⚠ {name}：存到 {dest.parent} 失败（{e}）")
                    dest = None
            blob = self._anchor_blob(dest or src, clip.id)
            gp = clip.gen_params
            if dest is not None:
                gp["local"] = str(dest)
            if blob is not None:
                gp["blob"] = str(blob)
            gp["voice_anchor"] = name
            self.store.put(clip)
            if dest is None and blob is None:
                rows.append(f"⚠ {name}：只剩 {src} 这一份，没能另存")
            else:
                rows.append(f"♪ {name}：{dest or blob}")
        return rows

    def _anchor_blob(self, src: Path, asset_id: str) -> Path | None:
        """Agent 自己留的那份：同盘硬链接，跨盘复制 —— 锚点只有几秒，多大都留（keep_blob 跨盘
        超过 50MB 不复制）。"""
        blob = keep_blob(self.store, src, asset_id)
        if blob is not None:
            return blob
        root = getattr(self.store, "root", None)
        if root is None:
            return None
        try:
            d = Path(root) / "blobs"
            d.mkdir(parents=True, exist_ok=True)
            dest = d / f"{asset_id}{src.suffix}"
            if not dest.exists():
                shutil.copy2(src, dest)
            return dest
        except OSError:
            return None

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
                else f"⚠ 音色锚点 {name}：重新托管失败（{err}），"
                + (
                    "这一集临时用本集的独白段，你定的锚点不动（本地副本还在）" if a.pinned
                    else "这次会重新定锚点"
                )
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
        pack_sig: str = "",
        aspect: str = DEFAULT_ASPECT,
        fps: dict[tuple[str, str], str] | None = None,
        recheck: list[tuple[tuple[str, str], str]] | None = None,
    ) -> dict[tuple[str, str], str]:
        """这份提示词此前渲过、**过了质检门**的片段：(场次, 镜头范围) → 片段资产 id，最新的优先。

        重跑只补失败/缺的段 —— 之前一段网络错误就要整集重生成，成功的段也跟着再付一遍钱。

        2026-09-23 审查后改的三处：
          · 不再只认整批跑完才写的片段索引：每段视频生成时就带着标签（所属提示词、场次、
            镜头、参考图包），渲到一半被 /stop、超时、崩溃，已付费的段下次照样能复用
          · 参考图包换了（换脸、补了服装）就不复用旧片段 —— 否则新脸永远到不了视频
          · redo 里点名的段强制重渲（之前结果里让人「去掉问题段」，却没有参数能做到）
        2026-09-26：fps（场次, 镜头范围 → 段指纹）给了就先按指纹找 —— 这一段自己的内容和参考图
        没变就复用，不管片段属于哪份提示词、哪个参考图包。recheck 给了，就把「只因为字幕没查成」
        被判 ⛔ 的段放进去，调用方补下载重查（过了就复用，不重付）。
        """
        cand: dict[tuple[str, str], tuple[int, str]] = {}

        def offer(key: tuple[str, str], seq: int, cid: str) -> None:
            if not cid or not self._clip_playable(cid):
                return
            if key not in cand or seq > cand[key][0]:
                cand[key] = (seq, cid)

        def usable(key: tuple[str, str], a: Any, tags: dict[str, Any]) -> bool:
            """过了质检门的才复用；没过的除非用户点名放行（accept）。被换掉的废片永远不用。"""
            if tags.get("accepted") and not tags.get("rejected"):
                return True
            if tags.get("rejected"):
                return False
            if _label_hit(key, accept):
                tags["accepted"] = True
                tags["accepted_by"] = "human"
                a.gen_params["tags"] = tags
                self.store.put(a)
                return True
            if recheck is not None and _only_unchecked(tags) and self._clip_playable(a.id):
                recheck.append((key, a.id))
            return False

        # ⓪' 用户按片段 id 点名要回来的（redo 换掉的旧版本新版更差时，2026-09-26）：
        #     恢复成可用，压过同一段后来渲的版本
        for raw in accept or []:
            aid = str(raw).strip()
            if not aid.startswith("as_"):
                continue
            try:
                a = self.store.get(aid)
            except KeyError:
                continue
            tags = dict((a.gen_params or {}).get("tags") or {})
            if a.type is not AssetType.VIDEO or tags.get("shots_id") is None:
                continue
            tags.pop("rejected", None)
            tags.update({"accepted": True, "accepted_by": "human"})
            a.gen_params["tags"] = tags
            a.status = AssetStatus.ACTIVE
            a.status_note = "用户点名要回（accept）"
            self.store.put(a)
            key = (str(tags.get("scene") or ""), str(tags.get("name") or ""))
            offer(key, 10**12 + a.seq, a.id)

        # ⓪ 按段指纹（2026-09-26）
        if fps:
            by_fp = {fp: key for key, fp in fps.items() if fp}
            for a in self.store.find(type_=AssetType.VIDEO):
                tags = (a.gen_params or {}).get("tags") or {}
                if not isinstance(tags, dict):
                    continue
                key = by_fp.get(str(tags.get("fp") or ""))
                if key is not None and usable(key, a, tags):
                    offer(key, a.seq, a.id)

        sig_cache: dict[str, str] = {}

        def same_pack(tag_pack: str, tag_sig: str) -> bool:
            """片段渲时用的包和这次的是不是同一套图。按内容签名比，不按包 id：重新托管 /
            刷新链接会新建一个包 id 但图没换，之前按 id 比，托管一次就整集重渲、重付
            （2026-09-24 审查）。老标签只记了包 id 的，读那个包算签名再比。"""
            if not pack_id:
                return True
            if tag_sig and pack_sig:
                return tag_sig == pack_sig
            if not tag_pack or tag_pack == pack_id:
                return True
            if not pack_sig:
                return False
            if tag_pack not in sig_cache:
                try:
                    sig_cache[tag_pack] = pack_identity(load_pack(self.store.content(tag_pack)))
                except KeyError:
                    sig_cache[tag_pack] = ""
            return sig_cache[tag_pack] == pack_sig

        # ① 每段自带标签的片段（新口径）
        for a in self.store.find(type_=AssetType.VIDEO):
            tags = (a.gen_params or {}).get("tags") or {}
            if not isinstance(tags, dict) or tags.get("shots_id") != shots_id:
                continue
            if not same_pack(str(tags.get("pack") or ""), str(tags.get("pack_sig") or "")):
                continue
            # 画幅不同的片段不复用（之前没记画幅的都是竖屏 9:16）
            if str(tags.get("aspect") or DEFAULT_ASPECT) != aspect:
                continue
            key = (str(tags.get("scene") or ""), str(tags.get("name") or ""))
            if fps and tags.get("fp"):
                continue  # 带指纹的片段上面已经按指纹认过（指纹对不上 = 这段改过了，不复用）
            # 没过质检门（带字幕 / 变脸）的段不复用 —— 除非用户看过后点名放行；
            # 被换掉的废片（rejected）永远不复用
            if usable(key, a, tags):
                offer(key, a.seq, a.id)

        # ② 老口径：整批跑完落的片段索引
        for a in self.store.find(creator="tool:drama_render_shots"):
            if not a.parent_ids or a.parent_ids[0] != shots_id:
                continue
            if len(a.parent_ids) > 1 and not same_pack(a.parent_ids[1], ""):
                continue
            if str(a.gen_params.get("aspect") or DEFAULT_ASPECT) != aspect:
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
                # 点名重渲的旧段：标 rejected、挪进 废弃/，主文件名让给新段（之前旧段仍
                # active、占着名字，新段落成 -v2，按文件名剪片会剪进被替换的段）
                self._retire_version(out.pop(key), "人点名重渲（redo）")
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
        blocks = {x.strip() for x in block.split(",") if x.strip()}
        if self.cut_block:
            blocks.add("cut")  # 这个项目用 /cut 拦 打开了
        return {"block": blocks, "max_regen": regen, "frames": frames}

    async def _recover_segment(self, fp: str, args: dict[str, Any]) -> tuple[Any, str]:
        """这一段上次提交过、没等到结果的任务（轮询超时 / 进程被打断，服务端多半已生成、已计费）：
        按段指纹在任务台账里找，取回成片段。返回 (工具结果, 说明)；没有可取回的返回 (None, "")。"""
        ledger = self.task_ledger
        if ledger is None or not fp or not hasattr(ledger, "recoverable_where"):
            return None, ""
        rec = ledger.recoverable_where(
            lambda r: r.kind == "video" and ((r.params or {}).get("tags") or {}).get("fp") == fp
        )
        if rec is None:
            return None, ""
        ledger.recovered(rec.task_id)
        got = await self.registry_invoke(
            "media_recover",
            {"task_id": rec.task_id, "summary": str(args.get("summary") or ""),
             "local_name": str(args.get("local_name") or "")},
        )
        if not getattr(got, "ok", False) or not getattr(got, "asset_ref", ""):
            return None, ""
        return got, f"取回了上次没等到的任务 {rec.task_id}（已付过费，没重新生成）"

    async def _gen_clip(
        self,
        args: dict[str, Any],
        retries: int,
        sub_gate: bool,
        sub_retries: int,
        id_refs: list[tuple[str, str]] | None = None,
        first: Any = None,
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
        # first：取回的旧任务（已付过费）—— 不再生成第一版，直接过质检门
        r = first if first is not None else await self._invoke_video(args, retries)
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
                if getattr(v, "unchecked", False):
                    # 没查成不算过（2026-09-27 用户定的）：和字幕的「查不成」一样，
                    # 不进成片、重跑补查
                    issues["identity_unchecked"] = v.note or "没查成"
                elif v.note:
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
        if "identity_unchecked" in final:
            notes.append(
                ("⛔ " if _gate_blocks("identity_unchecked", gc["block"]) else "⚠ ")
                + f"{_ID_UNCHECKED}（{final['identity_unchecked']}）"
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
        """量这段视频里最长的镜头有几秒。返回 (秒数, 检查不了的原因)。

        没本地副本也要说一声（2026-09-24 审查：之前静默跳过，结果里看不出这段根本没量）。"""
        local = self._local_video(asset_id)
        if local is None:
            return None, "没有本地副本，镜头时长没查"
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
            "save_draft 新版本）再渲染。短视频 / 广告里要出现剧中角色：给 short_video_produce "
            "传 ref_images（角色主形象的资产 id）或 shot_refs，逐镜带着参考图生成。"
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
        """参考图 → 给视觉模型看的 data URL：优先本地副本（远端链接会过期），没有就原样给链接
        （由 _vision_ref 下载转换，不直接发）。"""
        if ref.startswith("as_"):
            return self._image_payload(ref, "")
        a = self._asset_by_uri(ref)
        if a is not None:
            return self._image_payload(a.id, ref)
        return ref

    async def _vision_ref(self, ref: str) -> tuple[str, str]:
        """参考图 → (data URL, 拿不到的原因)。链接和本地路径都在这里转成 data URL：远端链接
        视觉模型多半拉不到，路径更不能当链接发（2026-09-29 审查 1.2）。"""
        return await image_data_url(self._ref_payload(ref))

    async def _vision_image(self, asset_id: str, url: str) -> tuple[str, str]:
        """要校验的图 → (data URL, 拿不到的原因)。本地副本优先，没有就把链接下载下来再转。"""
        return await image_data_url(self._image_payload(asset_id, url))

    async def _check_identity(
        self, target: str, refs: list[tuple[str, str]], is_video: bool, pass_score: int
    ) -> Any:
        """把参考图和生成结果一起给视觉模型，判是不是同一个人。

        做不了：配置里就没开（没有文本网关 / 没配视觉角色）→ note 说明、按通过；这次没查成（参考图
        拿不到、没本地副本、抽不出帧、调用失败、输出读不出来）→ unchecked=True，视频段按「没过」算、
        重跑时补查（2026-09-27 用户定的：之前一律按通过，没核对过的脸悄悄进成片）。"""
        from ..drama.identity import IdentityVerdict

        if self.gateway is None:
            return IdentityVerdict(note="没有文本网关，跳过一致性校验")
        ref_parts: list[tuple[str, str]] = []
        missing: list[str] = []
        for label, u in refs:
            payload, why = await self._vision_ref(u)
            if payload:
                ref_parts.append((label, payload))
            else:
                missing.append(why)
        if not ref_parts:
            why = missing[0] if missing else "没有参考图"
            return IdentityVerdict(note=f"参考图拿不到（{why}）", unchecked=True)
        if is_video:
            local = self._local_video(target)
            if local is None:
                return IdentityVerdict(note="片段没有本地副本", unchecked=True)
            # 抽几帧跟字幕门同一个配置（check_frames）：之前一致性门固定 4 帧（2026-09-26）
            frames = await self._frames_of(local, self._gate_cfg()["frames"])
            if not frames:
                return IdentityVerdict(note="抽不出画面帧（检查 ffmpeg）", unchecked=True)
            targets = ["data:image/jpeg;base64," + base64.b64encode(b).decode() for b in frames]
        else:
            payload, why = await self._vision_image(target, "")
            if not payload:
                return IdentityVerdict(note=f"生成图拿不到（{why}）", unchecked=True)
            targets = [payload]
        try:
            resp = await self.gateway.chat(
                REALISM_ROLE, identity_check_messages(ref_parts, targets, is_video)
            )
        except KeyError:
            return IdentityVerdict(note=f"models.yaml 没配 {REALISM_ROLE} 角色，跳过一致性校验")
        except Exception as e:  # noqa: BLE001
            return IdentityVerdict(note=f"一致性校验调用失败：{type(e).__name__}", unchecked=True)
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

    def render_need(self, shots_id: str, episode: int = 0) -> tuple[int, float]:
        """渲这一集最坏要生成几段、几秒：扣掉这份提示词已经渲过、过了质检的段，每段按质检最多
        再生成几次算。按集流水没人可问时的预算预检用（2026-09-26：之前不算重生成，还把能复用的段
        算进去）。按指纹跨提示词复用的段这里不扣 —— 预检宁可多算。"""
        try:
            shots, err = parse_shots(self.store.content(shots_id))
        except KeyError:
            return 1, float(FALLBACK_SECONDS)
        if err or not shots:
            return 1, float(FALLBACK_SECONDS)
        if episode:
            shots = [
                s for s in shots if s.scene_index.strip("[]").startswith(f"第{episode}集-")
            ] or shots
        done: set[tuple[str, str]] = set()
        for a in self.store.find(type_=AssetType.VIDEO):
            tags = (a.gen_params or {}).get("tags") or {}
            if (
                isinstance(tags, dict) and tags.get("shots_id") == shots_id
                and tags.get("accepted") and not tags.get("rejected")
            ):
                done.add((str(tags.get("scene") or ""), str(tags.get("name") or "")))
        todo = [s for s in shots if (s.scene_index, s.video_name) not in done]
        k = 1 + self._worst_video_regen()
        return len(todo) * k, float(sum(s.seconds or FALLBACK_SECONDS for s in todo)) * k

    @staticmethod
    def _short_on_refs(
        shots: list[Any],
        todo: list[int],
        resolved: dict[str, Any],
        images: dict[str, dict[str, str]],
        owner_of: dict[str, str],
        scene_names: set[str],
        speakers_by_shot: list[dict[str, int]],
        anchor_urls: dict[str, str],
        plan: AnchorPlan,
        vmax: int,
        max_refs: int,
    ) -> list[str]:
        """这次要渲的段里，参考位不够、要省掉场景 / 道具参考图的：「场次 镜头：省了哪几张」。
        花钱之前放进报价（2026-09-27 用户定的：「这段会少用哪张参考图」一律先让你知道；之前渲完
        才在备注里看到。seedance 一段 9 个参考，参考视频占 3 个，三人同框 3 张服装 + 3 张脸
        就满了）。

        参考视频按渲染时的规矩估：前序引入（排在前面的同场景段）+ 说话角色的音色锚点（已有的，
        或这批里更早的段定下的），不超过 vmax。"""
        if not max_refs:
            return []
        first_at: dict[str, int] = {}
        for i, s in enumerate(shots):
            first_at.setdefault(s.scene_index.strip("[]"), i)
        out: list[str] = []
        for i in todo:
            s = shots[i]
            n_carry = sum(1 for c in s.carries() if first_at.get(c, len(shots)) < i)
            n_anchor = sum(
                1 for n in speakers_by_shot[i]
                if n in anchor_urls or plan.births.get(n, i) != i
            )
            n_videos = min(vmax, n_carry + n_anchor)
            _, gone = _trim_refs(
                _segment_refs(s, resolved, images, owner_of, scene_names), n_videos, max_refs
            )
            if gone:
                out.append(f"{s.scene_index} {s.video_name}：{'、'.join(gone)}")
        return out

    def _passed_keys(self, episode: int) -> set[tuple[str, str]]:
        """以前渲过、过了质检门的段：(场次, 镜头范围)。报价时把「这次要重付的已通过段」说出来。"""
        out: set[tuple[str, str]] = set()
        for a in self.store.find(type_=AssetType.VIDEO):
            tags = (a.gen_params or {}).get("tags") or {}
            if not isinstance(tags, dict) or not tags.get("accepted") or tags.get("rejected"):
                continue
            if episode and int(tags.get("episode") or 0) not in (0, episode):
                continue
            out.add((str(tags.get("scene") or ""), str(tags.get("name") or "")))
        return out

    async def _recheck_unchecked(
        self,
        cands: list[tuple[tuple[str, str], str]],
        id_refs_of: Any = None,
    ) -> dict[tuple[str, str], str]:
        """上次只因为「没查成」被判 ⛔ 的段（视觉接口抖了 / 本地副本没下载下来）：先补下载、
        再把没查成的那几道门查一次（字幕：没字；人物一致：对得上），都过了就记成通过、直接复用。
        之前重跑只复用通过的段，这些段整段重新生成、重新付费，其实远端链接都还在
        （字幕 2026-09-26、人物一致 2026-09-27 用户定的）。
        id_refs_of：(场次, 镜头范围) → 这段的人物参考 [(名字, url)]。
        返回 (场次, 镜头范围) → 片段 id。"""
        out: dict[tuple[str, str], str] = {}
        if not cands:
            return out
        sub_on, _ = self._subtitle_cfg()
        id_on, _, pass_score = self._identity_cfg()
        for key, aid in cands:
            if key in out:
                continue
            try:
                notes = [str(n) for n in (self.store.get(aid).gen_params.get("tags") or {})
                         .get("notes") or []]
            except KeyError:
                continue
            # 那道门现在关着：没有要补查的（和新渲的段一样，关着的门不查）
            need_sub = sub_on and any(_SUB_UNCHECKED in n for n in notes)
            need_id = id_on and any(_ID_UNCHECKED in n for n in notes)
            refs = list(id_refs_of(key) or []) if (need_id and id_refs_of is not None) else []
            if need_id and not refs:
                continue  # 这段的人物参考找不到了：没法补查，照常重渲
            if self._local_video(aid) is None and self.registry is not None:
                await self.registry_invoke("fetch_asset_file", {"asset_ids": [aid]})
            if self._local_video(aid) is None:
                continue
            fixed: list[str] = []
            if need_sub:
                found, why = await self._check_subtitles(aid)
                if why or found:
                    continue
                fixed.append("重查字幕：没有字（上次没查成）")
            if need_id:
                v = await self._check_identity(aid, refs, is_video=True, pass_score=pass_score)
                if getattr(v, "unchecked", False) or not v.passed:
                    continue
                fixed.append(f"重查人物一致：{v.score}/10（上次没查成）")
            try:
                a = self.store.get(aid)
            except KeyError:
                continue
            tags = dict((a.gen_params or {}).get("tags") or {})
            tags["accepted"] = True
            tags["notes"] = [n for n in (tags.get("notes") or []) if not _is_unchecked(n)] + fixed
            a.gen_params["tags"] = tags
            self.store.put(a)
            out[key] = aid
        return out

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
        # 只看这个项目自己的图：没迁移的旧剧（没有项目键）哪个项目都看得见，两部剧角色同名
        # （唐僧、悟空…）时旧剧的图会被当成这部剧的「另一张脸」挪进回收站（2026-09-26）
        scope = str(getattr(self.store, "project", "") or "")
        for a in self.store.find(type_=AssetType.IMAGE, newest_first=False):
            if a.id in seen:
                continue
            if scope and a.project != scope:
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
        # 有本地副本就传资产 id，_ref_payload 才会转成 data URL；之前传的是本地路径，原样当
        # 图片链接发给视觉模型必然失败，每张都记成「比不了」，审查从没真正比过（2026-09-29 审查）
        refs = [(f"角色「{who}」本人", base.asset_id if base.local else base.url)]
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
            hosted = self.hosting is not None and getattr(self.hosting, "enabled", False)
            for a in anchors.values():
                local = None
                try:
                    local = local_copy(self.store.get(a.asset))
                except KeyError:
                    pass
                if a.pinned:
                    lines.append(
                        f"  ♪ {a.character}（人定的）本地：{local}" if local
                        else f"  ⚠ {a.character}（人定的）本地没有副本：链接过期后就用不上了"
                    )
                _, why = self._anchor_url(a, vcfg)
                if not why:
                    continue
                if not a.pinned:
                    lines.append(f"  ⚠ {a.character}：{why}，下次渲染会用新片段接替")
                elif local and hosted:
                    lines.append(f"  · {a.character}：{why}，下次渲染前会从本地副本重新上传")
                else:
                    lines.append(
                        f"  ⚠ {a.character}：{why}；"
                        + ("没配托管（config/hosting.yaml），" if local else "")
                        + "渲染时这个角色只能临时用本集的独白段，你定的锚点不动"
                    )
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
            kept = self._keep_anchor_files({character: anchors[character]})
            return ToolResult(
                content=f"已把 {character} 的声音锚在 {clip_id}（{clip.summary}）。锚点表 {aid}\n"
                + "本地永久存一份：" + "；".join(kept),
                asset_ref=aid,
            )
        return ToolResult(ok=False, error=f"未知 action {action!r}，用 list / pin / clear")

    # ---------- 定音（2026-09-26 用户定的：主角先渲一段独白，人听过后固定成音色锚点） ----------

    def _character_lines(self, lib: Any) -> dict[str, list[str]]:
        """各角色在这个项目最新分镜里的台词（按「引号前最后一个出现的角色名」认说话人）。"""
        names = sorted({c.name for c in lib.characters}, key=len, reverse=True)
        out: dict[str, list[str]] = {}
        for ep, sid in latest_storyboards(self.store).items():
            try:
                eps, err = parse_episodes(self.store.content(sid))
            except KeyError:
                continue
            for e in eps if not err else []:
                if e.index != ep:
                    continue
                for ln in shot_lines(e.desc):
                    m = _QUOTED.search(ln)
                    if not m:
                        continue
                    head = ln[: m.start()]
                    who = max(
                        ((head.rfind(n), n) for n in names if n in head), default=(-1, "")
                    )[1]
                    if who:
                        out.setdefault(who, []).append(m.group(1).strip())
        return out

    async def _fn_drama_voice_casting(
        self, assets_id: str, characters: list[str] | None = None, max_characters: int = 6
    ) -> ToolResult:
        try:
            lib, err = parse_assets(self.store.content(assets_id))
        except KeyError:
            return ToolResult(ok=False, error=f"没有资产库 {assets_id}")
        if err:
            return ToolResult(ok=False, error=f"资产库读不出来：{err}")
        images: dict[str, dict[str, str]] = {}
        for lid in library_chain(self.store, assets_id):
            _, images = self._previous_pack(lid)
            if images:
                break
        if not images:
            return ToolResult(
                ok=False, meta={"charged": False},
                error="这套资产库还没渲参考图：先 drama_render_assets，定音要拿主形象当参考",
            )
        said = self._character_lines(lib)
        if characters:
            want = [c.name for x in characters if (c := lib.character_of(str(x))) is not None]
        else:
            ranked = sorted(((len(v), n) for n, v in said.items() if v), reverse=True)
            want = [n for _, n in ranked][: max(1, int(max_characters or 6))]
        jobs: list[tuple[Any, str, str]] = []
        skipped: list[str] = []
        for name in dict.fromkeys(want):
            ch = lib.character_of(name)
            url = str((images.get(name) or {}).get("url") or "")
            line = _casting_line(said.get(name, []))
            if ch is None or not url:
                skipped.append(f"{name}：参考图包里没有主形象")
            elif not line:
                skipped.append(f"{name}：分镜里没找到 TA 的台词")
            else:
                jobs.append((ch, url, line))
        if not jobs:
            why = "；".join(skipped) or "分镜里没找到有台词的角色（先拆分镜）"
            return ToolResult(ok=False, error=f"没有能定音的角色：{why}", meta={"charged": False})

        regen = self._worst_video_regen()
        each = self._price(MediaKind.VIDEO, self.video_model, {"duration": 5, "resolution": "720p"})
        n = len(jobs)
        bp, deny = await self._quote(
            "drama_voice_casting", f"定音 {n} 段", "video",
            f"定音：给 {n} 个主角各渲一段 5 秒的独白（{'、'.join(ch.name for ch, _, _ in jobs)}），"
            f"模型 {self.video_model} · 720p",
            (n * (1 + regen), 5.0 * n * (1 + regen),
             each * n * (1 + regen) if each is not None else None),
            each * n if each is not None else None,
            f"每段质检最多再生成 {regen} 次" if regen else "",
            {"assets_id": assets_id, "角色": [ch.name for ch, _, _ in jobs]},
        )
        if deny:
            return ToolResult(ok=False, error=deny, meta={"charged": False})

        ratio = self.aspect_ratio or DEFAULT_ASPECT
        retries = self._video_retries()
        sub_gate, sub_retries = self._subtitle_cfg()

        async def one(job: tuple[Any, str, str]) -> tuple[Any, str, Any, list[str]]:
            ch, url, line = job
            core = (
                f"{reference_block([('角色', ch.name)])}\n"
                f"【音色锁定】({ch.name})：{ch.voice or '按角色设定的声音'}\n"
                f"【定音独白】({ch.name}) 近景，面对镜头，神情自然，用自己的声音清楚地说出这一句"
                f"台词：“{line}”。只有 TA 一个人说话，没有背景音乐，环境安静，镜头固定。"
            )
            args: dict[str, Any] = {
                "prompt": person_video_prompt(
                    core, level=self._realism_level(), minor=is_minor(ch.body)
                ),
                "model": self.video_model,
                "aspect_ratio": ratio,
                "resolution": "720p",
                "duration": 5,
                "image": [url],
                "summary": f"定音·{ch.name}",
                "local_name": f"定音_{ch.name}",
                "tags": {"casting": ch.name, "library": assets_id},
            }
            r, notes = await self._gen_clip(
                args, retries, sub_gate, sub_retries, id_refs=[(ch.name, url)]
            )
            return ch, line, r, notes

        with batch_scope(bp):
            results = await _run_parallel(jobs, one, self._max_concurrency("video"))
        pending: dict[str, Anchor] = {}
        rows: list[str] = []
        for ch, line, r, notes in results:
            if not r.ok or not r.asset_ref:
                rows.append(f"  ✗ {ch.name}：没渲成（{(r.error or '')[:80]}）")
                continue
            bad = any(_blocking_note(x) for x in notes)
            local = self._local_video(r.asset_ref)
            where = str(local) if local is not None else r.asset_ref
            rows.append(
                f"  {'⛔' if bad else '♪'} {ch.name}：「{line}」→ {where}"
                + (f"（{'；'.join(notes)}）" if notes else "")
            )
            if not bad:
                pending[ch.name] = Anchor(
                    character=ch.name, asset=r.asset_ref, scene="定音", clip="独白",
                    lines=1, solo=True, pinned=True,
                )
        rows += [f"  · {s}" for s in skipped]
        if not pending:
            return ToolResult(
                ok=False, error="定音一段都没渲成（或都没过质检）：\n" + "\n".join(rows)
            )
        self._pending_casting = pending
        home = self._anchor_home("x")
        question = (
            "听一下这几段独白（本地文件在下面），声音对就采纳 —— 都会固定成各自的音色锚点，"
            "之后每一集都跟它对齐"
            + (f"，并在 {home.parent} 里永久存一份（<角色>.mp4）" if home is not None else "")
            + "；哪个角色的声音不对，就在采纳附言里写他的名字（那几个不固定），或者整体打回：\n"
            + "\n".join(rows)
        )
        return ToolResult(
            content=question + "\n\n已暂停等用户听。采纳后自动固定，不用再调 drama_voice_anchors。",
            suspend=True,
            suspend_payload={
                "question": question,
                "stage": CASTING_STAGE,
                "target": CASTING_STAGE,
                "assets": [a.asset for a in pending.values()],
                "major": True,
            },
        )

    def on_event(self, event: Any) -> None:
        """总线回调：人在「定音」上采纳 → 独白段固定成音色锚点（附言里点名的角色除外）。
        打回、/auto 自动采纳都不固定。"""
        if getattr(event, "type", None) is not EventType.CHECKPOINT_DECIDED:
            return
        data = getattr(event, "data", None) or {}
        if data.get("node") != CASTING_STAGE:
            return
        pending, self._pending_casting = self._pending_casting, {}
        if not pending or data.get("decision") != "adopt":
            return
        if str(data.get("decided_by") or "human") == "auto":
            return
        reason = str(data.get("reason") or "")
        keep = {n: a for n, a in pending.items() if n not in reason}
        if keep:
            self._merge_save_anchors(keep, [a.asset for a in keep.values()], override_pinned=True)
            # 人定的声音在本地永久存一份（2026-09-27）；存不成的当场说（存到哪在定音的问题里说过了）
            bad = [r for r in self._keep_anchor_files(keep) if r.startswith("⚠")]
            if bad and self.bus is not None:
                # 总线替任务留着引用：之前 create_task 不留引用，任务可能半路被回收（审查 1.9）
                self.bus.emit_soon(EventType.WARNING, message="定音：" + "；".join(bad))

    async def registry_invoke(self, name: str, args: dict[str, Any]) -> Any:
        if self.registry is None:
            return ToolResult(ok=False, error=f"没接注册表，调不了 {name}")
        return await self.registry.invoke(name, args)

    def permission_for(
        self, tool: str, args: dict[str, Any]
    ) -> tuple[PermissionLevel, str] | None:
        """注册表的提权钩子：定稿的资产库要整份重做（rebuild=true），当场问人。
        重做 = 重新设计全部角色 / 服装 / 场景，描述变了的参考图全要重生成、已渲的片段对不上
        （2026-09-26 用户定的：资产库定稿后冻结，只增量补新集）。"""
        if tool == "drama_assets" and args.get("rebuild"):
            return (
                PermissionLevel.EXTERNAL,
                "整份重做这部剧的资产库：角色 / 服装 / 场景会重新设计，描述变了的参考图都要"
                "重生成、已经渲好的片段可能对不上 —— 需要你确认",
            )
        return None

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            r = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
            r = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        r.duration_ms = int((time.perf_counter() - started) * 1000)
        return r

    def timeout_for(self, tool: str, args: dict[str, Any]) -> float | None:
        """这一次调用的超时按参数放宽（注册表的钩子）。只管渲参考图：张数事先数得出来。

        2026-09-29 审查 1.4：工具超时写死 1 小时，《不渡》174 张按实测速度要 50–90 分钟。
        按资产库里要生成的张数给（上限估计：不扣复用的），不少于工具声明的超时。"""
        if tool != "drama_render_assets":
            return None
        try:
            lib, err = parse_assets(self.store.content(str(args.get("assets_id") or "")))
        except KeyError:
            return None
        if err or lib is None:
            return None
        want = _RENDER_ONLY.get(str(args.get("only") or "all"), _RENDER_ONLY["all"])
        n = 0
        if "characters" in want:
            n += len(lib.characters)
        if "costumes" in want:
            n += sum(len(c.costumes) for c in lib.characters)
        if "scenes" in want:
            n += len(lib.scenes)
        if "props" in want:
            n += len(lib.props)
        base = self._specs["drama_render_assets"].timeout
        return max(base, _RENDER_ASSETS_PER_IMAGE_S * n)


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
                    "  写完把剧本给用户看，他确认了再进拆解流程（drama_storyboard + drama_assets）"
                    "—— 别拿没确认过的剧本往下拆。"
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
        # 集长只有一个来源：项目规格（/length，或人采纳的创作方案里写的每集时长）。之前模型能
        # 自己传 minutes，和项目规格、人审里问的时长三处各说各的（2026-09-26）—— 参数已从工具
        # 说明里去掉，老对话里还传的一律不用
        if not idea.strip():
            return ToolResult(ok=False, error="idea 是空的")
        resp = await self._prose(
            [{"role": "user", "content": write_prompt(idea, episodes, fmt=self.fmt)}]
        )
        stop = _finish_problem(resp, "剧本")
        if stop:
            return ToolResult(ok=False, error=stop)
        r = self._save_script(resp.text, f"剧本·{episodes}集", ["idea"], idea)
        if minutes and r.ok:
            r.content += (
                f"\n\n（没按 minutes={minutes:g} 写：集长按这个项目的规格 —— {self.fmt.brief()}；"
                "要改请用户 /length）"
            )
        return r

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
        parent_ids = [p for p in parents if p and p != "idea"]
        base_gp: dict[str, Any] = {"idea": idea} if idea else {}
        # 自查：写出来的东西得真像剧本，否则下一步拆分镜会出洋相
        warn = ""
        if not r.is_script:
            warn = (
                f"\n\n⚠ 自查：写出来的内容结构上不像剧本（{r.brief()}）。"
                "可能缺场景标头或对白格式，进拆解前先看一眼。"
            )
        # 按「第N集」拆开、一集一份（摘要「第N集·剧本」、记 episode）：之前整份存成「剧本·N集」，
        # 项目卡和按集流水认不出集号，项目卡整张不出现、⚠ 防线也跟着没了（2026-09-26）
        by_ep = split_script(body)
        if by_ep:
            tail = "剧本·扩写修改" if "扩写" in summary else "剧本"
            made = [
                (n, self.store.create(
                    text,
                    type_=AssetType.SCRIPT,
                    summary=f"第{n}集·{tail}",
                    parents=parent_ids,
                    creator="tool:drama_write",
                    gen_params={**base_gp, "episode": n},
                ))
                for n, text in sorted(by_ep.items())
            ]
            ids = "、".join(f"第{n}集 {a.id}" for n, a in made)
            return ToolResult(
                content=f"{body}\n\n---\n{summary} · {len(body)} 字 · 按集存好：{ids}{warn}",
                asset_ref=made[0][1].id,
            )
        asset = self.store.create(
            body,
            type_=AssetType.SCRIPT,
            summary=summary,
            parents=parent_ids,
            creator="tool:drama_write",
            gen_params=base_gp,
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
        resp, softened = await self._chat_soft(messages)
        stop = _finish_problem(resp, "分镜脚本")
        if stop:
            return ToolResult(ok=False, error=stop + (_SOFT_TRIED if softened else ""))
        resp, eps, err = await self._reparse(messages, resp, parse_episodes, "分镜脚本")
        if err:
            return ToolResult(ok=False, error=f"分镜解析失败（已自动重发一次）：{err}")

        # 规格检查（一集的镜头数与总时长、台词念不念得完、开场 15 秒高潮点）+ 比剧本少没少台词：
        # 不合格让模型改一次。之前只查总时长 —— 模型把 7–10 分钟的分镜压成 1.5 秒一镜「修」达标，
        # 改的那版问题变少了就被采用（2026-09-25：《不渡》5 集这样压快了台词、1 集删了 21 句）
        eps = self._fix_episode_index(eps, script, script_id)
        if softened:
            # 被内容过滤拦过、用克制措辞重写的：核对台词和场次，剧情变了就不用这一版
            # （2026-09-26：之前结果里只写一句「剧情和台词没改」，其实没核对）
            changed = _softened_changes(eps, script)
            if changed:
                return ToolResult(
                    ok=False,
                    error=(
                        "分镜被模型服务商的内容安全过滤拦了一次，自动用克制措辞重写的那版改了剧情"
                        f"（{changed}），没有采用。如实告诉用户哪段戏触发了过滤，让他决定怎么改"
                        "措辞或删改这段，不要自己删台词绕过去"
                    ),
                )
        problems = storyboard_problems(eps, script, self.fmt)
        if problems:
            text2 = await self._revise_once(_ROLE, messages, resp.text, problems)
            eps2, err2 = parse_episodes(text2)
            if not err2:
                eps2 = self._fix_episode_index(eps2, script, script_id)
                problems2 = storyboard_problems(eps2, script, self.fmt)
                # 问题少了还不够：比第一版多丢了台词 / 字幕卡的不要（2026-09-26：之前按问题条数挑，
                # 删了一两句台词的修订版照样胜出 —— 丢得少的不算问题，却是实打实丢了）
                if len(problems2) < len(problems) and _lost_count(eps2, script) <= _lost_count(
                    eps, script
                ):
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
                "spec": self.fmt.stamp,  # 规格戳：流水线据此认旧规格的产物
            },
        )
        body = "\n".join(
            f"{e.title}　{e.shot_count} 镜 / {len(e.scenes)} 场" for e in eps
        )
        hint = _length_hint(eps, self.fmt)
        # 只是比规格长（剧本本身长）时不说「达标」—— 总时长的事写在下面的 📏 里
        checked = "单镜时长、台词时长" + ("" if self.fmt.follow_script else "、开场高潮点")
        note = (
            f"\n\n✓ {checked}都达标；总时长见下"
            if hint and not problems
            else _format_note(problems, self.fmt)
        )
        # 跟剧本走的剧：开场没高潮点只提醒、加了闪前的说出来（2026-09-27 用户定的）
        opening = ""
        if self.fmt.follow_script:
            flat = [x for e in eps if (x := opening_note(e, self.fmt))]
            flash = [e.title for e in eps if has_flash(e.desc)]
            if flash:
                opening += (
                    f"\n\n📌 {'、'.join(flash)} 开场是铺垫，加了一段无台词画面闪前（不要就说一声）"
                )
            if flat:
                opening += "\n\n📌 " + "；".join(flat)
        return ToolResult(
            content=f"拆出 {len(eps)} 集：\n{body}{note}{opening}"
            f"{hint}{_SOFT_NOTE if softened else ''}"
            f"\n\n资产 {asset.id}（第③步要用）",
            asset_ref=asset.id,
        )

    def _fix_episode_index(self, eps: list[Any], script: str, script_id: str = "") -> list[Any]:
        """模型把集号标错（第 5 集的剧本标成 episodeIndex 1）：按剧本里的「第N集」标题、没有标题
        就按剧本资产的集号纠正。标错不纠正，分镜会串到别的集上，台词核对也对不上号、悄悄跳过
        （2026-09-26）。集数对不上（剧本 3 集、分镜 2 集）不猜，原样返回。"""
        want = sorted(split_script(script))
        if not want and script_id:
            try:
                n = self.store.episode_of(self.store.get(script_id))
            except KeyError:
                n = 0
            want = [n] if n > 0 else []
        if not want or len(want) != len(eps) or [e.index for e in eps] == want:
            return eps
        out: list[Any] = []
        for e, n in zip(eps, want, strict=True):
            if e.index != n:
                title = (
                    re.sub(r"第\s*[0-9零〇一二两三四五六七八九十百]+\s*集", f"第{n}集", e.title,
                           count=1)
                    if e.title else f"第{n}集"
                )
                e = replace(e, index=n, title=title)
            out.append(e)
        return out

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

    async def _reparse(
        self, messages: list[dict[str, Any]], resp: Any, parse: Any, what: str
    ) -> tuple[Any, Any, str]:
        """解析模型输出（parse 先在本地修漏逗号、尾逗号、没转义的引号）；还不行就原样重发一次，
        最后一条用户消息后面追加格式提醒。返回 (用上的回复, 解析结果, 错误)；
        两次都不行报第一次的错。
        """
        result, err = parse(resp.text)
        if not err:
            return resp, result, ""
        brief = err.split("（", 1)[0][:160]
        last = messages[-1]
        content = last["content"] + _JSON_AGAIN.format(err=brief)
        again = [*messages[:-1], {**last, "content": content}]
        resp2, _ = await self._chat_soft(again)
        if _finish_problem(resp2, what):
            return resp, None, err
        result2, err2 = parse(resp2.text)
        if err2:
            return resp, None, err
        return resp2, result2, ""

    def _script_episodes(self, ids: list[str], text: str) -> list[int]:
        """这次传进来的剧本覆盖哪几集：剧本资产的集号；没传 id 就按正文里的「第N集」标题。"""
        eps: set[int] = set()
        for aid in ids:
            try:
                n = self.store.episode_of(self.store.get(aid))
            except KeyError:
                continue
            if n > 0:
                eps.add(n)
        if not eps:
            eps = set(split_script(text))
        return sorted(eps)

    def _own_script_episodes(self) -> set[int]:
        """这个项目自己的剧本有哪几集（没迁移的旧剧没有项目键，不算）。"""
        scope = self.store.project
        return {
            n
            for a in self.store.find(type_=AssetType.SCRIPT)
            if (not scope or a.project == scope) and (n := self.store.episode_of(a)) > 0
        }

    def _library_covers(self, assets_id: str) -> set[int]:
        """资产库生成时用了哪几集的剧本（2026-09-26 起记在 gen_params.covers）。

        老的库没记，返回空。"""
        try:
            raw = self.store.get(assets_id).gen_params.get("covers") or []
        except KeyError:
            return set()
        return {int(x) for x in raw if isinstance(x, int) or str(x).isdigit()}

    # ---------- ② 资产库 ----------

    async def _fn_drama_assets(
        self,
        script: str = "",
        ethnicity: str = "",
        language: str = "",
        note: str = "",
        script_id: str = "",
        script_ids: list[str] | None = None,
        rebuild: bool = False,
    ) -> ToolResult:
        ids = list(script_ids or []) + ([script_id] if script_id else [])
        script, err = self._script_input(script, ids)
        if err:
            return ToolResult(ok=False, error=err, meta={"charged": False})
        opts = normalize(ethnicity, language)
        if not opts.ready:
            return ToolResult(ok=False, error=ask_text())
        # 渲过参考图的资产库已经定稿：只增量补新集（2026-09-26 用户定的）。rebuild 才整份重做
        frozen = None if rebuild else self._frozen_library()
        if frozen is not None:
            return await self._extend_library(frozen, ids, script, opts, note)
        # 整份生成也记上一版（base）：渲参考图时沿 base 往上找渲过的包，名字和描述都没变的图照用。
        # 之前只有 rebuild 记 —— 重做完还没渲图又整份生成一次，base 断了，旧图一张都沿用不上
        # （mode 不是 incremental，不算同一套库，library_chain 不跟它）
        previous = self._latest_library_id()
        user = script if not note else f"{script}\n\n【额外要求】{note}"
        messages = [
            {"role": "system", "content": assets_system(opts, level=self._realism_level())},
            {"role": "user", "content": user},
        ]
        resp, softened = await self._chat_soft(messages)
        stop = _finish_problem(resp, "资产库")
        if stop:
            return ToolResult(ok=False, error=stop + (_SOFT_TRIED if softened else ""))
        # 输出不是合法 JSON：本地修（漏逗号、尾逗号、没转义的引号），修不好原样重发一次。
        # 2026-09-26《不渡》：20 集一次生成，模型在第 790 行漏了个逗号，2 分钟的调用整份作废；
        # 主模型于是拆成 1–10、11–20 两段各生成一份 —— 7 个主角各被描述成了两张脸
        resp, lib, err = await self._reparse(messages, resp, parse_assets, "资产库")
        if err:
            return ToolResult(
                ok=False, error=f"资产库解析失败（已自动重发一次）：{err}。{_ONE_LIBRARY}"
            )
        covers = self._script_episodes(ids, script)

        payload = {
            "characters": [as_dict(c) for c in lib.characters],
            "scenes": [as_dict(s) for s in lib.scenes],
            "props": [as_dict(p) for p in lib.props],
        }
        asset = self.store.create(
            json.dumps(payload, ensure_ascii=False, indent=2),
            type_=AssetType.STORYBOARD,
            summary=f"资产库·{lib.counts}",
            parents=[i for i in ids if i],
            creator="tool:drama_assets",
            gen_params={
                "characters": len(lib.characters),
                "ethnicity": opts.ethnicity,
                "language": opts.language,
                # 用了哪几集的剧本：项目卡和第③步据此认出「只覆盖半部剧」的库
                "covers": covers,
                # 整份重做：记下上一版，渲参考图时名字和描述都没变的图接着用（不算同一套库）
                "mode": "rebuild" if rebuild else "full",
                **({"base": previous} if previous else {}),
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
        # 只拿了部分剧本生成：同一部剧再生成另一段，会把同一个角色描述成另一张脸
        rest = sorted(self._own_script_episodes() - set(covers)) if covers else []
        if rest:
            lines.append(
                f"⚠ 这份资产库只用了第 {_fmt_eps(covers)} 集的剧本，"
                f"项目里还有第 {_fmt_eps(rest)} 集。{_ONE_LIBRARY}"
            )
        return ToolResult(
            content="\n".join(lines) + f"\n\n资产 {asset.id}（第③步要用）",
            asset_ref=asset.id,
        )

    # ---------- 资产库定稿后只增量补新集（2026-09-26 用户定的） ----------

    def _own_libraries(self) -> list[Any]:
        """这个项目自己的资产库，新的在前（没迁移的旧剧没有项目键，不算）。"""
        scope = self.store.project
        return [
            a for a in self.store.find(creator="tool:drama_assets")
            if not scope or a.project == scope
        ]

    def _latest_library_id(self) -> str:
        libs = self._own_libraries()
        return libs[0].id if libs else ""

    def _frozen_library(self) -> tuple[str, Any, set[int]] | None:
        """已经定稿的资产库：这个项目最新的那一版，它（或它增量链上的某一版）渲过参考图。
        返回 (id, 资产库, 覆盖哪几集)；没有定稿的返回 None。"""
        libs = self._own_libraries()
        if not libs:
            return None
        latest = libs[0]
        if not any(self._previous_pack(x)[1] for x in library_chain(self.store, latest.id)):
            return None
        lib, err = parse_assets(self.store.content(latest.id))
        if err:
            return None
        return latest.id, lib, self._library_covers(latest.id)

    async def _extend_library(
        self, frozen: tuple[str, Any, set[int]], ids: list[str], script: str, opts: Any,
        note: str,
    ) -> ToolResult:
        """定稿的资产库只补新集里新出现的角色 / 服装 / 场景 / 道具，已有条目原样冻结 ——
        之前改一套戏服、补一个角色、续写一集，都得整份重做：174 张参考图全部重生成、全员换脸、
        已渲的片段对不上。新一版记 mode=incremental、base=上一版，渲参考图时已有的图直接沿用。"""
        base_id, base_lib, base_covers = frozen
        covers = self._script_episodes(ids, script)
        new_eps = [n for n in covers if n not in base_covers] if base_covers else list(covers)
        if covers and base_covers and not new_eps:
            return ToolResult(
                content=(
                    f"这部剧的资产库已经定稿（{base_id}，覆盖第 {_fmt_eps(base_covers)} 集，参考图"
                    "渲过了），这几集都在里面，不用重做 —— 第③步直接用它。用户明确要整份重做才调 "
                    "drama_assets(…, rebuild=true)（会先问用户）"
                ),
                asset_ref=base_id,
                meta={"charged": False},
            )
        by_ep = split_script(script)
        chunks = [by_ep[n] for n in new_eps if n in by_ep] if by_ep else []
        fresh = "\n\n".join(chunks) or script
        user = (
            f"【已定稿的资产库（原样沿用，不要改写、不要重复输出）】\n{_frozen_digest(base_lib)}"
            f"\n\n【新增的剧本】\n{fresh}" + (f"\n\n【额外要求】{note}" if note else "")
        )
        messages = [
            {
                "role": "system",
                "content": assets_system(opts, level=self._realism_level()) + _EXTEND_RULES,
            },
            {"role": "user", "content": user},
        ]
        resp, softened = await self._chat_soft(messages)
        stop = _finish_problem(resp, "资产库增量")
        if stop:
            return ToolResult(ok=False, error=stop + (_SOFT_TRIED if softened else ""))
        resp, add, err = await self._reparse(messages, resp, _parse_additions, "资产库增量")
        if err:
            return ToolResult(ok=False, error=f"资产库增量解析失败（已自动重发一次）：{err}")
        merged, added = _merge_library(base_lib, add)
        payload = {
            "characters": [as_dict(c) for c in merged.characters],
            "scenes": [as_dict(s) for s in merged.scenes],
            "props": [as_dict(p) for p in merged.props],
        }
        asset = self.store.create(
            json.dumps(payload, ensure_ascii=False, indent=2),
            type_=AssetType.STORYBOARD,
            summary=f"资产库·{merged.counts}",
            parents=[base_id] + [i for i in ids if i],
            creator="tool:drama_assets",
            gen_params={
                "characters": len(merged.characters),
                "ethnicity": opts.ethnicity,
                "language": opts.language,
                "covers": sorted(set(base_covers) | set(covers)),
                "mode": "incremental",
                "base": base_id,
                "added": {k: len(v) for k, v in added.items()},
            },
        )
        n_new = sum(len(v) for v in added.values())
        lines = [
            f"这部剧的资产库已经定稿（{base_id}），这次只补新集"
            + (f"（第 {_fmt_eps(new_eps)} 集）" if new_eps else "")
            + "里新出现的：" + (
                "；".join(f"{k} {len(v)}：{'、'.join(v[:6])}" for k, v in added.items() if v)
                or "没有新东西"
            ),
            f"已有的 {base_lib.counts} 原样沿用（描述没动，参考图 drama_render_assets 会直接沿用，"
            f"只生成新增的 {n_new} 张）。",
        ]
        blank = [c.name for c in merged.characters if c.name in added["角色"] and not c.body]
        if blank:
            lines.append(f"⚠ 新角色没写形象描述：{'、'.join(blank)} —— 生图前补上")
        if softened:
            lines.append(_SOFT_NOTE.strip())
        return ToolResult(
            content="\n".join(lines) + f"\n\n资产 {asset.id}（第③步要用；是 {base_id} 的新一版）",
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

        # 资产库只用了其中几集的剧本（分段生成的）：别的集的角色、服装、场景在里面对不上
        covers = self._library_covers(assets_id)
        outside = [e.index for e in eps if covers and e.index > 0 and e.index not in covers]
        if outside:
            return ToolResult(
                ok=False,
                error=(
                    f"资产库 {assets_id} 只用第 {_fmt_eps(sorted(covers))} 集的剧本生成，"
                    f"第 {_fmt_eps(outside)} 集不在里面（角色、服装、场景对不上）。{_ONE_LIBRARY}"
                ),
                meta={"charged": False},
            )

        # 语言选择从第①步继承。重新问一遍是多余的，而且用户答得不一样时
        # 会把已经翻译好的台词又译回去 —— 台词必须在整条链上保持一致。
        opts = self._inherit_options(storyboard_id, assets_id)

        # 一集一集地生成；一集的分镜太长就按场分批（每批一次模型调用，并行）。视频提示词要覆盖
        # 分镜的每个镜头、时长跟着分镜走（2026-09-25：之前整季一次生成、按「一集 4 分钟」压总
        # 时长 —— 第 3 集 217 镜的分镜只写到第 128 镜，后半集整个没了，工具还报成功；第 1 集那份
        # 「达标」的也漏了最后 7 镜。输出太长还会被截断，流式传输也更容易被中途断开）
        shots: list[Any] = []
        problems: list[str] = []
        infos: list[str] = []
        chunks = 0
        note_fmt = self.fmt
        # 分镜从哪份剧本拆的：分镜里丢了的上屏字（字幕卡）按剧本补回来（2026-09-26）
        script_of = self._storyboard_script(storyboard_id)
        # 各批的输入指纹：出好的批先存着，这次没成、再跑只补没出好的（2026-09-29 审查 2.4）
        batch_keys: list[str] = []
        for e in eps:
            script = script_of.get(e.index) or (script_of.get(0, "") if len(eps) == 1 else "")
            got, probs, err, info, n_chunks, used = await self._episode_shots(
                e, lib, opts, note, script, batch_keys=batch_keys
            )
            if err:
                kept = sum(1 for k in set(batch_keys) if self._scratch_load("shots", k))
                if kept:
                    err += (
                        f"\n\n已经出好的 {kept} 批提示词先存着：参数不变再跑一次 drama_shots，"
                        "只补没出好的，出好的不重付"
                    )
                return ToolResult(ok=False, error=err)
            shots += got
            problems += probs
            chunks += n_chunks
            if info:
                infos.append(info)
            if len(eps) == 1:
                note_fmt = used

        # 全角括号里的资产名换成半角：提示词示例曾写成全角，模型照抄，场景引用和按场景换装
        # 就全部失效（2026-09-23 审查）
        known = lib.all_names()
        for s in shots:
            s.description = normalize_ref_parens(s.description, known)
        # 服装 ID 的集数范围被模型写短了（「孙悟空-黄直裰虎皮围裙-[1]」，库里是「…-[1-18]」）：
        # 同一角色同一套服装认得准就换成库里的全名，不用再带一长串全名当 note 重跑（2026-09-25）
        costume_names = {cos.name for c in lib.characters for cos in c.costumes}
        costume_fixes: list[str] = []
        for s in shots:
            ep_no, _ = parse_scene(s.scene_index)
            s.description, got_fixed = fix_costume_refs(s.description, costume_names, ep_no)
            costume_fixes += got_fixed
        # 服装按场景绑定（确定性，不靠模型自觉）：镜头所在场景有绑定服装就换成它的 ID，
        # 同一场景内穿搭一致、换场景才换装；裸角色名也换成服装 ID 让参考图对得上
        bindings, wardrobe_warns, costume_kept = bind_costumes(shots, lib)
        bad = audit_refs(shots, lib)
        gp: dict[str, Any] = {
            "shots": len(shots),
            "bad_refs": len(bad),
            "costume_bindings": len(bindings),
            "wardrobe_gaps": len(wardrobe_warns),
            "format_problems": len(problems),
            "spec": self.fmt.stamp,  # 规格戳：流水线见到旧戳的不直接拿去渲
            "chunks": chunks,
            "costume_fixes": len(costume_fixes),
            "costume_kept": len(costume_kept),
            # 上屏字（字幕卡）排进了几段的时间线：成片后叠。有这个键 = 按新规则出的提示词
            "screen_text": sum(len(s.screen_text) for s in shots),
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
        # 整份落库了，各批的半成品不用再留：之后再跑是要重出，不该拿到原样的旧结果
        for k in set(batch_keys):
            self._scratch_drop("shots", k)
        lines = []
        for s in shots:
            carry = f"　引入 {'/'.join(s.carries())}" if s.carries() else ""
            cuts = f"/{len(s.cuts)}镜" if s.cuts else ""
            lines.append(
                f"{s.scene_index} {s.video_name} {s.seconds}s{cuts}　"
                f"引用 {len(s.refs())} 个资产{carry}"
            )
        warn = ""
        if costume_fixes:
            warn += (
                f"\n\n服装全名纠正了 {len(costume_fixes)} 处（模型把集数范围写短了），"
                f"如 {costume_fixes[0]}"
            )
        if bindings:
            shown = bindings[:10]
            more = f"\n  …还有 {len(bindings) - 10} 处" if len(bindings) > 10 else ""
            warn += (
                f"\n\n服装已按场景绑定，改了 {len(bindings)} 处引用：\n  "
                + "\n  ".join(shown) + more
            )
        if costume_kept:
            shown = costume_kept[:8]
            more = f"\n  …还有 {len(costume_kept) - 8} 处" if len(costume_kept) > 8 else ""
            warn += (
                f"\n\n服装按剧情保留了 {len(costume_kept)} 处（模型按剧情点名、这一集能穿，"
                "但和资产库的场景分配表不一致 —— 回忆戏、婚礼这类临时换装多半是对的，"
                "请用户看一眼）：\n  " + "\n  ".join(shown) + more
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
        rushed = [e.title for e in eps if dialogue_rushed(e)]
        if rushed:
            warn += (
                f"\n\n⚠ {'、'.join(rushed)} 的分镜把台词压快了（光念台词就占满了镜头总时长）："
                "提示词跟着分镜走，渲出来台词会念不完，算不上完成。先请用户用 /length 放宽集长"
                "（或 /length auto 跟剧本走）、重拆这几集的分镜，再重出提示词"
            )
        total = sum(s.seconds for s in shots)
        warn += _format_note(problems, note_fmt)
        head = f"{len(shots)} 段 · 共 {total}s\n" + ("".join(i + "\n" for i in infos))
        return ToolResult(
            content=head + "\n".join(lines) + warn + f"\n\n资产 {asset.id}",
            asset_ref=asset.id,
        )

    def _storyboard_script(self, storyboard_id: str) -> dict[int, str]:
        """分镜是从哪份剧本拆的（drama_storyboard 落库时记的第一个父资产）：
        {集号: 这一集的剧本}；剧本里没有「第N集」标题时 {0: 全文}。找不到返回空。"""
        try:
            parents = self.store.get(storyboard_id).parent_ids
            src = self.store.get(parents[0]) if parents else None
        except (KeyError, IndexError):
            return {}
        if src is None or src.type is not AssetType.SCRIPT:
            return {}
        text = self.store.content(src.id)
        by_ep = split_script(text)
        return by_ep or ({0: text} if text else {})

    async def _place_missing_cards(
        self, e: Any, blocks: list[Any], missing: list[Card], script: str
    ) -> tuple[dict[int, list[Card]], str]:
        """剧本里有、分镜里丢了的上屏字：让模型对着带镜号的分镜挑它该在哪个镜头出现（一次小调用）。
        不用重拆分镜 —— 《不渡》20 集的分镜 9-26 才重拆过，第 1 集 6 张卡丢了 5 张。
        返回 ({镜号: [卡]}, 错误)。"""
        if not missing or self.gateway is None:
            return {}, ""
        n_lines = sum(b.count for b in blocks)
        board = strip_cards("\n".join(b.text for b in blocks))
        ctx = "\n".join(f"- {c.marker}（剧本里前后：{_card_context(script, c)}）" for c in missing)
        prompt = (
            "下面是一集短剧的分镜（每个镜头行开头的〔N〕是镜号）和剧本里几张分镜没写进去的上屏字"
            "（字幕卡）。按剧本上下文判断每张卡该在哪个镜头出现：地点卡 / 时间卡放在那场戏的第一个"
            "镜头，片尾卡放在最后一个镜头。只输出 JSON："
            '{"cards": [{"text": "卡片文字，原样照抄", "shot": 镜号}]}，不要解释。\n\n'
            f"【上屏字】\n{ctx}\n\n【分镜】\n{board}"
        )
        try:
            resp, _ = await self._chat_soft([{"role": "user", "content": prompt}])
        except Exception as ex:  # noqa: BLE001 — 放不上就如实报，不挡提示词
            return {}, f"{type(ex).__name__}: {ex}"
        rows = _json_rows(getattr(resp, "text", ""), "cards")
        by_key = {card_key(c.text): c for c in missing}
        placed: dict[int, list[Card]] = {}
        for row in rows:
            c = by_key.pop(card_key(str(row.get("text") or "")), None)
            try:
                no = int(float(row.get("shot")))
            except (TypeError, ValueError):
                no = 0
            if c is not None and 1 <= no <= n_lines:
                placed.setdefault(no, []).append(c)
        # 片尾卡模型没放上：按规矩放最后一镜
        for k, c in list(by_key.items()):
            if c.ending and n_lines:
                placed.setdefault(n_lines, []).append(c)
                by_key.pop(k)
        return placed, ("" if not by_key else f"{len(by_key)} 张没对上镜头")

    def _coverage_gate(self, shots_id: str, shots: list[Any]) -> str:
        """这份视频提示词有没有覆盖分镜的全部镜头；漏了返回拦截说明，空串 = 放行。

        提示词的第一个父资产是它的分镜（drama_shots 落库时记的）。读不到分镜的（手写的提示词、
        测试数据）不查。
        """
        try:
            parents = self.store.get(shots_id).parent_ids
            eps, err = parse_episodes(self.store.content(parents[0]))
        except (KeyError, IndexError):
            return ""
        if err or not eps:
            return ""
        missing = prompt_problems(shots, eps)
        if missing:
            return (
                "这份视频提示词和分镜对不上（" + "；".join(missing) + "），照着渲会缺镜头、丢台词，"
                "没有发起生成。先重跑 drama_shots 生成完整的提示词再渲（现在的 drama_shots 会"
                "逐镜核对、漏了自动补写、时长跟着分镜走）"
            )
        # 分镜重拆过：这份提示词和它自己的旧分镜对得上，但照着渲就是旧版的内容（2026-09-26：
        # 20 集全部重拆后，第 11、15 集的旧提示词「完整」，上面的检查拦不住）
        newest = latest_storyboards(self.store)
        here = sorted({parse_scene(s.scene_index)[0] for s in shots} - {0})
        moved = [n for n in here if newest.get(n) and newest[n] != parents[0]]
        if moved:
            return (
                f"这份视频提示词是按旧分镜 {parents[0]} 出的，"
                f"第 {_fmt_eps(moved)} 集的分镜已经重拆过"
                f"（最新的是 {newest[moved[0]]}），照着渲出来是旧版的内容，没有发起生成。"
                "先用新分镜重跑 drama_shots 再渲"
            )
        # 2026-09-26 之前出的提示词：剧本里的上屏字（字幕卡）被写进了画面描述让模型画 ——
        # 和「画面不许有字」冲突（画了字幕门判 ⛔、缺段不成片），成片上也不会叠字
        try:
            old = "screen_text" not in (self.store.get(shots_id).gen_params or {})
        except KeyError:
            old = False
        if old:
            script = self._storyboard_script(parents[0])
            eps_here = set(here) or {0}
            has_cards = any(find_cards(e.desc) for e in eps if e.index in eps_here or not here)
            has_cards = has_cards or any(
                find_cards(script.get(n, "")) for n in (eps_here | {0})
            )
            if has_cards:
                return (
                    "这份视频提示词是按旧规则出的：剧本里的上屏字（【字幕：…】这类字卡）被写进了"
                    "画面描述让视频模型画 —— 和「画面不许有字」冲突，成片上也不会叠字，"
                    "没有发起生成。"
                    "先重跑 drama_shots（会把字幕卡抽出来，拼完成片后叠上）再渲"
                )
        return ""

    async def _episode_shots(
        self,
        e: Any,
        lib: Any,
        opts: Any,
        note: str,
        script: str = "",
        batch_keys: list[str] | None = None,
    ) -> tuple[list[Any], list[str], str, str, int, EpisodeFormat]:
        """一集的视频提示词：分镜太长就按场分批并行生成，漏掉的镜头补写一次，最后按集检查。
        分镜里的上屏字（字幕卡）不给模型看、排进 screen_text 成片后叠；script 给了就把分镜
        丢掉的卡按剧本补回来。batch_keys 给了就把各批的输入指纹记进去（调用方落库后清缓存）。

        返回 (提示词, 规格问题, 错误, 给人看的说明, 分了几批, 检查用的规格)。
        """
        blocks, secs_by_no = scene_blocks(e.desc, e.index)
        n_lines = sum(b.count for b in blocks)
        if not n_lines:
            return [], [], f"{e.title} 的分镜里一个镜头行都没有", "", 0, self.fmt
        cards = card_lines(e.desc)
        missing = missing_cards(script, e.desc) if script else []
        card_texts = [c.text for cs in cards.values() for c in cs] + [c.text for c in missing]
        lo, hi = self.fmt.duration_range
        sb_total = sum(secs_by_no.values())
        timed = len(secs_by_no) >= 0.8 * n_lines
        # 分镜标了时长、又和一集的规格对不上：总时长跟着分镜走。分镜是定好的内容和节奏 ——
        # 比规格长时按规格压只能合并、删镜头（丢台词）；比规格短时按规格拉只能凭空加戏、多花钱
        # （按 4 分钟拆的第 1 集，项目改成 8 分钟后重出提示词会被拉成两倍）。集长由分镜那一步
        # 定：长了提醒用户 /length 放宽，短了重拆分镜。集长跟剧本走（/length auto）时一律按分镜
        target = (
            sb_total if timed and (self.fmt.follow_script or not lo <= sb_total <= hi) else 0.0
        )
        faithful = timed  # 分镜标了时长：每个镜头保留分镜标的时长
        fmt = (
            replace(self.fmt, minutes=round(target / 60, 2), follow_script=False)
            if target else self.fmt
        )
        groups = plan_chunks(blocks)
        results = await asyncio.gather(
            *(
                self._chunk_shots(
                    e, g, k, len(groups), lib, opts, note, secs_by_no, target, faithful,
                    strip=card_texts, used=batch_keys,
                )
                for k, g in enumerate(groups)
            ),
            return_exceptions=True,
        )
        shots: list[Any] = []
        softened = 0
        for r in results:
            if isinstance(r, BaseException):
                if not isinstance(r, Exception):
                    raise r  # /stop、超时的取消照样往上抛
                # 出好的批已经各自存下了（_chunk_shots），这里如实报是哪一类错
                return [], [], (
                    f"{e.title} 有一批视频提示词调用失败（{type(r).__name__}: {str(r)[:200]}）"
                ), "", len(groups), fmt
            got, err, soft = r
            softened += soft
            if err:
                return [], [], err, "", len(groups), fmt
            shots += got
        expected = set(range(1, n_lines + 1))
        gaps = coverage_gaps(shots, expected)
        fill_err = ""
        if gaps:
            # 模型偶尔跳过一截：只把漏掉的镜头补写一次
            keep = {n for a, b in gaps for n in range(a, b + 1)}
            touched = [b for b in blocks if any(b.first <= n <= b.last for n in keep)]
            got, fill_err, soft = await self._chunk_shots(
                e, touched, -1, len(groups), lib, opts, note, secs_by_no, target, faithful,
                keep=keep, strip=card_texts, used=batch_keys,
            )
            softened += soft
            if not fill_err:
                filled = [s for s in got if covered_numbers(s.video_name) & keep]
                shots = _merge_by_number(shots, filled)
            gaps = coverage_gaps(shots, expected)
        if gaps:
            why = f"：{fill_err}" if fill_err else ""
            return [], [], (
                f"{e.title} 的视频提示词漏了分镜第 {format_ranges(gaps)} 镜"
                f"（补写一次后仍缺{why}），没有保存 —— 存下来会渲出缺一截的成片。"
                "再跑一次 drama_shots"
            ), "", len(groups), fmt
        problems = check_shots(shots, fmt) + compression_problems(
            shots, secs_by_no, f"{e.title}："
        )
        # ---- 上屏字（字幕卡）：描述里不留字，排进各段的 screen_text，成片后叠 ----
        for s in shots:
            s.description = strip_cards(s.description, card_texts)
            s.screen_text = []
        lost_cards = place_cards(shots, cards, secs_by_no)
        placed_back, place_err = await self._place_missing_cards(e, blocks, missing, script)
        lost_cards += place_cards(shots, placed_back, secs_by_no)
        back = sum(len(v) for v in placed_back.values())
        n_cards = sum(len(s.screen_text) for s in shots)
        parts: list[str] = []
        if len(groups) > 1:
            parts.append(f"分镜 {n_lines} 镜，按场分 {len(groups)} 批生成")
        if softened:
            parts.append(f"内容安全过滤拦了 {softened} 次，已自动用克制的措辞重写通过")
        if target and not self.fmt.follow_script:
            why = (
                "剧本本身更长；要按更长的集长走请用户 /length 放宽，或 /length auto 跟剧本走"
                if sb_total > hi
                else "分镜比规格短，多半是按更短的集长拆的；要补足时长就重拆这一集的分镜"
            )
            parts.append(
                f"时长按分镜 {sb_total:g}s（一集 {self.fmt.minutes:g} 分钟的规格是 {lo}–{hi}s，"
                f"{why}）"
            )
        if n_cards:
            parts.append(
                f"上屏字 {n_cards} 张不进画面、成片后叠"
                + (f"（其中 {back} 张分镜里丢了，按剧本补回）" if back else "")
            )
        not_placed = [c.marker for c in lost_cards] + (
            [c.marker for c in missing if not any(c in v for v in placed_back.values())]
            if place_err else []
        )
        if not_placed:
            parts.append(
                f"⚠ 上屏字 {len(not_placed)} 张没排进时间线（{'、'.join(not_placed[:3])}），"
                "成片上不会有 —— 重拆这一集的分镜可以补上"
            )
        info = f"{e.title}：" + "；".join(parts) if parts else ""
        return shots, problems, "", info, len(groups), fmt

    async def _chunk_shots(
        self,
        e: Any,
        group: list[Any],
        k: int,
        n_groups: int,
        lib: Any,
        opts: Any,
        note: str,
        secs_by_no: dict[int, float],
        target: float,
        faithful: bool,
        keep: set[int] | None = None,
        strip: list[str] | None = None,
        used: list[str] | None = None,
    ) -> tuple[list[Any], str, bool]:
        """一批（或补写）的视频提示词：一次模型调用，这一批自己不合格就改一次。
        分镜里的上屏字（字幕卡）先剥掉再给模型看 —— 画面里不许有字，字卡成片后叠。
        出好的批按输入指纹存一份，同样的输入再跑直接沿用（used 给了就把指纹记进去）。
        返回 (提示词, 错误, 被内容过滤拦过又自动重试通过了没有)。"""
        text = strip_cards("\n".join(b.text for b in group), strip)
        if keep is not None:
            text = only_lines(text, keep)
            nums = set(keep)
        else:
            nums = set(range(group[0].first, group[-1].last + 1))
        secs = sum(secs_by_no.get(n, 0.0) for n in nums)
        partial = keep is not None or n_groups > 1
        user = shots_prompt(lib.digest(), f"{e.title}\n{text}") + _SHOTS_NUMBERING
        if faithful:
            user += _SHOTS_FAITHFUL
        if keep is not None:
            user += (
                f"\n\n【补写】上一轮漏了第 {format_ranges(number_ranges(sorted(keep)))} 镜："
                "只为这些镜头写视频提示词，别的镜头已经写好了、不要重复；"
                "这些不是开场，不要加 hook 字段。"
            )
        elif n_groups > 1:
            user += (
                f"\n\n【分批】这一集分镜较长，分 {n_groups} 批生成，这是第 {k + 1} 批：只写第 "
                f"{min(nums)}–{max(nums)} 镜（合计约 {secs:g} 秒），其余镜头别的批会写，不要补写。"
                + ("" if k == 0 else "这一批不是开场，不要加 hook 字段。")
            )
        elif target:
            user += (
                f"\n\n【时长以分镜为准】这一集分镜的镜头时长合计 {target:g} 秒：各段加起来也要"
                f" ≈ {target:g} 秒。不要为了凑标准时长合并、删镜头或添加分镜里没有的镜头。"
            )
        if note:
            user += f"\n\n【额外要求】{note}"
        fmt = (
            replace(self.fmt, minutes=max(secs, 1.0) / 60, follow_script=False)
            if (partial or target) else self.fmt
        )
        messages = [
            {
                "role": "system",
                "content": shots_system(opts, level=self._realism_level(), fmt=fmt),
            },
            {"role": "user", "content": user},
        ]
        # 同一批上次出好了、只是别的批失败了：直接沿用，不再付一次（2026-09-29 审查 2.4：
        # 之前任一批失败，已经成功的批全部丢掉，重跑整集重付，日志里白付了 21 次调用）
        key = _batch_key(messages)
        if used is not None:
            used.append(key)
        hit = self._scratch_load("shots", key)
        if isinstance(hit, dict) and hit.get("shots"):
            cached, err = parse_shots(json.dumps(hit["shots"], ensure_ascii=False))
            if not err:
                return cached, "", bool(hit.get("softened"))
        resp, softened = await self._chat_soft(messages)
        stop = _finish_problem(resp, "视频提示词")
        if stop:
            return [], stop + (_SOFT_TRIED if softened else ""), softened
        resp, shots, err = await self._reparse(messages, resp, parse_shots, "视频提示词")
        if err:
            return [], f"镜头提示词解析失败（已自动重发一次）：{err}", softened
        if softened:
            # 被内容过滤拦过、用克制措辞重写的：台词一句都不能少，少了就不用这一版（2026-09-26）
            lost = quoted_lost(text, "\n".join(s.description for s in shots))
            if lost:
                shown = "、".join(f"「{q[:12]}」" for q in lost[:3])
                return [], (
                    f"{e.title} 的视频提示词被内容安全过滤拦了一次，用克制措辞重写的那版丢了 "
                    f"{len(lost)} 句台词（如 {shown}），没有采用。如实告诉用户哪段戏触发了过滤，"
                    "让他决定怎么改"
                ), softened
        hook = k == 0 and keep is None
        problems = _chunk_problems(shots, fmt, nums, secs_by_no, hook)
        if problems:
            text2 = await self._revise_once(_ROLE, messages, resp.text, problems)
            shots2, err2 = parse_shots(text2)
            if not err2:
                problems2 = _chunk_problems(shots2, fmt, nums, secs_by_no, hook)
                before = (len(coverage_gaps(shots, nums)), len(problems))
                after = (len(coverage_gaps(shots2, nums)), len(problems2))
                if after < before:
                    shots = shots2
        self._scratch_save(
            "shots", key, {"shots": [as_dict(s) for s in shots], "softened": softened}
        )
        return shots, "", softened

    async def _chat_soft(self, messages: list[dict[str, Any]]) -> tuple[Any, bool]:
        """调一次 drama 角色；被内容安全过滤拦下就带着「措辞克制」的提醒自动再试一次。

        2026-09-25：Gemini 的过滤带随机性，一晚上拦了 3 次（第 7、20 集的视频提示词、第 18 集
        的分镜），主模型每次原样重跑一遍就过了 —— 工具自己重试一次，省掉一轮来回。
        返回 (响应, 有没有重试过)。
        """
        resp = await self.gateway.chat(_ROLE, messages)
        if not _is_filtered(resp):
            return resp, False
        soft = [*messages[:-1], {**messages[-1], "content": messages[-1]["content"] + _SOFTEN}]
        return await self.gateway.chat(_ROLE, soft), True

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
            passed, score, issues, note, direction, unchecked = await self._check_realism(
                aid, url, level, minor=minor
            )
            entry.update({"checked": True, "pass": passed, "score": score, "issues": issues})
            if note:
                entry["note"] = note
            # 没查成照旧按通过放行（拦不拦待用户拍板），但报告里单列，不算「一次通过」
            if unchecked:
                entry["unchecked"] = True
            else:
                entry.pop("unchecked", None)
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
            if getattr(v, "unchecked", False):
                # 图片没查成照旧按通过（拦不拦待用户拍板），报告里单列「没查成」，不写成「一致」
                entry["identity"]["unchecked"] = True
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
        """给视觉模型看的图：本地副本转 data URL（不依赖对方能不能拉外链），没有就给 url
        （没传 url 就用资产自己的链接）。链接由 _vision_image 下载转换，不直接发。"""
        try:
            asset = self.store.get(asset_id)
        except KeyError:
            asset = None
        local = local_copy(asset) if asset is not None else None
        if local:
            try:
                data = Path(str(local)).read_bytes()
                mime = "image/png" if str(local).lower().endswith(".png") else "image/jpeg"
                return f"data:{mime};base64," + base64.b64encode(data).decode()
            except OSError:
                pass
        if not url and asset is not None and (asset.uri or "").startswith(("http://", "https://")):
            return asset.uri
        return url

    def _realism_level(self) -> str:
        """真实感档位：subtle（默认，真实但干净）/ natural / strong。

        见 media_models.yaml 的 drama.realism_level。
        """
        cfg = getattr(self.catalog, "drama", {}) or {}
        return norm_level(cfg.get("realism_level"))

    async def _check_realism(
        self, asset_id: str, url: str, level: str = "", minor: bool = False
    ) -> tuple[bool, int, list[str], str, str, bool]:
        """视觉模型判一张人物图的皮肤质感是否达标。
        返回 (通过, 分数, 问题, 备注, 不合格方向, 没查成)。

        不合格方向：smooth 磨皮了 / heavy 做旧过头 / light 光太柔 —— 决定重生成往哪边改。
        校验本身失败（角色没配、调用异常、输出不是 JSON）按通过处理但写备注 ——
        校验是保险，不能把生图链路一起挂掉。图拿不到、调用失败、输出读不出来记「没查成」，
        报告里单独列出来，不和「查过、通过」混在一起（2026-09-29 审查 1.2；拦不拦待用户拍板）。
        """
        image, why = await self._vision_image(asset_id, url)
        if not image:
            return True, -1, [], f"没查成：{why}", "", True
        try:
            resp = await self.gateway.chat(
                REALISM_ROLE, realism_check_messages(image, level, minor=minor)
            )
        except KeyError:
            return True, -1, [], f"models.yaml 没配 {REALISM_ROLE} 角色，跳过校验", "", False
        except Exception as e:  # noqa: BLE001
            return True, -1, [], f"校验调用失败：{type(e).__name__}: {e}", "", True
        v = parse_realism_report(resp.text)
        note = "；".join(i for i in v.issues if "不是" in i and "JSON" in i)
        return v.passed, v.score, v.issues, note, v.direction, bool(note)

    # ---------- 整批报价（2026-09-26 用户定的） ----------
    # 之前超出开工额度后，每张参考图、每段视频各问一次 y/N（《不渡》一套参考图 174 张、额度
    # 80 张 → 一百多次），答 N 只拒一张、其余照问；而且媒体没填单价，问的时候也看不到钱。
    # 现在渲参考图、渲每一集之前把这一批报一次：要生成多少、质检重生成的最坏情况、金额（有单价
    # 时）、额度还剩多少 —— 人确认一次。批内调用照样过闸门、照样记账，超出额度时在单子范围内
    # 不再逐个问；中途要是比最坏情况还多、又问到人，答 N 整批停（见 gate.BatchPass）。

    def _price(self, kind: MediaKind, model: str, params: dict[str, Any]) -> float | None:
        price_of = getattr(self.catalog, "price_of", None)
        if not callable(price_of):
            return None
        try:
            v = price_of(kind, model, params)
        except Exception:  # noqa: BLE001 — 算不出价就当没单价
            return None
        return float(v) if v is not None else None

    def _worst_video_regen(self) -> int:
        """一段视频因质检最多再生成几次（字幕 / 镜头 / 人物一致合计，封顶 gate_max_regen）。"""
        sub_on, sub_n = self._subtitle_cfg()
        cut_on, cut_n, _, _ = self._cut_cfg()
        id_on, id_n, _ = self._identity_cfg()
        total = (sub_n if sub_on else 0) + (cut_n if cut_on else 0) + (id_n if id_on else 0)
        return max(0, min(self._gate_cfg()["max_regen"], total))

    async def _quote(
        self,
        tool: str,
        label: str,
        kind: str,
        head: str,
        worst: tuple[int, float, float | None],
        money: float | None,
        regen_note: str,
        args: dict[str, Any],
    ) -> tuple[BatchPass | None, str]:
        """整批报价问人一次。返回 (单子, 没确认时给模型的错误)。

        没有询问入口（脚本 / 测试 / 没接闸门）→ (None, "")：不开整批，照常逐次过闸门。
        worst = (最坏次数, 最坏视频秒数, 最坏金额；没单价为 None)。
        """
        gate = getattr(self.registry, "gate", None)
        confirm = getattr(gate, "confirm_batch", None)
        if not callable(confirm):
            return None, ""
        worst_n, worst_s, worst_m = worst
        unit = "段" if kind == "video" else "张"
        lines = [head]
        if regen_note:
            lines.append(
                f"质检最坏情况：{regen_note} → 最多 {worst_n} {unit}"
                + (f" / {worst_s:.0f} 秒" if worst_s else "")
            )
        if money is not None:
            lines.append(
                f"金额：约 ¥{money:.2f}"
                + (f"，最坏 ¥{worst_m:.2f}" if worst_m and worst_m > money + 0.005 else "")
            )
        else:
            lines.append(
                "金额：media_models.yaml 没填单价，这里看不到媒体要花多少钱，只按次数 / 秒数算"
            )
        guard = getattr(gate, "guard", None)
        room_fn = getattr(guard, "room", None)
        if callable(room_fn):
            room = room_fn(kind)
            parts: list[str] = []
            over = False
            if room.get("calls") is not None:
                parts.append(f"{room['calls']:.0f} {unit}")
                over = over or worst_n > room["calls"]
            if kind == "video" and room.get("seconds") is not None:
                parts.append(f"{room['seconds']:.0f} 秒")
                over = over or worst_s > room["seconds"]
            if room.get("money") is not None:
                # 已花的钱里有按估价计的（没配单价的文本模型，09-29 审查 2.1）：剩多少也跟着是估的
                usage = getattr(guard, "usage", None)
                est = getattr(usage, "estimate_note", None)
                parts.append(f"¥{room['money']:.2f}" + (est() if callable(est) else ""))
                over = over or bool(worst_m and worst_m > room["money"])
            if parts:
                lines.append(
                    "额度还剩：" + "、".join(parts) + "（本次开工与单日取更紧的）"
                    + ("；最坏情况超出的部分，确认即为这一批放行（只这一批，不改额度设置）"
                       if over else "")
                )
        lines.append(
            f"回 y 开始；回 N 这一批一{unit}都不发。"
            "中途要是比最坏情况还多、又要问你，答 N 就整批停。"
        )
        ok = await confirm(tool, "\n".join(lines), args)
        if ok is None:
            return None, ""
        if not ok:
            return None, (
                f"用户在整批报价上没有确认（{label}），一{unit}都没有生成、没有花钱。"
                "不要换个方式绕过去（拆成小批、改用 gen_image / gen_video）；问用户要怎么调整"
                "（少渲一些、换档位、先渲一部分），他同意了再调"
            )
        return BatchPass(label=label, units={kind: worst_n}, seconds=worst_s, money=worst_m), ""

    def _previous_pack(self, assets_id: str) -> tuple[str, dict[str, dict[str, str]]]:
        """这套资产库最新的参考图包 (id, images)。没有就 ("", {})。"""
        for a in self.store.find(creator="tool:drama_render_assets"):
            if assets_id in a.parent_ids:
                images = _load_pack(self.store.content(a.id))
                if images:
                    return a.id, images
        return "", {}

    def _inherited_pack(
        self, assets_id: str, lib: Any
    ) -> tuple[str, dict[str, dict[str, str]], list[str]]:
        """渲参考图时能沿用的旧包：这套库自己的；没有就沿血缘（增量补充 / 整份重做都记了 base）
        往上找最近一套渲过包的库，只留**名字和描述都没变**的条目（2026-09-26 用户定的：资产库
        定稿后只增量补新集，参考图按名字沿用；之前换一版资产库就是 174 张全部重生成、全员换脸）。
        返回 (包 id, 能沿用的图, 描述改了没沿用的名字)。"""
        pid, images = self._previous_pack(assets_id)
        if images:
            return pid, images, []
        seen: set[str] = {assets_id}
        cur = assets_id
        for _ in range(30):
            try:
                cur = str((self.store.get(cur).gen_params or {}).get("base") or "")
            except KeyError:
                break
            if not cur or cur in seen:
                break
            seen.add(cur)
            pid, images = self._previous_pack(cur)
            if not images:
                continue
            try:
                old_lib, err = parse_assets(self.store.content(cur))
            except KeyError:
                old_lib, err = None, "gone"
            if old_lib is None or err:
                return "", {}, []
            before, now = _entry_prompts(old_lib), _entry_prompts(lib)
            keep = {k: v for k, v in images.items() if k in now and before.get(k) == now[k]}
            changed = sorted(k for k in images if k in now and k not in keep)
            return pid, keep, changed
        return "", {}, []

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
        want = _RENDER_ONLY.get(only, _RENDER_ONLY["all"])
        # 本地文件名：类别-序号_名字，按资产库原序编号，后期按名字就能排
        names = library_reference_names(lib)

        # 增量：上次已经生成过的图直接复用，只补缺的。
        # 实测模型先 only=characters 省钱，之后再补服装/场景 —— 不复用的话主形象
        # 会再花一遍钱，而且新主形象和旧的不是同一张脸。
        # 上一个包不管 reuse 与否都读：这次没要求生成的类别要原样带进新包（2026-09-23 审查：
        # 之前 only=scenes 出的新包里只有场景，下游默认取最新的包，渲视频时人物全没了、被引用门
        # 拦下；再跑 all 又换脸又重复付费）。reuse 只管要生成的这几类能不能沿用旧图
        # 这套库自己没渲过包的（增量补新集 / 整份重做出的新一版）：沿血缘拿上一版的包，
        # 名字和描述都没变的图接着用，描述改了的重生成（2026-09-26）
        prev_id, prev, desc_changed = self._inherited_pack(assets_id, lib)
        # 上次渲到一半断了（超时 / /stop / 关了窗口）：出好的图记在进度里，这次接着用、不重付
        # （2026-09-29 审查 1.4：之前要到整批渲完才落包，一超时已付费的图不进任何包，重跑全部
        # 重付）。进度按资产库 id 记，资产库内容不会变，名字对上就是同一个条目
        resumed_from = self._scratch_load("refpack", assets_id)
        if not isinstance(resumed_from, dict):
            resumed_from = {}
        if resumed_from:
            prev = {**prev, **{k: dict(v) for k, v in resumed_from.items() if isinstance(v, dict)}}
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

        def remember(name: str, entry: dict[str, str]) -> None:
            """出好一张就记进进度（不进资产库：半成品包会被流水线当成渲完了去渲视频）。"""
            so_far = dict(self._scratch_load("refpack", assets_id) or {})
            so_far[name] = entry
            self._scratch_save("refpack", assets_id, so_far)

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
        if "characters" in want:
            # 不管这次要不要服装都得算：only=characters 重生成了脸，旧脸服装留在新包里
            # （portrait 对不上）就会一直传到视频（2026-09-24 审查）
            face_changed = {c.name for c in lib.characters if c.name not in reused}
        if "costumes" in want:
            for c in lib.characters:
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

        # ---- 整批报价（2026-09-26 用户定的）：一共要新生成几张、质检最坏几张，确认一次 ----
        bp: BatchPass | None = None
        if total:
            rr = max(0, self._realism_retries())  # -1 = 真实感校验关着
            id_on, id_n, _ = self._identity_cfg()
            ir = id_n if id_on else 0
            n_char = sum(1 for p in phase_a if p[0] == "角色")
            n_cos = len(planned_costumes)
            n_other = len(phase_a) - n_char
            worst_n = n_char * (1 + rr) + n_cos * (1 + rr + ir) + n_other
            each = self._price(MediaKind.IMAGE, self.image_model, {})
            regen = "、".join(
                x for x in (
                    f"人物图不够真实每张最多重生成 {rr} 次" if rr and (n_char or n_cos) else "",
                    f"服装图和主形象不像再重生成 {ir} 次" if ir and n_cos else "",
                ) if x
            )
            bp, deny = await self._quote(
                "drama_render_assets",
                f"参考图 {total} 张",
                "image",
                f"参考图要新生成 {total} 张（角色主形象 {n_char}、服装 {n_cos}、场景道具 {n_other}"
                + (f"；另有 {len(reused)} 张复用上次的" if reused else "")
                + f"），生图模型 {self.image_model}",
                (worst_n, 0.0, each * worst_n if each is not None else None),
                each * total if each is not None else None,
                regen,
                {"assets_id": assets_id, "新生成": total, "最坏": worst_n},
            )
            if deny:
                return ToolResult(ok=False, error=deny, meta={"charged": False})

        async def gen_tracked(label: str, name: str, *args: Any, **kwargs: Any) -> Any:
            nonlocal done
            if bp is not None and bp.stopped:
                r: Any = ("", "", "这一批已被你叫停，没有生成")
            else:
                r = await self._gen_image(*args, report=gate_log, **kwargs)
            done += 1
            await self._progress("渲染参考图", done, total, f"{label}·{name}")
            return r

        async def one_a(p: tuple[str, str, str, str, bool]) -> Any:
            r = await gen_tracked(
                p[0], p[1], p[2], p[3], f"{p[0]}·{p[1]}",
                person=p[4], local_name=names.get(p[1], ""),
            )
            aid, url, e = r
            if not e:
                remember(p[1], {"asset": aid, "url": url, "kind": p[0]})
            return r

        await self._progress("渲染参考图", 0, total)
        with batch_scope(bp):
            res_a = await _run_parallel(phase_a, one_a, limit)
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
            async def one_b(p: tuple[str, str, list[str], str, bool]) -> Any:
                r = await gen_tracked(
                    "服装", p[0], p[1], "16:9", f"服装·{p[0]}",
                    ref=p[2], person=True, local_name=names.get(p[0], ""), minor=p[4],
                )
                aid, url, e = r
                verdict = next(
                    (x.get("identity") or {} for x in gate_log
                     if isinstance(x, dict) and x.get("asset") == aid),
                    {},
                )
                # 和主形象不像的不记（下面也不进包），其余的出好一张记一张
                if not e and verdict.get("pass") is not False:
                    remember(p[0], {"asset": aid, "url": url, "kind": "服装", "portrait": p[3]})
                return r

            with batch_scope(bp):
                res_b = await _run_parallel(phase_b, one_b, limit)
            idcheck = {
                str(x.get("asset")): x.get("identity") or {}
                for x in gate_log if isinstance(x, dict) and x.get("asset")
            }
            for (name, _p, _r, face_id, _m), (aid, url, e) in zip(phase_b, res_b, strict=True):
                if e:
                    failed.append(f"服装 {name}：{e[:70]}")
                    continue
                verdict = idcheck.get(aid) or {}
                if verdict.get("pass") is False:
                    # 和主形象不像（重生成过还是不像）：不进包 —— 进了包，所有穿这套的段一致性
                    # 都过不了，每段多付一次、最后还是 ⛔。渲视频时退回主形象（2026-09-26）
                    score = verdict.get("score")
                    failed.append(
                        f"服装 {name}：和主形象不像（{score}/10），没进参考图包，渲视频时先退回"
                        f"主形象；要补就 drama_render_assets(only=\"costumes\", reuse=true) 重生成"
                    )
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
                if rec.get("unchecked"):
                    entry["realism"]["unchecked"] = True
        # 人物一致没查成的服装图：包里记一笔，别让人以为比对过（2026-09-29 审查 1.2）
        for x in gate_log:
            ident = x.get("identity") if isinstance(x, dict) else None
            if isinstance(ident, dict) and ident.get("unchecked"):
                for entry in merged.values():
                    if entry.get("asset") == x.get("asset"):
                        entry["identity"] = {"unchecked": True, "note": ident.get("note", "")}

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
        # 整批落进包里了，进度不用再留
        self._scratch_drop("refpack", assets_id)

        fresh = len(images) - len(reused)
        resumed = [n for n in reused if n in resumed_from]
        head = (
            f"参考图包共 {len(merged)} 张（本次新生成 {fresh}，复用 {len(reused)}"
            + (f"，其中 {len(resumed)} 张是上次渲到一半留下的、接着用没重付" if resumed else "")
            + (f"，沿用上一个包 {len(kept)}" if kept else "")
            + "）：\n"
        )
        warn = ("\n\n未生成：\n  " + "\n  ".join(failed)) if failed else ""
        if desc_changed:
            shown = "、".join(desc_changed[:6]) + (" 等" if len(desc_changed) > 6 else "")
            warn += (
                f"\n\n这一版资产库里 {len(desc_changed)} 个条目的描述和上一版不同，旧图没沿用"
                f"（{shown}）"
            )
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
        aspect_ratio: str = "",
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

        # 花钱之前核对：提示词有没有漏掉分镜里的镜头（2026-09-25：第 3 集的提示词只写到分镜
        # 第 128 镜，照着渲就是半集；第 1 集那份也缺最后 7 镜）
        gap_err = self._coverage_gate(shots_id, shots)
        if gap_err:
            return ToolResult(ok=False, error=gap_err, meta={"charged": False})
        # 画幅：这次指定的 > 项目设置（/ratio）> 默认竖屏。锁定的视频模型不支持就花钱之前拦下
        ratio = parse_aspect(aspect_ratio) or self.aspect_ratio or DEFAULT_ASPECT
        if aspect_ratio and not parse_aspect(aspect_ratio):
            return ToolResult(
                ok=False, meta={"charged": False},
                error=f"认不出画幅 {aspect_ratio!r}：可选 16:9（横屏）/ 9:16（竖屏）/ 1:1（方形）",
            )
        bad_ratio = unsupported_note(self.catalog, self.video_model, ratio)
        if bad_ratio:
            return ToolResult(ok=False, error=bad_ratio, meta={"charged": False})

        total_shots = len(shots)
        if limit > 0:
            shots = shots[:limit]
        # 只渲了前几段（limit）：这一集不算渲完 —— 之前照样 complete=True 落带集号的索引，
        # 按集流水把只渲了 2 段的集当渲完，默认还把这 2 段拼成整集成片（2026-09-24 审查）
        truncated = 0 < limit < total_shots

        # ---- 参考图包：定位 + 校验 + 引用预匹配（花钱之前做完）----
        pack_id, images, pack_notes, err = self._locate_pack(rendered_id, shots_id)
        if err:
            return ToolResult(ok=False, error=err)
        # 参考图包必须是这份提示词那套资产库的（2026-09-25：没迁移的旧剧包在新剧里看得见，
        # 同名角色（唐僧、悟空…）能对上，会拿旧剧的脸渲新剧）
        if pack_id:
            shots_lib = library_of_shots(self.store, shots_id)
            pack_lib = pack_library(self.store, pack_id)
            # 同一条增量链上的算同一套库（定稿后只增量补新集，已有条目原样不动，2026-09-26）
            if shots_lib and pack_lib and not same_library(self.store, shots_lib, pack_lib):
                return ToolResult(
                    ok=False,
                    meta={"charged": False},
                    error=(
                        f"参考图包 {pack_id} 属于另一套资产库（{pack_lib}），"
                        f"这份提示词用的资产库是 {shots_lib} —— 拿别的剧的参考图渲，"
                        "人物会换成别的剧的脸，没有发起生成。"
                        "不传 rendered_id，让它自动找这套资产库的包；这套资产库还没渲参考图就先 "
                        f'drama_render_assets(assets_id="{shots_lib}")'
                    ),
                )
        # 参考图链接过期/本地：配了托管就先重新上传（图不变、链接换新），不然模型拿不到参考
        images, host_notes = await self._rehost_pack(pack_id, images)
        pack_notes += host_notes
        # 包的内容签名（每个键用的是哪份图片资产）：片段复用按它判断「参考图换没换」，
        # 重新托管只换链接不换资产，签名不变
        pack_sig = pack_identity(images) if images else ""
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
        # ---- 段指纹（2026-09-26 用户定的：按段复用，承接的段连带重渲）----
        # 前序引入「{第X集-Y场}」指向列表里排在自己前面、同场景的最后一段（和下面的分层同一套）
        owner_early = (
            {cos.name: c.name for c in lib.characters for cos in c.costumes} if lib else {}
        )
        scene_last: dict[str, int] = {}
        fps: list[str] = []
        for i, s in enumerate(shots):
            carried = [fps[j] for c in s.carries() if (j := scene_last.get(c)) is not None]
            fps.append(
                _segment_fp(s, resolved, images, owner_early, ratio, self.video_model, carried)
            )
            scene_last[s.scene_index.strip("[]")] = i
        fp_of = {(s.scene_index, s.video_name): fps[i] for i, s in enumerate(shots)}
        # 增量重跑：上次已成功的段直接复用，只生成失败/缺的 —— 引用门只查这次要生成的段
        recheck: list[tuple[tuple[str, str], str]] = []
        prev_clips = (
            self._previous_clips(
                shots_id, episode, pack_id, redo, accept, pack_sig, ratio,
                fps=fp_of, recheck=recheck,
            )
            if reuse
            else {}
        )
        # 上次只因为「没查成」（字幕 / 人物一致）被判 ⛔ 的段：先补下载、重查一次，过了就复用，
        # 不重付（字幕 2026-09-26、人物一致 2026-09-27）
        early_scenes = lib.scene_names() if lib else set()
        shot_at = {(s.scene_index, s.video_name): s for s in shots}

        def id_refs_of(key: tuple[str, str]) -> list[tuple[str, str]]:
            s = shot_at.get(key)
            if s is None:
                return []
            return _segment_refs(s, resolved, images, owner_early, early_scenes).id_refs

        rechecked = await self._recheck_unchecked(
            [(k, aid) for k, aid in recheck if k not in prev_clips and not _label_hit(k, redo)],
            id_refs_of,
        )
        prev_clips.update(rechecked)
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
        # 人定的锚点（定音）链接过期、又没能重新托管：这一集临时用本集的独白段当锚点（集内声音
        # 一致），不改人定的。报价时先说、结果里照实说（2026-09-27：之前说「已接替」，锚点表其实
        # 没动，还白存一份一样的表）
        held = [
            n for n in plan.stale if (a := anchors.get(n)) is not None and a.pinned
        ] if vcfg["enabled"] else []
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
        blocked_scenes: set[str] = set()  # 最新一段没过质检门的场次：后面承接它的段先不渲
        vlimit = self._max_concurrency("video")
        vmax = vcfg["max_videos"]
        done_v = 0
        # 字幕检查连续几段都没查成（视觉接口多半不通）：熔断，这一批后面的段先不渲（2026-09-26）
        halt = {"why": "", "streak": 0}
        try:
            unchecked_break = max(
                1, int((getattr(self.catalog, "drama", {}) or {}).get("unchecked_break", 3))
            )
        except (TypeError, ValueError):
            unchecked_break = 3

        # ---- 整批报价（2026-09-26 用户定的）：这一集要新渲几段、几秒、质检最坏情况，确认一次 ----
        bp: BatchPass | None = None
        if todo:
            regen = self._worst_video_regen()
            secs = [float(shots[i].seconds or FALLBACK_SECONDS) for i in todo]
            n_new, total_s = len(todo), sum(secs)
            prices = [
                self._price(
                    MediaKind.VIDEO, self.video_model, {"duration": x, "resolution": "720p"}
                )
                for x in secs
            ]
            priced = [p for p in prices if p is not None]
            money = float(sum(priced)) if len(priced) == len(prices) else None
            where = f"第 {episode} 集" if episode else "这批镜头"
            # 以前渲过、过了质检，这次因为内容 / 参考图 / 前序段变了要重付的段：单独说出来
            # （2026-09-26 用户定的兜底：重付已通过的段之前先问）
            passed = self._passed_keys(episode)
            repay = [
                f"{shots[i].scene_index} {shots[i].video_name}" for i in todo
                if (shots[i].scene_index, shots[i].video_name) in passed
            ]
            repay_note = (
                f"；其中 {len(repay)} 段以前渲过、过了质检，这次因为提示词 / 参考图 / 前序段变了"
                f"要重渲（{'、'.join(repay[:4])}{' 等' if len(repay) > 4 else ''}）"
                if repay else ""
            )
            short = self._short_on_refs(
                shots, todo, resolved, images, owner_of, lib_scene_names,
                speakers_by_shot, anchor_urls, plan, vmax, max_refs,
            )
            short_note = (
                f"；⚠ 参考位不够：{len(short)} 段要省掉场景 / 道具参考图"
                f"（{'；'.join(short[:3])}{' 等' if len(short) > 3 else ''}），"
                "这几段的场景和道具可能和别的段对不上"
                if short else ""
            )
            if short:
                pack_notes.append(
                    f"⚠ 参考位不够（上限 {max_refs}）：{len(short)} 段省掉了场景 / 道具参考图"
                )
            bp, deny = await self._quote(
                "drama_render_shots",
                f"{where} {n_new} 段",
                "video",
                f"{where}要新渲 {n_new} 段视频"
                + (f"（另有 {len(shots) - n_new} 段复用上次的）" if len(shots) > n_new else "")
                + f"，共约 {total_s:.0f} 秒，模型 {self.video_model} · 720p · {ratio}"
                + repay_note
                + short_note
                + (
                    f"；⚠ 你定音的 {'、'.join(held)} 锚点链接过期了、没能重新托管，这一集他们的声音"
                    "只能按音色卡和本集的独白段定，会和你定的不一样"
                    if held else ""
                ),
                (
                    n_new * (1 + regen),
                    total_s * (1 + regen),
                    money * (1 + regen) if money is not None else None,
                ),
                money,
                f"每段质检（字幕 / 镜头时长 / 人物一致）最多再生成 {regen} 次" if regen else "",
                {"shots_id": shots_id, "集": episode or "全部", "新渲": n_new,
                 "最坏": n_new * (1 + regen)},
            )
            if deny:
                return ToolResult(ok=False, error=deny, meta={"charged": False})

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
                if bp is not None and bp.stopped:
                    # 中途被问到时你答了 N：整批停，后面的一段都不发
                    return i, "", (0, 0), "这一批已被你叫停，没有生成", False, [], False
                if halt["why"]:
                    return i, "", (0, 0), halt["why"], False, [], False
                # 要接的前序段没过质检门（⛔）：先不渲 —— 接着一段错脸 / 带字的尾帧渲出来也得重渲
                # （2026-09-26：之前 ⛔ 的段照样当前序参考和音色锚点，错往后传）
                bad_prev = [c for c in s.carries() if c in blocked_scenes]
                if bad_prev:
                    return (
                        i, "", (0, 0),
                        f"要接的前序段（{'、'.join(bad_prev)}）没过质检，这段先没渲 —— "
                        "那段放行（accept）或重渲（redo）后再跑一次会补上",
                        False, [], False,
                    )
                # 引用到的资产图当参考（人物优先、服装配脸、再场景道具）：和报价、补查共用一套
                seg = _segment_refs(s, resolved, images, owner_of, lib_scene_names)
                label_of = seg.label_of
                id_refs = seg.id_refs
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
                # 参考位上限：人物必保，位子不够先省道具、再省场景（报价时已经说过哪几段要省）
                refs, gone = _trim_refs(seg, len(videos), max_refs)
                ref_labels = [label_of[u] for u in refs if u in label_of]
                trim_notes: list[str] = []
                if gone:
                    trim_notes.append(f"参考位不够（上限 {max_refs}），省略 {'、'.join(gone)}")
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
                    "aspect_ratio": ratio,
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
                    "pack_sig": pack_sig,
                    "episode": episode,
                    "aspect": ratio,
                    "fp": fps[i],  # 段指纹：内容、参考图、前序段都没变的段下次直接复用
                }
                # 上次提交过、没等到结果的任务（含质检重生成的那次）：先取回，不重新付费
                first, got_note = await self._recover_segment(fps[i], args)
                # 网络类失败自动重试 + 人物一致性门 + 字幕门
                r, clip_notes = await self._gen_clip(
                    args, retries, sub_gate, sub_retries, id_refs=id_refs, first=first
                )
                if got_note:
                    clip_notes = [got_note, *clip_notes]
                if not r.ok and (r.meta or {}).get("quota") and not halt["why"]:
                    # 余额 / 额度用完（2026-09-29 审查 1.1）：后面的段一段都别发，原因写在最前面
                    halt["why"] = (
                        f"余额 / 额度用完了（{(r.error or '')[:160]}），这一批后面的段没发。"
                        "重试、换模型、拆小批都没用：告诉用户去充值（或等额度窗口恢复），"
                        "渲好的段都留着，恢复后原样再跑一次只补没渲的"
                    )
                    halt["quota"] = True
                blocked = False
                if r.ok and r.asset_ref:
                    blocked = not self._mark_clip(r.asset_ref, clip_notes)
                    # 字幕 / 人物一致「没查成」一起算熔断（2026-09-27）
                    if any(_is_unchecked(n) for n in clip_notes):
                        halt["streak"] += 1
                        if halt["streak"] >= unchecked_break and not halt["why"]:
                            halt["why"] = (
                                f"质检（字幕 / 人物一致）连续 {halt['streak']} 段没查成"
                                "（视觉接口多半不通），这一批后面的段先没渲 —— 查不成的段会判 ⛔、"
                                "不进成片，接着渲只是白花钱。接口恢复后再跑一次：没查成的段会先补查，"
                                "过了就复用"
                            )
                    elif sub_gate or id_refs:
                        halt["streak"] = 0
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
            with batch_scope(bp):
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
                # 只有远端且未过期的链接能当后面镜头的参考（前序引入 / 音色锚点）。
                # 没过质检门的段不当前序、不当锚点：错脸 / 带字的尾帧会一路传下去（2026-09-26）
                url, _ = self._clip_ref_url(aid, vcfg)
                scene_key = s.scene_index.strip("[]")
                if was_blocked:
                    by_scene[scene_key] = ""
                    blocked_scenes.add(scene_key)
                    url = ""
                else:
                    by_scene[scene_key] = url
                    blocked_scenes.discard(scene_key)
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

        # 余额 / 额度用完停下的：原因放最前面，后面一串失败都是它（2026-09-29 审查 1.1）
        quota_head = f"⛔ {halt['why']}\n\n" if halt.get("quota") else ""
        if not done:
            return ToolResult(
                ok=False, error=quota_head + "一段视频都没生成：\n" + "\n".join(failed)
            )

        anchors_id = ""
        keep_new = {n: a for n, a in new_anchors.items() if n not in held}
        if keep_new:
            anchors_id = self._merge_save_anchors(keep_new, [pack_id, shots_id])

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
            "complete": not failed and not blocked and not truncated,
            "failed": len(failed),
            "blocked": len(blocked),
            "aspect": ratio,
        }
        if truncated:
            rgp["limit"] = limit
            rgp["total_shots"] = total_shots
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
        too_long: list[str] = []
        for i in sorted(done):
            hit = next((x for x in notes_of.get(i, []) if "仍有超过" in x), "")
            if not hit:
                continue
            m = re.search(r"最长 ([\d.]+)s", hit)
            too_long.append(
                f"{shots[i].scene_index} {shots[i].video_name}"
                + (f"（最长 {m.group(1)}s）" if m else "")
            )
        if too_long:
            warn += (
                f"\n\n⚠ {len(too_long)} 段重生成后仍有超过 {self.fmt.max_cut_seconds:g} 秒的镜头"
                f"（用户要求每镜 ≤3 秒）：{'；'.join(too_long)}"
            )
            if not self.cut_block:
                warn += (
                    "\n  现在只标出来、照样进成片（检测准不准还没验证）。请用户抽看 3–5 段：5 段里"
                    "至少 4 段确实超了，就用 /cut 拦（以后超了自动重生成 1 次、仍超不进成片）；"
                    "误判多就先调 media_models.yaml 的 cut_scene_threshold 再看一集"
                )
        notes = ("\n\n" + "\n".join(pack_notes)) if pack_notes else ""
        notes += _voice_note(lib, plan, new_anchors, no_anchor, anchors_id, vcfg, held)
        if truncated:
            warn += (
                f"\n\n只渲了前 {len(shots)}/{total_shots} 段（limit），这一集不算渲完、没有拼接；"
                "去掉 limit 再跑会复用这几段、只补其余的"
            )
        fresh = len(ordered) - len(reused)
        head = quota_head + f"生成 {len(ordered)} 段视频"
        if reused:
            head += f"（复用 {len(reused)}，新生成 {fresh}）"
        head += "：\n" + "\n".join(lines) + notes + warn

        state = {"complete": not failed and not blocked and not truncated,
                 "failed": len(failed), "blocked": len(blocked)}
        if truncated:
            state["truncated"] = f"{len(shots)}/{total_shots}"
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
        if not compose or truncated:
            clips = " ".join(d["asset"] for d in ordered)
            return ToolResult(
                content=f"{head}\n\n片段资产：{clips}\n资产 {asset.id}", asset_ref=asset.id,
                meta=state,
            )

        # 片段**自带音轨** —— seedance 原生出声，台词和环境音都在里面。
        # 拼接会保留它，所以**不要再单独做 TTS 配音**：那等于用合成语音
        # 盖掉模型生成的原声，口型也对不上。
        name = filename or (episode_export_name(episode) if episode else "短剧.mp4")
        # 上屏字（字幕卡，2026-09-26）：生成的画面里不许有字，拼完先出无字版，再叠字出正式成片
        overlays = await self._overlay_items(shots, done)
        res = await self.registry_invoke(
            "compose_video",
            {
                "clips": [d["asset"] for d in ordered],
                "out_dir": out_dir,
                "filename": _no_text_name(name) if overlays else name,
                # 片段生成时已按时间线硬切过（每镜 ≤3s），拼接时不能再切，否则台词被剪断
                "keep_whole": True,
                "aspect_ratio": ratio,
            },
        )
        if not res.ok:
            return ToolResult(
                content=f"{head}\n\n⚠ 拼接失败：{res.error}", asset_ref=asset.id,
                meta={**state, "complete": False},
            )
        if not overlays:
            return ToolResult(
                content=f"{head}\n\n{res.content}", asset_ref=res.asset_ref, meta=state
            )
        ov = await self.registry_invoke(
            "overlay_text",
            {"video": res.asset_ref, "items": overlays, "out_dir": out_dir, "filename": name},
        )
        shown = "、".join(f"「{x['text']}」" for x in overlays[:4]) + (
            f" 等 {len(overlays)} 张" if len(overlays) > 4 else ""
        )
        if not ov.ok:
            return ToolResult(
                content=(
                    f"{head}\n\n{res.content}\n\n⚠ 上屏字没叠上（{(ov.error or '')[:120]}）："
                    f"上面是无字版，{shown} 还没上屏 —— 修好后用 overlay_text 叠"
                ),
                asset_ref=res.asset_ref,
                meta={**state, "complete": False},
            )
        return ToolResult(
            content=(
                f"{head}\n\n{res.content}\n\n上屏字 {shown} 已叠上（本地叠字，不花钱）：\n"
                f"{ov.content}\n（上面那条 _无字 的是没叠字的版本，要改字就对它重跑 overlay_text）"
            ),
            asset_ref=ov.asset_ref,
            meta=state,
        )

    async def _overlay_items(
        self, shots: list[Any], done: dict[int, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """这一集要叠的上屏字，换算成成片上的绝对时间：各段按实际片长（量不出来按计划时长）
        首尾相接。片尾卡叠到这一集结束。"""
        if not any(shots[i].screen_text for i in done):
            return []
        items: list[dict[str, Any]] = []
        start = 0.0
        for i in sorted(done):
            s = shots[i]
            length = float(s.seconds or FALLBACK_SECONDS)
            local = self._local_video(done[i]["asset"])
            if local is not None:
                try:
                    measured = (await ffmpeg.probe(local)).duration
                except Exception:  # noqa: BLE001 — 量不出来按计划时长
                    measured = 0.0
                if measured > 0:
                    length = measured
            for x in s.screen_text:
                at = start + float(x.get("at") or 0.0)
                items.append({
                    "text": str(x.get("text") or ""),
                    "start": round(at, 2),
                    "end": round(at + float(x.get("dur") or 2.5), 2),
                    "ending": bool(x.get("end")),
                })
            start += length
        for it in items:
            if it.pop("ending"):
                it.update({"end": round(start, 2), "position": "center", "size": "xl"})
            else:
                it.update({"end": min(it["end"], round(start, 2)), "position": "lower_third",
                           "size": "l"})
        return [it for it in items if it["text"]]


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
    """参考图包 = {名字: {asset, url, kind?}}。不是这个形状就返回空（实现在 drama/refpack）。"""
    return load_pack(content)


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


# 第③步喂给模型的分镜带镜号和场次（format.scene_blocks）：怎么用这些编号
_SHOTS_NUMBERING = (
    "\n\n【镜号与场次】分镜里每个镜头行开头的〔N〕是这一集的镜号，video_name 用它写覆盖范围"
    "（如 \"12-16\"）；每场开头的【第N集-M场】是场次，scene_index 写成 [第N集-M场]。〔N〕和"
    "【…】只是编号，不要写进 description。分镜里的每个镜头都要写进某一段，一个都不能少。"
)
_SHOTS_FAITHFUL = (
    "每个镜头保留分镜标的时长（cuts 按它来），一段放不下就多分几段，不要合并或删镜头。"
)


def _chunk_problems(
    shots: list[Any], fmt: EpisodeFormat, nums: set[int], secs_by_no: dict[int, float],
    hook: bool,
) -> list[str]:
    """一批视频提示词的问题：规格（这一批的时长）+ 漏掉的镜头 + 压缩了的段。"""
    problems = check_shots(shots, fmt, hook=hook)
    gaps = coverage_gaps(shots, nums)
    if gaps:
        problems.insert(
            0,
            f"分镜第 {format_ranges(gaps)} 镜没有写进任何一段："
            "分镜里的每个镜头都要写，一个都不能少",
        )
    return problems + compression_problems(shots, secs_by_no)


def _softened_changes(eps: list[Any], script: str) -> str:
    """用克制措辞重写过的分镜和剧本比：丢了哪几句台词、少了几场。没变返回空串。"""
    by_ep = split_script(script)
    parts: list[str] = []
    for e in eps:
        src = by_ep.get(e.index) or (script if not by_ep and len(eps) == 1 else "")
        if not src:
            continue
        _, miss = missing_dialogue(src, e.desc)
        if miss:
            shown = "、".join(f"「{m[:12]}」" for m in miss[:3])
            parts.append(f"{e.title} 少了 {len(miss)} 句台词，如 {shown}")
        want, got = scene_count(src), len(e.scenes)
        if want and got < want:
            parts.append(f"{e.title} 场次从 {want} 场变成 {got} 场")
    return "；".join(parts)


# 资产库定稿后只增量补新集（2026-09-26 用户定的）：追加在资产库系统提示词后面
_EXTEND_RULES = """

【增量补充（这部剧的资产库已经定稿）】
下面【已定稿的资产库】里的角色、服装、场景、道具都已经生成过参考图，**原样沿用：不要改写描述、
不要重复输出、不要换个叫法另起一条**（「观音」和「观音菩萨」是同一个人，「如来」和「如来佛祖」也是）。
只为【新增的剧本】里出现、名单里没有的东西写新条目，JSON 结构和上面的要求一样：
· 新角色：完整的 roleTotalDesc、音色卡、按场景分配的服装；
· 已有角色在新剧情里要穿的新服装：baseRoleName 用名单里的原名，roleTotalDesc 留空，
  roleCostumeList 里只写新服装（scenes / episodes 照常标）；
· 新场景、新道具。
没有新东西的类别输出空数组，比如 {"characters": [], "scenes": [], "props": []}。"""


def _frozen_digest(lib: Any) -> str:
    """给增量补充的模型看的定稿名单：角色（带一句形象）、各自的服装、场景、道具 —— 用来认出
    「这个人已经有了」，不给完整描述（那些不许改）。"""
    lines = ["角色："]
    for c in lib.characters:
        look = " ".join((c.body or "").split())[:30]
        cos = "、".join(x.name for x in c.costumes) or "（无）"
        lines.append(f"  - {c.name}（{look}…）｜服装：{cos}")
    if lib.scenes:
        lines.append("场景：" + "、".join(s.name for s in lib.scenes))
    if lib.props:
        lines.append("道具：" + "、".join(p.name for p in lib.props))
    return "\n".join(lines)


def _parse_additions(text: str) -> tuple[Any, str]:
    """增量补充的输出：没有新角色是正常的（只补了场景 / 道具，或什么都不用补）。"""
    lib, err = parse_assets(text)
    if err.startswith("没解析出任何角色"):
        return lib, ""
    return lib, err


def _merge_library(base: Any, add: Any) -> tuple[Any, dict[str, list[str]]]:
    """定稿的库 + 增量：已有的名字一律不动（模型重复输出的丢掉），已有角色只追加新服装。
    返回 (新一版资产库, 各类新增了哪些)。"""
    import copy

    merged = copy.deepcopy(base)
    names = merged.all_names()
    by_char = {c.name: c for c in merged.characters}
    added: dict[str, list[str]] = {"角色": [], "服装": [], "场景": [], "道具": []}
    for c in add.characters:
        target = by_char.get(c.name)
        if target is None:
            if c.name in names:
                continue  # 和已有的服装 / 场景重名：不是新角色
            fresh = copy.deepcopy(c)
            fresh.costumes = [x for x in c.costumes if x.name not in names]
            merged.characters.append(fresh)
            by_char[c.name] = fresh
            names.add(c.name)
            added["角色"].append(c.name)
            for x in fresh.costumes:
                names.add(x.name)
                added["服装"].append(x.name)
            continue
        for x in c.costumes:
            if x.name not in names:
                target.costumes.append(copy.deepcopy(x))
                names.add(x.name)
                added["服装"].append(x.name)
    for bucket, label, extra in (
        (merged.scenes, "场景", add.scenes), (merged.props, "道具", add.props)
    ):
        for n in extra:
            if n.name not in names:
                bucket.append(copy.deepcopy(n))
                names.add(n.name)
                added[label].append(n.name)
    return merged, added


def _casting_line(lines: list[str]) -> str:
    """定音用哪句台词：5 秒念得完（按偏快语速）的里面挑最长的；都太长就挑最短的那句。"""
    fits = [x for x in lines if 2.0 <= speech_seconds(x) <= 4.5]
    if fits:
        return max(fits, key=len)
    return min(lines, key=len) if lines else ""


def _entry_prompts(lib: Any) -> dict[str, str]:
    """资产库每个条目（角色 / 服装 / 场景 / 道具）生图用的提示词：名字 → 提示词。
    沿用旧参考图时按它判「描述变没变」。"""
    out: dict[str, str] = {}
    for c in lib.characters:
        out[c.name] = c.prompt()
        for cos in c.costumes:
            out[cos.name] = cos.prompt()
    for n in [*lib.scenes, *lib.props]:
        out[n.name] = n.prompt()
    return out


def _lost_count(eps: list[Any], script: str) -> int:
    """分镜比剧本少了几句台词、几张字幕卡（逐句逐张数，不设门槛）。挑修订版时用：
    修订版不能比上一版丢得多。"""
    by_ep = split_script(script)
    n = 0
    for e in eps:
        src = by_ep.get(e.index) or (script if not by_ep and len(eps) == 1 else "")
        if src:
            n += len(missing_dialogue(src, e.desc)[1]) + len(missing_cards(src, e.desc))
    return n


def _segment_fp(
    s: Any,
    resolved: dict[str, Any],
    images: dict[str, dict[str, str]],
    owner_of: dict[str, str],
    ratio: str,
    model: str,
    carry_fps: list[str],
) -> str:
    """一段视频的指纹（2026-09-26 用户定的：按段复用）：画面描述、切镜时长、段长、实际用到的
    参考图（按图片资产 id，不按会换的链接；服装图连带角色主形象）、画幅、模型、承接的前序段的
    指纹。指纹没变的段直接复用，不管它属于哪份提示词、哪个参考图包 —— 之前复用要求提示词 id
    和整包签名都不变，换一张道具图、改一段提示词，没渲完的集就整集重付。
    前序段的指纹算在里面：前序段重渲了，承接它尾帧的段跟着重渲。"""
    refs: set[str] = set()
    for r in s.refs():
        m = resolved.get(r)
        if m is None:
            continue
        refs.add(str((images.get(m.key) or {}).get("asset") or m.url))
        who = owner_of.get(m.key.removesuffix("·三视图"), "")
        face = str((images.get(who) or {}).get("asset") or "") if who else ""
        if face:
            refs.add(face)
    payload = {
        "d": " ".join((s.description or "").split()),
        "c": [float(x) for x in s.cuts],
        "t": s.seconds or FALLBACK_SECONDS,
        "r": sorted(refs),
        "a": ratio,
        "m": model,
        "p": list(carry_fps),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


@dataclass
class _SegRefs:
    """一段视频要带的参考图（裁参考位之前）。"""

    refs: list[str]  # 人物（服装 / 主形象）→ 服装配的脸 → 场景道具，去重
    label_of: dict[str, tuple[str, str]]  # url → (类别, 名字)
    people: set[str]  # 人物图的 url（参考位不够也必保）
    id_refs: list[tuple[str, str]]  # 一致性校验用：(角色名, url)，脸优先，最多 3 张


def _segment_refs(
    s: Any,
    resolved: dict[str, Any],
    images: dict[str, dict[str, str]],
    owner_of: dict[str, str],
    scene_names: set[str],
) -> _SegRefs:
    """一段要带哪些参考图。渲染、报价（花钱之前说清参考位够不够）、补查人物一致共用这一套。

    人物 / 服装优先（脸最要紧），再场景道具 —— 接口若截断参考图数，先丢的是最不影响一致性的。
    服装全身图里脸只占一小块：同时带上这个角色的主形象当脸的参考（2026-09-23 审查：之前只传
    服装图，「参考图传了照样漂」的主因之一）。"""
    people: list[str] = []
    faces: list[str] = []  # 服装对应角色的主形象（脸）：可选参考，位子够才带
    others: list[str] = []
    label_of: dict[str, tuple[str, str]] = {}
    id_refs: list[tuple[str, str]] = []
    for r in s.refs():
        m = resolved.get(r)
        if not m:
            continue
        (people if m.person else others).append(m.url)
        kind = str((images.get(m.key) or {}).get("kind") or "")
        if not kind:
            kind = "角色" if m.person else ("场景" if r in scene_names else "道具")
        key = m.key.removesuffix("·三视图")
        label_of.setdefault(m.url, (kind, key))
        if m.person and kind in ("角色", "服装"):
            id_refs.append((key, m.url))
        if m.person and kind == "服装":
            who = owner_of.get(key, "")
            fu = str((images.get(who) or {}).get("url") or "") if who else ""
            if fu and fu != m.url:
                faces.append(fu)
                label_of.setdefault(fu, ("角色", who))
    refs = list(dict.fromkeys(people))
    face_refs = [u for u in dict.fromkeys(faces) if u not in refs]
    refs += face_refs
    refs += [u for u in dict.fromkeys(others) if u not in refs]
    # 一致性校验优先拿脸比（主形象比全身服装图清楚得多）
    id_refs = list(dict.fromkeys([(label_of[u][1], u) for u in face_refs] + id_refs))[:3]
    return _SegRefs(refs=refs, label_of=label_of, people=set(people), id_refs=id_refs)


def _trim_refs(seg: _SegRefs, n_videos: int, max_refs: int) -> tuple[list[str], list[str]]:
    """参考位上限（seedance 9 个，含参考视频；媒体层双送参考图时会自动只送一份）：人物必保，
    位子不够先省道具、再省场景，最后才省脸。人物本身就超位的在引用门已经拦下。
    返回 (留下的参考图, 省掉的「类别「名字」」)。"""
    if not max_refs or len(seg.refs) + n_videos <= max_refs:
        return list(seg.refs), []
    budget = max(0, max_refs - n_videos)
    keep = [u for u in seg.refs if u in seg.people]
    rest = [u for u in seg.refs if u not in seg.people]
    rank = {"角色": 0, "场景": 1}
    rest.sort(key=lambda u: rank.get(seg.label_of.get(u, ("", ""))[0], 2))
    room = max(0, budget - len(keep))
    gone = [
        f"{seg.label_of[u][0]}「{seg.label_of[u][1]}」" for u in rest[room:] if u in seg.label_of
    ]
    return keep + rest[:room], gone


def _is_unchecked(note: str) -> bool:
    """这条质检备注是不是「没查成」（字幕 / 人物一致）。"""
    return _SUB_UNCHECKED in note or _ID_UNCHECKED in note


def _only_unchecked(tags: dict[str, Any]) -> bool:
    """这段没过质检门，只是因为「没查成」（字幕 / 人物一致：视觉服务不通、没本地副本），
    别的门都过了。"""
    blocking = [n for n in (tags.get("notes") or []) if _blocking_note(str(n))]
    return bool(blocking) and all(_is_unchecked(str(n)) for n in blocking)


def _no_text_name(name: str) -> str:
    """成片文件名 → 无字版的名字：第01集.mp4 → 第01集_无字.mp4。"""
    p = Path(name)
    return f"{p.stem}_无字{p.suffix or '.mp4'}"


def _card_context(script: str, c: Card, width: int = 40) -> str:
    """字幕卡在剧本里前后各 width 个字（给模型判断它该放哪个镜头）。"""
    s = script or ""
    i = s.find(c.marker)
    if i < 0:
        i = s.find(c.text)
    if i < 0:
        return "（剧本里没找到原位置）"
    before = s[max(0, i - width): i]
    after = s[i + len(c.marker): i + len(c.marker) + width]
    return " ".join(f"{before} ▲ {after}".split())


def _json_rows(text: str, key: str) -> list[dict[str, Any]]:
    """模型回的 JSON（{key: [...]} 或直接 [...]，可能包着 ```json 围栏）→ 行。解析不了返回空。"""
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    for cand in (s, s[s.find("{"): s.rfind("}") + 1], s[s.find("["): s.rfind("]") + 1]):
        try:
            data = json.loads(cand)
        except (ValueError, TypeError):
            continue
        rows = data.get(key) if isinstance(data, dict) else data
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
    return []


def _merge_by_number(first: list[Any], extra: list[Any]) -> list[Any]:
    """补写的段按镜号插回原来的位置（按每段覆盖的最小镜号排，同号保持原顺序）。"""

    def key(item: tuple[int, Any]) -> tuple[int, int]:
        i, s = item
        nums = covered_numbers(s.video_name)
        return (min(nums) if nums else 10**9, i)

    return [s for _, s in sorted(enumerate([*first, *extra]), key=key)]


def _length_hint(eps: list[Any], fmt: EpisodeFormat) -> str:
    """分镜比一集的规格长，或者光台词就超出规格（分镜被压快了）：按实际需要建议放宽集长
    （只有用户能改，/length）。集长跟剧本走（/length auto）时不用建议。"""
    if fmt.follow_script:
        return ""
    _, hi = fmt.duration_range
    over: list[tuple[str, float]] = []  # (说明, 这一集该有多长)
    for e in eps:
        blocks, secs = scene_blocks(e.desc, e.index)
        n = sum(b.count for b in blocks)
        total = sum(secs.values())
        need, _ = dialogue_timing(e)
        if n and len(secs) >= 0.8 * n and total > hi:
            over.append((f"{e.title} 约 {total / 60:.1f} 分钟", max(total, need / 0.75)))
        elif need / DIALOGUE_SHARE_MAX > hi:
            # 分镜被压进了规格、台词念不完：按台词占一集四分之三估该多长
            over.append((f"{e.title} 光念台词就要约 {need / 60:.1f} 分钟", need / 0.75))
    if not over:
        return ""
    minutes = math.ceil(max(t for _, t in over) / 60)
    shown = "；".join(s for s, _ in over)
    return (
        f"\n\n📏 {shown}，比「一集 {fmt.minutes:g} 分钟」长，多半是剧本本身就长。台词要全保留的话，"
        f"请用户用 /length {minutes} 放宽这个项目的集长，或 /length auto 让集长跟剧本走"
        "（只有用户能改）；不要反复重拆，也不要删台词或把镜头压短。"
    )


def _format_note(problems: list[str], fmt: EpisodeFormat) -> str:
    """一集规格（4 分钟 / 开场 15 秒高潮点）的检查结果，给人看。"""
    if not problems:
        return f"\n\n✓ 规格：{fmt.brief()} —— 达标"
    return (
        f"\n\n⚠ 规格检查未过（{fmt.brief()}；已让模型改过一次仍未达标，"
        "可带 note 重跑或人工调）：\n  " + "\n  ".join(problems)
    )


_FILTERED = ("content_filter", "content-filter", "safety", "sensitive", "blocked")
# 被内容安全过滤拦下后，自动重试时追加在最后一条用户消息后面的提醒
_SOFTEN = (
    "\n\n【措辞】上一次输出被模型服务商的内容安全过滤拦下了。这次用克制的镜头语言："
    "打斗和伤害只写动作与结果，不写伤口、血腥和痛苦的细节；涉及孩子只写日常的行为和表情；"
    "亲密的戏只写情绪和距离。剧情、台词和镜头数量都不变。"
)
_SOFT_NOTE = (
    "\n\n（内容安全过滤拦了一次，已自动用克制的措辞重写通过；剧情和台词没改，"
    "打斗、伤害类画面写得更含蓄了）"
)
_SOFT_TRIED = "（已经自动用克制的措辞重试过一次，仍被拦下）"
# 输出不是合法 JSON、本地也修不好：原样重发一次，在最后一条用户消息后面追加这段
_JSON_AGAIN = (
    "\n\n【格式】上一次输出的 JSON 不合法（{err}），整份作废了。这次只输出一份完整、合法的 JSON："
    "相邻的元素之间都要有逗号，最后一个元素后面不加逗号；字符串里需要引号时一律用「」，"
    "不要用英文双引号。内容和要求不变。"
)
# 一部剧一份资产库。2026-09-26 分两段生成的两份里：7 个主角各被描述成两张脸（痣换了边、年龄
# 不同、音色不同），沙僧的行者装前十集蓝灰、后十集赭黄，同一个人还有「观音」「观音菩萨」两条
_ONE_LIBRARY = (
    "一部剧只要一份资产库：把全剧各集的剧本 id 一次传给 drama_assets（失败了原样再调一次）。"
    "不要拆成几段分别生成 —— 各段会把同一个角色描述成不同的脸、同一套戏服设计成不同的样子；"
    "也不要读出资产库 JSON 手工合并"
)


def _fmt_eps(nums: Any) -> str:
    """集号 → 「1–10、12」。"""
    return format_ranges(number_ranges(sorted(set(nums))), limit=8)


def _is_filtered(resp: Any) -> bool:
    return str(getattr(resp, "finish_reason", "") or "").strip().lower() in _FILTERED


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
    if fr in _FILTERED:
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


def _label_norm(s: str) -> str:
    """比对前去掉方括号、「镜」字和空白："[第1集-2场] 5-9"、"第1集-2场 镜5-9"、"镜5-9" 都能对上。"""
    return re.sub(r"[\[\]【】镜\s]", "", str(s or ""))


def _label_hit(key: tuple[str, str], wants: list[str] | str | None) -> bool:
    """(场次, 镜头范围) 是否被 redo / accept 列表点到：写全称、只写场次或只写镜头都认。

    工具说明教的写法是「第1集-2场 镜5-9」「镜5-9」，之前按原样子串比对不上（场次带方括号、
    镜头不带「镜」），accept 写了等于没写、redo 静默无效；模型传成字符串时按字符迭代，
    "5" 命中一大片（2026-09-24 审查）。"""
    if isinstance(wants, str):
        wants = [wants]
    scene, name = _label_norm(key[0]), _label_norm(key[1])
    for want in wants or []:
        w = _label_norm(want)
        # 「第1集」= 这一集的所有段（场次写成「第1集-2场」，以「第1集-」开头）
        if w and (
            w == name or w == scene or w == scene + name
            or (w.endswith("集") and scene.startswith(w + "-"))
        ):
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
    held: list[str] | None = None,
) -> str:
    """音色锁定汇总：锁了几段、锚点沿用/新定/过期重定、缺音色卡的角色、没带上锚点的段。

    held：人定（定音）的锚点链接过期、没能重新托管的角色 —— 这一集用的是临时锚点，没替换人定的。"""
    held = held or []
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
    def _where(a: Anchor) -> str:
        return f"{a.character} ← {a.scene} 镜{a.clip}" + ("·独白段" if a.solo else "")

    saved = {n: a for n, a in born.items() if n not in held}
    if saved:
        parts.append(f"；本次新定 {len(saved)} 人：{'、'.join(_where(a) for a in saved.values())}")
    replaced = [n for n in plan.stale if n in born and n not in held]
    kept = [n for n in plan.stale if n not in born and n not in held]
    if replaced:
        parts.append(f"；{len(replaced)} 人的旧锚点链接过期已接替（{', '.join(replaced)}）")
    if held:
        temp = [_where(born[n]) for n in held if n in born]
        parts.append(
            f"\n  ⚠ 你定音的 {'、'.join(held)} 锚点链接过期了、没能重新托管：这一集"
            + (f"临时用本集的独白段当锚点（{'；'.join(temp)}），" if temp else "只锁了音色卡文字，")
            + "你定的锚点没动。要跨集一致：配好能传视频的托管（本地副本会自动重新上传），"
            "或重新定音（drama_voice_casting）"
        )
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
        # 没查成的（视觉模型没拿到图 / 没给出结论）单列：之前按通过算进「一致」，包里、报告里
        # 都写一致，其实从没比过（2026-09-29 审查 1.2）
        id_unchecked = [e for e in ident if e["identity"].get("unchecked")]
        id_graded = [e for e in ident if not e["identity"].get("unchecked")]
        ok = [e for e in id_graded if e["identity"].get("pass")]
        bad = [e for e in id_graded if not e["identity"].get("pass")]
        lines.append(f"；人物一致性：{len(ok)} 张与主形象一致")
        if id_unchecked:
            lines.append(
                f"，{len(id_unchecked)} 张没查成（按一致放行了，请自己看一眼）："
                + "、".join(e["name"] for e in id_unchecked[:6])
                + (" 等" if len(id_unchecked) > 6 else "")
                + f"（{id_unchecked[0]['identity'].get('note') or '没给出结论'}）"
            )
        if bad:
            lines.append(f"，{len(bad)} 张仍不像同一个人：")
            for e in bad:
                why = "；".join(e["identity"].get("issues") or [])[:80]
                lines.append(f"\n  ⚠ {e['name']}（{e['identity'].get('score', -1)}/10）{why}")
    unchecked = [e for e in checked if e.get("unchecked")]
    graded = [e for e in checked if not e.get("unchecked")]
    first = [e for e in graded if e.get("pass") and e.get("attempts", 1) == 1]
    retried_ok = [e for e in graded if e.get("pass") and e.get("attempts", 1) > 1]
    failed = [e for e in graded if not e.get("pass")]
    lines.append(
        f"；校验：一次通过 {len(first)} · 重生成后通过 {len(retried_ok)} · 仍不达标 {len(failed)}"
        + (f" · 没查成 {len(unchecked)}" if unchecked else "")
    )
    for e in failed:
        issues = "；".join(e.get("issues") or [])[:120]
        lines.append(f"\n  ⚠ {e['name']}（{e.get('score', -1)} 分）{issues}")
    if unchecked:
        lines.append(
            f"\n  ⚠ 没查成 {len(unchecked)} 张（按通过放行了，请自己看一眼 images/）："
            + "、".join(e["name"] for e in unchecked[:6])
            + (" 等" if len(unchecked) > 6 else "")
        )
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

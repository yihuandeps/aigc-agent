"""媒体生成 functions —— 模型自己挑模型。

暴露三个 function：
  list_media_models  看有哪些模型、各自擅长什么 → 模型据此自由选型
  gen_image          生图
  gen_video          生视频（异步，内部轮询）

两条设计：
  1. **产物一律落 Asset**，和文本 function 同一套底座。血缘、版本、
     成本归因、单步重跑全部照旧 —— 这是 M21 的硬约束。
  2. **不认识的模型 id 直接报错，不静默替换。** 悄悄换成别的模型会
     让人对着结果百思不得其解。
"""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from pathlib import Path
from typing import Any

from ...harness.events.bus import EventType
from ...harness.model.media import MediaGateway, MediaKind
from ...harness.model.quota import is_quota_error
from ...harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
    ToolSpec,
)
from ..assets.store import AssetStore, AssetType
from ..generators.catalog import MediaCatalog
from ..media.naming import safe_name, unique_path
from ..media.no_text import no_text_prompt

_LOCAL_NAME_DESC = (
    "本地副本的文件名（不含扩展名），产物目录里按它排序；留空用资产 id。"
    "批量生成时按顺序号命名，后期剪辑才排得出先后"
)
# 人审挂起的环节名：换视频模型必须人拍板（2026-09-20 用户定的规则），总线上按它认决策
VIDEO_SWITCH_STAGE = "视频模型切换"
# 生图同理（2026-09-22 用户定）：模型**可以主动提议换**生图模型，但不能自己换了算。
# 起因是短剧里某个角色的主形象被服务商的内容护栏拒了 —— 换一家模型是合理的工程选择，
# 可画风、质感会跟着变，而且之后所有图都用新模型，这种事只能人拍板。
IMAGE_SWITCH_STAGE = "生图模型切换"

# 模型锁的参数表：锁字段 / 持久化回调字段 / 挂起环节名 / 模态名 / 产物名。
# 视频和生图共用同一套门，只有这几个字不一样 —— 加第三种模态（音频）照抄一行即可。
_LOCK_SPECS: dict[MediaKind, tuple[str, str, str, str, str]] = {
    MediaKind.VIDEO: ("video_lock", "on_video_lock", VIDEO_SWITCH_STAGE, "视频", "镜头"),
    MediaKind.IMAGE: ("image_lock", "on_image_lock", IMAGE_SWITCH_STAGE, "生图", "图"),
}

# 人采纳换模型时附言里带这些词 = 只这一次：这一轮放行新模型，锁不动。
# 2026-09-23 审查：之前一采纳就写进会话快照，短剧链也跟着换 —— 想「先用快档试一镜」
# 或「这一张被护栏拒了换一家出」，之后所有生成都被带走了
_ONCE = re.compile(r"(只|仅)[换用]?(这|此)?一?(次|回|镜|张|段)|就(这|此)一?次|临时|暂时|once", re.I)
# 「之后都换」「全部用它」之类是要长期换，哪怕句子里也出现了「只…一次」
_FOR_GOOD = re.compile(r"(之后|以后|后面|往后)(也)?都|全都|全部|都换")


def _adopt_once(reason: str) -> bool:
    s = reason or ""
    return bool(_ONCE.search(s)) and not _FOR_GOOD.search(s)


def _not_public(refs: list[str]) -> list[str]:
    """参考素材里不是公网链接的那些。

    生成接口只认 http/https（seedance 原话：Only http/https URL or asset:// private
    asset URL），本地路径和 base64 一律拒。不在这儿拦就得等接口返回一句英文报错，
    一段视频的等待时间全白搭 —— 而真正该做的是先 host_file 换成链接。
    """
    return [r for r in refs if r and not str(r).startswith(("http://", "https://", "asset://"))]


def _local_ref_error(bad: list[str]) -> str:
    shown = "、".join(str(b)[:60] for b in bad[:3])
    more = f" 等 {len(bad)} 个" if len(bad) > 3 else ""
    return (
        f"参考素材不是公网链接（{shown}{more}），没有提交 —— "
        "生成接口只收 http/https，本地路径和 base64 都会被拒。\n"
        '先把本地文件换成链接：host_file(path="...")；'
        '短剧里用 drama_use_local_ref(name="角色名", path="...") 直接进参考图包。'
        "报「没有配置素材托管」就先看 hosting_status。"
    )


class MediaFunctions:
    """媒体领域的 ToolProvider。加新 modality（TTS/音乐）在这里加 function。"""

    name = "media"
    namespaced = False
    disclosure = "full"

    def __init__(
        self,
        gateway: MediaGateway,
        catalog: MediaCatalog,
        store: AssetStore,
        prefs: Any = None,
    ) -> None:
        self.gateway = gateway
        self.catalog = catalog
        self.store = store
        # 产物目录偏好：生成完把本地副本落到用户的产物目录，按 local_name 命名
        self.prefs = prefs
        # 参考图门（2026-09-20 用户定的规则）：短剧镜头没带参考图不许生成。模型绕过
        # gen_videos 无参考生成，成片后半段人物全变脸。None = 不拦（脚本 / 测试）
        self.ref_guard = None
        # 模型锁：留空一律用锁定的模型，换模型要先问人
        # （视频 2026-09-20 定、生图 2026-09-22 定，两边共用 _model_gate）。
        # on_*_lock 把换锁写回会话快照；_pending_switch 按环节名记着正在等人拍板的那个
        self.video_lock = ""
        self.on_video_lock = None
        # 项目画幅（/ratio）：gen_video 没传 aspect_ratio 时用它；空 = 交给模型默认
        self.default_aspect = ""
        self.image_lock = ""
        self.on_image_lock = None
        self._pending_switch: dict[str, str] = {}
        # 人说了「只这一次」的：环节名 → 这一轮放行的模型；这一轮真正结束（LOOP_END）就收回
        self._once: dict[str, str] = {}
        self._specs: dict[str, ToolSpec] = {}
        self._build()

    def _build(self) -> None:
        self._specs["list_media_models"] = ToolSpec(
            name="list_media_models",
            summary="列出可用的图像/视频生成模型及各自擅长什么",
            permission=PermissionLevel.READ,
            description=(
                "生成图片或视频**之前先看这个**，按任务性质挑合适的模型："
                "出成片挑 quality 档，批量试方向挑 fast 档省钱。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["image", "video", "all"]},
                },
            },
        )

        self._specs["gen_image"] = ToolSpec(
            name="gen_image",
            summary="按提示词生成图片，产物自动存为资产",
            permission=PermissionLevel.COMPUTE,
            cost_kind="image",
            timeout=1500,  # 生成 600s + 排队 600s + 余量
            description=(
                "生成**一张**图片。model 留空则按 prefer 自动选。要出多个候选就把 n 调大 —— "
                "关键节点给人多个选择，别只给一个。\n"
                "⚠️ **要生成两张以上就改用 gen_images 批量并发**：逐张调是串行的，几十张会等很久。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "画面描述，越具体越好"},
                    "model": {
                        "type": "string",
                        "description": "留空 = 本会话锁定的生图模型。传一个不同的模型会"
                        "**暂停问用户**，同意后才换 —— 某个模型拒绝出图时可以主动提议换一家，"
                        "但不要自己换了算",
                    },
                    "prefer": {
                        "type": "string",
                        "enum": ["quality", "balanced", "fast"],
                        "description": "未指定 model 时的选型倾向，默认 balanced",
                    },
                    "aspect_ratio": {"type": "string", "description": "如 16:9 / 9:16 / 1:1"},
                    "n": {"type": "integer", "description": "生成几个候选，默认 1，最多 4"},
                    "image": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "参考图 url。**保持人物/物体一致性的唯一手段** —— "
                        "同一个角色换装、换场景时把上一张图传进来",
                    },
                    "summary": {"type": "string", "description": "这张图是干什么用的，一句话"},
                    "parent_id": {
                        "type": "string",
                        "description": "若基于某份脚本/分镜生成，填它的 id 以建立血缘",
                    },
                    "local_name": {"type": "string", "description": _LOCAL_NAME_DESC},
                },
                "required": ["prompt"],
            },
        )

        self._specs["gen_video"] = ToolSpec(
            name="gen_video",
            summary="按提示词生成视频（异步，可能要等几分钟）",
            permission=PermissionLevel.COMPUTE,
            cost_kind="video",
            timeout=1500,  # 生成 600s + 排队 600s + 余量
            description=(
                "生成**一段**视频。**很慢也很贵**，调之前先确认脚本/分镜已经定稿、"
                "最好已经过人审。可以先用 fast 档验证方向，定了再用 quality 档出成片。\n"
                "⚠️ **要生成两段以上一律改用 gen_videos 批量并发**：一次工具调用就是一个迭代，"
                "逐段调必然串行 —— 实测 19 段串行 2.3 小时，并发约 20 分钟。"
                "整集短剧优先用 drama_render_shots，它自带依赖排队与失败复用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "model": {
                        "type": "string",
                        "description": "留空 = 本会话锁定的视频模型（用户指定的）。传一个不同的模型"
                        "会**暂停问用户**，用户同意后才换；不要为了省钱或绕过失败自己换模型",
                    },
                    "prefer": {"type": "string", "enum": ["quality", "balanced", "fast"]},
                    "aspect_ratio": {
                        "type": "string",
                        "description": "画幅，如 16:9 / 9:16；留空用项目设置（用户用 /ratio 选的）",
                    },
                    "duration": {"type": "integer", "description": "秒，受模型上限约束"},
                    "resolution": {
                        "type": "string",
                        "enum": ["480p", "720p", "1080p", "4k"],
                        # 480p 是 2026-09-22 对着 seedance-2.0-fast 实测过的：
                        # 出 496×864，最省钱，验证分镜方向用它
                        "description": "验证方向用 480p 最省（seedance-2.0-fast 实测支持，"
                        "各模型支持哪些看 list_media_models），成片再上 720p",
                    },
                    "image_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "参考图/首帧 url。人物镜头必须带角色参考图；"
                        "参考素材总数受模型上限约束（seedance 9 个，含参考视频），超了直接拒绝",
                    },
                    "allow_no_refs": {
                        "type": "boolean",
                        "description": (
                            "没有参考图也允许生成（默认 false：短剧镜头不带参考图会被拦下）。"
                            "只对没有任何人物的空镜有效 —— 提示词里出现资产库里的角色/服装时"
                            "传了也会被拦"
                        ),
                    },
                    "video_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "参考视频 url（seedance 最多 3 个），提示词里用 @视频1 "
                        "引用：取其音色/动作/运镜；短剧的音色锚点走这里",
                    },
                    "audio_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "参考音频 url（最多 3 个、合计 ≤15s），提示词里用 "
                        "@音频1 引用",
                    },
                    "summary": {"type": "string"},
                    "parent_id": {"type": "string"},
                    "local_name": {"type": "string", "description": _LOCAL_NAME_DESC},
                    "allow_text": {
                        "type": "boolean",
                        "description": "允许画面出现文字。默认 false：提示词会带最高优先级的"
                        "禁字幕/禁文字约束（成片字幕由后期压，不能让模型烧进画面）。"
                        "传 true 会当场问用户，用户不同意就不生成；"
                        "上屏文字请用 overlay_text 后期叠",
                    },
                },
                "required": ["prompt"],
            },
        )

        self._specs["gen_videos"] = ToolSpec(
            name="gen_videos",
            summary="一次提交多段视频并发生成（要生成多段时用它，不要逐段调 gen_video）",
            permission=PermissionLevel.COMPUTE,
            cost_kind="video",
            cost_units_arg="jobs",  # 几段就记几次，预算护栏按真实数量算
            timeout=7200,
            description=(
                "**要生成两段以上视频时一律用这个，不要一段一段调 gen_video。**"
                "逐段调会一段跑完才发下一段（一次工具调用就是一个迭代），"
                "实测 19 段串行跑了 2.3 小时；并发跑同样的量约 20 分钟。\n"
                "jobs 是任务数组，每项至少有 prompt，可覆盖 image / duration / summary / "
                "local_name 等；公共参数（模型、比例、分辨率）写在外层。"
                "返回按顺序的资产 id，可直接喂给 compose_video。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "jobs": {
                        "type": "array",
                        "description": "每段一个对象：{prompt, image?, duration?, summary?, "
                        "local_name?, video_urls?, audio_urls?}",
                        "items": {"type": "object"},
                    },
                    "model": {
                        "type": "string",
                        "description": "公共模型。留空 = 本会话锁定的视频模型；"
                        "传不同的模型会暂停问用户",
                    },
                    "prefer": {"type": "string", "enum": ["quality", "balanced", "fast"]},
                    "aspect_ratio": {
                        "type": "string",
                        "description": "公共画幅，如 16:9 / 9:16；留空用项目设置（/ratio）",
                    },
                    "resolution": {
                        "type": "string",
                        "enum": ["480p", "720p", "1080p", "4k"],
                        # 480p 是 2026-09-22 对着 seedance-2.0-fast 实测过的：
                        # 出 496×864，最省钱，验证分镜方向用它
                        "description": "验证方向用 480p 最省（seedance-2.0-fast 实测支持，"
                        "各模型支持哪些看 list_media_models），成片再上 720p",
                    },
                    "duration": {"type": "integer", "description": "公共时长秒，可被单项覆盖"},
                    "parent_id": {"type": "string"},
                    "allow_text": {
                        "type": "boolean",
                        "description": "允许画面出现文字（默认 false）。传 true 会当场问用户；"
                        "上屏文字请用 overlay_text 后期叠",
                    },
                    "allow_no_refs": {
                        "type": "boolean",
                        "description": (
                            "没有参考图也允许生成（默认 false，短剧镜头不带参考图会被拦）；"
                            "只对没有人物的空镜有效，点了角色名的照样拦"
                        ),
                    },
                },
                "required": ["jobs"],
            },
            max_result_chars=12_000,
        )

        self._specs["gen_images"] = ToolSpec(
            name="gen_images",
            summary="一次提交多张图并发生成（要生成多张时用它，不要逐张调 gen_image）",
            permission=PermissionLevel.COMPUTE,
            cost_kind="image",
            cost_units_arg="jobs",
            timeout=3600,
            description=(
                "**要生成两张以上图片时一律用这个，不要一张一张调 gen_image。**"
                "逐张调会串行，几十张要等很久。jobs 每项至少有 prompt，"
                "可覆盖 image（参考图）/ summary / local_name / n；公共参数写在外层。"
                "返回按顺序的资产 id。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "jobs": {
                        "type": "array",
                        "description": "每张一个对象：{prompt, image?, summary?, local_name?, n?}",
                        "items": {"type": "object"},
                    },
                    "model": {
                        "type": "string",
                        "description": "公共模型，留空 = 本会话锁定的生图模型；"
                        "传不同的模型会暂停问用户（整批都不发）",
                    },
                    "prefer": {"type": "string", "enum": ["quality", "balanced", "fast"]},
                    "aspect_ratio": {"type": "string"},
                    "parent_id": {"type": "string"},
                },
                "required": ["jobs"],
            },
            max_result_chars=12_000,
        )

        self._specs["media_tasks"] = ToolSpec(
            name="media_tasks",
            summary="列出提交了但没取回结果的图/视频任务（多半已生成、已计费）",
            permission=PermissionLevel.READ,
            description=(
                "生成中途被 /stop、工具超时、轮询放弃时，服务端的任务照跑、照扣费，"
                "task_id 记在台账里。**重做之前先看这里**：能取回就用 media_recover 取回，"
                "不要原样重新提交（那是再付一份钱）。"
            ),
            parameters={
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "最多列几个，默认 20"}},
            },
        )
        self._specs["media_recover"] = ToolSpec(
            name="media_recover",
            summary="按 task_id 取回之前没拿到结果的生成任务，登记成资产（不重新付费）",
            permission=PermissionLevel.WRITE,
            timeout=1500,
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "summary": {"type": "string"},
                    "local_name": {"type": "string", "description": _LOCAL_NAME_DESC},
                    "parent_id": {"type": "string"},
                },
                "required": ["task_id"],
            },
        )

    # ---------- ToolProvider 协议 ----------

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        n_i, n_v = len(self.catalog.image), len(self.catalog.video)
        return ProviderHealth(ok=True, detail=f"图 {n_i} 个 / 视频 {n_v} 个模型")

    def permission_for(
        self, tool: str, args: dict[str, Any]
    ) -> tuple[PermissionLevel, str] | None:
        """注册表的提权钩子：要放开「画面不许有字」时，这次调用按 L-external 过闸门、当场问人。

        画面禁字是用户 2026-09-18 定的最高优先级规则，allow_text 却是模型自己就能填的参数
        （2026-09-26 审查）。放开它得人点头：闸门在终端里问（/auto 也问、默认不放行），
        模型没法替人答。批量版单项里写的 allow_text 同样算。
        """
        if tool not in ("gen_video", "gen_videos"):
            return None
        jobs = args.get("jobs") if tool == "gen_videos" else None
        wanted = bool(args.get("allow_text")) or any(
            isinstance(j, dict) and bool(j.get("allow_text")) for j in (jobs or [])
        )
        if not wanted:
            return None
        return (
            PermissionLevel.EXTERNAL,
            "要放开「视频画面不许有字幕 / 文字」这条你定的最高优先级规则（allow_text=true），"
            "这次生成的画面允许出现文字 —— 需要你确认",
        )

    def estimate_cost(self, tool: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """这次调用要花多少（给闸门事前拦用）：{"units", "seconds", "money"}。

        模型没指定就按锁定的算；目录没填单价 money 就是 None（只按段数 / 秒数拦）。
        """
        def model_for(kind: MediaKind, want: str) -> str:
            lock = self.video_lock if kind is MediaKind.VIDEO else self.image_lock
            chosen = want or lock
            if not chosen:
                chosen, _ = self.catalog.choose(kind, "", str(args.get("prefer") or "balanced"))
            return chosen or ""

        def add(total: dict[str, Any], money: float | None) -> None:
            if money is None:
                total["unpriced"] = True
            else:
                total["money"] = round((total.get("money") or 0.0) + money, 4)

        total: dict[str, Any] = {"units": 0, "seconds": 0.0}
        if tool in ("gen_video", "gen_videos"):
            jobs = (
                [j for j in (args.get("jobs") or []) if isinstance(j, dict)]
                if tool == "gen_videos"
                else [args]
            )
            for job in jobs:
                p = {**args, **job}
                model = model_for(MediaKind.VIDEO, str(p.get("model") or ""))
                total["units"] += 1
                total["seconds"] += self.catalog.seconds_of(model, p)
                add(total, self.catalog.price_of(MediaKind.VIDEO, model, p))
        elif tool in ("gen_image", "gen_images"):
            jobs = (
                [j for j in (args.get("jobs") or []) if isinstance(j, dict)]
                if tool == "gen_images"
                else [args]
            )
            for job in jobs:
                p = {**args, **job}
                n = max(1, min(int(p.get("n") or 1), 4))
                model = model_for(MediaKind.IMAGE, str(p.get("model") or ""))
                total["units"] += n
                add(total, self.catalog.price_of(MediaKind.IMAGE, model, p, n=n))
        else:
            return None
        if total.get("unpriced"):
            total.pop("money", None)  # 有一项算不出钱，金额就不装作知道
        total.pop("unpriced", None)
        return total if total["units"] else None

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        fn = getattr(self, f"_fn_{tool}", None)
        try:
            if fn is None:
                raise TypeError(f"没有工具 {tool}")
            inspect.signature(fn).bind(**args)
        except TypeError as e:
            # 参数对不上（模型多传 / 漏传了参数）：根本没提交，闸门按整批记的账要退回来
            # （2026-09-24 审查：之前照样计次计秒）
            result = ToolResult(ok=False, error=f"参数不对：{e}", meta={"charged": False})
        else:
            try:
                result = await fn(**args)
            except Exception as e:  # noqa: BLE001
                # 执行中途出错：可能已经提交、付了钱，不退额度
                result = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result

    # ---------- 实现 ----------

    # ---------- 批量并发 ----------

    async def _run_batch(
        self, kind: MediaKind, jobs: list[dict[str, Any]], shared: dict[str, Any]
    ) -> ToolResult:
        """一次提交多个生成任务，按配置的并发上限同时跑。

        2026-09-19：用户发现 19 段视频跑了 2.3 小时、零重叠。根因是模型一次只调一个
        gen_video，一次调用就是一个迭代，必然串行——并发逻辑当时只存在于
        drama_render_shots 里，手搓链路享受不到。所以把并发下沉到这一层。
        """
        if not jobs:
            return ToolResult(ok=False, error="jobs 是空的", meta={"charged": False})
        if len(jobs) > 40:
            return ToolResult(
                ok=False, error=f"一次最多 40 个，收到 {len(jobs)} 个；分批调用",
                meta={"charged": False},
            )
        label = "视频" if kind is MediaKind.VIDEO else "图片"
        fn = self._fn_gen_video if kind is MediaKind.VIDEO else self._fn_gen_image
        limit = self.catalog.max_concurrency("video" if kind is MediaKind.VIDEO else "image")
        sem = asyncio.Semaphore(limit) if limit > 0 else None
        # 余额 / 额度用完（2026-09-29 审查 1.1）：第一个撞上之后，还在排队的不再发 —— 之前一批
        # 8 段并发全部 402，后面排队的照样一个个去撞
        quota: dict[str, str] = {}

        async def run(args: dict[str, Any]) -> ToolResult:
            if quota:
                return ToolResult(
                    ok=False, error="没发：前面已经撞上余额 / 额度用完",
                    meta={"charged": False, "quota": True},
                )
            r = await fn(**args)
            if not r.ok and (r.meta or {}).get("quota") and not quota:
                quota["why"] = (r.error or "")[:200]
            return r

        async def one(i: int, job: dict[str, Any]) -> tuple[int, ToolResult]:
            args = {**shared, **{k: v for k, v in job.items() if v is not None}}
            args.setdefault("summary", f"{label} {i + 1}")
            if sem is None:
                return i, await run(args)
            async with sem:
                return i, await run(args)

        started = time.perf_counter()
        results = await asyncio.gather(
            *(one(i, j) for i, j in enumerate(jobs)), return_exceptions=True
        )
        lines: list[str] = []
        ok_ids: list[str] = []
        failed = 0
        refund = {"refund_units": 0, "refund_seconds": 0.0}
        for item in results:
            if isinstance(item, BaseException):
                failed += 1
                lines.append(f"  ✗ {type(item).__name__}: {item}")
                continue
            i, r = item
            name = jobs[i].get("summary") or f"{label} {i + 1}"
            unpaid = (r.meta or {}).get("charged") is False
            if r.ok and r.asset_ref:
                ok_ids.append(r.asset_ref)
                tail = "（取回上次的任务，没重新付费）" if unpaid else ""
                lines.append(f"  ✓ {name} → {r.asset_ref}{tail}")
            else:
                failed += 1
                lines.append(f"  ✗ {name}：{(r.error or '未知原因')[:100]}")
            if unpaid:
                # 这一项没花钱（被拦 / 被拒收 / 取回了之前付过费的任务），闸门按整批记的账
                # 要退回这部分（2026-09-24 审查：取回的照样计次计秒，同一段算两次）
                job = {**shared, **jobs[i]}
                if kind is MediaKind.VIDEO:
                    refund["refund_units"] += 1
                    model = str(job.get("model") or self.video_lock or "")
                    refund["refund_seconds"] += self.catalog.seconds_of(model, job)
                else:
                    refund["refund_units"] += max(1, min(int(job.get("n") or 1), 4))
        dt = time.perf_counter() - started
        head = (
            f"批量生成{label} {len(jobs)} 个：成功 {len(ok_ids)}，失败 {failed}"
            f"（并发上限 {limit or '不限'}，耗时 {dt / 60:.1f} 分钟）"
        )
        if quota:
            # 原因写在最前面：后面一串 ✗ 都是同一个原因，别让模型挨个去查
            head = (
                f"⛔ 余额 / 额度用完了，这一批停在这里（{quota['why']}）。"
                "重试、换模型、拆小批都没用：告诉用户去充值（或等额度窗口恢复）；"
                "出好的（✓）都留着，恢复后只把 ✗ 的几项再发一次\n"
                + head
            )
        if not ok_ids:
            return ToolResult(ok=False, error=head + "\n" + "\n".join(lines), meta=refund)
        tail = f"\n\n按顺序的资产 id：{' '.join(ok_ids)}"
        return ToolResult(
            content=head + "\n" + "\n".join(lines) + tail, asset_ref=ok_ids[0], meta=refund
        )

    async def _fn_gen_images(
        self,
        jobs: list[dict[str, Any]],
        model: str = "",
        prefer: str = "balanced",
        aspect_ratio: str = "",
        parent_id: str = "",
    ) -> ToolResult:
        shared = {
            "model": model,
            "prefer": prefer,
            "aspect_ratio": aspect_ratio,
            "parent_id": parent_id,
        }
        # 生图模型锁：整批先过一遍门 —— 任何一项要换模型都先问人，一张都不发
        for job in jobs or []:
            if not isinstance(job, dict):
                continue
            want = str(job.get("model") or model or "")
            _, _, gate = self._image_model_gate(want, prefer, str(job.get("summary") or ""))
            if gate is not None:
                return gate
        if not model and self.image_lock:
            shared["model"] = self.image_lock
        result = await self._run_batch(MediaKind.IMAGE, jobs, shared)
        note = self._prefer_note(MediaKind.IMAGE, self.image_lock, prefer) if not model else ""
        if note and result.ok:
            result.content = f"{result.content}\n{note}"
        return result

    # ---------- 模型锁：换模型之前必须问用户 ----------
    #
    # 视频 2026-09-20 定、生图 2026-09-22 定，两条规则一模一样，所以共用一套门。
    # 各自的真实来由：
    #   视频 —— 模型补段时 model 留空 + prefer=quality，自动选型从用户指定的
    #           seedance-2.0 换成了 veo3.1-quality，12 段白渲。
    #   生图 —— 某个角色的主形象被服务商内容护栏拒了。换一家模型是对的，但画风会变，
    #           而且之后所有图都跟着换，得人点头。
    # 两边都不禁止模型**提议**换，只是提议要走人审，不能自己换了算。

    def set_video_lock(self, model: str) -> None:
        """锁定本会话的视频模型，并通知持久化（会话快照）。"""
        self._set_lock(MediaKind.VIDEO, model)

    def set_image_lock(self, model: str) -> None:
        """锁定本会话的生图模型，并通知持久化（会话快照）。"""
        self._set_lock(MediaKind.IMAGE, model)

    def _set_lock(self, kind: MediaKind, model: str) -> None:
        attr, cb, _, _, _ = _LOCK_SPECS[kind]
        setattr(self, attr, model)
        fn = getattr(self, cb, None)
        if fn is not None:
            try:
                fn(model)
            except Exception:  # noqa: BLE001
                pass

    def on_event(self, event: Any) -> None:
        """总线回调：人对「模型切换」拍板后，**采纳**才真正换锁；打回/退回不换。

        采纳时附言说「只这一次」（a 只这一次）：这一轮放行新模型，锁不动、不写快照；
        这一轮真正结束（LOOP_END，挂起等人审的不算）就收回。
        """
        etype = getattr(event, "type", None)
        data = getattr(event, "data", None) or {}
        if etype is EventType.LOOP_END:
            if data.get("stop_reason") != "awaiting_review":
                self._once.clear()
            return
        if etype is not EventType.CHECKPOINT_DECIDED:
            return
        node = data.get("node")
        for kind, (_, _, stage, _, _) in _LOCK_SPECS.items():
            if node != stage:
                continue
            target = self._pending_switch.pop(stage, "")
            if data.get("decision") != "adopt" or not target:
                continue
            if _adopt_once(str(data.get("reason") or "")):
                self._once[stage] = target
            else:
                self._set_lock(kind, target)

    def _video_model_gate(
        self, model: str, prefer: str, summary: str = ""
    ) -> tuple[str, str, ToolResult | None]:
        return self._model_gate(MediaKind.VIDEO, model, prefer, summary)

    def _image_model_gate(
        self, model: str, prefer: str, summary: str = ""
    ) -> tuple[str, str, ToolResult | None]:
        return self._model_gate(MediaKind.IMAGE, model, prefer, summary)

    def _model_gate(
        self, kind: MediaKind, model: str, prefer: str, summary: str = ""
    ) -> tuple[str, str, ToolResult | None]:
        """决定这次用哪个模型。返回 (模型, 首次锁定的说明, 要先返回的结果)。

        · 已锁定 + 留空      → 用锁定的，不再自动选型
        · 已锁定 + 要换      → 挂起问人（major，/auto 也停）；采纳后 on_event 换锁，模型再调一次即可
        · 未锁定             → 显式指定或自动选型，选中的就成为锁
        """
        attr, _, stage, label, thing = _LOCK_SPECS[kind]
        lock = str(getattr(self, attr, "") or "")
        if lock and model and model != lock and self._once.get(stage) == model:
            # 人说了「只这一次」：这一轮放行，锁不动
            return model, f"（用户只同意这一轮用 {model}；本会话{label}模型仍锁定为 {lock}）", None
        if lock and model and model != lock:
            chosen, why = self.catalog.choose(kind, model, prefer)
            if not chosen:
                return "", "", ToolResult(ok=False, error=why, meta={"charged": False})
            self._pending_switch[stage] = chosen
            question = (
                f"要把{label}模型从 {lock} 换成 {chosen} 吗？（本次生成：{summary or '未命名'}）\n"
                f"换模型会改变画质与风格。回复 a 同意换，之后的{thing}也都用新模型；"
                f"「a 只这一次」只在这一轮用 {chosen}，之后仍用 {lock}；"
                f"r 或 j 不换，继续用 {lock}。"
            )
            payload = {
                "question": question,
                "stage": stage,
                "target": stage,
                "assets": [],
                "major": True,  # /auto 模式也要停：换模型只能人定
                "model_from": lock,
                "model_to": chosen,
            }
            return "", "", ToolResult(
                content=(
                    f"{label}模型切换需要用户确认（{lock} → {chosen}），已暂停等待用户决定。"
                    f"用户同意后再调用一次即可（他说「只这一次」就只在这一轮生效）；"
                    f"不同意就继续用 {lock}。"
                ),
                suspend=True,
                suspend_payload=payload,
            )
        if lock and not model:
            return lock, self._prefer_note(kind, lock, prefer), None
        chosen, why = self.catalog.choose(kind, model, prefer)
        if not chosen:
            return "", "", ToolResult(ok=False, error=why, meta={"charged": False})
        if not lock:
            self._set_lock(kind, chosen)
            note = f"{label}模型已锁定为 {chosen}（{why}）：之后都用它，要换会先问用户。"
            return chosen, note, None
        return chosen, "", None

    def _prefer_note(self, kind: MediaKind, lock: str, prefer: str) -> str:
        """有锁时 prefer 不参与选型。之前悄悄忽略 ——「先用 fast 档验证方向」实际还是锁定的
        quality 模型在跑，钱照花（2026-09-23 审查）。现在说出来，并告诉模型怎么真换。"""
        if prefer not in ("quality", "fast"):
            return ""  # balanced 是默认值，分不出是不是有意传的
        spec = self.catalog.get(kind, lock)
        if spec is None or spec.tier == prefer:
            return ""
        # 优先推荐同一家的那一档（seedance-2.0 → seedance-2.0-fast），换家画风变得更多
        family = lock.split("-")[0]
        same = [
            m for m in self.catalog.models(kind) if m.tier == prefer and m.id.startswith(family)
        ]
        if same:
            alt = min(same, key=lambda m: m.cost).id
        else:
            alt, _ = self.catalog.choose(kind, "", prefer)
        label = _LOCK_SPECS[kind][3]
        how = (
            f"要按 {prefer} 档出，带 model={alt} 再调一次 —— 会先问用户（可以只同意这一次）"
            if alt and alt != lock
            else "要换模型得先问用户"
        )
        return (
            f"prefer={prefer} 没生效：本会话{label}模型锁定为 {lock}（{spec.tier} 档），"
            f"这次仍用它。{how}。"
        )

    async def _fn_gen_videos(
        self,
        jobs: list[dict[str, Any]],
        model: str = "",
        prefer: str = "balanced",
        aspect_ratio: str = "",
        resolution: str = "",
        duration: int | None = None,
        parent_id: str = "",
        allow_text: bool = False,
        allow_no_refs: bool = False,
    ) -> ToolResult:
        shared: dict[str, Any] = {
            "model": model,
            "prefer": prefer,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "parent_id": parent_id,
            "allow_text": allow_text,
            "allow_no_refs": allow_no_refs,
        }
        if duration:
            shared["duration"] = duration
        # 视频模型锁：整批先过一遍门 —— 任何一项要换模型都先问人，一段都不发
        for job in jobs or []:
            if not isinstance(job, dict):
                continue
            want = str(job.get("model") or model or "")
            _, _, gate = self._video_model_gate(want, prefer, str(job.get("summary") or ""))
            if gate is not None:
                return gate
        if not model and self.video_lock:
            shared["model"] = self.video_lock
        result = await self._run_batch(MediaKind.VIDEO, jobs, shared)
        note = self._prefer_note(MediaKind.VIDEO, self.video_lock, prefer) if not model else ""
        if note and result.ok:
            result.content = f"{result.content}\n{note}"
        return result

    async def _fn_list_media_models(self, kind: str = "all") -> ToolResult:
        parts = []
        if kind in ("image", "all"):
            parts.append("## 图像模型\n" + self.catalog.render(MediaKind.IMAGE))
        if kind in ("video", "all"):
            parts.append("## 视频模型\n" + self.catalog.render(MediaKind.VIDEO))
        return ToolResult(content="\n\n".join(parts) or "目录为空")

    async def _fn_gen_image(
        self,
        prompt: str,
        model: str = "",
        prefer: str = "balanced",
        aspect_ratio: str = "",
        n: int = 1,
        summary: str = "",
        parent_id: str = "",
        image: list[str] | None = None,
        local_name: str = "",
        tags: dict[str, Any] | None = None,
    ) -> ToolResult:
        bad = _not_public(list(image or []))
        if bad:
            return ToolResult(ok=False, error=_local_ref_error(bad), meta={"charged": False})
        # 生图模型锁：留空用锁定的；要换先挂起问人（2026-09-22 用户定的规则）
        chosen, lock_note, gate = self._image_model_gate(model, prefer, summary)
        if gate is not None:
            return gate
        result = await self._generate(
            MediaKind.IMAGE,
            prompt,
            chosen,
            prefer,
            summary,
            parent_id,
            AssetType.IMAGE,
            local_name=local_name,
            tags=tags,
            n=max(1, min(int(n or 1), 4)),
            aspect_ratio=aspect_ratio or None,
            # 参考图。实测 2026-09-12 接口收的字段是 image，不是 image_url/reference_images。
            # 对照实验：同一句"参考图里的物体放到沙滩上"，带参考图出苹果，不带出相机。
            image=(image or None),
        )
        if lock_note and result.ok:
            result.content = f"{result.content}\n{lock_note}"
        return result

    async def _fn_gen_video(
        self,
        prompt: str,
        model: str = "",
        prefer: str = "balanced",
        aspect_ratio: str = "",
        duration: int | None = None,
        resolution: str = "",
        image_urls: list[str] | None = None,
        image: list[str] | None = None,
        video_urls: list[str] | None = None,
        audio_urls: list[str] | None = None,
        summary: str = "",
        parent_id: str = "",
        local_name: str = "",
        allow_text: bool = False,
        allow_no_refs: bool = False,
        tags: dict[str, Any] | None = None,
    ) -> ToolResult:
        pics = list(image or image_urls or [])
        bad = _not_public(pics + list(video_urls or []) + list(audio_urls or []))
        if bad:
            return ToolResult(ok=False, error=_local_ref_error(bad), meta={"charged": False})
        # 参考图门（2026-09-20 用户定的规则）：短剧镜头没带参考图不许生成，在提交前拦，
        # 不能生成完再说"没引用成功"。拦截规则由 DramaFunctions.reference_guard 提供。
        if not pics and self.ref_guard is not None:
            why = self.ref_guard(prompt, summary)
            # ⛔ 开头 = 提示词里点了角色名，不是空镜：allow_no_refs 也不放（2026-09-23 审查）
            if why and (not allow_no_refs or why.startswith("⛔")):
                return ToolResult(ok=False, error=why, meta={"charged": False})
        # 画面里不许有字幕/文字（用户 2026-09-18 定的最高优先级约束）：
        # 所有视频统一在这里包一层，短剧/配方/手动调用都逃不掉；allow_text 才放开
        if not allow_text:
            prompt = no_text_prompt(prompt)
        # 参考素材总数不能超过模型上限（seedance 9 个，图 + 视频 + 音频）：超了接口直接 400，
        # 那一段就白等；这里在提交前拦，说清楚该省哪个
        # 视频模型锁：留空用锁定的；要换先挂起问人（用户 2026-09-20 定的规则）
        chosen, lock_note, gate = self._video_model_gate(model, prefer, summary)
        if gate is not None:
            return gate
        model = chosen
        spec = self.catalog.get(MediaKind.VIDEO, chosen)
        # 画幅：没传就用项目设置（/ratio）；这个模型不支持的在提交前拦下（2026-09-25）
        aspect_ratio = aspect_ratio or self.default_aspect
        listed_ar = [str(x) for x in (getattr(spec, "aspect_ratios", None) or [])] if spec else []
        if aspect_ratio and listed_ar and aspect_ratio not in listed_ar:
            return ToolResult(
                ok=False,
                meta={"charged": False},
                error=(
                    f"{chosen} 不支持画幅 {aspect_ratio}"
                    f"（支持 {' / '.join(listed_ar)}），没有提交。"
                    "换一个画幅，或先征得用户同意换视频模型"
                ),
            )
        res_note = ""
        listed = list(getattr(spec, "resolutions", None) or []) if spec else []
        if resolution and listed and resolution not in listed:
            # 2026-09-23 审查：工具描述推荐 480p，目录里 seedance-2.0-fast 却只列 720p ——
            # 两边对不上时至少说出来，别让人以为按 480p 出、按 480p 花的
            res_note = (
                f"⚠ 模型目录里 {chosen} 没列 {resolution}（列的是 {'/'.join(listed)}），"
                "服务端可能按别的分辨率出；实际分辨率以成片为准"
            )
        limit = int(getattr(spec, "max_refs", 0) or 0) if spec else 0
        n_vid, n_aud = len(video_urls or []), len(audio_urls or [])
        if limit and len(pics) + n_vid + n_aud > limit:
            return ToolResult(
                ok=False,
                meta={"charged": False},
                error=(
                    f"参考素材 {len(pics) + n_vid + n_aud} 个超过 {chosen} 的上限 {limit}"
                    f"（图 {len(pics)} + 视频 {n_vid} + 音频 {n_aud}），没有提交。"
                    "先删减：人物参考图优先保留，再场景，最后道具 / 参考视频。"
                ),
            )
        # 参考素材的字段名走目录配置（media_models.yaml）：图片实测是 image；
        # 视频/音频参考按 seedance 文档默认 video_urls / audio_urls，接口对不上时改配置不改代码。
        refs: dict[str, list[str]] = {}
        for field_name, urls in (
            (self.catalog.image_ref_field, pics),
            (self.catalog.video_ref_field, video_urls or []),
            (self.catalog.audio_ref_field, audio_urls or []),
        ):
            if urls:
                refs.setdefault(field_name, []).extend(urls)  # 同名字段就合并成一个列表
        # 参考图再按 seedance 文档的字段名（image_urls）送一份：接口只认其中一个时另一个被忽略，
        # 两个都送不会出错，但能避免「传了参考图其实零参考」（2026-09-18 用户反馈人物漂移）。
        # 但接口把两份**都算进参考数**（实测 5 张双送 = 10 个 → 400），会超上限时只送实测有效的那份
        alias = getattr(self.catalog, "image_ref_alias", "") or ""
        alias_fits = not limit or 2 * len(pics) + n_vid + n_aud <= limit
        if alias and alias not in refs and pics and alias_fits:
            refs[alias] = list(pics)
        result = await self._generate(
            MediaKind.VIDEO,
            prompt,
            model,
            prefer,
            summary,
            parent_id,
            AssetType.VIDEO,
            local_name=local_name,
            tags=tags,
            aspect_ratio=aspect_ratio or None,
            duration=duration,
            resolution=resolution or None,
            **refs,
        )
        notes = [n for n in (lock_note, res_note) if n]
        if notes and result.ok:
            result.content = "\n".join([result.content, *notes])
        return result

    async def _generate(
        self,
        kind: MediaKind,
        prompt: str,
        model: str,
        prefer: str,
        summary: str,
        parent_id: str,
        asset_type: AssetType,
        *,
        local_name: str = "",
        tags: dict[str, Any] | None = None,
        **params: Any,
    ) -> ToolResult:
        chosen, why = self.catalog.choose(kind, model, prefer)
        if not chosen:
            # 不静默替换成别的模型
            return ToolResult(ok=False, error=why, meta={"charged": False})

        spec = self.catalog.get(kind, chosen)
        clamp_note = ""
        if spec and spec.max_duration and params.get("duration"):
            want = int(params["duration"])
            if want > spec.max_duration:
                params["duration"] = spec.max_duration
                # 之前静默截断：换了个单段上限更短的模型，时间线按 15s 排的镜头尾巴全被砍掉，
                # 一集只剩约 2/3 却没有任何提示（2026-09-23 审查）
                clamp_note = (
                    f"⚠ {chosen} 单段最长 {spec.max_duration}s，这段从 {want}s 截到了 "
                    f"{spec.max_duration}s —— 按 {want}s 排的镜头时间线尾巴会被砍掉"
                )

        task = await self.gateway.generate(
            self.catalog.provider,
            kind,
            chosen,
            prompt,
            max_wait_s=self.catalog.max_wait(kind),
            **params,
        )

        if not task.ok:
            quota = is_quota_error(task.http_status, task.error or "")
            return ToolResult(
                ok=False,
                error=(
                    f"{chosen} 生成失败（{task.status.value}，{task.elapsed_s}s，"
                    f"轮询 {task.polls} 次）：{task.error or '未知原因'}"
                ),
                # 给调用方程序看：能不能原样重提（2026-09-23 审查：之前靠错误文本猜，
                # 把「轮询失败」也当网络抖动重提，同一个镜头付两份钱）
                meta={
                    "retryable": task.retryable and not quota,
                    "task_id": task.task_id,
                    "stage": task.stage,
                    "status": task.status.value,
                    # 提交阶段被拒收 / 请求没送到 / 请求本身有错（4xx）：服务端没建任务、没计费
                    "charged": not (
                        task.stage == "submit"
                        and (task.retryable or 400 <= task.http_status < 500)
                    ),
                    # 余额 / 额度用完：调用方据此停下整批，不再一段段撞（2026-09-29 审查 1.1）
                    "quota": quota,
                },
            )

        result = await self._register(
            task, kind, prompt, chosen, why, summary, parent_id, asset_type,
            local_name=local_name, tags=tags, params=params,
        )
        if clamp_note:
            result.content = f"{result.content}\n{clamp_note}"
        return result

    async def _register(
        self,
        task: Any,
        kind: MediaKind,
        prompt: str,
        chosen: str,
        why: str,
        summary: str,
        parent_id: str,
        asset_type: AssetType,
        *,
        local_name: str = "",
        tags: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> ToolResult:
        """把一次成功的生成任务登记成资产（+ 本地副本 + Agent 自留备份）。

        _generate 和 media_recover 共用：取回的旧任务和新生成的走同一套落库口径。
        """
        spec = self.catalog.get(kind, chosen)
        # 单价来自目录（media_models.yaml 的 price，参考价）：网关不回传金额，
        # 目录没填就留空 —— 那时 Cost Guard 只剩次数口径，但次数口径始终有效。
        price = self.catalog.price_of(kind, chosen, params or {}) if hasattr(
            self.catalog, "price_of"
        ) else (spec.price if spec else None)
        gp_base: dict[str, Any] = {"prompt": prompt, "model": chosen, **(params or {})}
        if tags:
            gp_base["tags"] = dict(tags)
        if task.task_id and task.task_id != "sync":
            gp_base["task_id"] = task.task_id
        assets = []
        for i, url in enumerate(task.urls):
            a = self.store.create(
                content="",
                type_=asset_type,
                summary=summary or f"{prompt[:24]}…" + (f" #{i + 1}" if len(task.urls) > 1 else ""),
                parents=[parent_id] if parent_id else [],
                creator=f"model:{chosen}",
                gen_params=dict(gp_base),
                gen_cost=price,
            )
            a.uri = url
            a.mime = "image/png" if kind is MediaKind.IMAGE else "video/mp4"
            self.store.put(a)
            assets.append(a)

        # 金额在闸门**放行时**就按目录单价记过了（提交即计费，2026-09-23），这里不再补
        # COST —— 否则同一笔记两遍。gen_cost 仍写在资产上，成本归因照旧。

        # 本地副本：落一份到用户产物目录。uri 保持远端 URL 不动 —— 后续镜头的
        # 参考图链要的是 URL；本地路径记进 gen_params["local"]。
        # 下载失败不拖累生成（远端链接仍在）。
        local_note = ""
        if self.prefs is not None:
            sub = "images" if kind is MediaKind.IMAGE else "videos"
            ext = ".png" if kind is MediaKind.IMAGE else ".mp4"
            folder = self.prefs.dir_for(sub)
            got: list[str] = []
            for i, (a, url) in enumerate(zip(assets, task.urls, strict=True)):
                # 文件名：调用方给了 local_name 就用它（带顺序号，后期能排序），
                # 否则退回资产 id。同名已存在（重渲）加 -v2，不覆盖旧版。
                base = safe_name(local_name) if local_name else a.id
                if len(assets) > 1:
                    base += f"-{i + 1}"
                target = unique_path(folder, base + ext)
                p = await self.prefs.download(url, folder, target.name)
                if p and kind is MediaKind.IMAGE:
                    # 生图接口回的不一定是 PNG（jpg / webp 都有）：按文件头改成真实扩展名 ——
                    # 之前一律存 .png，按扩展名认格式的看图软件、图床会出错（2026-09-24 审查）
                    p = _fix_image_ext(p)
                    a.mime = _IMAGE_MIME.get(p.suffix.lower(), a.mime)
                if p:
                    a.gen_params["local"] = str(p)
                    # Agent 自己留一份（同盘硬链接不占空间）：产物目录是用户的，
                    # 整理/删掉之后远端链接也早过期了，参考图就再也找不回来
                    blob = keep_blob(self.store, p, a.id)
                    if blob:
                        a.gen_params["blob"] = str(blob)
                    self.store.put(a)
                    got.append(p.name)
            if got:
                local_note = f"\n本地副本：{folder}（{len(got)}/{len(assets)}）：{', '.join(got)}"
            if len(got) < len(assets):
                local_note += (
                    f"\n⚠ {len(assets) - len(got)} 个本地副本没下载下来（远端链接约 24 小时失效，"
                    "字幕/镜头检查也要靠本地副本）"
                )

        ledger = getattr(self.gateway, "ledger", None)
        if ledger is not None and task.task_id and task.task_id != "sync":
            ledger.delivered(task.task_id)

        lines = [f"{a.id} → {a.uri}" for a in assets]
        head = f"用 {chosen}（{why}）生成了 {len(assets)} 个，耗时 {task.elapsed_s}s：\n"
        if getattr(task, "recovered", False):
            head = (
                f"取回了之前没拿到结果的任务 {task.task_id}（没有重新付费），"
                f"{chosen} 共 {len(assets)} 个：\n"
            )
        recovered = bool(getattr(task, "recovered", False))
        meta: dict[str, Any] = {"task_id": task.task_id, "recovered": recovered}
        if recovered:
            meta["charged"] = False  # 取回的没重新付费：闸门记的这次额度退回
        return ToolResult(
            content=head + "\n".join(lines) + local_note,
            asset_ref=assets[0].id,
            meta=meta,
        )

    # ---------- 任务台账：钱花了没拿到结果的，能取回 ----------

    async def _fn_media_tasks(self, limit: int = 20) -> ToolResult:
        ledger = getattr(self.gateway, "ledger", None)
        if ledger is None:
            return ToolResult(content="没有接任务台账（脚本 / 测试环境）")
        rows = ledger.pending(limit=max(1, min(int(limit or 20), 100)))
        if not rows:
            return ToolResult(content="台账里没有未取回的媒体任务。")
        now = time.time()
        lines = [
            f"- {r.task_id} · {r.kind}/{r.model} · {r.status} · {r.age_h(now):.1f} 小时前 · "
            f"{(r.prompt or '')[:40]}" + (f" · {r.error[:60]}" if r.error else "")
            for r in rows
        ]
        return ToolResult(
            content=(
                f"未取回的媒体任务 {len(rows)} 个（新的在前）。它们多半已在服务端生成并计费，"
                "用 media_recover(task_id=…) 取回成资产；超过 24 小时的链接可能已失效：\n"
                + "\n".join(lines)
            )
        )

    async def _fn_media_recover(
        self, task_id: str, summary: str = "", local_name: str = "", parent_id: str = ""
    ) -> ToolResult:
        ledger = getattr(self.gateway, "ledger", None)
        rec = ledger.get(task_id) if ledger is not None else None
        if rec is None:
            return ToolResult(
                ok=False, error=f"台账里没有任务 {task_id}（media_tasks 列出可取回的）"
            )
        if rec.delivered:
            return ToolResult(ok=False, error=f"任务 {task_id} 的结果已经登记过资产，不用再取回")
        try:
            kind = MediaKind(rec.kind)
        except ValueError:
            return ToolResult(ok=False, error=f"任务 {task_id} 的类型 {rec.kind!r} 不认识")
        task = await self.gateway.recover(
            self.catalog.provider, task_id, kind, rec.model, max_wait_s=self.catalog.max_wait(kind)
        )
        if not task.ok:
            return ToolResult(
                ok=False,
                error=f"任务 {task_id} 取不回来（{task.status.value}）：{task.error or '未知原因'}",
                meta={"retryable": False, "task_id": task_id},
            )
        params = dict(rec.params or {})
        tags = params.pop("tags", None)
        return await self._register(
            task, kind, rec.prompt, rec.model, "取回", summary or f"取回·{rec.prompt[:20]}…",
            parent_id, AssetType.IMAGE if kind is MediaKind.IMAGE else AssetType.VIDEO,
            local_name=local_name, tags=tags, params=params,
        )


# 请求确实没送到服务端的错误（重提安全）。其余一律不自动重提。
_SEND_FAILED = ("ConnectError", "ConnectTimeout", "PoolTimeout", "连不上服务端")


def retryable_failure(result: Any) -> bool:
    """这次失败能不能**原样重新提交**。

    优先看结构化标记（ToolResult.meta["retryable"]，媒体网关按「请求到没到服务端」判）；
    没有标记（旧调用方 / 非媒体工具）才看错误文本，而且只认「连不上」这一类 ——
    「轮询失败」「HTTP 5xx」都不算：那时任务多半已在服务端跑、已计费（2026-09-23 审查）。
    """
    if result is None or getattr(result, "ok", False):
        return False
    meta = getattr(result, "meta", None) or {}
    if "retryable" in meta:
        return bool(meta["retryable"])
    err = str(getattr(result, "error", "") or "")
    if "轮询" in err or "超时" in err:
        return False  # 轮询阶段的连接错误同样叫 ConnectError，但那时任务已在服务端
    return any(k in err for k in _SEND_FAILED)


def keep_blob(store: Any, src: Any, asset_id: str) -> Any:
    """在资产库的 blobs/ 下给本地副本留一份：同盘硬链接（不占空间），跨盘复制小文件。

    用户的产物目录会被整理、移动、删除（2026-09-23 审查：E:\\内容测试\\images 整个没了，
    194 个图片资产的本地副本全断，远端链接也早过期）—— Agent 自己得留一份。
    """
    import os
    import shutil
    from pathlib import Path

    root = getattr(store, "root", None)
    if root is None:
        return None
    try:
        src = Path(src)
        d = Path(root) / "blobs"
        d.mkdir(parents=True, exist_ok=True)
        dest = d / f"{asset_id}{src.suffix}"
        if dest.exists():
            return dest
        try:
            os.link(src, dest)
        except OSError:
            if src.stat().st_size > 50 * 1024 * 1024:
                return None  # 跨盘的大视频不复制（占空间），靠远端/产物目录
            shutil.copy2(src, dest)
        return dest
    except OSError:
        return None


# ---------------------------------------------------------------- 图片格式


_IMAGE_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif",
}


def _sniff_image_ext(head: bytes) -> str:
    """按文件头认图片格式。认不出返回空串。"""
    if head.startswith(b"\x89PNG"):
        return ".png"
    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return ".webp"
    if head[:4] == b"GIF8":
        return ".gif"
    return ""


def _fix_image_ext(p: Path) -> Path:
    """扩展名和真实格式对不上就改名（同名已存在加 -v2）。改不了原样返回。"""
    try:
        with p.open("rb") as f:
            head = f.read(12)
    except OSError:
        return p
    ext = _sniff_image_ext(head)
    have = p.suffix.lower()
    if not ext or have == ext or (ext == ".jpg" and have == ".jpeg"):
        return p
    target = unique_path(p.parent, p.stem + ext)
    try:
        p.rename(target)
    except OSError:
        return p
    return target

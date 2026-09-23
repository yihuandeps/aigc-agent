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
import time
from typing import Any

from ...harness.events.bus import EventType
from ...harness.model.media import MediaGateway, MediaKind
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
        self.image_lock = ""
        self.on_image_lock = None
        self._pending_switch: dict[str, str] = {}
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
                    "aspect_ratio": {"type": "string", "description": "短视频用 9:16"},
                    "duration": {"type": "integer", "description": "秒，受模型上限约束"},
                    "resolution": {
                        "type": "string",
                        "enum": ["480p", "720p", "1080p", "4k"],
                        # 480p 是 2026-09-22 对着 seedance-2.0-fast 实测过的：
                        # 出 496×864，最省钱，验证分镜方向用它
                        "description": "验证方向用 480p 最省，成片再上 720p",
                    },
                    "image_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "参考图/首帧 url。人物镜头必须带角色参考图；"
                        "参考素材总数受模型上限约束（seedance 9 个，含参考视频），超了直接拒绝",
                    },
                    "allow_no_refs": {
                        "type": "boolean",
                        "description": "没有参考图也允许生成（默认 false：短剧镜头/提到角色的提示词"
                        "不带参考图会被拦下）。只有确定是无人物的空镜才传 true",
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
                        "禁字幕/禁文字约束（成片字幕由后期压，不能让模型烧进画面）",
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
                    "aspect_ratio": {"type": "string", "description": "公共比例，短视频用 9:16"},
                    "resolution": {
                        "type": "string",
                        "enum": ["480p", "720p", "1080p", "4k"],
                        # 480p 是 2026-09-22 对着 seedance-2.0-fast 实测过的：
                        # 出 496×864，最省钱，验证分镜方向用它
                        "description": "验证方向用 480p 最省，成片再上 720p",
                    },
                    "duration": {"type": "integer", "description": "公共时长秒，可被单项覆盖"},
                    "parent_id": {"type": "string"},
                    "allow_text": {"type": "boolean"},
                    "allow_no_refs": {
                        "type": "boolean",
                        "description": "没有参考图也允许生成"
                        "（默认 false，短剧镜头不带参考图会被拦）",
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

    # ---------- ToolProvider 协议 ----------

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        n_i, n_v = len(self.catalog.image), len(self.catalog.video)
        return ProviderHealth(ok=True, detail=f"图 {n_i} 个 / 视频 {n_v} 个模型")

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            result = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
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
            return ToolResult(ok=False, error="jobs 是空的")
        if len(jobs) > 40:
            return ToolResult(ok=False, error=f"一次最多 40 个，收到 {len(jobs)} 个；分批调用")
        label = "视频" if kind is MediaKind.VIDEO else "图片"
        fn = self._fn_gen_video if kind is MediaKind.VIDEO else self._fn_gen_image
        limit = self.catalog.max_concurrency("video" if kind is MediaKind.VIDEO else "image")
        sem = asyncio.Semaphore(limit) if limit > 0 else None

        async def one(i: int, job: dict[str, Any]) -> tuple[int, ToolResult]:
            args = {**shared, **{k: v for k, v in job.items() if v is not None}}
            args.setdefault("summary", f"{label} {i + 1}")
            if sem is None:
                return i, await fn(**args)
            async with sem:
                return i, await fn(**args)

        started = time.perf_counter()
        results = await asyncio.gather(
            *(one(i, j) for i, j in enumerate(jobs)), return_exceptions=True
        )
        lines: list[str] = []
        ok_ids: list[str] = []
        failed = 0
        for item in results:
            if isinstance(item, BaseException):
                failed += 1
                lines.append(f"  ✗ {type(item).__name__}: {item}")
                continue
            i, r = item
            name = jobs[i].get("summary") or f"{label} {i + 1}"
            if r.ok and r.asset_ref:
                ok_ids.append(r.asset_ref)
                lines.append(f"  ✓ {name} → {r.asset_ref}")
            else:
                failed += 1
                lines.append(f"  ✗ {name}：{(r.error or '未知原因')[:100]}")
        dt = time.perf_counter() - started
        head = (
            f"批量生成{label} {len(jobs)} 个：成功 {len(ok_ids)}，失败 {failed}"
            f"（并发上限 {limit or '不限'}，耗时 {dt / 60:.1f} 分钟）"
        )
        if not ok_ids:
            return ToolResult(ok=False, error=head + "\n" + "\n".join(lines))
        tail = f"\n\n按顺序的资产 id：{' '.join(ok_ids)}"
        return ToolResult(content=head + "\n" + "\n".join(lines) + tail, asset_ref=ok_ids[0])

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
        return await self._run_batch(MediaKind.IMAGE, jobs, shared)

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
        """总线回调：人对「模型切换」拍板后，**采纳**才真正换锁；打回/退回不换。"""
        if getattr(event, "type", None) is not EventType.CHECKPOINT_DECIDED:
            return
        data = getattr(event, "data", None) or {}
        node = data.get("node")
        for kind, (_, _, stage, _, _) in _LOCK_SPECS.items():
            if node != stage:
                continue
            target = self._pending_switch.pop(stage, "")
            if data.get("decision") == "adopt" and target:
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
        if lock and model and model != lock:
            chosen, why = self.catalog.choose(kind, model, prefer)
            if not chosen:
                return "", "", ToolResult(ok=False, error=why)
            self._pending_switch[stage] = chosen
            question = (
                f"要把{label}模型从 {lock} 换成 {chosen} 吗？（本次生成：{summary or '未命名'}）\n"
                f"换模型会改变画质与风格，之后的{thing}也都用新模型。"
                f"回复 a 同意换；r 或 j 不换，继续用 {lock}。"
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
                    f"用户同意后再调用一次即可；不同意就继续用 {lock}。"
                ),
                suspend=True,
                suspend_payload=payload,
            )
        if lock and not model:
            return lock, "", None
        chosen, why = self.catalog.choose(kind, model, prefer)
        if not chosen:
            return "", "", ToolResult(ok=False, error=why)
        if not lock:
            self._set_lock(kind, chosen)
            note = f"{label}模型已锁定为 {chosen}（{why}）：之后都用它，要换会先问用户。"
            return chosen, note, None
        return chosen, "", None

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
        return await self._run_batch(MediaKind.VIDEO, jobs, shared)

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
    ) -> ToolResult:
        bad = _not_public(list(image or []))
        if bad:
            return ToolResult(ok=False, error=_local_ref_error(bad))
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
    ) -> ToolResult:
        pics = list(image or image_urls or [])
        bad = _not_public(pics + list(video_urls or []) + list(audio_urls or []))
        if bad:
            return ToolResult(ok=False, error=_local_ref_error(bad))
        # 参考图门（2026-09-20 用户定的规则）：短剧镜头没带参考图不许生成，在提交前拦，
        # 不能生成完再说"没引用成功"。拦截规则由 DramaFunctions.reference_guard 提供。
        if not pics and not allow_no_refs and self.ref_guard is not None:
            why = self.ref_guard(prompt, summary)
            if why:
                return ToolResult(ok=False, error=why)
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
        limit = int(getattr(spec, "max_refs", 0) or 0) if spec else 0
        n_vid, n_aud = len(video_urls or []), len(audio_urls or [])
        if limit and len(pics) + n_vid + n_aud > limit:
            return ToolResult(
                ok=False,
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
            aspect_ratio=aspect_ratio or None,
            duration=duration,
            resolution=resolution or None,
            **refs,
        )
        if lock_note and result.ok:
            result.content = f"{result.content}\n{lock_note}"
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
        **params: Any,
    ) -> ToolResult:
        chosen, why = self.catalog.choose(kind, model, prefer)
        if not chosen:
            return ToolResult(ok=False, error=why)  # 不静默替换成别的模型

        spec = self.catalog.get(kind, chosen)
        if spec and spec.max_duration and params.get("duration"):
            params["duration"] = min(int(params["duration"]), spec.max_duration)

        task = await self.gateway.generate(
            self.catalog.provider,
            kind,
            chosen,
            prompt,
            max_wait_s=self.catalog.max_wait(kind),
            **params,
        )

        if not task.ok:
            return ToolResult(
                ok=False,
                error=(
                    f"{chosen} 生成失败（{task.status.value}，{task.elapsed_s}s，"
                    f"轮询 {task.polls} 次）：{task.error or '未知原因'}"
                ),
            )

        # 产物落 Asset —— 血缘、成本归因照旧。
        # 单价来自目录（media_models.yaml 的 price，参考价）：网关不回传金额，
        # 目录没填就留空 —— 那时 Cost Guard 只剩次数口径，但次数口径始终有效。
        price = spec.price if spec else None
        assets = []
        for i, url in enumerate(task.urls):
            a = self.store.create(
                content="",
                type_=asset_type,
                summary=summary or f"{prompt[:24]}…" + (f" #{i + 1}" if len(task.urls) > 1 else ""),
                parents=[parent_id] if parent_id else [],
                creator=f"model:{chosen}",
                gen_params={"prompt": prompt, "model": chosen, **params},
                gen_cost=price,
            )
            a.uri = url
            a.mime = "image/png" if kind is MediaKind.IMAGE else "video/mp4"
            self.store.put(a)
            assets.append(a)

        if price is not None and assets:
            # 走 COST 事件补金额。次数已经在闸门处记过，这里不重复计次。
            await self.gateway.bus.emit(
                EventType.COST,
                modality=kind.value,
                model=chosen,
                cost=price * len(assets),
                outputs=len(assets),
            )

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
                if p:
                    a.gen_params["local"] = str(p)
                    self.store.put(a)
                    got.append(p.name)
            if got:
                local_note = f"\n本地副本：{folder}（{len(got)}/{len(assets)}）：{', '.join(got)}"

        lines = [f"{a.id} → {a.uri}" for a in assets]
        return ToolResult(
            content=(
                f"用 {chosen}（{why}）生成了 {len(assets)} 个，耗时 {task.elapsed_s}s：\n"
                + "\n".join(lines)
                + local_note
            ),
            asset_ref=assets[0].id,
        )

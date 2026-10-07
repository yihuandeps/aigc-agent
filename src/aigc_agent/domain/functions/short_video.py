"""抖音短视频分支的 functions（2026-09-18，用户定的新路径）。

  关键词 + 风格 ──RPA 抓抖音/小红书热点──▶ short_video_brief（分析归纳成新内容，定做法与时长）
      │
      ├─ 需要实拍/外部素材的镜头 ──▶ request_materials（挂起问人）──▶ resolve_materials
      │                               └─ 联网：stock_media_search / fetch_stock_media
      └─ short_video_produce（缺的镜头并发生成 + 配音 + 字幕 + 快切合成）──▶ view_video 复核

和 `agent video make` 那条命令行链的区别：这里全部是**工具**，模型在聊天里按 skill
（skills/douyin-short）逐步调；热点来自 RPA 真实抓取而不是接口；镜头可以用真实素材；
生成有并发/重试/复用。配方（config/recipes/*.yaml）= 风格，仍然是数据。
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from ...harness.events.bus import EventBus, EventType
from ...harness.model.media import MediaKind
from ...harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
    ToolSpec,
)
from ..aspect import DEFAULT_ASPECT, parse_aspect, still_size
from ..assets.store import AssetStore, AssetType, local_copy
from ..media import ffmpeg
from ..media.naming import recipe_shot_name, safe_name
from ..media.no_text import no_text_retry, parse_subtitle_verdict, subtitle_check_messages
from ..pipeline.recipe import Recipe, list_recipes, load_recipe
from ..pipeline.short_video import (
    Brief,
    brief_prompt,
    materials_question,
    parse_brief,
    parse_material_reply,
)
from ..pipeline.subtitle import align_script
from ..pipeline.voice_pick import parse_pick as parse_voice
from ..pipeline.voice_pick import pick_prompt as voice_prompt
from ..realism import REALISM_ROLE, norm_level
from .drama import _run_parallel
from .media import retryable_failure

PLANNER_ROLE = "short_video_planner"
# 带产品参考图时加在镜头提示词最前面（身份锁：只继承产品本身，不继承参考图的背景构图）
_REF_LOCK = (
    "【参考锁定】@图片1 是产品本体（身份锁）：外形轮廓、颜色、材质、盖子与铭牌位置必须与参考图"
    "一致；只继承产品本身，不继承参考图的背景、构图和光线。"
)
_MAX_SOURCE_CHARS = 20_000  # 喂给规划模型的热点材料总上限
CONFIRM_STAGE = "出片确认"  # 花钱前的确认单（major：/auto 也停）
REVIEW_STAGE = "成片审核"  # 配方 review.after_compose：成片出来先给人看
SCRIPT_STAGE = "文案审核"  # 配方 review.after_script：文案和分镜出来先给人看


class ShortVideoFunctions:
    name = "short_video"
    namespaced = False
    disclosure = "full"

    def __init__(
        self,
        gateway: Any,
        store: AssetStore,
        registry: Any = None,
        catalog: Any = None,
        bus: EventBus | None = None,
        output: Any = None,
        files: Any = None,
        recipes_dir: Path | None = None,
        hosting: Any = None,
    ) -> None:
        self.gateway = gateway
        self.store = store
        self.registry = registry
        self.catalog = catalog
        self.bus = bus
        self.output = output  # OutputPrefs：图转镜头等中间产物落哪
        self.files = files  # FileFunctions：用户贴的素材路径按同一套边界解析
        self.recipes_dir = recipes_dir
        # 素材托管：产品参考图是本地文件时换成公网链接（生成接口只收 http(s)）
        self.hosting = hosting
        # 会话锁定的视频模型（装配层接 MediaFunctions.video_lock）：锁着时 tier 不参与选型
        self.video_lock_source: Any = None
        # 项目画幅（/ratio）：比配方的 output.aspect_ratio 优先；空 = 按配方
        self.aspect_ratio = ""
        # 出片确认只认人的决定（2026-09-26 审查）：confirm=true 是模型自己就能填的参数，
        # 之前第一次就带上它，确认单根本不出现、直接花钱。现在挂起时记下这张确认单（按简报），
        # 人采纳（总线 CHECKPOINT_DECIDED，/auto 自动采纳的不算）才转成已确认，用一次就收回；
        # 这一轮结束（LOOP_END）没用掉的也收回。命令行 `agent video make` 在终端问完调 approve()
        self._pending_confirm: dict[str, dict[str, Any]] = {}
        self._confirmed: dict[str, dict[str, Any]] = {}
        self._specs: dict[str, ToolSpec] = {}
        self._build()

    # ---------- 人的确认 ----------

    def approve(self, brief_id: str) -> None:
        """人在别处（终端）确认了这份简报的出片：下一次带 confirm=true 的调用放行一次。"""
        self._confirmed[brief_id] = self._pending_confirm.pop(brief_id, None) or {"any": True}

    def on_event(self, event: Any) -> None:
        """总线回调：人在出片确认单上**采纳**才算确认；打回 / 退回、/auto 自动采纳都不算。"""
        etype = getattr(event, "type", None)
        data = getattr(event, "data", None) or {}
        if etype is EventType.LOOP_END:
            if data.get("stop_reason") != "awaiting_review":
                self._confirmed.clear()
            return
        if etype is not EventType.CHECKPOINT_DECIDED or data.get("node") != CONFIRM_STAGE:
            return
        for bid in data.get("candidates") or []:
            plan = self._pending_confirm.pop(str(bid), None)
            if plan is None or data.get("decision") != "adopt":
                continue
            if str(data.get("decided_by") or "human") == "auto":
                continue
            self._confirmed[str(bid)] = plan

    def _take_confirmation(self, brief_id: str, plan: dict[str, Any]) -> bool:
        """这次要生成的是不是人确认过的那张单子（同样的生成条件、镜头只少不多）。用一次就收回。"""
        ok = self._confirmed.pop(brief_id, None)
        if ok is None:
            return False
        if ok.get("any"):
            return True
        return ok.get("gen") == plan.get("gen") and set(plan.get("shots") or []) <= set(
            ok.get("shots") or []
        )

    def permission_for(
        self, tool: str, args: dict[str, Any]
    ) -> tuple[PermissionLevel, str] | None:
        """注册表的提权钩子：skip_shots（缺着镜头出片）要当场问人 —— 丢哪几镜是内容决定。"""
        if tool != "short_video_produce":
            return None
        skip = [s for s in (args.get("skip_shots") or []) if str(s).strip()]
        if not skip:
            return None
        return (
            PermissionLevel.EXTERNAL,
            f"不要第 {'、'.join(str(s) for s in skip)} 镜、缺着它们出片 —— 成片会少这几个镜头，"
            "需要你确认",
        )

    # ---------- 声明 ----------

    def _build(self) -> None:
        self._specs["list_video_styles"] = ToolSpec(
            name="list_video_styles",
            summary="列出可选的短视频风格（= 配方）：适合什么、时长范围、画面靠生成还是实拍",
            permission=PermissionLevel.READ,
            description="做抖音短视频第一步：让用户从这里选一种风格并确认，再去抓热点。",
            parameters={"type": "object", "properties": {}},
        )
        self._specs["short_video_brief"] = ToolSpec(
            name="short_video_brief",
            summary="把抓到的抖音/小红书热点分析归纳成新内容，定切入角度、时长、口播、分镜与素材来源",
            permission=PermissionLevel.COMPUTE,
            # 不设 cost_kind：文本花费由模型调用的 COST 事件记，闸门再记一次就重了（2026-09-23）
            description=(
                "热点抓完后调。sources 传热点资产 id（douyin_hot_rpa / xhs_collect / "
                "browse_and_copy / douyin_hot_list 的产物），模型会交叉比对归纳成一段新内容，"
                "事实逐条标出处；"
                "时长在风格允许范围内按内容密度决定；每个镜头标明 AI 生成还是需要实拍/外部素材。"
                "产物是选题简报资产，后面 request_materials / short_video_produce 都靠它。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "用户给的关键词"},
                    "style": {
                        "type": "string",
                        "description": "风格 = 配方名，见 list_video_styles",
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "热点资产 id 列表",
                    },
                    "notes": {"type": "string", "description": "用户的额外要求，可省略"},
                    "grounding": {
                        "type": "boolean",
                        "description": "配方要求接热榜、又没传 sources 时，是否自动拉一次抖音热榜"
                        "（默认按配方）。用户说不用热点才传 false",
                    },
                },
                "required": ["keyword", "style"],
            },
            max_result_chars=12_000,
        )
        self._specs["request_materials"] = ToolSpec(
            name="request_materials",
            summary="简报里需要实拍/外部素材的镜头 → 挂起向用户要（给文件 / 让我联网找 / 改生成）",
            permission=PermissionLevel.WRITE,
            description=(
                "简报里有 source=real 的镜头就调这个。会列出每个镜头要什么素材、可搜的关键词，"
                "等用户回复。用户回复后把原话交给 resolve_materials。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "brief_id": {"type": "string"},
                    "shots": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "只问哪几个镜头（1 起），省略 = 全部需要实拍的镜头",
                    },
                    "note": {"type": "string", "description": "附加说明，可省略"},
                },
                "required": ["brief_id"],
            },
        )
        self._specs["resolve_materials"] = ToolSpec(
            name="resolve_materials",
            summary="读用户对素材请求的回复：贴的文件登记成素材、改生成的改标记、要联网找的给出搜索词",
            permission=PermissionLevel.WRITE,
            description=(
                "把用户的回复原话传进来。文件路径会登记成资产并绑到镜头；「生成」会把该镜头改为 "
                "AI 生成；「联网找」会返回该镜头的搜索词 —— 接着用 stock_media_search 挑一条、"
                "fetch_stock_media 下载，再把资产 id 通过 short_video_produce 的 materials 传入。"
                "返回更新后的简报 id。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "brief_id": {"type": "string"},
                    "reply": {"type": "string", "description": "用户的回复原话"},
                },
                "required": ["brief_id", "reply"],
            },
        )
        self._specs["short_video_produce"] = ToolSpec(
            name="short_video_produce",
            summary="按简报出片：缺的镜头并发生成（网络失败重试、重跑复用），配音、字幕、快切合成",
            permission=PermissionLevel.COMPUTE,
            # 不设 cost_kind：它里面每段 gen_video 都单独过闸门计次计秒，外层再记一次就重了
            timeout=7200,
            description=(
                "素材都定了再调。materials 传 {镜头序号: 资产id}（用户给的或素材站下的；"
                "图片会自动做成静止镜头），没传素材的实拍镜头会改用 AI 生成并在结果里标出来。"
                "口播出镜：用户的出镜视频传 aroll（它的原声当音轨、字幕从它转写，"
                "B-roll 穿插）。广告：产品图传 ref_images（每个生成镜头都带上当身份锁），"
                "个别镜头不同就用 shot_refs。"
                "要新生成镜头时会先停下来报镜头数和秒数，人在确认单上采纳后带 confirm=true "
                "再调（没经过人确认的 confirm=true 不算数，会再出一次确认单）。"
                "有镜头没生成出来（失败 / 画面有字）就**不合成成片**，结果里列出缺哪几镜："
                "修好原因再调一次（成功的镜头复用、只补缺的）；用户说缺的不要了才传 skip_shots。"
                "画面按简报顺序快切（每刀 ≤3s），长度以配音 / 出镜原声为准。"
                "配方要求成片审核的（如产品广告）合成后会停下来请用户看。"
                "上屏文字（slogan / 卖点 / 片尾字卡）不让模型画，成片出来后用 overlay_text 叠。"
                "跑完用 view_video 看成片。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "brief_id": {"type": "string"},
                    "materials": {
                        "type": "object",
                        "description": '{"3": "as_xxx"} 镜头序号 → 素材资产 id',
                        "additionalProperties": {"type": "string"},
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["quality", "balanced", "fast"],
                        "description": "生成档位，省略用配方的",
                    },
                    "out_dir": {"type": "string", "description": "成片导出目录"},
                    "filename": {"type": "string", "description": "成片文件名"},
                    "reuse": {"type": "boolean", "description": "复用上次已生成的镜头，默认 true"},
                    "no_voiceover": {"type": "boolean"},
                    "no_subtitle": {"type": "boolean"},
                    "voice": {"type": "string", "description": "指定音色，省略按文案自动挑"},
                    "aroll": {
                        "type": "string",
                        "description": "口播出镜素材（用户对着镜头说话、带原声的视频）的资产 id",
                    },
                    "ref_images": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "所有生成镜头都带的参考图（产品图资产 id 或链接）",
                    },
                    "shot_refs": {
                        "type": "object",
                        "description": '个别镜头单独的参考图：{"3": ["as_xxx"]}',
                        "additionalProperties": {"type": "array", "items": {"type": "string"}},
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "人已经在确认单上采纳了这次要生成的镜头数和成本（第一次调"
                        "不要传）。只有人采纳过才生效；镜头数变多、档位 / 参考图 / 画幅变了"
                        "要重新确认",
                    },
                    "skip_shots": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "这次不要的镜头序号（1 起），缺着它们出片。只在用户明确说"
                        "某几镜不要了时传；会当场再问用户一次",
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "description": "画幅，如 16:9 横屏 / 9:16 竖屏。留空用项目设置（/ratio），"
                        "项目没设就按配方",
                    },
                },
                "required": ["brief_id"],
            },
        )

    # ---------- ToolProvider ----------

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        n = len(self._recipes())
        return ProviderHealth(ok=n > 0, detail=f"{n} 种风格" if n else "config/recipes 里没有配方")

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            result = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
            result = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result

    # ---------- 小工具 ----------

    def _recipes(self) -> list[Recipe]:
        return list_recipes(self.recipes_dir)

    def _recipe(self, name: str) -> tuple[Recipe | None, str]:
        try:
            return load_recipe(name, self.recipes_dir), ""
        except FileNotFoundError as e:
            return None, str(e)

    def _brief(self, brief_id: str) -> tuple[Brief | None, Any, str]:
        try:
            asset = self.store.get(brief_id)
            data = json.loads(self.store.content(brief_id))
        except KeyError:
            return None, None, f"没有简报 {brief_id}"
        except json.JSONDecodeError:
            return None, None, f"{brief_id} 不是选题简报"
        if not isinstance(data, dict) or "shots" not in data:
            return None, None, f"{brief_id} 不是选题简报（要 short_video_brief 的产物）"
        return Brief.from_dict(data), asset, ""

    def _save_brief(self, brief: Brief, parents: list[str], base: str = "") -> Any:
        content = json.dumps(brief.as_dict(), ensure_ascii=False, indent=2)
        gp = {
            "keyword": brief.keyword,
            "style": brief.style,
            "duration": brief.duration,
            "shots": len(brief.shots),
            "real_shots": len(brief.material_needs()),
        }
        if base:
            a = self.store.revise(
                base, content, summary=f"选题简报·{brief.title}", creator="tool:resolve_materials"
            )
            a.gen_params.update(gp)
            return self.store.put(a)
        return self.store.create(
            content,
            type_=AssetType.OUTLINE,
            summary=f"选题简报·{brief.title}",
            parents=parents,
            creator="tool:short_video_brief",
            gen_params=gp,
        )

    async def _progress(self, stage: str, done: int, total: int, item: str = "") -> None:
        if self.bus is not None:
            await self.bus.emit(
                EventType.BATCH_PROGRESS, stage=stage, done=done, total=total, item=item
            )

    def _cfg(self, key: str, default: Any) -> Any:
        return (getattr(self.catalog, "drama", {}) or {}).get(key, default)

    def _subtitle_gate(self) -> tuple[bool, int]:
        """(开没开, 发现字后重生成几次)。和短剧共用 media_models.yaml drama.subtitle_gate /
        subtitle_retries；没有文本网关（脚本 / 测试）就关。"""
        if self.gateway is None:
            return False, 0
        on = str(self._cfg("subtitle_gate", "true")).strip().lower() in ("1", "true", "yes", "on")
        try:
            n = max(0, int(self._cfg("subtitle_retries", 1)))
        except (TypeError, ValueError):
            n = 1
        return on, n

    async def _burned_text(self, asset_id: str) -> tuple[bool | None, str]:
        """抽几帧看画面里有没有叠上去的字。返回 (发现了没有, 说明)；查不了返回 (None, 原因)。

        2026-09-23 审查：抽帧查字幕之前只有短剧链有 —— 画面里不许有字是用户定的最高优先级，
        短视频镜头照样会被模型画上字幕条、标题。
        """
        try:
            local = local_copy(self.store.get(asset_id))
        except KeyError:
            local = None
        if local is None:
            return None, "片段没有本地副本"
        tmp = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="svsub_"))
        try:
            paths = await ffmpeg.extract_frames(local, tmp, count=4, width=512)
            frames = await asyncio.to_thread(lambda: [p.read_bytes() for p in paths])
        except Exception:  # noqa: BLE001
            frames = []
        finally:
            await asyncio.to_thread(shutil.rmtree, tmp, True)
        if not frames:
            return None, "抽不出画面帧（检查 ffmpeg）"
        urls = ["data:image/jpeg;base64," + base64.b64encode(b).decode() for b in frames]
        note = ""
        for _ in range(2):  # 判读输出不是 JSON：再问一次，还不行才算查不成
            try:
                resp = await self.gateway.chat(REALISM_ROLE, subtitle_check_messages(urls))
            except Exception as e:  # noqa: BLE001
                return None, f"字幕检查调用失败：{type(e).__name__}"
            found, note = parse_subtitle_verdict(resp.text)
            if found is not None:
                return found, note
        return None, note or "字幕检查没给出结论"

    def _tier_note(self, vtier: str, explicit: bool) -> tuple[str, str]:
        """(锁定的视频模型, 档位没生效的说明)。

        会话锁着视频模型时 gen_video 不按 prefer 选型 —— 之前 tier 参数和配方的 video_tier
        悄悄不起作用，确认单上还写着「档位 fast」（2026-09-23 审查）。"""
        try:
            locked = str(self.video_lock_source() or "") if self.video_lock_source else ""
        except Exception:  # noqa: BLE001
            locked = ""
        if not locked:
            return "", ""
        get = getattr(self.catalog, "get", None)
        spec = get(MediaKind.VIDEO, locked) if callable(get) else None
        tier = str(getattr(spec, "tier", "") or "")
        if not tier or tier == vtier:
            return locked, ""
        origin = "" if explicit else "（配方默认）"
        return locked, (
            f"档位 {vtier}{origin} 没生效：本会话视频模型锁定为 {locked}（{tier} 档），"
            "镜头都用它生成；要换档得先征得用户同意换模型"
        )

    # ---------- 风格 ----------

    async def _fn_list_video_styles(self) -> ToolResult:
        rs = self._recipes()
        if not rs:
            return ToolResult(ok=False, error="config/recipes 里没有配方")
        lines = ["可选风格（配方名 · 适合什么 · 时长范围 · 画面来源）："]
        foot = {"generate": "画面靠 AI 生成", "real": "实拍为主", "mixed": "实拍与生成混用"}
        for r in rs:
            lo, hi = r.duration_bounds
            fo = foot.get(str(r.style.get("footage") or "generate"), "画面靠 AI 生成")
            lines.append(f"  {r.key} · {r.style_label}：{r.style_desc} · {lo}–{hi}s · {fo}")
        lines.append(
            "让用户确认一种，再去抓热点（douyin_hot_rpa / xhs_collect / browse_and_copy）。"
        )
        return ToolResult(content="\n".join(lines))

    # ---------- 简报 ----------

    def _sources_text(self, ids: list[str]) -> tuple[str, list[str], list[str]]:
        """把热点资产拼成材料文本。返回 (文本, 用到的 id, 找不到的 id)。"""
        parts: list[str] = []
        used: list[str] = []
        missing: list[str] = []
        budget = _MAX_SOURCE_CHARS
        for aid in ids:
            try:
                a = self.store.get(aid)
                body = self.store.content(aid)
            except KeyError:
                missing.append(aid)
                continue
            piece = body.strip()[: max(1000, budget // max(1, len(ids)))]
            kw = str((a.gen_params or {}).get("keyword") or "")
            if kw:
                scope = f" · 按关键词「{kw}」搜的"
            elif a.creator in ("tool:douyin_hot_rpa", "tool:douyin_hot_list"):
                scope = " · 全站热榜，不一定和关键词相关"
            else:
                scope = ""
            parts.append(f"### 来源 {a.summary or aid}（{aid}）{scope}\n{piece}")
            used.append(aid)
            budget -= len(piece)
            if budget <= 0:
                break
        return "\n\n".join(parts), used, missing

    async def _ground(self, recipe: Recipe) -> tuple[str, str]:
        """配方要求接热榜（grounding.enabled）而模型没先抓热点：自动拉一次抖音热榜。
        返回 (热榜资产 id, 没拉到的原因)。

        之前 grounding 只有命令行 `agent video make` 在读，对话里模型跳过抓热点，简报就只
        提醒一句「内容只能按常识写」照样出（2026-09-26 审查）—— 配方里写着「上一版效果差的
        根因就是热点是编的」。拉不到不拦：热榜是加分项，但要明说降级了。"""
        if self.registry is None:
            return "", "没接工具注册表"
        g = recipe.grounding
        try:
            args = {
                "count": int(g.get("top") or 15),
                "days": int(g.get("days") or 1),
                "category": str(g.get("category") or ""),
            }
        except (TypeError, ValueError):
            args = {"count": 15, "days": 1, "category": ""}
        hot = await self.registry.invoke("douyin_hot_list", args)
        if not hot.ok or not str(hot.content or "").strip():
            return "", (hot.error or "热榜返回空")[:120]
        if hot.asset_ref:
            return str(hot.asset_ref), ""
        a = self.store.create(hot.content, summary="抖音热榜", creator="tool:douyin_hot_list")
        return a.id, ""

    async def _fn_short_video_brief(
        self,
        keyword: str,
        style: str,
        sources: list[str] | None = None,
        notes: str = "",
        grounding: bool | None = None,
    ) -> ToolResult:
        if not keyword.strip():
            return ToolResult(ok=False, error="keyword 是空的")
        recipe, err = self._recipe(style)
        if recipe is None:
            return ToolResult(ok=False, error=err)
        if self.gateway is None:
            return ToolResult(ok=False, error="没有文本网关，写不了简报")
        sources = list(sources or [])
        ground_note = ""
        if not sources and recipe.grounded and grounding is not False:
            hot_id, why = await self._ground(recipe)
            if hot_id:
                sources.append(hot_id)
                ground_note = "配方要求接热榜，已自动拉了一次抖音热榜" + (
                    f"（{recipe.grounding.get('category')} 类）"
                    if recipe.grounding.get("category") else ""
                )
            else:
                ground_note = f"配方要求接热榜，但热榜没拉到（{why}）"
        text, used, missing = self._sources_text(sources)
        pick_hint = str(recipe.grounding.get("pick_hint") or "").strip()
        if used and pick_hint:
            # 配方的选题要求（ugc-vlog「选一件小事」这种）：之前只有老链的选题那步在用
            notes = f"{notes}\n选题要求：{pick_hint}".strip()
        lo, hi = recipe.duration_bounds
        voiceover = bool(recipe.voiceover.get("enabled"))
        prompt = brief_prompt(
            keyword, recipe.style_label, recipe.style_desc, text, lo, hi, notes,
            prompt_hint=str(recipe.shots.get("prompt_hint") or ""),
            script_hint=str(recipe.voiceover.get("script_hint") or ""),
            voiceover=voiceover,
            shot_seconds=recipe.cut_max or 3.0,
        )
        try:
            resp = await self.gateway.chat(PLANNER_ROLE, [{"role": "user", "content": prompt}])
        except KeyError:
            resp = await self.gateway.chat("main_agent", [{"role": "user", "content": prompt}])
        brief, warns = parse_brief(
            resp.text, keyword, recipe.key, lo, hi, voiceover=voiceover
        )
        if brief is None:
            return ToolResult(ok=False, error="简报解析失败：" + "；".join(warns))
        brief.sources = used
        if ground_note and not used:
            warns.insert(0, ground_note)
        if not used:
            warns.append(
                "没有热点材料，内容只能按常识写 —— 数字和事实都没有出处，建议先抓热点再重跑"
            )
        if missing:
            warns.append(f"这些来源资产不存在：{', '.join(missing)}")
        asset = self._save_brief(brief, parents=used)
        out = brief.render()
        if ground_note and used:
            out = f"（{ground_note}）\n\n{out}"
        if warns:
            out += "\n\n⚠ " + "\n⚠ ".join(warns)
        needs = brief.material_needs()
        nxt = (
            f"request_materials(brief_id=\"{asset.id}\") 向用户要素材"
            if needs
            else f"short_video_produce(brief_id=\"{asset.id}\") 出片"
        )
        content = f"{out}\n\n简报资产 {asset.id}。下一步：{nxt}"
        # 配方 review.after_script：文案和分镜出来先给人看（之前这个字段没有代码读）。
        # 不算大节点：/auto 下自动过，花钱前的出片确认单照样会停
        if recipe.review.get("after_script"):
            return ToolResult(
                content=content + "\n\n已暂停：这个风格要求文案和分镜先给用户看。用户采纳再往下走，"
                "打回就按意见改（带 notes 重新 short_video_brief）。",
                asset_ref=asset.id,
                suspend=True,
                suspend_payload={
                    "question": f"「{brief.title}」的文案和分镜，看一下方向对不对：\n{out}",
                    "stage": SCRIPT_STAGE,
                    "target": SCRIPT_STAGE,
                    "assets": [asset.id],
                    "major": False,
                },
            )
        return ToolResult(content=content, asset_ref=asset.id)

    # ---------- 素材：问人 / 读回复 ----------

    async def _fn_request_materials(
        self, brief_id: str, shots: list[int] | None = None, note: str = ""
    ) -> ToolResult:
        brief, _, err = self._brief(brief_id)
        if brief is None:
            return ToolResult(ok=False, error=err)
        needs = brief.material_needs()
        if shots:
            needs = [(i, s) for i, s in needs if i in set(shots)]
        if not needs:
            return ToolResult(
                content="这份简报没有需要实拍/外部素材的镜头，直接 short_video_produce 即可"
            )
        question = materials_question(brief, needs)
        if note:
            question += f"\n\n{note}"
        payload = {
            "question": question,
            "stage": "素材",
            "target": "素材",
            "assets": [brief_id],
            "materials": [
                {"shot": i, "need": s.real_need, "search_terms": s.search_terms} for i, s in needs
            ],
            "major": True,  # /auto 模式下也要停：素材只能人给
        }
        return ToolResult(
            content=f"已向用户请求 {len(needs)} 个镜头的素材（挂起等待回复）\n{question}",
            suspend=True,
            suspend_payload=payload,
        )

    async def _fn_resolve_materials(self, brief_id: str, reply: str) -> ToolResult:
        brief, asset, err = self._brief(brief_id)
        if brief is None:
            return ToolResult(ok=False, error=err)
        needs = brief.material_needs()
        indices = [i for i, _ in needs]
        parsed = parse_material_reply(reply, indices)
        if not parsed:
            return ToolResult(
                ok=False,
                error="没读懂回复。请按「第3镜 E:\\素材\\x.mp4」「第3镜 联网找」「第3镜 生成」"
                "的格式，或「都联网找」「都生成」",
            )
        lines: list[str] = []
        online: list[tuple[int, list[str]]] = []
        for i, (how, arg) in parsed.items():
            shot = brief.shots[i - 1]
            if how == "generate":
                shot.source = "generate"
                lines.append(f"  第{i}镜：改为 AI 生成")
            elif how == "online":
                terms = shot.search_terms or [shot.desc]
                online.append((i, terms))
                lines.append(f"  第{i}镜：联网找 —— 搜索词 {', '.join(terms)}")
            else:
                aid, why = await self._import_material(arg, i, shot.desc)
                if not aid:
                    lines.append(f"  第{i}镜：文件没登记成功 —— {why}")
                    continue
                brief.materials[str(i)] = aid
                lines.append(f"  第{i}镜：已登记 {aid}（{arg}）")
        unanswered = [i for i in indices if i not in parsed and str(i) not in brief.materials]
        new = self._save_brief(brief, parents=[brief_id], base=brief_id)
        out = "素材处理：\n" + "\n".join(lines)
        if online:
            out += (
                "\n\n联网找的镜头：先 stock_media_search(query=搜索词, kind=video) 挑一条，"
                "fetch_stock_media 下载拿到资产 id，再 short_video_produce 时通过 materials 传入。"
            )
        if unanswered:
            left = ", ".join(map(str, unanswered))
            out += f"\n\n未答复的镜头：{left} —— 出片时会改用 AI 生成并标出"
        return ToolResult(content=f"{out}\n\n简报已更新：{new.id}", asset_ref=new.id)

    async def _import_material(self, path: str, shot_no: int, desc: str) -> tuple[str, str]:
        """用户贴的路径 → 资产 id。走 fs_import 同一套边界。"""
        if self.files is not None:
            r = await self.files.invoke(
                "fs_import", {"path": path, "summary": f"第{shot_no}镜素材·{desc[:20]}"}
            )
            return (r.asset_ref or "", "") if r.ok else ("", r.error or "登记失败")
        return await asyncio.to_thread(self._import_fallback, path, shot_no)

    def _import_fallback(self, path: str, shot_no: int) -> tuple[str, str]:
        """没接文件工具（脚本/测试）时的直接登记。"""
        p = Path(path).expanduser()
        if not p.exists():
            return "", f"{path} 不存在"
        ext = p.suffix.lower()
        type_ = AssetType.IMAGE if ext in (".png", ".jpg", ".jpeg", ".webp") else AssetType.VIDEO
        a = self.store.create(
            "", type_=type_, summary=f"第{shot_no}镜素材", creator="human:import",
            gen_params={"local": str(p)},
        )
        a.uri = str(p)
        self.store.put(a)
        return a.id, ""

    # ---------- 出片 ----------

    async def _fn_short_video_produce(
        self,
        brief_id: str,
        materials: dict[str, str] | None = None,
        tier: str = "",
        out_dir: str = "",
        filename: str = "",
        reuse: bool = True,
        no_voiceover: bool = False,
        no_subtitle: bool = False,
        voice: str = "",
        aroll: str = "",
        ref_images: list[str] | None = None,
        shot_refs: dict[str, list[str]] | None = None,
        confirm: bool = False,
        aspect_ratio: str = "",
        skip_shots: list[int] | None = None,
    ) -> ToolResult:
        brief, _, err = self._brief(brief_id)
        if brief is None:
            return ToolResult(ok=False, error=err)
        if self.registry is None:
            return ToolResult(ok=False, error="没接工具注册表，出不了片")
        recipe, err = self._recipe(brief.style)
        if recipe is None:
            return ToolResult(ok=False, error=err)
        vtier = tier or str(recipe.models.get("video_tier", "fast"))
        # 画幅：这次指定的 > 项目设置（/ratio）> 配方（2026-09-25 用户要的：比例可以个性化选）
        ratio = (
            parse_aspect(aspect_ratio)
            or self.aspect_ratio
            or str(recipe.output.get("aspect_ratio") or DEFAULT_ASPECT)
        )
        mapping = {**brief.materials, **{str(k): v for k, v in (materials or {}).items()}}
        notes: list[str] = []
        locked, tier_note = self._tier_note(vtier, explicit=bool(tier))
        if tier_note:
            notes.append(tier_note)

        # ---- 出镜素材（口播出镜）：它的原声是主音轨，字幕从它转写，B-roll 穿插在它上面 ----
        if aroll:
            why = await self._aroll_problem(aroll)
            if why:
                return ToolResult(ok=False, error=why, meta={"charged": False})

        # ---- 镜头数不能多于刀数：多出来的镜头生成了也进不了成片（2026-09-23 审查：8 个镜头
        #      只切 6 刀，第 7、8 镜付了钱不出现）。出镜模式的时间线跟着出镜素材走，不在此列 ----
        shots = list(enumerate(brief.shots, 1))
        # 用户明确不要的镜头（skip_shots 经闸门问过人）：这次不生成、不进成片
        skipped: set[int] = set()
        for s in skip_shots or []:
            try:
                skipped.add(int(s))
            except (TypeError, ValueError):
                continue
        if skipped:
            shots = [(i, s) for i, s in shots if i not in skipped]
            notes.append(f"按用户要求去掉第 {'、'.join(map(str, sorted(skipped)))} 镜")
        cut_min = max(0.5, float(recipe.cut_min or 1.2))
        room = max(1, math.floor(brief.duration / cut_min + 1e-9))
        if not aroll and len(shots) > room:
            cut = ", ".join(str(i) for i, _ in shots[room:])
            notes.append(
                f"简报有 {len(shots)} 个镜头，{brief.duration}s 按每刀 ≥{cut_min:g}s 最多切 "
                f"{room} 刀 —— 第 {cut} 镜进不了成片，没有生成"
            )
            shots = shots[:room]

        # ---- 每个镜头：用户/素材站的素材 → 复用上次生成的 → 现在生成 ----
        # 复用只认同样的生成条件：参考图（产品身份锁）、档位一样才复用 —— 之前第一次没带
        # 产品图、第二次带上再调，旧的无产品片段全部复用，连确认单都不弹（2026-09-24 审查）
        gen_cond = {
            # 锁着视频模型时档位不起作用（镜头都用锁定的模型）：按实际用的模型比，别因为换了个
            # 不生效的档位就把能复用的片段重新付费生成
            "model": locked,
            "tier": "" if locked else vtier,
            "aspect": ratio,  # 画幅不同的旧片段不复用
            "refs": sorted(str(x).strip() for x in (ref_images or []) if str(x).strip()),
            "shot_refs": {
                str(k): sorted(str(x).strip() for x in v if str(x).strip())
                for k, v in (shot_refs or {}).items() if v
            },
        }
        prev = self._previous_clips(brief_id, gen_cond) if reuse else {}
        if reuse and not prev and self._previous_clips(brief_id):
            notes.append("上次出片的生成条件不同（参考图 / 档位），旧片段没有复用")
        clips: dict[int, str] = {}
        to_generate: list[int] = []
        by_aroll: set[int] = set()  # 出镜素材本身就是这些镜头，不算缺
        for i, shot in shots:
            mat = mapping.get(str(i))
            if mat:
                cid, why = await self._material_clip(
                    mat, shot.seconds or recipe.seconds_each, i, size=still_size(ratio)
                )
                if cid:
                    clips[i] = cid
                    continue
                notes.append(f"第{i}镜素材 {mat} 用不了（{why}），改用 AI 生成")
            elif shot.source == "real":
                if aroll:
                    by_aroll.add(i)
                    continue  # 真人出镜的镜头就是出镜素材本身，不另生成
                need = shot.real_need or shot.desc
                notes.append(f"第{i}镜需要实拍素材但没提供（{need}），已改用 AI 生成")
            if i in prev:
                clips[i] = prev[i]
            else:
                to_generate.append(i)

        # ---- 参考图（广告的产品身份锁）：换成公网链接，生成时逐镜带上 ----
        common, err = await self._ref_urls(ref_images or [])
        if err:
            return ToolResult(ok=False, error=err, meta={"charged": False})
        per_shot: dict[str, list[str]] = {}
        for k, ids in (shot_refs or {}).items():
            urls, err = await self._ref_urls(list(ids or []))
            if err:
                return ToolResult(ok=False, error=err, meta={"charged": False})
            per_shot[str(k)] = urls

        # ---- 出片前报成本，人点头了再花钱（2026-09-23 审查：之前不报镜头数也不预估调用次数）----
        # confirm=true 只认人在确认单上的采纳（2026-09-26 审查：之前模型第一次就带上它，
        # 确认单根本不出现）；生成条件变了、镜头变多了也要重新确认
        plan = {
            "shots": sorted(to_generate),
            "gen": json.dumps(gen_cond, sort_keys=True, ensure_ascii=False),
        }
        if to_generate and not (confirm and self._take_confirmation(brief_id, plan)):
            secs = len(to_generate) * recipe.seconds_each
            q = (
                f"「{brief.title}」要新生成 {len(to_generate)} 段视频"
                f"（第 {'、'.join(map(str, sorted(to_generate)))} 镜，"
                f"每段 {recipe.seconds_each}s，共约 {secs}s，画幅 {ratio}，"
                + (f"模型 {locked}（会话锁定）" if locked else f"档位 {vtier}")
                + "），"
                f"复用 / 用素材 {len(clips)} 段"
                + ("，并出一次配音" if recipe.voiceover.get("enabled") else "")
                + "。确认就开始生成。"
            )
            if brief.script:
                head = " ".join(brief.script.split())
                q += f"\n口播：{head[:120]}{'…' if len(head) > 120 else ''}"
            self._pending_confirm[brief_id] = plan
            unconfirmed = (
                "\n（这次带了 confirm=true，但用户还没在这张确认单上采纳过 —— 没经过人确认的 "
                "confirm 不算数。）"
                if confirm else ""
            )
            return ToolResult(
                content=(
                    f"{q}\n已暂停等人确认。人采纳后用同样的参数加 confirm=true 再调一次 "
                    "short_video_produce；打回就按人的意见改简报。"
                    + unconfirmed
                    + ("\n" + "\n".join(f"  · {n}" for n in notes) if notes else "")
                ),
                suspend=True,
                suspend_payload={
                    "question": q,
                    "stage": CONFIRM_STAGE,
                    "target": CONFIRM_STAGE,
                    "assets": [brief_id],
                    "major": True,
                },
                meta={"charged": False},
            )

        # ---- 生成：并发 + 网络类失败重试 ----
        failed: list[str] = []
        if to_generate:
            level = norm_level(self._cfg("realism_level", ""))
            try:
                retries = max(0, int(self._cfg("video_retries", 1)))
            except (TypeError, ValueError):
                retries = 1
            limit = self.catalog.max_concurrency("video") if self.catalog is not None else 0
            sub_gate, sub_retries = self._subtitle_gate()
            unchecked: list[str] = []
            done = 0
            await self._progress("生成短视频素材", 0, len(to_generate))

            async def gen(i: int) -> tuple[int, str, str]:
                nonlocal done
                shot = brief.shots[i - 1]
                refs = per_shot.get(str(i)) or common
                prompt = recipe.shot_prompt(shot.desc, level)
                if refs:
                    prompt = _REF_LOCK + prompt
                args: dict[str, Any] = {
                    "prompt": prompt,
                    "prefer": vtier,
                    "aspect_ratio": ratio,
                    "duration": recipe.seconds_each,
                    "resolution": recipe.output.get("resolution", "720p"),
                    "summary": f"{brief.title}·第{i}镜",
                    "local_name": recipe_shot_name(brief.title, i),
                }
                if refs:
                    args["image"] = refs  # 产品图是身份锁：之前 produce 根本没有传参考图的通道
                r = None
                for attempt in range(retries + 1):
                    r = await self.registry.invoke("gen_video", args)
                    # 只有「请求没送到」才原样重提；轮询失败/超时时任务在服务端照跑、
                    # 已计费，重提就是付两份钱（2026-09-23 审查）
                    if r.ok or attempt >= retries or not retryable_failure(r):
                        break
                    await asyncio.sleep(3)
                # 画面里不许有字：抽帧查，有字就带着「上一版出了字」重生成；还有就不进成片
                if sub_gate and r is not None and r.ok and r.asset_ref:
                    found, where = await self._burned_text(r.asset_ref)
                    regen_err = ""
                    for _ in range(sub_retries):
                        if not found:
                            break
                        again = await self.registry.invoke(
                            "gen_video", {**args, "prompt": no_text_retry(args["prompt"])}
                        )
                        if not again.ok or not again.asset_ref:
                            regen_err = (again.error or "未知原因")[:80]
                            break
                        r = again
                        found, where = await self._burned_text(r.asset_ref)
                    if found is None:
                        unchecked.append(f"第{i}镜（{where}）")
                    elif found:
                        done += 1
                        await self._progress("生成短视频素材", done, len(to_generate), f"第{i}镜")
                        # 重生成没成功时如实说（之前一律报「重生成后仍有」）；有字的那版留在
                        # 资产库里给人看，但不进成片
                        why = f"重生成失败（{regen_err}）" if regen_err else "重生成后仍有"
                        return i, "", f"画面里有字（{where or '位置不明'}），{why}，没进成片"
                done += 1
                await self._progress("生成短视频素材", done, len(to_generate), f"第{i}镜")
                return i, (r.asset_ref if r.ok else ""), ("" if r.ok else (r.error or ""))

            for i, aid, e in await _run_parallel(to_generate, gen, limit):
                if aid:
                    clips[i] = aid
                else:
                    failed.append(f"第{i}镜：{e[:90]}")
            if unchecked:
                notes.append(
                    f"{len(unchecked)} 段没做成字幕检查：{'、'.join(sorted(unchecked)[:4])}"
                    " —— 成片里有没有字要人看一眼"
                )
        if not clips and not aroll:
            return ToolResult(ok=False, error="一个镜头都没有：\n" + "\n".join(failed))

        # ---- 缺镜头不合成（2026-09-26 审查，和短剧「缺段不成片」同一条规则）：之前失败的、
        #      画面有字的镜头直接跳过，剩下的镜头拉长了照样拼成片 —— 口播讲到的画面没了，
        #      广告缺了英雄帧，看起来却像做完了。成功的镜头记进出片记录，下次复用只补缺的 ----
        lost = sorted({i for i, _ in shots} - set(clips) - by_aroll)
        if lost:
            order = sorted(clips)
            record = self.store.create(
                json.dumps(
                    {"clips": {str(i): clips[i] for i in order}, "audio": "", "subtitle": "",
                     "aroll": aroll, "composed": "", "gen": gen_cond, "missing": lost},
                    ensure_ascii=False, indent=2,
                ),
                type_=AssetType.STORYBOARD,
                summary=f"短视频出片记录·{brief.title}（未成片）",
                parents=[brief_id],
                creator="tool:short_video_produce",
                gen_params={"clips": len(clips), "generated": len(to_generate),
                            "failed": len(failed), "missing": lost},
            )
            miss = "、".join(map(str, lost))
            body = "\n".join(f"  · {n}" for n in notes)
            return ToolResult(
                content=(
                    f"「{brief.title}」⚠ 未成片：第 {miss} 镜没生成出来，没有合成 —— "
                    "缺镜头拼出来的成片会让人以为做完了。\n"
                    + ("失败：\n  " + "\n  ".join(failed) + "\n" if failed else "")
                    + (f"{body}\n" if body else "")
                    + f"下一步：修好原因后再调一次 short_video_produce（已生成的 {len(clips)} 镜"
                    "复用、只补缺的，会再出一张确认单）；用户明确说缺的不要了，才传 "
                    f"skip_shots=[{', '.join(map(str, lost))}] 缺着出片。"
                    "不要自己改写提示词用 gen_video 补。\n"
                    f"出片记录 {record.id}"
                ),
                asset_ref=record.id,
                meta={"complete": False, "missing": len(lost)},
            )

        # ---- 配音：配方开了才配；「实拍为主」的配方没给出镜素材时用 TTS 兜底，不然成片没声音 ----
        audio_id = ""
        vo_wanted = bool(recipe.voiceover.get("enabled"))
        if not vo_wanted and not aroll and str(recipe.style.get("footage") or "") == "real":
            vo_wanted = True
            notes.append("没有出镜素材：用 TTS 口播兜底（不然成片没有声音）")
        vo_on = vo_wanted and not no_voiceover and bool(brief.script) and not aroll
        if vo_on:
            tts_model = str(recipe.models.get("tts_model") or "").strip()
            picked, speed, why = await self._pick_voice(recipe, brief, voice)
            r = await self.registry.invoke(
                "tts",
                {
                    "text": brief.script,
                    "voice": picked,
                    "speed": speed,
                    "instruct": str(recipe.voiceover.get("instruct") or ""),
                    "summary": f"{brief.title}·口播",
                    # 配方里写了 TTS 模型就用它（之前写了没人读）；留空按目录默认
                    **({"model": tts_model} if tts_model else {}),
                },
            )
            if r.ok:
                audio_id = r.asset_ref
                notes.append(f"配音：{picked} · 语速 {speed:g}" + (f"（{why}）" if why else ""))
            else:
                notes.append(f"配音失败，成片无声：{(r.error or '')[:100]}")

        # ---- 字幕：转写拿时间轴。TTS 的文字回贴原稿；出镜素材按出镜人实际说的 ----
        sub_id = ""
        sub_src = audio_id or aroll
        if recipe.subtitle.get("enabled") and sub_src and not no_subtitle:
            r = await self.registry.invoke(
                "transcribe",
                {"asset_id": sub_src, "format": "srt",
                 "language": str(recipe.subtitle.get("language") or "zh")},
            )
            if r.ok:
                sub_id = r.asset_ref
                if audio_id:
                    try:
                        fixed, changed = align_script(brief.script, self.store.content(sub_id))
                    except KeyError:
                        fixed, changed = "", 0
                    if changed:
                        rev = self.store.revise(
                            sub_id, fixed, summary="字幕·按原稿校正", creator="pipeline:align"
                        )
                        sub_id = rev.id
                        notes.append(f"字幕：按原稿校正 {changed} 条")
                else:
                    notes.append("字幕：按出镜人的原声转写")
            else:
                notes.append(f"字幕失败，成片无字幕：{(r.error or '')[:100]}")

        # ---- 合成：按简报顺序排刀（每镜至少一刀、以最后一镜收尾），画布按配方画幅 ----
        order = sorted(clips)
        ordered = [clips[i] for i in order]
        r = await self.registry.invoke(
            "compose_video",
            {
                "clips": ordered,
                "audio_id": audio_id,
                "subtitle_id": sub_id,
                "out_dir": out_dir,
                "filename": filename or recipe.filename(brief.title, vtier),
                "max_cut_seconds": recipe.cut_max,
                "min_cut_seconds": recipe.cut_min,
                "total_seconds": brief.duration,
                "aroll_id": aroll,
                "cut_order": "brief",
                "weights": [float(brief.shots[i - 1].seconds or 1.0) for i in order],
                "aspect_ratio": ratio,
            },
        )
        record = self.store.create(
            json.dumps(
                {"clips": {str(i): clips[i] for i in order}, "audio": audio_id,
                 "subtitle": sub_id, "aroll": aroll, "composed": r.asset_ref if r.ok else "",
                 "gen": gen_cond},
                ensure_ascii=False, indent=2,
            ),
            type_=AssetType.STORYBOARD,
            summary=f"短视频出片记录·{brief.title}",
            parents=[brief_id],
            creator="tool:short_video_produce",
            gen_params={"clips": len(clips), "generated": len(to_generate), "failed": len(failed)},
        )
        fresh = len(to_generate) - len(failed)
        kept = len(clips) - fresh
        head = (
            f"「{brief.title}」{brief.duration}s · 镜头 {len(clips)} 个"
            f"（新生成 {fresh}，复用/素材 {kept}）" + ("· 出镜素材为主" if aroll else "")
        )
        body = "\n".join(f"  · {n}" for n in notes)
        warn = ("\n\n失败：\n  " + "\n  ".join(failed)) if failed else ""
        if not r.ok:
            return ToolResult(
                content=f"{head}\n{body}{warn}\n\n⚠ 合成失败：{r.error}\n出片记录 {record.id}",
                asset_ref=record.id,
            )
        done_text = (
            f"{head}\n{body}{warn}\n\n{r.content}\n\n出片记录 {record.id}。"
            f"建议 view_video(source=\"{r.asset_ref}\") 看一遍：字幕、变脸、穿帮。"
        )
        # 配方 review.after_compose（产品广告 / 口播出镜）：成片出来先给人看（之前这个字段
        # 没有代码读，「投放级成片必须给用户审」只写在指引里）。major：/auto 也停
        if recipe.review.get("after_compose"):
            return ToolResult(
                content=done_text + "\n\n已暂停：这个风格要求成片先给用户审。用户采纳才算交付；"
                "要叠上屏文字用 overlay_text，打回就按意见改。",
                asset_ref=r.asset_ref,
                suspend=True,
                suspend_payload={
                    "question": f"「{brief.title}」成片出来了，请看一遍（画面、口型、产品细节、"
                    "穿帮）：\n" + r.content.splitlines()[0],
                    "stage": REVIEW_STAGE,
                    "target": REVIEW_STAGE,
                    "assets": [r.asset_ref],
                    "major": True,
                },
            )
        return ToolResult(content=done_text, asset_ref=r.asset_ref)

    async def _aroll_problem(self, asset_id: str) -> str:
        """出镜素材能不能用：要是视频、本地有文件（剪辑和转写都要读它）、有音轨（它的原声
        是整条音轨）。能用返回空串。"""
        try:
            a = self.store.get(asset_id)
        except KeyError:
            return f"出镜素材 {asset_id} 不存在"
        if a.type is not AssetType.VIDEO:
            return f"出镜素材 {asset_id} 是 {a.type.value}，要视频"
        lc = local_copy(a)
        if lc is None:
            return f"出镜素材 {asset_id} 本地没有文件（先 fs_import 登记用户给的视频）"
        # 没音轨的出镜素材：B-roll 生成付了钱才在混音处报错（2026-09-24 审查）。
        # ffprobe 读不出（没装 / 假文件）不拦，交给后面的步骤
        info = await ffmpeg.probe(Path(lc))
        if info.duration > 0 and not info.has_audio:
            return (
                f"出镜素材 {asset_id} 没有音轨 —— 口播出镜要用它的原声当整条音轨。"
                "没声音的实拍片段请当普通素材传 materials，不要传 aroll"
            )
        return ""

    async def _ref_urls(self, ids: list[str]) -> tuple[list[str], str]:
        """参考图 → 公网链接（生成接口只收 http(s)）。已是链接的原样用；本地图走托管。"""
        out: list[str] = []
        for x in ids:
            x = str(x or "").strip()
            if not x:
                continue
            if x.startswith(("http://", "https://")):
                out.append(x)
                continue
            try:
                a = self.store.get(x)
            except KeyError:
                return [], f"参考图 {x} 不存在"
            if self.hosting is not None and getattr(self.hosting, "enabled", False):
                url, err = await self.hosting.ensure_asset(self.store, a, 20)
                if url:
                    # err 只是「链接可能过期」的提醒（有远端链接、没本地副本时）：链接照用，
                    # 之前整次拒绝（2026-09-24 审查）
                    out.append(url)
                    continue
                return [], f"参考图 {x} 托管失败：{err}"
            if (a.uri or "").startswith(("http://", "https://")):
                out.append(str(a.uri))
                continue
            return [], (
                f"参考图 {x} 只有本地文件，生成接口只收公网链接 —— 先 host_file 拿链接，"
                "或在 config/hosting.yaml 配好素材托管"
            )
        return out, ""

    async def _pick_voice(self, recipe: Recipe, brief: Brief, voice: str) -> tuple[str, float, str]:
        want = voice or str(recipe.voiceover.get("voice") or "auto")
        speed = float(recipe.voiceover.get("speed") or 1.0)
        if want.lower() != "auto":
            return want, speed, ""
        voices = [(v.name, v.note) for v in getattr(self.catalog, "speech_voices", [])]
        fallback = str(getattr(self.catalog, "speech_default_voice", "") or "")
        if not voices or self.gateway is None:
            return fallback, speed, ""
        try:
            resp = await self.gateway.chat(
                "voice_select",
                [{"role": "user", "content": voice_prompt(brief.script, voices, brief.title)}],
            )
        except Exception:  # noqa: BLE001
            return fallback, speed, ""
        pick = parse_voice(resp.text, [n for n, _ in voices], fallback, speed)
        return pick.voice, pick.speed, pick.why

    def _previous_clips(
        self, brief_id: str, cond: dict[str, Any] | None = None
    ) -> dict[int, str]:
        """这份简报上次出片时成功的镜头：序号 → 片段资产 id（能拼的才算）。
        cond：只认同样生成条件（参考图 / 档位）的出片记录；老记录没记条件，只在这次
        也没带参考图时才算同样。"""
        out: dict[int, str] = {}
        # resolve_materials 用 revise 出新版简报（新 id）：之前只认 parent_ids[0] == brief_id，
        # 只改了一个镜头的素材，所有 AI 镜头都重新生成（2026-09-23 审查）
        lineage = self._brief_lineage(brief_id)
        for a in self.store.find(creator="tool:short_video_produce"):
            if not a.parent_ids or a.parent_ids[0] not in lineage:
                continue
            try:
                data = json.loads(self.store.content(a.id))
                rows = data.get("clips") or {}
            except (KeyError, json.JSONDecodeError, AttributeError):
                continue
            if cond is not None:
                gen = data.get("gen") if isinstance(data, dict) else None
                if gen is None:
                    # 老记录没记条件：都是配方默认的竖屏、没带参考图
                    if cond.get("refs") or cond.get("shot_refs") or (
                        cond.get("aspect", DEFAULT_ASPECT) != DEFAULT_ASPECT
                    ):
                        continue
                elif gen != cond:
                    continue
            for k, cid in rows.items():
                try:
                    i = int(k)
                except ValueError:
                    continue
                if i not in out and self._playable(cid):
                    out[i] = cid
        return out

    def _brief_lineage(self, brief_id: str) -> set[str]:
        """这份简报和它 revise 出来之前的各版（只沿 resolve_materials 的改版链往上走）。"""
        ids = {brief_id}
        cur = brief_id
        for _ in range(50):
            try:
                a = self.store.get(cur)
            except KeyError:
                break
            if a.creator != "tool:resolve_materials" or not a.parent_ids:
                break
            cur = a.parent_ids[0]
            ids.add(cur)
        return ids

    def _playable(self, asset_id: str) -> bool:
        try:
            a = self.store.get(asset_id)
        except KeyError:
            return False
        if local_copy(a) is not None:
            return True
        return (a.uri or "").startswith(("http://", "https://"))

    async def _material_clip(
        self, asset_id: str, seconds: float, shot_no: int, size: tuple[int, int] = (720, 1280)
    ) -> tuple[str, str]:
        """素材资产 → 可拼的视频片段 id。视频直接用；图片做成静止镜头。"""
        try:
            a = self.store.get(asset_id)
        except KeyError:
            return "", "资产不存在"
        if a.type is AssetType.VIDEO:
            return (asset_id, "") if self._playable(asset_id) else ("", "视频文件不在了")
        if a.type is not AssetType.IMAGE:
            return "", f"类型是 {a.type.value}，不是视频或图片"
        lc = local_copy(a)
        src = str(lc) if lc is not None else (a.uri or "")
        if not src or src.startswith(("http://", "https://")):
            return "", "图片没有本地文件"
        out_dir = (
            self.output.dir_for("videos")
            if self.output is not None and hasattr(self.output, "dir_for")
            else self.store.blob_dir
        )
        out = Path(out_dir) / f"{safe_name(a.summary or asset_id)}_镜{shot_no:02d}.mp4"
        # 多做 3 秒（全局快切上限）：排刀在素材里错开取，素材刚好等长时取不满、成片缩水、
        # 配音结尾被截（2026-09-24 审查）；静帧多做几秒不花钱
        ok, why = await ffmpeg.still_to_clip(Path(src), out, max(2.0, seconds) + 3.0, size=size)
        if not ok:
            return "", f"图片转镜头失败：{why}"
        clip = self.store.create(
            "",
            type_=AssetType.VIDEO,
            summary=f"第{shot_no}镜·图片镜头",
            parents=[asset_id],
            creator="tool:short_video_produce",
            gen_params={"local": str(out), "from_image": asset_id},
        )
        clip.uri = str(out)
        clip.mime = "video/mp4"
        self.store.put(clip)
        return clip.id, ""

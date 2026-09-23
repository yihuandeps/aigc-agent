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
import json
import time
from pathlib import Path
from typing import Any

from ...harness.events.bus import EventBus, EventType
from ...harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
    ToolSpec,
)
from ..assets.store import AssetStore, AssetType, local_copy
from ..media import ffmpeg
from ..media.naming import recipe_shot_name, safe_name
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
from ..realism import norm_level
from .drama import _run_parallel
from .media import retryable_failure

PLANNER_ROLE = "short_video_planner"
_MAX_SOURCE_CHARS = 20_000  # 喂给规划模型的热点材料总上限


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
    ) -> None:
        self.gateway = gateway
        self.store = store
        self.registry = registry
        self.catalog = catalog
        self.bus = bus
        self.output = output  # OutputPrefs：图转镜头等中间产物落哪
        self.files = files  # FileFunctions：用户贴的素材路径按同一套边界解析
        self.recipes_dir = recipes_dir
        self._specs: dict[str, ToolSpec] = {}
        self._build()

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
            cost_kind="text",
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
            cost_kind="video",
            timeout=7200,
            description=(
                "素材都定了再调。materials 传 {镜头序号: 资产id}（用户给的或素材站下的；"
                "图片会自动做成静止镜头），没传素材的实拍镜头会改用 AI 生成并在结果里标出来。"
                "画面按风格配方的快切规则剪，长度以配音为准。跑完用 view_video 看一遍成片。"
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
            parts.append(f"### 来源 {a.summary or aid}（{aid}）\n{piece}")
            used.append(aid)
            budget -= len(piece)
            if budget <= 0:
                break
        return "\n\n".join(parts), used, missing

    async def _fn_short_video_brief(
        self, keyword: str, style: str, sources: list[str] | None = None, notes: str = ""
    ) -> ToolResult:
        if not keyword.strip():
            return ToolResult(ok=False, error="keyword 是空的")
        recipe, err = self._recipe(style)
        if recipe is None:
            return ToolResult(ok=False, error=err)
        if self.gateway is None:
            return ToolResult(ok=False, error="没有文本网关，写不了简报")
        text, used, missing = self._sources_text(sources or [])
        lo, hi = recipe.duration_bounds
        prompt = brief_prompt(keyword, recipe.style_label, recipe.style_desc, text, lo, hi, notes)
        try:
            resp = await self.gateway.chat(PLANNER_ROLE, [{"role": "user", "content": prompt}])
        except KeyError:
            resp = await self.gateway.chat("main_agent", [{"role": "user", "content": prompt}])
        brief, warns = parse_brief(resp.text, keyword, recipe.key, lo, hi)
        if brief is None:
            return ToolResult(ok=False, error="简报解析失败：" + "；".join(warns))
        brief.sources = used
        if not used:
            warns.append("没有热点材料，内容只能按常识写 —— 建议先抓热点再重跑")
        if missing:
            warns.append(f"这些来源资产不存在：{', '.join(missing)}")
        asset = self._save_brief(brief, parents=used)
        out = brief.render()
        if warns:
            out += "\n\n⚠ " + "\n⚠ ".join(warns)
        needs = brief.material_needs()
        nxt = (
            f"request_materials(brief_id=\"{asset.id}\") 向用户要素材"
            if needs
            else f"short_video_produce(brief_id=\"{asset.id}\") 出片"
        )
        return ToolResult(
            content=f"{out}\n\n简报资产 {asset.id}。下一步：{nxt}", asset_ref=asset.id
        )

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
        mapping = {**brief.materials, **{str(k): v for k, v in (materials or {}).items()}}
        notes: list[str] = []

        # ---- 每个镜头：用户/素材站的素材 → 复用上次生成的 → 现在生成 ----
        prev = self._previous_clips(brief_id) if reuse else {}
        clips: dict[int, str] = {}
        to_generate: list[int] = []
        for i, shot in enumerate(brief.shots, 1):
            mat = mapping.get(str(i))
            if mat:
                cid, why = await self._material_clip(mat, shot.seconds or recipe.seconds_each, i)
                if cid:
                    clips[i] = cid
                    continue
                notes.append(f"第{i}镜素材 {mat} 用不了（{why}），改用 AI 生成")
            elif shot.source == "real":
                need = shot.real_need or shot.desc
                notes.append(f"第{i}镜需要实拍素材但没提供（{need}），已改用 AI 生成")
            if i in prev:
                clips[i] = prev[i]
            else:
                to_generate.append(i)

        # ---- 生成：并发 + 网络类失败重试 ----
        failed: list[str] = []
        if to_generate:
            level = norm_level(self._cfg("realism_level", ""))
            try:
                retries = max(0, int(self._cfg("video_retries", 1)))
            except (TypeError, ValueError):
                retries = 1
            limit = self.catalog.max_concurrency("video") if self.catalog is not None else 0
            done = 0
            await self._progress("生成短视频素材", 0, len(to_generate))

            async def gen(i: int) -> tuple[int, str, str]:
                nonlocal done
                shot = brief.shots[i - 1]
                args = {
                    "prompt": recipe.shot_prompt(shot.desc, level),
                    "prefer": vtier,
                    "aspect_ratio": recipe.output.get("aspect_ratio", "9:16"),
                    "duration": recipe.seconds_each,
                    "resolution": recipe.output.get("resolution", "720p"),
                    "summary": f"{brief.title}·第{i}镜",
                    "local_name": recipe_shot_name(brief.title, i),
                }
                r = None
                for attempt in range(retries + 1):
                    r = await self.registry.invoke("gen_video", args)
                    # 只有「请求没送到」才原样重提；轮询失败/超时时任务在服务端照跑、
                    # 已计费，重提就是付两份钱（2026-09-23 审查）
                    if r.ok or attempt >= retries or not retryable_failure(r):
                        break
                    await asyncio.sleep(3)
                done += 1
                await self._progress("生成短视频素材", done, len(to_generate), f"第{i}镜")
                return i, (r.asset_ref if r.ok else ""), ("" if r.ok else (r.error or ""))

            for i, aid, e in await _run_parallel(to_generate, gen, limit):
                if aid:
                    clips[i] = aid
                else:
                    failed.append(f"第{i}镜：{e[:90]}")
        if not clips:
            return ToolResult(ok=False, error="一个镜头都没有：\n" + "\n".join(failed))

        # ---- 配音 ----
        audio_id = ""
        vo_on = bool(recipe.voiceover.get("enabled")) and not no_voiceover and brief.script
        if vo_on:
            picked, speed, why = await self._pick_voice(recipe, brief, voice)
            r = await self.registry.invoke(
                "tts",
                {
                    "text": brief.script,
                    "voice": picked,
                    "speed": speed,
                    "instruct": str(recipe.voiceover.get("instruct") or ""),
                    "summary": f"{brief.title}·口播",
                },
            )
            if r.ok:
                audio_id = r.asset_ref
                notes.append(f"配音：{picked} · 语速 {speed:g}" + (f"（{why}）" if why else ""))
            else:
                notes.append(f"配音失败，成片无声：{(r.error or '')[:100]}")

        # ---- 字幕：转写拿时间轴，文字回贴原稿 ----
        sub_id = ""
        if recipe.subtitle.get("enabled") and audio_id and not no_subtitle:
            r = await self.registry.invoke(
                "transcribe",
                {"asset_id": audio_id, "format": "srt",
                 "language": str(recipe.subtitle.get("language") or "zh")},
            )
            if r.ok:
                sub_id = r.asset_ref
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
                notes.append(f"字幕失败，成片无字幕：{(r.error or '')[:100]}")

        # ---- 合成 ----
        ordered = [clips[i] for i in sorted(clips)]
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
            },
        )
        record = self.store.create(
            json.dumps(
                {"clips": {str(i): clips[i] for i in sorted(clips)}, "audio": audio_id,
                 "subtitle": sub_id, "composed": r.asset_ref if r.ok else ""},
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
            f"（新生成 {fresh}，复用/素材 {kept}）"
        )
        body = "\n".join(f"  · {n}" for n in notes)
        warn = ("\n\n失败：\n  " + "\n  ".join(failed)) if failed else ""
        if not r.ok:
            return ToolResult(
                content=f"{head}\n{body}{warn}\n\n⚠ 合成失败：{r.error}\n出片记录 {record.id}",
                asset_ref=record.id,
            )
        return ToolResult(
            content=f"{head}\n{body}{warn}\n\n{r.content}\n\n出片记录 {record.id}。"
            f"建议 view_video(source=\"{r.asset_ref}\") 看一遍：字幕、变脸、穿帮。",
            asset_ref=r.asset_ref,
        )

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

    def _previous_clips(self, brief_id: str) -> dict[int, str]:
        """这份简报上次出片时成功的镜头：序号 → 片段资产 id（能拼的才算）。"""
        out: dict[int, str] = {}
        for a in self.store.find(creator="tool:short_video_produce"):
            if not a.parent_ids or a.parent_ids[0] != brief_id:
                continue
            try:
                rows = json.loads(self.store.content(a.id)).get("clips") or {}
            except (KeyError, json.JSONDecodeError, AttributeError):
                continue
            for k, cid in rows.items():
                try:
                    i = int(k)
                except ValueError:
                    continue
                if i not in out and self._playable(cid):
                    out[i] = cid
        return out

    def _playable(self, asset_id: str) -> bool:
        try:
            a = self.store.get(asset_id)
        except KeyError:
            return False
        if local_copy(a) is not None:
            return True
        return (a.uri or "").startswith(("http://", "https://"))

    async def _material_clip(self, asset_id: str, seconds: float, shot_no: int) -> tuple[str, str]:
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
            else (self.store.root or Path(".")) / "blobs"
        )
        out = Path(out_dir) / f"{safe_name(a.summary or asset_id)}_镜{shot_no:02d}.mp4"
        ok, why = await ffmpeg.still_to_clip(Path(src), out, max(2.0, seconds))
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

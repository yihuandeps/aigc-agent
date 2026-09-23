"""分镜 functions —— 从 ai-character-passport 移植过来的能力。

四个工具，解决的是同一个问题：**让一条视频里的人物前后是同一个人**。

视频生成模型每次调用都是独立的，同一句"一位工程师"生成四次会得到四张
不同的脸。护照把外观拆成结构化字段逐镜注入，把这件事压到可控。

和现有链路的衔接：breakdown_script 出的 prompt 可以直接喂给 gen_video，
narration 可以直接喂给 tts。
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

from ...harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
    ToolSpec,
)
from ..assets.store import AssetStore, AssetType
from ..media import ffmpeg
from ..storyboard import (
    Character,
    frames_messages,
    parse_shots,
    render_passport,
    script_messages,
)

# 抽多少帧给视觉模型看。再多 token 涨得很快，而节奏和构图 12 帧已经够看。
_DEFAULT_FRAMES = 12
_CREATOR = "tool:character_passport"


class StoryboardFunctions:
    name = "storyboard"
    namespaced = False
    disclosure = "full"

    def __init__(self, gateway: Any, store: AssetStore, workspace: Path) -> None:
        self.gateway = gateway
        self.store = store
        self.workspace = workspace
        self._specs: dict[str, ToolSpec] = {}
        self._build()

    def _build(self) -> None:
        self._specs["save_character"] = ToolSpec(
            name="save_character",
            summary="建一张角色护照（固定外观，保证跨镜头人物一致）",
            permission=PermissionLevel.WRITE,
            description=(
                "把人物外观拆成结构化字段存下来，之后拆分镜时逐镜注入提示词。\n"
                "**不要把外观揉成一段话**：分开写模型自由发挥的空间才小，一致性才稳。\n"
                "同名角色会覆盖旧的。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "角色名，如 林夏"},
                    "base_prompt": {
                        "type": "string",
                        "description": "人物是谁，一句话，如 28岁女程序员，沉默寡言",
                    },
                    "appearance": {
                        "type": "string",
                        "description": (
                            "长什么样，逗号分隔的标签。**一致性的主力，越具体越好**，"
                            "如 短黑发, 圆框眼镜, 灰色连帽衫, 左耳银色耳钉"
                        ),
                    },
                    "style_lighting": {
                        "type": "string",
                        "description": "怎么拍：风格与光影，如 冷蓝夜景, 浅景深, 电影感",
                    },
                    "reference_image": {
                        "type": "string",
                        "description": "参考图的图片资产 id，可省略",
                    },
                    "notes": {"type": "string", "description": "备注，可省略"},
                },
                "required": ["name"],
            },
        )

        self._specs["list_characters"] = ToolSpec(
            name="list_characters",
            summary="列出已有的角色护照",
            permission=PermissionLevel.READ,
            description="拆分镜前先看有哪些角色可用。",
            parameters={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["natural", "tags", "json"],
                        "description": "护照渲染格式，默认 natural",
                    }
                },
            },
        )

        self._specs["breakdown_script"] = ToolSpec(
            name="breakdown_script",
            summary="把剧本拆成分镜（英文画面提示词 + 中文旁白）",
            permission=PermissionLevel.COMPUTE,
            description=(
                "剧本 → 逐镜的画面提示词和旁白。提示词是英文逗号短语，带运镜和光影，"
                "可直接喂给 gen_video；旁白是中文，可直接喂给 tts。\n"
                "**传 character 就会逐镜注入该角色的外貌**，这是跨镜头人物一致的关键。\n"
                "seconds_each 必须和你要调的生成模型对齐（veo3.1 最长 8 秒）——"
                "写大了模型会按更长的信息量编排画面，生成出来是截断的。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "script": {"type": "string", "description": "剧本或视频描述"},
                    "character": {"type": "string", "description": "角色名，用已存的护照"},
                    "seconds_each": {
                        "type": "integer",
                        "description": "单镜时长秒，默认 8（veo3.1 上限）",
                    },
                    "shot_count": {
                        "type": "integer",
                        "description": "要几个镜头。省略则由剧本内容决定",
                    },
                },
                "required": ["script"],
            },
        )

        self._specs["reference_to_shots"] = ToolSpec(
            name="reference_to_shots",
            summary="拆解参考视频的分镜，并把主角换成指定角色",
            permission=PermissionLevel.COMPUTE,
            description=(
                "抽参考视频的关键帧给视觉模型看，模仿它的构图、运镜、节奏，"
                "拆成分镜，同时把画面里的主角换成你指定的角色。\n"
                "用来做「这条爆款的节奏，换成我的人物重拍一遍」。\n"
                "视频要先是本地资产（fetch_asset_file / compose_video 的产物都行）。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "video_id": {"type": "string", "description": "参考视频的资产 id"},
                    "character": {"type": "string", "description": "要换成的角色名"},
                    "frames": {
                        "type": "integer",
                        "description": f"抽几帧给模型看，默认 {_DEFAULT_FRAMES}",
                    },
                    "seconds_each": {"type": "integer", "description": "单镜时长秒，默认 8"},
                },
                "required": ["video_id"],
            },
        )

    # ---------- ToolProvider ----------

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail=f"角色护照 {len(self._roster())} 个")

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            r = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
            r = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        r.duration_ms = int((time.perf_counter() - started) * 1000)
        return r

    # ---------- 角色护照 ----------

    def _roster(self) -> list[Character]:
        """护照存成 TEXT 资产。同名保留最新的那张。"""
        out: dict[str, Character] = {}
        for a in self.store.find(newest_first=False):  # 当前项目（缺口 A）
            if a.creator != _CREATOR:
                continue
            try:
                c = Character.from_dict(json.loads(self.store.content(a.id)))
            except (json.JSONDecodeError, TypeError):
                continue
            if c.name:
                out[c.name] = c
        return list(out.values())

    def _find(self, name: str) -> Character | None:
        if not name:
            return None
        low = name.strip().lower()
        return next((c for c in self._roster() if c.name.strip().lower() == low), None)

    def _no_such(self, name: str) -> ToolResult:
        have = "、".join(x.name for x in self._roster()) or "（一个都没有）"
        return ToolResult(ok=False, error=f"没有名为 {name!r} 的角色护照。已有：{have}")

    async def _fn_save_character(self, name: str, **kw: Any) -> ToolResult:
        c = Character(name=name.strip(), **{k: str(v or "") for k, v in kw.items()})
        if not c.filled:
            return ToolResult(
                ok=False,
                error="护照太空了：至少要有 base_prompt 或 appearance，否则注入进去等于没注入",
            )
        asset = self.store.create(
            json.dumps(c.to_dict(), ensure_ascii=False, indent=2),
            type_=AssetType.TEXT,
            summary=f"角色护照·{c.name}",
            creator=_CREATOR,
        )
        return ToolResult(
            content=f"已保存角色护照「{c.name}」\n\n{c.render('natural')}\n\n资产 {asset.id}",
            asset_ref=asset.id,
        )

    async def _fn_list_characters(self, format: str = "natural") -> ToolResult:  # noqa: A002
        roster = self._roster()
        if not roster:
            return ToolResult(content="还没有角色护照。用 save_character 建一个。")
        body = "\n\n".join(f"【{c.name}】\n{render_passport(c, format)}" for c in roster)
        return ToolResult(content=f"角色护照（{len(roster)} 个）\n\n{body}")

    # ---------- 拆分镜 ----------

    async def _fn_breakdown_script(
        self,
        script: str,
        character: str = "",
        seconds_each: int = 8,
        shot_count: int = 0,
    ) -> ToolResult:
        if not script.strip():
            return ToolResult(ok=False, error="script 是空的")
        c = self._find(character)
        if character and c is None:
            return self._no_such(character)

        resp = await self.gateway.chat(
            "storyboard", script_messages(script, c, seconds_each, shot_count)
        )
        return self._pack(resp.text, c, f"剧本拆解·{len(script)}字")

    async def _fn_reference_to_shots(
        self,
        video_id: str,
        character: str = "",
        frames: int = _DEFAULT_FRAMES,
        seconds_each: int = 8,
    ) -> ToolResult:
        if not ffmpeg.have_ffmpeg():
            return ToolResult(ok=False, error="环境里没有 ffmpeg，抽不了帧")
        c = self._find(character)
        if character and c is None:
            return self._no_such(character)

        path = self._local_video(video_id)
        if path is None:
            return ToolResult(
                ok=False,
                error=f"资产 {video_id!r} 不是本地视频文件。先用 fetch_asset_file 落盘。",
            )

        work = self.workspace / "storyboard" / f"f{int(time.time())}"
        shots = await ffmpeg.extract_frames(path, work, max(1, min(int(frames), 24)))
        if not shots:
            return ToolResult(ok=False, error="一帧都没抽出来，视频可能是坏的")

        urls = [_data_url(f) for f in shots]
        resp = await self.gateway.chat("storyboard", frames_messages(urls, c, seconds_each))
        return self._pack(resp.text, c, f"参考视频拆解·{len(shots)}帧")

    # ---------- 收尾 ----------

    def _pack(self, text: str, c: Character | None, summary: str) -> ToolResult:
        shots, warn = parse_shots(text)
        if not shots:
            return ToolResult(ok=False, error=f"{warn}\n模型原文前 300 字：{text[:300]}")

        body = "\n\n".join(s.line() for s in shots)
        asset = self.store.create(
            json.dumps([s.__dict__ for s in shots], ensure_ascii=False, indent=2),
            type_=AssetType.STORYBOARD,
            summary=f"{summary}·{len(shots)}镜" + (f"·{c.name}" if c else ""),
            creator="tool:breakdown_script",
            gen_params={"character": c.name if c else "", "shots": len(shots)},
        )
        hint = (
            f"\n\n共 {len(shots)} 镜。prompt 可直接喂 gen_video，"
            f"narration 可直接喂 tts。资产 {asset.id}"
        )
        return ToolResult(content=body + hint, asset_ref=asset.id)

    def _local_video(self, asset_id: str) -> Path | None:
        try:
            a = self.store.get(asset_id)
        except KeyError:
            return None
        if a.type != AssetType.VIDEO or not a.uri:
            return None
        p = Path(a.uri)
        return p if p.exists() else None


def _data_url(p: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()

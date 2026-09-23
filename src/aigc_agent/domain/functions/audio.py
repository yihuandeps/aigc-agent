"""语音 functions —— 配音与字幕。

三个 function：
  list_voices   看有哪些音色和 TTS/ASR 模型
  tts           文本 → 配音音频（存为资产）
  transcribe    音频 → 文本 / **srt / vtt 字幕**

短视频链路里这两个是串起来的：
  脚本 --tts--> 配音音频 --transcribe(srt)--> 时间轴对齐的字幕
后者比按字数估算时间轴准得多，因为它是从**真实音频**里对出来的。
function 描述里写了这条，让模型自己会用。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ...harness.model.audio import TRANSCRIPT_FORMATS, AudioGateway
from ...harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
    ToolSpec,
)
from ..assets.store import AssetStore, AssetType
from ..generators.catalog import MediaCatalog

_EXT = {"mp3": ".mp3", "wav": ".wav", "opus": ".opus", "pcm": ".pcm"}
_MIME = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/opus",
    "pcm": "audio/L16",
}


class AudioFunctions:
    name = "audio"
    namespaced = False
    disclosure = "full"

    def __init__(
        self, gateway: AudioGateway, catalog: MediaCatalog, store: AssetStore
    ) -> None:
        self.gateway = gateway
        self.catalog = catalog
        self.store = store
        self._specs: dict[str, ToolSpec] = {}
        self._build()

    # ---------- 声明 ----------

    def _build(self) -> None:
        self._specs["list_voices"] = ToolSpec(
            name="list_voices",
            summary="列出可用的音色、TTS 模型与转写模型",
            permission=PermissionLevel.READ,
            description="配音之前先看这个，按内容调性挑音色。",
            parameters={"type": "object", "properties": {}},
        )

        self._specs["tts"] = ToolSpec(
            name="tts",
            summary="把文本合成为配音音频，存为资产",
            permission=PermissionLevel.COMPUTE,
            cost_kind="audio",
            description=(
                "文本转语音。单次上限 4096 字，长稿要自己分段合成。"
                "instruct 可以指定情绪（如「轻松亲切」「冷静克制」），比只调语速自然得多。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要念的文本，≤4096 字"},
                    "voice": {"type": "string", "description": "音色名，见 list_voices"},
                    "model": {"type": "string", "description": "留空则用默认 TTS 模型"},
                    "language": {"type": "string", "description": "如 Chinese / English / Auto"},
                    "speed": {"type": "number", "description": "0.5-2.0，默认 1.0"},
                    "instruct": {"type": "string", "description": "情绪与语气指令"},
                    "response_format": {
                        "type": "string",
                        "enum": ["mp3", "wav", "opus", "pcm"],
                    },
                    "summary": {"type": "string"},
                    "parent_id": {
                        "type": "string",
                        "description": "念的是哪份脚本，填它的 id 以建立血缘",
                    },
                },
                "required": ["text"],
            },
        )

        self._specs["transcribe"] = ToolSpec(
            name="transcribe",
            summary="把音频转成文本，或直接生成 srt/vtt 字幕",
            permission=PermissionLevel.COMPUTE,
            cost_kind="audio",
            description=(
                "音频转写。**format 选 srt 就直接得到时间轴对齐的字幕文件**——"
                "对刚用 tts 生成的配音做一次转写，比按字数估算时间轴准得多。"
                "输入给 asset_id（本地资产）。单文件上限 25MB。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string", "description": "音频资产 id"},
                    "format": {
                        "type": "string",
                        "enum": list(TRANSCRIPT_FORMATS),
                        "description": "srt/vtt 出字幕，json 出纯文本，verbose_json 带分段时间",
                    },
                    "model": {"type": "string"},
                    "language": {"type": "string", "description": "ISO-639-1，如 zh / en"},
                    "prompt": {"type": "string", "description": "专有名词提示，提高准确率"},
                    "summary": {"type": "string"},
                },
                "required": ["asset_id"],
            },
        )

    # ---------- ToolProvider 协议 ----------

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            ok=True, detail=f"{len(self.catalog.voices)} 个音色 / {len(self.catalog.audio)} 个模型"
        )

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            result = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
            result = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result

    # ---------- 实现 ----------

    async def _fn_list_voices(self) -> ToolResult:
        return ToolResult(content=self.catalog.render_audio())

    async def _fn_tts(
        self,
        text: str,
        voice: str = "",
        model: str = "",
        language: str = "",
        speed: float | None = None,
        instruct: str = "",
        response_format: str = "mp3",
        summary: str = "",
        parent_id: str = "",
    ) -> ToolResult:
        chosen, why = self.catalog.choose_audio(model, kind="tts")
        if not chosen:
            return ToolResult(ok=False, error=why)

        picked_voice = voice or self.catalog.speech_default_voice
        if voice and not self.catalog.has_voice(voice):
            names = ", ".join(v.name for v in self.catalog.speech_voices)
            return ToolResult(ok=False, error=f"未知音色 {voice!r}。可用：{names}")

        r = await self.gateway.speak(
            self.catalog.speech_provider,
            chosen,
            text,
            voice=picked_voice or None,
            language=language or None,
            speed=speed,
            # 字段名是 instructions，不是 instruct —— 后者会被当成未知字段丢掉。
            # ⚠️ 实测 2026-09-12：即使名字对了，经 APIMart 转发的 gpt-4o-mini-tts
            #    也**不响应**这个参数（"每句停顿三秒"的极端指令对时长毫无影响）。
            #    所以别指望用它控语气 —— 真正有效的是 speed 和文案本身的写法。
            instructions=instruct or None,
            response_format=response_format,
        )
        if not r.ok:
            return ToolResult(ok=False, error=f"{chosen} 合成失败：{r.error}")

        asset = self.store.create_blob(
            r.audio,
            ext=_EXT.get(r.fmt, ".mp3"),
            type_=AssetType.AUDIO,
            mime=_MIME.get(r.fmt, "audio/mpeg"),
            summary=summary or f"配音·{picked_voice or '默认'}·{len(text)}字",
            parents=[parent_id] if parent_id else [],
            creator=f"model:{chosen}",
            gen_params={
                "voice": picked_voice,
                "language": language,
                "speed": speed,
                "instruct": instruct,
                "chars": len(text),
                # 把原文一起存下来。只记字数的话，事后想核对"到底念了什么"
                # 就只能靠 ASR 反推 —— 而 ASR 会有同音字和繁简问题，
                # 字幕校正正是要拿这份原文当唯一真相。
                "text": text,
            },
        )
        return ToolResult(
            content=(
                f"用 {chosen}（{why}）音色 {picked_voice or '默认'} 合成了 {len(text)} 字，"
                f"{len(r.audio) / 1024:.0f}KB，耗时 {r.elapsed_s}s\n{asset.id} → {asset.uri}"
            ),
            asset_ref=asset.id,
        )

    async def _fn_transcribe(
        self,
        asset_id: str,
        format: str = "json",  # noqa: A002 — 对齐 API 字段名
        model: str = "",
        language: str = "",
        prompt: str = "",
        summary: str = "",
    ) -> ToolResult:
        chosen, why = self.catalog.choose_audio(model, kind="asr")
        if not chosen:
            return ToolResult(ok=False, error=why)
        if format not in TRANSCRIPT_FORMATS:
            return ToolResult(
                ok=False,
                error=(
                    f"不支持的输出格式 {format!r}。"
                    f"可选：{', '.join(TRANSCRIPT_FORMATS)}"
                ),
            )

        try:
            data = self.store.blob(asset_id)
        except (KeyError, ValueError) as e:
            return ToolResult(ok=False, error=str(e))

        filename = Path(self.store.get(asset_id).uri or "audio.mp3").name
        r = await self.gateway.transcribe(
            self.catalog.provider,
            chosen,
            data,
            filename,
            response_format=format,
            language=language or None,
            prompt=prompt or None,
        )
        if not r.ok:
            return ToolResult(ok=False, error=f"{chosen} 转写失败：{r.error}")

        asset = self.store.create(
            r.text,
            type_=AssetType.SUBTITLE if r.is_subtitle else AssetType.TEXT,
            summary=summary or (f"字幕·{format}" if r.is_subtitle else f"转写·{len(r.text)}字"),
            parents=[asset_id],  # 血缘挂在音频下面
            creator=f"model:{chosen}",
            gen_params={"format": format, "language": r.language, "duration": r.duration},
        )

        head = (
            f"用 {chosen}（{why}）转写完成，耗时 {r.elapsed_s}s"
            + (f"，时长 {r.duration}s" if r.duration else "")
            + (f"，{len(r.segments)} 个分段" if r.segments else "")
        )
        return ToolResult(
            content=f"{head}\n{asset.id}\n---\n{r.text[:1500]}",
            asset_ref=asset.id,
            truncated=len(r.text) > 1500,
        )

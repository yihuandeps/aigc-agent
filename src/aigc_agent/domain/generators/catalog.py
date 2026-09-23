"""M13 生成适配器 —— 模型目录与选型。

与 M2 的分工：
  · M2 Model Gateway  管**怎么调通**（协议、轮询、重试、计费）
  · M13 这里          管**调什么、怎么调好**（选哪个模型、参数预设、多候选）

目录是 yaml 数据（config/media_models.yaml），加模型不用改代码。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ...harness.model.media import MediaKind


class MediaModel(BaseModel):
    id: str
    tier: str = "balanced"  # quality | balanced | fast
    cost: int = 3  # 1-5，5 最贵
    strengths: str = ""
    aspect_ratios: list[str] = Field(default_factory=list)
    resolutions: list[str] = Field(default_factory=list)
    max_duration: int | None = None
    # 参考素材总数上限（图 + 参考视频 + 参考音频）。seedance 2.0 实测：超过 9 个直接 HTTP 400
    # "This model accepts at most 9 reference images"，而且参考图按两个字段双送时**算两份**。
    # 没填 = 不限。媒体层按它在提交前拦，短剧渲染按它裁参考图（人物必保）。
    max_refs: int | None = None
    supports_edit: bool = False
    # 元/张（图）或 元/段（视频），**参考价**。填了 Cost Guard 才能按金额记，
    # 没填只按次数 / 秒数拦。网关不回传单价，只能靠这里。
    price: float | None = None
    # 视频按秒计价（元/秒，720p 口径）：填了就按「秒数 × 单价 × 分辨率系数」算，
    # 比按段计准 —— 5 秒和 15 秒一段的价钱差三倍（2026-09-23）
    price_per_second: float | None = None
    # 分辨率系数（相对 720p）：如 {"480p": 0.5, "1080p": 2.0}；没写的分辨率按 1
    resolution_factor: dict[str, float] = Field(default_factory=dict)
    # 没传时长时按几秒估（视频）
    default_duration: int | None = None

    def brief(self) -> str:
        """进上下文的形态，约 20 token。模型照着这个自己挑。"""
        extra = f"，最长 {self.max_duration}s" if self.max_duration else ""
        if self.resolutions:
            extra += "，" + "/".join(self.resolutions)
        return f"{self.id}（{self.tier}·成本{self.cost}/5）：{self.strengths}{extra}"


class Voice(BaseModel):
    name: str
    lang: str = ""
    note: str = ""

    def brief(self) -> str:
        return f"{self.name}（{self.lang}）：{self.note}"


class AudioModel(MediaModel):
    kind: str = "tts"  # tts | asr
    # 属于哪个 provider。留空 = 跟主 provider 走。
    # 不标的话，TTS 切到 minimax 之后 gpt-4o-mini-tts 仍会出现在候选里，
    # 选中它就是 "method t2a-v2 not have model" —— 这个错在目录层就该拦住。
    provider: str = ""


class Polling(BaseModel):
    interval: float = 3.0
    backoff: float = 1.3
    max_interval: float = 20.0
    max_wait_image: float = 180.0
    max_wait_video: float = 900.0
    # 排队 watchdog：任务提交后在服务端队列里最多等这么久。
    # 排队时间不吃生成预算（生成计时从状态变 running 才起算），
    # 这个只防「永远排不上」。没上报 running 状态的 provider 由它兜住总时长。
    max_queue: float = 600.0
    # 轮询请求次数上限（0 = 不限）；连续网络错误容忍次数；提交阶段网络重试次数
    max_polls: int = 0
    max_transient: int = 10
    submit_retries: int = 2


class Concurrency(BaseModel):
    """批量渲染的客户端并发上限。0 = 不限，实际并发由 provider 侧限流/排队决定。

    只压同一层里无依赖的任务：依赖永远排队（服装等角色主形象、
    镜头等前序片段），与这里的数无关。
    """

    image: int = 0
    video: int = 8


class MediaCatalog(BaseModel):
    provider: str = "apimart"
    # TTS 可以单独走别家。中文口播上 APIMart 只有 OpenAI 那套英文音色库，
    # 换中文原生的 provider 是唯一能根治"听着假"的办法；
    # 而转写（whisper）留在原 provider 就行，没必要一起换。
    tts_provider: str = ""
    image: list[MediaModel] = Field(default_factory=list)
    video: list[MediaModel] = Field(default_factory=list)
    image_defaults: dict[str, str] = Field(default_factory=dict)
    video_defaults: dict[str, str] = Field(default_factory=dict)
    audio: list[AudioModel] = Field(default_factory=list)
    audio_defaults: dict[str, str] = Field(default_factory=dict)
    voices: list[Voice] = Field(default_factory=list)
    default_voice: str = ""
    # 按 provider 分开的音色表与默认音色。换了 TTS provider 之后，
    # 音色名也整套换了 —— 拿 nova 去请求 MiniMax，它**不会报错**，
    # 而是静默换成自己的默认音色。你以为选了某个音色，实际拿到的是别的。
    provider_voices: dict[str, list[Voice]] = Field(default_factory=dict)
    provider_default_voice: dict[str, str] = Field(default_factory=dict)
    # 短剧流水线用哪个生图/生视频模型。配置驱动，不写死在代码里。
    drama: dict[str, str] = Field(default_factory=dict)
    # 参考素材走哪个请求字段。图片的 image 是 2026-09-12 对照实验证过的；
    # 视频/音频参考按 APIMart seedance 文档的 image_urls / video_urls / audio_urls 命名，
    # 提示词里用 @视频1 / @音频1 引用。字段名不对接口**不报错只是忽略**，所以留成配置。
    image_ref_field: str = "image"
    image_ref_alias: str = "image_urls"  # 参考图再按这个名字送一份；空 = 不送
    video_ref_field: str = "video_urls"
    audio_ref_field: str = "audio_urls"

    @property
    def speech_provider(self) -> str:
        """TTS 实际走哪家。没单独配就跟着主 provider。"""
        return self.tts_provider or self.provider

    @property
    def speech_voices(self) -> list[Voice]:
        """当前 TTS provider 的音色表。"""
        return self.provider_voices.get(self.speech_provider) or self.voices

    @property
    def speech_default_voice(self) -> str:
        return self.provider_default_voice.get(self.speech_provider) or self.default_voice
    polling: Polling = Field(default_factory=Polling)
    concurrency: Concurrency = Field(default_factory=Concurrency)

    # ---------- 查询 ----------

    def models(self, kind: MediaKind) -> list[MediaModel]:
        return self.image if kind is MediaKind.IMAGE else self.video

    def get(self, kind: MediaKind, model_id: str) -> MediaModel | None:
        return next((m for m in self.models(kind) if m.id == model_id), None)

    def defaults(self, kind: MediaKind) -> dict[str, str]:
        return self.image_defaults if kind is MediaKind.IMAGE else self.video_defaults

    def choose(self, kind: MediaKind, model: str = "", prefer: str = "balanced") -> tuple[str, str]:
        """选一个模型，返回 (model_id, 选择理由)。

        显式指定优先；没指定就按 tier 取默认；默认没配就在该 tier 里挑最便宜的。
        **不认识的模型 id 不静默替换** —— 直接报错，否则会悄悄用错模型还查不出来。
        """
        available = self.models(kind)
        if not available:
            return "", f"目录里没有任何 {kind.value} 模型"

        if model:
            if self.get(kind, model):
                return model, "按你指定"
            names = ", ".join(m.id for m in available)
            return "", f"未知模型 {model!r}。可用：{names}"

        want = prefer if prefer in ("quality", "balanced", "fast") else "balanced"
        picked = self.defaults(kind).get(want)
        if picked and self.get(kind, picked):
            return picked, f"{want} 档默认"

        tier_hits = [m for m in available if m.tier == want]
        pool = tier_hits or available
        best = min(pool, key=lambda m: m.cost)
        return best.id, f"{want} 档无默认配置，取该档最省的"

    # ---------- 语音 ----------

    def audio_models(self, kind: str) -> list[AudioModel]:
        """按 kind 过滤，并且只留当前 provider 能用的。

        TTS 走 speech_provider，ASR 走主 provider —— 两者可以是不同的家。
        """
        active = self.speech_provider if kind == "tts" else self.provider
        return [
            m
            for m in self.audio
            if m.kind == kind and (m.provider or self.provider) == active
        ]

    def choose_audio(self, model: str = "", kind: str = "tts") -> tuple[str, str]:
        """选 TTS 或 ASR 模型。同 choose()：未知 id 直接报错，不静默替换。"""
        available = self.audio_models(kind)
        if not available:
            return "", f"目录里没有任何 {kind} 模型"
        if model:
            if any(m.id == model for m in available):
                return model, "按你指定"
            names = ", ".join(m.id for m in available)
            return "", f"未知 {kind} 模型 {model!r}。可用：{names}"
        # 默认模型必须跟着 provider 走。TTS 换成 minimax 之后还拿
        # gpt-4o-mini-tts 去请求，会得到
        # "method t2a-v2 not have model: gpt-4o-mini-tts" —— provider 和
        # 模型各配各的，错配几乎必然发生，所以在这里绑死。
        picked = ""
        if kind == "tts" and self.speech_provider != self.provider:
            picked = self.audio_defaults.get(f"tts_{self.speech_provider}", "")
        picked = picked or self.audio_defaults.get(kind, "")
        if picked and any(m.id == picked for m in available):
            return picked, "默认"
        best = min(available, key=lambda m: m.cost)
        return best.id, "无默认配置，取最省的"

    def has_voice(self, name: str) -> bool:
        return any(v.name == name for v in self.speech_voices)

    def render_audio(self) -> str:
        nl = "\n"
        parts: list[str] = []

        tts = self.audio_models("tts")
        if tts:
            parts.append("## TTS 模型" + nl + nl.join(f"- {m.brief()}" for m in tts))

        if self.voices:
            block = "## 音色" + nl + nl.join(f"- {v.brief()}" for v in self.voices)
            if self.default_voice:
                block += nl + f"默认：{self.default_voice}"
            parts.append(block)

        asr = self.audio_models("asr")
        if asr:
            parts.append("## 转写模型" + nl + nl.join(f"- {m.brief()}" for m in asr))

        return (nl + nl).join(parts) or "语音目录为空"

    def render(self, kind: MediaKind) -> str:
        """给模型看的目录。"""
        lines = [m.brief() for m in self.models(kind)]
        d = self.defaults(kind)
        if d:
            lines.append(
                "默认：" + " / ".join(f"{k}→{v}" for k, v in d.items())
            )
        return "\n".join(f"- {x}" for x in lines)

    def max_wait(self, kind: MediaKind) -> float:
        return (
            self.polling.max_wait_image
            if kind is MediaKind.IMAGE
            else self.polling.max_wait_video
        )

    def price_of(
        self, kind: MediaKind, model_id: str, params: dict | None = None, n: int = 1
    ) -> float | None:
        """一次生成的参考价（元）。目录没填单价返回 None（金额口径看不见它）。"""
        m = self.get(kind, model_id)
        if m is None:
            return None
        params = params or {}
        if kind is MediaKind.VIDEO and m.price_per_second is not None:
            secs = self.seconds_of(model_id, params)
            factor = float(m.resolution_factor.get(str(params.get("resolution") or ""), 1.0))
            return round(m.price_per_second * secs * factor * max(1, n), 4)
        if m.price is not None:
            return round(m.price * max(1, n), 4)
        return None

    def seconds_of(self, model_id: str, params: dict | None = None) -> float:
        """一段视频按几秒算（传了 duration 用它，受模型上限约束；没传用目录默认 / 5 秒）。"""
        m = self.get(MediaKind.VIDEO, model_id)
        params = params or {}
        try:
            secs = float(params.get("duration") or 0)
        except (TypeError, ValueError):
            secs = 0.0
        if secs <= 0:
            secs = float((m.default_duration if m else None) or 5)
        if m is not None and m.max_duration:
            secs = min(secs, float(m.max_duration))
        return secs

    @property
    def priced(self) -> bool:
        """图 / 视频目录里有没有任何一个模型填了单价（没有 = 金额护栏看不见媒体）。"""
        return any(
            m.price is not None or m.price_per_second is not None
            for m in list(self.image) + list(self.video)
        )

    def max_concurrency(self, kind: str) -> int:
        """批量渲染并发上限（"image" / "video"）。0 = 不限。"""
        return self.concurrency.image if kind == "image" else self.concurrency.video

    # ---------- 加载 ----------

    @classmethod
    def load(cls, path: str | Path) -> MediaCatalog:
        path = Path(path)
        if not path.exists():
            return cls()
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        def section(key: str) -> tuple[list[MediaModel], dict[str, str]]:
            block = raw.get(key) or {}
            models = [MediaModel.model_validate(m) for m in block.get("models") or []]
            return models, dict(block.get("defaults") or {})

        img, img_def = section("image")
        vid, vid_def = section("video")

        aud_raw = raw.get("audio") or {}
        return cls(
            provider=raw.get("provider", "apimart"),
            image=img,
            video=vid,
            image_defaults=img_def,
            video_defaults=vid_def,
            audio=[AudioModel.model_validate(m) for m in aud_raw.get("models") or []],
            audio_defaults=dict(aud_raw.get("defaults") or {}),
            voices=[Voice.model_validate(v) for v in aud_raw.get("voices") or []],
            default_voice=aud_raw.get("default_voice", ""),
            provider_voices={
                k[len("voices_") :]: [Voice.model_validate(v) for v in (val or [])]
                for k, val in aud_raw.items()
                if k.startswith("voices_")
            },
            provider_default_voice={
                k[len("default_voice_") :]: str(val)
                for k, val in aud_raw.items()
                if k.startswith("default_voice_")
            },
            tts_provider=aud_raw.get("tts_provider", ""),
            drama={k: str(v) for k, v in (raw.get("drama") or {}).items()},
            image_ref_field=str((raw.get("ref_fields") or {}).get("image") or "image"),
            image_ref_alias=str(
                (raw.get("ref_fields") or {}).get("image_alias", "image_urls") or ""
            ),
            video_ref_field=str((raw.get("ref_fields") or {}).get("video") or "video_urls"),
            audio_ref_field=str((raw.get("ref_fields") or {}).get("audio") or "audio_urls"),
            polling=Polling.model_validate(raw.get("polling") or {}),
            concurrency=Concurrency.model_validate(raw.get("concurrency") or {}),
        )

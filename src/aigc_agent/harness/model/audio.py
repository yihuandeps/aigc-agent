"""M2 语音 —— TTS 与 ASR。

**为什么不复用 media.py 的异步任务模型：调用形状根本不同。**

M2 到这里一共有三种形状，硬塞进一个抽象只会互相别扭：

| 形状 | 谁 | 请求 | 响应 |
|---|---|---|---|
| 同步 JSON | chat | JSON | JSON |
| 异步任务 | 图像/视频 | JSON | task_id → 轮询 → URL |
| **同步二进制** | **TTS** | JSON | **音频字节流** |
| **多部分上传** | **ASR** | multipart 文件 | JSON / srt / vtt |

TTS 直接吐字节，没有 URL 可轮询；ASR 要上传文件。所以这里单独一层，
但仍归 M2 —— 重试、超时、计费、事件都走同一套。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx2 as httpx

from ..events.bus import EventBus, EventType
from .media import default_proxy

# ASR 支持的输入格式与上限（来自 APIMart 文档）
ASR_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
ASR_MAX_BYTES = 25 * 1024 * 1024

# 转写输出格式。srt / vtt 直接就是字幕文件 —— 短视频链路的字幕由此而来。
TRANSCRIPT_FORMATS = ("json", "text", "srt", "verbose_json", "vtt")


@dataclass
class SpeechResult:
    ok: bool = True
    audio: bytes = b""
    fmt: str = "mp3"
    model: str = ""
    voice: str = ""
    chars: int = 0
    elapsed_s: float = 0.0
    error: str | None = None


@dataclass
class TranscriptResult:
    ok: bool = True
    text: str = ""
    fmt: str = "json"
    language: str = ""
    duration: float = 0.0
    segments: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    elapsed_s: float = 0.0
    error: str | None = None

    @property
    def is_subtitle(self) -> bool:
        return self.fmt in ("srt", "vtt")


class AudioProvider(Protocol):
    name: str

    async def speak(self, model: str, text: str, **params: Any) -> SpeechResult: ...
    async def transcribe(
        self, model: str, audio: bytes, filename: str, **params: Any
    ) -> TranscriptResult: ...
    async def close(self) -> None: ...


class ApiMartAudioProvider:
    """APIMart 语音接口（OpenAI 兼容）。

      TTS  POST {base}/audio/speech          JSON  → 音频字节流
      ASR  POST {base}/audio/transcriptions  表单  → JSON / srt / vtt
    """

    name = "apimart"

    def __init__(
        self, base_url: str, api_key: str, timeout: float = 120.0, proxy: str | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.proxy = proxy or default_proxy()
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
                proxy=self.proxy,
            )
        return self._client

    async def speak(self, model: str, text: str, **params: Any) -> SpeechResult:
        fmt = str(params.get("response_format") or "mp3")
        body = {
            "model": model,
            "input": text,
            **{k: v for k, v in params.items() if v is not None},
        }
        resp = await self._http().post("/audio/speech", json=body)

        if resp.status_code >= 400:
            return SpeechResult(
                ok=False,
                model=model,
                error=f"HTTP {resp.status_code}：{_err_of(resp)}",
            )

        data = resp.content
        # 有些网关出错时仍返回 200 + JSON 错误体，别把它当音频存下来
        if data[:1] in (b"{", b"[") and b"error" in data[:200].lower():
            return SpeechResult(ok=False, model=model, error=_short(data))
        if not data:
            return SpeechResult(ok=False, model=model, error="返回了空音频")

        return SpeechResult(
            audio=data,
            fmt=fmt,
            model=model,
            voice=str(params.get("voice") or ""),
            chars=len(text),
        )

    async def transcribe(
        self, model: str, audio: bytes, filename: str, **params: Any
    ) -> TranscriptResult:
        fmt = str(params.get("response_format") or "json")
        form = {
            "model": model,
            **{k: str(v) for k, v in params.items() if v is not None},
        }
        resp = await self._http().post(
            "/audio/transcriptions",
            data=form,
            files={"file": (filename, audio, _mime_of(filename))},
        )

        if resp.status_code >= 400:
            return TranscriptResult(
                ok=False, model=model, fmt=fmt, error=f"HTTP {resp.status_code}：{_err_of(resp)}"
            )

        # srt / vtt / text 是纯文本；json / verbose_json 是结构体
        if fmt in ("srt", "vtt", "text"):
            return TranscriptResult(text=resp.text, fmt=fmt, model=model)

        try:
            payload = resp.json()
        except Exception:  # noqa: BLE001
            return TranscriptResult(text=resp.text, fmt=fmt, model=model)

        return TranscriptResult(
            text=str(payload.get("text") or ""),
            fmt=fmt,
            model=model,
            language=str(payload.get("language") or ""),
            duration=float(payload.get("duration") or 0.0),
            segments=list(payload.get("segments") or []),
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# voice_setting.voice_id 必填，没指定时用这个。
# 取值来自 /v1/get_voice 实拉的 303 个系统音色，不是猜的。
DEFAULT_MINIMAX_VOICE = "Chinese (Mandarin)_News_Anchor"


class MiniMaxSpeechProvider:
    """MiniMax 语音合成（T2A v2）。**只做 TTS，不做转写。**

    接进来的唯一理由：APIMart 那 364 个模型里没有一个中文原生 TTS，
    只有 OpenAI 那套英文音色库。英文音色念中文，那股"假"是先天的，
    调参数和改文案都只能缓解不能消除 —— 实测下来是这条链路的天花板。

    ⚠️ **国内站和国际站是两条产品线**，账号和余额都不通：
        platform.minimax.cn  → api.minimax.cn（本实现用这个）
        platform.minimaxi.com → api.minimaxi.com
    模型名也不一样：国内站没有 speech-2.5-*，是 speech-2.8/2.6/02/01 系列。
    和 Kimi Code vs platform.moonshot.cn 是同一类坑，别把域名改回去。

    和 OpenAI 形状的四处不同，照搬会坏：
      1. 参数是嵌套的 voice_setting / audio_setting，不是平铺
      2. voice_setting.voice_id **必填**（OpenAI 那边省略 voice 是合法的）
      3. 返回的是 JSON，音频在 data.audio 里，而且是 **hex 字符串**不是二进制流
         （base64 解会得到一堆噪音，这个坑很隐蔽 —— 文件能生成、能播放、
         全是白噪音）
      4. 失败时 HTTP 也是 200，错误在 base_resp.status_code 里
    """

    name = "minimax"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        group_id: str,
        timeout: float = 120.0,
        proxy: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.group_id = group_id
        self.timeout = timeout
        # 国内直连即可，配了代理反而可能更慢；沿用全局设置但允许为空
        self.proxy = proxy
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
                proxy=self.proxy,
            )
        return self._client

    async def speak(self, model: str, text: str, **params: Any) -> SpeechResult:
        if not self.api_key:
            return SpeechResult(
                ok=False,
                model=model,
                error=(
                    "MiniMax 没配好：.env 里需要 MINIMAX_API_KEY"
                    "（platform.minimaxi.com 账户页）。"
                ),
            )

        fmt = str(params.get("response_format") or "mp3")
        # voice_setting 是**必填**的。不传会被拒（2013 invalid params, empty field），
        # 所以没指定音色时也要给一个默认的，不能像 OpenAI 那样省略。
        voice = str(params.get("voice") or DEFAULT_MINIMAX_VOICE)
        voice_setting: dict[str, Any] = {"voice_id": voice}
        if params.get("speed") is not None:
            voice_setting["speed"] = float(params["speed"])
        if params.get("vol") is not None:
            voice_setting["vol"] = float(params["vol"])
        if params.get("pitch") is not None:
            voice_setting["pitch"] = int(params["pitch"])
        # emotion 是**枚举**不是自由文本。传 instruct 那种整句描述会被拒，
        # 所以由上层映射好再传进来，这里不做猜测。
        if params.get("emotion"):
            voice_setting["emotion"] = str(params["emotion"])

        body: dict[str, Any] = {
            "model": model,
            "text": text,
            "stream": False,
            "voice_setting": voice_setting,
            "audio_setting": {"format": fmt, "channel": 1},
        }
        # 粤语等非普通话音色必须带这个，否则会按普通话念
        if params.get("language_boost"):
            body["language_boost"] = str(params["language_boost"])

        # GroupId 只在老式 key 上需要。新的 sk-api- 前缀 key 自带归属，
        # 反而**不能传** —— 实测传了会 1004 token not match group。
        query = {"GroupId": self.group_id} if self.group_id else None
        resp = await self._http().post("/v1/t2a_v2", params=query, json=body)
        if resp.status_code >= 400:
            return SpeechResult(
                ok=False, model=model, error=f"HTTP {resp.status_code}：{_err_of(resp)}"
            )

        try:
            payload = resp.json()
        except Exception:  # noqa: BLE001
            return SpeechResult(ok=False, model=model, error=_short(resp.content))

        # MiniMax 失败时也返回 HTTP 200，错误在 base_resp 里 —— 必须单独查，
        # 否则会把错误响应当成音频存下来。
        base = payload.get("base_resp") or {}
        if int(base.get("status_code") or 0) != 0:
            return SpeechResult(
                ok=False,
                model=model,
                error=f"MiniMax {base.get('status_code')}：{base.get('status_msg') or payload}",
            )

        hexed = ((payload.get("data") or {}).get("audio")) or ""
        if not hexed:
            return SpeechResult(ok=False, model=model, error=f"没拿到音频：{str(payload)[:200]}")
        try:
            audio = bytes.fromhex(hexed)
        except ValueError as e:
            return SpeechResult(ok=False, model=model, error=f"音频不是合法 hex：{e}")

        return SpeechResult(audio=audio, fmt=fmt, model=model, voice=voice, chars=len(text))

    async def voices(self) -> list[dict[str, str]]:
        """拉系统音色表。

        **这个接口不扣费**，余额为 0 时也能调通 —— 排查"到底是 key 不对
        还是没钱"时特别有用：能列出音色说明鉴权没问题。
        """
        resp = await self._http().post("/v1/get_voice", json={"voice_type": "system"})
        if resp.status_code >= 400:
            return []
        payload = resp.json()
        if int((payload.get("base_resp") or {}).get("status_code") or 0) != 0:
            return []
        return [
            {"id": str(v.get("voice_id") or ""), "name": str(v.get("voice_name") or "")}
            for v in payload.get("system_voice") or []
        ]

    async def transcribe(
        self, model: str, audio: bytes, filename: str, **params: Any
    ) -> TranscriptResult:
        raise NotImplementedError(
            "MiniMax 这条通路只接了 TTS。转写请走 apimart 的 whisper —— "
            "catalog 里 tts_provider 和 provider 是分开配的，就是为了这个。"
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _err_of(resp: Any) -> str:
    try:
        payload = resp.json()
    except Exception:  # noqa: BLE001
        return _short(getattr(resp, "content", b"") or b"")
    for key in ("message", "error", "msg", "detail"):
        v = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict) and isinstance(v.get("message"), str):
            return v["message"]
    return str(payload)[:200]


def _short(data: bytes) -> str:
    return data[:200].decode("utf-8", errors="replace")


def _mime_of(filename: str) -> str:
    return {
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".mp4": "video/mp4",
        ".wav": "audio/wav",
        ".webm": "audio/webm",
        ".mpeg": "audio/mpeg",
        ".mpga": "audio/mpeg",
    }.get(Path(filename).suffix.lower(), "application/octet-stream")


class AudioGateway:
    """统一入口：校验、计时、事件、错误归一。"""

    def __init__(self, providers: dict[str, AudioProvider], bus: EventBus) -> None:
        self.providers = providers
        self.bus = bus

    async def speak(
        self, provider: str, model: str, text: str, **params: Any
    ) -> SpeechResult:
        p = self.providers.get(provider)
        if p is None:
            return SpeechResult(ok=False, error=f"未配置语音 provider {provider!r}")
        if not text.strip():
            return SpeechResult(ok=False, error="文本为空")
        if len(text) > 4096:
            return SpeechResult(
                ok=False,
                error=f"文本 {len(text)} 字，超过单次 4096 上限。请分段合成后再拼接。",
            )

        started = time.perf_counter()
        await self.bus.emit(
            EventType.MODEL_REQUEST, modality="tts", model=model, chars=len(text)
        )
        try:
            r = await p.speak(model, text, **params)
        except Exception as e:  # noqa: BLE001
            r = SpeechResult(ok=False, model=model, error=f"{type(e).__name__}: {e}")
        r.elapsed_s = round(time.perf_counter() - started, 2)

        await self.bus.emit(
            EventType.MODEL_RESPONSE,
            modality="tts",
            model=model,
            ok=r.ok,
            bytes=len(r.audio),
            elapsed_s=r.elapsed_s,
            error=r.error,
        )
        return r

    async def transcribe(
        self, provider: str, model: str, audio: bytes, filename: str, **params: Any
    ) -> TranscriptResult:
        p = self.providers.get(provider)
        if p is None:
            return TranscriptResult(ok=False, error=f"未配置语音 provider {provider!r}")

        suffix = Path(filename).suffix.lower()
        if suffix not in ASR_FORMATS:
            return TranscriptResult(
                ok=False,
                error=(
                    f"不支持的格式 {suffix or '(无扩展名)'}。"
                    f"支持：{', '.join(sorted(ASR_FORMATS))}"
                ),
            )
        if len(audio) > ASR_MAX_BYTES:
            mb = len(audio) / 1024 / 1024
            return TranscriptResult(
                ok=False,
                error=f"文件 {mb:.1f}MB，超过 25MB 上限。先转码或切段再转写。",
            )
        if not audio:
            return TranscriptResult(ok=False, error="音频为空")

        started = time.perf_counter()
        await self.bus.emit(
            EventType.MODEL_REQUEST, modality="asr", model=model, bytes=len(audio)
        )
        try:
            r = await p.transcribe(model, audio, filename, **params)
        except Exception as e:  # noqa: BLE001
            r = TranscriptResult(ok=False, model=model, error=f"{type(e).__name__}: {e}")
        r.elapsed_s = round(time.perf_counter() - started, 2)

        await self.bus.emit(
            EventType.MODEL_RESPONSE,
            modality="asr",
            model=model,
            ok=r.ok,
            chars=len(r.text),
            elapsed_s=r.elapsed_s,
            error=r.error,
        )
        return r

    async def close(self) -> None:
        for p in self.providers.values():
            await p.close()


class FakeAudioProvider:
    """测试用。"""

    name = "fake"

    def __init__(
        self,
        audio: bytes = b"ID3fake-audio-bytes",
        text: str = "这是转写出来的文本",
        srt: str = "1\n00:00:00,000 --> 00:00:02,000\n这是第一句\n",
        fail: str | None = None,  # speak | transcribe
        return_json_error: bool = False,
    ) -> None:
        self.audio = audio
        self.text = text
        self.srt = srt
        self.fail = fail
        self.return_json_error = return_json_error
        self.spoken: list[dict[str, Any]] = []
        self.transcribed: list[dict[str, Any]] = []

    async def speak(self, model, text, **params):
        self.spoken.append({"model": model, "text": text, **params})
        if self.fail == "speak":
            return SpeechResult(ok=False, model=model, error="模拟合成失败")
        if self.return_json_error:
            return SpeechResult(ok=False, model=model, error='{"error":"quota"}')
        return SpeechResult(
            audio=self.audio,
            fmt=str(params.get("response_format") or "mp3"),
            model=model,
            voice=str(params.get("voice") or ""),
            chars=len(text),
        )

    async def transcribe(self, model, audio, filename, **params):
        self.transcribed.append({"model": model, "filename": filename, **params})
        if self.fail == "transcribe":
            return TranscriptResult(ok=False, model=model, error="模拟转写失败")
        fmt = str(params.get("response_format") or "json")
        if fmt in ("srt", "vtt"):
            return TranscriptResult(text=self.srt, fmt=fmt, model=model)
        return TranscriptResult(
            text=self.text,
            fmt=fmt,
            model=model,
            language="zh",
            duration=2.0,
            segments=[{"id": 0, "start": 0.0, "end": 2.0, "text": self.text}],
        )

    async def close(self) -> None:
        return None

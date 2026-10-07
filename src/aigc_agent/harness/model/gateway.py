"""M2 Model Gateway —— 屏蔽模型提供方差异。

职责：请求归一、流式处理、重试退避、token 计数、成本记账。
上层只说「以 main_agent 这个角色调一次」，不关心背后是哪家、哪个型号。

P0 只实现 text/chat 一族。image / video / audio 的 Provider 接口留在
ModelProvider 协议里，P2 再补——尤其 video 是异步任务模型，见 ARCHITECTURE.md M2。
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx2 as httpx

try:  # httpx2 的依赖；流式中途断开时抛的是它的异常，不一定被映射成 httpx 的
    import httpcore2 as _httpcore
except ImportError:  # pragma: no cover
    _httpcore = None
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from ..events.bus import EventBus, EventType
from .config import ModelsConfig, ProviderConfig
from .media import default_proxy


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # 原始 JSON 字符串，解析交给 dispatcher


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost: float | None = None  # 按真实单价（provider 的 pricing）算的，元；没配单价是 None
    # 没配单价时按 text.pricing_estimate 的保守估价算的，元（这时 cost 仍是 None）。
    # 金额护栏和台账照样计入，展示时标「估算」（2026-10-07，09-29 审查 2.1）
    est_cost: float | None = None


@dataclass
class ModelResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    duration_ms: int = 0

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ModelGateway:
    def __init__(self, config: ModelsConfig, bus: EventBus) -> None:
        self.config = config
        self.bus = bus
        self._clients: dict[str, AsyncOpenAI] = {}
        self._pricing_warned: set[str] = set()

    # ---------- 客户端 ----------

    def _client(self, p: ProviderConfig) -> AsyncOpenAI:
        if p.key not in self._clients:
            if not p.api_key_resolved:
                raise RuntimeError(
                    f"provider {p.key!r} 的 api_key 未解析：{p.api_key!r}\n"
                    f"检查 .env 里是否设置了对应环境变量，且进程启动前已加载。"
                )
            proxy = default_proxy()
            self._clients[p.key] = AsyncOpenAI(
                api_key=p.api_key,
                base_url=p.base_url,
                timeout=p.timeout_seconds,
                max_retries=0,  # 重试由本 Gateway 统一管，不交给 SDK
                # 国内网络下不走代理多半连不通，见 default_proxy() 的说明
                http_client=httpx.AsyncClient(proxy=proxy, timeout=p.timeout_seconds)
                if proxy
                else None,
            )
        return self._clients[p.key]

    async def close(self) -> None:
        """关掉所有 provider 客户端。

        SDK 客户端内部握着一个 httpx.AsyncClient，不关的话进程退出时连接池的
        异步生成器被 GC 强行关闭，会刷一屏 "generator didn't stop after athrow()"
        堆栈 —— 无害，但会把真正的输出淹掉。
        """
        for c in self._clients.values():
            try:
                await c.close()
            except Exception:  # noqa: BLE001
                pass
        self._clients.clear()

    async def list_models(self, role: str = "main_agent") -> list[str]:
        """拉取服务端实际可用的模型 id。

        用来核对 config/models.yaml 里 model 字段填得对不对
        —— 这是 P0 阶段第一个该跑的命令。
        """
        provider, _ = self.config.text.resolve(role)
        client = self._client(provider)
        resp = await client.models.list()
        return sorted(m.id for m in resp.data)

    async def list_models_for(self, provider_key: str) -> list[str]:
        """按 provider 查**它自己端点**上的模型列表。

        doctor 用。之前是拿主角色的端点去核对所有 provider 的 model 字段，
        于是走 APIMart 的 gemini 被拿到 Kimi 的列表里比对，永远报「不在服务端
        列表里」—— 一个假阴性会让人不再信任自检。
        """
        p = self.config.text.providers[provider_key]
        resp = await self._client(p).models.list()
        return sorted(m.id for m in resp.data)

    async def probe_cache(
        self, role: str = "summarize", approx_tokens: int = 3000
    ) -> dict[str, Any]:
        """实测 provider 有没有**自动前缀缓存**。

        同一段长前缀连发两次，看第二次 usage 里有没有 cached_tokens。
        它决定 M3 批量驱逐（evict_at=15）的收益是否成立 —— 靠文档猜不如打两次看。
        两次调用都是小输出，代价可忽略。结果回填 config/models.yaml 的 cache.mode。
        """
        line = "内容生产的每份产出都是带版本与血缘的资产，回退只标记不删除，人审理由要落库。"
        filler = "\n".join(f"{i + 1}. {line}" for i in range(max(4, approx_tokens // 25)))
        messages = [
            {"role": "system", "content": "你是回声测试程序。无论收到什么，只回复：OK"},
            {"role": "user", "content": f"以下是背景资料：\n{filler}\n\n只回复 OK。"},
        ]
        first = await self.chat(role, messages, stream=False, max_output_tokens=512)
        second = await self.chat(role, messages, stream=False, max_output_tokens=512)
        hit = second.usage.cached_tokens
        total = second.usage.prompt_tokens
        return {
            "model": second.model,
            "prompt_tokens": total,
            "cached_first": first.usage.cached_tokens,
            "cached_second": hit,
            "hit_ratio": round(hit / total, 3) if total else 0.0,
            "mode": "auto_prefix" if hit > 0 else "none",
        }

    # ---------- 主调用 ----------

    async def chat(
        self,
        role: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool | None = None,
        **overrides: Any,
    ) -> ModelResponse:
        """按角色发起一次文本调用。

        参数优先级：overrides > role_params > provider 默认值。
        """
        provider, role_params = self.config.text.resolve(role)
        params = {**role_params, **overrides}

        use_stream = provider.stream if stream is None else stream
        # 带工具时统一走非流式：流式下累积 tool_call delta 容易踩各家实现差异，
        # 而工具轮本来就不需要给用户看字。纯文本轮才流式。
        if tools:
            use_stream = False

        kwargs: dict[str, Any] = {
            "model": provider.model,
            "messages": messages,
            "max_tokens": params.pop("max_output_tokens", provider.max_output_tokens),
        }
        if "temperature" in params:
            kwargs["temperature"] = params.pop("temperature")
        if params.pop("response_format", None) == "json":
            kwargs["response_format"] = {"type": "json_object"}
        # 强制思考的模型（Kimi K3 等）：按角色给思考强度。
        # 机械任务 low、创作类 high —— 推理 token 也是要付钱的。
        effort = params.pop("thinking_effort", None)
        if effort:
            # OpenAI SDK 的 create() 只收已知字段，厂商扩展参数必须走 extra_body
            kwargs.setdefault("extra_body", {})["thinking"] = {
                "type": "enabled",
                "effort": effort,
            }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = params.pop("tool_choice", "auto")
        kwargs.update(params)  # 其余透传

        # 丢掉该模型明确不支持的参数。不这么做的话，角色配置里一个
        # temperature 就能让整个 provider 全线 400。
        for bad in provider.unsupported_params:
            kwargs.pop(bad, None)

        # 超时按请求大小放宽：24 万 token 的请求实测要 90–290 秒，固定 180 秒
        # 会超时，超时重试又把整段重发一遍。每 10 万（估算）token 加一个基础时长，
        # 最多三倍。
        est = estimate_tokens(messages)
        kwargs["timeout"] = min(
            provider.timeout_seconds * 3,
            provider.timeout_seconds * (1 + est / 100_000),
        )

        await self.bus.emit(
            EventType.MODEL_REQUEST,
            role=role,
            model=provider.model,
            provider=provider.key,
            messages=len(messages),
            tools=len(tools) if tools else 0,
            stream=use_stream,
        )

        started = time.perf_counter()
        try:
            resp = await self._with_retry(provider, kwargs, use_stream)
        except Exception as e:
            # 模型在服务端用不了：换成一条说清「哪个角色、哪个 provider、怎么办」的错误往上抛。
            # 工具层会把它原样报给主模型 —— 之前报的是原始 400，主模型绕了 11 轮去改自己的配置
            if classify_model_error(e)[0] == "model_unavailable":
                raise ModelUnavailableError(role, provider.key, provider.model, str(e)) from e
            raise
        resp.duration_ms = int((time.perf_counter() - started) * 1000)
        resp.model = provider.model

        u = resp.usage
        u.cost = provider.pricing.cost(u.prompt_tokens, u.completion_tokens, u.cached_tokens)
        # 没配单价：按保守估价另算一份 est_cost，金额护栏和台账照样计入、标「估算」。之前按 0 记，
        # gemini 三天约 ¥90–100 护栏看不见（09-29 审查 2.1）。cost 仍是 None —— 它只放真实单价
        # 算出来的钱，只认 cost 的看板照旧标「无单价」，不会把估算当成真账
        estimate = self.config.text.pricing_estimate if u.cost is None else None
        if estimate is not None:
            u.est_cost = estimate.cost(u.prompt_tokens, u.completion_tokens, u.cached_tokens)
        if u.cost is None and provider.key not in self._pricing_warned:
            self._pricing_warned.add(provider.key)
            if estimate is not None and u.est_cost is not None:
                message = (
                    f"provider {provider.key!r} 没配 pricing：金额按保守估价计入"
                    f"（{estimate.brief()}），各处标「估算」。查到真实单价后填进 "
                    "config/models.yaml 这个 provider 的 pricing.input_per_mtok / "
                    "output_per_mtok，就改按真价算。"
                )
            else:
                message = (
                    f"provider {provider.key!r} 未配置 pricing，成本无法核算。"
                    f"在 config/models.yaml 填 pricing.input_per_mtok / output_per_mtok。"
                )
            await self.bus.emit(EventType.WARNING, message=message)

        await self.bus.emit(
            EventType.COST,
            role=role,
            provider=provider.key,
            model=provider.model,
            prompt_tokens=u.prompt_tokens,
            completion_tokens=u.completion_tokens,
            cached_tokens=u.cached_tokens,
            cost=u.cost,
            est_cost=u.est_cost,
            duration_ms=resp.duration_ms,
        )
        await self.bus.emit(
            EventType.MODEL_RESPONSE,
            role=role,
            finish_reason=resp.finish_reason,
            text_len=len(resp.text),
            tool_calls=[c.name for c in resp.tool_calls],
            duration_ms=resp.duration_ms,
        )
        return resp

    # ---------- 重试 ----------

    async def _with_retry(
        self, provider: ProviderConfig, kwargs: dict[str, Any], use_stream: bool
    ) -> ModelResponse:
        cfg = provider.retry
        delay = cfg.initial_delay_ms / 1000
        last: Exception | None = None
        attempt = 0
        budget = cfg.max_attempts  # 连接类错误会就地抬高（见下）

        while True:
            attempt += 1
            try:
                if use_stream:
                    return await self._call_stream(provider, kwargs)
                return await self._call_once(provider, kwargs)
            except APITimeoutError as e:
                # 超时不是网络抖动：请求多半送到了、服务端在算（上下文太大或拥堵），原样重发
                # 还是超时，而且每次整包重发。SDK 里它是 APIConnectionError 的子类，之前按连接
                # 错误给了 6 次，CLI 再续 3 次，最坏整包重发 24 次（2026-09-23 审查）
                last = e
                retryable = True
                budget = min(cfg.max_attempts, 2)
            except (APIConnectionError, RateLimitError) as e:
                last = e
                retryable = True
                # 纯网络问题：服务端没收到请求，重发无副作用 —— 值得多等一会儿
                budget = cfg.attempts_for(isinstance(e, APIConnectionError))
            except APIStatusError as e:
                last = e
                retryable = e.status_code in cfg.retry_status
            except _STREAM_BROKEN as e:
                # 流式输出中途被服务端断开（incomplete chunked read）：这次的输出已经丢了，只能整包
                # 重发。2026-09-25 实测：拆视频提示词时 40 秒左右断过两次，之前直接判失败，主模型
                # 只好手动重跑。只重发一次 —— 反复断多半是服务端的问题，再重发只是再花一次钱
                last = e
                retryable = True
                budget = min(cfg.max_attempts, 2)
            except Exception as e:  # noqa: BLE001
                await self.bus.emit(
                    EventType.MODEL_ERROR, error=f"{type(e).__name__}: {e}", attempt=attempt
                )
                raise

            if not retryable or attempt >= budget:
                await self.bus.emit(
                    EventType.MODEL_ERROR,
                    error=f"{type(last).__name__}: {last}",
                    attempt=attempt,
                    gave_up=True,
                )
                raise last

            cap = cfg.max_delay_ms / 1000
            wait = min(delay, cap)
            if cfg.jitter > 0:
                wait *= 1 + random.uniform(-cfg.jitter, cfg.jitter)
            # 下限防空转，但不能越过上限（上限可能本来就很小）
            wait = min(max(min(0.1, cap), wait), cap * (1 + cfg.jitter))
            await self.bus.emit(
                EventType.MODEL_RETRY,
                attempt=attempt,
                max_attempts=budget,
                delay_s=round(wait, 2),
                error=f"{type(last).__name__}: {last}",
            )
            await asyncio.sleep(wait)
            delay = delay * 2 if cfg.backoff == "exponential" else delay

    # ---------- 两种调用形态 ----------

    async def _call_once(self, provider: ProviderConfig, kwargs: dict[str, Any]) -> ModelResponse:
        raw = await self._client(provider).chat.completions.create(**kwargs)
        choice = raw.choices[0]
        msg = choice.message

        calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            for tc in (msg.tool_calls or [])
        ]
        return ModelResponse(
            text=msg.content or "",
            tool_calls=calls,
            finish_reason=choice.finish_reason or "",
            usage=self._usage(raw),
        )

    async def _call_stream(self, provider: ProviderConfig, kwargs: dict[str, Any]) -> ModelResponse:
        kwargs = {**kwargs, "stream": True, "stream_options": {"include_usage": True}}
        client = self._client(provider)

        try:
            stream = await client.chat.completions.create(**kwargs)
        except APIStatusError as e:
            # 部分兼容实现不认 stream_options，去掉重试一次
            if "stream_options" in str(e):
                kwargs.pop("stream_options", None)
                stream = await client.chat.completions.create(**kwargs)
            else:
                raise

        text_parts: list[str] = []
        finish = ""
        usage = Usage()

        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = self._usage(chunk)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish = choice.finish_reason
            delta = choice.delta
            if delta and delta.content:
                text_parts.append(delta.content)
                await self.bus.emit(EventType.TEXT_DELTA, text=delta.content)

        text = "".join(text_parts)
        if usage.prompt_tokens == 0:
            # 服务端没回 usage，用估算兜底，标记为估算值
            usage.prompt_tokens = estimate_tokens(kwargs["messages"])
            usage.completion_tokens = estimate_tokens(text)
        return ModelResponse(text=text, finish_reason=finish, usage=usage)

    @staticmethod
    def _usage(raw: Any) -> Usage:
        u = getattr(raw, "usage", None)
        if not u:
            return Usage()
        cached = 0
        details = getattr(u, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        # Moonshot 风格的缓存命中字段
        cached = cached or getattr(u, "cached_tokens", 0) or 0
        return Usage(
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            cached_tokens=cached,
        )


_CONTEXT_OVERFLOW = re.compile(
    r"context|too long|maximum.*tokens|tokens.*(exceed|limit)|supports only \d+\s*k",
    re.I,
)
_QUOTA = re.compile(r"usage limit|quota|insufficient|额度|余额|balance", re.I)
# 流式输出被服务端中途断开（httpx 和 httpcore 各有一套异常类，互不继承）
_STREAM_BROKEN: tuple[type[BaseException], ...] = (
    httpx.RemoteProtocolError,
    httpx.ReadError,
    *((_httpcore.RemoteProtocolError, _httpcore.ReadError) if _httpcore is not None else ()),
)
# 模型在服务端用不了：没开通 / 没定价 / 下架 / 型号写错。重试、换写法都没用，只能换模型。
# 2026-09-25 实例（APIMart，new-api 系网关）：「模型 gemini-3.1-pro-preview 的价格尚未由管理员配置，
# 暂时无法使用」——drama 工具把原始 400 报给主模型，主模型绕了 11 轮去翻、去改 Agent 自己的配置
_MODEL_UNAVAILABLE = re.compile(
    r"has not been priced|价格尚未由管理员配置|倍率或价格未配置|model_price_error"
    r"|无可用渠道|no available channel|model_not_found|模型不存在"
    r"|the model .{0,80}does not exist|model .{0,80}not found",
    re.I,
)


class ModelUnavailableError(Exception):
    """某个角色配的模型在服务端用不了。消息本身写清楚该怎么办 —— 工具层照例只把
    「异常类名: 消息」报给主模型，这段话会原样到它眼前。"""

    def __init__(self, role: str, provider: str, model: str, detail: str) -> None:
        self.role, self.provider, self.model, self.detail = role, provider, model, detail
        super().__init__(
            f"角色「{role}」用的模型 {model} 在服务端用不了（config/models.yaml 里 provider "
            f"「{provider}」的 model）。服务端原话：{detail[:200]}。"
            "这不是参数、剧本或网络的问题：重试、换写法、换工具都没用，也不要去读写 Agent 自己的"
            "配置文件。请把这句话告诉用户，由用户决定换成哪个可用型号"
            f"（agent models --role {role} 能列出服务端的型号）。"
        )


def classify_model_error(exc: BaseException) -> tuple[str, str]:
    """把模型侧异常分成几类，Loop 据此决定重试还是交回给人。返回 (类别, 给人的提示)。

      context_overflow  请求超过模型上下文上限 —— 不该重发，压缩后再试
      quota             订阅额度/余额用尽 —— 等额度恢复，进度不会丢
      timeout           响应超时 —— 多半是上下文太大或服务端拥堵
      other             其余
    Kimi 对超长上下文返回的是 401（"k3-256k supports only 256K context"），
    所以判类别看的是报错文本，不看状态码。
    """
    if isinstance(exc, ModelUnavailableError):
        return "model_unavailable", ""  # 消息本身已经写清楚怎么办
    if isinstance(exc, APITimeoutError):
        return "timeout", "模型响应超时，多半是上下文太大或服务端拥堵；发「继续」会重试。"
    if isinstance(exc, APIConnectionError):
        return "connection", (
            "连不上模型服务（网络/代理抖动）。已在同一轮里自动等着重试，"
            "本轮做过的活不会丢；一直连不上就检查代理与 .env 里的 HTTPS_PROXY。"
        )
    msg = str(exc)
    status = getattr(exc, "status_code", None)
    if _CONTEXT_OVERFLOW.search(msg) and status in (None, 400, 401, 413, 422):
        return "context_overflow", "请求超过模型上下文上限。"
    if status in (402, 403, 429) and _QUOTA.search(msg):
        return "quota", "模型额度用尽（订阅窗口或余额）。等额度恢复后发「继续」即可，进度不会丢。"
    if status in (400, 403, 404, 503) and _MODEL_UNAVAILABLE.search(msg):
        return "model_unavailable", (
            "这个模型在服务端用不了（没开通 / 没定价 / 下架）。重试没用：换 config/models.yaml 里"
            "对应 provider 的 model（agent models 能列出服务端的型号）。"
        )
    return "other", ""


def estimate_tokens(payload: Any) -> int:
    """粗略估算。

    中文约 1.5 字/token，英文约 4 字符/token。只用于窗口裁剪的预判和
    服务端没回 usage 时的兜底；真实计费一律以 API 返回的 usage 为准。
    """
    if isinstance(payload, str):
        text = payload
    elif isinstance(payload, list):
        parts: list[str] = []
        for m in payload:
            if isinstance(m, dict):
                c = m.get("content")
                if isinstance(c, str):
                    parts.append(c)
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    parts.append(str(fn.get("name", "")) + str(fn.get("arguments", "")))
        text = "\n".join(parts)
    else:
        text = str(payload)

    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    other = len(text) - cjk
    return int(cjk / 1.5 + other / 4) + 1

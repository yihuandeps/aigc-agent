"""M2 模型配置加载 —— 解析 config/models.yaml。

两条硬规则：
  1. 密钥只从 ${ENV_VAR} 取，配置文件里永远不出现明文。
  2. 上层按「角色」引用模型（main_agent / memory_extract / ...），
     不直接引用模型 id。换模型只改 yaml 的 roles 段。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# thinking.effort 的合法取值（强制思考的模型）。写错了服务端一律 400，而且很快就回、
# 重试也没用 —— 抖音线简报角色配了 medium，6 次调用 6 次 400（2026-09-29 审查）。
# 所以在加载配置时就拦住，启动即报错，而不是等到调用时才失败。
THINKING_EFFORTS = ("low", "high", "max")

# yaml 里未填写的项统一写成 TODO，解析时视为 None
_PLACEHOLDERS = {"TODO", "todo", "TBD", None, ""}


def _expand_env(value: Any) -> Any:
    """递归展开 ${VAR}。变量不存在时保留原样，由调用方报错。"""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _num_or_none(value: Any) -> float | None:
    if value in _PLACEHOLDERS:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class RetryConfig(BaseModel):
    max_attempts: int = 3
    # 连接类错误（代理抖动、DNS、VPN 重连）单独给更深的重试：这类是**纯网络**问题，
    # 服务端没收到请求，重发没有副作用，而 3 次 0.5s/1.0s 总共才等 1.5 秒 ——
    # 家用代理抖一下就是十几秒，等不到就把整轮判死（2026-09-19 用户实测）。
    connect_max_attempts: int = 6
    backoff: str = "exponential"
    initial_delay_ms: int = 500
    max_delay_ms: int = 15_000  # 单次等待上限，指数退避不至于一次等几分钟
    jitter: float = 0.25  # ±25% 抖动，多个并发调用不要同时重试
    retry_on: list[Any] = Field(default_factory=lambda: [429, 500, 502, 503, 504, "timeout"])

    @property
    def retry_status(self) -> set[int]:
        return {c for c in self.retry_on if isinstance(c, int)}

    def attempts_for(self, connection: bool) -> int:
        if not connection:
            return self.max_attempts
        return max(self.max_attempts, self.connect_max_attempts)


class Pricing(BaseModel):
    """单位：元 / 百万 token。未填写则为 None —— 网关改按 text.pricing_estimate 的保守估价
    另算一份「估算」金额（2026-10-07），估价也关了才是「无单价」，成本计算跳过并告警一次。"""

    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    cached_input_per_mtok: float | None = None

    @property
    def known(self) -> bool:
        return self.input_per_mtok is not None and self.output_per_mtok is not None

    def cost(
        self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0
    ) -> float | None:
        if not self.known:
            return None
        fresh = max(prompt_tokens - cached_tokens, 0)
        c = fresh / 1e6 * self.input_per_mtok
        if cached_tokens and self.cached_input_per_mtok is not None:
            c += cached_tokens / 1e6 * self.cached_input_per_mtok
        elif cached_tokens:
            c += cached_tokens / 1e6 * self.input_per_mtok
        c += completion_tokens / 1e6 * self.output_per_mtok
        return c

    def brief(self) -> str:
        """「输入 ¥6.48 / 输出 ¥21.6 每百万 token」，给告警和开工面板看。"""
        if not self.known:
            return "未配单价"
        parts = [f"输入 ¥{self.input_per_mtok:g}"]
        if self.cached_input_per_mtok is not None:
            parts.append(f"命中缓存 ¥{self.cached_input_per_mtok:g}")
        parts.append(f"输出 ¥{self.output_per_mtok:g}")
        return " / ".join(parts) + " 每百万 token"


# 没配 pricing 的文本 provider 按这组**保守估价**算金额，照样计入金额护栏和台账，各处标「估算」
# （2026-10-07，09-29 审查 2.1：gemini 没配单价，9-25 到 9-27 三天约 ¥90–100 按 0 记了）。
# 来源：用 APIMart 余额两次见底之间的变化和 402 预扣金额反推 gemini-3.8-flash，输入约 $0.6–0.9、
# 输出约 $2.8–3.0 每百万 token；取上沿，按 1 美元 = 7.2 元换算：0.9 × 7.2 = 6.48，3.0 × 7.2 = 21.6。
# 命中缓存不打折（cached_input_per_mtok 留空 = 按输入价算）：估价宁高勿低。
# 它不是真实单价：查到后填进那个 provider 的 pricing，那一家就改按真价算。
# models.yaml 的 text.pricing_estimate 能改这组数；那一段没写也按这组估 ——
# 金额护栏不该因为少填一项配置就看不见一家模型
DEFAULT_PRICING_ESTIMATE = Pricing(input_per_mtok=6.48, output_per_mtok=21.6)


def _pricing_estimate(raw: Any) -> Pricing | None:
    """text.pricing_estimate：没写、写成 TODO 的项用默认估价；整段写 false 才是不估
    （不建议：没配单价的那家又会按 0 记）。"""
    if raw is False:
        return None
    raw = raw if isinstance(raw, dict) else {}
    d = DEFAULT_PRICING_ESTIMATE

    def pick(key: str) -> float | None:
        v = _num_or_none(raw.get(key))
        return getattr(d, key) if v is None else v

    return Pricing(
        input_per_mtok=pick("input_per_mtok"),
        output_per_mtok=pick("output_per_mtok"),
        cached_input_per_mtok=pick("cached_input_per_mtok"),
    )


class ProviderConfig(BaseModel):
    key: str
    vendor: str = ""
    protocol: str = "openai_compatible"
    base_url: str
    api_key: str
    model: str
    context_window: int = 128_000
    max_output_tokens: int = 4096
    timeout_seconds: float = 120.0
    stream: bool = True
    retry: RetryConfig = Field(default_factory=RetryConfig)
    pricing: Pricing = Field(default_factory=Pricing)
    # 该模型不接受的参数，Gateway 会静默丢弃。
    # 强制思考型模型（Kimi K3、OpenAI o 系列）通常不许调 temperature / top_p。
    unsupported_params: list[str] = Field(default_factory=list)

    @property
    def api_key_resolved(self) -> bool:
        """${VAR} 没被替换掉说明环境变量缺失。"""
        return bool(self.api_key) and not _ENV_PATTERN.search(self.api_key)


class TextConfig(BaseModel):
    providers: dict[str, ProviderConfig]
    roles: dict[str, str]
    role_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    fallback_chain: list[str] = Field(default_factory=list)
    # providers 里没配 pricing 的按这组保守估价算「估算」金额（见 DEFAULT_PRICING_ESTIMATE）。
    # None = 不估，那些调用照旧记「无单价」、金额按 0
    pricing_estimate: Pricing | None = Field(
        default_factory=lambda: DEFAULT_PRICING_ESTIMATE.model_copy()
    )

    @field_validator("role_params")
    @classmethod
    def _check_thinking_effort(cls, v: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        for role, params in v.items():
            effort = (params or {}).get("thinking_effort")
            if effort and effort not in THINKING_EFFORTS:
                raise ValueError(
                    f"role_params.{role}.thinking_effort = {effort!r} 不是合法值，"
                    f"只能是 {' / '.join(THINKING_EFFORTS)}（写错了服务端会直接返回 400）"
                )
        return v

    def resolve(self, role: str) -> tuple[ProviderConfig, dict[str, Any]]:
        """角色 → (provider 配置, 该角色的参数覆盖)。"""
        if role not in self.roles:
            raise KeyError(
                f"未知角色 {role!r}。config/models.yaml 的 text.roles 里可用："
                f"{', '.join(sorted(self.roles))}"
            )
        provider_key = self.roles[role]
        if provider_key not in self.providers:
            raise KeyError(f"角色 {role!r} 指向的 provider {provider_key!r} 未定义")
        return self.providers[provider_key], dict(self.role_params.get(role, {}))

    def estimated_providers(self) -> list[str]:
        """没配 pricing、金额按 pricing_estimate 估算的 provider（只列有角色在用的），
        给开工面板点名，如「gemini（gemini-3.8-flash，按输入 ¥6.48 / 输出 ¥21.6
        每百万 token 估）」。估价关了返回空。"""
        est = self.pricing_estimate
        if est is None or not est.known:
            return []
        used = set(self.roles.values())
        return [
            f"{key}（{p.model}，按{est.brief()} 估）"
            for key, p in self.providers.items()
            if key in used and not p.pricing.known
        ]


class CacheConfig(BaseModel):
    # auto_prefix | explicit | none | None(未确认)
    mode: str | None = None
    enabled: bool = True


class CostGuard(BaseModel):
    per_task_limit: float | None = None
    per_project_limit: float | None = None
    daily_limit: float | None = None
    on_exceed: str = "pause_and_ask"
    # 次数口径：单次任务内各类媒体调用的上限。没填的类别用 budget.py 的默认值。
    call_limits: dict[str, int] = Field(default_factory=dict)
    # 单日各类媒体调用上限（跨会话，靠台账）。没填 = 不限
    daily_call_limits: dict[str, int] = Field(default_factory=dict)
    # 视频秒数口径（2026-09-23）：本次开工 / 单日。不依赖单价的视频刹车
    seconds_limit: float | None = None
    daily_seconds_limit: float | None = None


class ModelsConfig(BaseModel):
    text: TextConfig
    cache: CacheConfig = Field(default_factory=CacheConfig)
    cost_guard: CostGuard = Field(default_factory=CostGuard)
    node_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # image / video / audio / embedding 暂未接入，保留原始 dict 不做校验
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @classmethod
    def load(cls, path: str | Path) -> ModelsConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"找不到模型配置：{path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw = _expand_env(raw)

        text_raw = raw.get("text") or {}
        providers: dict[str, ProviderConfig] = {}
        for key, p in (text_raw.get("providers") or {}).items():
            pricing_raw = p.get("pricing") or {}
            providers[key] = ProviderConfig(
                key=key,
                vendor=p.get("vendor", ""),
                protocol=p.get("protocol", "openai_compatible"),
                base_url=p["base_url"],
                api_key=p.get("api_key", ""),
                model=p["model"],
                context_window=p.get("context_window", 128_000),
                max_output_tokens=p.get("max_output_tokens", 4096),
                timeout_seconds=p.get("timeout_seconds", 120.0),
                stream=p.get("stream", True),
                unsupported_params=list(p.get("unsupported_params") or []),
                retry=RetryConfig(**(p.get("retry") or {})),
                pricing=Pricing(
                    input_per_mtok=_num_or_none(pricing_raw.get("input_per_mtok")),
                    output_per_mtok=_num_or_none(pricing_raw.get("output_per_mtok")),
                    cached_input_per_mtok=_num_or_none(pricing_raw.get("cached_input_per_mtok")),
                ),
            )

        cache_raw = raw.get("cache") or {}
        mode = cache_raw.get("mode")
        guard_raw = raw.get("cost_guard") or {}

        return cls(
            text=TextConfig(
                providers=providers,
                roles=text_raw.get("roles") or {},
                role_params=text_raw.get("role_params") or {},
                fallback_chain=text_raw.get("fallback_chain") or [],
                pricing_estimate=_pricing_estimate(text_raw.get("pricing_estimate")),
            ),
            cache=CacheConfig(
                mode=None if mode in _PLACEHOLDERS else mode,
                enabled=cache_raw.get("enabled", True),
            ),
            cost_guard=CostGuard(
                per_task_limit=_num_or_none(guard_raw.get("per_task_limit")),
                per_project_limit=_num_or_none(guard_raw.get("per_project_limit")),
                daily_limit=_num_or_none(guard_raw.get("daily_limit")),
                seconds_limit=_num_or_none(guard_raw.get("seconds_limit")),
                daily_seconds_limit=_num_or_none(guard_raw.get("daily_seconds_limit")),
                on_exceed=guard_raw.get("on_exceed", "pause_and_ask"),
                call_limits={
                    str(k): int(v)
                    for k, v in (guard_raw.get("call_limits") or {}).items()
                    if v not in _PLACEHOLDERS
                },
                daily_call_limits={
                    str(k): int(v)
                    for k, v in (guard_raw.get("daily_call_limits") or {}).items()
                    if v not in _PLACEHOLDERS
                },
            ),
            node_overrides=raw.get("node_overrides") or {},
            raw=raw,
        )

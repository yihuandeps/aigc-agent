"""M2 媒体生成 —— 异步任务模型。

**这是架构里点名「最容易在设计初期漏掉、后期改起来最痛」的一处。**

图像和视频生成不是同步返回的：提交拿到 task_id，然后轮询直到完成。
视频动辄几分钟。如果 Gateway 假设同步返回，接第一个视频模型时就要把
整层重写。所以这套 submit → poll → fetch 在 P2 一次做对。

Provider 抽象出来是为了三件事：
  1. 换供应商只改配置（APIMart / 官方渠道 / 自建都实现同一个协议）
  2. 测试不用真花钱（FakeMediaProvider）
  3. 同步型接口（部分图像模型直接返回）也能套进同一个 MediaTask 形状
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import httpx2 as httpx

from ..events.bus import EventBus, EventType


def default_proxy() -> str | None:
    """从环境变量取代理。

    国内网络下多数外部 API 要走代理才通 —— 这层不做的话，
    连通性问题会被误诊成「DNS 污染」「key 无效」之类，白查半天。
    HTTPS_PROXY / HTTP_PROXY 是事实标准，大小写两种写法都认。
    """
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        v = os.environ.get(key)
        if v:
            return v
    return None


class MediaKind(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class TaskStatus(StrEnum):
    SUBMITTED = "submitted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"

    @property
    def done(self) -> bool:
        return self in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.TIMEOUT)


# 各家状态字符串五花八门，统一映射到上面四个
_STATUS_MAP = {
    "submitted": TaskStatus.SUBMITTED,
    "pending": TaskStatus.SUBMITTED,
    "queued": TaskStatus.SUBMITTED,
    "in_queue": TaskStatus.SUBMITTED,
    "running": TaskStatus.RUNNING,
    "processing": TaskStatus.RUNNING,
    "in_progress": TaskStatus.RUNNING,
    "generating": TaskStatus.RUNNING,
    "succeeded": TaskStatus.SUCCEEDED,
    "success": TaskStatus.SUCCEEDED,
    "completed": TaskStatus.SUCCEEDED,
    "complete": TaskStatus.SUCCEEDED,
    "finished": TaskStatus.SUCCEEDED,
    "failed": TaskStatus.FAILED,
    "failure": TaskStatus.FAILED,
    "error": TaskStatus.FAILED,
    "cancelled": TaskStatus.FAILED,
}


def normalize_status(raw: Any) -> TaskStatus:
    return _STATUS_MAP.get(str(raw or "").strip().lower(), TaskStatus.RUNNING)


@dataclass
class MediaTask:
    task_id: str
    kind: MediaKind
    model: str
    status: TaskStatus = TaskStatus.SUBMITTED
    urls: list[str] = field(default_factory=list)
    error: str | None = None
    cost: float | None = None
    elapsed_s: float = 0.0
    polls: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is TaskStatus.SUCCEEDED and bool(self.urls)


class MediaProvider(Protocol):
    name: str

    async def submit(
        self, kind: MediaKind, model: str, prompt: str, **params: Any
    ) -> MediaTask: ...
    async def poll(self, task: MediaTask) -> MediaTask: ...
    async def close(self) -> None: ...


# ---------------------------------------------------------------- URL 提取


def extract_urls(payload: Any) -> list[str]:
    """从任意响应结构里递归捞出媒体 URL。

    各家返回结构差异很大（data[].url / output[] / video_url / result.urls…），
    与其为每家写一个解析器，不如认「长得像 URL 的字符串」—— 加新模型时不用改代码。
    """
    found: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, str):
            s = v.strip()
            if s.startswith(("http://", "https://")):
                found.append(s)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(payload)
    seen: set[str] = set()
    return [u for u in found if not (u in seen or seen.add(u))]


def _find_task_id(payload: Any) -> str:
    for key in ("task_id", "taskId", "id", "request_id"):
        v = _deep_get(payload, key)
        if isinstance(v, str) and v:
            return v
    return ""


def _deep_get(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for v in payload.values():
            r = _deep_get(v, key)
            if r is not None:
                return r
    elif isinstance(payload, (list, tuple)):
        for v in payload:
            r = _deep_get(v, key)
            if r is not None:
                return r
    return None


# ---------------------------------------------------------------- APIMart


class ApiMartProvider:
    """APIMart 聚合网关。

    契约（来自官方文档）：
      提交  POST {base}/videos/generations  或  {base}/images/generations
            → {"code":200,"data":[{"status":"submitted","task_id":"..."}]}
      轮询  GET  {base}/tasks/{task_id}
            → 直到 status 为 completed / failed

    部分图像模型可能同步返回结果（响应里直接带 URL），
    这种情况下 submit 会直接把 task 标成 SUCCEEDED，不进轮询。
    """

    name = "apimart"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 60.0,
        proxy: str | None = None,
        submit_path: str = "/{kind}s/generations",
        poll_path: str = "/tasks/{task_id}",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.submit_path = submit_path
        self.poll_path = poll_path
        self.proxy = proxy or default_proxy()
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

    async def submit(self, kind: MediaKind, model: str, prompt: str, **params: Any) -> MediaTask:
        body = {
            "model": model,
            "prompt": prompt,
            **{k: v for k, v in params.items() if v is not None},
        }
        path = self.submit_path.format(kind=kind.value)
        resp = await self._http().post(path, json=body)
        payload = _safe_json(resp)

        if resp.status_code >= 400:
            return MediaTask(
                task_id="",
                kind=kind,
                model=model,
                status=TaskStatus.FAILED,
                error=f"HTTP {resp.status_code}：{_err_text(payload) or resp.text[:200]}",
                raw=payload,
            )

        urls = extract_urls(payload)
        task_id = _find_task_id(payload)

        # 同步返回：响应里已经带 URL，不用轮询
        if urls and not task_id:
            return MediaTask(
                task_id="sync",
                kind=kind,
                model=model,
                status=TaskStatus.SUCCEEDED,
                urls=urls,
                raw=payload,
            )

        if not task_id:
            return MediaTask(
                task_id="",
                kind=kind,
                model=model,
                status=TaskStatus.FAILED,
                error=f"响应里既没有 task_id 也没有 URL：{str(payload)[:300]}",
                raw=payload,
            )

        return MediaTask(
            task_id=task_id,
            kind=kind,
            model=model,
            status=normalize_status(_deep_get(payload, "status") or "submitted"),
            urls=urls,
            raw=payload,
        )

    async def poll(self, task: MediaTask) -> MediaTask:
        resp = await self._http().get(self.poll_path.format(task_id=task.task_id))
        payload = _safe_json(resp)
        task.polls += 1
        task.raw = payload

        if resp.status_code >= 400:
            task.status = TaskStatus.FAILED
            task.error = f"HTTP {resp.status_code}：{_err_text(payload) or resp.text[:200]}"
            return task

        task.status = normalize_status(_deep_get(payload, "status"))
        urls = extract_urls(payload)
        if urls:
            task.urls = urls
            # 有些实现拿到结果时状态字段仍是 running，以有无产物为准
            if task.status is TaskStatus.RUNNING:
                task.status = TaskStatus.SUCCEEDED
        if task.status is TaskStatus.FAILED and not task.error:
            task.error = _err_text(payload) or "生成失败，未给出原因"
        return task

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _safe_json(resp: Any) -> dict[str, Any]:
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        return {"_raw_text": getattr(resp, "text", "")[:500]}
    return data if isinstance(data, dict) else {"data": data}


def _err_text(payload: Any) -> str:
    for key in ("message", "error", "msg", "detail"):
        v = _deep_get(payload, key)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict):
            inner = _deep_get(v, "message")
            if isinstance(inner, str) and inner:
                return inner
    return ""


# ---------------------------------------------------------------- 轮询编排


class MediaGateway:
    """submit → poll → 取回。带退避、超时、事件。"""

    def __init__(
        self,
        providers: dict[str, MediaProvider],
        bus: EventBus,
        poll_interval: float = 3.0,
        poll_backoff: float = 1.3,
        max_poll_interval: float = 20.0,
        max_wait_s: float = 600.0,
        max_queue_s: float | None = None,
        max_polls: int = 0,
        max_transient: int = 10,
        submit_retries: int = 2,
    ) -> None:
        self.providers = providers
        self.bus = bus
        self.poll_interval = poll_interval
        self.poll_backoff = poll_backoff
        self.max_poll_interval = max_poll_interval
        self.max_wait_s = max_wait_s
        # 排队 watchdog 的预算。None = 与生成预算相同。
        self.max_queue_s = max_queue_s
        # 轮询请求次数上限（0 = 不限）。用户 2026-09-18 定的口径：每 3 秒一次、最多 200 次。
        self.max_polls = max_polls
        # 网络抖动容忍：轮询连续多少次网络错误才判失败。之前一次 ConnectError 就把任务
        # 判死 —— 服务端其实还在跑、也照样计费，片段却丢了。
        self.max_transient = max_transient
        # 提交阶段的网络错误重试次数（HTTP 4xx 不重试，那是请求本身的问题）
        self.submit_retries = submit_retries

    async def generate(
        self,
        provider: str,
        kind: MediaKind,
        model: str,
        prompt: str,
        max_wait_s: float | None = None,
        **params: Any,
    ) -> MediaTask:
        p = self.providers.get(provider)
        if p is None:
            return MediaTask(
                task_id="",
                kind=kind,
                model=model,
                status=TaskStatus.FAILED,
                error=(
                    f"未配置 provider {provider!r}。"
                    f"可用：{', '.join(self.providers) or '（无）'}"
                ),
            )

        started = time.perf_counter()
        await self.bus.emit(
            EventType.MODEL_REQUEST,
            modality=kind.value,
            provider=provider,
            model=model,
            prompt_len=len(prompt),
        )

        task = await self._submit(p, kind, model, prompt, params)
        budget = max_wait_s if max_wait_s is not None else self.max_wait_s
        interval = self.poll_interval
        polls = 0  # 本次发出的轮询请求数（网络失败的也算一次请求）
        transient = 0  # 连续网络错误次数

        # 计时规则（用户 2026-09-17 定的）：排队/等依赖的时间**不吃生成预算**。
        # 生成计时从任务真正开跑（状态变 running）那一刻起算 —— 服务端队列
        # 拥堵时任务可能排很久才开跑，从提交就计时会把排队时间误算成生成超时。
        # 排队也不是无限等：queue_budget 是 watchdog，超了按「排队超时」报，
        # 和「生成超时」区分开 —— 一个是服务商太忙，一个是任务本身卡了。
        queue_budget = self.max_queue_s or budget
        queued_since = time.perf_counter()
        run_started: float | None = None

        while not task.status.done:
            now = time.perf_counter()
            if task.status is TaskStatus.RUNNING:
                if run_started is None:
                    run_started = now
                if now - run_started > budget:
                    task.status = TaskStatus.TIMEOUT
                    task.error = f"生成超时（开跑后 >{budget:.0f}s，已轮询 {task.polls} 次）"
                    break
            elif now - queued_since > queue_budget:
                task.status = TaskStatus.TIMEOUT
                task.error = (
                    f"排队超时（>{queue_budget:.0f}s 仍未开跑，已轮询 {task.polls} 次）"
                )
                break
            if self.max_polls and polls >= self.max_polls:
                task.status = TaskStatus.TIMEOUT
                task.error = (
                    f"轮询 {polls} 次（每 {self.poll_interval:g}s 一次）仍未完成，放弃等待"
                )
                break
            await asyncio.sleep(interval)
            interval = min(interval * self.poll_backoff, self.max_poll_interval)
            polls += 1
            try:
                task = await p.poll(task)
                transient = 0
            except (httpx.TransportError, OSError) as e:
                # 网络抖动：任务在服务端照常跑，这里只是没问到 —— 接着问，别把任务判死
                transient += 1
                if transient > self.max_transient:
                    task.status = TaskStatus.FAILED
                    task.error = (
                        f"轮询连续 {transient} 次网络错误，放弃：{type(e).__name__}: {e}"
                    )
                    break
            except Exception as e:  # noqa: BLE001
                task.status = TaskStatus.FAILED
                task.error = f"轮询失败 {type(e).__name__}: {e}"
                break

        task.elapsed_s = round(time.perf_counter() - started, 1)
        await self.bus.emit(
            EventType.MODEL_RESPONSE,
            modality=kind.value,
            model=model,
            status=task.status.value,
            urls=len(task.urls),
            elapsed_s=task.elapsed_s,
            polls=task.polls,
            error=task.error,
        )
        return task

    async def _submit(
        self, p: MediaProvider, kind: MediaKind, model: str, prompt: str, params: dict[str, Any]
    ) -> MediaTask:
        """提交，网络错误重试几次（HTTP 4xx/5xx 由 provider 转成 FAILED，不在这里重试）。"""
        for attempt in range(self.submit_retries + 1):
            try:
                return await p.submit(kind, model, prompt, **params)
            except (httpx.TransportError, OSError) as e:
                if attempt >= self.submit_retries:
                    return MediaTask(
                        task_id="",
                        kind=kind,
                        model=model,
                        status=TaskStatus.FAILED,
                        error=f"提交失败（网络错误，已重试 {attempt} 次）：{type(e).__name__}: {e}",
                    )
                await asyncio.sleep(min(2.0 * (attempt + 1), 10.0))
        raise AssertionError("unreachable")

    async def close(self) -> None:
        for p in self.providers.values():
            await p.close()


# ---------------------------------------------------------------- 测试替身


class FakeMediaProvider:
    """测试用。可控制需要轮询几次、是否失败。"""

    name = "fake"

    def __init__(
        self,
        polls_needed: int = 2,
        urls: list[str] | None = None,
        fail_at: str | None = None,  # submit | poll
        sync: bool = False,
    ) -> None:
        self.polls_needed = polls_needed
        self.urls = urls or ["https://example.com/out.png"]
        self.fail_at = fail_at
        self.sync = sync
        self.submitted: list[dict[str, Any]] = []

    async def submit(self, kind, model, prompt, **params):
        self.submitted.append({"kind": kind, "model": model, "prompt": prompt, **params})
        if self.fail_at == "submit":
            return MediaTask(
                task_id="", kind=kind, model=model, status=TaskStatus.FAILED, error="模拟提交失败"
            )
        if self.sync:
            return MediaTask(
                task_id="sync",
                kind=kind,
                model=model,
                status=TaskStatus.SUCCEEDED,
                urls=list(self.urls),
            )
        return MediaTask(task_id="t_fake", kind=kind, model=model)

    async def poll(self, task):
        task.polls += 1
        if self.fail_at == "poll":
            task.status = TaskStatus.FAILED
            task.error = "模拟生成失败"
            return task
        if task.polls >= self.polls_needed:
            task.status = TaskStatus.SUCCEEDED
            task.urls = list(self.urls)
        else:
            task.status = TaskStatus.RUNNING
        return task

    async def close(self) -> None:
        return None

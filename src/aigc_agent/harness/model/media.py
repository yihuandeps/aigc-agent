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
    "canceled": TaskStatus.FAILED,
    "expired": TaskStatus.FAILED,
    "rejected": TaskStatus.FAILED,
    "timeout": TaskStatus.FAILED,
    "timed_out": TaskStatus.FAILED,
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
    # 失败发生在哪一步：submit（服务端大概率没建任务）/ poll（任务已建、还在计费）
    stage: str = ""
    # 原样重新提交安全吗 —— 只有「请求确实没送到 / 被明确拒收（429）」才安全。
    # 轮询阶段的失败永远不安全：任务在服务端照跑，重提就是付两份钱（2026-09-23 审查）
    retryable: bool = False
    http_status: int = 0
    # 这次结果是从台账里取回的旧任务（没有重新付费）
    recovered: bool = False

    @property
    def ok(self) -> bool:
        return self.status is TaskStatus.SUCCEEDED and bool(self.urls)


class PollTransientError(Exception):
    """轮询时服务端/网关的暂时性错误（429、5xx）：任务本身没事，接着问。"""


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
            code = resp.status_code
            return MediaTask(
                task_id="",
                kind=kind,
                model=model,
                status=TaskStatus.FAILED,
                error=f"HTTP {code}：{_err_text(payload) or resp.text[:200]}",
                raw=payload,
                stage="submit",
                http_status=code,
                # 429 是被拒收（没建任务、不扣费）；503 是服务不可用（没转到后端）。
                # 502/504 是上游没及时回 —— 和 ReadTimeout 一样，上游可能已经建了任务，
                # 原样重提可能付两份（2026-09-24 审查）。400/401/403/500 等说明请求本身或
                # 服务端有问题，重提没意义或有风险
                retryable=code in (429, 503),
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
            code = resp.status_code
            detail = f"HTTP {code}：{_err_text(payload) or resp.text[:200]}"
            if code in (408, 425, 429) or code >= 500:
                # 网关偶发 5xx / 限流：任务在服务端照常跑，只是这次没问到。之前一次就判死，
                # 上层再把它当网络抖动重新提交 —— 同一个镜头付两份钱（2026-09-23 审查）
                raise PollTransientError(detail)
            task.status = TaskStatus.FAILED
            task.error = detail
            task.http_status = code
            task.stage = "poll"
            return task

        task.status = normalize_status(_deep_get(payload, "status"))
        urls = extract_urls(payload)
        if urls:
            task.urls = urls
            # 有些实现拿到结果时状态字段仍是 running，以有无产物为准 —— 但进度明说没到头的不算：
            # 有的实现先吐预览/中间产物的链接，之前一见 URL 就判成功（2026-09-23 审查）
            if task.status is TaskStatus.RUNNING and not _in_progress(payload):
                task.status = TaskStatus.SUCCEEDED
        if task.status is TaskStatus.FAILED and not task.error:
            task.error = _err_text(payload) or "生成失败，未给出原因"
        return task

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _in_progress(payload: Any) -> bool:
    """进度字段明说还没完：0<p<1（小数制）或 1<p<100（百分制），「45%」这种也认。

    缺字段、0、1、100 都不算没完 —— 0 常见于「这家不更新进度」，1 可能是小数制的 100%。
    """
    raw = _deep_get(payload, "progress")
    if isinstance(raw, str):
        raw = raw.strip().rstrip("%")
    try:
        p = float(raw)
    except (TypeError, ValueError):
        return False
    return 0 < p < 1 or 1 < p < 100


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
    """submit → poll → 取回。带退避、超时、事件、任务台账。

    2026-09-23 审查后加的四样：
      · **任务台账**（ledger）：提交成功就记下 task_id；取消、超时、轮询放弃也照记。
        钱花了拿不到结果的任务，事后能用同一个 task_id 取回。
      · **先取回、再提交**：同一份请求（指纹相同）在台账里还有没交付的任务，就去取回它，
        不重新付费 —— 上层的「失败重试」因此不会再把还在跑的任务又提交一遍。
      · **按模态共享并发**：信号量放在网关里，两批渲染同时跑也不会把服务商的并发打爆
        （之前每次调用各建一个信号量，两批并发就翻倍）。
      · **分级重试**：提交只在「请求确实没送到」时重试，429 退避后再试；轮询时的
        429/5xx 当网络抖动接着问，不判死。
    """

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
        ledger: Any = None,
        concurrency: dict[str, int] | None = None,
        rate_limit_retries: int = 4,
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
        # 提交阶段「请求没送到」的重试次数（HTTP 4xx 不重试，那是请求本身的问题）
        self.submit_retries = submit_retries
        # 提交被 429 拒收（没建任务、不扣费）时退避重试几次
        self.rate_limit_retries = rate_limit_retries
        # 任务台账（MediaTaskLedger）。None = 不记（脚本 / 测试）
        self.ledger = ledger
        # 各模态同时在跑的任务上限（image / video / audio）。0 或缺 = 不限
        self.concurrency = dict(concurrency or {})
        self._sems: dict[str, asyncio.Semaphore] = {}
        # 本进程正在轮询的 task_id：台账里「没交付」的任务如果正被本进程等着，不能拿去取回
        self._inflight: set[str] = set()

    def _sem(self, kind: MediaKind) -> asyncio.Semaphore | None:
        n = int(self.concurrency.get(kind.value) or 0)
        if n <= 0:
            return None
        sem = self._sems.get(kind.value)
        if sem is None:
            sem = self._sems[kind.value] = asyncio.Semaphore(n)
        return sem

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
        sem = self._sem(kind)
        if sem is None:
            return await self._generate(p, provider, kind, model, prompt, max_wait_s, params)
        async with sem:
            return await self._generate(p, provider, kind, model, prompt, max_wait_s, params)

    async def _generate(
        self,
        p: MediaProvider,
        provider: str,
        kind: MediaKind,
        model: str,
        prompt: str,
        max_wait_s: float | None,
        params: dict[str, Any],
    ) -> MediaTask:
        started = time.perf_counter()
        fingerprint = task_fingerprint(kind, model, prompt, params)
        await self.bus.emit(
            EventType.MODEL_REQUEST,
            modality=kind.value,
            provider=provider,
            model=model,
            prompt_len=len(prompt),
        )

        # 1) 同一份请求在台账里还有没交付的任务：先取回它（服务端可能早就生成完了）
        rec = None
        if self.ledger is not None:
            rec = self.ledger.recoverable(fingerprint, exclude=self._inflight)
        if rec is not None:
            await self.bus.emit(
                EventType.WARNING,
                message=(
                    f"同一份{kind.value}请求在台账里有没取回的任务 {rec.task_id}"
                    f"（{rec.status}），先取回它，不重新提交"
                ),
            )
            self.ledger.recovered(rec.task_id)
            task = MediaTask(
                task_id=rec.task_id, kind=kind, model=rec.model or model,
                status=TaskStatus.RUNNING, recovered=True,
            )
            task = await self._await_task(p, task, max_wait_s)
            if task.ok or task.status is not TaskStatus.FAILED:
                return await self._finish(task, started)
            # 旧任务确实失败了（服务端判的，不是我们没问到）：这次正常提交一个新的
            await self.bus.emit(
                EventType.WARNING, message=f"台账里的任务 {rec.task_id} 已失败，重新提交"
            )

        # 2) 正常提交
        task = await self._submit(p, kind, model, prompt, params)
        if task.task_id and task.task_id != "sync" and self.ledger is not None:
            self.ledger.submitted(
                task.task_id, kind=kind.value, model=model, provider=provider,
                fingerprint=fingerprint, prompt=prompt, params=params,
            )
        task = await self._await_task(p, task, max_wait_s)
        return await self._finish(task, started)

    async def recover(
        self, provider: str, task_id: str, kind: MediaKind, model: str,
        max_wait_s: float | None = None,
    ) -> MediaTask:
        """按 task_id 取回一个之前没拿到结果的任务（轮询到结束，不提交新的）。"""
        p = self.providers.get(provider)
        if p is None:
            return MediaTask(
                task_id=task_id, kind=kind, model=model, status=TaskStatus.FAILED,
                error=f"未配置 provider {provider!r}",
            )
        started = time.perf_counter()
        task = MediaTask(
            task_id=task_id, kind=kind, model=model, status=TaskStatus.RUNNING, recovered=True
        )
        task = await self._await_task(p, task, max_wait_s)
        return await self._finish(task, started)

    async def _await_task(
        self, p: MediaProvider, task: MediaTask, max_wait_s: float | None
    ) -> MediaTask:
        """轮询到结束。被取消（/stop、工具超时）时把 task_id 记成 abandoned 再往外抛。"""
        budget = max_wait_s if max_wait_s is not None else self.max_wait_s
        interval = self.poll_interval
        polls = 0  # 本次发出的轮询请求数（网络失败的也算一次请求）
        transient = 0  # 连续网络错误次数
        tracked = bool(task.task_id) and task.task_id != "sync"

        # 计时规则（用户 2026-09-17 定的）：排队/等依赖的时间**不吃生成预算**。
        # 生成计时从任务真正开跑（状态变 running）那一刻起算 —— 服务端队列
        # 拥堵时任务可能排很久才开跑，从提交就计时会把排队时间误算成生成超时。
        # 排队也不是无限等：queue_budget 是 watchdog，超了按「排队超时」报，
        # 和「生成超时」区分开 —— 一个是服务商太忙，一个是任务本身卡了。
        queue_budget = self.max_queue_s or budget
        queued_since = time.perf_counter()
        run_started: float | None = None
        if tracked:
            self._inflight.add(task.task_id)
        try:
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
                except (httpx.TransportError, OSError, PollTransientError) as e:
                    # 网络抖动 / 网关 5xx / 限流：任务在服务端照常跑，这里只是没问到 ——
                    # 接着问，别把任务判死
                    transient += 1
                    if transient > self.max_transient:
                        task.status = TaskStatus.TIMEOUT
                        task.error = (
                            f"轮询连续 {transient} 次没问到（{type(e).__name__}: {e}），放弃等待"
                        )
                        break
                except Exception as e:  # noqa: BLE001
                    task.status = TaskStatus.TIMEOUT
                    task.error = f"轮询出错 {type(e).__name__}: {e}"
                    break
        except asyncio.CancelledError:
            if tracked and self.ledger is not None:
                self.ledger.update(
                    task.task_id, "abandoned",
                    error="本地放弃等待（/stop 或工具超时），服务端可能仍在生成",
                )
            raise
        finally:
            if tracked:
                self._inflight.discard(task.task_id)

        if tracked:
            if task.status is TaskStatus.TIMEOUT:
                # 本地没等到 ≠ 任务失败：它多半还在服务端跑、照样计费。记下来，事后能取回；
                # 也明确告诉上层**不要原样重提**
                task.stage = "poll"
                task.retryable = False
                task.error = (
                    f"{task.error}。任务 {task.task_id} 可能仍会在服务端完成（已计费）："
                    "用 media_tasks 查看、media_recover 取回，不要原样重新提交"
                )
                if self.ledger is not None:
                    self.ledger.update(task.task_id, "timeout", error=task.error or "")
            elif self.ledger is not None:
                if task.ok:
                    self.ledger.update(task.task_id, "succeeded", urls=task.urls)
                else:
                    self.ledger.update(task.task_id, "failed", error=task.error or "")
        return task

    async def _finish(self, task: MediaTask, started: float) -> MediaTask:
        task.elapsed_s = round(time.perf_counter() - started, 1)
        await self.bus.emit(
            EventType.MODEL_RESPONSE,
            modality=task.kind.value,
            model=task.model,
            status=task.status.value,
            urls=len(task.urls),
            elapsed_s=task.elapsed_s,
            polls=task.polls,
            error=task.error,
            task_id=task.task_id,
            recovered=task.recovered,
        )
        return task

    async def _submit(
        self, p: MediaProvider, kind: MediaKind, model: str, prompt: str, params: dict[str, Any]
    ) -> MediaTask:
        """提交。只在「请求确实没送到」时重试；被 429 拒收时退避后再试。

        · 连不上 / 连接超时 / 连接池等不到：请求没发出去，重试安全
        · 读超时 / 连接中途断：请求**可能已经送达**、服务端可能已经建了任务 ——
          不重试（之前会重试，一个镜头最多被提交 6 次），交给上层看台账决定
        · HTTP 429：被拒收，没建任务、不扣费，按退避重试
        """
        attempt = limited = 0
        while True:
            try:
                task = await p.submit(kind, model, prompt, **params)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
                if attempt >= self.submit_retries:
                    return MediaTask(
                        task_id="", kind=kind, model=model, status=TaskStatus.FAILED,
                        error=(
                            f"提交失败（连不上服务端，已重试 {attempt} 次）："
                            f"{type(e).__name__}: {e}"
                        ),
                        stage="submit", retryable=True,
                    )
                attempt += 1
                await asyncio.sleep(min(2.0 * attempt, 10.0))
                continue
            except (httpx.TransportError, OSError) as e:
                return MediaTask(
                    task_id="", kind=kind, model=model, status=TaskStatus.FAILED,
                    error=(
                        f"提交后没收到响应（{type(e).__name__}: {e}）：服务端可能已经建了任务、"
                        "开始计费，所以没有自动重提。过几分钟看服务商控制台或重试前先确认"
                    ),
                    stage="submit", retryable=False,
                )
            if (
                task.status is TaskStatus.FAILED
                and task.http_status == 429
                and limited < self.rate_limit_retries
            ):
                limited += 1
                await asyncio.sleep(min(5.0 * (2 ** (limited - 1)), 60.0))
                continue
            return task

    async def close(self) -> None:
        # 2026-09-24 审查：这个方法曾被误缩进到模块级函数 task_fingerprint 的 return 之后，
        # 成了死代码 —— httpx 连接池从此没人关，每次退出刷一屏 athrow 堆栈
        for p in self.providers.values():
            await p.close()


def task_fingerprint(kind: MediaKind, model: str, prompt: str, params: dict[str, Any]) -> str:
    """同一份生成请求的指纹：模态 + 模型 + 提示词 + 会影响产物的参数。"""
    import hashlib
    import json

    body = json.dumps(
        {"k": kind.value, "m": model, "p": prompt, "x": params},
        ensure_ascii=False, sort_keys=True, default=str,
    )
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:20]


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

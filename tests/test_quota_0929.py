"""余额 / 额度用完：不重试、说清楚、停下整批（2026-09-29 审查 1.1）。

现场：APIMart 余额不足先回 500（insufficient quota: balance=…, required=…）再回 402；500 在 retry_on
里照样重试，9-27 两次 drama_shots 共重试 12 次，原始 402 直接甩给主模型；9-23 连着 20 次 402，
主模型当成模型的问题换了两次模型；媒体一批 8 段并发全部 402，排队的照样一个个去撞。
"""

from __future__ import annotations

from typing import Any

import httpx2 as httpx
import pytest
from openai import APIStatusError, RateLimitError

from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.functions.drama import DramaFunctions
from aigc_agent.domain.functions.media import MediaFunctions
from aigc_agent.domain.generators.catalog import Concurrency, MediaCatalog
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.model.gateway import classify_model_error
from aigc_agent.harness.model.media import MediaGateway, MediaKind, MediaTask, TaskStatus
from aigc_agent.harness.model.quota import QuotaExhaustedError, is_quota_error, quota_advice
from aigc_agent.harness.tools.provider import ToolResult
from tests.test_budget_gate import CATALOG
from tests.test_drama_render import SHOTS, FakeRegistry, _shots_asset
from tests.test_net_resilience import Flaky, _gateway
from tests.test_review_0924 import _ChunkGateway, _shots_setup

APIMART_500 = "insufficient quota: balance=85356, required=102911"


def _err(code: int, msg: str, cls: type = APIStatusError) -> Any:
    req = httpx.Request("POST", "https://api.example.com/v1")
    return cls(msg, response=httpx.Response(code, request=req), body=None)


# ---------------------------------------------------------------- 识别


def test_认得出余额和额度用完_不误伤限流和服务端故障():
    assert is_quota_error(402, "")
    assert is_quota_error(500, APIMART_500)
    assert is_quota_error(429, "You exceeded your current quota")
    assert is_quota_error(403, "usage limit exceeded for this billing window")
    assert not is_quota_error(500, "internal server error")
    assert not is_quota_error(500, "insufficient resources, try later")
    assert not is_quota_error(429, "rate limit reached, slow down")
    assert not is_quota_error(400, "insufficient quota")
    assert not is_quota_error(None, "quota")


def test_提示分余额和额度窗口_都说别换模型():
    assert "充值" in quota_advice("apimart", 402) and "充值" in quota_advice("apimart", 500)
    assert "额度窗口" in quota_advice("kimi", 403) and "额度窗口" in quota_advice("kimi", 429)
    e = QuotaExhaustedError("apimart", 500, APIMART_500, role="drama")
    assert "drama" in str(e) and "apimart" in str(e) and "换模型" in str(e)
    assert classify_model_error(e)[0] == "quota"


# ---------------------------------------------------------------- 文本网关


async def test_余额不足的500不重试():
    gw, provider = _gateway(EventBus(), max_attempts=3)
    flaky = Flaky(fails=9, exc=lambda: _err(500, APIMART_500))
    gw._call_once = flaky  # type: ignore[method-assign]
    with pytest.raises(APIStatusError):
        await gw._with_retry(provider, {}, use_stream=False)
    assert flaky.calls == 1


async def test_额度用完的429不重试_普通限流照旧重试():
    gw, provider = _gateway(EventBus(), max_attempts=3, connect_max_attempts=3)
    quota = Flaky(fails=9, exc=lambda: _err(429, "You exceeded your current quota", RateLimitError))
    gw._call_once = quota  # type: ignore[method-assign]
    with pytest.raises(RateLimitError):
        await gw._with_retry(provider, {}, use_stream=False)
    assert quota.calls == 1
    limited = Flaky(fails=1, exc=lambda: _err(429, "rate limit reached", RateLimitError))
    gw._call_once = limited  # type: ignore[method-assign]
    assert (await gw._with_retry(provider, {}, use_stream=False)).text == "ok"
    assert limited.calls == 2


async def test_chat把余额用完包成说人话的错误():
    gw, _ = _gateway(EventBus())
    gw._call_once = Flaky(fails=9, exc=lambda: _err(402, "insufficient balance"))  # type: ignore
    with pytest.raises(QuotaExhaustedError) as ei:
        await gw.chat("main_agent", [{"role": "user", "content": "hi"}])
    assert ei.value.provider == "p" and ei.value.status == 402
    assert "充值" in str(ei.value) and "main_agent" in str(ei.value)


# ---------------------------------------------------------------- 媒体


class _QuotaFirst:
    """第一次提交回 402；之后的提交真发出去会成功 —— 用来证明后面排队的没发。"""

    name = "fake"

    def __init__(self) -> None:
        self.submitted = 0

    async def submit(self, kind: MediaKind, model: str, prompt: str, **_: Any) -> MediaTask:
        self.submitted += 1
        if self.submitted == 1:
            return MediaTask(
                task_id="", kind=kind, model=model, status=TaskStatus.FAILED,
                error="HTTP 402：insufficient balance", stage="submit", http_status=402,
            )
        return MediaTask(task_id="sync", kind=kind, model=model,
                         status=TaskStatus.SUCCEEDED, urls=["https://x/a.png"])

    async def poll(self, task: MediaTask) -> MediaTask:
        return task

    async def close(self) -> None:
        return None


async def test_批量生图_第一个撞上余额用完_排队的不再发():
    catalog = MediaCatalog.load(CATALOG)
    catalog.concurrency = Concurrency(image=1)  # 一张一张排队：才看得出后面的没发
    provider = _QuotaFirst()
    gw = MediaGateway({catalog.provider: provider}, EventBus(), poll_interval=0.01)
    fns = MediaFunctions(gw, catalog, AssetStore())
    fns.image_lock = next(iter(catalog.image)).id
    r = await fns._fn_gen_images(jobs=[{"prompt": f"图{i}"} for i in range(4)])
    assert provider.submitted == 1, "撞上余额用完之后一张都不再发"
    assert not r.ok and r.error.startswith("⛔ 余额 / 额度用完了")
    assert "充值" in r.error and r.error.count("没发") == 3
    assert r.meta["refund_units"] == 4, "一张都没花钱：额度全退"


async def test_媒体网关_额度用完的429不退避重提():
    class Q429(_QuotaFirst):
        async def submit(self, kind: MediaKind, model: str, prompt: str, **_: Any) -> MediaTask:
            self.submitted += 1
            return MediaTask(
                task_id="", kind=kind, model=model, status=TaskStatus.FAILED,
                error="HTTP 429：quota exceeded", stage="submit", http_status=429,
            )

    p = Q429()
    gw = MediaGateway({"fake": p}, EventBus())
    task = await gw._submit(p, MediaKind.IMAGE, "m", "x", {})
    assert task.status is TaskStatus.FAILED and p.submitted == 1


# ---------------------------------------------------------------- 短剧


class _QuotaVideos(FakeRegistry):
    async def invoke(self, name: str, args: dict[str, Any]) -> ToolResult:
        if name == "gen_video":
            self.calls[args.get("summary", name)] = args
            return ToolResult(
                ok=False,
                error="seedance 生成失败（failed，0s，轮询 0 次）：HTTP 402：insufficient balance",
                meta={"quota": True, "charged": False, "retryable": False},
            )
        return await super().invoke(name, args)


async def test_渲分镜视频_撞上余额用完_后面的段不发_原因写在最前面():
    store = AssetStore()
    fake = _QuotaVideos(store)
    catalog = MediaCatalog(concurrency=Concurrency(video=1))
    fns = DramaFunctions(None, store, registry=fake, catalog=catalog)
    r = await fns._fn_drama_render_shots(_shots_asset(store, [SHOTS[0], SHOTS[2]]))
    assert not r.ok
    assert len(fake.calls) == 1, "第一段撞上之后，第二段没发"
    assert r.error.startswith("⛔ 余额 / 额度用完了") and "充值" in r.error


async def test_分批出提示词_一批余额用完_报清楚还说出好的存着():
    class Quota(_ChunkGateway):
        async def chat(self, role: str, messages: list[dict[str, Any]], **kw: Any) -> Any:
            if "〔51〕" in messages[1]["content"]:
                self.calls.append(messages)
                raise QuotaExhaustedError("apimart", 402, "insufficient balance", role=role)
            return await super().chat(role, messages, **kw)

    gw = Quota()
    fns, _, sb, lib = _shots_setup(gw)
    r = await fns.invoke("drama_shots", {"storyboard_id": sb, "assets_id": lib})
    assert not r.ok and "QuotaExhaustedError" in r.error and "充值" in r.error
    assert "已经出好的 2 批提示词先存着" in r.error

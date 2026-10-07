"""没配单价的文本 provider 不再按 0 记（2026-10-07，docs/优化-2026-09-29.md 2.1）。

9-25 到 9-27 短剧写作（gemini-3.8-flash，走 APIMart）调用 206 次、输入 374 万 / 输出 369 万 token，
models.yaml 没给它配 pricing，网关算出 cost=None —— 金额护栏、台账都按 0 记，三天约 ¥90–100
没进账（同期账本只记了 ¥24），开工面板却写着「金额（含文本模型）」。

现在：
  · 没配价的按 text.pricing_estimate 的保守估价另算一份 est_cost（cost 仍是 None），
    金额护栏、台账照样计入；展示金额的地方标「估算」、点名是哪几家
  · 台账把输入 / 输出 / 缓存命中 token 分开记，以后补了真实单价能回算
  · 金额口径不再因为它判「失效」
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.model.budget import CostGuard
from aigc_agent.harness.model.config import (
    DEFAULT_PRICING_ESTIMATE,
    ModelsConfig,
    Pricing,
    ProviderConfig,
    TextConfig,
)
from aigc_agent.harness.model.gateway import ModelGateway, ModelResponse, Usage
from aigc_agent.harness.model.ledger import CostLedger, today
from aigc_agent.interfaces.cli.budget_prompt import render_budget

ROOT = Path(__file__).resolve().parents[1]

# 9-25 到 9-27 三天 gemini 的量（docs/优化-2026-09-29.md 2.1）
_IN, _OUT = 3_740_000, 3_690_000
_MSG = [{"role": "user", "content": "拆第 1 集分镜"}]


def _provider(key: str, pricing: Pricing | None = None) -> ProviderConfig:
    return ProviderConfig(
        key=key, model=f"{key}-model", api_key="sk-x", base_url="https://api.example.com/v1",
        stream=False, pricing=pricing or Pricing(),
    )


def _gateway(bus: EventBus, estimate: Pricing | None = DEFAULT_PRICING_ESTIMATE) -> ModelGateway:
    kimi = Pricing(input_per_mtok=20, output_per_mtok=100, cached_input_per_mtok=2)
    cfg = ModelsConfig(
        text=TextConfig(
            providers={"kimi_k3": _provider("kimi_k3", kimi), "gemini": _provider("gemini")},
            roles={"main_agent": "kimi_k3", "drama": "gemini"},
            pricing_estimate=estimate,
        )
    )
    return ModelGateway(cfg, bus)


def _answer(prompt: int, completion: int, cached: int = 0) -> Any:
    """顶替 _call_once：不连网，回一份带 usage 的响应。"""

    async def call(provider: Any, kwargs: dict[str, Any]) -> ModelResponse:
        return ModelResponse(text="ok", usage=Usage(prompt, completion, cached))

    return call


def _warnings(bus: EventBus) -> list[str]:
    return [str(e.data.get("message")) for e in bus.history if e.type is EventType.WARNING]


# ---------------------------------------------------------------- 配置


def test_真实配置_没配单价的provider都按保守估价():
    cfg = ModelsConfig.load(ROOT / "config" / "models.yaml")
    est = cfg.text.pricing_estimate
    assert est is not None
    # 反推的上沿：输入 $0.9、输出 $3.0 每百万 token，按 1 美元 = 7.2 元换算
    assert est.input_per_mtok == 6.48 and est.output_per_mtok == 21.6
    assert est.cached_input_per_mtok is None, "缓存命中不打折：估价宁高勿低"
    assert est == DEFAULT_PRICING_ESTIMATE, "yaml 和代码里的默认估价是同一组数"
    # 有角色在用、没配单价的每一家都要点名（现在是 gemini；用户查到真价填上后它就不在了）
    used = set(cfg.text.roles.values())
    unpriced = [k for k, p in cfg.text.providers.items() if k in used and not p.pricing.known]
    named = cfg.text.estimated_providers()
    assert [n.split("（")[0] for n in named] == unpriced
    assert all("6.48" in n and "21.6" in n for n in named)
    assert cfg.text.providers["kimi_k3"].pricing.known, "配了参考价的不算估价"


def test_估价段没写或写TODO都按默认估_写false才不估(tmp_path: Path):
    base = (
        "text:\n  providers:\n    g:\n      base_url: https://x\n      api_key: k\n      model: m\n"
        "  roles:\n    drama: g\n"
    )

    def load(extra: str) -> ModelsConfig:
        p = tmp_path / "models.yaml"
        p.write_text(base + extra, encoding="utf-8")
        return ModelsConfig.load(p)

    # 这一段删掉了也照样估 —— 金额护栏不该因为少填一项配置就看不见一家模型
    assert load("").text.pricing_estimate == DEFAULT_PRICING_ESTIMATE
    assert load("").text.estimated_providers() == [
        "g（m，按输入 ¥6.48 / 输出 ¥21.6 每百万 token 估）"
    ]
    half = load("  pricing_estimate:\n    input_per_mtok: 3\n    output_per_mtok: TODO\n")
    est = half.text.pricing_estimate
    assert est is not None and est.input_per_mtok == 3
    assert est.output_per_mtok == DEFAULT_PRICING_ESTIMATE.output_per_mtok, "没填的项用默认值"
    off = load("  pricing_estimate: false\n")
    assert off.text.pricing_estimate is None and off.text.estimated_providers() == []


# ---------------------------------------------------------------- 网关


async def test_没配单价的provider_网关按估价另给est_cost_cost仍是None():
    bus = EventBus()
    gw = _gateway(bus)
    gw._call_once = _answer(_IN, _OUT)  # type: ignore[method-assign]
    resp = await gw.chat("drama", _MSG)

    assert resp.usage.cost is None, "cost 只放真实单价算出来的钱"
    expected = _IN / 1e6 * 6.48 + _OUT / 1e6 * 21.6  # ≈ ¥103.9：和反推的 ¥90–100 同一量级、偏高
    assert resp.usage.est_cost is not None and abs(resp.usage.est_cost - expected) < 1e-6
    cost = [e for e in bus.history if e.type is EventType.COST][-1].data
    assert cost["cost"] is None and abs(cost["est_cost"] - expected) < 1e-6
    assert cost["provider"] == "gemini" and cost["role"] == "drama"
    assert (cost["prompt_tokens"], cost["completion_tokens"]) == (_IN, _OUT)

    warns = _warnings(bus)
    assert len(warns) == 1
    assert "gemini" in warns[0] and "估价" in warns[0] and "6.48" in warns[0]
    assert "pricing" in warns[0], "告诉人查到真实单价往哪填"
    await gw.chat("drama", _MSG)
    assert len(_warnings(bus)) == 1, "每家只提醒一次"


async def test_配了单价的照旧按真价算_不出估算():
    bus = EventBus()
    gw = _gateway(bus)
    gw._call_once = _answer(1_000_000, 100_000, cached=500_000)  # type: ignore[method-assign]
    resp = await gw.chat("main_agent", _MSG)
    assert resp.usage.cost is not None
    assert abs(resp.usage.cost - (0.5 * 20 + 0.5 * 2 + 0.1 * 100)) < 1e-9
    assert resp.usage.est_cost is None
    cost = [e for e in bus.history if e.type is EventType.COST][-1].data
    assert cost["est_cost"] is None and cost["provider"] == "kimi_k3"
    assert _warnings(bus) == []


async def test_估价关掉时照旧记无单价():
    bus = EventBus()
    guard = CostGuard(money_limit=1.0)
    guard.attach(bus)
    gw = _gateway(bus, estimate=None)
    gw._call_once = _answer(_IN, _OUT)  # type: ignore[method-assign]
    resp = await gw.chat("drama", _MSG)
    assert resp.usage.cost is None and resp.usage.est_cost is None
    assert guard.usage.unpriced == 1 and guard.usage.money == 0
    assert "成本无法核算" in _warnings(bus)[0]


# ---------------------------------------------------------------- 金额护栏


async def test_金额护栏看得见没配价的gemini_超限时说清其中多少是估算():
    bus = EventBus()
    guard = CostGuard(money_limit=50.0)
    guard.attach(bus)
    gw = _gateway(bus)
    gw._call_once = _answer(_IN, _OUT)  # type: ignore[method-assign]
    await gw.chat("drama", _MSG)

    assert guard.usage.money > 100, "之前这里是 0：三天 ¥90–100 护栏看不见"
    assert guard.usage.unpriced == 0
    assert abs(guard.usage.estimated - guard.usage.money) < 1e-9
    assert not guard.blind, "有了估价，金额口径不再算失效"
    v = guard.check()
    assert not v and v.money and v.level == "task"
    assert "其中估算 ¥" in v.reason and "gemini" in v.reason, v.reason


async def test_估算计入单日上限_超了也说清是估算(tmp_path: Path):
    bus = EventBus()
    guard = CostGuard(daily_limit=10.0, ledger=CostLedger(tmp_path / "l.jsonl"), project_id="p")
    guard.attach(bus)
    gw = _gateway(bus)
    gw._call_once = _answer(_IN, _OUT)  # type: ignore[method-assign]
    await gw.chat("drama", _MSG)
    v = guard.check()
    assert not v and v.level == "day" and "其中估算 ¥" in v.reason, v.reason
    assert guard.room("video")["money"] == 0.0, "额度还剩多少也要扣掉估算的钱"


async def test_brief分开写真价和估算_点名哪几家(tmp_path: Path):
    bus = EventBus()
    guard = CostGuard(
        money_limit=200, daily_limit=800, project_limit=400,
        ledger=CostLedger(tmp_path / "l.jsonl"), project_id="p1", session_id="s1",
    )
    guard.attach(bus)
    await bus.emit(
        EventType.COST, role="main_agent", provider="kimi_k3", model="k3",
        prompt_tokens=1000, completion_tokens=100, cached_tokens=0, cost=0.5,
    )
    await bus.emit(
        EventType.COST, role="drama", provider="gemini", model="gemini-3.8-flash",
        prompt_tokens=100_000, completion_tokens=50_000, cached_tokens=10_000,
        cost=None, est_cost=1.728,
    )
    text = guard.brief()
    assert "本次开工 ¥2.2280（其中估算 ¥1.7280：gemini 没配单价）" in text, text
    assert "今日 ¥2.2280（其中估算 ¥1.7280；上限 ¥800.00）" in text, text
    assert "项目 p1 ¥2.2280（其中估算 ¥1.7280；上限 ¥400.00）" in text, text
    assert "无单价" not in text
    assert guard.usage.calls["text"] == 2


# ---------------------------------------------------------------- 台账


async def test_台账分开记输入输出token_标出估算_重读还在(tmp_path: Path):
    path = tmp_path / "costs" / "ledger.jsonl"
    bus = EventBus()
    guard = CostGuard(ledger=CostLedger(path), project_id="p1", session_id="s1")
    guard.attach(bus)
    await bus.emit(
        EventType.COST, role="main_agent", provider="kimi_k3", model="k3",
        prompt_tokens=1000, completion_tokens=100, cached_tokens=600, cost=0.5,
    )
    await bus.emit(
        EventType.COST, role="drama", provider="gemini", model="gemini-3.8-flash",
        prompt_tokens=100_000, completion_tokens=50_000, cached_tokens=10_000,
        cost=None, est_cost=1.728,
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    gem = next(r for r in rows if r["provider"] == "gemini")
    assert gem["estimated"] is True and abs(gem["cost"] - 1.728) < 1e-9
    assert (gem["prompt_tokens"], gem["completion_tokens"], gem["cached_tokens"]) == (
        100_000, 50_000, 10_000,
    )
    assert gem["tokens"] == 150_000 and gem["role"] == "drama"
    kimi = next(r for r in rows if r["provider"] == "kimi_k3")
    assert kimi["estimated"] is False and kimi["cost"] == 0.5 and kimi["cached_tokens"] == 600

    # 以后补了真实单价：估算的那几笔按 token 回算
    real = Pricing(input_per_mtok=5, output_per_mtok=20)
    redo = real.cost(gem["prompt_tokens"], gem["completion_tokens"], gem["cached_tokens"])
    assert redo is not None and abs(redo - (0.1 * 5 + 0.05 * 20)) < 1e-9

    again = CostLedger(path)  # 另一个进程 / 下次启动
    assert abs(again.money(day=today()) - 2.228) < 1e-9, "估算照样计入单日 / 单项目金额"
    assert abs(again.estimated(day=today()) - 1.728) < 1e-9
    assert abs(again.estimated(project="p1") - 1.728) < 1e-9
    assert again.estimated(project="别的项目") == 0.0


# ---------------------------------------------------------------- 开工面板


def test_开工面板点名哪些provider按估价():
    limits = {"money": 200, "video_calls": 60, "video_seconds": 600, "image_calls": 80}
    plain = render_budget(limits, priced=True)
    assert "估" not in plain, "都配了单价就不提估价"

    text_cfg = _gateway(EventBus()).config.text  # kimi_k3 配了价，gemini 没配
    text = render_budget(limits, priced=True, estimated=text_cfg.estimated_providers())
    assert "¥200" in text and "估算" in text
    assert "gemini（gemini-model，按输入 ¥6.48 / 输出 ¥21.6 每百万 token 估）" in text
    assert "kimi_k3" not in text
    assert "pricing" in text, "告诉人查到真实单价往哪填"

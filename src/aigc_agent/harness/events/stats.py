"""M6 会话统计 —— 成本看板的数据层。

从一串事件里算出：按角色/模型的文本成本与 token、按 modality 的媒体调用、
工具调用次数/耗时/失败、轮次与停机原因、人审决策、预算拦截、子代理运行。
在线（bus.history）和离线（EventLog 读回）都能算 —— 一套双用。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .bus import Event, EventType


def session_stats(events: list[Event]) -> dict[str, Any]:
    by_role: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "prompt": 0, "completion": 0, "cached": 0, "cost": 0.0, "unpriced": 0}
    )
    media: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "ok": 0, "failed": 0, "elapsed_s": 0.0, "cost": 0.0}
    )
    tools: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "ok": 0, "failed": 0, "total_ms": 0, "max_ms": 0}
    )
    stops: dict[str, int] = defaultdict(int)
    decisions: dict[str, int] = defaultdict(int)
    subagents: dict[str, dict[str, int]] = defaultdict(lambda: {"runs": 0, "ok": 0})
    turns = iterations = budget_hits = checkpoints = 0
    total_cost = 0.0

    for e in events:
        d = e.data
        t = e.type
        if t is EventType.COST:
            modality = str(d.get("modality") or "text")
            cost = d.get("cost")
            if modality == "text":
                r = by_role[str(d.get("role") or "?")]
                r["calls"] += 1
                r["prompt"] += int(d.get("prompt_tokens") or 0)
                r["completion"] += int(d.get("completion_tokens") or 0)
                r["cached"] += int(d.get("cached_tokens") or 0)
                if cost is None:
                    r["unpriced"] += 1
                else:
                    r["cost"] += float(cost)
            else:
                media[modality]["cost"] += float(cost or 0.0)
            total_cost += float(cost or 0.0)
        elif t is EventType.MODEL_RESPONSE and d.get("modality"):
            m = media[str(d["modality"])]
            m["calls"] += 1
            ok = d.get("ok", d.get("status") in ("succeeded", None))
            m["ok" if ok else "failed"] += 1
            m["elapsed_s"] += float(d.get("elapsed_s") or 0.0)
        elif t in (EventType.TOOL_RESULT, EventType.TOOL_ERROR):
            tl = tools[str(d.get("tool") or "?")]
            tl["calls"] += 1
            tl["ok" if d.get("ok", t is EventType.TOOL_RESULT) else "failed"] += 1
            ms = int(d.get("duration_ms") or 0)
            tl["total_ms"] += ms
            tl["max_ms"] = max(tl["max_ms"], ms)
        elif t is EventType.LOOP_START:
            turns += 1
        elif t is EventType.ITERATION_START:
            iterations += 1
        elif t is EventType.LOOP_STOP_REASON:
            stops[str(d.get("reason") or "?")] += 1
        elif t is EventType.CHECKPOINT_REACHED:
            checkpoints += 1
        elif t is EventType.CHECKPOINT_DECIDED:
            decisions[str(d.get("decision") or "?")] += 1
        elif t is EventType.BUDGET_EXCEEDED:
            budget_hits += 1
        elif t is EventType.SUBAGENT_END:
            s = subagents[str(d.get("name") or "?")]
            s["runs"] += 1
            s["ok"] += 1 if d.get("ok") else 0

    prompt = sum(int(r["prompt"]) for r in by_role.values())
    cached = sum(int(r["cached"]) for r in by_role.values())
    first = events[0].ts if events else 0.0
    last = events[-1].ts if events else 0.0
    return {
        "events": len(events),
        "duration_s": round(last - first, 1) if events else 0.0,
        "turns": turns,
        "iterations": iterations,
        "total_cost": round(total_cost, 4),
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "cache_hit": round(cached / prompt, 3) if prompt else 0.0,
        "by_role": dict(by_role),
        "media": dict(media),
        "tools": dict(tools),
        "stops": dict(stops),
        "checkpoints": checkpoints,
        "decisions": dict(decisions),
        "budget_hits": budget_hits,
        "subagents": dict(subagents),
    }


def render_stats(s: dict[str, Any]) -> str:
    lines = [
        f"事件 {s['events']} · 时长 {s['duration_s']}s · 轮次 {s['turns']} · "
        f"迭代 {s['iterations']} · 人审 {s['checkpoints']} · 预算拦截 {s['budget_hits']}",
        f"总成本 ¥{s['total_cost']:.4f} · 输入 {s['prompt_tokens']:,} token · "
        f"缓存命中 {s['cache_hit']:.0%}",
    ]
    if s["by_role"]:
        lines.append("")
        lines.append("按角色（文本）：")
        for role, r in sorted(s["by_role"].items(), key=lambda kv: -kv[1]["cost"]):
            unpriced = f"（{int(r['unpriced'])} 次无单价）" if r["unpriced"] else ""
            lines.append(
                f"  {role:18} {int(r['calls']):3} 次 · in {int(r['prompt']):,} / out "
                f"{int(r['completion']):,} · cached {int(r['cached']):,} · "
                f"¥{r['cost']:.4f}{unpriced}"
            )
    if s["media"]:
        lines.append("")
        lines.append("媒体：")
        for kind, m in sorted(s["media"].items()):
            lines.append(
                f"  {kind:8} {int(m['calls'])} 次 · 成功 {int(m['ok'])} · "
                f"失败 {int(m['failed'])} · 耗时 {m['elapsed_s']:.0f}s · ¥{m['cost']:.4f}"
            )
    if s["tools"]:
        lines.append("")
        lines.append("工具：")
        for name, tl in sorted(s["tools"].items(), key=lambda kv: -kv[1]["calls"])[:15]:
            avg = int(tl["total_ms"] / tl["calls"]) if tl["calls"] else 0
            lines.append(
                f"  {name:28} {int(tl['calls']):3} 次 · 失败 {int(tl['failed'])} · "
                f"平均 {avg}ms · 最长 {int(tl['max_ms'])}ms"
            )
    if s["stops"] or s["decisions"]:
        lines.append("")
        stops = ", ".join(f"{k} {v}" for k, v in s["stops"].items()) or "—"
        dec = ", ".join(f"{k} {v}" for k, v in s["decisions"].items()) or "—"
        lines.append(f"停机原因：{stops} · 人审决策：{dec}")
    if s["subagents"]:
        sub = ", ".join(f"{k} {v['ok']}/{v['runs']}" for k, v in s["subagents"].items())
        lines.append(f"子代理：{sub}")
    return "\n".join(lines)

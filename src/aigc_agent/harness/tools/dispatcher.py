"""M4 工具调度 —— 权限前置 + 并发执行 + 超时。

关键设计：**权限串行、执行并发。**
权限询问要跟人交互，并发问会让终端提示交错、人根本看不清在批准什么。
所以先顺序过一遍闸门，再把放行的一批并发跑掉。

单轮内并发是内容场景的性能命门：同时生成 6 张图 vs 串行 6 次，差 6 倍。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ..context.compaction import find_fold_marks
from ..events.bus import EventBus, EventType
from ..model.gateway import ToolCall
from ..permission.gate import PermissionGate
from .provider import ToolMeta, ToolResult
from .registry import ToolRegistry

# 工具结果进上下文的长度上限。超出部分截断并留指针（P2 接 M12 后指向 asset_id）。
MAX_RESULT_CHARS = 4000


class ToolDispatcher:
    def __init__(
        self,
        registry: ToolRegistry,
        gate: PermissionGate,
        bus: EventBus,
        timeout: float = 120.0,
        max_parallel: int = 16,
    ) -> None:
        self.registry = registry
        self.gate = gate
        self.bus = bus
        self.timeout = timeout
        # 同一次迭代里最多同时跑几个工具（花钱的媒体调用另有网关里的按模态并发上限）
        self._parallel = asyncio.Semaphore(max(1, max_parallel))
        # 已经跑完、结果还没被 Loop 回填的调用（call_id → 结果）。
        # 2026-09-23 审查：/stop 取消一整批时，之前连同批里**已经完成**的结果一起丢，
        # 下一轮被补记成「没有执行结果」—— 已付费的 4 段视频，模型以为没做又生成一遍。
        # Loop 修补悬空调用时先来这里取真实结果。
        self._completed: dict[str, ToolResult] = {}

    def take_completed(self, call_id: str) -> ToolResult | None:
        """取走一个已完成但没被回填的结果（被中断的那一批里先跑完的）。"""
        return self._completed.pop(call_id, None)

    async def run(self, calls: list[ToolCall]) -> list[tuple[ToolCall, ToolResult]]:
        # ---- 第一阶段：串行过闸门 ----
        approved: list[tuple[ToolCall, dict[str, Any], ToolMeta]] = []
        results: dict[str, ToolResult] = {}

        for call in calls:
            args, err = _parse_args(call.arguments)
            if err:
                results[call.id] = ToolResult(ok=False, error=err)
                continue

            # 按参数算等级：同一个工具写产物目录和写 Agent 自己的配置，风险不是一回事
            meta = self.registry.meta_for_call(call.name, args)
            if meta is None:
                results[call.id] = ToolResult(ok=False, error=f"未知工具 {call.name!r}")
                continue

            ok, reason = await self.gate.check(meta, args)
            if not ok:
                results[call.id] = ToolResult(ok=False, error=reason)
                continue

            approved.append((call, args, meta))

        # ---- 第二阶段：并发执行 ----
        # 超时按工具来：批量渲染一次调用跑几十分钟，默认 120s 会误杀它。
        if approved:
            async with asyncio.TaskGroup() as tg:
                tasks = {
                    call.id: tg.create_task(
                        self._invoke_one(
                            call,
                            args,
                            meta.timeout or self.timeout,
                            meta.max_result_chars or MAX_RESULT_CHARS,
                        )
                    )
                    for call, args, meta in approved
                }
            for cid, task in tasks.items():
                results[cid] = task.result()

        # 正常返回：这批结果由调用方回填，缓存里的就不需要了
        for cid in results:
            self._completed.pop(cid, None)
        return [(c, results[c.id]) for c in calls if c.id in results]

    async def _invoke_one(
        self,
        call: ToolCall,
        args: dict[str, Any],
        timeout_s: float,
        max_chars: int = MAX_RESULT_CHARS,
    ) -> ToolResult:
        await self.bus.emit(EventType.TOOL_CALL, tool=call.name, args=args, call_id=call.id)
        try:
            async with self._parallel:
                result = await asyncio.wait_for(
                    # 上面第一阶段已经串行过闸门了，这里不再问第二遍
                    self.registry.invoke_ungated(call.name, args), timeout=timeout_s
                )
        except TimeoutError:
            result = ToolResult(
                ok=False,
                error=(
                    f"工具执行超时（>{timeout_s:.0f}s）：本地不再等了，但它调用的模型/生成任务"
                    "可能仍在服务端跑完并计费 —— 重做前先核对（生图/生视频看 media_tasks）"
                ),
            )
        except Exception as e:  # noqa: BLE001 — 单个工具崩溃不能打断整轮
            result = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        self._completed[call.id] = result

        if result.ok and len(result.content) > max_chars:
            # 截断要让模型知道被截了、截了多少 —— 否则它会把"看到的"当成"全部"
            total = len(result.content)
            result.content = (
                result.content[:max_chars]
                + f"\n\n[结果已截断：共 {total} 字，只显示前 {max_chars} 字。"
                "列表类结果请加过滤条件或分页再查]"
            )
            result.truncated = True

        await self.bus.emit(
            EventType.TOOL_RESULT if result.ok else EventType.TOOL_ERROR,
            tool=call.name,
            call_id=call.id,
            ok=result.ok,
            error=result.error,
            duration_ms=result.duration_ms,
            truncated=result.truncated,
            preview=result.content[:200],
            # 产物引用。执行痕迹靠它 + 入参里的资产 id 拼出依赖 DAG。
            asset_ref=result.asset_ref,
            suspend=result.suspend,
            args=args,
        )
        return result


def _parse_args(raw: str) -> tuple[dict[str, Any], str | None]:
    if not raw or not raw.strip():
        return {}, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return {}, f"参数不是合法 JSON：{e}。原始内容：{raw[:200]}"
    if not isinstance(parsed, dict):
        return {}, f"参数必须是 JSON 对象，收到 {type(parsed).__name__}"
    # 抄回来的折叠标记不是内容：放过去就是把占位符当正文存盘/发给下游模型
    marks = find_fold_marks(parsed)
    if marks:
        return {}, (
            f"参数 {', '.join(marks)} 是上下文折叠留下的占位标记（形如「<N 字已折叠>」），"
            "不是真实内容 —— 历史里被折叠的正文不会自动带上，抄回来存进去就是一份空壳。"
            "三条路选一条：① 把完整内容重新写出来再调用；"
            "② 要沿用已存的内容，传它的资产 id，或先 read_asset 取回原文；"
            "③ 内容太长一次写不完，就 fs_write 分几块写到本地文件（第二块起 append=true），"
            "再 fs_import 登记成资产，然后传资产 id —— 长正文别硬塞进工具参数。"
        )
    return parsed, None

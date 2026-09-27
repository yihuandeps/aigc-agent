"""输入枢纽 —— 生成过程中输入框不消失（2026-09-18，用户要的）。

之前一轮跑起来（渲视频动辄十几分钟）终端就只剩进度窗，想说句话只能 Ctrl+C。
现在 stdin 由一根常驻线程读，所有输入统一从这里过：

  · 空闲时：正常问答（`next_message` / `ask`）
  · 一轮在跑时打字 → **排队**，本轮结束后按序自动发送（不用再敲）
  · `/stop`（或 /pause）→ 立刻停掉当前这一轮
  · `/now 消息` → 停掉当前，并把这条消息插到队首立即发送
  · `/queue` 看排队，`/queue clear` 清空

所有问人的提示（权限确认、金额护栏、人审决策、产物目录）也都走这里，
否则两处同时读 stdin 会互相抢行。停掉的一轮里没执行完的工具调用，由 Loop 在
下一轮开始时补记为「已中断」（`_repair_interrupted_calls`），上下文不会缺 tool 响应。
"""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from rich.console import Console
from rich.markup import escape

STOP_WORDS = {"/stop", "/pause", "/停", "/暂停", "/停止"}
NOW_PREFIXES = ("/now ", "/插队 ")


class InputHub:
    def __init__(
        self,
        console: Console,
        reader: Callable[[], str] | None = None,
        use_thread: bool = True,
    ) -> None:
        self.console = console
        self.pending: list[str] = []  # 排队的消息，本轮结束后按序发送
        self.stopped_by_user = False  # 最近一次 watch 是不是被 /stop 或 /now 停掉的
        self.eof = False
        self._reader = reader or sys.stdin.readline
        self._use_thread = use_thread
        self._q: asyncio.Queue[str | None] | None = None
        # 在等一行输入的：(future, 提示语)。**栈**：最后问的那个先拿到下一行。
        # 2026-09-23 审查实测：之前只有一个等待位，后台（流水线经闸门）一问，主循环等着的
        # 「你」就被顶掉了 —— 你打给主循环的话被当成那个提问的回答，主循环再也不返回
        self._waiters: list[tuple[asyncio.Future[str], str]] = []
        self._task: asyncio.Task[Any] | None = None
        self._router: asyncio.Task[None] | None = None
        # /stop 之类被按下时通知一声（CLI 接到事件总线上，复盘能看到是人停的）
        self.on_stop: Callable[[str], None] | None = None

    # ---------- 生命周期 ----------

    def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._q = asyncio.Queue()
        self._router = loop.create_task(self._route())
        if not self._use_thread:
            return

        def pump() -> None:
            while True:
                try:
                    line = self._reader()
                except Exception:  # noqa: BLE001 — stdin 关了/被重定向，按 EOF 处理
                    line = ""
                if not line:
                    loop.call_soon_threadsafe(self._q.put_nowait, None)  # type: ignore[union-attr]
                    return
                loop.call_soon_threadsafe(self._q.put_nowait, line.rstrip("\r\n"))  # type: ignore[union-attr]

        threading.Thread(target=pump, name="stdin-reader", daemon=True).start()

    def close(self) -> None:
        if self._router is not None:
            self._router.cancel()
            self._router = None

    def feed(self, line: str | None) -> None:
        """不走线程直接投一行（测试 / 其他前端接进来用）。None = EOF。"""
        assert self._q is not None, "先 start()"
        self._q.put_nowait(line)

    # ---------- 路由 ----------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _route(self) -> None:
        assert self._q is not None
        while True:
            line = await self._q.get()
            if line is None:
                self.eof = True
                for fut, _ in self._waiters:
                    if not fut.done():
                        fut.set_exception(EOFError())
                return
            text = line.strip()
            # 停止/插队优先：哪怕此刻正在问权限，也先把任务停掉
            if self.running and self._handle_control(text):
                top = self._top()
                if top is not None:
                    top.set_result("")
                continue
            top = self._top()
            if top is not None:
                top.set_result(line)
                continue
            if not text:
                continue
            if self.running:
                self._queue_while_running(text)
            else:
                self.pending.append(text)

    def _top(self) -> asyncio.Future[str] | None:
        """最近问的、还在等的那个。"""
        while self._waiters and self._waiters[-1][0].done():
            self._waiters.pop()
        return self._waiters[-1][0] if self._waiters else None

    def _handle_control(self, text: str) -> bool:
        """/stop、/now：返回 True 表示已处理并发起了取消。

        /stop 停下来的同时**清掉排队的消息**（2026-09-23 审查：排队里有「再渲第 3 集」，
        /stop 停完会立刻开下一轮接着花钱）。/now 是改道不是叫停：插到队首，排着的照旧。
        """
        low = text.lower()
        if low in STOP_WORDS:
            self.stopped_by_user = True
            dropped = len(self.pending)
            self.pending.clear()
            self.console.print(
                "[yellow]■ 正在停止当前这一轮…[/]"
                + (f"[dim]（排队的 {dropped} 条也清掉了）[/]" if dropped else "")
            )
            self._notify_stop(text)
            self._task.cancel()  # type: ignore[union-attr]
            return True
        for p in NOW_PREFIXES:
            if low.startswith(p):
                msg = text[len(p):].strip()
                if msg:
                    self.pending.insert(0, msg)
                    self.stopped_by_user = True
                    self.console.print(f"[yellow]■ 停止当前这一轮，随后立即发送：[/]{msg}")
                    self._notify_stop(text)
                    self._task.cancel()  # type: ignore[union-attr]
                    return True
        return False

    def _notify_stop(self, text: str) -> None:
        if self.on_stop is not None:
            try:
                self.on_stop(text)
            except Exception:  # noqa: BLE001 — 通知失败不能挡住停止
                pass

    def _queue_while_running(self, text: str) -> None:
        low = text.lower()
        if low == "/queue":
            self.print_queue()
            return
        if low in {"/queue clear", "/queue 清空"}:
            self.pending.clear()
            self.console.print("[dim]排队已清空[/]")
            return
        self.pending.append(text)
        self.console.print(
            f"[dim]⏳ 已排队（第 {len(self.pending)} 条），本轮结束后自动发送。"
            "/stop 立即停止当前 · /now <消息> 停止并插队发送 · /queue 看排队[/]"
        )

    def print_queue(self) -> None:
        if not self.pending:
            self.console.print("[dim]排队为空[/]")
            return
        for i, m in enumerate(self.pending, 1):
            self.console.print(f"  [dim]{i}.[/] {escape(m[:120])}")

    # ---------- 读输入 ----------

    async def ask(self, prompt: str) -> str:
        """问一句、等一行**新**输入（不消费排队的消息）。stdin 关了抛 EOFError。"""
        if self.eof:
            raise EOFError
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        self._waiters.append((fut, prompt))
        self.console.print(prompt, end="")
        try:
            return await fut
        finally:
            self._waiters = [(f, p) for f, p in self._waiters if f is not fut]
            # 下面还有人在等（比如主循环的「你」）：把它的提示再打一遍，别让人以为卡住了
            below = [(f, p) for f, p in self._waiters if not f.done()]
            if below:
                self.console.print()
                self.console.print(below[-1][1], end="")

    async def next_message(self, prompt: str) -> str:
        """下一条要发给模型的话：有排队的先发排队的，没有就等人打。"""
        if self.pending:
            msg = self.pending.pop(0)
            self.console.print(f"{prompt}[dim]▶ 排队消息：[/]{escape(msg)}")
            return msg
        return await self.ask(prompt)

    # ---------- 跑一轮 ----------

    async def watch(self, coro: Awaitable[Any]) -> Any:
        """把一轮执行包成任务跑；期间输入照常收：排队 / 停止 / 插队。

        被 /stop、/now 停掉时向调用方抛 CancelledError，调用方看 stopped_by_user 区分
        「人停的」和「外部取消」。
        """
        self.stopped_by_user = False
        self._task = asyncio.ensure_future(coro)
        try:
            return await self._task
        finally:
            self._task = None

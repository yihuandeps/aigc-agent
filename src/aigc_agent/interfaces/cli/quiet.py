"""压掉 httpcore2 关连接时的假异常。

**这不是我们的 bug**，是环境里这套 httpx2 / httpcore2 分支的问题：
读完响应体收尾时，`safe_async_iterate` 向异步生成器抛 GeneratorExit，
生成器没按预期停下，contextlib 就抛 `generator didn't stop after athrow()`。
请求本身是成功的，数据也拿到了 —— 纯粹是清理路径上的噪音。

不压的话每次调模型都会刷十几行堆栈，把真正的报错淹掉，排查时很误导。

压的范围刻意收得很窄：只吃这一种 RuntimeError，且必须来自 httpcore2。
别的异常照常往外抛 —— 无差别静音比噪音更危险。

**什么时候删掉这个文件**：环境里换回上游 httpx（或 httpx2 修了这个），
届时 `_is_shutdown_noise` 会一条都匹配不到，删了即可。
注意本环境里的 openai SDK 自己 import 的就是 httpx2，标准 httpx 没装，
所以换的时候要连 openai 一起换。
"""

from __future__ import annotations

import asyncio
from typing import Any

_NOISE = "generator didn't stop after athrow()"


def _is_shutdown_noise(context: dict[str, Any]) -> bool:
    exc = context.get("exception")
    if not isinstance(exc, RuntimeError) or _NOISE not in str(exc):
        return False
    # 必须确实来自 httpcore2 的收尾路径，否则放行
    tb = exc.__traceback__
    while tb is not None:
        if "httpcore2" in (tb.tb_frame.f_code.co_filename or ""):
            return True
        tb = tb.tb_next
    return False


def install() -> None:
    """在当前事件循环上装过滤器。要在协程里调（此时 loop 才存在）。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # 没有运行中的循环，不装
        return

    default = loop.get_exception_handler() or (lambda lp, ctx: lp.default_exception_handler(ctx))

    def handler(lp: asyncio.AbstractEventLoop, ctx: dict[str, Any]) -> None:
        if _is_shutdown_noise(ctx):
            return
        default(lp, ctx)

    loop.set_exception_handler(handler)

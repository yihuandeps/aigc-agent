"""常驻输入框 + 命令模糊搜索（2026-09-22，用户要的）。

之前的输入是裸 `sys.stdin.readline`：一轮跑起来，进度窗和你敲的字会叠在一起，
输入框也没有固定位置，想输命令得先记住有哪些、拼对全名。

现在底部钉一个输入框：
  · **不消失** —— 生成、渲染、进度窗刷新时它都在最下面，随时能打字
  · 所有输出从它上方走（patch_stdout），不再互相盖
  · 打 `/` 就弹命令菜单，**模糊匹配**（`/bg` 能搜到 `/budget`、`/rb` 能搜到 `/rollback`），
    命中的命令连同说明列在输入框下方，上下键选、Tab 或回车补全

输入框只负责"读一行"，读到的行照旧交给 InputHub 路由（排队 / /stop / 插队），
所以没有 prompt_toolkit 的环境（重定向 stdin、CI、测试）退回 readline 也能跑。
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from typing import Any

# 命令表：唯一出处。补全菜单、/help、文档都读它。
# (命令, 一句话说明)
COMMANDS: tuple[tuple[str, str], ...] = (
    ("/help", "列出全部命令"),
    ("/exit", "退出"),
    ("/quit", "退出"),
    ("/stat", "看状态：模型、工具数、窗口、产物目录、锁定的模型"),
    ("/tools", "列出当前可用的工具"),
    ("/trace", "看上一轮的执行痕迹"),
    ("/mermaid", "导出流程图"),
    ("/skills", "已加载的 skill 与能力区占用"),
    ("/brief", "记忆简报"),
    ("/out", "改产物目录：/out E:\\西游记"),
    ("/type", "选产线：短剧 / 抖音短视频 / 广告 / 设计；/type off 不限定"),
    ("/line", "同 /type"),
    ("/rename", "把已生成的图/视频改成带序号的可读文件名（/rename dry 只看计划）"),
    ("/rollback", "回退到某个资产版本重来：/rollback as_xxx"),
    ("/retry", "同一轮续跑（网络失败后不用重发话）"),
    ("/budget", "预算用量；/budget set 金额 300 视频秒 900 改额度；/budget allow 50 临时追加；"
                "/budget reset 清零本次开工的用量"),
    ("/auto", "自动人审：/auto on 打开、/auto off 关闭、/auto retry 重派流水线失败的环节"),
    ("/stop", "停掉正在跑的这一轮"),
    ("/pause", "同 /stop"),
    ("/now", "停掉当前并插队发送：/now 先渲第 3 段"),
    ("/queue", "看排队的消息；/queue clear 清空"),
)

# 同义的写法（中文、别名）：分发时认，补全菜单不列
ALIASES: dict[str, str] = {
    "/停": "/stop", "/暂停": "/stop", "/停止": "/stop", "/重试": "/retry", "/继续跑": "/retry",
    "/插队": "/now",
}


def known_command(text: str) -> bool:
    """是不是命令表里的命令（带参数也认）。不在表里的「/xxx」不该当成聊天发给模型。"""
    head = (text or "").strip().split(" ", 1)[0].lower()
    return head in {c for c, _ in COMMANDS} or head in ALIASES


def help_text() -> str:
    width = max(len(c) for c, _ in COMMANDS)
    return "\n".join(f"  {c.ljust(width)}  {d}" for c, d in COMMANDS)


# 输入框配色。**每个值都要能被 prompt_toolkit 解析**，否则整个 CLI 起不来 ——
# 2026-09-22 就是这么炸的一次：写了 "noinverse"（正确的是 noreverse），
# Style.from_dict 当场抛 ValueError，而它当时还在 try 外面。
# test_promptbox.py 里有一条专门逐个解析这张表，改配色先跑它。
STYLE: dict[str, str] = {
    "frame": "ansibrightblack",      # 输入框的边
    "caret": "bold ansicyan",        # 提示符 >
    "bottom-toolbar": "noreverse ansibrightblack",
    "completion-menu.completion": "bg:#1f2430 ansiwhite",
    "completion-menu.completion.current": "bg:#3b4252 bold ansiwhite",
    "completion-menu.meta.completion": "bg:#1f2430 ansibrightblack",
    "completion-menu.meta.completion.current": "bg:#3b4252 ansiwhite",
}


def fuzzy_match(query: str, name: str) -> int | None:
    """子序列模糊匹配，返回"有多散"的分数（越小越好）。不匹配返回 None。

    `bg` → `/budget`：b…g 按顺序出现即可。分数 = 首次命中位置 + 各次命中之间的间隔，
    所以 `/budget` 排在 `/rollback` 前面（前者更紧凑、起始更靠前）。
    """
    q = query.lstrip("/").lower()
    target = name.lstrip("/").lower()
    if not q:
        return 0
    score = 0
    pos = -1
    for ch in q:
        nxt = target.find(ch, pos + 1)
        if nxt < 0:
            return None
        score += nxt - pos
        pos = nxt
    return score


def search(query: str, commands: Iterable[tuple[str, str]] = COMMANDS) -> list[tuple[str, str]]:
    """模糊搜命令，按匹配紧凑度排序。query 不以 / 开头时返回空 —— 正常聊天不弹菜单。"""
    if not query.startswith("/"):
        return []
    head = query.split(" ", 1)[0]
    if " " in query:  # 已经在写参数了，不再弹菜单
        return []
    if head == "/":  # 光敲一个 / —— 按命令表的顺序列，常用的在前，别按名字长短排
        return list(commands)
    hits: list[tuple[int, int, str, str]] = []
    for name, desc in commands:
        s = fuzzy_match(head, name)
        if s is not None:
            hits.append((s, len(name), name, desc))
    hits.sort()
    return [(name, desc) for _, _, name, desc in hits]


# 写在下边框上的快捷键提示。宽终端给全的，窄了退精简版，再窄就只剩一条光边 ——
# 80 列（默认宽度）刚好放得下全的，70 列只放得下精简版。
HINT = "/ 命令 · ↑↓ 选 · Tab 补全 · /stop 停当前轮 · /now 插队 · /exit 退出"
HINT_SHORT = "/ 命令 · Tab 补全 · /stop 停 · /exit 退出"


def _width(default: int = 80) -> int:
    import shutil

    return max(28, shutil.get_terminal_size((default, 24)).columns)


def top_border(width: int) -> str:
    """输入框的上边：╭────…────╮"""
    return "╭" + "─" * max(0, width - 2) + "╮"


def bottom_border(width: int, hints: tuple[str, ...] = (HINT, HINT_SHORT)) -> str:
    """输入框的下边，顺带把快捷键写在边上：╰─ / 命令 … ───╯

    中文是双宽字符，用 get_cwidth 量显示宽度，不能用 len ——
    用 len 算，中文越多右边越歪。装不下就退到更短的那条提示。
    """
    from prompt_toolkit.utils import get_cwidth

    inner = width - 2  # 去掉两个拐角
    for hint in hints:
        left = f"─ {hint} "
        used = get_cwidth(left)
        if used <= inner:
            return "╰" + left + "─" * (inner - used) + "╯"
    return "╰" + "─" * max(0, inner) + "╯"  # 再窄就只留一条光边


def _completer() -> Any:
    from prompt_toolkit.completion import Completer, Completion

    class _Cmd(Completer):
        def get_completions(self, document: Any, complete_event: Any) -> Any:
            text = document.text_before_cursor
            for name, desc in search(text):
                yield Completion(
                    name,
                    start_position=-len(text.split(" ", 1)[0]),
                    display=name,
                    display_meta=desc,
                )

    return _Cmd()


def make_reader(placeholder: str = "说点什么，或敲 / 看命令") -> Callable[[], str] | None:
    """造一个"读一行"的函数给 InputHub 用。装不了 prompt_toolkit 就返回 None（退回 readline）。

    返回值遵守 InputHub 的约定：空串 = EOF；空行是 "\\n"，不能是 ""。
    """
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.patch_stdout import patch_stdout
        from prompt_toolkit.styles import Style
    except ImportError:
        return None
    if not sys.stdin.isatty():
        return None  # 重定向 / 管道：没有终端可钉，老实用 readline

    # 上边框 + 提示符当作"提示语"，下边框走 bottom_toolbar —— 合起来就是一个看得见的框。
    # 两个都做成**可调用**：终端一拉宽拉窄，边框跟着重算。
    def message() -> Any:
        w = _width()
        return HTML(f"<frame>{top_border(w)}</frame>\n<frame>│</frame> <caret>&gt;</caret> ")

    def toolbar() -> Any:
        return HTML(f"<frame>{bottom_border(_width())}</frame>")

    try:
        session: Any = PromptSession(
            completer=_completer(),
            style=Style.from_dict(STYLE),
            complete_while_typing=True,  # 边打边搜，结果就在输入框下方
            # 不预留菜单空间：留了就是一大块常驻空白，框看着不像框；
            # 菜单该弹的时候 prompt_toolkit 会自己把屏幕顶上去
            reserve_space_for_menu=0,
            bottom_toolbar=toolbar,
            placeholder=HTML(f"<ansibrightblack>{placeholder}</ansibrightblack>"),
        )
    except Exception:  # noqa: BLE001 — 配色/终端有任何问题都退回 readline，别把 CLI 带崩
        return None

    def read() -> str:
        try:
            text = session.prompt(message)
        except EOFError:
            return ""
        except KeyboardInterrupt:
            return "/stop\n"  # Ctrl+C 当成停掉这一轮，不是退出整个会话
        except Exception:  # noqa: BLE001
            return sys.stdin.readline()
        return (text or "") + "\n"

    return PromptReader(read, patch_stdout)


class PromptReader:
    """输入框的「读一行」+ 整个会话只进一次 patch_stdout。

    2026-09-23 审查（实测·模拟）：之前每读一行就在读线程里进出一次 patch_stdout，进度窗
    又在主线程里反复开关 Live 重定向 —— 两个线程交替改 sys.stdout，Live 停下时可能把
    sys.stdout 恢复成已经失效的代理，最终回复要么永远不显示、要么和输入框叠在一起。
    现在 chat 开始时进一次、结束时出一次；所有输出都经同一个代理走到输入框上方。
    """

    def __init__(self, read: Callable[[], str], patcher: Any) -> None:
        self._read = read
        self._patcher = patcher
        self._ctx: Any = None

    def __call__(self) -> str:
        return self._read()

    def start(self) -> None:
        if self._ctx is None:
            self._ctx = self._patcher(raw=True)
            self._ctx.__enter__()

    def stop(self) -> None:
        if self._ctx is not None:
            ctx, self._ctx = self._ctx, None
            try:
                ctx.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 — 退出时终端状态异常不要再抛
                pass

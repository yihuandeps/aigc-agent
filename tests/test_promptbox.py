"""常驻输入框 + 命令模糊搜索（2026-09-22，用户要的）。

用户要的三件事：输入框始终在底部不消失、随时能敲命令、命令能模糊搜且结果列在下方。
这里测得了的是模糊匹配与菜单内容（纯逻辑）；输入框本身要真终端，靠 make_reader
在没有 tty 时返回 None 来保证"装不了就退回 readline"，不会把 CLI 带崩。
"""

from __future__ import annotations

import sys

from aigc_agent.interfaces.cli.promptbox import (
    COMMANDS,
    HINT,
    HINT_SHORT,
    STYLE,
    bottom_border,
    fuzzy_match,
    make_reader,
    search,
    top_border,
)


def test_子序列就能命中_不用打全名():
    assert fuzzy_match("/bg", "/budget") is not None
    assert fuzzy_match("/rb", "/rollback") is not None
    assert fuzzy_match("/qc", "/queue") is None, "q…c 在 queue 里凑不齐"
    assert fuzzy_match("/xyz", "/stat") is None


def test_越紧凑越靠前():
    """/bg 该先给 /budget，不是碰巧也含 b…g 的别的命令。"""
    hits = [n for n, _ in search("/bg")]
    assert hits and hits[0] == "/budget"
    hits = [n for n, _ in search("/st")]
    assert hits[:2] == ["/stat", "/stop"], hits


def test_菜单带说明():
    hits = search("/budget")
    assert hits and "预算" in hits[0][1]


def test_正常聊天不弹菜单():
    assert search("你好") == []
    assert search("帮我写第 3 集") == []
    assert search("") == []


def test_开始写参数后就不再弹():
    """/out E:\\西游记 —— 已经在填路径了，菜单该让位。"""
    assert search("/out ") == []
    assert search("/out E:/西游记") == []


def test_空的斜杠列出全部命令_按命令表顺序():
    """常用的排前面，不是按名字长短排 —— 这是给人看的菜单。"""
    hits = search("/")
    assert len(hits) == len(COMMANDS)
    assert [n for n, _ in hits] == [n for n, _ in COMMANDS]


def test_命令表覆盖了主要命令():
    names = {n for n, _ in COMMANDS}
    for must in ("/exit", "/stat", "/stop", "/now", "/queue", "/out", "/budget", "/auto"):
        assert must in names, f"命令表缺 {must}"
    assert all(d.strip() for _, d in COMMANDS), "每个命令都得有一句说明，菜单才有用"


def test_没有终端时退回readline(monkeypatch):
    """CI / 管道 / 测试里没有 tty，不能硬上 prompt_toolkit 把 CLI 带崩。"""
    import sys

    monkeypatch.setattr(sys, "stdin", type("S", (), {"isatty": staticmethod(lambda: False)})())
    assert make_reader() is None


def test_装不上依赖时也退回readline(monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name: str, *a: object, **k: object) -> object:
        if name.startswith("prompt_toolkit"):
            raise ImportError("没装")
        return real(name, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake)
    assert make_reader() is None


def test_配色表每一行都能被解析():
    """2026-09-22 真炸过：写了 "noinverse"（正确是 noreverse），Style.from_dict 抛
    ValueError，整个 agent chat 起不来 —— 配色是字符串，拼错编译器不会管，只能靠这条。"""
    from prompt_toolkit.styles import Style

    for key, value in STYLE.items():
        Style.from_dict({key: value})  # 有一行不合法就在这儿炸，指名道姓
    Style.from_dict(STYLE)


def test_终端起不来时不抛异常只退回readline(monkeypatch):
    """输入框是锦上添花，任何环节出问题都不该把 CLI 带崩。"""
    import prompt_toolkit

    monkeypatch.setattr(prompt_toolkit, "PromptSession", lambda **k: 1 / 0)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    assert make_reader() is None


def test_有终端时真的能造出输入框(monkeypatch):
    """之前的用例只覆盖了"退回 readline"那几条路，成功那条从没跑过 ——
    结果配色写错时 make_reader 直接抛异常，agent chat 起不来。这条补上。"""
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    class _Tty:
        @staticmethod
        def isatty() -> bool:
            return True

    monkeypatch.setattr(sys, "stdin", _Tty())
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        reader = make_reader()
    assert callable(reader), "有终端时应该造出输入框，而不是退回 readline"


# ---------------------------------------------------------------- 看得见的框


def test_上下边框宽度对得齐():
    """中文是双宽字符，边框里又写着中文提示 —— 用 len 算右边一定歪，必须按显示宽度。"""
    from prompt_toolkit.utils import get_cwidth

    for w in (30, 55, 72, 120, 200):
        assert get_cwidth(top_border(w)) == w, f"上边框 w={w} 不齐"
        assert get_cwidth(bottom_border(w)) == w, f"下边框 w={w} 不齐"


def test_边框长得像个框():
    top, bot = top_border(100), bottom_border(100)
    assert top.startswith("╭") and top.endswith("╮")
    assert bot.startswith("╰") and bot.endswith("╯")
    assert "/stop" in bot and "Tab" in bot, "快捷键提示写在下边框上，省一行"


def test_终端越窄提示越短_最后只剩光边():
    """80 列（默认宽度）要放得下全提示，不能一到常见宽度就退化成光秃秃一条线。"""
    from prompt_toolkit.utils import get_cwidth

    assert "插队" in bottom_border(80), "80 列该放得下全提示"
    mid = bottom_border(60)
    assert "插队" not in mid and "/stop" in mid, "60 列退到精简提示"
    bare = bottom_border(30)
    assert "/stop" not in bare and set(bare) <= {"╰", "─", "╯"}
    for w in (30, 60, 80):
        assert get_cwidth(bottom_border(w)) == w, f"退档后宽度也要对齐（w={w}）"


def test_提示里不带会被HTML吃掉的字符():
    """提示语要塞进 prompt_toolkit 的 HTML() 里，带 < & 会被当成标签解析。"""
    for hint in (HINT, HINT_SHORT):
        assert "<" not in hint and ">" not in hint and "&" not in hint

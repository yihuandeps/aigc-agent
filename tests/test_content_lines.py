"""产线标签（2026-09-23 用户要的）：用户自己选这次做哪类内容 —— 短剧 / 抖音短视频 / 广告 / 设计。

选定后三件事都要成立：skill 目录按 content_type 预筛、路由指引 pin 进上下文、会话快照记住。
不限定（空）= 老行为，模型自己判断、所有 skill 都在目录里。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from aigc_agent.app import Agent
from aigc_agent.capabilities.memory.session import SessionSnapshot
from aigc_agent.capabilities.skill_hub import SkillHub
from aigc_agent.domain.lines import (
    LINE_PIN,
    LINES,
    get_line,
    is_off,
    line_block,
    menu,
    parse_line,
)
from aigc_agent.interfaces.cli.promptbox import COMMANDS, search

SKILLS = Path(__file__).resolve().parents[1] / "skills"


# ---------------------------------------------------------------- 纯逻辑


def test_四条产线_各自映射到skill筛选用的内容形态():
    assert [ln.key for ln in LINES] == ["drama", "douyin", "ad", "design"]
    assert get_line("design").content_type == "图像"
    # 三条视频产线各有各的形态（都属于「短视频」），skill 预筛才分得开（2026-09-23 审查）
    assert [get_line(k).content_type for k in ("drama", "douyin", "ad")] == ["短剧", "抖音", "广告"]
    assert get_line("nope") is None and get_line("") is None


def test_用户怎么敲都认得出():
    assert parse_line("1").key == "drama" and parse_line("4").key == "design"
    assert parse_line("短剧").key == "drama"
    assert parse_line("抖音").key == "douyin" and parse_line("短视频").key == "douyin"
    assert parse_line("广告").key == "ad" and parse_line("我想做个带货的").key == "ad"
    assert parse_line("海报").key == "design" and parse_line("封面").key == "design"
    assert parse_line("DESIGN").key == "design"
    assert parse_line("随便") is None and parse_line("") is None


def test_不限定的几种说法():
    for t in ("", "off", "不限定", "清除", "auto", "0"):
        assert is_off(t), t
    assert not is_off("短剧")


def test_菜单四行_指引说清工具和方法论():
    m = menu()
    assert m.count("\n") == 7 and "1  短剧" in m and "4  设计" in m
    blk = line_block(get_line("design"))
    assert blk.startswith("## 当前产线：设计") and "不要再问" in blk
    assert "gen_image" in blk and "make_poster" in blk and "不带任何文字" in blk
    ad = line_block(get_line("ad"))
    assert "product-ad" in ad and "ugc-vlog" in ad and "别混" in ad and "host_file" in ad
    drama = line_block(get_line("drama"))
    assert "drama_render_shots" in drama and "不要用 gen_video 补" in drama


# ---------------------------------------------------------------- skill 预筛


def test_选了设计_短视频类skill不进目录():
    hub = SkillHub(SKILLS)
    hub.load()
    names = {s.name for s in hub.candidates(get_line("design").content_type)}
    assert "drama-script" not in names and "douyin-short" not in names
    assert "seedance-prompting" not in names, "视频提示词库和海报无关，别占预算"
    assert "compliance-redlines" in names, "合规红线是 global，任何产线都在"


def test_选了短剧_创作与提示词skill都在():
    hub = SkillHub(SKILLS)
    hub.load()
    names = {s.name for s in hub.candidates(get_line("drama").content_type)}
    assert {"drama-script", "seedance-prompting", "compliance-redlines"} <= names
    assert "douyin-short" not in names, "抖音的抓热点流程不该进短剧的视野"


def test_选了广告或抖音_各拿各的skill():
    hub = SkillHub(SKILLS)
    hub.load()
    ad = {s.name for s in hub.candidates(get_line("ad").content_type)}
    assert "seedance-prompting" in ad and "douyin-short" not in ad and "drama-script" not in ad
    dy = {s.name for s in hub.candidates(get_line("douyin").content_type)}
    assert {"douyin-short", "seedance-prompting"} <= dy and "drama-script" not in dy


def test_不限定时全部skill都在():
    hub = SkillHub(SKILLS)
    hub.load()
    assert {s.name for s in hub.candidates("")} == {s.name for s in hub.available}


# ---------------------------------------------------------------- Agent 侧


class _Mem:
    def __init__(self) -> None:
        self.pins: dict[str, tuple[str, str]] = {}

    def pin(self, name: str, text: str, position: str = "") -> None:
        self.pins[name] = (text, position)

    def unpin(self, name: str) -> None:
        self.pins.pop(name, None)


def _fake_agent() -> SimpleNamespace:
    """不起整个 Agent（要装配十几个模块），只借它的两个方法：set_content_line / _pin_line。"""
    a = SimpleNamespace(
        content_line="",
        allocator=SimpleNamespace(content_type=""),
        memory=_Mem(),
        session_store=SimpleNamespace(saved=[], set_content_line=lambda k: None),
    )
    a._pin_line = lambda: Agent._pin_line(a)
    return a


def test_选定产线_三件事一起生效():
    a = _fake_agent()
    saved: list[str] = []
    a.session_store = SimpleNamespace(set_content_line=saved.append)
    Agent.set_content_line(a, "design")
    assert a.content_line == "design"
    assert a.allocator.content_type == "图像", "skill 预筛跟着走"
    text, pos = a.memory.pins[LINE_PIN]
    assert pos == "pre_input" and "当前产线：设计" in text
    assert saved == ["design"], "写回会话快照"


def test_清除产线_回到不限定():
    a = _fake_agent()
    saved: list[str] = []
    a.session_store = SimpleNamespace(set_content_line=saved.append)
    Agent.set_content_line(a, "ad")
    Agent.set_content_line(a, "")
    assert a.content_line == "" and a.allocator.content_type == ""
    assert LINE_PIN not in a.memory.pins, "不限定就不 pin"
    assert saved == ["ad", ""]


def test_从快照恢复时不重复写盘():
    a = _fake_agent()
    saved: list[str] = []
    a.session_store = SimpleNamespace(set_content_line=saved.append)
    Agent.set_content_line(a, "drama", persist=False)
    assert a.content_line == "drama" and saved == []


def test_认不得的键当不限定():
    a = _fake_agent()
    Agent.set_content_line(a, "nope", persist=False)
    assert a.content_line == "" and a.allocator.content_type == ""


# ---------------------------------------------------------------- 会话快照 / 命令菜单


def test_会话快照记住产线_重启沿用(tmp_path: Path):
    s = SessionSnapshot(tmp_path, "西游记")
    assert s.content_line == ""
    s.set_content_line("ad")
    again = SessionSnapshot(tmp_path, "西游记")
    assert again.content_line == "ad"
    again.set_content_line("")
    assert SessionSnapshot(tmp_path, "西游记").content_line == ""


def test_命令菜单能搜到type():
    assert any(n == "/type" for n, _ in COMMANDS)
    assert [n for n, _ in search("/ty")][0] == "/type"

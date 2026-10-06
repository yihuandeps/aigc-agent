"""抖音拆解 skill + functions 验收。

移植自 Claude Code skill，验两件事：
  1. 方法论完整搬过来了，且环境特定内容（macOS 路径、别的工具名）清干净了
  2. functions 能把素材落成资产，失败路径不炸也不留垃圾
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.functions.douyin import RENDER_SCRIPT, DouyinFunctions, _slug
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "douyin-viral-analyzer.md"


def _parts() -> tuple[dict, str]:
    raw = SKILL.read_text(encoding="utf-8")
    _, fm, body = raw.split("---", 2)
    return yaml.safe_load(fm), body


# ---------------------------------------------------------------- skill 本体


def test_frontmatter符合skill_hub约定():
    fm, _ = _parts()
    assert fm["name"] == "douyin-viral-analyzer"
    assert fm["status"] == "active"
    assert fm["scope"] == "content_type"
    assert fm["applies_to"] == ["抖音"]
    assert set(fm["stage"]) <= {"选题", "策划", "脚本", "分镜", "正文", "配图", "排版", "审核"}
    assert 0 <= fm["priority"] <= 100


def test_description写成了触发场景而非自我介绍():
    """常驻上下文的只有 description，它决定这篇会不会被翻开。"""
    fm, _ = _parts()
    d = fm["description"]
    assert 20 <= len(d) <= 200
    assert any(k in d for k in ["时用", "用户贴", "说"]), "要说清什么情况下该用它"
    assert "抖音" in d


def test_核心方法论全部保留():
    """这些是这份 skill 真正的价值，移植时一条都不能丢。"""
    _, body = _parts()
    for key in [
        "归因铁律",  # 作者得意之笔 ≠ 用户爆款驱动力
        "抽帧密度铁律",
        "单评论级爆款",
        "评赞比",
        "藏赞比",
        "转赞比",
        "情绪曲线",
        "脚本公式",
        "选题变体",
        "风险点",
        "脱敏",
    ]:
        assert key in body, f"丢了：{key}"

    # 归因铁律的实战案例是最有说服力的部分
    assert "非遗" in body and "零人提到" in body
    # 比例基准按视频类型分档，不是统一标准
    assert "干货/工具型" in body and "情感共鸣/表态型" in body


def test_环境特定内容已清除():
    """原 skill 是给 Claude Code 用的，带 macOS 路径和别的工具链。"""
    raw = SKILL.read_text(encoding="utf-8")
    for bad in [
        "C:\\Users\\17307",
        "~/.claude",
        ".claude/skills",
        "sharelink",
        "whisper-cli",
        "dy_transcribe",
        "settings.json",
        "python3 ~",
    ]:
        assert bad not in raw, f"残留环境细节：{bad}"


def test_改为引用本agent自己的function():
    _, body = _parts()
    assert "fetch_douyin" in body
    assert "transcribe" in body  # 用 agent 的 ASR，不用原 skill 的本地 whisper
    assert "render_douyin_report" in body


def test_脱敏清单更新为本agent的工具名():
    """报告要发给同事客户，不能漏出任何内部实现。"""
    _, body = _parts()
    sec = body[body.index("必须脱敏") :]
    assert "fetch_douyin" in sec and "transcribe" in sec
    assert "as_xxx" in sec  # 资产 id 也不能出现在报告里


def test_能被skill_hub加载():
    from aigc_agent.capabilities.skill_hub import load_skills  # noqa: PLC0415

    skills = load_skills(ROOT / "skills")
    got = next((s for s in skills if s.name == "douyin-viral-analyzer"), None)
    assert got is not None and got.status == "active"
    # 占位版本已被真货取代
    assert all(s.name != "douyin-viral-structure" for s in skills)


# ---------------------------------------------------------------- functions


async def _funcs(tmp_path: Path):
    bus = EventBus()
    store = AssetStore(tmp_path / "assets")
    fns = DouyinFunctions(store, tmp_path)
    registry = ToolRegistry(bus)
    registry.register(fns)
    await registry.refresh()
    return registry, store, fns


async def test_三个function注册且权限正确(tmp_path: Path):
    registry, _, _ = await _funcs(tmp_path)
    perms = {m.name: m.permission.value for m in registry.catalog()}
    assert set(perms) == {"fetch_douyin", "read_frames", "render_douyin_report"}
    assert perms["read_frames"] == "L-read"
    assert perms["fetch_douyin"] == "L-compute"  # 下载 + 抽帧 + 可能调外部 API


async def test_抓取脚本已随项目落地(tmp_path: Path):
    """vendor 脚本要在项目里，不能依赖用户机器上某个全局路径。"""
    _, _, fns = await _funcs(tmp_path)
    h = await fns.health()
    assert h.ok, h.detail


async def test_读帧清单先给首尾(tmp_path: Path):
    registry, store, _ = await _funcs(tmp_path)
    pack_dir = tmp_path / "douyin" / "demo"
    (pack_dir / "frames").mkdir(parents=True)
    (pack_dir / "frames" / "index.json").write_text(
        json.dumps(
            {
                "fps": 2,
                "scene_threshold": 0.15,
                "first": {"t": 0.0, "file": "first.jpg"},
                "last": {"t": 30.0, "file": "last.jpg"},
                "uniform": [{"t": 0.5, "file": "uniform_0001.jpg"}],
                "scene": [{"t": 1.2, "file": "scene_0001.jpg"}],
            }
        ),
        encoding="utf-8",
    )
    pack = store.create("摘要", gen_params={"out_dir": str(pack_dir)})

    r = await registry.invoke("read_frames", {"pack_id": pack.id})
    assert r.ok
    lines = [ln for ln in r.content.splitlines() if ln.strip()]
    body = [ln for ln in lines if ln.startswith(("first", "last", "uniform", "scene"))]
    assert body[0].startswith("first") and body[1].startswith("last")
    assert "先读 first/last" in r.content


async def test_只要关键帧时不给全部(tmp_path: Path):
    registry, store, _ = await _funcs(tmp_path)
    pack_dir = tmp_path / "douyin" / "d2"
    (pack_dir / "frames").mkdir(parents=True)
    (pack_dir / "frames" / "index.json").write_text(
        json.dumps(
            {
                "first": {"t": 0, "file": "a.jpg"},
                "last": {"t": 9, "file": "z.jpg"},
                "uniform": [{"t": 1, "file": "u1.jpg"}],
                "scene": [{"t": 2, "file": "s1.jpg"}],
            }
        ),
        encoding="utf-8",
    )
    pack = store.create("x", gen_params={"out_dir": str(pack_dir)})
    r = await registry.invoke("read_frames", {"pack_id": pack.id, "kind": "key"})
    assert "u1.jpg" not in r.content and "s1.jpg" not in r.content


async def test_没有抽帧数据时给可读错误(tmp_path: Path):
    registry, store, _ = await _funcs(tmp_path)
    pack = store.create("x", gen_params={"out_dir": str(tmp_path / "空目录")})
    r = await registry.invoke("read_frames", {"pack_id": pack.id})
    assert not r.ok and "没有抽帧数据" in r.error


async def test_抓取失败时提示走手动清单(tmp_path: Path):
    """链接抓不到是常态（没配 key、视频删了），要引导而不是干报错。"""
    registry, store, _ = await _funcs(tmp_path)
    r = await registry.invoke("fetch_douyin", {"ref": "这不是一个有效的抖音链接", "name": "bad"})
    assert not r.ok
    assert "手动清单" in r.error
    assert len(store) == 0  # 失败不留垃圾资产


async def test_渲染空报告被拦(tmp_path: Path):
    registry, store, _ = await _funcs(tmp_path)
    empty = store.create("   ")
    r = await registry.invoke("render_douyin_report", {"report_id": empty.id})
    assert not r.ok and "空的" in r.error


def test_PDF样式文件在渲染脚本找的位置():
    """render_pdf.py 按「脚本目录的上一级/assets/report.css」找样式，找不到就静默用空样式 ——
    之前样式放在仓库根的 assets/，脚本读不到，PDF 一直没有样式（2026-09-29 审查）。"""
    src = RENDER_SCRIPT.read_text(encoding="utf-8")
    assert "SKILL_DIR = Path(__file__).resolve().parent.parent" in src
    assert 'CSS_PATH = SKILL_DIR / "assets" / "report.css"' in src
    assert (RENDER_SCRIPT.resolve().parent.parent / "assets" / "report.css").is_file()


async def test_没装markdown时给出安装提示(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Python-Markdown 不在核心依赖里；没装时不能只甩给用户一段子进程的 ImportError。"""
    registry, store, _ = await _funcs(tmp_path)
    report = store.create("# 报告\n\n正文")
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a, **k: None if name == "markdown" else real(name, *a, **k),
    )
    r = await registry.invoke("render_douyin_report", {"report_id": report.id})
    assert not r.ok and "pip install markdown" in r.error


def test_文件名清洗防止越出工作目录():
    """filename 是模型给的，不能让它写到 workspace 外面去。"""
    assert _slug("../../etc/passwd") == "passwd"
    assert _slug("a/b/c.pdf", keep_ext=True) == "c.pdf"
    assert _slug("正常名字_2026.pdf", keep_ext=True) == "正常名字_2026.pdf"
    assert "/" not in _slug("x/y") and "\\" not in _slug("x\\y")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

"""三级披露验收 —— 带 references 的大型 skill。

短剧创作 skill 全量约 20K token，整篇塞进上下文会把能力预算吃光。
拆成「主文档说流程 + 参考说细节」后，任一时刻只有当前阶段那 1–2 篇在场。

验收点：
  1. 目录形态的 skill（SKILL.md + references/）能加载
  2. 参考文档**不随主文档一起进上下文**，只给目录
  3. 按名字能取到单篇，取错名字给出可用清单
  4. 移植时产物改走 Asset，不再直接写文件
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aigc_agent.capabilities.skill_hub import SkillHub, load_skills
from aigc_agent.capabilities.skill_hub.functions import SkillFunctions
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"
DRAMA = SKILLS / "drama-script" / "SKILL.md"


def _hub() -> SkillHub:
    h = SkillHub(SKILLS)
    h.load()
    return h


async def _registry():
    bus = EventBus()
    reg = ToolRegistry(bus)
    reg.register(SkillFunctions(_hub()))
    await reg.refresh()
    return reg


# ---------------------------------------------------------------- 加载


def test_目录形态的skill能被加载():
    skills = load_skills(SKILLS)
    drama = next((s for s in skills if s.name == "drama-script"), None)
    assert drama is not None and drama.status == "active"
    assert len(drama.references) == 8


def test_单文件与目录两种形态共存():
    names = {s.name for s in load_skills(SKILLS)}
    assert "drama-script" in names  # 目录形态
    assert "douyin-viral-analyzer" in names  # 单文件形态


def test_frontmatter符合约定():
    fm = yaml.safe_load(DRAMA.read_text(encoding="utf-8").split("---", 2)[1])
    assert fm["name"] == "drama-script"
    assert fm["status"] == "active"
    assert set(fm["stage"]) <= {"选题", "策划", "脚本", "分镜", "正文", "配图", "排版", "审核"}


def test_参考文档的用途从主文档表格里抽出来():
    """目录里得说清每篇干什么，模型才知道该拉哪篇。"""
    drama = next(s for s in load_skills(SKILLS) if s.name == "drama-script")
    by_name = {r.name: r.purpose for r in drama.references}
    assert "13" in by_name["genre-guide"] or "题材" in by_name["genre-guide"]
    assert "合规" in by_name["compliance-checklist"]
    assert all(p for p in by_name.values()), f"有参考没抽到用途：{by_name}"


# ---------------------------------------------------------------- 预算


def test_主文档不含参考正文():
    """这是三级披露的关键：主文档只给目录，正文按需拉。"""
    body = DRAMA.read_text(encoding="utf-8")
    # 参考文档里的特征内容不该出现在主文档
    assert "反派越强，主角越燃" not in body
    assert "前5秒定生死" not in body
    assert len(body) < 12000, f"主文档 {len(body)} 字符，超出单篇预算"


def test_全量拉取的代价确实很大():
    """量化一下为什么必须分级：全量是主文档的好几倍。"""
    drama = next(s for s in load_skills(SKILLS) if s.name == "drama-script")
    main = len(drama.body)
    refs = sum(len(r.read()) for r in drama.references)
    assert refs > main * 2, f"主文档 {main} / 参考合计 {refs}"


def test_digest标注了带几篇参考():
    drama = next(s for s in load_skills(SKILLS) if s.name == "drama-script")
    assert "8 篇参考" in drama.digest()


# ---------------------------------------------------------------- functions


async def test_加载主文档时附带参考目录而非正文():
    reg = await _registry()
    r = await reg.invoke("load_skill", {"names": ["drama-script"]})
    assert r.ok
    assert "本 skill 的参考文档" in r.content
    assert "genre-guide" in r.content
    assert "load_skill_reference" in r.content
    # 正文不在
    assert "反派越强，主角越燃" not in r.content


async def test_按需取单篇参考():
    reg = await _registry()
    r = await reg.invoke(
        "load_skill_reference", {"skill": "drama-script", "name": "villain-design"}
    )
    assert r.ok
    assert "反派设计体系" in r.content
    # 只给了这一篇，没顺带给别的
    assert "前5秒定生死" not in r.content


async def test_省略md后缀也能取到():
    reg = await _registry()
    a = await reg.invoke("load_skill_reference", {"skill": "drama-script", "name": "hook-design"})
    b = await reg.invoke(
        "load_skill_reference", {"skill": "drama-script", "name": "hook-design.md"}
    )
    assert a.ok and b.ok and a.content == b.content


async def test_取错名字给出可用清单():
    reg = await _registry()
    r = await reg.invoke("load_skill_reference", {"skill": "drama-script", "name": "不存在"})
    assert not r.ok
    assert "villain-design" in r.error and "genre-guide" in r.error


async def test_取错skill名给出可用清单():
    reg = await _registry()
    r = await reg.invoke("load_skill_reference", {"skill": "没这个", "name": "x"})
    assert not r.ok and "drama-script" in r.error


async def test_无参考的skill取参考时说清楚():
    reg = await _registry()
    r = await reg.invoke(
        "load_skill_reference", {"skill": "douyin-viral-analyzer", "name": "whatever"}
    )
    assert not r.ok and "没有参考文档" in r.error


# ---------------------------------------------------------------- 移植适配


def test_产物改走Asset不再直接写文件():
    """原 skill 直接写 {项目目录}/creative-plan.md，绕过 Asset 层就没有血缘和回退。"""
    body = DRAMA.read_text(encoding="utf-8")
    assert "save_draft" in body
    assert "parent_id" in body
    assert "{项目目录}/" not in body
    assert "creative-plan.md" not in body


def test_关键节点要求人审():
    """50-100 集的工作量，方向错了返工成本极高。"""
    body = DRAMA.read_text(encoding="utf-8")
    assert "request_review" in body


def test_参考资料那节改成了按需加载():
    body = DRAMA.read_text(encoding="utf-8")
    sec = body[body.index("## 参考资料") : body.index("## 命令定义")]
    assert "不要一次全读" in sec
    assert "load_skill_reference" in sec
    assert "创作前必须阅读" not in sec  # 原文是要求全读


def test_九个命令都还在():
    body = DRAMA.read_text(encoding="utf-8")
    for cmd in [
        "/start",
        "/plan",
        "/characters",
        "/outline",
        "/episode",
        "/review",
        "/export",
        "/overseas",
        "/compliance",
    ]:
        assert f"### {cmd}" in body, f"丢了命令 {cmd}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

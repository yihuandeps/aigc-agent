"""M15 机审验收。

机审只出风险清单 + 定位 + 建议改法，**不做自动删改**。
六类检查各验一遍，外加：规则是数据（yaml 可加载）、报告落成资产且血缘挂在被审资产下。
"""

from __future__ import annotations

from pathlib import Path

from aigc_agent.capabilities.memory.store import (
    Category,
    Keyword,
    Layer,
    Memory,
    MemoryStore,
    Polarity,
    Source,
)
from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.compliance import ComplianceChecker, ComplianceRules, Severity
from aigc_agent.domain.functions.compliance import ComplianceFunctions
from aigc_agent.harness.events.bus import EventBus, EventType
from aigc_agent.harness.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
RULES = ComplianceRules.load(ROOT / "config" / "compliance.yaml")
LABEL = "本内容由 AI 生成。"


def _check(text: str, **kw):
    return ComplianceChecker(RULES).check(text, **kw)


def _by_rule(report, rule: str):
    return [f for f in report.findings if f.rule == rule]


# ---------------------------------------------------------------- 规则


def test_规则文件可加载():
    assert RULES.version
    keys = {r.key for r in RULES.word_rules}
    assert {"absolute_claims", "medical_claims", "finance_claims", "sensitive_topics"} <= keys
    assert RULES.aigc_markers and RULES.facts_markers
    assert "极限词" in RULES.summary()


def test_极限词是block并带行号与片段():
    r = _check(LABEL + "\n这款耳机音质第一，全球首发。")
    hits = _by_rule(r, "absolute_claims")
    assert {f.message for f in hits} >= {"出现「第一」", "出现「全球首」"}
    assert all(f.severity is Severity.BLOCK and f.line == 2 for f in hits)
    assert any("第一" in f.excerpt for f in hits)
    assert all(f.suggestion for f in hits)
    assert not r.passed and r.counts()["block"] >= 2


def test_医疗与金融承诺是block():
    r = _check(LABEL + "三天根治失眠，投资稳赚不赔。")
    assert _by_rule(r, "medical_claims") and _by_rule(r, "finance_claims")
    assert not r.passed


def test_敏感与诱导只是warn():
    r = _check(LABEL + "这个话题涉及宗教。点赞关注领取资料。")
    assert _by_rule(r, "sensitive_topics")[0].severity is Severity.WARN
    assert _by_rule(r, "inducement")[0].severity is Severity.WARN
    assert r.passed, "warn 不拦打包，交给人判断"


def test_数字缺出处():
    r = _check(LABEL + "今年销量涨了 300%。")
    assert _by_rule(r, "facts") and _by_rule(r, "facts")[0].severity is Severity.WARN
    r = _check(LABEL + "据财报，今年销量涨了 300%。")
    assert not _by_rule(r, "facts"), "句内有来源标记就不报"
    r = _check(LABEL + "一句没有数字的话。")
    assert not _by_rule(r, "facts")


def test_数字缺出处有上限():
    text = LABEL + "".join(f"第{i}条涨了 {i * 10}%。" for i in range(1, 10))
    assert len(_by_rule(_check(text), "facts")) == RULES.facts_max


def test_AIGC显式标识():
    r = _check("一段没有标识的 AI 文案。", generated=True)
    hit = _by_rule(r, "aigc_label")
    assert hit and hit[0].severity is Severity.BLOCK and not r.passed
    assert not _by_rule(_check("一段人写的文案。", generated=False), "aigc_label")
    assert not _by_rule(_check("文案正文（AIGC）", generated=True), "aigc_label")


def test_素材版权unknown():
    r = _check(LABEL, media_rights=[("as_aaaaaaaaaa", "unknown"), ("as_bbbbbbbbbb", "generated")])
    hits = _by_rule(r, "rights")
    assert len(hits) == 1 and "as_aaaaaaaaaa" in hits[0].message


def test_账号禁忌词来自账号层记忆():
    mems = MemoryStore()
    m = mems.put(
        Memory(
            layer=Layer.ACCOUNT,
            content="这个号不写硬广",
            keywords=[Keyword(term="硬广", polarity=Polarity.NEGATIVE,
                              category=Category.CONSTRAINT, origin_quote="别写硬广")],
            source=Source.HUMAN,
        )
    )
    mems.put(
        Memory(
            layer=Layer.PROJECT,
            content="项目层的不算",
            keywords=[Keyword(term="正文", polarity=Polarity.NEGATIVE)],
        )
    )
    r = ComplianceChecker(RULES, mems).check(LABEL + "这是一段硬广正文")
    hits = _by_rule(r, "brand")
    assert len(hits) == 1 and m.id in hits[0].message and "别写硬广" in hits[0].message
    assert not ComplianceChecker(RULES).check(LABEL + "硬广").findings, "没接记忆库就不查这项"


def test_报告渲染block在前():
    r = _check("第一名。今年涨了 50%。")  # 无标识：block；极限词：block；数字：warn
    text = r.render()
    assert text.startswith("合规机审") and "未通过" in text
    lines = [ln for ln in text.splitlines() if ln.startswith(("✗", "⚠", "·"))]
    assert lines[0].startswith("✗") and lines[-1].startswith("⚠")
    assert "未发现风险" in _check(LABEL + "一句干净的话。").render()


# ---------------------------------------------------------------- function


async def test_function产出报告资产并挂血缘():
    store = AssetStore()
    bus = EventBus()
    draft = store.create("最好的产品，AI 生成的文案", creator="model:k3", summary="草稿")
    img = store.create("", type_=AssetType.IMAGE, summary="配图", creator="")  # 版权 unknown
    reg = ToolRegistry(bus)
    reg.register(ComplianceFunctions(ComplianceChecker(RULES), store, bus))
    await reg.refresh()

    r = await reg.invoke(
        "check_compliance",
        {"asset_id": draft.id, "platform": "wechat", "media_asset_ids": [img.id]},
    )
    assert r.ok and "未通过" in r.content and "素材版权" in r.content
    report = store.get(r.asset_ref)
    assert report.type is AssetType.REPORT
    assert report.parent_ids == [draft.id, img.id]
    assert report.gen_params["passed"] is False and report.gen_params["platform"] == "wechat"
    assert draft.inline == "最好的产品，AI 生成的文案", "机审不改正文"

    ev = [e for e in bus.history if e.type is EventType.COMPLIANCE_CHECKED]
    assert ev and ev[-1].data["passed"] is False and ev[-1].data["report"] == report.id

    rules = await reg.invoke("list_compliance_rules", {})
    assert rules.ok and "极限词" in rules.content

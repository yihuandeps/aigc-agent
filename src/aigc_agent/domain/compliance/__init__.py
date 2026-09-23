"""M15 Compliance & Review —— 发布前的机审。

**机审只出风险清单 + 定位 + 建议改法，不做自动删改。** 半自动定位下，改不改是人的决定。
哪怕是 block 级的问题，这里也只是拦住"打包发布"这一步，正文一个字不动。

规则全在 config/compliance.yaml（数据不是代码），分六类：
  词表   广告法极限词 / 医疗功效 / 金融承诺 / 敏感话题 / 诱导互动
  事实   数字断言缺出处（句内没有来源标记）
  标识   AIGC 显式标识 —— 《人工智能生成合成内容标识办法》要求显式 + 隐式，
         显式在这里查，隐式由发布包 manifest 写（M16）
  版权   素材版权状态为 unknown 的不得直接用
  品牌   账号层记忆里 polarity=negative 的关键词出现在正文里
  （平台格式限制归 M16，不在这里）

不调模型：这是确定性检查，每次打包前都要跑，必须快、必须可复现。
需要判断的（"这个敏感话题到底能不能发"）标 warn 交给人，不替人判。
"""

from __future__ import annotations

import re
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ...capabilities.memory.store import Layer, MemoryStore, Polarity


class Severity(StrEnum):
    BLOCK = "block"  # 不改不能打包
    WARN = "warn"  # 需人工判断
    INFO = "info"


_ORDER = {Severity.BLOCK: 0, Severity.WARN: 1, Severity.INFO: 2}


class Finding(BaseModel):
    rule: str
    label: str = ""
    severity: Severity = Severity.WARN
    message: str
    line: int = 0  # 1 起；0 = 整篇
    excerpt: str = ""
    suggestion: str = ""

    def render(self) -> str:
        where = f"第 {self.line} 行" if self.line else "整篇"
        mark = {"block": "✗", "warn": "⚠", "info": "·"}[self.severity.value]
        head = f"{mark} [{self.severity.value}] {self.label or self.rule} · {where}：{self.message}"
        if self.excerpt:
            head += f"\n    …{self.excerpt}…"
        if self.suggestion:
            head += f"\n    → {self.suggestion}"
        return head


class ComplianceReport(BaseModel):
    asset_id: str = ""
    platform: str = "generic"
    findings: list[Finding] = Field(default_factory=list)
    checked_at: float = Field(default_factory=time.time)
    rules_version: str = ""

    @property
    def blocks(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.BLOCK]

    @property
    def warns(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARN]

    @property
    def passed(self) -> bool:
        """没有 block 级问题。warn 不拦打包，但会带进上传清单让人看。"""
        return not self.blocks

    def counts(self) -> dict[str, int]:
        return {
            "block": len(self.blocks),
            "warn": len(self.warns),
            "info": len([f for f in self.findings if f.severity is Severity.INFO]),
        }

    def render(self) -> str:
        c = self.counts()
        head = (
            f"合规机审（{self.platform}）：{'通过' if self.passed else '未通过'} · "
            f"block {c['block']} · warn {c['warn']} · info {c['info']}"
        )
        if not self.findings:
            return head + "\n未发现风险。机审覆盖有限，仍需人工终审。"
        ordered = sorted(self.findings, key=lambda f: (_ORDER[f.severity], f.line))
        return head + "\n\n" + "\n\n".join(f.render() for f in ordered)


# ---------------------------------------------------------------- 规则


def _applies(applies_to: list[str], asset_type: str) -> bool:
    """规则按资产类型生效：applies_to 为空 = 全部类型；没告诉类型 = 全查。"""
    return not applies_to or not asset_type or asset_type in applies_to


class WordRule(BaseModel):
    key: str
    label: str = ""
    severity: Severity = Severity.WARN
    words: list[str] = Field(default_factory=list)
    suggestion: str = ""
    # 只对这些资产类型生效（text / outline / script / storyboard …）。空 = 全部。
    # 广告法极限词是给营销文案的：打在剧本台词上，「最强」「绝对」「第一」全判 block，
    # 实测模型花了三个会话、四十多元把 60 集重写一遍去删这些词。
    applies_to: list[str] = Field(default_factory=list)


class ComplianceRules(BaseModel):
    version: str = ""
    word_rules: list[WordRule] = Field(default_factory=list)
    facts_severity: Severity = Severity.WARN
    facts_pattern: str = r"\d+(?:\.\d+)?\s*(?:%|％|倍|万|亿)"
    facts_markers: list[str] = Field(default_factory=list)
    facts_max: int = 5
    facts_suggestion: str = ""
    facts_applies_to: list[str] = Field(default_factory=list)
    aigc_severity: Severity = Severity.BLOCK
    aigc_markers: list[str] = Field(default_factory=list)
    aigc_applies_to: list[str] = Field(default_factory=list)
    aigc_text_label: str = "本内容由 AI 生成"
    aigc_media_label: str = ""
    aigc_suggestion: str = ""
    rights_severity: Severity = Severity.WARN
    rights_blocked: list[str] = Field(default_factory=lambda: ["unknown"])
    rights_suggestion: str = ""
    brand_severity: Severity = Severity.WARN
    brand_suggestion: str = ""

    @classmethod
    def load(cls, path: str | Path) -> ComplianceRules:
        path = Path(path)
        if not path.exists():
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        rules = []
        for key, spec in (raw.get("rules") or {}).items():
            spec = spec or {}
            rules.append(
                WordRule(
                    key=str(key),
                    label=str(spec.get("label") or key),
                    severity=Severity(str(spec.get("severity", "warn"))),
                    words=[str(w) for w in (spec.get("words") or []) if str(w).strip()],
                    suggestion=str(spec.get("suggestion") or ""),
                    applies_to=[str(t) for t in (spec.get("applies_to") or [])],
                )
            )
        facts = raw.get("facts") or {}
        aigc = raw.get("aigc_label") or {}
        rights = raw.get("rights") or {}
        brand = raw.get("brand") or {}
        return cls(
            version=str(raw.get("version") or ""),
            word_rules=rules,
            facts_severity=Severity(str(facts.get("severity", "warn"))),
            facts_pattern=str(
                facts.get("number_pattern") or cls.model_fields["facts_pattern"].default
            ),
            facts_markers=[str(m) for m in (facts.get("source_markers") or [])],
            facts_max=int(facts.get("max_findings", 5)),
            facts_suggestion=str(facts.get("suggestion") or ""),
            facts_applies_to=[str(t) for t in (facts.get("applies_to") or [])],
            aigc_severity=Severity(str(aigc.get("severity", "block"))),
            aigc_markers=[str(m) for m in (aigc.get("explicit_markers") or [])],
            aigc_applies_to=[str(t) for t in (aigc.get("applies_to") or [])],
            aigc_text_label=str(aigc.get("text_label") or "本内容由 AI 生成"),
            aigc_media_label=str(aigc.get("media_label") or ""),
            aigc_suggestion=str(aigc.get("suggestion") or ""),
            rights_severity=Severity(str(rights.get("severity", "warn"))),
            rights_blocked=[str(x) for x in (rights.get("blocked_states") or ["unknown"])],
            rights_suggestion=str(rights.get("suggestion") or ""),
            brand_severity=Severity(str(brand.get("severity", "warn"))),
            brand_suggestion=str(brand.get("suggestion") or ""),
        )

    def summary(self, platform: str = "") -> str:
        lines = [f"规则版本 {self.version or '—'}"]
        for r in self.word_rules:
            scope = f"，只查 {'/'.join(r.applies_to)}" if r.applies_to else ""
            lines.append(
                f"- {r.label}（{r.severity.value}，{len(r.words)} 个词{scope}）：{r.suggestion}"
            )
        lines.append(f"- 数字缺出处（{self.facts_severity.value}）：{self.facts_suggestion}")
        lines.append(f"- AIGC 显式标识（{self.aigc_severity.value}）：{self.aigc_suggestion}")
        lines.append(f"- 素材版权（{self.rights_severity.value}）：{self.rights_suggestion}")
        lines.append(f"- 品牌调性（{self.brand_severity.value}）：{self.brand_suggestion}")
        return "\n".join(lines)


# ---------------------------------------------------------------- 检查器

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")


def _locate(content: str, pos: int, span: int = 14) -> tuple[int, str]:
    line = content.count("\n", 0, pos) + 1
    start, end = max(0, pos - span), min(len(content), pos + span)
    return line, content[start:end].replace("\n", " ")


class ComplianceChecker:
    """确定性检查。memories 给了才做品牌调性那一项。"""

    def __init__(self, rules: ComplianceRules, memories: MemoryStore | None = None) -> None:
        self.rules = rules
        self.memories = memories

    def check(
        self,
        content: str,
        *,
        platform: str = "generic",
        generated: bool = True,
        media_rights: list[tuple[str, str]] | None = None,
        asset_id: str = "",
        asset_type: str = "",
    ) -> ComplianceReport:
        """asset_type（text / script / outline …）决定哪些规则生效；不给就全查。"""
        findings: list[Finding] = []
        findings += self._words(content, asset_type)
        if _applies(self.rules.facts_applies_to, asset_type):
            findings += self._facts(content)
        if _applies(self.rules.aigc_applies_to, asset_type):
            findings += self._aigc(content, generated)
        findings += self._rights(media_rights or [])
        findings += self._brand(content)
        return ComplianceReport(
            asset_id=asset_id,
            platform=platform,
            findings=findings,
            rules_version=self.rules.version,
        )

    # ---------- 各项 ----------

    def _words(self, content: str, asset_type: str = "") -> list[Finding]:
        out: list[Finding] = []
        for rule in self.rules.word_rules:
            if not _applies(rule.applies_to, asset_type):
                continue
            for word in rule.words:
                for m in re.finditer(re.escape(word), content):
                    line, excerpt = _locate(content, m.start())
                    out.append(
                        Finding(
                            rule=rule.key,
                            label=rule.label,
                            severity=rule.severity,
                            message=f"出现「{word}」",
                            line=line,
                            excerpt=excerpt,
                            suggestion=rule.suggestion,
                        )
                    )
        return out

    def _facts(self, content: str) -> list[Finding]:
        try:
            pat = re.compile(self.rules.facts_pattern)
        except re.error:
            return []
        out: list[Finding] = []
        pos = 0
        for sentence in _SENTENCE_SPLIT.split(content):
            start = pos
            pos += len(sentence)
            s = sentence.strip()
            if not s or not pat.search(s):
                continue
            if any(mk and mk in s for mk in self.rules.facts_markers):
                continue
            line, _ = _locate(content, start)
            out.append(
                Finding(
                    rule="facts",
                    label="数字缺出处",
                    severity=self.rules.facts_severity,
                    message="这句里的数字没有来源标记",
                    line=line,
                    excerpt=s[:40],
                    suggestion=self.rules.facts_suggestion,
                )
            )
            if len(out) >= self.rules.facts_max:
                break
        return out

    def _aigc(self, content: str, generated: bool) -> list[Finding]:
        if not generated:
            return []
        if any(mk and mk in content for mk in self.rules.aigc_markers):
            return []
        return [
            Finding(
                rule="aigc_label",
                label="AIGC 显式标识",
                severity=self.rules.aigc_severity,
                message=f"AI 生成的内容没有显式标识（如「{self.rules.aigc_text_label}」）",
                suggestion=self.rules.aigc_suggestion,
            )
        ]

    def _rights(self, media_rights: list[tuple[str, str]]) -> list[Finding]:
        out: list[Finding] = []
        for ref, rights in media_rights:
            if rights in self.rules.rights_blocked:
                out.append(
                    Finding(
                        rule="rights",
                        label="素材版权",
                        severity=self.rules.rights_severity,
                        message=f"素材 {ref} 的版权状态为 {rights}",
                        suggestion=self.rules.rights_suggestion,
                    )
                )
        return out

    def _brand(self, content: str) -> list[Finding]:
        """账号层的禁忌：polarity=negative 的关键词出现在正文里。"""
        if self.memories is None:
            return []
        out: list[Finding] = []
        seen: set[str] = set()
        for m in self.memories.all(layer=Layer.ACCOUNT):
            for k in m.keywords:
                term = k.term.strip()
                if k.polarity is not Polarity.NEGATIVE or len(term) < 2 or term in seen:
                    continue
                pos = content.find(term)
                if pos < 0:
                    continue
                seen.add(term)
                line, excerpt = _locate(content, pos)
                out.append(
                    Finding(
                        rule="brand",
                        label="品牌调性",
                        severity=self.rules.brand_severity,
                        message=(
                            f"出现账号禁忌词「{term}」"
                            f"（记忆 {m.id}：{k.origin_quote or m.content}）"
                        ),
                        line=line,
                        excerpt=excerpt,
                        suggestion=self.rules.brand_suggestion,
                    )
                )
        return out


def report_to_params(report: ComplianceReport) -> dict[str, Any]:
    """存进资产 gen_params 的紧凑形态。"""
    return {
        "platform": report.platform,
        "passed": report.passed,
        "counts": report.counts(),
        "findings": [f.model_dump(mode="json") for f in report.findings],
        "rules_version": report.rules_version,
    }

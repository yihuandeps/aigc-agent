"""Memory Brief —— M8.2 的输出契约。

格式定死，否则主 Agent 不知道怎么用。四个分区优先级递减，外加一个 conflicts：

    must      硬约束：账号层的禁忌、法务红线、人明确定下的要求
    must_not  避雷：本项目已否决的方案 + 打回理由 + 历史踩过的坑
    should    软偏好：倾向性、经数据验证的做法、以及模型推测出的约束（带标记）
    refs      参考：相关历史资产 id，按需取回
    conflicts 未消解的矛盾 —— 显式暴露给人，不由 Agent 私自裁决

`must_not` 是最重要的一块：半自动模式下人会反复打回，Agent 不记得为什么被打回，
第二次就会生成几乎一样的东西。这是这类工具最常见的体验崩塌点。

**规则版不调模型。** 分区靠 Memory 上已有的字段（layer / source / polarity / category）
就能判，每次节点启动、每轮对话前都要跑，加一次模型调用会让每轮都变慢。
需要判断的活（去重合并、冲突原因）由 MemoryAgent.consolidate() 交给子代理做。
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from .store import Category, Layer, Memory, MemoryStore, Polarity, Source

_ASSET_RE = re.compile(r"\bas_[0-9a-f]{10}\b")
INFERRED_TAG = "（推测）"
# 一次性的事实：报错码、余额、限流 —— 过两天就不成立了（「seedance 402 余额不足」之前永久生效）
_ERROR_FACT = re.compile(r"(?<!\d)(40[0-9]|42[0-9]|5\d\d)(?!\d)|余额不足|额度用尽|限流|宕机|超时")
ERROR_TTL = 2 * 86400
ASSET_TTL = 14 * 86400  # 指着具体资产 id 的推测：资产会被替代，两周后多半过期


def transient_ttl(content: str) -> float | None:
    """这条记忆多久后作废（秒）。不是一次性事实返回 None。"""
    if _ERROR_FACT.search(content or ""):
        return ERROR_TTL
    if _ASSET_RE.search(content or ""):
        return ASSET_TTL
    return None


def _stale(m: Memory, now: float) -> bool:
    """老数据没设 valid_until 的：推测来源的一次性事实按创建时间算过期。人说的不动。"""
    if m.source is not Source.INFERRED or m.valid_until is not None:
        return False
    ttl = transient_ttl(m.content)
    return ttl is not None and now - m.created_at > ttl


_PUNCT = re.compile(r"[\s“”\"'‘’「」『』（）()《》【】\[\]！!？?，,。.、；;：:…—\-]+")


def _norm(s: str) -> str:
    return _PUNCT.sub("", s.replace(INFERRED_TAG, "")).lower()


def _similar(a: str, b: str, threshold: float = 0.72) -> bool:
    """两条记忆说的是不是同一件事。

    提取是按批跑的，同一句话常被记成措辞略有出入的两条；实测简报里
    「开头不要写成硬广式震惊腔调」出现了两遍（差在"文案""的"和中英引号），
    「时长 30 秒以内」还因为一条标 positive 一条标 negative 被当成矛盾。
    字符级判据够用，不上模型：
      · 先去标点再比，引号风格不同不该算两条
      · 短句被长句覆盖 85% 以上就是同一条（长句只是多了修饰）
      · 数字不同的一律不算同一条 —— 「30 秒」和「60 秒」是真矛盾，不能合并掉
    """
    x, y = _norm(a), _norm(b)
    if not x or not y:
        return False
    if re.findall(r"\d+", x) != re.findall(r"\d+", y):
        return False
    if x in y or y in x:
        return True
    sm = SequenceMatcher(None, x, y)
    matched = sum(block.size for block in sm.get_matching_blocks())
    if matched / min(len(x), len(y)) >= 0.85:
        return True
    return sm.ratio() >= threshold


class MemoryBrief(BaseModel):
    must: list[str] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)
    should: list[str] = Field(default_factory=list)
    refs: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    # 每个分区各条来自哪些记忆 id，可回溯
    sources: dict[str, list[str]] = Field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not (self.must or self.must_not or self.should or self.refs or self.conflicts)

    def render(self) -> str:
        """注入节点契约 / 上下文的完整形态。"""
        if self.empty:
            return ""
        blocks: list[str] = []
        if self.must:
            blocks.append("### 必须遵守\n" + "\n".join(f"- {x}" for x in self.must))
        if self.must_not:
            blocks.append(
                "### 避雷（已被打回或明确禁止，不要重犯）\n"
                + "\n".join(f"- {x}" for x in self.must_not)
            )
        if self.should:
            blocks.append("### 建议\n" + "\n".join(f"- {x}" for x in self.should))
        if self.refs:
            blocks.append("### 参考资产（需要时用 read_asset 取回）\n" + ", ".join(self.refs))
        if self.conflicts:
            blocks.append(
                "### 未消解的矛盾（请向人确认，不要自行裁决）\n"
                + "\n".join(f"- {x}" for x in self.conflicts)
            )
        return "\n\n".join(blocks)

    def pin_text(self, limit: int = 8) -> str:
        """pin 进上下文的紧凑形态：只有 must / must_not。

        pin 区不占窗口配额、常驻，所以只放硬约束；should/refs 随召回走。
        """
        lines: list[str] = []
        if self.must:
            lines.append("必须：" + "；".join(self.must[:limit]))
        if self.must_not:
            lines.append("禁止/避雷：" + "；".join(self.must_not[:limit]))
        if self.conflicts:
            lines.append("待人裁决的矛盾：" + "；".join(self.conflicts[:3]))
        return "## 记忆简报\n" + "\n".join(lines) if lines else ""


def build_brief(
    store: MemoryStore,
    project_id: str = "",
    topic: str = "",
    stage: str = "",
    limit: int = 8,
) -> MemoryBrief:
    """规则版 Brief。不调模型。

    取哪些记忆：账号层全部 + 本项目（含未标项目的）全部；有 topic/stage 时按关键词
    过滤 should/refs，但 **must / must_not 不做相关性过滤** —— 避雷漏掉一条就是重犯一次。
    """
    now = time.time()
    pool = [
        m
        for m in store.all()
        if (m.layer is Layer.ACCOUNT or m.project_id in ("", project_id)) and not _stale(m, now)
    ]
    if not pool:
        return MemoryBrief()

    query = " ".join(x for x in (topic, stage) if x).strip()
    relevant: set[str] | None = None
    if query:
        terms = store.match_terms(query)
        relevant = set()
        if terms:
            relevant = {m.id for m in store.recall(terms, limit=50, project_id=project_id)}
        # 直接包含的也算相关（match_terms 只看索引词）
        for m in pool:
            if any(t and t in m.content for t in query.split()):
                relevant.add(m.id)

    def rank(m: Memory) -> tuple[float, float]:
        recency = 1.0 / (1.0 + (now - m.created_at) / 86400)
        return (m.weight * (0.5 + 0.5 * recency), m.created_at)

    must: list[tuple[Memory, str]] = []
    must_not: list[tuple[Memory, str]] = []
    should: list[tuple[Memory, str]] = []
    refs: list[str] = []
    polarity_by_term: dict[str, dict[Polarity, list[Memory]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for m in sorted(pool, key=rank, reverse=True):
        neg = any(k.polarity is Polarity.NEGATIVE for k in m.keywords)
        cats = {k.category for k in m.keywords}
        tag = INFERRED_TAG if m.source is Source.INFERRED else ""
        text = m.content.strip() + tag

        for k in m.keywords:
            if k.category in (Category.CONSTRAINT, Category.PREFERENCE) and k.polarity in (
                Polarity.POSITIVE,
                Polarity.NEGATIVE,
            ):
                polarity_by_term[k.term][k.polarity].append(m)

        for aid in _ASSET_RE.findall(m.content + " " + m.origin_ref):
            if aid not in refs and (relevant is None or m.id in relevant):
                refs.append(aid)

        if neg:
            # 避雷只认人说的（打回理由）和数据验证过的。推测出来的「别这么做」只当建议 ——
            # 2026-09-23 审查：「需逐镜 gen_video 传 image_urls（推测）」进了 must_not，
            # 和「镜头只能走 drama_render_shots」正面冲突，每轮 pin 着误导模型
            if m.source in (Source.HUMAN, Source.DATA):
                must_not.append((m, text))
            elif relevant is None or m.id in relevant:
                should.append((m, "避免：" + text))
        elif Category.CONSTRAINT in cats:
            # 硬约束只认人说的和数据验证过的；推测出的只能当建议 —— 推测不得晋升
            if m.source in (Source.HUMAN, Source.DATA):
                must.append((m, text))
            else:
                should.append((m, text))
        elif Category.PREFERENCE in cats:
            if relevant is None or m.id in relevant:
                should.append((m, text))
        # topic / entity 类不进 Brief：它们是召回索引，不是指导

    conflicts: list[str] = []
    for term, by_pol in polarity_by_term.items():
        pos, neg_ = by_pol.get(Polarity.POSITIVE, []), by_pol.get(Polarity.NEGATIVE, [])
        if not (pos and neg_):
            continue
        # 同一句话被打成两种极性不算矛盾 —— 那是提取抖动，不是人改口
        pairs = [(p_, n_) for p_ in pos for n_ in neg_ if not _similar(p_.content, n_.content)]
        if not pairs:
            continue
        p_, n_ = pairs[0]
        a = p_.keywords[0].origin_quote or p_.content
        b = n_.keywords[0].origin_quote or n_.content
        conflicts.append(f"「{term}」：既有要求（{a[:40]}）也有禁止（{b[:40]}）")

    def dedupe(items: list[tuple[Memory, str]]) -> tuple[list[str], list[str]]:
        """近似重复只留排在前面的那条（rank 高的先来）。"""
        texts: list[str] = []
        ids: list[str] = []
        for m, t in items:
            if any(_similar(t, kept) for kept in texts):
                continue
            texts.append(t)
            ids.append(m.id)
            if len(texts) >= limit:
                break
        return texts, ids

    must_t, must_ids = dedupe(must)
    must_not_t, must_not_ids = dedupe(must_not)
    should_t, should_ids = dedupe(should)
    return MemoryBrief(
        must=must_t,
        must_not=must_not_t,
        should=should_t,
        refs=refs[:limit],
        conflicts=conflicts[:limit],
        sources={"must": must_ids, "must_not": must_not_ids, "should": should_ids},
    )

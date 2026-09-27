"""M8.1 Memory Store —— 被动存储层。

**纯 CRUD + 检索，不含任何智能。** 主 Agent 可以直接读（快路径），
不必经过 Memory Agent。判断类的活（冲突消解、晋升、提炼、关键词提取）
全在 M8.2 Memory Agent，P3 才建。

P1 只做一件事，但这件事必须现在做：**把打回理由落库。**
Checkpoint 一旦存在就会产生打回理由，而这类数据是**补不回来**的——
等 P3 做 Memory Agent 时，前两个阶段积累的理由如果没存，冷启动就没素材。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections import defaultdict
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class Layer(StrEnum):
    ACCOUNT = "account"  # 人设、调性、禁忌、爆款特征 —— 长期，人工维护
    PROJECT = "project"  # 本次选题、已定方案、**已否决的方案及原因**
    SESSION = "session"  # 当前对话临时状态


class Source(StrEnum):
    HUMAN = "human"  # 人明确说的 —— 最高可信
    DATA = "data"  # 从回流数据推出的
    INFERRED = "inferred"  # 模型推测 —— **不得自动晋升到账号层**


class Polarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Category(StrEnum):
    CONSTRAINT = "constraint"  # 要求与禁止 —— 最高价值
    PREFERENCE = "preference"  # 倾向与评价
    TOPIC = "topic"
    ENTITY = "entity"


class Keyword(BaseModel):
    """压缩产物，同时是召回索引。

    `polarity` 是纯关键词方案的补丁：用户说「开头太硬广了，别这么写」，
    只存 [开头, 硬广] 的话，下次召回分不清是**要**还是**不要**硬广。
    `origin_quote` 更值——召回时把原话直接给模型，比一堆孤立词管用得多。
    """

    term: str
    polarity: Polarity = Polarity.NEUTRAL
    category: Category = Category.TOPIC
    origin_quote: str = ""


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: "mem_" + uuid.uuid4().hex[:10])
    layer: Layer = Layer.PROJECT
    content: str  # 一条记忆只装一个事实
    keywords: list[Keyword] = Field(default_factory=list)
    origin_ref: str = ""  # 指向被压缩的原始对话/checkpoint，可回溯

    source: Source = Source.INFERRED
    confidence: float = 0.5
    weight: float = 1.0

    valid_until: float | None = None
    supersedes: list[str] = Field(default_factory=list)
    superseded_by: str | None = None

    project_id: str = ""
    # 打回发生在哪个环节（内容类打回挂在环节上，2026-09-26）；其余记忆留空
    stage: str = ""
    created_at: float = Field(default_factory=time.time)
    last_hit_at: float | None = None
    hit_count: int = 0

    @property
    def alive(self) -> bool:
        if self.superseded_by:
            return False
        return self.valid_until is None or self.valid_until > time.time()


# 内容类打回理由的有效期（天）。2026-09-26：之前打回理由一律永久有效、每轮 pin ——
# 蜘蛛精剧的「控制在十二集」到《不渡》里还每轮 pin 着。人要长期生效就 /remember 或晋升账号层
REJECTION_TTL_DAYS = 30.0


def in_scope(m: Memory, project_id: str) -> bool:
    """这条记忆在这个项目里算不算数（召回、简报、pin 都按它，2026-09-26）：
      · 账号层：全局，哪个项目都算
      · 有项目上下文：只认本项目的 —— **没有项目键的旧记忆隔离**（不召回、不 pin），等人认领
        （`agent memory claim`，或对话里 /remember 记忆id）。之前它们对所有剧都生效
      · 没有项目上下文（脚本、测试）：只看没有项目键的
    """
    if m.layer is Layer.ACCOUNT:
        return True
    return m.project_id == project_id


_NEGATIVE_WORDS = ("不要", "别", "不许", "不准", "禁止", "不能", "不得", "严禁", "避免", "不用")
_TERM_SPLIT = re.compile(r"[\s，。！？、；：,.!?;:\n\"'“”（）()【】\[\]「」]+")


class MemoryStore:
    """P1 形态：JSON 文件 + 内存倒排索引。

    为什么先不上向量库：关键词精确匹配 + 倒排在内部工具的数据量级下够用。
    等召回准确率成为瓶颈再加 —— 见 ARCHITECTURE.md M8.1。
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root
        self._items: dict[str, Memory] = {}
        self._index: dict[str, set[str]] = defaultdict(set)  # term -> memory ids
        if root:
            root.mkdir(parents=True, exist_ok=True)
            self._load()

    # ---------- 写 ----------

    def put(self, memory: Memory) -> Memory:
        self._items[memory.id] = memory
        for kw in memory.keywords:
            self._index[kw.term].add(memory.id)
        if self.root:
            (self.root / f"{memory.id}.json").write_text(
                memory.model_dump_json(indent=2), encoding="utf-8"
            )
        return memory

    def record_rejection(
        self,
        reason: str,
        node_id: str,
        run_id: str,
        target_node: str = "",
        project_id: str = "",
        decision: str = "revise",
        candidates: list[str] | None = None,
        ttl_days: float | None = REJECTION_TTL_DAYS,
    ) -> Memory:
        """打回理由落库 —— P1 的核心动作。

        `target_node` 是被退回去重跑的那个节点。**索引主要挂在它身上**：
        真正需要这条记忆的是即将重跑的节点，不是做决策的人审节点。
        打回发生在 review，但「开头太硬广」这条要在 draft 下一次跑之前拿到。

        P1 不做关键词提取（需要模型，是 M8.2 的活，P3 建）。这里把**原文存下来**，
        `keywords` 的语义字段等 P3 回填 —— 原文不存就永远补不回来，词晚点提没关系。

        2026-09-26：挂到项目和环节上、默认 30 天后过期（ttl_days=None 永久）——
        一句打回理由是对那一版内容的意见，不是这个号永远的规矩。
        """
        target = target_node or node_id
        terms = {target, node_id}
        return self.put(
            Memory(
                layer=Layer.PROJECT,
                content=f"「{target}」曾被打回（{decision}）：{reason}",
                keywords=[
                    Keyword(
                        term=t,
                        polarity=Polarity.NEGATIVE,
                        category=Category.CONSTRAINT,
                        origin_quote=reason[:60],
                    )
                    for t in sorted(terms)
                ],
                origin_ref=f"run:{run_id}#{node_id}->{target}",
                source=Source.HUMAN,  # 人明确说的，最高可信
                confidence=1.0,
                weight=2.0,  # 打回理由权重高于一般记忆
                project_id=project_id,
                stage=target,
                valid_until=(time.time() + ttl_days * 86400) if ttl_days else None,
            )
        )

    def remember(self, text: str, project_id: str = "", everywhere: bool = False) -> Memory:
        """人在对话里亲口定的规则（/remember）：来源 human、硬约束、不过期，每轮 pin。

        默认只对这个项目生效；everywhere=True 记到账号层（所有项目都算）。
        含「不要 / 别 / 禁止…」的记成禁止（进避雷），其余记成要求（进必须遵守）。
        """
        body = " ".join((text or "").split())
        if not body:
            raise ValueError("要记的话是空的")
        neg = any(w in body for w in _NEGATIVE_WORDS)
        terms = [p for p in _TERM_SPLIT.split(body) if 2 <= len(p) <= 12][:6] or [body[:12]]
        return self.put(
            Memory(
                layer=Layer.ACCOUNT if everywhere else Layer.PROJECT,
                content=body,
                keywords=[
                    Keyword(
                        term=t,
                        polarity=Polarity.NEGATIVE if neg else Polarity.POSITIVE,
                        category=Category.CONSTRAINT,
                        origin_quote=body[:60],
                    )
                    for t in dict.fromkeys(terms)
                ],
                origin_ref="human:/remember",
                source=Source.HUMAN,
                confidence=1.0,
                weight=2.0,
                project_id="" if everywhere else project_id,
            )
        )

    def claim(self, memory_id: str, project_id: str) -> Memory:
        """认领一条没有项目键的旧记忆：挂到 project_id 上，从此只在这个项目里生效。"""
        if not project_id:
            raise ValueError("认领要给项目键")
        m = self.get(memory_id)
        if m.layer is Layer.ACCOUNT:
            raise ValueError(f"{memory_id} 是账号层记忆，本来就对所有项目生效，不用认领")
        m.project_id = project_id
        return self.put(m)

    def unclaimed(self) -> list[Memory]:
        """没有项目键、又不是账号层的有效记忆 —— 有项目的会话里被隔离、等人认领的那些。"""
        return [m for m in self.all() if m.layer is not Layer.ACCOUNT and not m.project_id]

    def find(self, query: str, project_id: str = "") -> list[Memory]:
        """/forget 用：按记忆 id 或内容里的关键词找这个项目看得到的记忆（含账号层、
        和等人认领的旧记忆 —— 人要作废的多半就是它们）。"""
        q = (query or "").strip()
        if not q:
            return []
        if q in self._items and self._items[q].alive:
            return [self._items[q]]
        key = "".join(q.split()).lower()
        return [
            m for m in self.all()
            if (in_scope(m, project_id) or not m.project_id)
            and key in "".join(m.content.split()).lower()
        ]

    def promote(self, memory_id: str, layer: Layer = Layer.ACCOUNT, by: str = "human") -> Memory:
        """晋升到账号层：从"这个项目这么要求"变成"这个号一直这么要求"。

        **只能由人或数据回流触发。** inferred 来源的记忆不得自动晋升 —— 否则
        记忆库会被模型的臆测慢慢污染，而这种污染是渐进的、很难察觉的。
        """
        if by not in ("human", "data"):
            raise ValueError("晋升只能由 human 或 data 触发，模型不能自动晋升")
        m = self.get(memory_id)
        m.layer = layer
        m.source = Source.HUMAN if by == "human" else Source.DATA
        if by == "human":
            m.confidence = 1.0
        return self.put(m)

    def forget(self, memory_id: str, by: str = "human") -> Memory:
        """作废一条记忆。只标记不删：留着复盘，且冲突链要能回溯。"""
        m = self.get(memory_id)
        m.superseded_by = f"{by}:forget"
        return self.put(m)

    # ---------- 读（快路径，无判断）----------

    def get(self, memory_id: str) -> Memory:
        if memory_id not in self._items:
            raise KeyError(f"记忆不存在：{memory_id}")
        return self._items[memory_id]

    def all(self, layer: Layer | None = None, project_id: str = "") -> list[Memory]:
        out = [m for m in self._items.values() if m.alive]
        if layer:
            out = [m for m in out if m.layer is layer]
        if project_id:
            out = [m for m in out if m.project_id == project_id]
        return sorted(out, key=lambda m: m.created_at)

    def match_terms(self, query: str, min_len: int = 2) -> list[str]:
        """从自由文本里找出**索引里真实存在**的词。

        倒排索引是精确匹配，而中文没有空格分词：存的是「短视频开头」，
        用户问「帮我写个开头」，直接切词一个都对不上 —— 实测召回全空。

        做法是反过来：拿索引里已有的词去查询串里找。词表是有界的
        （最多几千个），扫一遍很便宜，而且不用引入分词器或调模型
        （召回在每轮的快路径上，加一次模型调用会让每轮都变慢）。

        双向包含：索引词出现在查询里算命中（「硬广」⊂「别写硬广」），
        查询片段出现在索引词里也算（「开头」⊂「短视频开头」）。
        """
        q = (query or "").strip()
        if len(q) < min_len:
            return []
        hits = [t for t in self._index if len(t) >= min_len and t in q]
        if hits:
            return hits
        # 反向：查询里的片段是不是某个索引词的一部分
        grams = {q[i : i + n] for n in (2, 3, 4) for i in range(len(q) - n + 1)}
        return [t for t in self._index if any(g in t for g in grams)]

    def recall(
        self, terms: list[str], limit: int = 5, project_id: str = "", touch: bool = True
    ) -> list[Memory]:
        """关键词召回。按 权重 × 命中数 × 时间新近 排序。touch=False：只查，不计命中
        （简报的主题预筛每轮拉 50 条，不是真正注入上下文的，之前也算命中、每轮落盘 50 次）。"""
        hits: dict[str, int] = defaultdict(int)
        for t in terms:
            for mid in self._index.get(t, ()):
                hits[mid] += 1

        scored = []
        now = time.time()
        for mid, n in hits.items():
            m = self._items[mid]
            # 有项目上下文：只召回本项目 + 账号层，没有项目键的旧记忆隔离（2026-09-26：之前
            # 放行，蜘蛛精剧的规则在《不渡》里照样召回）
            if not m.alive or (project_id and not in_scope(m, project_id)):
                continue
            recency = 1.0 / (1.0 + (now - m.created_at) / 86400)
            scored.append((m.weight * n * (0.5 + 0.5 * recency), m))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [m for _, m in scored[:limit]]
        if not touch:
            return picked
        for m in picked:
            m.hit_count += 1
            m.last_hit_at = now
            # 落盘：之前只改内存，重启就清零，「哪些记忆真的有用」永远统计不出来
            self.put(m)
        return picked

    def render_brief(self, memories: list[Memory]) -> str:
        """极简版 Memory Brief。P3 由 Memory Agent 产出完整四区结构。

        避雷只放人说的 / 数据验证过的「不要」。模型推测出来的「不要」放参考、标「推测」——
        之前一律标成「本项目已被打回过」（2026-09-26）：「需逐镜 gen_video 传 image_urls」
        这种推测被当成人打回过的规矩召回，和系统提示词正面冲突。"""
        if not memories:
            return ""

        def negative(m: Memory) -> bool:
            return any(k.polarity is Polarity.NEGATIVE for k in m.keywords)

        must_not = [m for m in memories if negative(m) and m.source in (Source.HUMAN, Source.DATA)]
        other = [m for m in memories if m not in must_not]

        parts = []
        if must_not:
            parts.append(
                "### 避雷（本项目已被打回过，不要重犯）\n"
                + "\n".join(f"- {m.content}" for m in must_not)
            )
        if other:
            parts.append(
                "### 参考\n"
                + "\n".join(
                    f"- {'避免：' if negative(m) else ''}{m.content}"
                    + ("（推测）" if m.source is Source.INFERRED else "")
                    for m in other
                )
            )
        return "\n\n".join(parts)

    # ---------- 持久化 ----------

    def _load(self) -> None:
        if not self.root:
            return
        for f in self.root.glob("mem_*.json"):
            try:
                m = Memory.model_validate(json.loads(f.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001 — 单条坏数据不该让整个库起不来
                continue
            self._items[m.id] = m
            for kw in m.keywords:
                self._index[kw.term].add(m.id)

    def __len__(self) -> int:
        return len(self._items)

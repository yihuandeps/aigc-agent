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
    created_at: float = Field(default_factory=time.time)
    last_hit_at: float | None = None
    hit_count: int = 0

    @property
    def alive(self) -> bool:
        if self.superseded_by:
            return False
        return self.valid_until is None or self.valid_until > time.time()


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
    ) -> Memory:
        """打回理由落库 —— P1 的核心动作。

        `target_node` 是被退回去重跑的那个节点。**索引主要挂在它身上**：
        真正需要这条记忆的是即将重跑的节点，不是做决策的人审节点。
        打回发生在 review，但「开头太硬广」这条要在 draft 下一次跑之前拿到。

        P1 不做关键词提取（需要模型，是 M8.2 的活，P3 建）。这里把**原文存下来**，
        `keywords` 的语义字段等 P3 回填 —— 原文不存就永远补不回来，词晚点提没关系。
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
            )
        )

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

    def recall(self, terms: list[str], limit: int = 5, project_id: str = "") -> list[Memory]:
        """关键词召回。按 权重 × 命中数 × 时间新近 排序。"""
        hits: dict[str, int] = defaultdict(int)
        for t in terms:
            for mid in self._index.get(t, ()):
                hits[mid] += 1

        scored = []
        now = time.time()
        for mid, n in hits.items():
            m = self._items[mid]
            if not m.alive or (project_id and m.project_id and m.project_id != project_id):
                continue
            recency = 1.0 / (1.0 + (now - m.created_at) / 86400)
            scored.append((m.weight * n * (0.5 + 0.5 * recency), m))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [m for _, m in scored[:limit]]
        for m in picked:
            m.hit_count += 1
            m.last_hit_at = now
        return picked

    def render_brief(self, memories: list[Memory]) -> str:
        """极简版 Memory Brief。P3 由 Memory Agent 产出完整四区结构。"""
        if not memories:
            return ""
        must_not = [m for m in memories if any(k.polarity is Polarity.NEGATIVE for k in m.keywords)]
        other = [m for m in memories if m not in must_not]

        parts = []
        if must_not:
            parts.append(
                "### 避雷（本项目已被打回过，不要重犯）\n"
                + "\n".join(f"- {m.content}" for m in must_not)
            )
        if other:
            parts.append("### 参考\n" + "\n".join(f"- {m.content}" for m in other))
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

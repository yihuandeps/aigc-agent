"""M8.2 Memory Agent —— 长期记忆的生成、召回与整理。

**P1 缺的就是这一块。** 在此之前：滑窗到 15 轮、淘汰 5 轮，那 5 轮
直接丢弃 —— 长期记忆永远不会被写入。`ShortTermMemory.on_evict` 钩子
留着但没人挂，`memory_extract` / `memory_recall` 两个角色配了但没人调。

它是**独立 SubAgent**，不是主循环里的一个函数调用：

  · 独立上下文 —— 提关键词时不该看见主对话的工具定义和能力目录，
    那些既贵又会干扰判断
  · 异步执行 —— 触发驱逐的那一轮恰好也是缓存击穿的那一轮，
    两个开销叠在同一轮上，人会明显感觉卡顿
  · 失败不上抛 —— 记忆提取失败不该让用户的这一轮对话跟着失败

压缩方式是**提取关键词 + 留原话**。纯关键词会丢掉"要还是不要"：
用户说「开头太硬广了，别这么写」，只存 [开头, 硬广] 的话，
下次召回分不清是要还是不要。所以 Keyword 带 polarity，并且把
origin_quote 一起存 —— 召回时把原话给模型，比一堆孤立词管用得多。

三条路径（ARCHITECTURE M8.2）：
  快路径   recall()       每轮：切词 → 倒排索引 → 渲染，不调模型
  压缩路径 submit()       每 5 轮：异步队列 → memory_extract → 落库
  慢路径   brief() / consolidate()   节点边界：产出 Memory Brief。
           brief() 规则版不调模型；consolidate() 走 M10 子代理做去重合并与标冲突，
           失败退回规则版。这是 M10 的参考实现：无状态、只读、输出 schema 定死。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from typing import Any

from ..subagents import SubAgentDef, SubAgentRunner
from .brief import MemoryBrief, _similar, build_brief, transient_ttl
from .store import Category, Keyword, Layer, Memory, MemoryStore, Polarity, Source, in_scope

EXTRACT_ROLE = "memory_extract"
RECALL_ROLE = "memory_recall"
CONSOLIDATE_ROLE = "memory_consolidate"

# 一次最多存几条。提太多等于没提 —— 召回时全是噪音。
MAX_PER_BATCH = 6


def extract_prompt(transcript: str) -> str:
    return (
        "下面是一段已经滑出上下文窗口的对话。把其中**值得长期记住的事实**抽出来。\n\n"
        f"{transcript}\n\n"
        "判断标准（按价值从高到低）：\n"
        "  constraint  用户明确的要求或禁止（"
        "「开头别写成硬广」「时长控制在 30 秒」）—— 最高价值\n"
        "  preference  用户的倾向和评价（「我更喜欢口语一点的」）\n"
        "  entity      具体的人、项目、账号、文件路径\n"
        "  topic       正在做的选题方向\n\n"
        "**只抽真正会影响后续工作的**。寒暄、一次性的中间过程、"
        "模型自己的推理，都不要。宁可少抽几条，也不要塞一堆噪音 ——\n"
        "记忆是要被召回注入上下文的，噪音会挤掉真正有用的。\n\n"
        "每条记忆还要给 polarity：\n"
        "  positive 用户要的 / negative 用户明确不要的 / neutral 中性事实\n"
        "**这个字段不能省**：「别写成硬广」只存关键词的话，"
        "下次召回分不清是要还是不要硬广。\n\n"
        f"最多 {MAX_PER_BATCH} 条。没有值得记的就返回空数组。\n\n"
        "只输出 JSON 数组：\n"
        '[{"content": "一句话事实", "terms": ["关键词1", "关键词2"], '
        '"polarity": "negative", "category": "constraint", '
        '"quote": "用户的原话片段", "confidence": 0.9}]'
    )


def parse_memories(raw: str, origin_ref: str, project_id: str = "") -> list[Memory]:
    """解析并**过滤**。宁可丢掉一条可疑的，也不要污染长期记忆 ——
    记忆是会被反复注入上下文的，错一条会持续影响后面每一轮。
    """
    try:
        data = json.loads(_unwrap(raw))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []

    out: list[Memory] = []
    for item in data[:MAX_PER_BATCH]:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if len(content) < 4:
            continue
        terms = [str(t).strip() for t in (item.get("terms") or []) if str(t).strip()]
        if not terms:
            continue  # 没有关键词就召不回来，存了也是死数据

        pol = _enum(item.get("polarity"), Polarity, Polarity.NEUTRAL)
        cat = _enum(item.get("category"), Category, Category.TOPIC)
        quote = str(item.get("quote") or "")[:80]
        try:
            conf = float(item.get("confidence") or 0.6)
        except (TypeError, ValueError):
            conf = 0.6

        ttl = transient_ttl(content)
        out.append(
            Memory(
                layer=Layer.PROJECT,
                content=content,
                # 一次性事实（报错码、余额、某份资产）设过期：之前永久有效，两天前的
                # 「余额不足」到今天还 pin 着（2026-09-23 审查）
                valid_until=(time.time() + ttl) if ttl else None,
                keywords=[
                    Keyword(term=t, polarity=pol, category=cat, origin_quote=quote)
                    for t in dict.fromkeys(terms)
                ],
                origin_ref=origin_ref,
                # 从对话里提的一律算推测，**不得自动晋升到账号层** ——
                # 模型把"这次这么说"当成"一直都这样"是最常见的记忆污染。
                source=Source.INFERRED,
                confidence=min(1.0, max(0.1, conf)),
                weight=1.5 if cat is Category.CONSTRAINT else 1.0,
                project_id=project_id,
            )
        )
    return out


def _enum(raw: Any, cls: Any, default: Any) -> Any:
    v = str(raw or "").strip().lower()
    for m in cls:
        if v == m.value:
            return m
    return default


def _unwrap(text: str) -> str:
    raw = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if m:
        raw = m.group(1).strip()
    i, j = raw.find("["), raw.rfind("]")
    return raw[i : j + 1] if i >= 0 and j > i else raw


def terms_of(text: str) -> list[str]:
    """从用户输入里切出召回用的词。

    不分词、不调模型 —— 召回在主循环的**快路径**上，每轮都要跑，
    加一次模型调用会让每轮都变慢。按标点和空白切段，再按长度过滤，
    倒排索引那边是精确匹配，切得糙一点不影响命中已存的词。
    """
    parts = re.split(r"[\s，。！？、；：,.!?;:\n\"'“”（）()【】\[\]]+", text or "")
    return [p for p in parts if 2 <= len(p) <= 12][:20]


# ---------------------------------------------------------------- 完整模式（M10 参考实现）

CONSOLIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["must", "must_not", "should", "conflicts"],
    "properties": {
        "must": {"type": "array"},
        "must_not": {"type": "array"},
        "should": {"type": "array"},
        "conflicts": {"type": "array"},
    },
}

CONSOLIDATE_DEF = SubAgentDef(
    name="memory_consolidate",
    description="整理一份记忆简报：合并重复、去掉过期、把矛盾标出来交给人",
    system_prompt=(
        "你是记忆整理员。收到一份按规则拼出来的记忆简报，把它整理成主 Agent 能直接用的形态。\n"
        "你要做的只有三件事：\n"
        "  1. 合并说的是同一件事的条目，措辞压成一句话\n"
        "  2. 明显过期或互相覆盖的，保留新的\n"
        "  3. 互相矛盾的**不要裁决**，写进 conflicts 交给人\n"
        "两条硬规则：\n"
        "  · 避雷（must_not）只能合并措辞，**不能删掉任何一条**——漏一条就是重犯一次\n"
        "  · 带「（推测）」标记的不能升级成硬约束，最多留在 should 里"
    ),
    tools=[],  # 只做判断，不碰任何工具；写库由 MemoryAgent 校验后自己做
    role=CONSOLIDATE_ROLE,
    output_schema=CONSOLIDATE_SCHEMA,
    stateless=True,  # 硬规则：代理本身无记忆，状态全在 Store
    max_iterations=2,
)


def consolidate_prompt(brief: MemoryBrief) -> str:
    return "以下是规则拼出的记忆简报：\n\n" + brief.render() + "\n\n按输出契约整理它。"


class MemoryAgent:
    """独立上下文的记忆代理。主循环只往队列里丢，不等它。"""

    def __init__(
        self,
        gateway: Any,
        store: MemoryStore,
        project_id: str = "",
        queue_size: int = 32,
        runner: SubAgentRunner | None = None,
    ) -> None:
        self.gateway = gateway
        self.store = store
        self.project_id = project_id
        self.runner = runner  # M10 运行器；没有就只用规则版 Brief
        # 队列里带着入队时的项目键：/out 换了项目之后，排着没提取完的旧项目对话不能记到
        # 新项目名下（2026-09-26）
        self._queue: asyncio.Queue[tuple[str, str, str]] = asyncio.Queue(maxsize=queue_size)
        self._task: asyncio.Task[None] | None = None
        self.extracted = 0  # 统计用，也方便测试断言

    # ---------- 生命周期 ----------

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._worker())

    async def close(self) -> None:
        """把队列里剩的处理完再退。

        不等的话，最后几轮的记忆会丢 —— 而对话结尾往往正是
        用户给明确要求的地方（"下次别这么写"）。
        """
        if self._task is None:
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._queue.join(), timeout=30)
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    # ---------- 写：淘汰的轮次 → 长期记忆 ----------

    def submit(self, transcript: str, origin_ref: str = "", project_id: str | None = None) -> bool:
        """非阻塞入队。队列满了就丢弃并返回 False —— 记忆提取是
        锦上添花，不值得为它把主循环卡住。

        project_id 不给就按入队这一刻的项目键记（提取是异步的，处理时项目可能已经换了）。
        """
        project = self.project_id if project_id is None else project_id
        try:
            self._queue.put_nowait((transcript, origin_ref, project))
        except asyncio.QueueFull:
            return False
        self.start()
        return True

    async def _worker(self) -> None:
        while True:
            transcript, origin_ref, project = await self._queue.get()
            try:
                await self._extract_one(transcript, origin_ref, project)
            except Exception:  # noqa: BLE001
                # 记忆提取失败不该让用户那一轮跟着失败，吞掉
                pass
            finally:
                self._queue.task_done()

    async def _extract_one(
        self, transcript: str, origin_ref: str, project_id: str | None = None
    ) -> int:
        if not transcript.strip():
            return 0
        project = self.project_id if project_id is None else project_id
        resp = await self.gateway.chat(
            EXTRACT_ROLE, [{"role": "user", "content": extract_prompt(transcript)}]
        )
        mems = parse_memories(resp.text, origin_ref, project)
        kept = 0
        # 只和这个项目看得到的比（本项目 + 账号层）：之前也和没有项目键的旧记忆去重 ——
        # 用户在新剧里又说了一遍，结果只给那条被隔离的旧记忆加了分量，新剧里还是没有
        existing = [m for m in self.store.all() if in_scope(m, project)]
        for m in mems:
            # 写入去重：同一件事之前记过（措辞略有出入）就不再存一条，给旧的加点分量 ——
            # 之前每批提取都新存，同一句「开头别写硬广」在简报里出现两三遍
            dup = next((old for old in existing if _similar(old.content, m.content)), None)
            if dup is not None:
                dup.weight = min(3.0, dup.weight + 0.2)
                dup.hit_count += 1
                self.store.put(dup)
                continue
            self.store.put(m)
            existing.append(m)
            kept += 1
        self.extracted += kept
        return kept

    async def extract_now(self, transcript: str, origin_ref: str = "") -> int:
        """同步提取，给测试和 CLI 用。主循环不要调这个。"""
        return await self._extract_one(transcript, origin_ref)

    # ---------- 读：召回 ----------

    def recall(self, query: str, limit: int = 5) -> str:
        """快路径召回：切词 → 倒排索引 → 渲染成 Brief。

        **不调模型**。这在主循环的每一轮都要跑，加一次模型调用
        会让每轮都变慢，而且召回本身是检索问题不是生成问题。
        """
        # 先按索引里真实存在的词去匹配（中文没分词，直接切词对不上），
        # 匹配不到再退回朴素切词 —— 后者对英文和带标点的输入仍然有效。
        terms = self.store.match_terms(query) or terms_of(query)
        if not terms:
            return ""
        hits = self.store.recall(terms, limit=limit, project_id=self.project_id)
        return self.store.render_brief(hits) if hits else ""

    # ---------- 慢路径：Memory Brief ----------

    def brief(self, topic: str = "", stage: str = "", limit: int = 8) -> MemoryBrief:
        """规则版 Brief，不调模型。节点启动前、每轮对话前都可以跑。"""
        return build_brief(self.store, self.project_id, topic=topic, stage=stage, limit=limit)

    async def consolidate(self, topic: str = "", stage: str = "") -> MemoryBrief:
        """完整模式：规则先出 Brief，再让子代理做需要判断的部分。

        子代理只做去重合并、标冲突，**不裁决、不删避雷、不晋升**。
        返回的东西要过三道校验，任何一道不过就退回规则版：
          1. 结构合法（output_schema）
          2. must_not 条数不少于规则版 —— 避雷只增不减
          3. 每条都是非空字符串
        """
        base = self.brief(topic, stage)
        if self.runner is None or base.empty:
            return base

        r = await self.runner.run(CONSOLIDATE_DEF, task=consolidate_prompt(base))
        if not r.ok or not isinstance(r.data, dict):
            return base

        def strs(key: str) -> list[str]:
            return [str(x).strip() for x in (r.data.get(key) or []) if str(x).strip()]

        must, must_not, should, conflicts = (
            strs("must"), strs("must_not"), strs("should"), strs("conflicts")
        )
        if len(must_not) < len(base.must_not):
            must_not = base.must_not  # 模型删了避雷，不采信
        if len(must) < len(base.must):
            must = base.must  # 硬约束同样只增不减
        return MemoryBrief(
            must=must,
            must_not=must_not,
            should=should,
            refs=base.refs,
            conflicts=conflicts or base.conflicts,
            sources=base.sources,
        )

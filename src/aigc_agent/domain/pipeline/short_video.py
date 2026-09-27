"""抖音短视频分支（2026-09-18 用户定的新路径）—— 纯逻辑部分。

用户定的路径：
  关键词 + 确认风格 → RPA 抓抖音/小红书相关热点 → 分析归纳成**新的内容**（选题简报）
  → 据此决定怎么做、做多长 → 需要实拍/外部素材的镜头：① 向用户要 ② 联网找可用素材
  → 生成/配音/字幕/合成 → 看片复核

这里只放不依赖网关和资产库的东西：给规划模型的提示词、简报的解析与校验、
时长决策、素材需求清单、向用户要素材的问法。调模型、存资产、跑生成在
functions/short_video.py。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

CHARS_PER_SECOND = 4.5  # 中文口播语速估算
SOURCES = ("generate", "real")  # 镜头素材来源：AI 生成 / 实拍或外部素材


@dataclass
class Shot:
    desc: str
    seconds: float = 0.0
    source: str = "generate"
    real_need: str = ""  # source=real 时：需要什么实拍/外部素材
    search_terms: list[str] = field(default_factory=list)  # 素材站搜索词（英文为主）

    def as_dict(self) -> dict[str, Any]:
        return {
            "desc": self.desc,
            "seconds": self.seconds,
            "source": self.source,
            "real_need": self.real_need,
            "search_terms": list(self.search_terms),
        }


@dataclass
class Brief:
    """选题简报：抓来的热点经分析归纳后的**新内容**，以及据此定下的做法。"""

    keyword: str = ""
    style: str = ""  # 配方名（= 风格）
    angle: str = ""
    title: str = ""
    summary: str = ""
    key_facts: list[dict[str, str]] = field(default_factory=list)
    hook: str = ""
    duration: int = 30
    why_duration: str = ""
    script: str = ""
    shots: list[Shot] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # 热点资产 id
    materials: dict[str, str] = field(default_factory=dict)  # 镜头序号 → 素材资产 id

    # ---------- 派生 ----------

    @property
    def script_chars(self) -> int:
        return int(self.duration * CHARS_PER_SECOND)

    def material_needs(self) -> list[tuple[int, Shot]]:
        """需要实拍/外部素材的镜头（1 起的序号）。"""
        return [(i, s) for i, s in enumerate(self.shots, 1) if s.source == "real"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "keyword": self.keyword,
            "style": self.style,
            "angle": self.angle,
            "title": self.title,
            "summary": self.summary,
            "key_facts": list(self.key_facts),
            "hook": self.hook,
            "duration": self.duration,
            "why_duration": self.why_duration,
            "script": self.script,
            "shots": [s.as_dict() for s in self.shots],
            "risks": list(self.risks),
            "sources": list(self.sources),
            "materials": dict(self.materials),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Brief:
        shots = [
            Shot(
                desc=str(s.get("desc") or ""),
                seconds=float(s.get("seconds") or 0),
                source=str(s.get("source") or "generate"),
                real_need=str(s.get("real_need") or ""),
                search_terms=[str(t) for t in (s.get("search_terms") or [])],
            )
            for s in (d.get("shots") or [])
            if isinstance(s, dict)
        ]
        return cls(
            keyword=str(d.get("keyword") or ""),
            style=str(d.get("style") or ""),
            angle=str(d.get("angle") or ""),
            title=str(d.get("title") or ""),
            summary=str(d.get("summary") or ""),
            key_facts=[x for x in (d.get("key_facts") or []) if isinstance(x, dict)],
            hook=str(d.get("hook") or ""),
            duration=int(d.get("duration") or 30),
            why_duration=str(d.get("why_duration") or ""),
            script=str(d.get("script") or ""),
            shots=shots,
            risks=[str(r) for r in (d.get("risks") or [])],
            sources=[str(s) for s in (d.get("sources") or [])],
            materials={str(k): str(v) for k, v in (d.get("materials") or {}).items()},
        )

    def render(self) -> str:
        """给人看的简报。"""
        lines = [f"【{self.title or self.keyword}】{self.angle}", ""]
        lines.append(self.summary)
        if self.key_facts:
            lines.append("")
            lines.append("依据：")
            for f in self.key_facts:
                src = f.get("source") or "未标来源"
                lines.append(f"  · {f.get('fact', '')}（{src}）")
        lines += ["", f"时长 {self.duration}s —— {self.why_duration}", f"钩子：{self.hook}"]
        lines += ["", f"口播（{len(self.script)} 字，目标 ≤{self.script_chars}）：", self.script]
        lines += ["", f"分镜 {len(self.shots)} 个："]
        for i, s in enumerate(self.shots, 1):
            tag = "生成" if s.source == "generate" else "实拍/外部"
            extra = f"　需要：{s.real_need}" if s.source == "real" and s.real_need else ""
            lines.append(f"  {i}. [{tag}] {s.desc}（{s.seconds:g}s）{extra}")
        needs = self.material_needs()
        if needs:
            lines += [
                "",
                f"需要实拍/外部素材的镜头：{len(needs)} 个"
                "（用 request_materials 向用户要，或 stock_media_search 联网找）",
            ]
        if self.risks:
            lines += ["", "风险提示：" + "；".join(self.risks)]
        return "\n".join(lines)


# ---------------------------------------------------------------- 提示词


def brief_prompt(
    keyword: str,
    style_label: str,
    style_desc: str,
    sources_text: str,
    min_seconds: int,
    max_seconds: int,
    notes: str = "",
    *,
    prompt_hint: str = "",
    script_hint: str = "",
    voiceover: bool = True,
    shot_seconds: float = 3.0,
) -> str:
    """给规划模型：从抓来的热点里分析归纳出新内容，并决定怎么做、做多长。

    prompt_hint / script_hint：配方里这个风格最关键的写法要求（2026-09-23 审查：之前只喂风格名和
    适用场景，ugc-vlog / product-ad 的写法要点只有老链在用，新链的简报根本没看到）。
    voiceover=False：这个风格不配 TTS 口播（口播出镜靠出镜人原声、产品片靠画面）。
    """
    facts = sources_text.strip() or "（没有抓到任何热点数据 —— 只能按常识写，务必在 risks 里注明）"
    extra = f"\n\n用户额外要求：{notes}" if notes else ""
    chars = int(max_seconds * CHARS_PER_SECOND)
    hints = ""
    if prompt_hint.strip():
        hints += f"\n\n这个风格的画面写法（配方要求，照做）：\n{prompt_hint.strip()}"
    if script_hint.strip():
        hints += "\n\n这个风格的口播写法：\n" + script_hint.strip().replace("{chars}", str(chars))
    vo = (
        "口播按每秒 4.5 字控制。"
        if voiceover
        else "**这个风格不配 TTS 口播**：script 写出镜人对着镜头要说的话（口播出镜）或旁白备选，"
        "时长按画面和内容定，不按字数拉长。"
    )
    return (
        f"关键词：{keyword}\n风格：{style_label} —— {style_desc}{extra}{hints}\n\n"
        "下面是抓到的热点材料（真实数据，带热度或互动数）。注意：标着「全站热榜」的和关键词"
        "不一定有关 —— 只拿和关键词真正相关的条目当依据，无关的不要硬套成出处：\n\n"
        f"{facts}\n\n"
        "请完成三件事：\n"
        "1. **分析归纳**：把这些热点交叉比对，找出真正在发生的事、大家关心的点、"
        "还没被讲透的角度，归纳成一段**新的内容**（不是复述某一条）。事实只能来自上面的材料，"
        "每条依据标明出处；材料里没有的数字不要编，用定性描述。\n"
        f"2. **决定做法与时长**：时长在 {min_seconds}–{max_seconds} 秒之间，由内容密度决定"
        f"（信息点多、需要举例就长；一个爆点讲透就短），说明理由。{vo}\n"
        "3. **分镜与素材来源**：每个镜头写成具体可拍的画面。能用 AI 生成的标 source=generate；"
        "必须用真实画面的（真实产品/人物/事件现场、真实数据截图、用户本人出镜）标 source=real，"
        "写清 real_need（要什么素材）和 search_terms（去素材站搜的英文关键词，2–4 个）。"
        "画面里不要出现文字、字幕、logo。镜头按播放顺序排（成片按这个顺序剪），"
        "镜头数不要多于「时长 ÷ 每刀秒数」—— 多出来的镜头进不了成片。\n\n"
        "只输出一个 JSON：\n"
        "{\n"
        '  "angle": "切入角度一句话",\n'
        '  "title": "视频标题，20 字内",\n'
        '  "summary": "归纳出的新内容，3–6 句",\n'
        '  "key_facts": [{"fact": "一条依据", "source": "出处（热榜第几条/哪条笔记）"}],\n'
        '  "hook": "开头 3 秒的钩子",\n'
        '  "duration_seconds": 30,\n'
        '  "why_duration": "为什么是这个时长",\n'
        '  "script": "口播全文",\n'
        f'  "shots": [{{"desc": "画面", "seconds": {shot_seconds:g}, "source": "generate", '
        '"real_need": "", "search_terms": []}],\n'
        '  "risks": ["事实存疑/合规/版权提示"]\n'
        "}"
    )


def _unwrap_json(text: str) -> str:
    raw = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if m:
        return m.group(1).strip()
    i, j = raw.find("{"), raw.rfind("}")
    return raw[i : j + 1] if i >= 0 and j > i else raw


def parse_brief(
    text: str,
    keyword: str,
    style: str,
    min_seconds: int,
    max_seconds: int,
    voiceover: bool = True,
) -> tuple[Brief | None, list[str]]:
    """解析规划模型的输出并校验。返回 (简报, 警告)。解析不出来简报为 None。

    voiceover=False 的风格不按口播字数拉长时长（2026-09-23 审查：实测关了口播的配方
    15 秒被拉到 20 秒）。"""
    try:
        data = json.loads(_unwrap_json(text))
    except json.JSONDecodeError as e:
        return None, [f"模型输出不是合法 JSON：{e}"]
    if not isinstance(data, dict):
        return None, ["模型输出不是 JSON 对象"]
    warns: list[str] = []
    b = Brief(keyword=keyword, style=style)
    b.angle = str(data.get("angle") or "").strip()
    b.title = str(data.get("title") or keyword).strip()[:40]
    b.summary = str(data.get("summary") or "").strip()
    b.key_facts = [
        {"fact": str(x.get("fact") or ""), "source": str(x.get("source") or "")}
        for x in (data.get("key_facts") or [])
        if isinstance(x, dict) and x.get("fact")
    ]
    b.hook = str(data.get("hook") or "").strip()
    b.why_duration = str(data.get("why_duration") or "").strip()
    b.script = str(data.get("script") or "").strip()
    b.risks = [str(r) for r in (data.get("risks") or []) if str(r).strip()]
    if not b.script:
        return None, ["输出里没有 script（口播全文）"]

    # 时长：模型定，但夹在风格允许的范围内；口播写长了以口播为准往上抬（画面要盖住旁白）
    try:
        want = int(float(data.get("duration_seconds") or 0))
    except (TypeError, ValueError):
        want = 0
    if want <= 0:
        want = max(min_seconds, min(max_seconds, math.ceil(len(b.script) / CHARS_PER_SECOND)))
        warns.append(f"模型没给时长，按口播字数定为 {want}s")
    if want < min_seconds or want > max_seconds:
        clamped = max(min_seconds, min(max_seconds, want))
        warns.append(
            f"模型给的时长 {want}s 超出风格范围 {min_seconds}–{max_seconds}s，改为 {clamped}s"
        )
        want = clamped
    need = math.ceil(len(b.script) / CHARS_PER_SECOND)
    if voiceover and need > want * 1.15:
        new = min(max_seconds, need)
        warns.append(f"口播 {len(b.script)} 字约需 {need}s，时长从 {want}s 调到 {new}s")
        if need > max_seconds:
            warns.append(f"口播仍超过风格上限 {max_seconds}s，成片会以配音长度为准，建议精简文案")
        want = new
    b.duration = want

    shots: list[Shot] = []
    for s in data.get("shots") or []:
        if not isinstance(s, dict) or not str(s.get("desc") or "").strip():
            continue
        src = str(s.get("source") or "generate").strip().lower()
        if src not in SOURCES:
            src = "real" if src in ("user", "stock", "footage", "photo") else "generate"
        try:
            sec = float(s.get("seconds") or 0)
        except (TypeError, ValueError):
            sec = 0.0
        terms = [str(t).strip() for t in (s.get("search_terms") or []) if str(t).strip()]
        shots.append(
            Shot(
                desc=str(s["desc"]).strip(),
                seconds=sec,
                source=src,
                real_need=str(s.get("real_need") or "").strip(),
                search_terms=terms[:4],
            )
        )
    if not shots:
        return None, warns + ["输出里没有 shots（分镜）"]
    # 没给秒数的按时长均分
    if any(s.seconds <= 0 for s in shots):
        each = round(b.duration / len(shots), 1)
        for s in shots:
            if s.seconds <= 0:
                s.seconds = each
    b.shots = shots
    return b, warns


# ---------------------------------------------------------------- 向用户要素材


def materials_question(brief: Brief, items: list[tuple[int, Shot]] | None = None) -> str:
    """挂起问人的文案：需要哪些素材、三种回应方式。"""
    needs = items if items is not None else brief.material_needs()
    lines = [f"「{brief.title}」有 {len(needs)} 个镜头需要实拍或外部素材："]
    for i, s in needs:
        terms = f"（素材站可搜：{'、'.join(s.search_terms)}）" if s.search_terms else ""
        lines.append(f"  第{i}镜 · {s.desc}\n      需要：{s.real_need or '真实画面'}{terms}")
    lines += [
        "",
        "请按镜头回复，三种方式任选：",
        "  ① 给文件：把素材放到任意目录后贴路径，如「第3镜 E:\\素材\\产品.mp4」"
        "（图片也行，会做成镜头）",
        "  ② 联网找：回复「第3镜 联网找」，我去 Pexels / Pixabay 这类可商用素材站搜",
        "  ③ 改生成：回复「第3镜 生成」，改用 AI 生成画面",
        "全部让我联网找就回复「都联网找」，全部改生成回复「都生成」。",
    ]
    return "\n".join(lines)


def parse_material_reply(text: str, indices: list[int]) -> dict[int, tuple[str, str]]:
    """解析人的回复 → {镜头序号: (方式, 参数)}。方式：file / online / generate。

    看不懂的镜头不出现在结果里，调用方按「未答复」处理。
    """
    t = (text or "").strip()
    out: dict[int, tuple[str, str]] = {}
    if not t:
        return out
    if re.search(r"(都|全部|全都)\s*(联网|上网)", t):
        return {i: ("online", "") for i in indices}
    if re.search(r"(都|全部|全都)\s*生成", t):
        return {i: ("generate", "") for i in indices}
    # 逐镜：第3镜 xxx  /  3: xxx。一段到下一个「第N镜」为止 —— 之前只按换行和分号断，
    # 「第1镜 生成，第2镜 联网找」只认出第 1 镜；路径后面跟着逗号还会把后半句吞进路径
    marks = _marks(t)
    for k, m in enumerate(marks):
        i = int(m.group(1) or m.group(2))
        if i not in indices:
            continue
        end = marks[k + 1].start() if k + 1 < len(marks) else len(t)
        body = re.split(r"[\n；;]", t[m.end() : end], maxsplit=1)[0]
        body = body.strip().lstrip("：:=").strip().strip("「」\"' ").rstrip("，,。、 ")
        body = body.strip("「」\"' ")
        # 先认路径：目录名里可能带"生成""联网"这种词（如 E:\AI生成\...），不能先按关键词判
        if _looks_like_path(body):
            out[i] = ("file", body)
        elif re.search(r"联网|上网|素材站|去找", body):
            out[i] = ("online", "")
        elif re.search(r"生成|AI", body, re.I):
            out[i] = ("generate", "")
    return out


_MARK = re.compile(r"第\s*(\d+)\s*镜|(?<![\d.\w])(\d+)\s*[:：、]")
_SEP = " \t\r\n,，;；。、"


def _marks(t: str) -> list[re.Match[str]]:
    """段落标记（「第N镜」「N:」）。标记所在的词（前后到空白 / 分隔符为止）像路径、标记又不在
    词首的，是文件名的一部分，不算：「第3镜 E:/素材/第2镜.mp4」之前被切成两个镜头；
    「第1镜生成第2镜联网找」这种不带分隔符的照样认。"""
    out: list[re.Match[str]] = []
    for m in _MARK.finditer(t):
        s = max(t.rfind(c, 0, m.start()) for c in _SEP) + 1
        e = min((x for x in (t.find(c, m.end()) for c in _SEP) if x >= 0), default=len(t))
        if s < m.start() and _looks_like_path(t[s:e]):
            continue
        out.append(m)
    return out

_PATH_HINT = re.compile(
    r"^(?:[A-Za-z]:[\\/]|[\\/]{1,2}|~[\\/]|\./)|\.(?:mp4|mov|mkv|webm|avi|m4v|png|jpe?g|webp|gif)$",
    re.I,
)


def _looks_like_path(text: str) -> bool:
    return bool(_PATH_HINT.search(text.strip()))

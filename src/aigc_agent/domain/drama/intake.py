"""判断用户给的是**一句话想法**还是**完整剧本**。

这一步判错的代价不对称，两个方向都很痛：

  想法误判成剧本 → 拿"一个霸总爱上我"去拆分镜，模型只能硬编，
                    出来的东西和用户脑子里的完全不是一回事
  剧本误判成想法 → 让用户重写他已经写好的剧本，最直接的冒犯

所以**不靠模型猜**，用可解释的结构信号打分。剧本这种东西结构特征极强：
场景标头、`角色：台词` 的冒号行、分集标记、多段落 —— 一句话想法一个都没有。
信号足够强才下判断，中间地带如实说"看不准"并让用户确认。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- 结构信号

# 场景标头。三种常见写法：[夜] [内] [走廊] / 夜。走廊。/ 场景：走廊
_SLUG = re.compile(
    r"^\s*(?:\[[^\]]{1,8}\]\s*){2,}"  # [夜] [内] [走廊]
    r"|^\s*(?:日|夜|清晨|傍晚|黄昏|深夜)[。．.，,]\s*\S"  # 夜。酒店走廊。
    r"|^\s*(?:场景|地点|时间)\s*[:：]",
    re.M,
)

# 台词行：`角色名：台词`。名字不长、冒号后有实质内容。
_DIALOGUE = re.compile(r"^\s*[^\s：:]{1,12}\s*[：:]\s*\S{2,}", re.M)

# 分集/场次标记
_EPISODE = re.compile(r"第\s*\d+\s*(?:集|场|幕)|EP\s*\d+|Episode\s*\d+", re.I)

# 明显在描述"想做什么"而不是剧本本身
_INTENT = re.compile(
    r"我想|我要|帮我|做一[部个条]|想做|题材|大概讲|讲的是|构思|点子|idea", re.I
)


@dataclass
class Intake:
    kind: str  # "script" | "idea" | "unclear"
    score: int  # 剧本结构得分，越高越像剧本
    signals: list[str] = field(default_factory=list)

    @property
    def is_script(self) -> bool:
        return self.kind == "script"

    @property
    def is_idea(self) -> bool:
        return self.kind == "idea"

    def brief(self) -> str:
        name = {"script": "完整剧本", "idea": "一句话想法", "unclear": "看不准"}[self.kind]
        return f"{name}（{'、'.join(self.signals) if self.signals else '无明显结构特征'}）"


def classify(text: str) -> Intake:
    """按结构信号判断。阈值取得保守：宁可说"看不准"也不要判错。"""
    t = (text or "").strip()
    if not t:
        return Intake("idea", 0, ["空输入"])

    lines = [ln for ln in t.splitlines() if ln.strip()]
    slugs = len(_SLUG.findall(t))
    dialogues = len(_DIALOGUE.findall(t))
    episodes = len(_EPISODE.findall(t))

    score = 0
    signals: list[str] = []
    if slugs:
        score += 3
        signals.append(f"{slugs} 个场景标头")
    if dialogues >= 2:
        score += 3
        signals.append(f"{dialogues} 行对白")
    elif dialogues == 1:
        score += 1
    if episodes:
        score += 2
        signals.append(f"{episodes} 处分集标记")
    if len(lines) >= 8:
        score += 2
        signals.append(f"{len(lines)} 行")
    elif len(lines) >= 4:
        score += 1
    if len(t) >= 400:
        score += 2
        signals.append(f"{len(t)} 字")
    elif len(t) >= 150:
        score += 1

    # 明确在陈述需求的措辞，是想法的强信号 —— 但只在结构很弱时才压分，
    # 否则"我想做的这个剧本：[夜][内]..."会被误判。
    if _INTENT.search(t) and score <= 3:
        score -= 2
        signals.append("措辞像在描述需求")

    if score >= 5:
        return Intake("script", score, signals)
    if score <= 2:
        return Intake("idea", score, signals or [f"{len(t)} 字、{len(lines)} 行"])
    return Intake("unclear", score, signals)


# ---------------------------------------------------------------- 提问文本


def ask_unclear(text: str) -> str:
    n = len((text or "").strip())
    return (
        f"这段内容 {n} 字，结构上看不准是**完整剧本**还是**一句话想法**。\n\n"
        "  · 如果是剧本 → 我直接进拆解流程\n"
        "  · 如果是想法 → 我先按短剧方法论帮你写成剧本，再往下走\n\n"
        "请用户确认一下。"
    )


def ask_script_next() -> str:
    """拿到完整剧本时问：直接进工程，还是先扩写修改。"""
    return (
        "收到完整剧本。往下走之前先定一件事：\n\n"
        "  [bold]1. 直接进工程[/] —— 按现有剧本拆分镜、建资产、生成视频。\n"
        "     适合剧本已经定稿。\n\n"
        "  [bold]2. 先扩写／修改[/] —— 按短剧方法论过一遍：补钩子、调节奏曲线、\n"
        "     加爽点、查合规红线，改完再进工程。\n"
        "     适合初稿，或者自己也觉得还差点意思。\n\n"
        "后面的生成很贵很慢（一集六段视频约 22 分钟），"
        "剧本不满意就往下跑是最亏的 —— 所以这一步值得停一下。"
    )

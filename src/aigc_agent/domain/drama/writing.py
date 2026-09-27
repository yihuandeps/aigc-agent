"""一句话想法 → 剧本 / 剧本扩写修改。

方法论取自 `skills/drama-script`（用户移植进来的那套）：钩子设计、节奏曲线、
爽点矩阵、反派设计、开篇规则、付费点、合规红线。这里把要点压成提示词，
不走 skill 的完整状态机 —— 那套是给对话式创作用的，一步步确认；
这里要的是一次调用出稿，让用户看到东西再决定改不改。
"""

from __future__ import annotations

from typing import Any

_METHOD = """短剧的方法论要点（来自 drama-script）：

开篇：**前 3 秒定生死**。第一个镜头就要有冲突或反常，不要交代背景、
不要铺垫人物关系 —— 观众没有耐心等你铺。

钩子：每集结尾必须断在**信息缺口**上（真相揭一半、危机刚落下、
身份即将暴露），不是断在情绪高点。情绪会散，缺口不会。

节奏曲线：短剧不是匀速的。每 30-40 秒要有一个小反转或信息增量，
每集至少一个大转折。连续两分钟没有新信息，观众就划走了。

爽点：**憋屈要具体，反击要即时**。憋屈铺得越具体（谁羞辱、怎么羞辱、
当着谁的面），反击的爽感越强；但憋屈不能超过一集，隔集兑现就凉了。

反派：反派要有**可理解的动机**，不能纯坏。纯坏的反派让观众觉得假，
而"他这么做有他的道理"才让冲突立得住。

人物：主角要有一个**具体的软肋**（母亲的手术费、弟弟的学费、
一句没说出口的道歉），所有选择都要能追溯到它。"""

_FORMAT = """输出格式（直接出剧本，不要大纲、不要解释）：

第X集

[时间] [内外] [地点]

动作描写。用具体可拍的动作和神态，不写心理活动。

角色名：台词

继续动作描写。

角色名：台词

要求：
· 场景标头单独成行，格式就是 [夜] [内] [酒店走廊] 这样
· 台词写成 `角色名：内容`，一行一句
· 动作描写和台词交替，不要整段动作后面跟一堆台词
· 台词要口语，允许结巴、语气词、不完整句
· 不要写运镜和景别 —— 那是下一步分镜师的活，这里只要剧本"""


def write_prompt(idea: str, episodes: int = 1, minutes: float = 0, fmt: Any = None) -> str:
    """一句话 → 剧本。minutes 不传就按一集规格（config/drama.yaml，默认 4 分钟）。"""
    from dataclasses import replace

    from .format import DEFAULT_FORMAT, writing_rules

    spec = fmt or DEFAULT_FORMAT
    if minutes and minutes > 0:
        spec = replace(spec, minutes=float(minutes), follow_script=False)
    length = (
        f"每集时长按剧情需要定（参考约 {spec.minutes:g} 分钟、{spec.script_chars} 字）"
        if spec.follow_script
        else f"每集约 {spec.minutes:g} 分钟（约 {spec.script_chars} 字）"
    )
    return (
        f"用户的想法：{idea}\n\n"
        f"{_METHOD}\n\n"
        f"按这套方法写 **{episodes} 集**短剧剧本，{length}。\n\n"
        f"{writing_rules(spec)}\n\n"
        f"{_FORMAT}"
    )


def expand_prompt(script: str, note: str = "", fmt: Any = None) -> str:
    """已有剧本 → 扩写/修改。"""
    from .format import DEFAULT_FORMAT, writing_rules

    spec = fmt or DEFAULT_FORMAT
    extra = f"\n\n用户的额外要求：{note}" if note else ""
    extra += f"\n\n{writing_rules(spec)}"
    return (
        f"这是用户已有的剧本：\n\n{script}\n\n"
        f"{_METHOD}\n\n"
        "按上面的方法论过一遍，**改写并输出完整剧本**。重点看四件事：\n"
        "  1. 开篇够不够狠 —— 前 3 秒有没有冲突或反常\n"
        "  2. 结尾断在不断在信息缺口上\n"
        "  3. 中间有没有连续两分钟不给新信息的段落\n"
        "  4. 憋屈和反击是不是同一集内兑现\n\n"
        "**保留原剧本的人物、设定和主线**，不要重起炉灶 —— 用户要的是改，不是换。"
        "改动要能看出来是在原基础上加固，而不是另写一个故事。"
        f"{extra}\n\n"
        f"{_FORMAT}"
    )

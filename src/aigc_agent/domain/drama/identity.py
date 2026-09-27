"""人物一致性（2026-09-18 用户反馈：参考图传了也漂，人物漂移严重）—— 纯逻辑部分。

三件事：
  1. **参考锁定段**：参考图传给模型只是"给了"，模型不一定用。seedance 要在提示词里用
     @图片N 点名"这张是谁、要一致什么"，所以每段视频提示词开头加一段参考锁定。
  2. **一致性校验**：生成后把参考图和生成结果（抽帧或图）一起给视觉模型，逐人判断
     是不是同一个人（脸型五官、发型发色、体型、服装），给 0–10 分。
  3. **重生成提示**：不合格时把具体差异写进提示词再生成一次（"发型变短了" 比 "要一致" 有用）。

调模型、抽帧、存资产在 functions/drama.py；这里只有文本与解析，可离线测试。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

REFERENCE_HEAD = "【参考锁定】"

_KIND_DESC = {
    "角色": "角色「{name}」本人：脸型五官、发型发色、体型必须与这张图完全一致",
    "服装": "「{name}」这套服装：款式、颜色、材质一致（脸仍以角色主形象为准）",
    # 三视图：2026-09-22 起不再生成，留着是为了老参考图包还能渲
    "三视图": "角色「{name}」的体型比例参考",
    "场景": "场景「{name}」的环境与布光",
    "道具": "道具「{name}」的外形",
}


# 不继承声明（2026-09-23，来自 awesome-seedance 案例库：能不能锁住人，差别就在这一句）。
# 我们的参考图是纯白底证件照和白底全身服装图 —— 不说清楚，模型会把白背景、站姿、
# 正面构图和影棚光一起搬进镜头里。参考图只负责"这个人长什么样、穿什么"，别的都不是。
NOT_INHERIT = (
    "参考图只取上述属性；**不要**把参考图里的纯白背景、站姿、正面构图、画角和原始光线搬进画面，"
    "那些是资产图不是分镜。转头、低头、说话、手靠近脸、快速运动时仍必须是同一张脸。"
)


def reference_block(labels: list[tuple[str, str]]) -> str:
    """参考图清单 → 提示词里的参考锁定段。labels 与传给模型的 image 列表同序：[(类别, 名字)]。"""
    if not labels:
        return ""
    parts = []
    for i, (kind, name) in enumerate(labels, 1):
        desc = _KIND_DESC.get(kind, "{name}").format(name=name)
        parts.append(f"@图片{i} = {desc}")
    return (
        f"{REFERENCE_HEAD}画面中的人物、服装、场景、道具严格以参考图为准，"
        "不得自行更换长相、发型或服装：" + "；".join(parts) + "。" + NOT_INHERIT
    )


IDENTITY_RETRY = (
    "【重新生成】上一版画面里的人物与参考图不是同一个人（{issues}），属于不合格。"
    "这次必须严格按参考图：脸型五官、发型发色、体型、服装都与 @图片 中的角色一致，"
    "不要美化、不要换人、不要换装。 "
    "REGENERATION — the previous output did not match the reference character ({issues}). "
    "Match the reference images exactly: same face, hairstyle, body and outfit; do not "
    "beautify or swap the person."
)


_RETRY_PREFIX = IDENTITY_RETRY.split("（", 1)[0]


def identity_retry_prompt(prompt: str, issues: list[str]) -> str:
    """把差异写进提示词最前面（禁字幕的最外层禁令由 gen_video 再包一次）。幂等。"""
    if prompt.startswith(_RETRY_PREFIX):
        return prompt
    why = "；".join(i for i in issues if i)[:200] or "长相或服装变了"
    return f"{IDENTITY_RETRY.format(issues=why)}\n{prompt}"


_CHECK_PROMPT = (
    "下面先给参考图（每张标了是谁/什么），然后给生成结果（一段视频按时间均匀抽出的几帧，"
    "或一张生成图）。请逐个判断生成结果里出现的人物是否与参考图是**同一个人**："
    "脸型与五官、发型与发色、体型、服装款式与颜色。\n"
    "给 0–10 的一致性分：10 = 同一个人同一装扮；7 分以下 = 明显不是同一个人、或发型/服装变了。"
    "画面里没出现某个参考角色不算扣分。\n"
    '只输出 JSON：{"pass": true, "score": 8, "issues": ["陆离的发型从长发变成短发"], '
    '"characters": {"陆离": 8}}'
)


def identity_check_messages(
    refs: list[tuple[str, str]], targets: list[str], is_video: bool
) -> list[dict[str, Any]]:
    """refs: [(标签, 图片 data URL 或链接)]；targets: 生成结果的帧/图 data URL。"""
    content: list[dict[str, Any]] = [{"type": "text", "text": _CHECK_PROMPT}]
    for label, url in refs:
        content.append({"type": "text", "text": f"参考图：{label}"})
        content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
    content.append(
        {"type": "text", "text": "生成结果（视频抽帧，按时间顺序）：" if is_video else "生成结果："}
    )
    for u in targets:
        content.append(
            {"type": "image_url", "image_url": {"url": u, "detail": "low" if is_video else "high"}}
        )
    return [{"role": "user", "content": content}]


@dataclass
class IdentityVerdict:
    passed: bool = True
    score: int = -1
    issues: list[str] = field(default_factory=list)
    characters: dict[str, int] = field(default_factory=dict)
    note: str = ""  # 解析不了时的说明
    # 这次没查成（调用失败 / 没本地副本 / 抽不出帧 / 输出读不出来）—— 视频段按「没过」算：
    # 不进成片、重跑时补查（2026-09-27 用户定的）。只有 note 没有 unchecked = 配置里就没开，
    # 不算没查成
    unchecked: bool = False


def parse_identity_verdict(text: str, pass_score: int = 7) -> IdentityVerdict:
    """解析视觉模型的判定。解析不出来按通过处理但写 note —— 校验不能把生成链路一起挂掉。"""
    raw = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if m:
        raw = m.group(1).strip()
    i, j = raw.find("{"), raw.rfind("}")
    if i < 0 or j <= i:
        return IdentityVerdict(note="一致性校验输出不是 JSON", unchecked=True)
    try:
        data = json.loads(raw[i : j + 1])
    except json.JSONDecodeError:
        return IdentityVerdict(note="一致性校验输出不是合法 JSON", unchecked=True)
    try:
        score = int(float(data.get("score", -1)))
    except (TypeError, ValueError):
        score = -1
    chars: dict[str, int] = {}
    raw_chars = data.get("characters")
    for k, v in (raw_chars.items() if isinstance(raw_chars, dict) else []):
        try:
            chars[str(k)] = int(float(v))
        except (TypeError, ValueError):
            continue
    issues = [str(x) for x in (data.get("issues") or []) if str(x).strip()]
    if score >= 0:
        passed = score >= pass_score and all(v >= pass_score for v in chars.values())
    else:
        p = data.get("pass")
        passed = p if isinstance(p, bool) else True
    return IdentityVerdict(passed=passed, score=score, issues=issues, characters=chars)

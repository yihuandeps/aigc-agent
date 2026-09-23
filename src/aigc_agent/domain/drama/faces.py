"""角色面容审查（2026-09-22 用户提的问题）—— 纯逻辑部分。

用户的原话：生成过程里会出现"同一个人物但面容不同"的图，需要只保留一张，
否则后面的内容会打架。真实现场：角色「阿蛛」的主形象被一家模型的内容护栏拒了，
换了三家模型各出一张，产物目录里就躺着三张脸；谁当准，后面 12 集全跟着走。

这里只做三件事，都不碰网络、不调模型，可离线测：
  1. **归拢候选**：同一个角色名下散落在参考图包 / 资产库 / 产物目录里的图
  2. **定基准**：谁有资格当"那张脸"——用户亲自钉的 > 参考图包在用的 > 最新的
  3. **分组与裁决**：拿视觉模型给出的两两判定，分出"同一人"和"另一张脸"，
     并判断这次能不能自动定（能就清理，不能就挂起问人）

调模型、搬文件、改参考图包在 functions/drama.py。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 候选的来源优先级：数字越小越有资格当基准。
# user 是用户自己钉进来的（drama_use_local_ref），他已经表过态，不该被自动换掉。
SOURCE_RANK = {"user": 0, "pack": 1, "asset": 2}
# 同一来源里再按类别分先后：**主形象才是那张脸的准绳**，服装图哪怕更新也不能当基准
# （它自己就是照着主形象生的）。"" 是散落产物，摸不清是什么，排中间。
KIND_RANK = {"角色": 0, "": 1, "服装": 2}


@dataclass
class FaceCandidate:
    """一张"疑似这个角色"的图。"""

    character: str
    asset_id: str
    summary: str = ""
    url: str = ""
    local: str = ""  # 本地副本路径，没有就空
    source: str = "asset"  # user（用户钉的）/ pack（参考图包在用）/ asset（散落的产物）
    pack_key: str = ""  # 在参考图包里的键，没有就空
    kind: str = ""  # 角色 / 服装 / ...
    seq: int = 0  # 资产序号，越大越新

    @property
    def rank(self) -> tuple[int, int, int]:
        """排序键：来源 > 类别（主形象优先）> 新旧（新的在前）。"""
        return (SOURCE_RANK.get(self.source, 9), KIND_RANK.get(self.kind, 1), -self.seq)

    def label(self) -> str:
        names = {"user": "用户钉的", "pack": "参考图包", "asset": "产物"}
        where = names.get(self.source, self.source)
        return f"{self.asset_id}（{where}{'·' + self.kind if self.kind else ''}）"


@dataclass
class FaceAudit:
    """一个角色的审查结论。"""

    character: str
    keeper: FaceCandidate | None = None
    same: list[FaceCandidate] = field(default_factory=list)  # 和基准同一个人
    drift: list[FaceCandidate] = field(default_factory=list)  # 另一张脸
    scores: dict[str, int] = field(default_factory=dict)  # asset_id → 一致性得分
    issues: dict[str, list[str]] = field(default_factory=dict)  # asset_id → 差异描述
    ambiguous: bool = False  # 定不了，要问人
    why: str = ""  # 基准是怎么定的 / 为什么定不了
    skipped: str = ""  # 没法比对时的说明（没本地副本、没配视觉模型……）

    @property
    def conflicted(self) -> bool:
        return bool(self.drift)


def pick_baseline(cands: list[FaceCandidate]) -> tuple[FaceCandidate | None, str]:
    """选基准：用户钉的 > 参考图包在用的 > 最新的产物。返回 (基准, 理由)。"""
    if not cands:
        return None, "没有候选"
    best = sorted(cands, key=lambda c: c.rank)[0]
    why = {
        "user": "你自己钉过的那张（drama_use_local_ref）",
        "pack": "参考图包里正在用的主形象",
        "asset": "产物里最新的一张",
    }.get(best.source, "第一张")
    return best, why


def decide(
    character: str,
    cands: list[FaceCandidate],
    verdicts: dict[str, tuple[bool, int, list[str]]],
    baseline: FaceCandidate,
    why: str,
) -> FaceAudit:
    """按视觉模型的判定分组，并决定这次能不能自动定。

    verdicts: asset_id → (是否同一人, 分数, 差异描述)。基准自己不在里面。

    自动定的条件（用户 2026-09-22 定：Agent 自动挑，歧义才问）：
      · 基准是用户钉的或参考图包在用的 —— 他已经表过态，不问
      · 基准只是散落的产物，但跟它一致的占多数 —— 多数派说了算，不问
      · 否则（没人表过态、又是各自成群）—— 挂起问人
    """
    a = FaceAudit(character=character, keeper=baseline, why=why)
    for c in cands:
        if c.asset_id == baseline.asset_id:
            continue
        ok, score, issues = verdicts.get(c.asset_id, (True, -1, []))
        a.scores[c.asset_id] = score
        if issues:
            a.issues[c.asset_id] = issues
        (a.same if ok else a.drift).append(c)
    if not a.drift:
        return a
    if baseline.source in ("user", "pack"):
        return a
    # 基准没有权威性：跟它一致的（含它自己）要过半，才认它是多数派
    if len(a.same) + 1 <= len(a.drift):
        a.ambiguous = True
        a.why = (
            f"「{character}」有 {len(a.drift) + 1} 张互不一致的脸，"
            "而且没有一张是你钉过或参考图包在用的 —— 定不了谁是准的"
        )
    return a


def group_candidates(rows: list[Any], names: list[str]) -> dict[str, list[FaceCandidate]]:
    """把候选按角色名归堆，并按基准优先级排好。names 是资产库里的角色名。"""
    out: dict[str, list[FaceCandidate]] = {}
    for c in rows:
        if c.character in names:
            out.setdefault(c.character, []).append(c)
    for name in out:
        out[name].sort(key=lambda c: c.rank)
    return out


def matches_character(text: str, name: str, others: list[str]) -> bool:
    """这段文字（资产摘要 / 文件名）说的是不是这个角色。

    别的角色名更长且包含它时不算 —— 比如「朱锦」和「朱锦娘」同时存在，
    摘要里出现「朱锦娘」不该被算进「朱锦」。
    """
    if not name or name not in (text or ""):
        return False
    for other in others:
        if other != name and name in other and other in text:
            return False
    return True


def audit_lines(audits: list[FaceAudit]) -> list[str]:
    """给人看的审查结果。没有冲突的一行带过，有冲突的展开。"""
    lines: list[str] = []
    for a in audits:
        if a.skipped:
            lines.append(f"  · {a.character}：{a.skipped}")
            continue
        n = len(a.same) + len(a.drift) + 1
        if not a.conflicted:
            lines.append(f"  ✓ {a.character}　{n} 张，同一个人")
            continue
        mark = "⚠" if a.ambiguous else "✂"
        lines.append(f"  {mark} {a.character}　{n} 张里有 {len(a.drift)} 张是另一张脸")
        lines.append(f"      保留：{a.keeper.label() if a.keeper else '—'}（{a.why}）")
        for c in a.drift:
            score = a.scores.get(c.asset_id, -1)
            issues = "；".join(a.issues.get(c.asset_id) or [])[:60]
            tail = f"　{issues}" if issues else ""
            lines.append(f"      弃用：{c.label()}　{score} 分{tail}")
    return lines


def conflict_question(audits: list[FaceAudit]) -> str:
    """歧义时问用户的话。只问定不了的那些。"""
    bad = [a for a in audits if a.ambiguous]
    parts = ["这些角色有多张互不一致的脸，定不了保留哪张，请你点一张："]
    for a in bad:
        parts.append(f"\n【{a.character}】{a.why}")
        options = ([a.keeper] if a.keeper else []) + a.drift
        for i, c in enumerate(options, 1):
            parts.append(f"  {i}) {c.label()}　{c.summary[:40]}")
    parts.append("\n回复要保留的那张的资产 id（多个角色就逐个写，如「阿蛛 as_xxx」）。")
    return "\n".join(parts)

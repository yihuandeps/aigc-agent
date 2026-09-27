"""存量资产分项目（2026-09-23 审查缺口 A 的迁移）。

加了项目维度之后，新资产都带项目键；之前的 1100 多份没有 —— 它们哪个项目都看得见，
新剧照样会看到旧剧的剧本和资产库。这里把它们分到各自的项目。

**按整条链判**（2026-09-26 用户定的规则）：血缘（剧本 → 分镜 / 资产库 → 参考图包 → 提示词 →
片段，含片段标签里的 shots_id / 包、参考图包和渲染索引里列的资产）连成一条链，整条链归同一个
项目。链的归属看链上所有资产的证据，按硬度分四级，最硬的那一级说了算：

  1. state   产物目录里 `.drama-state.json` 的结构化 id（assetsId / refPackId / scriptIds …），
             以及链上已经带项目键的资产（新代码写的）
  2. local   gen_params.local / 本地 uri 落在某个产物目录的 texts/ images/ videos/ … 下
  3. anchor  内容里提到某个项目的剧名 / 角色名（只认唯一命中一个项目的）
  4. mirror  文本镜像：产物目录 texts/ 里有 `<摘要>-<资产id>.md` —— 最后才参考
都追溯不到的 → legacy（不属于任何项目，按项目查询看不见，按 id 仍取得到）。
**最硬的那一级同时指向两个项目 → 不自动判，标「待你定」**（弱一级的反证据写进依据，不算冲突）。

之前是逐份判、镜像排在血缘前面：《姐姐们抢着给我当妈》的剧本镜像在 E:\\内容测试\\texts，
剧本、资产库、参考图包全被判给了「内容测试」，而 E:\\西游记\\.drama-state.json 里清清楚楚
列着它们的 id。

**先出映射表、人改完再执行**：plan 之后 write_plan_table 写一张可编辑的 CSV（每份资产一行：
建议项目、依据、冲突），人在表里改项目（填产物目录路径 / legacy / 测试桩），plan_from_table
按改过的表执行；「待你定」没改的行不动（不写项目键，下次迁移重新进表）。

顺带三件事：
  · 测试桩（测试进程写进真实库的「创作方案 / 角色档案 / 分集目录」短桩等）移进回收站
  · 没有项目的记忆：内容里提到的资产 id、剧名、角色名能指认出唯一一个项目的，归到那个项目；
    指认不出的保持全局
  · 会话快照：老的 default 快照记着产物目录，复制一份成那个目录的项目会话（在那个文件夹
    启动就接着聊），原文件不动

**全部可撤销**：改动前的值记进 workspace/trash/migrate-<时间>/manifest.json，restore 按它恢复。
必须在没有 Agent 进程运行时做 —— 老进程内存里的资产没有项目键，它一回写就把迁移结果盖掉。
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..drama.card import read_state
from ..project import normalize_root, project_key, project_title
from .store import LEGACY_PROJECT, _atomic_write

# 产物目录下的分类子目录（OutputPrefs.dir_for 的 kind，外加各链路自己建的）
KIND_DIRS = {
    "texts", "images", "videos", "downloads", "exports", "posters", "audio", "clips",
    "subtitles", "refs", "shots",
}
_ID_RE = re.compile(r"as_[0-9a-f]{10}")
_MIRROR_RE = re.compile(r"-(as_[0-9a-f]{10})\.md$")

# 测试桩：测试进程在 gap E 修好之前写进真实库的固定内容（tests/test_p6_drama_tools.py 的 _seed、
# _episode，以及 stub 生成器）。按原文认，不按「短」认 —— 用户自己存的短文档不能误伤
_STUB_OUTLINE_BODIES = (
    "三幕结构，7 个付费卡点",
    "陆离：守门人，沉默。苏晏：记者。",
    "第1集：隧道里的呼吸声 —— 主角发现异象",
)
_STUB_SCRIPT = re.compile(r"# 第\d+集：\S{1,6}\n\n正文…\n\n> 🎣 本集钩子：第\d+集的悬念")


def is_test_stub(a: dict[str, Any]) -> bool:
    body = str(a.get("inline") or "")
    creator = str(a.get("creator") or "")
    if creator == "stub":
        return True
    if a.get("type") == "outline" and not creator and len(body) < 120:
        return body.startswith(_STUB_OUTLINE_BODIES)
    if a.get("type") == "script" and creator in ("", "model") and len(body) < 80:
        return bool(_STUB_SCRIPT.fullmatch(body.strip()))
    return False


@dataclass
class MigrationPlan:
    assign: dict[str, str] = field(default_factory=dict)  # 资产 id → 项目键（或 legacy）
    # 资产 id → state / local / anchor / mirror（自己有这一级的证据）/ lineage（跟着链走）/ legacy
    how: dict[str, str] = field(default_factory=dict)
    roots: dict[str, str] = field(default_factory=dict)  # 项目键 → 产物目录
    stubs: list[str] = field(default_factory=list)  # 测试桩 → 回收站
    memories: dict[str, str] = field(default_factory=dict)  # 记忆 id → 项目键
    snapshots: list[tuple[str, str]] = field(default_factory=list)  # (源快照, 新快照) 文件名
    seq_max: int = 0
    already: int = 0  # 已经带项目键的（新代码写的），不动
    # 待你定：链上最硬的证据同时指向两个以上项目。资产 id → 候选项目键（不进 assign，不动它）
    pending: dict[str, list[str]] = field(default_factory=dict)
    why: dict[str, str] = field(default_factory=dict)  # 资产 id → 依据（给人看，进映射表）
    chain: dict[str, str] = field(default_factory=dict)  # 资产 id → 链编号（映射表里按链筛）
    skipped: list[str] = field(default_factory=list)  # 按映射表执行时：表里没有 / 已不在库里的
    _names: dict[str, str] = field(default_factory=dict, repr=False)

    def counts(self) -> dict[str, Counter[str]]:
        out: dict[str, Counter[str]] = {}
        for aid, key in self.assign.items():
            out.setdefault(key, Counter())[self.how.get(aid, "?")] += 1
        return out

    def label(self, key: str) -> str:
        """项目给人看的名字：剧名（.drama-state）或文件夹名；认不出目录的用键。"""
        if key not in self._names:
            root = self.roots.get(key, "")
            self._names[key] = project_title(root) if root else key
        return self._names[key]

    def render(self) -> str:
        lines = ["资产分项目计划："]
        for key, c in sorted(self.counts().items(), key=lambda kv: -sum(kv[1].values())):
            root = self.roots.get(key, "")
            name = project_title(root) if root else "（追溯不到出处）"
            detail = "、".join(f"{how} {n}" for how, n in c.most_common())
            lines.append(f"  {name} [{key}] {sum(c.values())} 份（{detail}）")
            if root:
                lines.append(f"      {root}")
        if self.pending:
            by = Counter(
                " / ".join(self.label(k) for k in sorted(c)) for c in self.pending.values()
            )
            lines.append(
                f"待你定（链上的证据同时指向几个项目，不自动判）：{len(self.pending)} 份 —— "
                + "；".join(f"{who} {n} 份" for who, n in by.most_common(5))
            )
        lines.append(f"已带项目键、不动：{self.already} 份")
        lines.append(f"测试桩移进回收站：{len(self.stubs)} 份")
        if self.skipped:
            lines.append(f"映射表里没有 / 已不在库里、没动：{len(self.skipped)} 份")
        if self.memories:
            by = Counter(self.memories.values())
            lines.append(
                "记忆归项目：" + "、".join(f"{k} {n} 条" for k, n in by.most_common())
                + "（其余保持全局）"
            )
        for src, dst in self.snapshots:
            lines.append(f"会话快照 {src} 复制成 {dst}（在那个文件夹启动就接着上次聊）")
        return "\n".join(lines)


# ---------------------------------------------------------------- 读


def _read_json(p: Path) -> dict[str, Any] | None:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _assets(workspace: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for f in sorted((workspace / "assets").glob("as_*.json")):
        d = _read_json(f)
        if d and d.get("id"):
            out[str(d["id"])] = d
    return out


def _snapshot_roots(workspace: Path) -> dict[str, str]:
    """会话快照文件名 → 它记着的产物目录。"""
    out: dict[str, str] = {}
    for f in sorted((workspace / "memory" / "sessions").glob("*.json")):
        d = _read_json(f) or {}
        if d.get("output_dir"):
            out[f.name] = str(d["output_dir"])
    return out


def _local_paths(a: dict[str, Any]) -> list[str]:
    gp = a.get("gen_params") or {}
    paths = [str(gp.get("local") or "")]
    uri = str(a.get("uri") or "")
    if uri and not uri.startswith(("http://", "https://", "asset://", "data:")):
        paths.append(uri)
    return [p for p in paths if p]


def _root_of(path: str, known: list[str], workspace: Path) -> str:
    """文件属于哪个产物目录：在已知产物目录下 → 它（最长前缀）；否则往上找分类子目录
    （images/ videos/ texts/ …）的上一级。Agent 自己的工作区（blobs、配音临时目录…）不算。"""
    norm = normalize_root(path)
    best, best_len = "", -1
    for r in known:
        nr = normalize_root(r)
        if (norm == nr or norm.startswith(nr + os.sep)) and len(nr) > best_len:
            best, best_len = r, len(nr)
    if best:
        return best
    p = Path(path)
    for parent in p.parents:
        if parent.name.lower() in KIND_DIRS:
            root = parent.parent
            ws = normalize_root(workspace)
            nr = normalize_root(root)
            inside_ws = nr == ws or nr.startswith(ws + os.sep)
            if inside_ws and not nr.startswith(normalize_root(workspace / "output") + os.sep):
                return ""
            return str(root)
    return ""


_TITLE = re.compile(r"《([^》\n]{2,20})》")
# 角色档案里的人物小标题：「### 陆离（男主，28岁）」「### 大姐·朱锦（面妈）」
_HEADING = re.compile(r"^#{2,4}\s*(?:[^\s（(#]*·)?([一-鿿]{2,4})\s*[（(]", re.M)
_GENERIC_DIRS = {
    "output", "default", "texts", "images", "videos", "downloads", "exports", "新建文件夹",
}


def _anchors(root: str, assets: dict[str, dict[str, Any]], ids: list[str]) -> set[str]:
    """能指认这个项目的词：剧名（状态文件、文档里的《》）、角色名（资产库、角色档案的
    人物小标题）、产物目录的路径和不太泛的目录名。一个文件夹里做过几部剧的，几部剧的都算。"""
    words: set[str] = {str(root)}
    title = project_title(root)
    if title and title.lower() not in _GENERIC_DIRS:
        words.add(title)
    for i in ids:
        a = assets[i]
        body = str(a.get("inline") or "")
        summary = str(a.get("summary") or "")
        if a.get("creator") == "tool:drama_assets" and body:
            try:
                data = json.loads(body)
            except ValueError:
                continue
            for c in (data.get("characters") or []) if isinstance(data, dict) else []:
                name = str((c or {}).get("baseRoleName") or (c or {}).get("name") or "").strip()
                words.add(name)
        elif a.get("type") == "outline" and any(
            k in summary for k in ("角色档案", "创作方案", "分集目录")
        ):
            words |= set(_TITLE.findall(body[:400]))
            if "角色档案" in summary:
                words |= set(_HEADING.findall(body))
    return {w for w in words if len(w) >= 2}


# ---------------------------------------------------------------- 计划


_TIERS = ("state", "local", "anchor", "mirror")  # 越靠前越硬
_TIER_LABEL = {
    "state": ".drama-state / 已有项目键",
    "local": "本地文件",
    "anchor": "剧名 / 角色名",
    "mirror": "文本镜像",
}
# 这些资产的正文里列着别的资产 id，也算血缘：参考图包 → 各张参考图；渲染索引 → 各段片段；
# 短视频出片记录 → 各镜头
_LISTS_IDS = ("tool:drama_render_assets", "tool:drama_render_shots", "tool:short_video_produce")
_ANCHOR_TEXT = 3000  # 认剧名 / 角色名时每份资产看多少字（摘要 + 正文开头）


def _links(a: dict[str, Any]) -> set[str]:
    """一份资产在血缘上直接连着哪些资产：父资产、片段标签里的 shots_id / 包、正文里列的资产。"""
    out = {str(p) for p in a.get("parent_ids") or []}
    tags = (a.get("gen_params") or {}).get("tags")
    if isinstance(tags, dict):
        out |= {str(tags[k]) for k in ("shots_id", "pack") if tags.get(k)}
    if a.get("creator") in _LISTS_IDS:
        out |= set(_ID_RE.findall(str(a.get("inline") or "")))
    return out


def _state_ids(root: str) -> dict[str, str]:
    """产物目录的 .drama-state.json 里结构化记着的资产 id → 哪个字段。

    只认结构化字段（名字以 Id / Ids 结尾的：assetsId、refPackId、scriptIds …），不认备注
    里的自由文本 —— 备注里常有「作废资产勿用：as_xxx（别的剧分镜）」这种反着说的。"""
    out: dict[str, str] = {}
    for k, v in read_state(Path(root)).items():
        if not str(k).endswith(("Id", "Ids")):
            continue
        vals = list(v.values()) if isinstance(v, dict) else v if isinstance(v, list) else [v]
        for x in vals:
            if isinstance(x, str) and _ID_RE.fullmatch(x.strip()):
                out[x.strip()] = str(k)
    return out


def _chains(ids: set[str], assets: dict[str, dict[str, Any]]) -> list[list[str]]:
    """按血缘连成链（连通分量），大的在前；链内按 seq 排。"""
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for aid in ids:
        for other in _links(assets[aid]):
            if other in parent:
                ra, rb = find(aid), find(other)
                if ra != rb:
                    parent[ra] = rb
    groups: dict[str, list[str]] = {}
    for aid in ids:
        groups.setdefault(find(aid), []).append(aid)
    seq = lambda i: (int(assets[i].get("seq") or 0), i)  # noqa: E731
    return sorted(
        (sorted(g, key=seq) for g in groups.values()), key=lambda g: (-len(g), seq(g[0]))
    )


def plan_migration(workspace: Path) -> MigrationPlan:
    plan = MigrationPlan()
    assets = _assets(workspace)
    snap_roots = _snapshot_roots(workspace)
    known = sorted(set(snap_roots.values()))

    # 测试桩先挑出来：不分项目，直接进回收站；也不参与连链（它们挂在真资产的血缘上会把
    # 不相干的剧连成一条）
    plan.stubs = sorted(aid for aid, a in assets.items() if is_test_stub(a))
    stubs = set(plan.stubs)
    todo = {aid: a for aid, a in assets.items() if aid not in stubs}
    plan.seq_max = max((int(a.get("seq") or 0) for a in assets.values()), default=0)
    plan.already = sum(1 for a in todo.values() if a.get("project"))

    # ---- 每份资产自己的证据：(级别, 项目键, 说明) ----
    ev: dict[str, list[tuple[str, str, str]]] = {}

    def add(aid: str, tier: str, key: str, why: str, root: str = "") -> None:
        ev.setdefault(aid, []).append((tier, key, why))
        if root:
            plan.roots.setdefault(key, root)

    for root in known:
        plan.roots.setdefault(project_key(root), root)
    for aid, a in todo.items():
        key = str(a.get("project") or "")
        if key and key != LEGACY_PROJECT:
            add(aid, "state", key, "已有项目键")
            continue
        for path in _local_paths(a):
            root = _root_of(path, known, workspace)
            if root:
                add(aid, "local", project_key(root), f"本地文件在 {root}", root)
                break
    candidates = sorted(set(known) | set(plan.roots.values()))
    for root in candidates:
        key = project_key(root)
        for aid, fld in _state_ids(root).items():
            if aid in todo:
                add(aid, "state", key, f"{root}\\.drama-state.json 的 {fld}", root)
        texts = Path(root) / "texts"
        if texts.is_dir():
            for md in texts.glob("*.md"):
                m = _MIRROR_RE.search(md.name)
                if m and m.group(1) in todo:
                    add(m.group(1), "mirror", key, f"镜像在 {texts}", root)

    # ---- 连链，按最硬的一级判 ----
    chains = _chains(set(todo), todo)
    decided: dict[int, tuple[str, list[str]]] = {}  # 链序号 → (级别, 项目键们)

    def tally(chain: list[str], tier: str) -> Counter[str]:
        return Counter(k for i in chain for t, k, _ in ev.get(i, []) if t == tier)

    # 剧名 / 角色名的词表：用各链暂定的项目（state > local > 镜像，唯一的才算）攒
    tentative: dict[str, list[str]] = {}
    for chain in chains:
        for tier in ("state", "local", "mirror"):
            keys = sorted(tally(chain, tier))
            if keys:
                if len(keys) == 1:
                    tentative.setdefault(keys[0], []).extend(chain)
                break
    words = {
        key: _anchors(plan.roots[key], todo, ids)
        for key, ids in tentative.items() if plan.roots.get(key)
    }
    # 每份资产的内容命中哪个项目的剧名 / 角色名（同时命中几个项目的不算）
    for chain in chains:
        for i in chain:
            if todo[i].get("project"):
                continue
            a = todo[i]
            text = f"{a.get('summary') or ''} {str(a.get('inline') or '')[:_ANCHOR_TEXT]}"
            got = {k: next(w for w in ws if w in text) for k, ws in words.items()
                   if any(w in text for w in ws)}
            if len(got) == 1:
                k, w = next(iter(got.items()))
                add(i, "anchor", k, f"内容里提到「{w}」")

    for n, chain in enumerate(chains):
        state, local = tally(chain, "state"), tally(chain, "local")
        anchor, mirror = tally(chain, "anchor"), tally(chain, "mirror")
        if state:
            decided[n] = ("state", sorted(state))
        elif local:
            keys = sorted(local)
            # 文件放在哪和内容讲的是哪部剧对不上（之前的产物目录指错了地方，一部剧的图落进
            # 了另一个文件夹）：两个项目都碰到了，不自动判
            other = sorted(set(anchor) - set(keys))
            decided[n] = ("local", sorted(set(keys) | set(other)) if other else keys)
        elif anchor:
            decided[n] = ("anchor", sorted(anchor))
        elif mirror:
            decided[n] = ("mirror", sorted(mirror))

    # ---- 落到每份资产 ----
    for n, chain in enumerate(chains):
        cid = f"C{n + 1:03d}"
        tier, keys = decided.get(n, ("", []))

        def evidence(tiers: tuple[str, ...], chain: list[str] = chain) -> list[str]:
            return [
                f"{_TIER_LABEL[t]} → "
                + "、".join(f"{plan.label(k)}×{v}" for k, v in c.most_common(3))
                for t in tiers if (c := tally(chain, t))
            ]

        weaker = evidence(tuple(t for t in _TIERS if t != tier))
        tail = f"；另有 {'；'.join(weaker)}" if weaker else ""
        for aid in chain:
            if todo[aid].get("project"):
                continue
            plan.chain[aid] = cid
            if not keys:
                plan.assign[aid] = LEGACY_PROJECT
                plan.how[aid] = "legacy"
                plan.why[aid] = f"链 {cid}（{len(chain)} 份）：追溯不到出处"
            elif len(keys) > 1:
                plan.pending[aid] = keys
                plan.how[aid] = "conflict"
                plan.why[aid] = (
                    f"链 {cid}（{len(chain)} 份）：证据指向不止一个项目 —— "
                    + "；".join(evidence(_TIERS))
                )
            else:
                own = [w for t, k, w in ev.get(aid, []) if t == tier and k == keys[0]]
                plan.assign[aid] = keys[0]
                plan.how[aid] = tier if own else "lineage"
                plan.why[aid] = (
                    f"链 {cid}（{len(chain)} 份）：{_TIER_LABEL[tier]} → {plan.label(keys[0])}"
                    + (f"（这份：{own[0]}）" if own else "（跟着链走）") + tail
                )

    _plan_memories(plan, workspace, assets)

    # 会话快照：记着产物目录的，复制一份成那个目录的项目会话
    sessions = workspace / "memory" / "sessions"
    for name, root in snap_roots.items():
        dst = f"{project_key(root)}.json"
        if dst != name and not (sessions / dst).exists():
            plan.snapshots.append((name, dst))
            plan.roots.setdefault(project_key(root), root)
    return plan


def _plan_memories(plan: MigrationPlan, workspace: Path, assets: dict[str, dict[str, Any]]) -> None:
    """记忆：没有项目的，按提到的资产 id / 剧名 / 角色名指认（按最终的分配算）。"""
    plan.memories = {}
    by_project: dict[str, list[str]] = {}
    for aid, key in plan.assign.items():
        by_project.setdefault(key, []).append(aid)
    anchors = {
        key: _anchors(root, assets, by_project.get(key, []))
        for key, root in plan.roots.items()
    }
    for f in sorted((workspace / "memory").glob("mem_*.json")):
        m = _read_json(f)
        if not m or m.get("project_id") or m.get("layer") == "account":
            continue
        text = f"{m.get('content') or ''} {m.get('origin_ref') or ''}"
        votes = Counter(
            k for i in _ID_RE.findall(text)
            if (k := plan.assign.get(i, str((assets.get(i) or {}).get("project") or "")))
            and k != LEGACY_PROJECT
        )
        if not votes:
            hits = [k for k, words in anchors.items() if any(w in text for w in words)]
            if len(hits) == 1:
                votes[hits[0]] += 1
        if votes:
            plan.memories[str(m["id"])] = votes.most_common(1)[0][0]


# ---------------------------------------------------------------- 映射表（先出表、人改完再执行）

TABLE_GLOB = "migrate-plan-*.csv"
_COLS = ("资产id", "类型", "摘要", "链", "项目（改这一列）", "剧名", "依据", "冲突")
PENDING = "待你定"
STUB_CELL = "测试桩（移进回收站）"
_LEGACY_CELLS = ("legacy", "不属于任何项目", "无")
_KEY_RE = re.compile(r"^.+-[0-9a-f]{8}$")


def _cell(plan: MigrationPlan, key: str) -> str:
    """项目键 → 表里「项目」那一格：认得出目录的写目录路径（人好认、好改），否则写键。"""
    if key == LEGACY_PROJECT:
        return "legacy"
    return plan.roots.get(key) or key


def write_plan_table(plan: MigrationPlan, workspace: Path, path: Path | None = None) -> Path:
    """把计划写成一张可编辑的映射表（UTF-8 带 BOM 的 CSV，Excel / WPS 直接打开不乱码）。

    一份资产一行。人改「项目（改这一列）」：填产物目录路径（如 E:\\西游记）、项目键、剧名，
    或 legacy（不属于任何项目）/ 测试桩（移进回收站）。「待你定」那几行不改就不动它们。
    排序：待你定在最前，其余按项目、按链、按先后。"""
    assets = _assets(workspace)
    path = path or workspace / f"migrate-plan-{time.strftime('%Y%m%d-%H%M%S')}.csv"
    size = Counter(plan.assign.values())
    rows: list[tuple[tuple[Any, ...], list[str]]] = []

    def info(aid: str) -> tuple[str, str, int]:
        a = assets.get(aid) or {}
        summary = " ".join(str(a.get("summary") or "").split())[:60]
        return str(a.get("type") or ""), summary, int(a.get("seq") or 0)

    for aid, cands in plan.pending.items():
        t, s, seq = info(aid)
        cell = f"{PENDING}：" + " | ".join(_cell(plan, k) for k in cands)
        rows.append(((0, plan.chain.get(aid, ""), seq), [
            aid, t, s, plan.chain.get(aid, ""), cell, "", plan.why.get(aid, ""), PENDING,
        ]))
    for aid, key in plan.assign.items():
        t, s, seq = info(aid)
        legacy = key == LEGACY_PROJECT
        order = (2 if legacy else 1, -size[key], key, plan.chain.get(aid, ""), seq)
        rows.append((order, [
            aid, t, s, plan.chain.get(aid, ""), _cell(plan, key),
            "" if legacy else plan.label(key), plan.why.get(aid, ""), "",
        ]))
    for aid in plan.stubs:
        t, s, seq = info(aid)
        rows.append(((3, "", seq), [aid, t, s, "", STUB_CELL, "", "测试进程写进真实库的桩", ""]))
    rows.sort(key=lambda r: r[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(_COLS)
        w.writerows(r for _, r in rows)
    return path


def latest_plan_table(workspace: Path) -> Path | None:
    tables = sorted(workspace.glob(TABLE_GLOB), key=lambda p: p.stat().st_mtime)
    return tables[-1] if tables else None


def read_plan_table(path: Path) -> dict[str, str]:
    """映射表 → {资产 id: 「项目」那一格}。认 UTF-8（带不带 BOM）和 GBK（Excel 另存常见）。"""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "gbk"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"映射表 {path} 的编码认不出（要 UTF-8 或 GBK）")
    reader = csv.reader(io.StringIO(text))
    head = next(reader, [])
    try:
        id_col = next(i for i, h in enumerate(head) if h.strip().lower().startswith("资产id"))
        proj_col = next(i for i, h in enumerate(head) if h.strip().startswith("项目"))
    except StopIteration:
        raise ValueError(f"映射表 {path} 的表头不对：要有「资产id」和「项目」两列") from None
    out: dict[str, str] = {}
    for row in reader:
        if len(row) > max(id_col, proj_col) and _ID_RE.fullmatch(row[id_col].strip()):
            out[row[id_col].strip()] = row[proj_col].strip()
    return out


def plan_from_table(workspace: Path, table: Path) -> MigrationPlan:
    """按人改过的映射表出执行计划。表里「待你定」没改的、填得认不出的、表里没有的（表写出之后
    新出现的）都不动，列在 pending / skipped 里。"""
    base = plan_migration(workspace)
    rows = read_plan_table(table)
    assets = _assets(workspace)
    by_label = Counter(base.label(k) for k in base.roots)
    labels = {base.label(k): k for k in base.roots if by_label[base.label(k)] == 1}
    known_keys = set(base.roots) | {
        str(a.get("project")) for a in assets.values() if a.get("project")
    }
    plan = MigrationPlan(
        roots=dict(base.roots), snapshots=list(base.snapshots), seq_max=base.seq_max,
        already=base.already, why=dict(base.why), chain=dict(base.chain),
    )
    suggested = {aid: _cell(base, k) for aid, k in base.assign.items()}
    suggested.update({aid: STUB_CELL for aid in base.stubs})
    for aid, a in assets.items():
        if a.get("project"):
            continue
        cell = rows.get(aid)
        if cell is None:
            plan.skipped.append(aid)
            continue
        changed = aid in suggested and cell != suggested[aid]
        if not cell or cell.startswith(PENDING):
            plan.pending[aid] = base.pending.get(aid, [])
            plan.how[aid] = "conflict"
        elif cell.lower() in _LEGACY_CELLS:
            plan.assign[aid] = LEGACY_PROJECT
            plan.how[aid] = "table" if changed else base.how.get(aid, "legacy")
        elif cell.startswith("测试桩"):
            plan.stubs.append(aid)
        else:
            key = cell if cell in known_keys else labels.get(cell, "")
            if not key and (":" in cell or "\\" in cell or "/" in cell):
                key = project_key(cell)
                plan.roots.setdefault(key, cell)
            if not key:
                # 填的东西认不出（不是路径、项目键，也不是认得出的剧名）：不猜，不动
                plan.pending[aid] = base.pending.get(aid, [])
                plan.how[aid] = "unknown"
                plan.why[aid] = f"表里填的「{cell}」认不出（要产物目录路径 / 项目键 / 剧名）"
                continue
            plan.assign[aid] = key
            by_hand = changed or aid in base.pending
            plan.how[aid] = "table" if by_hand else base.how.get(aid, "table")
    plan.stubs.sort()
    _plan_memories(plan, workspace, assets)
    return plan


# ---------------------------------------------------------------- 执行 / 撤销


def agents_running(workspace: Path, within_s: float = 120.0) -> list[str]:
    """有 Agent 在跑的迹象：别的 `agent` 进程（chat / drama / video …），加上最近还在写的
    会话日志（闲着的 chat 不写日志，只看日志会漏）。迁移前要求一个都没有。"""
    out = agent_processes()
    now = time.time()
    for f in (workspace / "logs" / "sessions").glob("*.jsonl"):
        try:
            if now - f.stat().st_mtime < within_s:
                out.append(f"日志 {f.name}")
        except OSError:
            continue
    return out


def agent_processes() -> list[str]:
    """正在跑的 Agent 进程（命令行里有 aigc_agent.interfaces.cli，本进程和它的启动器除外）。

    Windows 的 venv python.exe 是个启动器，会再拉起一个真解释器子进程，所以父进程也要排除。
    查不了（没有 powershell / ps）就返回空 —— 只靠日志那一条兜底。
    """
    import subprocess

    mine = {os.getpid(), os.getppid()}
    if os.name == "nt":
        cmd = [
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process"
            " | Where-Object { $_.CommandLine -like '*aigc_agent.interfaces.cli*' }"
            " | ForEach-Object { \"$($_.ProcessId) $($_.CommandLine)\" }",
        ]
    else:
        cmd = ["ps", "-eo", "pid=,args="]
    try:
        text = subprocess.run(  # noqa: S603 — 固定命令，只读进程表
            cmd, capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace"
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if "aigc_agent.interfaces.cli" not in line or "Get-CimInstance" in line:
            continue  # 查进程的这条 powershell 命令行里也带着这个串
        pid, _, rest = line.partition(" ")
        if pid.isdigit() and int(pid) in mine:
            continue
        rows.append(f"进程 {pid}：{rest[:100]}")
    return rows


def apply_migration(plan: MigrationPlan, workspace: Path) -> Path:
    """按计划改。返回 manifest 路径（restore 用）。"""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    trash = workspace / "trash" / f"migrate-{stamp}"
    (trash / "assets").mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "stamp": stamp,
        "assets": {},
        "stubs": [],
        "memories": {},
        "snapshots": [],
        "seq_before": None,
    }
    adir = workspace / "assets"

    def _save() -> None:
        _atomic_write(trash / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    for aid, key in plan.assign.items():
        f = adir / f"{aid}.json"
        d = _read_json(f)
        if d is None or d.get("project"):
            continue
        manifest["assets"][aid] = d.get("project")
        d["project"] = key
        _atomic_write(f, json.dumps(d, ensure_ascii=False, indent=2))
    _save()

    for aid in plan.stubs:
        src = adir / f"{aid}.json"
        if not src.exists():
            continue
        dst = trash / "assets" / src.name
        shutil.move(str(src), str(dst))
        manifest["stubs"].append({"id": aid, "from": str(src), "to": str(dst)})
    _save()

    for mid, key in plan.memories.items():
        f = workspace / "memory" / f"{mid}.json"
        d = _read_json(f)
        if d is None or d.get("project_id"):
            continue
        manifest["memories"][mid] = d.get("project_id", "")
        d["project_id"] = key
        _atomic_write(f, json.dumps(d, ensure_ascii=False, indent=2))
    _save()

    sessions = workspace / "memory" / "sessions"
    for src, dst in plan.snapshots:
        target = sessions / dst
        if target.exists():
            continue
        shutil.copy2(sessions / src, target)
        manifest["snapshots"].append(str(target))

    counter = adir / ".seq"
    try:
        before = int(counter.read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        before = None
    manifest["seq_before"] = before
    if plan.seq_max > (before or 0):
        _atomic_write(counter, str(plan.seq_max))
    _save()
    return trash / "manifest.json"


def restore_migration(manifest_path: Path) -> dict[str, int]:
    """按 manifest 撤销：项目键改回原样、测试桩搬回、记忆改回、新复制的快照改名留底。"""
    m = _read_json(manifest_path)
    if m is None:
        raise ValueError(f"读不出迁移清单：{manifest_path}")
    adir = manifest_path.parent.parent.parent / "assets"
    done = Counter()
    for aid, before in (m.get("assets") or {}).items():
        f = adir / f"{aid}.json"
        d = _read_json(f)
        if d is None:
            continue
        if before is None:
            d.pop("project", None)
        else:
            d["project"] = before
        _atomic_write(f, json.dumps(d, ensure_ascii=False, indent=2))
        done["assets"] += 1
    for row in m.get("stubs") or []:
        src, dst = Path(row["to"]), Path(row["from"])
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            done["stubs"] += 1
    mem_dir = manifest_path.parent.parent.parent / "memory"
    for mid, before in (m.get("memories") or {}).items():
        f = mem_dir / f"{mid}.json"
        d = _read_json(f)
        if d is None:
            continue
        d["project_id"] = before or ""
        _atomic_write(f, json.dumps(d, ensure_ascii=False, indent=2))
        done["memories"] += 1
    for path in m.get("snapshots") or []:
        p = Path(path)
        if p.exists():
            p.rename(p.with_name(p.name + f".restored-{time.strftime('%Y%m%d-%H%M%S')}"))
            done["snapshots"] += 1
    return dict(done)

"""存量资产分项目（2026-09-23 审查缺口 A 的迁移）。

加了项目维度之后，新资产都带项目键；之前的 1100 多份没有 —— 它们哪个项目都看得见，
新剧照样会看到旧剧的剧本和资产库。这里把它们分到各自的项目：

  1. 本地文件：gen_params.local / 本地 uri 落在某个产物目录的 texts/ images/ videos/ … 下
     （或会话快照记着的产物目录下）→ 那个目录的项目
  2. 文本镜像：产物目录 texts/ 里有 `<摘要>-<资产id>.md` → 那个目录的项目
  3. 血缘：父母 / 子女已经分好的，跟着走（剧本 → 分镜 → 提示词 → 片段）
  4. 都追溯不到的 → legacy（不属于任何项目，按项目查询看不见，按 id 仍取得到）

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

import json
import os
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    how: dict[str, str] = field(default_factory=dict)  # 资产 id → local / mirror / lineage / legacy
    roots: dict[str, str] = field(default_factory=dict)  # 项目键 → 产物目录
    stubs: list[str] = field(default_factory=list)  # 测试桩 → 回收站
    memories: dict[str, str] = field(default_factory=dict)  # 记忆 id → 项目键
    snapshots: list[tuple[str, str]] = field(default_factory=list)  # (源快照, 新快照) 文件名
    seq_max: int = 0
    already: int = 0  # 已经带项目键的（新代码写的），不动

    def counts(self) -> dict[str, Counter[str]]:
        out: dict[str, Counter[str]] = {}
        for aid, key in self.assign.items():
            out.setdefault(key, Counter())[self.how.get(aid, "?")] += 1
        return out

    def render(self) -> str:
        lines = ["资产分项目计划："]
        for key, c in sorted(self.counts().items(), key=lambda kv: -sum(kv[1].values())):
            root = self.roots.get(key, "")
            name = project_title(root) if root else "（追溯不到出处）"
            detail = "、".join(f"{how} {n}" for how, n in c.most_common())
            lines.append(f"  {name} [{key}] {sum(c.values())} 份（{detail}）")
            if root:
                lines.append(f"      {root}")
        lines.append(f"已带项目键、不动：{self.already} 份")
        lines.append(f"测试桩移进回收站：{len(self.stubs)} 份")
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


def plan_migration(workspace: Path) -> MigrationPlan:
    plan = MigrationPlan()
    assets = _assets(workspace)
    snap_roots = _snapshot_roots(workspace)
    known = sorted(set(snap_roots.values()))

    # 测试桩先挑出来：不分项目，直接进回收站
    plan.stubs = sorted(aid for aid, a in assets.items() if is_test_stub(a))
    stubs = set(plan.stubs)
    todo = {aid: a for aid, a in assets.items() if aid not in stubs}
    plan.seq_max = max((int(a.get("seq") or 0) for a in assets.values()), default=0)

    # ① 本地文件
    for aid, a in todo.items():
        if a.get("project"):
            plan.already += 1
            continue
        for path in _local_paths(a):
            root = _root_of(path, known, workspace)
            if root:
                key = project_key(root)
                plan.assign[aid] = key
                plan.how[aid] = "local"
                plan.roots.setdefault(key, root)
                break

    # ② 文本镜像：已知的产物目录 + ① 里认出来的
    for root in sorted(set(known) | set(plan.roots.values())):
        texts = Path(root) / "texts"
        if not texts.is_dir():
            continue
        key = project_key(root)
        for md in texts.glob("*.md"):
            m = _MIRROR_RE.search(md.name)
            if not m:
                continue
            aid = m.group(1)
            if aid in todo and aid not in plan.assign and not todo[aid].get("project"):
                plan.assign[aid] = key
                plan.how[aid] = "mirror"
                plan.roots.setdefault(key, root)

    # ③ 血缘：父母优先，其次子女里最多的那个项目，直到不再变化
    children: dict[str, list[str]] = {}
    for aid, a in todo.items():
        for p in a.get("parent_ids") or []:
            children.setdefault(str(p), []).append(aid)

    def _proj(i: str) -> str:
        if i in plan.assign:
            return plan.assign[i]
        a = todo.get(i)
        return str(a.get("project") or "") if a else ""

    changed = True
    while changed:
        changed = False
        for aid, a in todo.items():
            if aid in plan.assign or a.get("project"):
                continue
            ups = [_proj(str(p)) for p in a.get("parent_ids") or []]
            ups = [k for k in ups if k and k != LEGACY_PROJECT]
            if ups:
                key = ups[0]
            else:
                downs = Counter(
                    k for c in children.get(aid, []) if (k := _proj(c)) and k != LEGACY_PROJECT
                )
                if not downs:
                    continue
                key = downs.most_common(1)[0][0]
            plan.assign[aid] = key
            plan.how[aid] = "lineage"
            changed = True

    # ④ 追溯不到的 → legacy
    for aid, a in todo.items():
        if aid not in plan.assign and not a.get("project"):
            plan.assign[aid] = LEGACY_PROJECT
            plan.how[aid] = "legacy"

    # 记忆：没有项目的，按提到的资产 id / 剧名 / 角色名指认
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

    # 会话快照：记着产物目录的，复制一份成那个目录的项目会话
    sessions = workspace / "memory" / "sessions"
    for name, root in snap_roots.items():
        dst = f"{project_key(root)}.json"
        if dst != name and not (sessions / dst).exists():
            plan.snapshots.append((name, dst))
            plan.roots.setdefault(project_key(root), root)
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

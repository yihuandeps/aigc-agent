"""M7 Skill Hub —— 方法论中枢。

Skill 是**方法论文档，不是可调用能力**：
  · Skill    回答「怎么做好」→ 进上下文
  · Function 回答「能做什么」→ 可调用
一份 skill 可以在正文里引用 function，但它自己不执行任何东西。

**三级披露**（同一个原则再深一层）：

| 级 | 内容 | 成本 | 何时加载 |
|---|---|---|---|
| L1 目录 | name + description | ~50 token/个 | 常驻 |
| L2 正文 | SKILL.md | 2–5K | 被选中时 |
| L3 参考 | references/*.md | 2–4K/篇 | 正文里点名、模型按需拉 |

第三级是给**大型多阶段 skill** 用的：一套完整创作方法论动辄 20K token，
整篇塞进去会把能力预算吃光，但拆成「主文档说流程 + 参考文档说细节」之后，
任一时刻只有当前阶段用得上的那 1–2 篇在上下文里。

Skill 有两种形态：
  · 单文件  `skills/foo.md`
  · 目录    `skills/foo/SKILL.md` + `skills/foo/references/*.md`

**这是非技术同事唯一能直接改的地方**，所以配套三件事（P3）：
  · 热加载   refresh_if_changed()：文件变了就重新解析，改完即生效
  · 版本记录 每个版本按内容哈希存进 history_dir，谁改的看 owner，何时改的看 mtime
  · 一键回滚 rollback()：把上一版内容写回文件
灰度（先在一个账号上试）留到有多账号时再做。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ...harness.model.gateway import estimate_tokens

# 以 _ 开头的不加载（如 _TEMPLATE.md）
_SKIP_PREFIX = "_"
_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.S)

# scope 特异性：冲突时越具体的越优先
_SCOPE_RANK = {"account": 3, "content_type": 2, "pipeline_stage": 1, "global": 0}

# 这个优先级及以上的 skill（合规类）不参与预算降级、永不被挤出
PROTECTED_PRIORITY = 100


@dataclass
class Reference:
    """skill 的参考文档。主文档说明何时该读它。"""

    name: str  # 文件名去掉 .md
    path: Path
    purpose: str = ""  # 从主文档的参考资料表里抽出来的用途说明

    def read(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return ""


@dataclass
class Skill:
    name: str
    description: str
    body: str
    scope: str = "global"
    applies_to: list[str] = field(default_factory=list)
    stage: list[str] = field(default_factory=list)
    priority: int = 50
    version: str = "0.0.0"
    owner: str = ""
    status: str = "draft"
    path: Path | None = None
    references: list[Reference] = field(default_factory=list)
    # 版本记录用：内容哈希 + 文件时间
    sha: str = ""
    mtime: float = 0.0
    size: int = 0

    @property
    def active(self) -> bool:
        return self.status == "active"

    @property
    def protected(self) -> bool:
        """合规类不参与降级。"""
        return self.priority >= PROTECTED_PRIORITY

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.body)

    def digest(self) -> str:
        """常驻上下文的形态。"""
        extra = f"（含 {len(self.references)} 篇参考）" if self.references else ""
        return f"- {self.name}{extra}：{self.description}"

    def reference(self, name: str) -> Reference | None:
        key = name.removesuffix(".md")
        return next((r for r in self.references if r.name == key), None)

    def reference_index(self) -> str:
        """参考文档目录，随主文档一起进上下文（每篇约 15 token）。"""
        if not self.references:
            return ""
        nl = "\n"
        lines = [f"- {r.name}{'：' + r.purpose if r.purpose else ''}" for r in self.references]
        head = (
            "## 本 skill 的参考文档"
            + nl
            + f'用 load_skill_reference(skill="{self.name}", name=...) 按需取，不要一次全拉。'
        )
        return head + nl + nl.join(lines)

    def matches(self, content_type: str = "", stage: str = "") -> bool:
        """规则预筛。模型判断之前先用规则滤掉不相关的，不消耗任何 token。"""
        if content_type and self.applies_to and content_type not in self.applies_to:
            return False
        return not (stage and self.stage and stage not in self.stage)

    @property
    def rank(self) -> tuple[int, int]:
        """冲突排序：scope 特异性 → priority。"""
        return (_SCOPE_RANK.get(self.scope, 0), self.priority)


def parse_skill(path: Path) -> Skill | None:
    """解析一个 skill 文件。格式不对返回 None，不抛异常——单篇写坏不该拖垮全部。"""
    try:
        raw = path.read_text(encoding="utf-8")
        st = path.stat()
    except Exception:  # noqa: BLE001
        return None

    m = _FM.match(raw)
    if not m:
        return None
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(fm, dict) or not fm.get("name"):
        return None

    return Skill(
        name=str(fm["name"]),
        description=str(fm.get("description", "")),
        body=m.group(2),
        scope=str(fm.get("scope", "global")),
        applies_to=[str(x) for x in (fm.get("applies_to") or [])],
        stage=[str(x) for x in (fm.get("stage") or [])],
        priority=int(fm.get("priority", 50)),
        version=str(fm.get("version", "0.0.0")),
        owner=str(fm.get("owner", "")),
        status=str(fm.get("status", "draft")),
        path=path,
        sha=hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12],
        mtime=st.st_mtime,
        size=st.st_size,
    )


def skill_files(skills_dir: Path) -> list[Path]:
    """所有会影响加载结果的文件：单文件 skill、目录 skill 的主文档与参考。"""
    if not skills_dir.exists():
        return []
    out: list[Path] = []
    for f in sorted(skills_dir.glob("*.md")):
        if f.name.startswith(_SKIP_PREFIX) or f.name == "README.md":
            continue
        out.append(f)
    for d in sorted(skills_dir.iterdir()):
        if not d.is_dir() or d.name.startswith(_SKIP_PREFIX):
            continue
        main = d / "SKILL.md"
        if not main.exists():
            continue
        out.append(main)
        ref_dir = d / "references"
        if ref_dir.is_dir():
            out.extend(sorted(ref_dir.glob("*.md")))
    return out


def load_skills(skills_dir: Path) -> list[Skill]:
    """扫描目录。只读不执行。

    两种形态都认：
      skills/foo.md                          单文件
      skills/foo/SKILL.md + foo/references/  目录（大型多阶段 skill）
    """
    if not skills_dir.exists():
        return []
    out: list[Skill] = []

    for f in sorted(skills_dir.glob("*.md")):
        if f.name.startswith(_SKIP_PREFIX) or f.name == "README.md":
            continue
        sk = parse_skill(f)
        if sk is not None:
            out.append(sk)

    for d in sorted(skills_dir.iterdir()):
        if not d.is_dir() or d.name.startswith(_SKIP_PREFIX):
            continue
        main = d / "SKILL.md"
        if not main.exists():
            continue
        sk = parse_skill(main)
        if sk is None:
            continue
        ref_dir = d / "references"
        if ref_dir.is_dir():
            purposes = _parse_reference_purposes(sk.body)
            sk.references = [
                Reference(name=f.stem, path=f, purpose=purposes.get(f.name, ""))
                for f in sorted(ref_dir.glob("*.md"))
            ]
        out.append(sk)

    return out


def _parse_reference_purposes(body: str) -> dict[str, str]:
    """从主文档的参考资料表里抽出每篇的用途。

    主文档通常有一张 `| 文件 | 用途 | 加载时机 |` 的表，
    把用途抽出来放进目录，模型才知道该拉哪篇。
    """
    out: dict[str, str] = {}
    for line in body.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and cells[0].endswith(".md"):
            out[cells[0]] = cells[1]
    return out


class SkillHub:
    def __init__(
        self, skills_dir: Path, max_active: int = 3, history_dir: Path | None = None
    ) -> None:
        self.dir = skills_dir
        self.max_active = max_active
        self.skills: list[Skill] = []
        # 版本记录目录。None = 不记录（测试或只读场景）
        self.history_dir = history_dir
        self._signature: tuple[tuple[str, int, int], ...] = ()

    # ---------- 加载与热加载 ----------

    def load(self) -> None:
        self.skills = load_skills(self.dir)
        self._signature = self._fingerprint()
        self._archive_all()

    def _fingerprint(self) -> tuple[tuple[str, int, int], ...]:
        """目录指纹：文件路径 + mtime + 大小。几十个 stat 调用，每轮跑得起。"""
        out = []
        for f in skill_files(self.dir):
            try:
                st = f.stat()
            except OSError:
                continue
            out.append((str(f), st.st_mtime_ns, st.st_size))
        return tuple(out)

    def refresh_if_changed(self) -> list[str]:
        """热加载：目录指纹变了就重新解析。返回内容变了的 skill 名（含新增/删除）。

        运营改完 markdown 存盘，下一轮就生效 —— 不用重启、不用发版。
        """
        fp = self._fingerprint()
        if fp == self._signature:
            return []
        before = {s.name: s.sha for s in self.skills}
        self.skills = load_skills(self.dir)
        self._signature = fp
        after = {s.name: s.sha for s in self.skills}
        changed = [n for n, sha in after.items() if before.get(n) != sha]
        changed += [n for n in before if n not in after]
        self._archive_all()
        return changed

    # ---------- 版本记录 ----------

    def _archive_all(self) -> None:
        if self.history_dir is None:
            return
        for s in self.skills:
            self._archive(s)

    def _archive(self, s: Skill) -> None:
        """按内容哈希存一份快照。同一内容只存一次，所以反复启动不会膨胀。"""
        if self.history_dir is None or s.path is None or not s.sha:
            return
        d = self.history_dir / s.name
        d.mkdir(parents=True, exist_ok=True)
        snap = d / f"{s.sha}.md"
        if snap.exists():
            return
        try:
            snap.write_text(s.path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:  # noqa: BLE001
            return
        entry = {
            "name": s.name,
            "sha": s.sha,
            "version": s.version,
            "owner": s.owner,
            "status": s.status,
            "mtime": s.mtime,
            "size": s.size,
            "tokens": s.tokens,
            "recorded_at": time.time(),
        }
        with (self.history_dir / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def history(self, name: str) -> list[dict]:
        """某个 skill 的版本记录，按时间先后。"""
        if self.history_dir is None:
            return []
        f = self.history_dir / "history.jsonl"
        if not f.exists():
            return []
        out = []
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("name") == name:
                out.append(e)
        return out

    def rollback(self, name: str, sha: str = "") -> Skill:
        """把某个历史版本写回文件。sha 留空 = 当前版本之前的那一版。"""
        if self.history_dir is None:
            raise RuntimeError("没有配置版本记录目录，无法回滚")
        current = next((s for s in self.skills if s.name == name), None)
        if current is None or current.path is None:
            raise KeyError(f"没有 skill {name!r}")
        entries = self.history(name)
        if not sha:
            older = [e["sha"] for e in entries if e["sha"] != current.sha]
            if not older:
                raise ValueError(f"{name} 没有更早的版本可回滚")
            sha = older[-1]
        snap = self.history_dir / name / f"{sha}.md"
        if not snap.exists():
            raise KeyError(f"{name} 没有版本 {sha}")
        current.path.write_text(snap.read_text(encoding="utf-8"), encoding="utf-8")
        self.refresh_if_changed()
        rolled = next((s for s in self.skills if s.name == name), None)
        if rolled is None:
            raise RuntimeError(f"回滚后 {name} 解析失败")
        return rolled

    # ---------- 查询 ----------

    @property
    def available(self) -> list[Skill]:
        return [s for s in self.skills if s.active]

    def get(self, name: str) -> Skill | None:
        return next((s for s in self.available if s.name == name), None)

    def candidates(self, content_type: str = "", stage: str = "") -> list[Skill]:
        """规则预筛后的候选集。"""
        return [s for s in self.available if s.matches(content_type, stage)]

    def catalog_digest(self, content_type: str = "", stage: str = "", limit: int = 0) -> str:
        """目录。规则先筛、模型再选——两级漏斗的第一级。

        limit>0 时只保留优先级最高的前 limit 个 —— 能力预算超标时的最后一级降级。
        """
        hits = self.candidates(content_type, stage)
        if limit and len(hits) > limit:
            hits = sorted(hits, key=lambda s: s.priority, reverse=True)[:limit]
        return "\n".join(s.digest() for s in hits)

    def select(self, names: list[str]) -> tuple[list[Skill], list[str]]:
        """按名字取正文。返回 (命中, 未知名)。"""
        by_name = {s.name: s for s in self.available}
        found = [by_name[n] for n in names if n in by_name]
        missing = [n for n in names if n not in by_name]
        # 冲突时高特异性/高优先级在后，注入上下文时更靠近末尾 = 更强
        found.sort(key=lambda s: s.rank)
        return found[-self.max_active :], missing

    def render(self, skills: list[Skill]) -> str:
        """注入上下文的形态。**显式标注优先级，不让模型自己猜。**"""
        if not skills:
            return ""
        ordered = sorted(skills, key=lambda s: s.rank)
        blocks: list[str] = []
        for i, s in enumerate(ordered):
            top = i == len(ordered) - 1 and len(ordered) > 1
            note = "（优先级最高，与前面冲突时以它为准）" if top else ""
            blocks.append(f"# Skill：{s.name}{note}\n\n{s.body.strip()}")
        return "\n\n---\n\n".join(blocks)

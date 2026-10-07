"""成本台账 —— 跨会话累计的那一份。

Cost Guard 的单任务口径只活在进程里，进程一退就归零。项目级和日级预算要跨会话
累计，就得落盘。这里是一份 JSONL 追加写的台账：每次模型调用、每次媒体生成记一行，
按天、按项目求和。

不做数据库：单机内部工具，一天几百行，一年也就十几万行，读一遍很快。
真到了要看趋势的时候再挪进 SQLite，接口不变。

文本 provider 没配单价的那几行，cost 是按保守估价算的（estimated=true），照样计入单日 / 单项目
金额；输入、输出 token 分开记，以后补了真实单价能回算（2026-10-07，09-29 审查 2.1）。
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from .task_ledger import append_jsonl


def today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


class LedgerEntry(BaseModel):
    ts: float = Field(default_factory=time.time)
    day: str = Field(default_factory=today)
    project: str = "default"
    session: str = ""
    kind: str = "text"  # text | image | video | audio
    # 元。文本 provider 没配单价时是按保守估价算的（estimated=True），照样计入单日 / 单项目金额；
    # 连估价都没有（媒体目录没填单价）就是 None
    cost: float | None = None
    estimated: bool = False
    calls: int = 1
    tokens: int = 0  # 输入 + 输出
    # 输入（含命中缓存）、输出、其中命中缓存的 token 分开记：以后补了真实单价，估算的那几笔
    # 能按 token 回算（2026-10-07，09-29 审查 2.1）。这之前的行只有 tokens 合计
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    role: str = ""
    provider: str = ""  # 文本：config/models.yaml 的 provider 键（单价按它配）
    model: str = ""
    seconds: float = 0.0  # 生成的视频秒数（退还时为负）


class CostLedger:
    """追加写。内存里只留聚合，不留全部明细 —— 明细在文件里。

    多个进程（两个终端各开一个会话）共用同一份台账：每次查询前把别的进程新追加的
    行读进来（按文件偏移增量读），单日 / 单项目上限才是真的「所有会话合计」。
    之前只在启动时读一次，另一个终端当天花的钱这边看不见（2026-09-23 审查）。
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._reset()
        if path is not None and path.exists():
            self._load()

    def _reset(self) -> None:
        self._money_by_day: dict[str, float] = defaultdict(float)
        self._money_by_project: dict[str, float] = defaultdict(float)
        # 金额里按保守估价算的那部分（已经含在上面两项里，单独记是为了展示时说清）
        self._est_by_day: dict[str, float] = defaultdict(float)
        self._est_by_project: dict[str, float] = defaultdict(float)
        self._calls_by_day: dict[tuple[str, str], int] = defaultdict(int)
        self._calls_by_project: dict[tuple[str, str], int] = defaultdict(int)
        self._seconds_by_day: dict[str, float] = defaultdict(float)
        self._seconds_by_project: dict[str, float] = defaultdict(float)
        self.entries = 0
        self._offset = 0  # 已经读进聚合的字节数

    def _load(self, until: int | None = None) -> None:
        """从上次读到的位置往后读完整的行（最后半行留着，等它写完）。until：只读到这个偏移。"""
        if self.path is None:
            return
        try:
            size = self.path.stat().st_size if until is None else until
        except OSError:
            return
        if size < self._offset:  # 文件被换掉 / 截断：从头重算
            self._reset()
        if size == self._offset:
            return
        try:
            with self.path.open("rb") as f:
                f.seek(self._offset)
                chunk = f.read(size - self._offset)
        except OSError:
            return
        end = chunk.rfind(b"\n")
        if end < 0:
            return
        for line in chunk[:end].decode("utf-8", "replace").splitlines():
            try:
                e = LedgerEntry.model_validate_json(line)
            except Exception:  # noqa: BLE001 — 单行坏数据不该让台账起不来
                continue
            self._absorb(e)
        self._offset += end + 1

    def _absorb(self, e: LedgerEntry) -> None:
        self.entries += 1
        if e.cost is not None:
            self._money_by_day[e.day] += e.cost
            self._money_by_project[e.project] += e.cost
            if e.estimated:
                self._est_by_day[e.day] += e.cost
                self._est_by_project[e.project] += e.cost
        self._calls_by_day[(e.day, e.kind)] += e.calls
        self._calls_by_project[(e.project, e.kind)] += e.calls
        if e.seconds:
            self._seconds_by_day[e.day] += e.seconds
            self._seconds_by_project[e.project] += e.seconds

    def append(self, e: LedgerEntry) -> LedgerEntry:
        self._load()  # 先把别的进程追加的读进来，再记自己这行
        self._absorb(e)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                start, n = append_jsonl(self.path, e.model_dump_json())
                if start > self._offset:
                    # 别的进程在我们 _load() 之后、写入之前追加的行：读进来，别跳过
                    # （之前直接把偏移设成文件大小，这些行本进程再也不读，合计偏小）
                    self._load(until=start)
                self._offset = start + n  # 自己这行已经在聚合里了
            except OSError:
                pass
        return e

    # ---------- 查询 ----------

    def money(self, day: str = "", project: str = "") -> float:
        """花了多少（元），含按估价算的部分 —— 单日 / 单项目上限要看得见没配单价的模型。"""
        self._load()
        if day:
            return self._money_by_day.get(day, 0.0)
        if project:
            return self._money_by_project.get(project, 0.0)
        return sum(self._money_by_day.values())

    def estimated(self, day: str = "", project: str = "") -> float:
        """money() 里按保守估价算的那部分（元）。展示金额时说清楚多少是估算。"""
        self._load()
        if day:
            return self._est_by_day.get(day, 0.0)
        if project:
            return self._est_by_project.get(project, 0.0)
        return sum(self._est_by_day.values())

    def calls(self, kind: str, day: str = "", project: str = "") -> int:
        self._load()
        if day:
            return self._calls_by_day.get((day, kind), 0)
        if project:
            return self._calls_by_project.get((project, kind), 0)
        return sum(n for (_, k), n in self._calls_by_day.items() if k == kind)

    def seconds(self, day: str = "", project: str = "") -> float:
        """生成的视频秒数合计（按天 / 按项目）。"""
        self._load()
        if day:
            return self._seconds_by_day.get(day, 0.0)
        if project:
            return self._seconds_by_project.get(project, 0.0)
        return sum(self._seconds_by_day.values())

    def today_money(self) -> float:
        return self.money(day=today())

    def summary(self, project: str = "") -> dict[str, object]:
        d = today()
        kinds = sorted({k for (_, k) in self._calls_by_day})
        return {
            "today": d,
            "today_money": round(self.money(day=d), 4),
            "today_estimated": round(self.estimated(day=d), 4),  # 已含在 today_money 里
            "today_calls": {k: self.calls(k, day=d) for k in kinds if self.calls(k, day=d)},
            "project": project,
            "project_money": round(self.money(project=project), 4) if project else 0.0,
            "project_estimated": round(self.estimated(project=project), 4) if project else 0.0,
            "entries": self.entries,
        }

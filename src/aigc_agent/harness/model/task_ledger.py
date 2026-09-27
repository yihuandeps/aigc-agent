"""媒体任务台账 —— 提交成功就记下 task_id，没拿到结果的任务事后还能取回。

2026-09-23 审查：视频生成动辄几分钟，期间 /stop、工具超时、轮询放弃、网关偶发 5xx，
任何一样都会让本地放弃等待 —— 服务端的任务却照跑、照扣费。之前 task_id 只活在内存里，
放弃就等于这笔钱白花；上层还会把「轮询失败」当成网络抖动**重新提交**，一个镜头付两份。

台账是 JSONL，一行记一次状态变化，读的时候按 task_id 折叠（后写的覆盖先写的）。
网关在提交之前先查：同一份请求（指纹相同）还有没交付的任务，就去取回它，不重新付费。

「没交付」= 提交后没拿到结果（submitted / running / timeout / abandoned / poll_failed），
或者拿到了结果但进程在登记资产前就没了（succeeded 且 delivered=False）。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

UNDELIVERED = {"submitted", "running", "timeout", "abandoned", "poll_failed", "succeeded"}


@dataclass
class TaskRecord:
    task_id: str
    kind: str = ""
    model: str = ""
    provider: str = ""
    fingerprint: str = ""
    status: str = "submitted"
    submitted_at: float = 0.0
    updated_at: float = 0.0
    prompt: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    urls: list[str] = field(default_factory=list)
    error: str = ""
    delivered: bool = False
    recoveries: int = 0  # 自动取回过几次（取回后又没等到的，不再无限取回）

    @property
    def undelivered(self) -> bool:
        return not self.delivered and self.status in UNDELIVERED

    def age_h(self, now: float | None = None) -> float:
        return ((now or time.time()) - (self.submitted_at or self.updated_at)) / 3600


class MediaTaskLedger:
    """一个 JSONL 文件。写入是追加（一行一次变化），读的时候折叠；按文件 mtime 缓存。"""

    def __init__(
        self, path: str | Path, max_age_h: float = 24.0, max_auto_recoveries: int = 2
    ) -> None:
        self.path = Path(path)
        # 同一个任务自动取回的上限：取回后又等超时的，多半卡死了（或状态词不认识），再取回只是
        # 每次再等满一轮（2026-09-24 审查）。人仍可用 media_recover 点名取回
        self.max_auto_recoveries = max_auto_recoveries
        # 超过这个时长的任务不再自动取回：生成链接约 24h 失效，取回来也下载不了
        self.max_age_h = max_age_h
        self._cache: dict[str, TaskRecord] = {}
        self._sig: tuple[int, int] | None = None

    # ---------- 读 ----------

    def _load(self) -> dict[str, TaskRecord]:
        try:
            st = self.path.stat()
        except OSError:
            self._cache, self._sig = {}, None
            return self._cache
        sig = (st.st_mtime_ns, st.st_size)
        if sig == self._sig:
            return self._cache
        out: dict[str, TaskRecord] = {}
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return self._cache
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # 写一半的行（进程被杀）跳过，不影响其余
            tid = str(row.get("task_id") or "")
            if not tid:
                continue
            rec = out.get(tid) or TaskRecord(task_id=tid)
            for k, v in row.items():
                if hasattr(rec, k) and v is not None:
                    setattr(rec, k, v)
            out[tid] = rec
        self._cache, self._sig = out, sig
        return out

    def get(self, task_id: str) -> TaskRecord | None:
        return self._load().get(task_id)

    def recoverable(self, fingerprint: str, exclude: set[str] | None = None) -> TaskRecord | None:
        """同一份请求最近一个没交付、还在有效期内的任务（本进程正在轮询的除外）。"""
        if not fingerprint:
            return None
        now = time.time()
        skip = exclude or set()
        cands = [
            r
            for r in self._load().values()
            if r.fingerprint == fingerprint
            and r.undelivered
            and r.task_id not in skip
            and r.age_h(now) <= self.max_age_h
            and int(r.recoveries or 0) < self.max_auto_recoveries
        ]
        cands.sort(key=lambda r: r.submitted_at or r.updated_at, reverse=True)
        return cands[0] if cands else None

    def recoverable_where(self, pick: Any) -> TaskRecord | None:
        """按条件找最近一个能自动取回的任务（没交付、在有效期内、没超自动取回次数）。

        按请求指纹找不到的：质检重生成时提示词改过（加了「上一版出了字」这类修正），轮询超时后
        重跑，指纹对不上、取不回来，白付一次（2026-09-26）。调用方按参数里的标签（比如片段的
        段指纹）来找。"""
        now = time.time()
        cands = [
            r
            for r in self._load().values()
            if r.undelivered
            and r.age_h(now) <= self.max_age_h
            and int(r.recoveries or 0) < self.max_auto_recoveries
            and pick(r)
        ]
        cands.sort(key=lambda r: r.submitted_at or r.updated_at, reverse=True)
        return cands[0] if cands else None

    def pending(self, limit: int = 20) -> list[TaskRecord]:
        """没交付的任务，新的在前（给人 / 模型看，决定要不要取回）。"""
        rows = [r for r in self._load().values() if r.undelivered]
        rows.sort(key=lambda r: r.submitted_at or r.updated_at, reverse=True)
        return rows[:limit]

    # ---------- 写 ----------

    def _append(self, row: dict[str, Any]) -> None:
        row["updated_at"] = time.time()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            append_jsonl(self.path, row)
        except OSError:
            pass  # 台账写不进去不能拖垮生成本身

    def submitted(
        self,
        task_id: str,
        *,
        kind: str,
        model: str,
        provider: str,
        fingerprint: str,
        prompt: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        rec = TaskRecord(
            task_id=task_id,
            kind=kind,
            model=model,
            provider=provider,
            fingerprint=fingerprint,
            status="submitted",
            submitted_at=time.time(),
            prompt=prompt,
            params=_jsonable(params or {}),
        )
        row = asdict(rec)
        row.pop("updated_at", None)
        self._append(row)

    def update(
        self, task_id: str, status: str, urls: list[str] | None = None, error: str = ""
    ) -> None:
        row: dict[str, Any] = {"task_id": task_id, "status": status}
        if urls:
            row["urls"] = list(urls)
        if error:
            row["error"] = error[:500]
        self._append(row)

    def recovered(self, task_id: str) -> None:
        """自动取回了一次：计数（超过上限就不再自动取回，改为正常提交）。"""
        rec = self.get(task_id)
        self._append({"task_id": task_id, "recoveries": int(rec.recoveries if rec else 0) + 1})

    def delivered(self, task_id: str) -> None:
        """结果已经登记成资产：以后不再自动取回它。"""
        self._append({"task_id": task_id, "delivered": True})


def append_jsonl(path: Path, row: dict[str, Any] | str) -> tuple[int, int]:
    """往 JSONL 追加一行。上一个进程被杀、最后一行只写了一半（没有换行）时先补换行 ——
    不补的话新记录会拼在半行后面，整行解析失败，这条新记录也跟着丢。

    返回 (这一行写在哪个偏移, 写了几个字节)：增量读台账的进程靠它知道自己这行在哪，
    别的进程在它读完和写入之间追加的行才不会被跳过。"""
    line = row if isinstance(row, str) else json.dumps(row, ensure_ascii=False)
    data = (line.rstrip("\n") + "\n").encode("utf-8")
    with Path(path).open("a+b") as f:
        f.seek(0, 2)
        if f.tell() > 0:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.write(b"\n")
        start = f.tell()
        f.write(data)
    return start, len(data)


def _jsonable(v: Any) -> Any:
    try:
        json.dumps(v, ensure_ascii=False)
        return v
    except (TypeError, ValueError):
        return json.loads(json.dumps(v, ensure_ascii=False, default=str))

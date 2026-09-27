"""会话现场快照 —— 把对话滑窗按 session 落盘，重启后装回。

跨会话真正会丢的只有这一块：资产（workspace/assets/）与长期记忆
（workspace/memory/）本来就落盘，但 ShortTermMemory 是纯内存 ——
重启后模型不知道上次聊到哪儿。这里把窗口里保留的轮次在每个 LOOP_END
存一次；同名 session 启动时装回，接着上次聊。

只存轮次，不存 pin：Memory Brief 每轮重 pin，rollback 说明是一次性的，
skill 目录由分配器装配 —— 复活旧 pin 只会把过期约束带回新会话。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import structlog

from ...harness.context.window import ShortTermMemory, Turn
from ...harness.model.gateway import estimate_tokens

FORMAT = 1


def _safe(name: str) -> str:
    # Windows 文件名非法字符；中文名合法，原样保留
    return "".join(c if c not in '\\/:*?"<>|' else "_" for c in name) or "default"


class SessionSnapshot:
    """一个 session 一份快照文件（JSON）。读写都是小文件同步 IO。"""

    def __init__(self, root: Path, name: str) -> None:
        self.root = root
        self.name = name
        self.path = root / f"{_safe(name)}.json"
        # 用户指定的产物目录（文本/图片/视频落盘到哪）。开工前问一次，之后沿用。
        self.output_dir: str = ""
        # 本会话锁定的视频 / 生图模型（用户定的规则：换模型之前必须先问用户 ——
        # 视频 2026-09-20、生图 2026-09-22）。用户同意切换后记在这里，重启沿用。
        self.video_model: str = ""
        self.image_model: str = ""
        # 用户选的产线标签（2026-09-23：短剧 / 抖音短视频 / 广告 / 设计；空 = 不限定）
        self.content_line: str = ""
        # 这个项目一集几分钟（/length 设；0 = 用 config/drama.yaml 的默认）。2026-09-25：《不渡》
        # 的剧本每集约 5600 字、光台词就四到六分钟，塞不进默认的 4 分钟
        self.episode_minutes: float = 0.0
        # 集长跟剧本走（/length auto，2026-09-26）：不按集长凑总时长
        self.episode_auto: bool = False
        # 这个项目的视频画幅（/ratio 设，如 16:9；空 = 默认：短剧竖屏 9:16、短视频按配方）
        self.aspect_ratio: str = ""
        # 镜头超 3 秒拦不拦成片（/cut，2026-09-27）：False = 只标 ⚠
        self.cut_block: bool = False
        # 上次开工确认的额度（金额 / 视频段数 / 视频秒数 / 图片张数），下次开工拿来当默认
        self.budget: dict[str, Any] = {}
        # 挂起中的人审（重启前没结案的）。之前只在内存里，重启后被悄悄补成「已中断」
        self.pending_review: dict[str, Any] | None = None
        # 上次加载过的 skill：重启后要重新激活，否则历史里写着「正文已常驻」其实没了
        self.active_skills: list[str] = []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.output_dir = str(data.get("output_dir") or "")
                self.video_model = str(data.get("video_model") or "")
                self.image_model = str(data.get("image_model") or "")
                self.content_line = str(data.get("content_line") or "")
                self.aspect_ratio = str(data.get("aspect_ratio") or "")
                try:
                    self.episode_minutes = float(data.get("episode_minutes") or 0.0)
                except (TypeError, ValueError):
                    self.episode_minutes = 0.0
                self.episode_auto = bool(data.get("episode_auto"))
                self.cut_block = bool(data.get("cut_block"))
                b = data.get("budget")
                self.budget = dict(b) if isinstance(b, dict) else {}
                pr = data.get("pending_review")
                self.pending_review = dict(pr) if isinstance(pr, dict) else None
                sk = data.get("active_skills")
                self.active_skills = [str(x) for x in sk] if isinstance(sk, list) else []
        except (OSError, json.JSONDecodeError):
            pass

    def set_output_dir(self, path: str) -> None:
        """记住用户指定的产物目录。读-改-写，不动轮次数据。"""
        self.output_dir = path
        self._patch({"output_dir": path})

    def set_video_model(self, model: str) -> None:
        """记住本会话的视频模型（用户指定或同意切换后的）。"""
        self.video_model = model
        self._patch({"video_model": model})

    def set_image_model(self, model: str) -> None:
        """记住本会话的生图模型（用户指定或同意切换后的）。"""
        self.image_model = model
        self._patch({"image_model": model})

    def set_content_line(self, key: str) -> None:
        """记住用户选的产线（/type）。空串 = 不限定。"""
        self.content_line = key
        self._patch({"content_line": key})

    def set_episode_minutes(self, minutes: float, auto: bool = False) -> None:
        """记住这个项目一集几分钟（/length）。0 = 恢复默认；auto = 跟剧本走（/length auto）。"""
        self.episode_minutes = 0.0 if auto else float(minutes or 0.0)
        self.episode_auto = bool(auto)
        self._patch({"episode_minutes": self.episode_minutes, "episode_auto": self.episode_auto})

    def set_aspect_ratio(self, ratio: str) -> None:
        """记住这个项目的视频画幅（/ratio）。空串 = 恢复默认。"""
        self.aspect_ratio = ratio or ""
        self._patch({"aspect_ratio": self.aspect_ratio})

    def set_cut_block(self, on: bool) -> None:
        """记住这个项目镜头超 3 秒拦不拦成片（/cut）。"""
        self.cut_block = bool(on)
        self._patch({"cut_block": self.cut_block})

    def set_budget(self, limits: dict[str, Any]) -> None:
        """记住这次开工确认的额度，下次开工当默认值拿出来问。"""
        self.budget = {k: v for k, v in limits.items() if v is not None}
        self._patch({"budget": self.budget})

    def inherit_settings(self, other: SessionSnapshot) -> None:
        """第二个窗口另起的会话：项目设置从主会话抄一份（对话、挂起的人审不抄）。"""
        fields: dict[str, Any] = {
            "output_dir": other.output_dir,
            "video_model": other.video_model,
            "image_model": other.image_model,
            "content_line": other.content_line,
            "episode_minutes": other.episode_minutes,
            "episode_auto": other.episode_auto,
            "aspect_ratio": other.aspect_ratio,
            "cut_block": other.cut_block,
            "budget": dict(other.budget),
        }
        for k, v in fields.items():
            setattr(self, k, v)
        self._patch(fields)

    def _patch(self, fields: dict[str, Any]) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, json.JSONDecodeError):
            data = {}
        data.update({"format": FORMAT, "name": self.name, **fields})
        self._write(data)

    def _write(self, data: dict[str, Any]) -> None:
        """原子写：先写临时文件再替换 —— 写一半断电/被杀，旧快照还在（2026-09-23 审查）。"""
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
            self.last_error = ""
        except OSError as e:
            # 记不住不挡聊天，但要留痕：同一文件夹两个终端同名快照互相撞上时会丢这一轮
            # （含 pending_review、模型锁），之前完全静默（2026-09-24 审查）
            self.last_error = f"{type(e).__name__}: {e}"
            structlog.get_logger(__name__).warning(
                "session.snapshot_write_failed", path=str(self.path), error=self.last_error
            )

    def load_into(self, memory: ShortTermMemory, max_tokens: int = 30_000) -> int:
        """装回上次保存的轮次，返回装了几轮。没有快照或文件坏了就当全新会话。

        装回时按窗口上限截断 —— 快照存的是驱逐后的窗口，正常不会超，
        但窗口策略可能在这两次会话之间被调小过。

        再按 max_tokens 从新往旧截：重启后的第一条请求没有任何缓存可吃，
        实测原样装回 10 轮就是 22 万 token 的冷启动，一次 ¥4.6。
        旧轮次的正文都在资产库里，装回太多只是花钱重读。至少留一轮。
        """
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0
        turns_raw = data.get("turns")
        if not isinstance(turns_raw, list):
            return 0

        keep = memory.policy.window_turns
        turns: list[Turn] = []
        for t in turns_raw[-keep:] if keep > 0 else []:
            try:
                turns.append(
                    Turn(
                        index=int(t["index"]),
                        messages=list(t["messages"]),
                        tokens=int(t.get("tokens") or 0),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue  # 坏轮次丢掉，不挡会话恢复
        if not turns:
            return 0
        if max_tokens > 0:
            total = 0
            picked: list[Turn] = []
            for t in reversed(turns):
                size = t.tokens or estimate_tokens(t.messages)
                if picked and total + size > max_tokens:
                    break
                picked.append(t)
                total += size
            dropped = turns[: len(turns) - len(picked)]
            turns = list(reversed(picked))
            # 截掉的轮次之前是直接丢 —— 它们从没进过记忆提取（2026-09-23 审查），
            # 下一次快照一覆盖就永久没了。先交给记忆提取再丢。
            if dropped and memory.on_evict is not None:
                try:
                    memory.on_evict(dropped)
                except Exception:  # noqa: BLE001 — 提取失败不挡会话恢复
                    pass
        memory.turns = turns
        # 轮次编号接续旧会话，新开的轮不会和装回的撞车
        memory._next_index = max(int(data.get("next_index") or 0), turns[-1].index + 1)
        return len(turns)

    def save(
        self,
        memory: ShortTermMemory,
        pending_review: dict[str, Any] | None = None,
        active_skills: list[str] | None = None,
    ) -> None:
        """落盘当前窗口。写失败不上抛 —— 快照是锦上添花，不该打断对话。"""
        self.pending_review = dict(pending_review) if pending_review else None
        if active_skills is not None:
            self.active_skills = list(active_skills)
        payload = {
            "format": FORMAT,
            "name": self.name,
            "next_index": memory._next_index,
            # 产物目录、模型锁、产线、额度跟着轮次一起存 —— save 是整体重写，不带就丢了
            "output_dir": self.output_dir,
            "video_model": self.video_model,
            "image_model": self.image_model,
            "content_line": self.content_line,
            "episode_minutes": self.episode_minutes,
            "episode_auto": self.episode_auto,
            "aspect_ratio": self.aspect_ratio,
            "cut_block": self.cut_block,
            "budget": self.budget,
            "pending_review": self.pending_review,
            "active_skills": self.active_skills,
            "turns": [
                {"index": t.index, "tokens": t.tokens, "messages": t.messages}
                for t in memory.turns
            ],
        }
        self._write(payload)


# ---------------------------------------------------------- 一份快照只给一个窗口（2026-09-26）
# 同一个文件夹开两个窗口：快照是整份覆盖写，后存的把先存的挂起人审、模型锁冲掉；两个 /auto 还会
# 重复派发渲染。第二个窗口另起一个会话（「名字~2」），项目设置从主会话抄一份，/auto 只在主窗口开。


class SessionLock:
    """操作系统级的文件锁：进程没了锁自动释放，不会留下崩溃残留，也不怕 PID 被复用。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: Any = None

    @property
    def held(self) -> bool:
        return self._fh is not None

    def acquire(self) -> bool:
        """占住。别的窗口占着返回 False；锁文件建不了（只读盘之类）当作占到了 —— 锁是保护，
        不是开工条件。"""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fh = self.path.open("a+b")
        except OSError:
            return True
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            fh.close()


def open_session(root: Path, name: str) -> tuple[SessionSnapshot, SessionLock, bool]:
    """打开会话快照并占住它，返回 (快照, 锁, 是不是第二个窗口)。

    别的窗口占着这份会话：另起「name~2」（~3 …）这样的会话，项目设置从主会话抄一份。"""
    lock = SessionLock(root / f"{_safe(name)}.lock")
    if lock.acquire():
        return SessionSnapshot(root, name), lock, False
    main = SessionSnapshot(root, name)
    for i in range(2, 20):
        alt = f"{name}~{i}"
        lock = SessionLock(root / f"{_safe(alt)}.lock")
        if lock.acquire():
            snap = SessionSnapshot(root, alt)
            snap.inherit_settings(main)
            return snap, lock, True
    snap = SessionSnapshot(root, f"{name}~{os.getpid()}")
    snap.inherit_settings(main)
    return snap, SessionLock(root / f"{_safe(snap.name)}.lock"), True

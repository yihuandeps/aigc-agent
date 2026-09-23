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

    def set_budget(self, limits: dict[str, Any]) -> None:
        """记住这次开工确认的额度，下次开工当默认值拿出来问。"""
        self.budget = {k: v for k, v in limits.items() if v is not None}
        self._patch({"budget": self.budget})

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
        except OSError:
            pass  # 记不住不挡聊天

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
            "budget": self.budget,
            "pending_review": self.pending_review,
            "active_skills": self.active_skills,
            "turns": [
                {"index": t.index, "tokens": t.tokens, "messages": t.messages}
                for t in memory.turns
            ],
        }
        self._write(payload)

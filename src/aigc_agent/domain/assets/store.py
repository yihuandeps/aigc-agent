"""M12 Asset Manager —— 一切产出皆 Asset。

它是「可重跑」「可回退」「可对比候选」三个能力的共同底座。
没有它，改一处就要从头生成，多模态场景下成本不可接受。

血缘链要能回答：这张图是从哪句脚本、用什么 prompt、哪个模型、花多少钱生成的。
复盘和成本归因全靠它。

L0 只持有 asset id 字符串（AssetRef），Asset 模型本身住在 L2 —— 依赖单向向下。
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ...harness.context.compaction import is_only_fold_mark
from ...harness.events.bus import EventType
from ..numerals import EPISODE_RE, episode_from
from ..output import slug


class AssetType(StrEnum):
    TEXT = "text"
    OUTLINE = "outline"
    SCRIPT = "script"
    STORYBOARD = "storyboard"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    SUBTITLE = "subtitle"  # srt / vtt
    REPORT = "report"  # 机审报告 / 复盘
    PACKAGE = "package"  # 待发布包（目录）


class AssetStatus(StrEnum):
    """资产状态（2026-09-23 审查缺口 A）：查询默认只取 active。

    之前没有状态字段：/rollback 作废的、质检门换掉的、面容审查弃用的版本，照样被当成
    「最新」取到 —— 只能靠一次性的 pin 提醒模型别用。
    """

    ACTIVE = "active"
    SUPERSEDED = "superseded"  # 有替代它的版本了（面容审查弃用的脸）
    REJECTED = "rejected"  # 质检门没过、被换掉的那一版
    VOID = "void"  # /rollback 作废的下游产物


# 迁移时追溯不到产物目录的旧资产。不属于任何项目，按项目查询时看不见（按 id 仍取得到）
LEGACY_PROJECT = "legacy"


# 集号识别（2026-09-20）：模型自己 save_draft 存的整集剧本、合规修订版都是 outline / text 类型、
# 没有 gen_params.episode，只有摘要里写着「第6集·合规修订版」。之前按类型 + gen_params 认集，
# 这批全部不算数 —— 用户目录里 6–37 集摆着，模型说「只写到第 5 集」。
# 「第12集」「第十二集」都认（2026-09-23 审查：中文数字的集号之前不算数）
_EPISODE_IN_SUMMARY = EPISODE_RE
_EPISODE_TYPES = ("text", "outline", "script", "storyboard")
# 摘要里带这些词的文本资产不是剧本正文（分集目录、审核报告、宣传文案…），不算「这一集写完了」
_NOT_SCRIPT_WORDS = ("目录", "大纲", "报告", "方案", "文案", "档案", "清单", "提示词")
_MIN_SCRIPT_CHARS = 300


class Asset(BaseModel):
    id: str = Field(default_factory=lambda: "as_" + uuid.uuid4().hex[:10])
    type: AssetType = AssetType.TEXT
    mime: str = "text/plain"

    # 小文本直接内联；大内容落盘走 uri。上下文里永远只带 id + summary。
    inline: str | None = None
    uri: str | None = None
    summary: str = ""

    version: int = 1
    parent_ids: list[str] = Field(default_factory=list)  # ← 血缘

    gen_params: dict[str, Any] = Field(default_factory=dict)
    gen_cost: float | None = None
    creator: str = ""  # model:kimi_k3 | human:xxx | tool:xxx

    created_at: float = Field(default_factory=time.time)
    # 单调递增序号，由 store 分配（跨进程也唯一，见 AssetStore._next_seq）。
    # 不能只靠 created_at 排序 —— Windows 上 time.time() 分辨率约 15ms，
    # 同一毫秒内产出的多个资产会撞在一起，"哪个是最新版"变成不确定的。
    seq: int = 0

    # 属于哪个项目（一部剧 / 一个选题），= 产物目录派生的项目键（domain/project.py）。
    # 空 = 没开项目维度的库（测试、一次性脚本）；legacy = 迁移时追溯不到出处的旧资产
    project: str = ""
    status: AssetStatus = AssetStatus.ACTIVE
    status_note: str = ""  # 为什么不是 active 了（被谁替代 / 哪道门没过 / 回退到哪）

    def brief(self) -> str:
        """进上下文的形态：引用 + 摘要，不放内容。"""
        return f"[{self.id}|{self.type.value}|v{self.version}] {self.summary}"


class AssetStore:
    """P1 形态：内存 + 可选落盘。P2 换 SQLite 时只改这个类。"""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root
        self._items: dict[str, Asset] = {}
        self._seq = 0
        # 当前项目键（装配层按产物目录设，/out 换目录时跟着换）。设了之后新资产打上它，
        # find / latest / episodes_done / script_of 默认只看它 —— 之前一律「全库最新」：
        # 新剧显示「已有 79 集」、/auto 拿旧剧的资产库渲新剧（2026-09-23 审查缺口 A）。
        # 空串 = 不分项目（测试、一次性脚本）
        self.project = ""
        # 读不出来的资产文件（坏 JSON / 旧版本写一半崩了）。之前静默跳过，资产凭空消失没人知道
        self.load_errors: list[str] = []
        # 多进程：两个终端同时开 Agent，对方新写 / 改过的资产要看得见 —— 按文件 mtime 增量重读，
        # 查询时最多每 refresh_interval 秒扫一次目录（Windows 上 scandir 自带 mtime，千份约 2ms）
        self._mtimes: dict[str, int] = {}
        self._scanned_at = 0.0
        self.refresh_interval = 1.0
        self._loop: asyncio.AbstractEventLoop | None = None  # 主事件循环（见 _emit_created）
        # 事件总线（装配层挂上才有）。挂上后每次落库发 ASSET_CREATED ——
        # 按集流水管线（EpisodePipeline）靠它实时知道「哪集就绪了」。
        self.bus: Any = None
        # 产物目录镜像（OutputPrefs，装配层挂上才有）。挂上后文本类资产
        # 自动往用户目录写一份可读的 .md —— 这里存的是系统记录（JSON），
        # 不是给人读的。
        self.mirror: Any = None
        if root:
            root.mkdir(parents=True, exist_ok=True)
            self._load()

    def bind_loop(self) -> None:
        """记下当前事件循环（装配时在主循环里调）：线程里建的资产要往它上面发事件。"""
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def _load(self) -> None:
        """启动时把已落盘的资产读回来。

        不做这个的话，资产只在单个进程内存活 —— 上一次生成的视频，
        下次启动就找不到了，跨会话的「拿上次那批素材来合成」直接失效。
        单条坏数据跳过，不让它拖垮整个库，但记进 load_errors（启动时告警）。
        """
        self._scan()

    def _scan(self) -> None:
        """把磁盘上新增的、被别的进程改过的资产读进来；被挪走 / 删掉的移出去。"""
        root = self.root
        if root is None:
            return
        self._scanned_at = time.monotonic()
        try:
            entries = list(os.scandir(root))
        except OSError:
            return
        present: set[str] = set()
        for e in entries:
            name = e.name
            if not (name.startswith("as_") and name.endswith(".json")):
                continue
            aid = name[:-5]
            present.add(aid)
            try:
                mtime = e.stat().st_mtime_ns
            except OSError:
                continue
            if self._mtimes.get(aid) == mtime:
                continue
            try:
                a = Asset.model_validate_json(Path(e.path).read_text(encoding="utf-8"))
            except Exception as ex:  # noqa: BLE001
                msg = f"{name}：{type(ex).__name__}"
                if msg not in self.load_errors:
                    self.load_errors.append(msg)
                continue
            self._items[a.id] = a
            self._mtimes[aid] = mtime
            self._seq = max(self._seq, a.seq)
        # 别的进程挪走的（迁移时移进回收站的测试桩）
        for aid in [k for k in self._mtimes if k not in present]:
            self._mtimes.pop(aid, None)
            self._items.pop(aid, None)

    def _maybe_refresh(self) -> None:
        if self.root is not None and time.monotonic() - self._scanned_at >= self.refresh_interval:
            self._scan()

    def _next_seq(self) -> int:
        """全局单调序号。多进程共用一个计数文件，加锁读改写 —— 之前各进程各算各的，
        测试进程和真实会话同时写，库里撞出 61 个重复 seq，「哪个是最新版」就不确定了。"""
        if self.root is None:
            self._seq += 1
            return self._seq
        counter = self.root / ".seq"
        with _file_lock(self.root / ".seq.lock"):
            try:
                disk = int(counter.read_text(encoding="utf-8").strip() or 0)
            except (OSError, ValueError):
                disk = 0
            n = max(disk, self._seq) + 1
            try:
                _atomic_write(counter, str(n))
            except OSError:
                pass
        self._seq = n
        return n

    def put(self, asset: Asset) -> Asset:
        if not asset.project and self.project:
            asset.project = self.project
        if not asset.seq:
            asset.seq = self._next_seq()
        self._items[asset.id] = asset
        if self.root:
            path = self.root / f"{asset.id}.json"
            # 先写临时文件再替换：写到一半崩溃不会留下半截 JSON
            _atomic_write(path, asset.model_dump_json(indent=2))
            try:
                self._mtimes[asset.id] = path.stat().st_mtime_ns
            except OSError:
                pass
        self._mirror_text(asset)
        self._emit_created(asset)
        return asset

    def _emit_created(self, asset: Asset) -> None:
        """资产落库事件：按集流水管线靠它做环节间的实时交接。

        fire-and-forget：put 是同步方法，事件用 create_task 发出，不阻塞落库；
        没有运行中的事件循环（纯同步上下文）就跳过 —— 没有循环也没有订阅者能消费。
        create_blob 会 put 两次（入库、补 uri），同一 id 来两条事件，
        消费方必须按 id 幂等。
        """
        if self.bus is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 在线程里建的（fs_import 走 to_thread）：之前事件直接丢了，按集流水漏接这批资产。
            # 投回主事件循环去发（2026-09-23 审查）
            main = self._loop
            if main is None or main.is_closed():
                return
            snapshot = dict(
                id=asset.id, type=asset.type.value, summary=asset.summary,
                gen_params=dict(asset.gen_params), parent_ids=list(asset.parent_ids),
                seq=asset.seq, project=asset.project, status=asset.status.value,
                creator=asset.creator,
            )
            main.call_soon_threadsafe(
                lambda: main.create_task(self.bus.emit(EventType.ASSET_CREATED, **snapshot))
            )
            return
        self._loop = loop  # 记住主循环，线程里建资产时往这里投
        loop.create_task(
            self.bus.emit(
                EventType.ASSET_CREATED,
                id=asset.id,
                type=asset.type.value,
                summary=asset.summary,
                gen_params=dict(asset.gen_params),
                parent_ids=list(asset.parent_ids),
                seq=asset.seq,
                project=asset.project,
                status=asset.status.value,
                creator=asset.creator,
            )
        )

    def _mirror_text(self, asset: Asset) -> None:
        """文本类资产往用户产物目录写一份可读的 .md（纯内容，不带元数据）。

        用户定的：生成的文本要落一份到指定文件夹。写失败不上抛 ——
        镜像是附加值，不是主链路。
        """
        prefs = self.mirror
        if prefs is None or asset.inline is None:
            return
        if asset.type not in (AssetType.TEXT, AssetType.SCRIPT, AssetType.OUTLINE):
            return
        try:
            d = prefs.dir_for("texts")
            d.mkdir(parents=True, exist_ok=True)
            name = f"{slug(asset.summary)}-{asset.id}.md"
            (d / name).write_text(asset.inline, encoding="utf-8")
        except OSError:
            pass

    def create(
        self,
        content: str,
        type_: AssetType = AssetType.TEXT,
        summary: str = "",
        parents: list[str] | None = None,
        creator: str = "",
        gen_params: dict[str, Any] | None = None,
        gen_cost: float | None = None,
    ) -> Asset:
        # 折叠占位符不许落库（2026-09-22）：参数层已经拦过一道，这里再兜一道 ——
        # 2026-09-21 就有 14 份「分镜」存成了 11 字的 <4939 字已折叠>，工具照样报成功，
        # 直到下游 drama_shots 读它才炸。整段只是占位符的内容不可能是真东西，零误判。
        if is_only_fold_mark(content):
            raise ValueError(
                f"内容只是上下文折叠留下的占位符（{content.strip()[:30]}），不是真正的正文，"
                "不存。把完整内容重新写出来；太长就 fs_write 分块写本地文件再 fs_import。"
            )
        return self.put(
            Asset(
                type=type_,
                inline=content,
                summary=summary or _auto_summary(content),
                parent_ids=parents or [],
                creator=creator,
                gen_params=gen_params or {},
                gen_cost=gen_cost,
            )
        )

    def create_blob(
        self,
        data: bytes,
        ext: str,
        type_: AssetType = AssetType.AUDIO,
        mime: str = "application/octet-stream",
        summary: str = "",
        parents: list[str] | None = None,
        creator: str = "",
        gen_params: dict[str, Any] | None = None,
    ) -> Asset:
        """存二进制产物（TTS 音频、下载回来的图/视频）。

        文件落到 blobs/ 下，Asset 只记路径 —— 上下文里永远不出现字节。
        """
        asset = Asset(
            type=type_,
            mime=mime,
            summary=summary or f"{type_.value} {len(data) / 1024:.0f}KB",
            parent_ids=parents or [],
            creator=creator,
            gen_params=gen_params or {},
        )
        self.put(asset)  # 先入库拿到 seq
        blob_dir = (self.root or Path(".")) / "blobs"
        blob_dir.mkdir(parents=True, exist_ok=True)
        path = blob_dir / f"{asset.id}{ext if ext.startswith('.') else '.' + ext}"
        path.write_bytes(data)
        asset.uri = str(path)
        return self.put(asset)

    def blob(self, asset_id: str) -> bytes:
        """读回二进制。ASR 要拿音频字节去转写。"""
        a = self.get(asset_id)
        if not a.uri:
            raise ValueError(f"{asset_id} 没有文件路径，它可能是内联文本资产")
        return Path(a.uri).read_bytes()

    def revise(self, base_id: str, content: str, summary: str = "", creator: str = "") -> Asset:
        """产出新版本，保留血缘。打回重做走这条路，不是覆盖原资产。"""
        # 同 create：折叠占位符不许落库（之前 revise 这条路没拦，2026-09-23 审查）
        if is_only_fold_mark(content):
            raise ValueError(
                f"内容只是上下文折叠留下的占位符（{content.strip()[:30]}），不是真正的正文，"
                "不存。把完整内容重新写出来；太长就 fs_write 分块写本地文件再 fs_import。"
            )
        base = self.get(base_id)
        return self.put(
            Asset(
                type=base.type,
                mime=base.mime,
                inline=content,
                summary=summary or _auto_summary(content),
                version=base.version + 1,
                parent_ids=[base.id],
                creator=creator,
                # 集号等标签跟着新版本走。之前不带：第 6 集改一稿就从 find_episode 里消失了
                gen_params=dict(base.gen_params),
                # 在当前项目里改的就归当前项目；没开项目维度时跟着原稿
                project=self.project or base.project,
            )
        )

    def set_status(self, asset_id: str, status: AssetStatus | str, note: str = "") -> Asset:
        """改状态（作废 / 被替代 / 质检没过）。不删任何东西，按 id 仍取得到。"""
        a = self.get(asset_id)
        a.status = AssetStatus(status)
        if note:
            a.status_note = note[:200]
        return self.put(a)

    def has(self, asset_id: str) -> bool:
        if asset_id in self._items:
            return True
        self._refresh_on_miss()
        return asset_id in self._items

    def _refresh_on_miss(self) -> None:
        """按 id 找不到时先重读一次目录：可能是另一个进程刚写的。限频，免得一串找不到的 id
        每个都扫一遍目录。"""
        if self.root is not None and time.monotonic() - self._scanned_at > 0.2:
            self._scan()

    def projects(self) -> dict[str, int]:
        """各项目的资产数（含 legacy 和没分项目的空串）。"""
        self._maybe_refresh()
        out: dict[str, int] = {}
        for a in self._items.values():
            out[a.project] = out.get(a.project, 0) + 1
        return out

    def get(self, asset_id: str) -> Asset:
        if asset_id not in self._items:
            self._refresh_on_miss()
        if asset_id not in self._items:
            near = self.nearest(asset_id)
            hint = ""
            if near:
                hint = f"。相近的 id：{', '.join(near)}（可能记错了几位，用 list_assets 核对）"
            raise KeyError(f"资产不存在：{asset_id}{hint}")
        return self._items[asset_id]

    def nearest(self, asset_id: str, n: int = 3) -> list[str]:
        """和给定 id 前缀最接近的几个真实 id。

        超长上下文下模型会把 as_d619a69d97 记成 as_d6191a0a80 —— 前几位对、后面编。
        报错时把相近的列出来，比一句「不存在」有用。
        """
        key = (asset_id or "").strip()
        if not key.startswith("as_") or len(key) < 6:
            return []

        def common(a: str, b: str) -> int:
            k = 0
            for x, y in zip(a, b, strict=False):
                if x != y:
                    break
                k += 1
            return k

        scored = [(common(key, a.id), a.seq, a.id) for a in self._items.values()]
        scored = [s for s in scored if s[0] >= 6]  # 至少 "as_" 后再对 3 位
        scored.sort(reverse=True)
        return [aid for _, _, aid in scored[:n]]

    def find(
        self,
        type_: AssetType | None = None,
        episode: int = 0,
        creator: str = "",
        contains: str = "",
        newest_first: bool = True,
        project: str | None = None,
        include_inactive: bool = False,
    ) -> list[Asset]:
        """按条件过滤。creator 按前缀匹配（"tool:" 能匹配所有工具产物）。

        默认只看**当前项目**的 **active** 资产（缺口 A）；project="*" 看全库，
        include_inactive=True 连作废 / 被替代 / 质检没过的一起看。
        """
        self._maybe_refresh()
        scope = self.project if project is None else project
        out = []
        # 先拷一份再遍历：fs_import 在线程里建资产，边遍历边插入会抛 RuntimeError
        for a in list(self._items.values()):
            # 没分项目的旧资产（迁移前的存量）哪个项目都看得见 —— 等于迁移前的老行为，
            # 不会一升级就「东西全没了」；agent assets migrate 分完就没有这种了
            if scope not in ("", "*") and a.project not in (scope, ""):
                continue
            if not include_inactive and a.status is not AssetStatus.ACTIVE:
                continue
            if type_ is not None and a.type is not type_:
                continue
            if episode and self.episode_of(a) != episode:
                continue
            if creator and not (a.creator or "").startswith(creator):
                continue
            if contains and contains not in (a.summary or ""):
                continue
            out.append(a)
        # seq 相同（旧版本多进程撞号留下的）再按创建时间分先后
        out.sort(key=lambda a: (a.seq, a.created_at), reverse=newest_first)
        return out

    # 短剧一集的产物分类：按创建者认，同一集多个版本取 seq 最大的
    _EPISODE_KINDS: tuple[tuple[str, str], ...] = (
        ("剧本", "script"),
        ("分镜脚本", "tool:drama_storyboard"),
        ("视频提示词", "tool:drama_shots"),
        ("视频片段", "tool:drama_render_shots"),
    )

    @staticmethod
    def episode_of(asset: Asset) -> int:
        """资产属于第几集：gen_params.episode 优先；文本类资产再按摘要里的「第N集」认。
        0 = 不属于某集。"""
        try:
            n = int(asset.gen_params.get("episode") or 0)
        except (TypeError, ValueError):
            n = 0
        if n > 0:
            return n
        if asset.type.value in _EPISODE_TYPES:
            return episode_from(asset.summary or "")
        return 0

    @staticmethod
    def _looks_like_script(asset: Asset) -> bool:
        """没标 type=script 的文本资产算不算这一集的剧本：
        摘要有集号、不是目录/报告/文案、有正文长度。"""
        if asset.type not in (AssetType.OUTLINE, AssetType.TEXT):
            return False
        summary = asset.summary or ""
        if not _EPISODE_IN_SUMMARY.search(summary):
            return False
        if any(w in summary for w in _NOT_SCRIPT_WORDS):
            return False
        return len(asset.inline or "") >= _MIN_SCRIPT_CHARS and not is_only_fold_mark(asset.inline)

    def script_of(self, episode: int) -> Asset | None:
        """某一集的剧本，最新版。type=script 优先；没有就退到摘要写着「第N集」的大纲 / 文本资产
        （模型用 save_draft(kind="outline") 存的整集、合规修订版都是这种）。"""
        # **标了 type=script 不等于它就是剧本。** 工作区里第 1 集躺着 41 份 script，
        # 其中 39 份是垃圾：30 字的「第1集·标题」（9-18 起测试往真实 workspace 写的）
        # 和 11 字的折叠空壳（9-21 那次事故），它们都比真剧本新，不筛就永远挡在前面，
        # 下游拿着 30 个字去拆分镜。所以两道筛：占位符空壳先去掉，再要够长度。
        typed = [a for a in self.find(type_=AssetType.SCRIPT, episode=episode)
                 if not is_only_fold_mark(a.inline)]
        real = [a for a in typed if len(a.inline or "") >= _MIN_SCRIPT_CHARS]
        if real:
            return real[0]
        loose = [a for a in self.find(episode=episode) if self._looks_like_script(a)]
        if loose:
            return loose[0]
        # 全都偏短：退回最长的那份，总比装作没有强（也可能真是一集很短的剧本）
        return max(typed, key=lambda a: len(a.inline or "")) if typed else None

    def best_doc(self, type_: AssetType, contains: str, min_chars: int = 300) -> Asset | None:
        """某类文档（创作方案 / 角色档案 / 分集目录）当前项目里最新的**真**版本：
        有创建者、不是占位符；够长的优先，都偏短就退到最长的那份。

        2026-09-23 审查：资产库里「最新」的这三类全是测试桩（12–59 字、creator 为空），
        项目卡每轮 pin 着、写剧本默认也取它 —— 真实的写第 3 集任务里出现过「陆离：守门人，沉默」。
        """
        cands = [
            a
            for a in self.find(type_=type_, contains=contains)
            if a.creator and not is_only_fold_mark(a.inline)
        ]
        if not cands:
            return None
        real = [a for a in cands if len(a.inline or "") >= min_chars]
        if real:
            return real[0]
        return max(cands, key=lambda a: (len(a.inline or ""), a.seq))

    def episode_assets(self, episode: int) -> dict[str, Asset]:
        """某一集的各类产物，各取最新版。剧本见 script_of，其余按创建者。"""
        found: dict[str, Asset] = {}
        for label, key in self._EPISODE_KINDS:
            if key == "script":
                script = self.script_of(episode)
                if script is not None:
                    found[label] = script
                continue
            items = self.find(episode=episode, creator=key)
            if items:
                found[label] = items[0]
        return found

    def episodes_done(self, project: str | None = None) -> list[int]:
        """已有剧本的集号，升序去重：type=script 的，
        加上摘要标着「第N集」的整集文本（见 script_of）。只看当前项目（缺口 A：
        之前全库算，混了 4 部剧，新剧开写就说「全剧已有剧本 79 集」）。"""
        eps: set[int] = set()
        for a in self.find(project=project):
            n = self.episode_of(a)
            if n <= 0:
                continue
            if a.type is AssetType.SCRIPT or self._looks_like_script(a):
                eps.add(n)
        return sorted(eps)

    def content(self, asset_id: str) -> str:
        a = self.get(asset_id)
        if a.inline is not None:
            return a.inline
        if a.uri:
            return Path(a.uri).read_text(encoding="utf-8")
        return ""

    def latest(self, type_: AssetType | None = None, project: str | None = None) -> Asset | None:
        """当前项目里最近产出的 active 资产。按 seq 排，稳定可靠。"""
        items = self.find(type_=type_, project=project)
        return items[0] if items else None

    def all(self) -> list[Asset]:
        """**全库**（跨项目、含已作废的），按 seq。按当前项目取用 find()。"""
        self._maybe_refresh()
        return sorted(self._items.values(), key=lambda a: (a.seq, a.created_at))

    def lineage(self, asset_id: str) -> list[Asset]:
        """回溯血缘链，从最早的祖先到自己。"""
        chain: list[Asset] = []
        seen: set[str] = set()
        cur = self.get(asset_id)
        while True:
            if cur.id in seen:
                break
            seen.add(cur.id)
            chain.append(cur)
            if not cur.parent_ids:
                break
            cur = self.get(cur.parent_ids[0])
        return list(reversed(chain))

    def __len__(self) -> int:
        return len(self._items)


def _atomic_write(path: Path, text: str) -> None:
    """先写临时文件再 os.replace：写到一半崩溃不会留下半截文件（之前那种 JSON 读的时候被
    静默跳过，资产凭空消失）。Windows 上目标文件正被别的进程读时 replace 会短暂失败，重试。"""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    for k in range(15):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.02 * (k + 1))
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def _file_lock(path: Path, timeout: float = 5.0, stale: float = 30.0) -> Iterator[None]:
    """跨进程互斥：O_CREAT|O_EXCL 建锁文件。拿不到就等；锁文件放了超过 stale 秒视为崩溃残留。
    等到 timeout 还拿不到就不等了 —— 宁可冒一次撞号，也不能把生成链卡死。"""
    deadline = time.monotonic() + timeout
    fd: int | None = None
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except (FileExistsError, PermissionError):
            try:
                if time.time() - path.stat().st_mtime > stale:
                    path.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.monotonic() > deadline:
                break
            time.sleep(0.01)
        except OSError:
            break
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
            try:
                path.unlink()
            except OSError:
                pass


def local_copy(asset: Asset) -> Path | None:
    """资产在本机的文件：产物目录里的副本（gen_params.local）优先，没了退到 Agent 在
    blobs/ 自留的那份（gen_params.blob），再退到 uri 本身是本地文件的（导入的素材、
    create_blob 落的音频）。都不在返回 None。

    2026-09-23 审查：产物目录是用户的，会被整理、移动、删除 —— E:\\内容测试\\images 整个
    没了之后 194 个图片资产找不到本地副本，远端链接也早过期。读本地副本一律走这里。
    """
    gp = asset.gen_params or {}
    for key in ("local", "blob"):
        v = str(gp.get(key) or "")
        if v:
            p = Path(v)
            if p.exists():
                return p
    uri = asset.uri or ""
    if uri and not uri.startswith(("http://", "https://", "asset://", "data:")):
        p = Path(uri)
        if p.exists():
            return p
    return None


# 从外面拿别人的内容回来的工具：它们的产物不是「本系统生成」
_FETCHERS = ("tool:fetch_douyin", "tool:fetch_stock_media", "tool:fetch_media_url",
             "tool:browse_and_copy")


def rights_of(asset: Asset) -> str:
    """版权状态：generated（本系统生成）/ human（人给的）/ licensed（素材站、授权写清了）/
    unknown（来源不明：别人的原片、授权没写清）。

    M9 检索和 M15 机审共用这一个口径，别各判各的。2026-09-23 审查：之前只看创建者 ——
    fetch_media_url / fetch_douyin 下回来的别人的原片 creator 是 tool:*，被判成「自己生成的」，
    机审查不出版权问题。
    """
    c = asset.creator or ""
    lic = str((asset.gen_params or {}).get("license") or "").strip().lower()
    if c in _FETCHERS:
        if not lic or lic.startswith("unknown") or "来源不明" in lic:
            return "unknown"
        return "licensed"
    if c.startswith(("model:", "tool:", "pipeline:", "stub")):
        return "generated"
    if c.startswith("human"):
        return "human"
    return "unknown"


def _auto_summary(content: str, limit: int = 40) -> str:
    one_line = " ".join(content.split())
    return one_line if len(one_line) <= limit else one_line[:limit] + "…"


def dump_index(store: AssetStore, path: Path) -> None:
    """导出一份人可读的索引，方便复盘时对着看。"""
    rows = [
        {
            "id": a.id,
            "type": a.type.value,
            "v": a.version,
            "summary": a.summary,
            "parents": a.parent_ids,
            "creator": a.creator,
            "cost": a.gen_cost,
            "project": a.project,
            "status": a.status.value,
        }
        for a in store.all()
    ]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

"""按集流水并行管线（EpisodePipeline）—— /auto 批量出稿的引擎。

剧本按集落资产（save_draft episode=N），ASSET_CREATED 事件驱动本管线：
某一集的剧本一落库就后台拆这一集的分镜；全剧齐就拆资产库、渲参考图；之后逐集出
视频提示词、逐集渲染出「第N集.mp4」。环节间不排队等齐，谁先就绪谁先跑。

2026-09-23 审查后分镜改成**逐集拆**（传这一集的 script_id）：之前 5 集并成一次调用，
一集 80–120 个镜头行，5 集必然超出模型单次输出上限被截断 → 解析失败 → 这一批标失败
永不重派，流水线就卡在那里。

设计要点：

- **交接单元是资产，事件只是触发器**。每次资产事件触发一次 reconcile：
  从 AssetStore 重建视图，把「该做还没做」的下一环派出去。天然幂等，
  重启后扫一遍存量资产就能接着跑（on_enable 也走同一条路）。
- **族裔/语言/总集数读 `<产物目录>/.drama-state.json`**（drama-script skill
  维护，本类是它第一个代码读者）。缺了不猜，打提示等人补 —— 与
  options.py「没给就停下来问」同一原则。
- **渲染花钱**：派发前查 CostGuard，超支即暂停渲染派发（刹车不是故障）。
- **只在 /auto on 时派发**（enabled）；off 时事件照收不动作。
- 派发全部走 `registry.invoke()`（同 drama_cmd 的直调模式），进 Trace。
- **失败的环节**记下当时的输入：输入变了（改了剧本、重拆了分镜、换了资产库）自动重派；
  文本环节（便宜）隔一会儿自动再试一次；花钱的不自动重试，/auto retry 由人发起。
- **旧规格的视频提示词不直接拿去渲**：规格戳（EpisodeFormat.stamp）对不上就按现行规格
  重出一遍（文本环节），每集只重出一次。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable, Coroutine
from typing import Any

from ...harness.events.bus import Event, EventBus, EventType
from ...harness.model.budget import CostGuard
from ...harness.tools.registry import ToolRegistry
from ..assets.store import AssetStore, AssetType
from ..drama.parse import parse_episodes
from ..output import OutputPrefs

_STATE_FILE = ".drama-state.json"
# 只有这些资产会改变流水线的视图：剧本 + drama 各环节的产物。
# 渲染时每段视频、每张图都会落库（一段要 put 2–3 次），之前每次都触发一次全量 reconcile
_RELEVANT_CREATORS = frozenset({
    "tool:drama_storyboard", "tool:drama_assets", "tool:drama_render_assets",
    "tool:drama_shots", "tool:drama_render_shots",
})
# 文本环节：失败了值得自动再试一次（便宜；偶发的超时 / 截断常见）
_TEXT_STAGES = ("storyboard", "assets", "shots")


class EpisodePipeline:
    """剧集流水线编排器：资产事件 → reconcile → 后台任务派发。"""

    def __init__(
        self,
        registry: ToolRegistry,
        assets: AssetStore,
        bus: EventBus,
        output_prefs: OutputPrefs,
        guard: CostGuard | None = None,
        batch_size: int = 5,
        enabled: bool = False,
        max_parallel_text: int = 3,
        max_parallel_renders: int = 2,
        spec: str = "",
    ) -> None:
        self.registry = registry
        self.assets = assets
        self.bus = bus
        self.output_prefs = output_prefs
        self.guard = guard
        self.batch_size = batch_size
        self.enabled = enabled
        # 人在控制台看到的派发/完成提示（main.py 注入 console.print）
        self.note: Callable[[str], None] | None = None

        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._inflight: set[str] = set()  # 已派发未完成（reconcile 不重复派）
        # 失败标记：环节 → 失败时的输入。输入变了自动重派；/auto retry 全部重派。
        # 之前是个 set，只有重启才清，修好了输入也不会重跑（2026-09-23 审查）
        self._failed: dict[str, str] = {}
        self._inputs: dict[str, str] = {}  # 派发时的输入（失败时照抄进 _failed）
        self._attempts: dict[str, int] = {}  # 连续失败次数（文本环节自动再试用）
        self._cooldown: dict[str, float] = {}  # 自动再试之前的冷却截止（monotonic）
        self.text_retries = 1
        self.retry_delay = 30.0
        # 现行规格戳（EpisodeFormat.stamp）；空 = 不校验。每集只按新规格重出一次提示词
        self.spec = spec
        self._respec_eps: set[int] = set()
        self._render_paused = False  # 预算刹车
        self._state_warned = False
        # 文本环节（分镜/资产库/提示词）与渲染环节分开限流：
        # 每个渲染调用内部还有自己的 concurrency.video 信号量，
        # 管线层再不压，N 集并行渲染会把远端接口打爆。
        self._text_sem = asyncio.Semaphore(max_parallel_text)
        self._render_sem = asyncio.Semaphore(max_parallel_renders)
        # 分镜资产 id → 它覆盖的集号。资产不可变，解析一次就够 —— 之前每条资产事件都把
        # 全库的分镜 JSON 重新解析一遍
        self._sb_eps: dict[str, list[int]] = {}

    # ---------- 生命周期 ----------

    def attach(self) -> None:
        """挂到事件总线上。attach 之后资产落库就会触发 reconcile。"""
        self.bus.subscribe(self._on_event)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run())

    async def aclose(self) -> None:
        for t in self._tasks:
            t.cancel()
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    @property
    def busy(self) -> bool:
        """还有派出去没跑完的活。"""
        return bool(self._inflight)

    def reset(self) -> None:
        """换项目时清掉按集号记的状态：失败标记、预算刹车、状态文件提示 ——
        不清的话上一部剧第 3 集的失败标记会挡住新剧的第 3 集（缺口 A）。"""
        self._failed.clear()
        self._inputs.clear()
        self._attempts.clear()
        self._cooldown.clear()
        self._respec_eps.clear()
        self._render_paused = False
        self._state_warned = False

    def retry(self) -> list[str]:
        """/auto retry：清掉失败标记和冷却，全部重派。返回被清掉的环节。

        渲染重派只补缺的段（drama_render_shots 默认复用过了质检门的段）；没过质检门的段会重渲。
        """
        keys = sorted(self._failed)
        self._failed.clear()
        self._attempts.clear()
        self._cooldown.clear()
        self._render_paused = False
        self.kick()
        return keys

    @property
    def failed(self) -> list[str]:
        return sorted(self._failed)

    def on_enable(self) -> None:
        """/auto on：踢一次 reconcile，扫存量资产补派（on 之前写的剧本不漏）。"""
        self.kick()

    def kick(self) -> None:
        """踢一次 reconcile。/budget reset 等外部状态变化后也靠它续派。"""
        self._queue.put_nowait("")

    # ---------- 事件入口 ----------

    def _on_event(self, ev: Event) -> None:
        # 同步回调只入队：EventBus.emit 串行等 handler，这里绝不能做重活。
        if ev.type is not EventType.ASSET_CREATED:
            return
        data = ev.data or {}
        creator = data.get("creator")
        # 只有剧本和 drama 各环节的产物会改变视图；没带创建者的（老事件）照旧踢
        if creator is None or creator in _RELEVANT_CREATORS or data.get("type") == "script":
            self.kick()

    async def _run(self) -> None:
        while True:
            await self._queue.get()
            # 事件常成群来（一批 5 集连存），排空合并成一次 reconcile
            while not self._queue.empty():
                self._queue.get_nowait()
            try:
                await self._reconcile()
            except Exception as e:  # noqa: BLE001 —— 管线崩溃不能拖垮主循环
                await self.bus.emit(EventType.WARNING, message=f"流水线 reconcile 失败：{e}")

    # ---------- 核心：reconcile ----------

    async def _reconcile(self) -> None:
        if not self.enabled:
            return
        state = self._load_state()
        if state is None:
            return  # _load_state 已打提示（只打一次）
        ethnicity, language, total = state
        view = self._scan()

        # ① 分镜：每一集的剧本一到就单独拆这一集
        for ep, script_id in sorted(view["scripts"].items()):
            if total and ep > total:
                continue
            key = f"storyboard:{ep}"
            if self._skip(key, script_id) or ep in view["storyboards"]:
                continue
            self._spawn(
                key, self._storyboard(ep, script_id, ethnicity, language), self._text_sem,
                inputs=script_id,
            )

        # ② 全剧剧本齐 → 拆资产库 → 渲参考图（各一次，跨集复用）
        if total and all(ep in view["scripts"] for ep in range(1, total + 1)):
            script_ids = ",".join(view["scripts"][ep] for ep in range(1, total + 1))
            if not view["assets_lib"] and not self._skip("assets", script_ids):
                self._spawn(
                    "assets",
                    self._assets_lib(view["scripts"], total, ethnicity, language),
                    self._text_sem,
                    inputs=script_ids,
                )
            elif (
                view["assets_lib"]
                and not view["refs"]
                and not self._skip("refs", view["assets_lib"])
                and await self._budget_ok("image", self._refs_need(view["assets_lib"]), 0.0,
                                          "渲参考图")
            ):
                self._spawn(
                    "refs", self._refs(view["assets_lib"]), self._render_sem,
                    inputs=view["assets_lib"],
                )

        # ③ 各集视频提示词：需要这一集的分镜 + 资产库（提示词要绑定资产 ID）。
        # 参考图是渲染的前置，不是提示词的 —— 不等它，提示词与渲图并行跑。
        # 已有提示词但规格戳是旧的、这一集又还没渲：按现行规格重出一次再渲。
        stale = view["stale_shots"]
        if view["assets_lib"]:
            for ep, sb_id in sorted(view["storyboards"].items()):
                key = f"shots:{ep}"
                inputs = f"{sb_id}|{view['assets_lib']}"
                if self._skip(key, inputs):
                    continue
                have = view["shots"].get(ep)
                respec = bool(
                    have and have in stale and ep not in view["rendered"]
                    and ep not in self._respec_eps
                )
                if have and not respec:
                    continue
                if respec:
                    self._respec_eps.add(ep)
                    self._say(
                        f"⚙ 第 {ep} 集的视频提示词是旧规格（{stale[have] or '没有规格戳'}），"
                        f"按现行规格（{self.spec}）重出一遍再渲"
                    )
                self._spawn(
                    key, self._shots(ep, sb_id, view["assets_lib"]), self._text_sem,
                    inputs=inputs,
                )

        # ④ 各集渲染：提示词 + 参考图都齐才派，预算超支就刹住（预算恢复后自动解除）
        if not view["refs"]:
            return
        for ep, shots_id in sorted(view["shots"].items()):
            key = f"render:{ep}"
            inputs = f"{shots_id}|{view['refs']}"
            if self._skip(key, inputs) or ep in view["rendered"]:
                continue
            shots_key = f"shots:{ep}"
            if shots_key in self._inflight:
                continue  # 提示词正在按新规格重出：等新的落库再渲，别拿旧的花钱
            if shots_id in stale and (shots_key in self._failed or shots_key in self._cooldown):
                continue  # 按新规格重出没成：旧规格的不自动渲，等自动再试或人处理
            units, seconds = self._render_need(ep, shots_id)
            if not await self._budget_ok("video", units, seconds, f"渲第 {ep} 集"):
                break
            self._spawn(
                key, self._render(ep, shots_id, view["refs"]), self._render_sem, inputs=inputs
            )

    async def _budget_ok(
        self, kind: str = "", units: int = 1, seconds: float = 0.0, what: str = "渲染"
    ) -> bool:
        """花钱的环节（参考图/视频渲染）派发前查预算闸：按这一步**要花多少**查。

        之前只查金额口径，而视频没有单价、金额口径看不见它 —— /auto 按集流水一集
        几十段视频没有任何刹车（2026-09-23 审查）。现在按这一集的段数和秒数查
        （参考图按张数）。不够：刹住（_render_paused）、说清差多少 —— 后台不弹问题
        （会和输入框抢输入），人用 /budget set 或 /budget allow 调完额度后自动续派。
        没装闸就放行。
        """
        if self.guard is None:
            return True
        verdict = self.guard.check(kind, units=max(1, units), seconds=seconds)
        if verdict:
            self._render_paused = False
            return True
        if not self._render_paused:
            self._render_paused = True
            need = f"要 {units} 个" + (f"、{seconds:.0f} 秒" if seconds else "")
            msg = f"额度不够{what}（{need}；{verdict.reason}），流水线暂停派发"
            self._say(
                f"⛔ {msg} —— /budget set 调额度或 /budget allow 追加后自动续派"
            )
            await self.bus.emit(EventType.WARNING, message=msg)
        return False

    def _render_need(self, ep: int, shots_id: str) -> tuple[int, float]:
        """渲这一集要几段、几秒（按提示词算；读不出来按 1 段 15 秒保守估）。"""
        try:
            from ..drama import parse_shots

            shots, err = parse_shots(self.assets.content(shots_id))
        except Exception:  # noqa: BLE001
            return 1, 15.0
        if err or not shots:
            return 1, 15.0
        mine = [s for s in shots if s.scene_index.strip("[]").startswith(f"第{ep}集-")] or shots
        return len(mine), float(sum((s.seconds or 15) for s in mine))

    def _refs_need(self, assets_id: str) -> int:
        """渲参考图要几张：角色主形象 + 各套服装 + 场景 + 道具。"""
        try:
            from ..drama import parse_assets

            lib, err = parse_assets(self.assets.content(assets_id))
        except Exception:  # noqa: BLE001
            return 1
        if err:
            return 1
        costumes = sum(len(c.costumes) for c in lib.characters)
        return max(1, len(lib.characters) + costumes + len(lib.scenes) + len(lib.props))

    def _skip(self, key: str, inputs: str = "") -> bool:
        """这一环现在不派：在跑 / 冷却中（等自动再试）/ 失败了且输入没变。

        失败时的输入和现在的不一样（人改了剧本、手动重拆了分镜、换了资产库）→ 失败标记作废，
        重派。之前失败标记只有重启才清，修好了输入也不会重跑（2026-09-23 审查）。
        """
        if key in self._inflight:
            return True
        if time.monotonic() < self._cooldown.get(key, 0.0):
            return True
        if key in self._failed:
            if inputs and self._failed[key] != inputs:
                del self._failed[key]
                self._attempts.pop(key, None)
                return False
            return True
        return False

    # ---------- 视图：从 AssetStore 重建（幂等 + 重启恢复） ----------

    def _scan(self) -> dict[str, Any]:
        scripts: dict[int, tuple[int, str]] = {}
        storyboards: dict[int, tuple[int, str]] = {}
        shots: dict[int, tuple[int, str]] = {}
        rendered: dict[int, tuple[int, str]] = {}
        stamps: dict[str, str] = {}  # 提示词资产 id → 规格戳
        assets_lib = (0, "")
        refs = (0, "")
        # 只看当前项目的 active 资产（缺口 A：之前全库，库里有任何一部剧的资产库，
        # 新剧就不再生成自己的，拿旧剧的脸渲新剧）
        for a in self.assets.find(newest_first=False):
            ep = a.gen_params.get("episode")
            if a.type is AssetType.SCRIPT and ep:
                _keep_latest(scripts, int(ep), a)
            elif a.creator == "tool:drama_storyboard":
                # 一份分镜可能覆盖多集（模型手动跑的全剧分镜）：按集登记，
                # drama_shots 内部按集过滤，兼容整本资产
                for e in self._storyboard_eps(a.id):
                    _keep_latest(storyboards, e, a)
            elif a.creator == "tool:drama_assets":
                if a.seq > assets_lib[0]:
                    assets_lib = (a.seq, a.id)
            elif a.creator == "tool:drama_render_assets":
                if a.seq > refs[0]:
                    refs = (a.seq, a.id)
            elif a.creator == "tool:drama_shots" and ep:
                _keep_latest(shots, int(ep), a)
                stamps[a.id] = str(a.gen_params.get("spec") or "")
            elif a.creator == "tool:drama_render_shots" and ep:
                # 只有完整的一集才算渲完（有失败段 / 被质检门拦下的段 → complete=False）。
                # 之前有索引就算完，缺段的集也打「✅ 完成」（2026-09-23 审查）
                if a.gen_params.get("complete", True):
                    _keep_latest(rendered, int(ep), a)
        latest_shots = {k: v[1] for k, v in shots.items()}
        return {
            "scripts": {k: v[1] for k, v in scripts.items()},
            "storyboards": {k: v[1] for k, v in storyboards.items()},
            "assets_lib": assets_lib[1],
            "refs": refs[1],
            "shots": latest_shots,
            "rendered": {k: v[1] for k, v in rendered.items()},
            # 规格戳对不上的最新提示词：id → 它的旧戳（没有戳的是旧版产物，记空串）
            "stale_shots": {
                sid: stamps.get(sid, "")
                for sid in latest_shots.values()
                if self.spec and stamps.get(sid, "") != self.spec
            },
        }

    def _storyboard_eps(self, asset_id: str) -> list[int]:
        """这份分镜覆盖哪几集（缓存）。"""
        cached = self._sb_eps.get(asset_id)
        if cached is not None:
            return cached
        try:
            eps, err = parse_episodes(self.assets.content(asset_id))
        except KeyError:
            eps, err = [], "gone"
        found = [] if err else [int(e.index) for e in eps]
        self._sb_eps[asset_id] = found
        return found

    def _batches(self, scripts: dict[int, str], total: int) -> dict[int, list[int]]:
        """该派分镜的完整批次。批次按集号划 (N-1)//size，不按到达顺序。"""
        out: dict[int, list[int]] = {}
        for b in range((total + self.batch_size - 1) // self.batch_size):
            members = self._batch_members(b, total)
            if members and all(ep in scripts for ep in members):
                out[b] = members
        return out

    def _batch_members(self, batch: int, total: int) -> list[int]:
        lo = batch * self.batch_size + 1
        hi = min((batch + 1) * self.batch_size, total)
        return list(range(lo, hi + 1))

    def _load_state(self) -> tuple[str, str, int] | None:
        """读 <产物目录>/.drama-state.json：ethnicity / language / totalEpisodes。

        缺了不猜 —— 打一行提示（只打一次），等下一条事件再试。
        """
        path = self.output_prefs.root / _STATE_FILE
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        ethnicity = str(data.get("ethnicity") or "")
        language = str(data.get("language") or "")
        try:
            total = int(data.get("totalEpisodes") or 0)
        except (TypeError, ValueError):
            total = 0
        if ethnicity and language and total > 0:
            self._state_warned = False
            return ethnicity, language, total
        if not self._state_warned:
            self._state_warned = True
            self._say(
                f"⚠ 流水线待命：{path} 里缺 ethnicity/language/totalEpisodes —— "
                "告诉我画面面孔和台词语言（或检查创作状态文件），补齐就接着自动跑"
            )
        return None

    # ---------- 派发（后台任务） ----------

    def _spawn(
        self, key: str, coro: Coroutine[Any, Any, None], sem: asyncio.Semaphore,
        inputs: str = "",
    ) -> None:
        self._inflight.add(key)
        self._inputs[key] = inputs
        self._cooldown.pop(key, None)

        async def runner() -> None:
            async with sem:
                try:
                    await coro
                except Exception as e:  # noqa: BLE001
                    self._fail(key, f"流水线环节 {key} 异常：{e}",
                               retry=key.split(":")[0] in _TEXT_STAGES)
                    await self.bus.emit(EventType.WARNING, message=f"流水线 {key} 异常：{e}")
                else:
                    if key not in self._failed and key not in self._cooldown:
                        self._attempts.pop(key, None)  # 成了：连续失败次数清零
                finally:
                    self._inflight.discard(key)

        t = asyncio.create_task(runner())
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def _storyboard(
        self, ep: int, script_id: str, ethnicity: str, language: str
    ) -> None:
        self._say(f"⚙ 第 {ep} 集剧本到了，开始拆这一集的分镜…")
        r = await self.registry.invoke(
            "drama_storyboard",
            {"script_id": script_id, "ethnicity": ethnicity, "language": language},
        )
        if not r.ok:
            self._fail(f"storyboard:{ep}", f"第 {ep} 集分镜失败：{r.error}", retry=True)
            return
        self._say(f"✓ 第 {ep} 集分镜完成")

    async def _assets_lib(
        self, scripts: dict[int, str], total: int, ethnicity: str, language: str
    ) -> None:
        self._say("⚙ 全剧剧本齐了，开始拆资产库（角色/服装/场景/道具）…")
        ids = [scripts[ep] for ep in range(1, total + 1)]
        r = await self.registry.invoke(
            "drama_assets",
            {"script_ids": ids, "ethnicity": ethnicity, "language": language},
        )
        if not r.ok:
            self._fail("assets", f"资产库拆解失败：{r.error}", retry=True)

    async def _refs(self, assets_id: str) -> None:
        self._say("⚙ 资产库完成，开始渲参考图（几分钟）…")
        r = await self.registry.invoke("drama_render_assets", {"assets_id": assets_id})
        if self._needs_human("refs", r, "渲参考图"):
            return
        if not r.ok:
            self._fail("refs", f"参考图渲染失败：{r.error}")
            return
        self._say("✓ 参考图就绪，各集渲染随提示词齐逐集开始")

    async def _shots(self, ep: int, storyboard_id: str, assets_id: str) -> None:
        r = await self.registry.invoke(
            "drama_shots",
            {"storyboard_id": storyboard_id, "assets_id": assets_id, "episode": ep},
        )
        if not r.ok:
            self._fail(f"shots:{ep}", f"第 {ep} 集提示词失败：{r.error}", retry=True)
            return
        self._say(f"✓ 第 {ep} 集视频提示词完成")

    async def _render(self, ep: int, shots_id: str, refs_id: str) -> None:
        self._say(f"🎬 开始渲染第 {ep} 集视频…")
        r = await self.registry.invoke(
            "drama_render_shots",
            {"shots_id": shots_id, "rendered_id": refs_id, "episode": ep},
        )
        if self._needs_human(f"render:{ep}", r, f"渲第 {ep} 集"):
            return
        if not r.ok:
            self._fail(f"render:{ep}", f"第 {ep} 集渲染失败：{r.error}")
            return
        meta = r.meta or {}
        if meta.get("complete") is False:
            # 有段没生成出来 / 没过质检门：不算完成，也不自动重派（重派会反复花钱），等人处理
            self._fail(
                f"render:{ep}",
                f"第 {ep} 集没渲完整（没生成 {meta.get('failed', 0)} 段，"
                f"质检没过 {meta.get('blocked', 0)} 段），没有拼成片 —— 看一下这几段，"
                "在对话里让我 accept 放行或 redo 重渲",
            )
            return
        self._say(f"✅ 第 {ep} 集.mp4 完成")

    def _needs_human(self, key: str, r: Any, what: str) -> bool:
        """工具挂起要人拍板（比如换生图模型、角色面容冲突）：流水线在后台没法替人答，
        停下这一步、把问题说清楚。之前无视 suspend，问题没人看见（2026-09-23 审查）。"""
        if not getattr(r, "suspend", False):
            return False
        q = str((getattr(r, "suspend_payload", None) or {}).get("question") or "")[:200]
        self._fail(key, f"{what}需要你拍板：{q or r.content[:200]} —— 在对话里告诉我怎么定")
        return True

    # ---------- 小工具 ----------

    def _fail(self, key: str, msg: str, retry: bool = False) -> None:
        """记失败。retry=True（文本环节）：前 text_retries 次隔 retry_delay 秒自动再试；
        之后、以及花钱的环节：标失败，输入变了自动重派，否则等 /auto retry。"""
        n = self._attempts.get(key, 0) + 1
        self._attempts[key] = n
        if retry and n <= self.text_retries:
            self._cooldown[key] = time.monotonic() + self.retry_delay
            self._say(f"⚠ {msg}（{self.retry_delay:.0f} 秒后自动再试一次）")
            with contextlib.suppress(RuntimeError):
                # 晚一点点再踢：事件循环的定时器可能提前一个时钟粒度（Windows 约 15ms）触发，
                # 正好踢在冷却截止之前就会被 _skip 挡掉，之后再没有人踢
                asyncio.get_running_loop().call_later(self.retry_delay + 0.1, self.kick)
            return
        self._failed[key] = self._inputs.get(key, "")
        self._say(
            f"⚠ {msg}（不再自动重试：改好输入会自动重派，或 /auto retry 重派所有失败的环节）"
        )

    def _say(self, msg: str) -> None:
        if self.note is not None:
            self.note(msg)


def _keep_latest(slot: dict[int, tuple[int, str]], key: int, asset: Any) -> None:
    """同一集/同一批可能有多个版本，seq 大的（最新的）赢。"""
    if key not in slot or asset.seq > slot[key][0]:
        slot[key] = (asset.seq, asset.id)

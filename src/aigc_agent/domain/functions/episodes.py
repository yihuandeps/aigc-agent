"""逐集写剧本的 function —— 写作交给子代理，主循环只做编排（2026-09-17）。

之前 60 集剧本是主 Agent 在对话里一集一集写的：每集正文通过 save_draft 的参数
进上下文，一个会话 100 次 save_draft，上下文从 3 万涨到 25 万 token，最后撞上
模型上限；一次「继续」只能写 1–2 集（输出额度被推理吃掉），用户敲了 45 次「继续」。

现在每集由一个**无状态子代理**写：契约只带分集目录里这一集的条目、角色档案、
创作方案要点、上一集的结尾 —— 约 1 万 token，写完直接落资产，主 Agent 只收到
id 和结尾钩子。60 集的 prompt 用量从 1300 万降到 100 万以内。

  drama_write_episode    写一集（返工某一集也用它）
  drama_write_episodes   连续写一批（默认 ≤ 10 集，一批 5 集正好对上按集流水的批次）

产物形态与 save_draft(kind="script", episode=N) 完全一致，按集流水管线照常接续。
"""

from __future__ import annotations

import re
import time
from typing import Any

from ...capabilities.subagents import SubAgentDef, SubAgentRunner
from ...harness.events.bus import EventBus, EventType
from ...harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
    ToolSpec,
)
from ..assets.store import Asset, AssetStore, AssetType
from ..drama.format import DEFAULT_FORMAT, EpisodeFormat, check_script, writing_rules

WRITER_ROLE = "episode_writer"
MAX_BATCH = 10
MIN_EPISODE_CHARS = 300

# 契约里各段的上限（字符）。角色档案是一致性的锚，给得最足。
CHARACTERS_MAX = 8_000
PLAN_MAX = 3_000
PREV_TAIL = 700

_FORMAT_ZH = """# 第{N}集：{集标题}

> 本集关键词：{3个关键词}
> 本集爽点：{爽点类型}
> ⚡ 前15秒高潮点：{一句话：开场第一个场次的前2-3个镜头就是什么冲突顶点/反转/危机}
> 前情提要：{上一集结尾悬念，1-2句}

---

## 场次一

**场景：** 内景/外景 · {地点} · 日/夜
**出场人物：** {人物列表}

△ （全景）{场景描写}

△ （中景）{人物动作描写}

**{角色名}**（{语气/动作指示}）："{台词}"

△ （特写）{关键细节}

♪ 音乐提示：{氛围}

---

## 场次二
…

---

> 🎣 本集钩子：{悬念描述}
> 📺 下集预告：{下一集核心看点，1句}"""

_FORMAT_EN = """# Episode {N}: {Title}

> Key Words: {3 keywords}
> Hook Type: {hook type}
> ⚡ Hook in first 15s: {one sentence: the peak conflict / twist / crisis the episode opens on}
> Previously: {last episode cliffhanger, 1-2 sentences}

---

## Scene 1

**INT./EXT. {LOCATION} - DAY/NIGHT**
**Characters: {character list}**

WIDE SHOT - {scene description}

MEDIUM SHOT - {action description}

**{CHARACTER NAME}** ({tone/action direction}): "{dialogue}"

CLOSE-UP - {key detail}

♪ Music cue: {atmosphere}

---

> 🎣 End Hook: {cliffhanger}
> 📺 Next: {next episode preview}"""

SYSTEM_PROMPT = """你是微短剧编剧。按给定的分集目录条目，写出**这一集**的完整剧本。

只输出剧本正文（markdown），不要解释、不要方案、不要"以下是"之类的引导语。

质量要求：
- 每集 3-6 个场次；字数按契约里「一集的规格」来（少了会被打回扩写，多了会被截）
- 景别提示至少用 3 种（全景 / 中景 / 近景 / 特写）；台词带语气或动作指示
- 结尾必须留悬念钩子；「前情提要」只写 1-2 句，放在高潮点之后回叙，不占开场
- 标了 💰 的付费卡点集，结尾要制造强悬念；标了 🔥 的关键集要有重大转折或揭秘
- 人物言行与角色档案一致；剧情推进与分集目录一致；**不要写到下一集的内容**
- **每一集**开场 15 秒必须是高潮点：第一个场次的前 2-3 个镜头就是冲突顶点、反转或危机现场，
  冷开场直接进入，铺垫后置；标题下一行写「> ⚡ 前15秒高潮点：…」说清是什么

格式严格按下面的模板，标题行必须是「# 第N集：标题」（English 模式为「# Episode N: Title」）。"""

_EP_LINE = re.compile(r"^\s*(?:[-*•]\s*)?(?:\*\*)?第\s*(\d+)\s*集", re.M)
_TITLE = re.compile(r"^#\s*(?:第\s*\d+\s*集\s*[:：]\s*|Episode\s*\d+\s*[:：]\s*)?(.+?)\s*$", re.M)

WRITER_DEF = SubAgentDef(
    name="episode_writer",
    description="按分集目录写一集完整剧本",
    system_prompt=SYSTEM_PROMPT,
    tools=[],  # 纯写作：不碰工具，素材由契约带进来
    allowed=[PermissionLevel.READ],
    role=WRITER_ROLE,
    stateless=True,
    max_iterations=2,
)


def outline_entry(outline: str, n: int) -> tuple[str, str, str]:
    """从分集目录里抽出第 n 集的条目，以及前后集的条目（承接与不越界用）。"""
    lines = outline.splitlines()
    found: dict[int, str] = {}
    for line in lines:
        m = _EP_LINE.match(line)
        if m:
            k = int(m.group(1))
            found.setdefault(k, line.strip())
    return found.get(n, ""), found.get(n - 1, ""), found.get(n + 1, "")


def episode_title(text: str, n: int) -> str:
    for line in text.splitlines():
        if line.startswith("#"):
            m = _TITLE.match(line.strip())
            if m and m.group(1).strip():
                return m.group(1).strip().strip("*")
            break
    return ""


def episode_tail(text: str, limit: int = PREV_TAIL) -> str:
    return text[-limit:].strip() if len(text) > limit else text.strip()


def build_contract(
    *,
    n: int,
    total: int,
    entry: str,
    prev_entry: str,
    next_entry: str,
    characters: str,
    plan: str,
    prev_tail: str,
    note: str,
    mode: str,
    rules: str = "",
) -> str:
    parts = [f"## 本集任务\n写第 {n} 集" + (f"（共 {total} 集）" if total else "") + "。"]
    if rules:
        parts.append("## 一集的规格（硬性）\n" + rules)
    if entry:
        parts.append("分集目录里这一集的条目：\n" + entry)
    else:
        parts.append(f"（分集目录里没找到第 {n} 集的条目，按前后集的走向和创作方案推进，不要跳集）")
    if prev_entry or next_entry:
        parts.append(
            "相邻集（只用来承接和避免越界，不要写它们的内容）：\n"
            + (f"上一集：{prev_entry}\n" if prev_entry else "")
            + (f"下一集：{next_entry}" if next_entry else "")
        )
    if characters:
        parts.append("## 角色档案（言行必须一致）\n" + characters[:CHARACTERS_MAX])
    if plan:
        parts.append("## 创作方案要点（节选）\n" + plan[:PLAN_MAX])
    if prev_tail:
        parts.append("## 上一集结尾（前情提要从这里承接）\n" + prev_tail)
    if note:
        parts.append("## 额外要求\n" + note)
    parts.append("## 输出模板\n" + (_FORMAT_EN if mode == "overseas" else _FORMAT_ZH))
    return "\n\n".join(parts)


class EpisodeFunctions:
    name = "episodes"
    namespaced = False
    disclosure = "full"

    def __init__(
        self,
        runner: SubAgentRunner,
        assets: AssetStore,
        bus: EventBus | None = None,
        fmt: EpisodeFormat | None = None,
    ) -> None:
        self.runner = runner
        self.assets = assets
        self.bus = bus
        self.fmt = fmt or DEFAULT_FORMAT  # 一集 4 分钟 / 开场 15 秒高潮点（config/drama.yaml）
        common = {
            "outline_id": {
                "type": "string",
                "description": "分集目录资产 id；省略则取最新的「分集目录」",
            },
            "characters_id": {
                "type": "string",
                "description": "角色档案资产 id；省略则取最新的「角色档案」",
            },
            "plan_id": {
                "type": "string",
                "description": "创作方案资产 id；省略则取最新的「创作方案」",
            },
            "total": {"type": "integer", "description": "全剧总集数（写进契约，帮子代理把握节奏）"},
            "mode": {
                "type": "string",
                "enum": ["domestic", "overseas"],
                "description": "domestic = 中文剧本格式（默认）；overseas = English 格式",
            },
            "note": {"type": "string", "description": "额外要求（返工时写清楚要改什么）"},
        }
        self._specs = {
            "drama_write_episode": ToolSpec(
                name="drama_write_episode",
                summary="让写作子代理写一集剧本并落资产（也用于返工某一集）",
                permission=PermissionLevel.COMPUTE,
                timeout=900,
                description=(
                    "写单集或返工单集。子代理只看这一集的目录条目、角色档案、方案要点和"
                    "上一集结尾，写完直接存为 script 资产（episode=N，parent 是分集目录），"
                    "你只会收到 id、标题和结尾钩子 —— **不要自己在对话里写正文**。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "episode": {"type": "integer", "description": "集号"},
                        **common,
                        "overwrite": {
                            "type": "boolean",
                            "description": "这一集已有剧本时是否重写（默认 false，直接返回已有的）",
                        },
                    },
                    "required": ["episode"],
                },
            ),
            "drama_write_episodes": ToolSpec(
                name="drama_write_episodes",
                summary="连续写一批剧本（≤10 集），逐集落资产，返回每集 id 与结尾钩子",
                permission=PermissionLevel.COMPUTE,
                timeout=7200,
                description=(
                    "批量出稿用这个：一批 5 集正好对上按集流水的批次。逐集串行写（后一集要"
                    "承接前一集的结尾），已有剧本的集默认跳过。写完一批接着调下一批，"
                    "直到最后一集；中途失败会停下并告诉你写到第几集。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "from_episode": {"type": "integer"},
                        "to_episode": {"type": "integer"},
                        **common,
                        "overwrite": {"type": "boolean", "description": "已有剧本的集也重写"},
                    },
                    "required": ["from_episode", "to_episode"],
                },
            ),
        }

    # ---------- ToolProvider ----------

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail=f"写作子代理 role={WRITER_ROLE}")

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            r = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
            r = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        r.duration_ms = int((time.perf_counter() - started) * 1000)
        return r

    # ---------- 素材定位 ----------

    def _latest_outline(self, keyword: str) -> Asset | None:
        """当前项目里这类文档的真版本（见 AssetStore.best_doc：不取测试桩和占位符）。"""
        return self.assets.best_doc(AssetType.OUTLINE, keyword)

    def _resolve(self, asset_id: str, keyword: str, required: bool) -> tuple[Asset | None, str]:
        if asset_id:
            try:
                return self.assets.get(asset_id), ""
            except KeyError as e:
                return None, str(e)
        a = self._latest_outline(keyword)
        if a is None and required:
            return None, (
                f"没找到「{keyword}」资产：先用 "
                f'save_draft(kind="outline", summary="{keyword}") 存下来，或显式传它的 id'
            )
        return a, ""

    async def _progress(self, done: int, total: int, item: str) -> None:
        if self.bus is not None:
            await self.bus.emit(
                EventType.BATCH_PROGRESS, stage="写剧本", done=done, total=total, item=item
            )

    # ---------- 写一集 ----------

    async def _write_one(
        self,
        n: int,
        *,
        outline: Asset,
        characters: Asset | None,
        plan: Asset | None,
        total: int,
        note: str,
        mode: str,
    ) -> tuple[Asset | None, str]:
        entry, prev_entry, next_entry = outline_entry(self.assets.content(outline.id), n)
        prev = self.assets.episode_assets(n - 1).get("剧本") if n > 1 else None

        def contract_with(extra: str) -> str:
            return build_contract(
                n=n,
                total=total,
                entry=entry,
                prev_entry=prev_entry,
                next_entry=next_entry,
                characters=self.assets.content(characters.id) if characters else "",
                plan=self.assets.content(plan.id) if plan else "",
                prev_tail=episode_tail(self.assets.content(prev.id)) if prev else "",
                note=(note + "\n" + extra).strip() if extra else note,
                mode=mode,
                rules=writing_rules(self.fmt),
            )

        r = await self.runner.run(WRITER_DEF, task=contract_with(""))
        if not r.ok:
            return None, f"第 {n} 集写作失败：{r.error}"
        text = (r.text or "").strip()
        if len(text) < MIN_EPISODE_CHARS:
            return None, f"第 {n} 集写出来的内容太短（{len(text)} 字）：{text[:120]}"
        cost = r.cost or 0.0
        # 规格检查：字数够不够一集 4 分钟、有没有写开场高潮点。太短就让子代理扩写一次
        problems = check_script(text, self.fmt)
        if any("只有" in p or "没有写" in p for p in problems):
            fix = "上一版不合规格：" + "；".join(problems) + (
                "。请按规格重写整集：字数不够就加场次、加冲突层次与对手戏（不要注水），"
                "开场 15 秒必须直接是高潮点并写清「> ⚡ 前15秒高潮点」那一行。"
            )
            r2 = await self.runner.run(WRITER_DEF, task=contract_with(fix))
            text2 = (r2.text or "").strip() if r2.ok else ""
            if text2 and len(check_script(text2, self.fmt)) < len(problems):
                text, problems = text2, check_script(text2, self.fmt)
            cost += (r2.cost or 0.0) if r2.ok else 0.0
        title = episode_title(text, n)
        asset = self.assets.create(
            text,
            type_=AssetType.SCRIPT,
            summary=f"第{n}集·{title}" if title else f"第{n}集",
            parents=[outline.id] + ([prev.id] if prev else []),
            creator="tool:drama_write_episode",
            gen_params={
                "episode": n,
                "role": WRITER_ROLE,
                "iterations": r.iterations,
                "chars": len(text),
                "format_problems": problems,
            },
            gen_cost=cost or None,
        )
        return asset, ""

    @staticmethod
    def _brief(asset: Asset, text: str) -> str:
        hook = ""
        for line in reversed(text.splitlines()):
            if "钩子" in line or "Hook" in line:
                hook = line.strip()
                break
        out = f"{asset.brief()} · {len(text)} 字" + (f"\n  {hook}" if hook else "")
        problems = asset.gen_params.get("format_problems") or []
        if problems:
            out += "\n  ⚠ 规格：" + "；".join(str(p) for p in problems)
        return out

    async def _fn_drama_write_episode(
        self,
        episode: int,
        outline_id: str = "",
        characters_id: str = "",
        plan_id: str = "",
        total: int = 0,
        mode: str = "domestic",
        note: str = "",
        overwrite: bool = False,
    ) -> ToolResult:
        n = int(episode)
        if n <= 0:
            return ToolResult(ok=False, error="episode 要是正整数")
        existing = self.assets.episode_assets(n).get("剧本")
        if existing is not None and not overwrite and not note:
            return ToolResult(
                content=(
                    f"第 {n} 集已有剧本：{existing.brief()}。"
                    "要重写请传 overwrite=true 或 note。"
                ),
                asset_ref=existing.id,
            )
        outline, err = self._resolve(outline_id, "分集目录", required=True)
        if err:
            return ToolResult(ok=False, error=err)
        characters, err = self._resolve(characters_id, "角色档案", required=False)
        if err:
            return ToolResult(ok=False, error=err)
        plan, err = self._resolve(plan_id, "创作方案", required=False)
        if err:
            return ToolResult(ok=False, error=err)

        await self._progress(0, 1, f"第{n}集")
        asset, err = await self._write_one(
            n,
            outline=outline,  # type: ignore[arg-type]
            characters=characters,
            plan=plan,
            total=int(total or 0),
            note=note,
            mode=mode or "domestic",
        )
        await self._progress(1, 1, f"第{n}集")
        if asset is None:
            return ToolResult(ok=False, error=err)
        text = self.assets.content(asset.id)
        return ToolResult(
            content=f"第 {n} 集已写完并落资产：\n{self._brief(asset, text)}",
            asset_ref=asset.id,
        )

    async def _fn_drama_write_episodes(
        self,
        from_episode: int,
        to_episode: int,
        outline_id: str = "",
        characters_id: str = "",
        plan_id: str = "",
        total: int = 0,
        mode: str = "domestic",
        note: str = "",
        overwrite: bool = False,
    ) -> ToolResult:
        a, b = int(from_episode), int(to_episode)
        if a <= 0 or b < a:
            return ToolResult(
                ok=False, error="集号范围不对：from_episode ≥ 1 且 to_episode ≥ from_episode"
            )
        if b - a + 1 > MAX_BATCH:
            return ToolResult(
                ok=False,
                error=f"一次最多写 {MAX_BATCH} 集（给了 {b - a + 1} 集），分批调",
            )
        outline, err = self._resolve(outline_id, "分集目录", required=True)
        if err:
            return ToolResult(ok=False, error=err)
        characters, err = self._resolve(characters_id, "角色档案", required=False)
        if err:
            return ToolResult(ok=False, error=err)
        plan, err = self._resolve(plan_id, "创作方案", required=False)
        if err:
            return ToolResult(ok=False, error=err)

        lines: list[str] = []
        written: list[Asset] = []
        count = b - a + 1
        await self._progress(0, count, f"第{a}集")
        for i, n in enumerate(range(a, b + 1), 1):
            existing = self.assets.episode_assets(n).get("剧本")
            if existing is not None and not overwrite:
                lines.append(f"- 第 {n} 集已有，跳过：{existing.brief()}")
                await self._progress(i, count, f"第{n}集")
                continue
            asset, err = await self._write_one(
                n,
                outline=outline,  # type: ignore[arg-type]
                characters=characters,
                plan=plan,
                total=int(total or 0),
                note=note,
                mode=mode or "domestic",
            )
            await self._progress(i, count, f"第{n}集")
            if asset is None:
                lines.append(f"- ✗ {err}")
                lines.append(f"停在第 {n} 集；修好后从这一集再调（from_episode={n}）。")
                break
            written.append(asset)
            lines.append("- " + self._brief(asset, self.assets.content(asset.id)))
        head = f"这批 {a}–{b} 集：写了 {len(written)} 集"
        done = self.assets.episodes_done()
        if done:
            head += f" · 全剧已有剧本 {len(done)} 集（最新到第 {done[-1]} 集）"
        return ToolResult(
            ok=bool(written) or all("跳过" in ln for ln in lines),
            content="\n".join([head, *lines]),
            asset_ref=written[-1].id if written else None,
            error=None if written or lines else head,
        )

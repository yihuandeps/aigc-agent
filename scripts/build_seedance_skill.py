"""把 awesome-seedance 的模板库生成成本地 skill（2026-09-23，用户要的：本地化 + 融进 Agent）。

上游：https://github.com/LearnPrompt/awesome-seedance —— 25 个从 goodcase.ai 人工核对案例里
蒸馏出来的 Seedance 提示词模板（结构 / 要点 / 常见坑），策划内容 CC BY 4.0。
数据已裁剪后放在 skills/seedance-prompting/data/（见那里的 SOURCE.md），本脚本只做一件事：
把两份 JSON 合并（上游 style-library + 本仓 templates-local + overrides，和上游生成脚本同一口径），
按类别写成中文参考文档，再写一份主文档做目录。

为什么生成而不是手写：上游会更新，重跑一次就同步；手改的东西全部在本脚本里，不在产物里。

用法：python scripts/build_seedance_skill.py
"""

from __future__ import annotations

import datetime
import io
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "seedance-prompting"
DATA = SKILL / "data"
REFS = SKILL / "references"

# 短剧链最先用得上的模板：锚定案例里给这些留提示词原文
DRAMA_FIRST = (
    "timeline-shot-script",
    "character-reference-lock",
    "dialogue-performance-beats",
    "cinematic-narrative-short",
    "handheld-ugc-vlog",
    "product-commercial-shotlist",
)
MAIN_BUDGET = 12_000  # 主文档字符预算，与 drama-script 同一条线（test_skill_references）

DESCRIPTION = (
    "Seedance 2.5 / 2.0 视频提示词写法：25 个从 goodcase.ai 人工核对案例蒸馏出的模板"
    "（结构/要点/常见坑），中文本地化。写 drama_shots 视频提示词、调 gen_video / gen_videos、"
    "用户说「seedance 提示词怎么写」「这段视频怎么描述」「镜头/运镜/对白/口型/参考图锁定/"
    "UGC 真实感/产品广告/动漫风格」时用。"
)

HOW_TO = """## 怎么用（五步）

1. **认意图**：这条片子是哪一类 —— 叙事短片、对白表演、UGC 手持、产品广告、动作、动漫……
   下面目录的「用于」一列扫一遍。真拿不准再问用户（比如「拍我的产品」可能是电影感环绕，
   也可能是 UGC 测评）。
2. **拉对应参考**：`load_skill_reference(skill="seedance-prompting", name="<文件名>")` 只拉那一类，
   全部拉进来约 60K 字，会把能力预算吃光。
3. **按「结构」逐块填**：模板的结构块（全局 / 固定 / 时间轴 / 约束……）一块都别跳，
   跳过的那块就是提示词变空泛的地方。
4. **对着「要点」和「常见坑」过一遍**：它们来自案例里真实成功和真实翻车的写法，不是一般性建议。
5. **产出**：一整段可直接用的提示词；说明用了哪个模板；有锚定案例就给链接。
"""

INTEGRATION = """## 和本 Agent 流水线怎么对接（重要，别重复包）

短剧链 `drama_shots` → `drama_render_shots` 已经在**渲染时**替每段提示词包好了这几层，
模板里的对应内容**不要再写进 description**：

- 参考图身份锁定 → 渲染层 `reference_block`（@图片N 逐张点名 + 不继承背景/姿势/构图）；
  description 里只用资产 ID 引用
- 逐秒时间轴 → `cuts` 字段 + 渲染层 `fast_cut_prompt`（每镜 ≤3s 的时间线）；
  description 按 [切镜：…] 逐镜写就行
- 音色 / 前序片段 → `voice_block`；真实感硬约束与负向清单 → `person_video_prompt` 尾巴；
  禁字幕 → 最外层

所以模板真正要用在 **description 的写法**上：一段一个主动作、反应写成因果链、一拍一个人说话、
每段收在一个结束状态、情绪写行为不写症状、机位术语中英混写、结尾硬切不淡出。
这些已经进了 `drama_shots` 的系统提示词（`SHOTS_SEEDANCE_RULES`），拉参考是为了看细节和案例。

手动 `gen_video` / `gen_videos`（配方短视频、单条广告、UGC）没有这些包装，**要按模板把全局块、
固定块、时间轴块、约束块都写全**，参考图仍走 `image` 参数并在正文里 @图片N 点名。
"""


def zh(v: Any) -> Any:
    """{en, zh, ja} → zh；不是多语言字典就原样返回。"""
    if isinstance(v, dict) and "zh" in v:
        return v["zh"]
    return v


def load() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """合并两份模板源 + overrides，返回 (模板, 类别, 案例索引)。"""
    up = json.loads((DATA / "style-library.json").read_text(encoding="utf-8"))
    loc = json.loads((DATA / "templates-local.json").read_text(encoding="utf-8"))
    cases = json.loads((DATA / "cases-top.json").read_text(encoding="utf-8"))["cases"]

    templates: dict[str, dict[str, Any]] = {t["id"]: dict(t) for t in up["templates"]}
    for t in loc.get("templates", []):
        templates[t["id"]] = dict(t)
    for tid, patch in (loc.get("overrides") or {}).items():
        base = templates.get(tid)
        if base is None:
            continue
        for k, v in patch.items():
            # 语言键字典逐语言合并（title.ja / copyPrompt.zh），其他字段整体覆盖
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = {**base[k], **v}
            else:
                base[k] = v
    cats = list(up.get("categories") or [])
    known = {c["id"] for c in cats}
    for t in templates.values():
        cid = t.get("category")
        if cid not in known:
            cats.append({"id": cid, "title": {"zh": cid}, "description": {"zh": ""}})
            known.add(cid)
    by_case: dict[str, list[dict[str, Any]]] = {}
    for c in cases:
        by_case.setdefault(c["template"], []).append(c)
    for rows in by_case.values():
        rows.sort(key=lambda r: -int(r.get("heatScore") or 0))
    return list(templates.values()), cats, by_case


def header(what: str) -> str:
    return (
        f"<!-- {what}。生成自 skills/seedance-prompting/data/（上游 LearnPrompt/awesome-seedance，"
        f"策划内容 CC BY 4.0，案例出自 goodcase.ai），{datetime.date.today()} 由 "
        "scripts/build_seedance_skill.py 生成 —— 勿手改，改脚本重跑 -->\n\n"
    )


def render_template(t: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    lines = [f"## {zh(t['title'])}", ""]
    lines.append(zh(t.get("description")) or "")
    lines.append("")
    lines.append(f"**适用场景：** {zh(t.get('useWhen')) or '—'}")
    lines.append("")
    lines.append("**结构（按块填，别跳）：**")
    for i, s in enumerate(zh(t.get("structure")) or [], 1):
        lines.append(f"{i}. {s}")
    lines.append("")
    lines.append("**要点（来自真实生效的案例）：**")
    for g in zh(t.get("guidance")) or []:
        lines.append(f"- {g}")
    lines.append("")
    lines.append("**常见坑：**")
    for p in zh(t.get("pitfalls")) or []:
        lines.append(f"- {p}")
    cp = zh(t.get("copyPrompt"))
    if isinstance(cp, str) and cp.strip():
        lines.append("")
        lines.append("**可复制引导语（把【】里的换成你的内容，连同下面这段模板一起发给模型）：**")
        lines.append("")
        lines.append(f"> {cp.strip()}")
    if cases:
        lines += ["", "**锚定案例（按热度）：**"]
        for c in cases[:3]:
            stab = c.get("stabilityScore")
            tail = f"，稳定 {stab}" if stab not in (None, "", "0", 0) else ""
            models = c.get("models") or ""
            if isinstance(models, list):
                models = " / ".join(models)
            lines.append(
                f"- [{c.get('title')}]({c.get('goodcaseUrl')})"
                f"　热度 {c.get('heatScore')}{tail}　{models}"
            )
    lines.append("")
    return "\n".join(lines)


def render_category_file(
    cat: dict[str, Any], ts: list[dict[str, Any]], by_case: dict[str, Any]
) -> str:
    out = [header(f"{zh(cat['title'])} 类模板参考")]
    out.append(f"# {zh(cat['title'])}（{len(ts)} 个模板）\n")
    if zh(cat.get("description")):
        out.append(zh(cat["description"]) + "\n")
    out.append("目录：" + " · ".join(zh(t["title"]) for t in ts) + "\n")
    for t in ts:
        out.append(render_template(t, by_case.get(t["id"], [])))
    return "\n".join(out)


def render_anchor_cases(templates: list[dict[str, Any]], by_case: dict[str, Any]) -> str:
    """短剧链最先用到的几类模板，给热度最高的案例留提示词原文（英文，照抄有效）。"""
    title_of = {t["id"]: zh(t["title"]) for t in templates}
    out = [header("锚定案例：提示词原文")]
    out.append("# 锚定案例（提示词原文）\n")
    out.append(
        "这些是各模板里热度最高、且经人工核对的案例，**提示词原文按原样保留（英文）**——"
        "案例库的规律是机位与结构用英文原词更稳。看它们怎么切段、怎么锁人、怎么写台词节拍，"
        "把写法搬到中文正文里；不要整段照抄进短剧提示词，我们的资产 ID、参考锁定、快切时间线"
        "都由流水线另外包。\n"
    )
    for tid in DRAMA_FIRST:
        rows = (by_case.get(tid) or [])[:2]
        if not rows:
            continue
        out.append(f"## {title_of.get(tid, tid)}\n")
        for c in rows:
            out.append(f"### {c.get('title')}　热度 {c.get('heatScore')}")
            if c.get("summary"):
                out.append(f"\n{c['summary']}\n")
            out.append(f"来源：{c.get('goodcaseUrl')}（作者 {c.get('creator') or '—'}）\n")
            out.append("```text\n" + (c.get("promptFull") or "").strip() + "\n```\n")
    return "\n".join(out)


def render_main(
    templates: list[dict[str, Any]],
    cats: list[dict[str, Any]],
    files: list[tuple[str, str, str]],
) -> str:
    """主文档：是什么、怎么用、怎么和本 Agent 的流水线对接、模板目录、参考资料表。"""
    owner = (
        "生成自 LearnPrompt/awesome-seedance（策划内容 CC BY 4.0，案例出自 goodcase.ai）；"
        f"scripts/build_seedance_skill.py，{datetime.date.today()}"
    )
    fm = (
        "---\n"
        "name: seedance-prompting\n"
        f"description: {DESCRIPTION}\n"
        "scope: content_type\n"
        "applies_to: [短视频]\n"
        "stage: [分镜, 正文]\n"
        "priority: 60\n"
        "version: 1.0.0\n"
        f"owner: {owner}\n"
        "status: active\n"
        "---\n"
    )
    body: list[str] = ["# Seedance 提示词写法（本地化模板库）\n"]
    body.append(
        "25 个模板，每个都是从真实、公开、人工核对过的 Seedance 案例里**倒推出来**的："
        "什么结构真的生效、什么写法真的会翻车。不是泛泛的视频生成常识 —— "
        "有匹配的模板就按模板走，别自己发明结构。\n"
    )
    body.append(HOW_TO)
    body.append(INTEGRATION)
    body.append("## 模板目录（按类别）\n")
    for c in cats:
        ts = [t for t in templates if t.get("category") == c["id"]]
        if not ts:
            continue
        body.append(f"### {zh(c['title'])}\n")
        body.append("| 模板 | 用于 |")
        body.append("|------|------|")
        for t in ts:
            use = (zh(t.get("description")) or "").replace("|", "／").strip()
            body.append(f"| {zh(t['title'])} | {use} |")
        body.append("")
    body.append("## 参考资料\n")
    body.append(
        f"本 skill 带 {len(files)} 篇参考。**按类别拉，不要一次全读。**"
        '用 `load_skill_reference(skill="seedance-prompting", name="<文件名>")`。\n'
    )
    body.append("| 文件 | 用途 | 加载时机 |")
    body.append("|------|------|---------|")
    for fname, purpose, when in files:
        body.append(f"| {fname} | {purpose} | {when} |")
    body.append("")
    body.append("## 出处\n")
    body.append(
        "上游 [LearnPrompt/awesome-seedance](https://github.com/LearnPrompt/awesome-seedance)"
        "（策划内容 CC BY 4.0，代码 MIT），案例出自 goodcase.ai 并经人工核对。"
        "本地副本在 `data/`，重新生成：`python scripts/build_seedance_skill.py`。"
    )
    return fm + "\n" + "\n".join(body) + "\n"


def main() -> None:
    templates, cats, by_case = load()
    REFS.mkdir(parents=True, exist_ok=True)
    for old in REFS.glob("*.md"):
        old.unlink()
    files: list[tuple[str, str, str]] = []
    for i, c in enumerate(cats, 1):
        ts = [t for t in templates if t.get("category") == c["id"]]
        if not ts:
            continue
        fname = f"{i:02d}-{zh(c['title']).replace(' ', '').replace('/', '与')}.md"
        (REFS / fname).write_text(render_category_file(c, ts, by_case), encoding="utf-8")
        names = "、".join(zh(t["title"]) for t in ts)
        files.append((fname, f"{len(ts)} 个模板：{names}", "写这一类片子的提示词时"))
    (REFS / "anchor-cases.md").write_text(
        render_anchor_cases(templates, by_case), encoding="utf-8"
    )
    files.append((
        "anchor-cases.md",
        "短剧最常用的 6 类模板里热度最高案例的提示词原文（英文）",
        "想看真实写法怎么落地时",
    ))
    main_text = render_main(templates, cats, files)
    if len(main_text) >= MAIN_BUDGET:
        sys.exit(f"主文档 {len(main_text)} 字符，超出 {MAIN_BUDGET} 预算 —— 精简目录一列的描述")
    (SKILL / "SKILL.md").write_text(main_text, encoding="utf-8")
    print(f"模板 {len(templates)} 个 / 类别 {len(cats)} 个 / 参考 {len(files)} 篇")
    print(f"SKILL.md {len(main_text)} 字符（预算 {MAIN_BUDGET}）")
    for p in sorted(REFS.glob("*.md")):
        print(f"  references/{p.name}  {p.stat().st_size} B")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()

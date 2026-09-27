"""三段产物的解析。

模型的 JSON 从来不干净，这里的容错都对应用户给的真实样例里出现过的问题：

  · 数组前面多一个全角顿号（`、[{...}]`）
  · 同一个对象里 `episodeTitle` 出现两次（JSON 允许，后者覆盖前者）
  · 正文里混进文档工具的 `[cite: 3, 4]` 引用标记
  · 用 ```json 围栏包起来
  · 该给数组却给了 {"episodes": [...]} 这种包一层的

**解析失败要返回原因，不要静默返回空**。这三步一步错步步错：
资产库空了，第三步就会编造引用；分镜空了，后面全塌。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .models import (
    AssetLibrary,
    Character,
    Costume,
    Episode,
    Named,
    ShotPrompt,
    normalize_costume_locked,
    split_locked,
    strip_cites,
)


def _unwrap(text: str) -> str:
    raw = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if m:
        raw = m.group(1).strip()
    # 数组优先。用户样例里出现过 `、[{...}]` —— 前面多一个顿号。
    i, j = raw.find("["), raw.rfind("]")
    k, ll = raw.find("{"), raw.rfind("}")
    if i >= 0 and j > i and (k < 0 or i < k):
        return raw[i : j + 1]
    if k >= 0 and ll > k:
        return raw[k : ll + 1]
    return raw


_CTRL = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _repair_quotes(raw: str) -> str:
    """修字符串值里没转义的双引号和裸控制字符。

    这是这套流水线**结构性的**故障点，不是偶发：规格要求台词写成
    `+ "完整台词内容"`，而这段文本本身又是 JSON 字符串的值。
    模型不转义内层引号时整个 JSON 就废了 —— 实测同一段剧本，
    跑三次有一次会这样。裸换行同理（该写 \\n 却写了真换行）。

    判据：处在字符串里遇到 `"`，往后看第一个非空白字符；
    是 `, : } ]` 才算真正的收尾引号，否则是内嵌引号，转义掉。
    """
    out: list[str] = []
    in_str = False
    esc = False
    n = len(raw)
    for i, ch in enumerate(raw):
        if esc:
            out.append(ch)
            esc = False
            continue
        if ch == "\\":
            out.append(ch)
            esc = True
            continue
        if ch == '"':
            if not in_str:
                in_str = True
                out.append(ch)
                continue
            j = i + 1
            while j < n and raw[j].isspace():
                j += 1
            nxt = raw[j] if j < n else ""
            # 收尾引号：后面紧跟 , : } ] 或到头；或者换了行、下一行以 " { [ 开头 —— 那是模型在
            # 行尾漏了逗号。2026-09-26 之前把这种当内嵌引号转义掉，整份输出从这里起全乱，
            # 报「字符串没收尾」（9-17 的资产库、9-22 的分镜都是这样废的）
            line_end = nxt in ('"', "{", "[") and "\n" in raw[i + 1 : j]
            if nxt in (",", ":", "}", "]", "") or line_end:
                in_str = False
                out.append(ch)
            else:
                out.append('\\"')  # 内嵌引号，转义
            continue
        if in_str and ch in _CTRL:
            out.append(_CTRL[ch])  # 裸换行/制表符，转义
            continue
        out.append(ch)
    return "".join(out)


# 值的最后一个字符 / 值（或键）的第一个字符。两者之间只隔着空白、没有逗号 = 模型漏了逗号
_VALUE_END = frozenset('"}]0123456789el')  # 字符串 / 对象 / 数组 / 数字 / true false / null
_VALUE_START = frozenset('"{[-0123456789tfn')
_MAX_FIXES = 40


def _fix_at(s: str, e: json.JSONDecodeError) -> str | None:
    """在报错的位置修一处模型常见的 JSON 手误；不是这几种就返回 None（不瞎修）。

    - 漏逗号：`"a": "x"⏎ "b": 1`、`}⏎ {` → 在前一个值后面补逗号
    - 多余的尾逗号：`[1, 2,]`、`{"a": 1,}` → 删掉
    """
    pos = e.pos
    j = pos - 1
    while j >= 0 and s[j].isspace():
        j -= 1
    prev = s[j] if j >= 0 else ""
    cur = s[pos] if pos < len(s) else ""
    if e.msg.startswith("Expecting ',' delimiter") and prev in _VALUE_END and cur in _VALUE_START:
        return s[: j + 1] + "," + s[j + 1 :]
    trailing = e.msg.startswith(("Expecting property name", "Expecting value"))
    if trailing and prev == "," and cur in ("}", "]"):
        return s[:j] + s[j + 1 :]
    return None


def _loads_lenient(s: str) -> tuple[Any, json.JSONDecodeError | None]:
    """json.loads；报的是漏逗号 / 尾逗号就就地修了再试（最多修 40 处）。

    返回 (数据, 第一次的报错)。"""
    first: json.JSONDecodeError | None = None
    for _ in range(_MAX_FIXES + 1):
        try:
            return json.loads(s), None
        except json.JSONDecodeError as e:
            first = first or e
            fixed = _fix_at(s, e)
            if fixed is None:
                break
            s = fixed
    return None, first


def _near(s: str, pos: int, width: int = 60) -> str:
    """报错位置前后各 width 个字，▲ 标在出错处。"""
    a, b = max(0, pos - width), min(len(s), pos + width)
    body = s[a:pos] + "▲" + s[pos:b]
    return ("…" if a else "") + body.replace("\n", "⏎") + ("…" if b < len(s) else "")


def _load(text: str) -> tuple[Any, str]:
    """模型输出 → JSON。依次试：原样；修漏逗号 / 尾逗号；再修没转义的引号和裸换行。

    2026-09-26：20 集的资产库在第 790 行漏了一个逗号，2 分钟的调用整份作废 —— 之前只会修引号。
    """
    raw = _unwrap(text)
    data, e = _loads_lenient(raw)
    if e is None:
        return data, ""
    data, e2 = _loads_lenient(_repair_quotes(raw))
    if e2 is None:
        return data, ""
    return None, f"不是合法 JSON：{e}（出错处附近：{_near(raw, e.pos)}）"


def _str_list(v: Any) -> list[str]:
    """scenes 字段：数组为主，模型偶尔给逗号/顿号分隔的字符串。"""
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [x.strip() for x in re.split(r"[，,、;；/]", v) if x.strip()]
    return []


def _episodes_text(v: Any) -> str:
    """episodes 字段：'1-3' / 3 / [1, 2, 3] 都收，统一成字符串。"""
    if v is None:
        return ""
    if isinstance(v, list):
        nums = [str(x).strip() for x in v if str(x).strip()]
        return ",".join(nums)
    return str(v).strip()


def _rows(data: Any, *keys: str) -> list[dict[str, Any]]:
    """拿到数组。模型有时会包一层 {"episodes": [...]}。"""
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        for k in keys:
            v = data.get(k)
            if isinstance(v, list):
                return [r for r in v if isinstance(r, dict)]
        # 只有一个对象时按单元素处理
        return [data]
    return []


# ---------------------------------------------------------------- ① 分镜脚本


def parse_episodes(text: str) -> tuple[list[Episode], str]:
    data, err = _load(text)
    if err:
        return [], err

    out: list[Episode] = []
    for i, row in enumerate(_rows(data, "episodes", "data"), 1):
        desc = strip_cites(str(row.get("episodeDesc") or "")).strip()
        if not desc:
            continue
        title = str(row.get("episodeTitle") or "").strip()
        try:
            idx = int(row.get("episodeIndex") or i)
        except (TypeError, ValueError):
            idx = i
        # 用户样例里出现过 episodeTitle 被写成 "1" 而不是 "第1集：标题"
        if title.isdigit():
            title = f"第{title}集"
        out.append(Episode(index=idx, title=title or f"第{idx}集", desc=desc))

    if not out:
        return [], "一集都没解析出来（可能缺 episodeDesc 字段）"
    return out, ""


# ---------------------------------------------------------------- ② 资产库


def parse_assets(text: str) -> tuple[AssetLibrary, str]:
    data, err = _load(text)
    if err:
        return AssetLibrary(), err
    if not isinstance(data, dict):
        return AssetLibrary(), "资产库应该是一个对象，含 characters / scenes / props"

    lib = AssetLibrary()

    for row in data.get("characters") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("baseRoleName") or "").strip()
        if not name:
            continue
        body, locked = split_locked(strip_cites(str(row.get("roleTotalDesc") or "")))
        costumes: list[Costume] = []
        for c in row.get("roleCostumeList") or []:
            if not isinstance(c, dict):
                continue
            cname = str(c.get("costumeName") or "").strip()
            if not cname:
                continue
            cbody, clocked = split_locked(strip_cites(str(c.get("costumeDesc") or "")))
            costumes.append(
                Costume(
                    name=cname,
                    body=cbody,
                    locked=normalize_costume_locked(clocked),
                    scenes=_str_list(c.get("scenes")),
                    episodes=_episodes_text(c.get("episodes")),
                )
            )
        voice_keys = ("voice", "voiceDesc", "roleVoice", "音色")
        voice = next((str(row[k]).strip() for k in voice_keys if row.get(k)), "")
        lib.characters.append(
            Character(name=name, body=body, locked=locked, costumes=costumes, voice=voice)
        )

    for key, bucket in (("scenes", lib.scenes), ("props", lib.props)):
        for row in data.get(key) or []:
            if not isinstance(row, dict):
                continue
            n = str(row.get("name") or "").strip()
            d = strip_cites(str(row.get("description") or "")).strip()
            if n and d:
                bucket.append(Named(name=n, desc=d))

    if not lib.characters:
        return lib, "没解析出任何角色 —— 后面两步都依赖角色 ID，不能继续"
    return lib, ""


# ---------------------------------------------------------------- ③ 视频提示词


def parse_shots(text: str) -> tuple[list[ShotPrompt], str]:
    data, err = _load(text)
    if err:
        return [], err

    out: list[ShotPrompt] = []
    for row in _rows(data, "shots", "scenes", "data"):
        desc = strip_cites(str(row.get("description") or "")).strip()
        if not desc:
            continue
        hook_raw = row.get("hook")
        hook = hook_raw is True or str(hook_raw or "").strip().lower() in ("true", "1", "yes", "是")
        out.append(
            ShotPrompt(
                scene_index=str(row.get("scene_index") or "").strip(),
                video_name=str(row.get("video_name") or "").strip(),
                duration=str(row.get("video_duration") or row.get("duration") or "").strip(),
                description=desc,
                hook=hook,
                cuts=_cuts(row.get("cuts")),
                screen_text=_screen_text(row.get("screen_text")),
            )
        )

    if not out:
        return [], "一个镜头提示词都没解析出来"
    return out, ""


def _screen_text(raw: Any) -> list[dict[str, Any]]:
    """一段的上屏字：[{"text", "shot"?, "at"?, "dur"?, "kind"?, "end"?}]。认不出的项丢掉。"""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        row: dict[str, Any] = {"text": text}
        for k in ("shot", "at", "dur"):
            try:
                if item.get(k) is not None:
                    row[k] = float(item[k]) if k != "shot" else int(float(item[k]))
            except (TypeError, ValueError):
                continue
        if item.get("kind"):
            row["kind"] = str(item["kind"])
        if item.get("end"):
            row["end"] = True
        out.append(row)
    return out


def _cuts(raw: Any) -> list[float]:
    """段内各镜头秒数：JSON 数组 [3, 2, 3] 或字符串 "3,2,3" / "3s 2s 3s"。解析不出来 → []。"""
    if raw is None:
        return []
    items: list[Any]
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = re.findall(r"\d+(?:\.\d+)?", str(raw))
    out: list[float] = []
    for it in items:
        try:
            v = float(str(it).strip().rstrip("sS秒"))
        except (TypeError, ValueError):
            continue
        if v > 0:
            out.append(round(v, 2))
    return out


def audit_refs(shots: list[ShotPrompt], lib: AssetLibrary) -> list[str]:
    """查有没有引用资产库里不存在的名字。

    这是第三步最容易出的错：模型把 `(Anne-酒店员工制服-[全集])` 写成
    `(Anne-制服-[1])`，看起来没毛病，但生视频时找不到参考图 ——
    人物就会变脸，而这正是整套流程要解决的问题。
    """
    known = lib.all_names()
    bad: list[str] = []
    for s in shots:
        for r in s.unknown_refs(known):
            bad.append(f"{s.scene_index or s.video_name}：{r}")
    return bad

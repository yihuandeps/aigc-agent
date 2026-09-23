"""短剧流水线的数据结构。

一条贯穿三步的设计：**描述文本分两段**——可编辑正文 + 锁定后缀。

后缀就是资产描述末尾那串方括号里的硬约束（纯白底、单张全身、
16:9、8K 写实……）。它不该让人看见，也不该让人改：
  · 让人看见 → 每个角色卡片都拖着一大段重复的技术参数，没法读
  · 让人改   → 改坏了出来的版式就不对，而生图失败时很难归因

所以存的时候就切开：`body` 给人看给人编，`locked` 由服务端在提交
生图前原样拼回去。前端只拿到 body。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# 锁定后缀的**标准值**。这是服务端持有的版式约束，不是模型输出 ——
# 模型吐不吐都不影响它。实测模型经常漏掉，漏了就没有白底、没有全身版式，
# 而这两样正是参考图能用的前提。所以解析不到就用这里的兜底。
CHARACTER_LOCKED = "[强制白底证件照]"
# 服装图的版式（2026-09-22 用户定）：**纯白底单张全身正面照，不要三视图**。
# 之前是「左 1/3 面部特写 + 右 2/3 正/侧/背」的三视图版式，现在一套服装一张图、
# 一个视角；脸靠参考主形象锁，体型靠全身入镜和「身高占满画面」锁。
COSTUME_LOCKED = (
    "[强制16:9 横版构图，纯白色背景，画面中只有这一个角色、只有一个视角。"
    "角色严格按照参考图片的面孔。单张全身站姿正面照，角色居中、身高占满画面高度，"
    "从头顶到鞋子完整入镜不裁切，服装的款式、剪裁、颜色、材质与褶皱细节清晰可辨。"
    "不要分格、不要拼图、不要并排多张，不要任何文字标注。超高清，超写实 8K，写实]"
)

# 模型会照抄自己见过的旧样例，老资产库里也存着三视图版式的后缀。用户定的是
# 「资产库里不许再出现三视图」—— 光改提示词挡不住，所以解析和拼提示词时都过一道：
# 认出三视图版式就换成上面的标准后缀。
_TURNAROUND_MARKS = ("三视图", "(Front)", "(Profile)", "(Back)", "三个维度", "多视角")


def normalize_costume_locked(locked: str) -> str:
    """服装的锁定后缀：带三视图版式的一律换成标准的单张全身后缀。"""
    t = (locked or "").strip()
    if not t:
        return ""
    return COSTUME_LOCKED if any(m in t for m in _TURNAROUND_MARKS) else t

# 描述末尾的锁定后缀：最后一个 [...] 块。
# 用贪婪匹配到最后一个 '['，因为正文里也会出现方括号（如 [强制白底证件照]
# 之前的视觉锚点描述），只有**最后**那个才是技术约束。
_LOCKED = re.compile(r"\[([^\[\]]*)\]\s*$", re.S)

# 模型偶尔会把文档工具的引用标记抄进来（用户给的样例里就有），
# 它们不是内容，进了提示词还会干扰生图。
_CITE = re.compile(r"\s*\[cite:[^\]]*\]")


def split_locked(text: str) -> tuple[str, str]:
    """切成 (可编辑正文, 锁定后缀)。没有后缀时后半为空。"""
    t = (text or "").strip()
    m = _LOCKED.search(t)
    if not m:
        return t, ""
    return t[: m.start()].rstrip(), m.group(0).strip()


def strip_cites(text: str) -> str:
    """去掉 [cite: 3, 4] 这类文档引用标记。"""
    return _CITE.sub("", text or "")


# ---------------------------------------------------------------- ① 分镜脚本


@dataclass
class Episode:
    index: int
    title: str  # 🟡 节点标题
    desc: str  # 🟢 节点正文（多场景，\n 分段）

    @property
    def shot_count(self) -> int:
        """镜头数 = 以 [ 开头的行数（每行一个镜头），排除场景标头。"""
        n = 0
        for line in self.desc.splitlines():
            s = line.strip()
            if s.startswith("【") and "】" in s:
                s = s.split("】", 1)[1].strip()  # 行首标记（如【高潮点】）不算镜头内容
            if s.startswith("[") and not _is_slug(s):
                n += 1
        return n

    @property
    def scenes(self) -> list[str]:
        """场景标头列表，如 '[夜] [内] [酒店走廊]'。"""
        return [ln.strip() for ln in self.desc.splitlines() if _is_slug(ln.strip())]


def _is_slug(line: str) -> bool:
    """场景标头长这样：[时间] [内外] [地点] —— 三个连续方括号且无中文动作描写。"""
    return bool(re.match(r"^\[[^\]]{1,6}\]\s*\[[内外]\]\s*\[[^\]]+\]\s*$", line))


# ---------------------------------------------------------------- ② 资产库


@dataclass
class Costume:
    name: str  # 🟡 如 Anne-酒店员工制服-[全集]
    body: str  # 🟢 可编辑正文
    locked: str = ""  # 🟣 锁定后缀，不透传前端
    # 这套服装出现在哪些场景（与资产库 scenes 的 name 一致）、哪些集。
    # 2026-09-18 加：之前一个角色只有一套服装，几十个场景一套穿到底；
    # 现在按场景分配，第③步按镜头所在场景选用对应的服装 ID。
    scenes: list[str] = field(default_factory=list)
    episodes: str = ""  # 如 "1-3" / "1,5-8" / "全集"

    def prompt(self) -> str:
        """提交生图时用的完整 prompt = 正文 + 锁定后缀。

        模型没吐后缀（或吐的是旧的三视图版式）就用标准值 —— 这段是服务端的
        版式约束，缺了出来就不是纯白底全身照，参考图也就没法用。
        """
        locked = normalize_costume_locked(self.locked) or COSTUME_LOCKED
        return f"{self.body} {locked}".strip()

    def covers_episode(self, episode: int) -> bool:
        return episode > 0 and any(a <= episode <= b for a, b in episode_ranges(self.episodes))


def episode_ranges(text: str) -> list[tuple[int, int]]:
    """'1-3' → [(1,3)]；'1,5-8' → [(1,1),(5,8)]；'前10集' → [(1,10)]；'全集' → [(1,9999)]。
    解析不出来 → []。"""
    t = (text or "").strip().strip("[]")
    if not t:
        return []
    if any(k in t for k in ("全集", "全部", "all")):
        return [(1, 9999)]
    out: list[tuple[int, int]] = []
    m = re.search(r"前\s*(\d+)\s*集", t)
    if m:
        out.append((1, int(m.group(1))))
    for a, b in re.findall(r"(\d+)\s*[-–~/至到]\s*(\d+)", t):
        lo, hi = int(a), int(b)
        out.append((min(lo, hi), max(lo, hi)))
    stripped = re.sub(r"(\d+)\s*[-–~/至到]\s*(\d+)", " ", t)
    stripped = re.sub(r"前\s*\d+\s*集", " ", stripped)
    for n in re.findall(r"\d+", stripped):
        out.append((int(n), int(n)))
    return out


@dataclass
class Character:
    name: str  # 🟡 baseRoleName
    body: str  # 🟢 roleTotalDesc 的可编辑部分
    locked: str = ""  # 🟣 如 [强制白底证件照]
    costumes: list[Costume] = field(default_factory=list)
    # 音色卡（2026-09-18）：性别 | 年龄感 | 音高与质地 | 语速节奏 | 口音习惯。
    # seedance 按提示词发声，每段视频各自生成 —— 不锁音色，集数一多同一角色的声音就漂。
    voice: str = ""

    def prompt(self) -> str:
        # 「证件照」会把生图模型带向均匀布光的修图脸，换个说法保留白底正面的版式要求
        locked = (self.locked or CHARACTER_LOCKED).replace("证件照", "正面半身照，原图直出无美颜")
        return f"{self.body} {locked}".strip()

    def costume_for(self, scene: str, episode: int = 0) -> Costume | None:
        """这个场景/这一集该穿哪套：场景显式绑定 > 集数覆盖 > None。"""
        if scene:
            for cos in self.costumes:
                if scene in cos.scenes:
                    return cos
        if episode:
            for cos in self.costumes:
                if cos.covers_episode(episode):
                    return cos
        return None


@dataclass
class Named:
    """场景和道具结构相同：名字 + 描述，没有锁定后缀。"""

    name: str  # 🟡
    desc: str  # 🟢

    def prompt(self) -> str:
        return self.desc


@dataclass
class AssetLibrary:
    characters: list[Character] = field(default_factory=list)
    scenes: list[Named] = field(default_factory=list)
    props: list[Named] = field(default_factory=list)

    def digest(self) -> str:
        """喂给第三步的摘要：**只给名字，不给描述**。

        第三步要的是"该引用哪个 ID"，喂全量描述会把上下文撑爆，
        而且模型会把服装细节抄进镜头描述里 —— 那是第二步的活，
        重复描述反而会和参考图打架。
        服装后面带它绑定的场景与集数 —— 第三步据此按场景选服装。
        """
        lines: list[str] = []
        for c in self.characters:
            for cos in c.costumes:
                tag = ""
                if cos.scenes or cos.episodes:
                    where = "、".join(cos.scenes) if cos.scenes else ""
                    when = f"第{cos.episodes}集" if cos.episodes else ""
                    tag = "   // 用于 " + "；".join(x for x in (where, when) if x)
                lines.append(f'  "costumeName": "{cos.name}"{tag}')
        if self.scenes:
            lines.append('  "scenes": [')
            lines += [f'      "name": "{s.name}"' for s in self.scenes]
        if self.props:
            lines.append('  "props": [')
            lines += [f'      "name": "{p.name}"' for p in self.props]
        return "\n".join(lines)

    def all_names(self) -> set[str]:
        """全部可被引用的 ID，用来校验第三步有没有编造引用。"""
        out = {c.name for c in self.characters}
        out |= {cos.name for c in self.characters for cos in c.costumes}
        out |= {s.name for s in self.scenes}
        out |= {p.name for p in self.props}
        return out

    def scene_names(self) -> set[str]:
        return {s.name for s in self.scenes}

    def character_of(self, ref: str) -> Character | None:
        """引用名（角色名或服装名）→ 角色。"""
        for c in self.characters:
            if ref == c.name or any(ref == cos.name for cos in c.costumes):
                return c
        return None

    def wardrobe_matrix(self) -> str:
        """给人看的服装-场景分配表。"""
        lines: list[str] = []
        for c in self.characters:
            lines.append(f"{c.name}：{len(c.costumes)} 套服装")
            for cos in c.costumes:
                where = "、".join(cos.scenes) if cos.scenes else "（未标场景）"
                when = f" · 第{cos.episodes}集" if cos.episodes else ""
                lines.append(f"  - {cos.name} ← {where}{when}")
        return "\n".join(lines)

    @property
    def counts(self) -> str:
        n_cos = sum(len(c.costumes) for c in self.characters)
        return (
            f"{len(self.characters)} 角色 / {n_cos} 套服装 / "
            f"{len(self.scenes)} 场景 / {len(self.props)} 道具"
        )


# ---------------------------------------------------------------- ③ 视频提示词

# description 里的引用：(Anne-酒店员工制服-[全集])、(洛杉矶顶奢酒店走廊)
_REF = re.compile(r"\(([^()]{2,80}?)\)")
# 🔴 视频引入：{第1集-1场}
_CARRY = re.compile(r"\{([^{}]{2,40}?)\}")
# 全角括号里的资产名（提示词示例曾写成全角「（奢华酒店顶层客厅）」，模型照抄 ——
# 场景参考图和按场景换装就全部失效，2026-09-23 审查）
_FULL_REF = re.compile(r"（([^（）]{2,80}?)）")
# 括号里的这些是台词 / 镜头标注，不是资产引用：之前「(OS)」被当成引用，包里找不到就整批拦下
_NOT_REF = (
    "切镜", "L-Cut", "VO", "V.O", "OS", "O.S", "Front", "Profile", "Back",
    "画外音", "旁白", "内心", "独白", "心声", "字幕", "音效", "SFX", "BGM",
)


def normalize_ref_parens(text: str, known: set[str]) -> str:
    """全角括号里是资产库 / 参考图包里的名字 → 换成半角 (名字)。
    不是资产名的全角括号（「陆离（低声）」）不动。"""
    if not text or not known:
        return text

    def sub(m: re.Match[str]) -> str:
        inner = m.group(1).strip()
        return f"({inner})" if inner in known else m.group(0)

    return _FULL_REF.sub(sub, text)


@dataclass
class ShotPrompt:
    scene_index: str  # 🟡 如 [第1集-1场]
    video_name: str  # 镜头范围，如 "1-8"
    duration: str  # 🟣 时长，不展示，仅提交模型时用
    description: str  # 🟢 正文
    hook: bool = False  # 开场高潮点段（2026-09-18：前 15 秒必须有）
    # 段内各镜头的秒数（2026-09-20 用户定的硬性要求：每个镜头 ≤3 秒）。
    # 一段 10–15s 的视频是多镜头快切，cuts 决定渲染时的时间线，也用来做规格检查。
    cuts: list[float] = field(default_factory=list)

    @property
    def seconds(self) -> int:
        """解析 '14s' → 14。解析不出来按 15 兜底（用户规格里定的默认值）。"""
        m = re.search(r"(\d+)", self.duration or "")
        return int(m.group(1)) if m else 15

    @property
    def shot_count(self) -> int:
        """video_name 覆盖了几个分镜脚本镜头：'9-13' → 5，'7' → 1，看不出来 → 0。"""
        m = re.search(r"(\d+)\s*[-–~至到]\s*(\d+)", self.video_name or "")
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return abs(b - a) + 1
        return 1 if re.fullmatch(r"\s*\d+\s*", self.video_name or "") else 0

    def refs(self) -> list[str]:
        """正文里引用到的全部资产名。前端据此显示缩略图。"""
        out: list[str] = []
        for r in _REF.findall(self.description):
            t = r.strip()
            # 切镜标记不是资产引用：(L-Cut Voice-over)、(VO) 之类
            if t and not t.startswith(_NOT_REF):
                out.append(t)
        return list(dict.fromkeys(out))

    def carries(self) -> list[str]:
        """🔴 引入的前序视频片段，如 {第1集-1场}。"""
        return list(dict.fromkeys(m.strip() for m in _CARRY.findall(self.description)))

    def unknown_refs(self, known: set[str]) -> list[str]:
        """引用了资产库里没有的名字 —— 生图时会找不到参考图。"""
        return [r for r in self.refs() if r not in known]


def as_dict(obj: Any) -> dict[str, Any]:
    """给资产落库用。dataclass → dict，嵌套的 costumes 一并展开。"""
    if isinstance(obj, Character):
        return {
            "baseRoleName": obj.name,
            "roleTotalDesc": obj.body,
            "locked": obj.locked,
            "voice": obj.voice,
            "roleCostumeList": [
                {
                    "costumeName": c.name,
                    "costumeDesc": c.body,
                    "locked": c.locked,
                    "scenes": list(c.scenes),
                    "episodes": c.episodes,
                }
                for c in obj.costumes
            ],
        }
    if isinstance(obj, Named):
        return {"name": obj.name, "description": obj.desc}
    if isinstance(obj, Episode):
        return {"episodeIndex": obj.index, "episodeTitle": obj.title, "episodeDesc": obj.desc}
    if isinstance(obj, ShotPrompt):
        return {
            "scene_index": obj.scene_index,
            "video_name": obj.video_name,
            "video_duration": obj.duration,
            "cuts": [float(c) if c != int(c) else int(c) for c in obj.cuts],
            "description": obj.description,
            "hook": obj.hook,
        }
    raise TypeError(f"不知道怎么序列化 {type(obj).__name__}")

"""产线标签（2026-09-23，用户要的）：让用户自己选这次要做哪一类内容。

之前只有两条产线（短视频 / 短剧），入口 `agent new` 问一次就分流到子命令；聊天里靠
系统提示词让模型自己判断。seedance 模板库进来之后，Agent 能做的不止这两类 ——
广告（投放级产品片 / UGC 带货）和设计（海报、封面、图文卡片）都有工具和方法论了。

一条产线 = 一个标签，选定后做三件事：
  1. **skill 预筛**：按 content_type 滤掉不相关的方法论，不进模型视野、不花 token
  2. **路由指引 pin 进上下文**：这条产线该走哪些工具、哪些 skill、先做什么后做什么
  3. **记进会话快照**：重启沿用；/type 随时改，/type off 回到"不限定"

不限定（空）= 老行为：模型按系统提示词自己判断产线，所有 skill 都在目录里。
"""

from __future__ import annotations

from dataclasses import dataclass

LINE_PIN = "content_line"


@dataclass(frozen=True)
class ContentLine:
    key: str  # drama / douyin / ad / design
    label: str  # 面板与菜单里的名字
    # skill frontmatter applies_to 的词汇：短剧 / 抖音 / 广告（都属于「短视频」）/ 图文 / 文案 /
    # 图像。2026-09-23 审查：之前三条视频产线都是「短视频」，预筛只分得出设计线 ——
    # 广告线也带着 douyin-short，被引导先去抓热点
    content_type: str
    note: str  # 菜单里的一句话
    guide: str  # pin 进上下文的路由指引


_DRAMA_GUIDE = (
    "流程：drama_intake 判断用户给的是什么 → 一句话想法先 drama_write 写成剧本（人确认）"
    "→ drama_storyboard 拆分镜 → drama_assets 拆资产库 → drama_shots 出视频提示词 "
    "→ drama_render_assets 渲参考图（自动做面容审查）→ drama_render_shots 渲视频。\n"
    "方法论：drama-script（创作）+ seedance-prompting（提示词写法，拉 04-叙事与表演）。\n"
    "用户自己的角色图：drama_use_local_ref 放进参考图包。"
    "镜头只能走 drama_render_shots，不要用 gen_video 补。"
)

_DOUYIN_GUIDE = (
    "流程：list_video_styles 列风格让用户选（资讯快切 / 口播出镜 / 情绪混剪 / 手持vlog）"
    "→ 抓热点：有关键词先 douyin_hot_list(keyword=…) / xhs_collect(keyword=…)"
    "（douyin_hot_rpa 只有全站热榜，带 keyword 只做筛选）→ short_video_brief 归纳成简报 "
    "→ request_materials 问实拍素材（或 stock_media_search 联网找）→ short_video_produce 出片。\n"
    "方法论：douyin-short；写镜头提示词拉 seedance-prompting 的 02-真实感与UGC。\n"
    "风格是用户确认的，不替他选。"
)

_AD_GUIDE = (
    "走短视频链，风格二选一：**product-ad**（投放级：广告美学词、微距指名拍什么、"
    "英雄帧收尾、文字后期加）或 **ugc-vlog**（创作者手持感、台词焊在动作里）"
    "—— 两者是两个极端，先问用户要哪种，别混。\n"
    "产品图是身份锁：先 host_file 拿到链接，gen_image / gen_video 的 image 参数带上，"
    "正文里 @图片1 点名并写清只继承产品轮廓、材质、盖子、铭牌，不继承背景与构图。\n"
    "上屏文字（slogan / 卖点）不让生成模型画：出图后用 make_poster 叠字，改字免费。\n"
    "方法论：seedance-prompting 的 03-商业与产品（含 UGC 口播测评带货）；"
    "配方在 config/recipes。投放级成片出来必须给用户审。"
)

_DESIGN_GUIDE = (
    "流程：gen_image 出**不带任何文字**的底图（比例按用途：封面 3:4、竖版海报 9:16、方图 1:1）"
    "→ make_poster 叠标题 / 要点 / 品牌名（模板 clean / bold / card，自动打 AIGC 角标）"
    "→ view_image 看一眼再给用户。\n"
    "不要让生图模型画中文，它画不对，改一个字就得重生成；字都本地叠。\n"
    "多张要风格一致：固定一段风格词 + 同一张参考图（image 参数）+ 同一个生图模型。\n"
    "方法论：poster-prompt-handbook。"
)

LINES: tuple[ContentLine, ...] = (
    ContentLine(
        key="drama",
        label="短剧",
        content_type="短剧",
        note="带剧情对白的连续剧：剧本 → 分镜 → 资产 → 参考图 → 逐镜生成，角色跨集一致",
        guide=_DRAMA_GUIDE,
    ),
    ContentLine(
        key="douyin",
        label="抖音短视频",
        content_type="抖音",
        note="30 秒左右：先确认风格 → 抓热点归纳 → 写文案分镜 → 补实拍素材 → 出片",
        guide=_DOUYIN_GUIDE,
    ),
    ContentLine(
        key="ad",
        label="广告",
        content_type="广告",
        note="产品广告：投放级产品片（product-ad）或 UGC 带货感（ugc-vlog），产品图当身份锁",
        guide=_AD_GUIDE,
    ),
    ContentLine(
        key="design",
        label="设计",
        content_type="图像",
        note="海报 / 封面 / 图文卡片：生图出不带字的底图，本地叠标题与要点，改字免费",
        guide=_DESIGN_GUIDE,
    ),
)

_BY_KEY = {ln.key: ln for ln in LINES}
_ALIASES = {
    "1": "drama",
    "2": "douyin",
    "3": "ad",
    "4": "design",
    "短剧": "drama",
    "drama": "drama",
    "剧": "drama",
    "抖音短视频": "douyin",
    "抖音": "douyin",
    "短视频": "douyin",
    "douyin": "douyin",
    "video": "douyin",
    "广告": "ad",
    "ad": "ad",
    "ads": "ad",
    "产品广告": "ad",
    "带货": "ad",
    "设计": "design",
    "design": "design",
    "海报": "design",
    "封面": "design",
    "图文": "design",
}
_OFF = {"", "0", "off", "none", "无", "不限", "不限定", "清除", "自动", "auto"}


def get_line(key: str) -> ContentLine | None:
    return _BY_KEY.get((key or "").strip().lower())


def parse_line(text: str) -> ContentLine | None:
    """用户敲的字 → 产线：序号、key、中文名、带"的"的口语都认。认不出返回 None。"""
    t = (text or "").strip().strip("。.").lower()
    if t in _ALIASES:
        return _BY_KEY[_ALIASES[t]]
    for alias, key in _ALIASES.items():
        if len(alias) >= 2 and alias in t:
            return _BY_KEY[key]
    return None


def is_off(text: str) -> bool:
    return (text or "").strip().lower() in _OFF


def menu() -> str:
    return "\n".join(f"{i}  {ln.label}\n   {ln.note}" for i, ln in enumerate(LINES, 1))


def line_block(line: ContentLine) -> str:
    """pin 进上下文的路由指引。用户已经选了产线，模型不要再问"做哪种内容"。"""
    return (
        f"## 当前产线：{line.label}（用户选定，不要再问做哪种内容；要换用户会说）\n"
        f"{line.guide}"
    )

"""真实感预设 —— 消掉 AI 味儿，但不把人做丑。

**所有会生成真人画面的链路都挂这个**：短剧资产、短剧分镜、角色护照分镜、
短视频配方。人一旦看出"这是 AI 生成的"，内容就废了一半，而暴露点高度集中
在三处：

  1. **皮肤太干净** —— 模型默认输出磨皮级的完美皮肤，没有毛孔、没有纹理。
  2. **光太柔** —— 默认给蝴蝶光/三点棚拍光，均匀、无硬边、无死角。
  3. **光学上太完美** —— 真镜头有色差和色散，一尘不染的成像本身就是"合成"的信号。

⚠️ 这套和短剧资产提示词原本的「明星美学干预 / 去瑕疵化指令」是**相反**的：
   那边明令禁止 freckles、要求皮肤"像剥壳鸡蛋"、强制柔和蝴蝶光。
   开了真实感预设就会把那几条换掉 —— 两套审美不能同时生效。

2026-09-18 两次校正：
  · 早上：7 张主形象的脸没听这套（提示词里写了毛孔雀斑，出来还是磨皮脸）。改成三道
    保险：硬约束段放人物提示词**开头**、描述先清洗、生成后视觉校验不合格重生成。
  · 下午：用户看成片，雀斑和皱纹**太明显、人物太丑**，要求减到原来的三分之一。
    所以真实感分了**档**（realism_level）：
      subtle  默认。真实但干净：细看有毛孔和纹理，雀斑至多零星几点极淡，不要明显皱纹
      natural 早上那版的力度（少量雀斑、可见毛孔、肤色不均）
      strong  明显做旧（老人、病人、流浪者这类人设才用）
    档位统一从 media_models.yaml 的 drama.realism_level 读；描述里写得过重的瑕疵
    （粗大毛孔、满脸雀斑、深深的皱纹）在 subtle 档会被 soften_flaws 压回去。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

DEFAULT_LEVEL = "subtle"
LEVELS = ("subtle", "natural", "strong")


def norm_level(level: str | None) -> str:
    lv = str(level or "").strip().lower()
    return lv if lv in LEVELS else DEFAULT_LEVEL


# ---------------------------------------------------------------- 人物

_PERSON = {
    "subtle": (
        "原图模式（无美颜、无磨皮、无滤镜）。"
        "皮肤保留自然纹理：细看有毛孔与极细的纹理、轻微油光，整体干净健康、状态好。"
        "雀斑与斑点只允许零星几点、极淡（远看几乎看不见），不要明显的皱纹、痘印或成片色斑；"
        "带轻微颗粒感，不要磨平。"
        "**严禁**磨皮、瓷娃娃皮肤、塑料质感、过度对称的五官。"
    ),
    "natural": (
        "原图模式（无美颜、无磨皮、无滤镜）。"
        "皮肤保留自然纹理：可见毛孔、细微油光、轻微的肤色不均匀。"
        "面部加入少量雀斑与颗粒感，不要磨平。"
        "**严禁**磨皮、瓷娃娃皮肤、塑料质感、过度对称的五官。"
    ),
    "strong": (
        "原图模式（无美颜、无磨皮、无滤镜）。"
        "皮肤纹理明显：清晰的毛孔、油光、肤色不均，"
        "面部有较多雀斑、清晰的细纹与颗粒感，不要磨平。"
        "**严禁**磨皮、瓷娃娃皮肤、塑料质感、过度对称的五官。"
    ),
}

# ---------------------------------------------------------------- 光

_LIGHT = {
    "subtle": (
        "布光以**侧光与局部光**为主，脸上有自然的明暗层次和阴影落点；"
        "个别镜头可用硬光或闪光灯直闪，但不要生硬夸张。"
        "**不要**均匀柔光、蝴蝶光、影棚三点光那种四平八稳的打法 —— 那是 AI 味儿的主要来源。"
    ),
    "natural": (
        "布光以**侧光与局部光**为主，让脸上有明确的明暗交界和阴影落点；"
        "部分镜头用硬光或闪光灯直闪，允许生硬的高光和清晰的投影边缘。"
        "**不要**均匀柔光、蝴蝶光、影棚三点光那种四平八稳的打法 —— 那是 AI 味儿的主要来源。"
    ),
    "strong": (
        "布光以**侧光与硬光**为主，明暗交界清晰、阴影浓重；多用闪光灯直闪与局部光，"
        "允许生硬的高光和清晰的投影边缘。"
        "**不要**均匀柔光、蝴蝶光、影棚三点光。"
    ),
}

# ---------------------------------------------------------------- 光学

_OPTICS = {
    "subtle": (
        "保留轻微的真实镜头光学特征：极轻的色差与色散，高反差边缘偶有一点红蓝溢出（紫边/青边），"
        "画面带细微感光颗粒。不要一尘不染的数字成像。"
    ),
    "natural": (
        "保留真实镜头的光学瑕疵：轻微色差与色散，高反差边缘有少量红蓝溢出（紫边/青边），"
        "画面带细微感光颗粒。不要一尘不染的数字成像。"
    ),
    "strong": (
        "保留明显的真实镜头光学瑕疵：可见的色差与色散，高反差边缘有红蓝溢出（紫边/青边），"
        "画面带明显感光颗粒。不要一尘不染的数字成像。"
    ),
}

PERSON = _PERSON[DEFAULT_LEVEL]
LIGHT = _LIGHT[DEFAULT_LEVEL]
OPTICS = _OPTICS[DEFAULT_LEVEL]

# ---------------------------------------------------------------- 硬约束（放开头）

# 中英双语、短句、祈使式；末尾那句"忽略下文相反要求"是关键 ——
# 模型写的人物描述里常混着"精致""光滑"，不声明优先级就会各听一半。
_HARD_HEAD = {
    "subtle": (
        "【硬约束，优先级高于下文所有描述】真实照片质感、未经修饰，但不要刻意做旧：皮肤细看有毛孔与"
        "极细纹理，整体干净、状态好；雀斑或斑点至多零星几点且极淡，不要明显的皱纹、痘印、成片色斑；"
        "五官自然不对称；光线有方向感和明暗层次（侧光为主，可少量硬光），不要影棚柔光；"
        "轻微颗粒与极轻色差。严禁磨皮、美颜、滤镜、瓷感或塑料感皮肤。"
        "下文若出现光滑、无瑕、精致、柔光、对称等要求，一律忽略；"
        "若出现粗大毛孔、明显皱纹、大片斑点等要求，也一律按轻微处理。"
        " RAW unretouched photo of an attractive, healthy-looking person: fine pores and subtle "
        "skin texture visible on close inspection, otherwise clean skin; at most a few faint "
        "freckles, no pronounced wrinkles, blemishes or blotchy patches; natural facial asymmetry; "
        "directional side light with soft shadow gradation (a little hard light is fine), "
        "no glossy studio glamour lighting; subtle film grain and a hint of chromatic aberration. "
        "No beauty retouching, no skin smoothing, no porcelain or plastic skin. "
        "Ignore any request below for flawless, smooth or perfectly symmetrical features, and tone "
        "down any request for heavy freckles, deep wrinkles or blotchy skin."
    ),
    "natural": (
        "【硬约束，优先级高于下文所有描述】原图直出的真实照片，未经修饰：面部皮肤必须可见毛孔、"
        "细纹与少量雀斑或斑点，肤色略不均匀，五官自然不对称；侧向硬光，明暗交界清晰；"
        "轻微色差与感光颗粒。严禁磨皮、美颜、滤镜、瓷感或塑料感皮肤、影棚柔光。"
        "下文若出现光滑、无瑕、精致、柔光、对称等要求，一律忽略。"
        " RAW unretouched photograph: visible pores, fine lines, a few freckles or blemishes, "
        "uneven skin tone, natural facial asymmetry; hard directional side light with crisp shadow "
        "edges; slight chromatic aberration and fine film grain. No beauty retouching, no skin "
        "smoothing, no airbrushing, no porcelain or plastic skin, no soft studio glamour lighting. "
        "Ignore any request below for flawless, smooth or perfectly symmetrical features."
    ),
    "strong": (
        "【硬约束，优先级高于下文所有描述】原图直出、明显未修饰的真实照片：面部皮肤有清晰的毛孔、"
        "细纹与较多雀斑或斑点，肤色不均匀，五官明显不对称；硬光，明暗交界锐利；"
        "可见色差与感光颗粒。严禁磨皮、美颜、滤镜、瓷感或塑料感皮肤、影棚柔光。"
        "下文若出现光滑、无瑕、精致、柔光、对称等要求，一律忽略。"
        " RAW unretouched photograph with pronounced skin texture: clearly visible pores, fine "
        "lines, many freckles or blemishes, uneven skin tone, marked facial asymmetry; hard light "
        "with crisp shadows; visible chromatic aberration and film grain. No beauty retouching, "
        "no skin smoothing, no porcelain or plastic skin, no soft studio glamour lighting. "
        "Ignore any request below for flawless, smooth or perfectly symmetrical features."
    ),
}
HARD_HEAD = _HARD_HEAD[DEFAULT_LEVEL]

# 视频版放末尾（画面描述先讲清楚发生了什么），比图片版短
_HARD_TAIL_VIDEO = {
    "subtle": (
        "Style: RAW unretouched footage, natural clean skin with subtle texture (fine pores on "
        "close-up, at most a few faint freckles, no heavy wrinkles or blemishes); natural "
        "asymmetry; directional light with gentle shadow gradation; subtle film grain that shifts "
        "every frame; no beauty filter, no skin smoothing, no glossy glamour lighting."
    ),
    "natural": (
        "Style: RAW unretouched footage, not a beauty shot — skin shows pores, freckles and uneven "
        "tone, natural asymmetry; hard side light with crisp shadows; film grain that shifts every "
        "frame; no beauty filter, no skin smoothing, no soft glamour lighting."
    ),
    "strong": (
        "Style: RAW unretouched footage with pronounced skin texture — clearly visible pores, "
        "freckles, fine lines and uneven tone; marked asymmetry; hard side light with crisp "
        "shadows; visible film grain that shifts every frame; no beauty filter, no skin smoothing."
    ),
}
HARD_TAIL_VIDEO = _HARD_TAIL_VIDEO[DEFAULT_LEVEL]

# 重生成时加在最前面。三个方向：上一版磨皮了（要质感）/ 做旧过头了（要减）/ 光太柔（要层次）
_RETRY = {
    "smooth": {
        "subtle": (
            "【第二次生成】上一版面部被磨得过于光滑、像美颜后的塑料脸，属于不合格。这次要有真实皮肤"
            "质感：近看可见毛孔与细微纹理、自然的肤色深浅，但仍要干净好看，不要加重雀斑和皱纹。 "
            "SECOND ATTEMPT — previous output was over-retouched. Skin MUST show fine pores and "
            "subtle natural texture on close inspection while staying clean and attractive; "
            "do not add heavy freckles or wrinkles."
        ),
        "natural": (
            "【第二次生成】上一版面部被磨得过于光滑、看不到毛孔与斑点，属于不合格。这次必须明显呈现："
            "鼻翼与颧骨处清晰的毛孔、至少几处雀斑或小色斑、眼下细纹、肤色不均，"
            "以及一侧脸上明确的硬光阴影。皮肤质感要像未修图的高清人像原片。 "
            "SECOND ATTEMPT — previous output was over-retouched. Skin MUST show clearly "
            "visible pores, several freckles or small blemishes, fine lines under the eyes and "
            "uneven tone, with a hard shadow on one side of the face. Unretouched "
            "high-resolution RAW portrait look."
        ),
        "strong": (
            "【第二次生成】上一版面部被磨得过于光滑，属于不合格。这次皮肤纹理必须非常明显："
            "清晰的毛孔、大量雀斑与色斑、细纹、肤色不均，硬光阴影浓重。 "
            "SECOND ATTEMPT — previous output was over-retouched. Skin MUST show heavy visible "
            "texture: clear pores, many freckles and blemishes, fine lines, uneven tone, "
            "hard shadows."
        ),
    },
    "heavy": {
        lv: (
            "【第二次生成】上一版面部瑕疵过重（雀斑、斑点或皱纹太明显，显老、显脏），属于不合格。"
            "这次把瑕疵减到原来的三分之一：皮肤整体干净健康、状态好，只保留细看可见的毛孔与极淡的纹理，"
            "雀斑至多零星几点，不要明显皱纹与痘印，人要好看。 "
            "SECOND ATTEMPT — previous output was over-aged and blotchy. Reduce imperfections to "
            "about a third: clean healthy attractive skin with only fine pores and faint texture, "
            "at most a few faint freckles, no visible wrinkles or blemishes."
        )
        for lv in LEVELS
    },
    "light": {
        lv: (
            "【第二次生成】上一版是影棚均匀柔光、脸上没有明暗层次，属于不合格。这次用有方向感的侧光，"
            "脸上要有明确的明暗过渡和阴影落点，不要蝴蝶光和三点光。 "
            "SECOND ATTEMPT — previous output had flat studio glamour lighting. Use directional "
            "side light with clear shadow gradation on the face; no butterfly or three-point "
            "lighting."
        )
        for lv in LEVELS
    },
}
RETRY_BOOST = _RETRY["smooth"][DEFAULT_LEVEL]


# ---------------------------------------------------------------- 未成年角色（2026-09-23）

# 上面那套（attractive / RAW / 毛孔 / 雀斑 / 硬光）是给成年人设计的。套到儿童角色上，既不该这么
# 描写孩子，也正好撞上生图服务的儿童护栏 —— 8 岁女主的主形象被拒，她的服装图全部连带跳过
# （2026-09-23 审查，因果属推测，但这套写法本来就不该用在孩子身上）。
_MINOR_WORDS = (
    "萌宝", "小孩", "孩子", "儿童", "幼童", "女童", "男童", "小女孩", "小男孩", "女孩", "男孩",
    "婴儿", "宝宝", "小学生", "初中生", "未成年", "少年", "少女",
    "child", "kid", "toddler", "baby", "teen", "schoolgirl", "schoolboy",
)
_AGE = re.compile(r"(\d{1,3})\s*(?:岁|周岁|years?[\s-]*old)", re.I)

_CHILD_HEAD = (
    "【硬约束，优先级高于下文所有描述】自然、健康、符合年龄的儿童真实照片：日常着装、神态自然，"
    "皮肤按真实儿童的样子干净自然（不刻意描写毛孔、雀斑、皱纹这类瑕疵），柔和的日常自然光；"
    "不要成人化的妆容、发型、服装或姿态，不要美颜滤镜与塑料感。"
    " Natural, wholesome, age-appropriate candid photo of a child: everyday clothes, natural "
    "expression, clean natural skin (no emphasis on pores, freckles or wrinkles), soft everyday "
    "daylight; no makeup, no mature styling, clothing or poses; no beauty filter, no plastic look."
)
_CHILD_RETRY = (
    "【第二次生成】上一版不像真实照片（塑料感、过度修饰或光线不自然），这次更像日常随手拍的真实"
    "儿童照片，自然光、神态自然。 SECOND ATTEMPT — make it look like a natural, everyday candid "
    "photo of a child; natural light and expression."
)
_CHILD_SUFFIX = "日常自然光，画面带极细微的感光颗粒，像真实照片而不是渲染图。"
_CHILD_TAIL_VIDEO = (
    "Style: natural, wholesome, age-appropriate footage of the child; everyday clothes; natural "
    "daylight; no makeup or mature styling; no beauty filter; subtle film grain."
)
_CHILD_TARGET = (
    "目标是「自然、符合年龄的儿童真实照片」：干净自然、不像塑料、没有成人化的妆容和打扮。"
)


def is_minor(text: str) -> bool:
    """人物描述里的这个人是不是未成年。按**外表**算：「外表 8 岁（实际元神 1400 年）」按孩子画。
    写了年龄就以第一个年龄为准，没写再看「萌宝 / 孩子 / 少年」这类词。"""
    t = text or ""
    m = _AGE.search(t)
    if m:
        return int(m.group(1)) < 18
    low = t.lower()
    return any(w in low for w in _MINOR_WORDS)


def image_suffix(level: str = "") -> str:
    """生图时追加的真实感段落（人物图用）。"""
    lv = norm_level(level)
    return f"{_PERSON[lv]} {_LIGHT[lv]} {_OPTICS[lv]}"


def skin_suffix(level: str = "") -> str:
    """只要皮肤真实感，不管布光和光学 —— 手机随拍类配方用：真实感段里的「侧光 / 闪光灯直闪 /
    感光颗粒」和「自然光、环境本来的颜色、iPhone 随手拍」正面冲突（2026-09-23 审查）。"""
    return _PERSON[norm_level(level)]


def video_suffix(level: str = "") -> str:
    """生视频时追加的真实感段落。

    比生图那版多一句动态颗粒 —— 静止的颗粒在视频里会像脏镜头，
    要求它随帧变化才像胶片。
    """
    return f"{image_suffix(level)} 颗粒随帧轻微浮动，不要静止的固定噪点图案。"


_FLAW_RULE = {
    "subtle": (
        "· 写人物外貌时，只写**一处轻微**的真实细节（如：鼻翼细看有毛孔、颧骨极淡的一两点小雀斑、"
        "笑起来眼角有浅浅纹路），程度要克制：远看是干净好看的脸，近看才有质感。"
        "严禁写粗大毛孔、明显皱纹、大片斑点、疤痕、痘印这类显丑的瑕疵（人设需要除外，"
        "如老人、病人、流浪者）；也严禁写「皮肤光滑 / 无瑕 / 精致 / 柔光 / 高度对称」。"
    ),
    "natural": (
        "· 写人物外貌时，必须写进至少两处**具体的**皮肤或面部瑕疵"
        "（如：鼻翼两侧毛孔粗大、颧骨一片淡淡晒斑、左眉尾一道浅疤、眼下青黑与细纹）。"
        "笼统的「皮肤真实」不算，生图模型只认具体描写；"
        "严禁写「皮肤光滑 / 无瑕 / 精致 / 柔光 / 高度对称」。"
    ),
    "strong": (
        "· 写人物外貌时，必须写进至少三处**明显的**皮肤或面部瑕疵"
        "（如：满脸雀斑、深深的法令纹、粗大毛孔、眼下浓重的青黑）。"
        "严禁写「皮肤光滑 / 无瑕 / 精致 / 柔光 / 高度对称」。"
    ),
}


def prompt_rules(level: str = "") -> str:
    """喂给"写提示词的模型"的规则段（第①②③步那种）。

    和 image_suffix 的区别：那个是直接拼到生图 prompt 尾巴上的成品文本，
    这个是让模型在**写描述时**就按这些原则去写。
    """
    lv = norm_level(level)
    return (
        "【真实感准则】写人物和画面时必须遵守，这几条是消 AI 味儿的关键：\n"
        f"· 皮肤：{_PERSON[lv]}\n"
        f"· 光线：{_LIGHT[lv]}\n"
        f"· 成像：{_OPTICS[lv]}\n"
        f"{_FLAW_RULE[lv]}"
    )


# ---------------------------------------------------------------- 描述清洗与瑕疵锚点

# 模型写描述时溜进来的美化措辞 → 替换成真实感说法。逐词替换，其余原样保留。
_BEAUTY_SWAPS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"(皮肤|肌肤)(细腻|光滑|无瑕|白皙无瑕|吹弹可破|零毛孔|如剥壳鸡蛋|像剥壳鸡蛋)"),
        r"\1有自然纹理与可见毛孔",
    ),
    (re.compile(r"(光滑|无瑕|零毛孔)(的)?(皮肤|肌肤)"), "有自然纹理与可见毛孔的皮肤"),
    (re.compile(r"磨皮|美颜|滤镜感|精修"), "原图直出"),
    (re.compile(r"(柔和的?蝴蝶光|蝴蝶光|影棚(级)?三点光|三点布光|均匀柔光|柔光)"), "侧向硬光"),
    (re.compile(r"(高度|完美|绝对)对称(的)?(五官|面部|脸)"), "自然略不对称的\\3"),
    (re.compile(r"(完美|精致|无瑕)(的)?(五官|面容|脸庞)"), "端正但有真实瑕疵的\\3"),
    (
        re.compile(
            r"\b(flawless|porcelain|airbrushed|retouched|smooth skin|glamour lighting)\b", re.I
        ),
        "unretouched, textured skin",
    ),
    (re.compile(r"\b(perfectly|highly) symmetrical\b", re.I), "naturally asymmetrical"),
]

# 反方向：描述里把瑕疵写得过重（早上那版规则要求"至少两处具体瑕疵"，模型就往狠里写），
# subtle 档把它们压回轻微 —— 用户看成片说雀斑皱纹太明显、人物太丑（2026-09-18）。
_HEAVY_SWAPS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"粗大(的)?毛孔|毛孔粗大|毛孔明显|明显(的)?毛孔"), "细看可见的毛孔"),
    (
        re.compile(r"(布满|密布|满脸|满面|大片|一片|成片|遍布|大量|密集)(的)?(雀斑|色斑|晒斑|斑点|痘印|老年斑)"),
        r"零星几点极淡的\3",
    ),
    (
        re.compile(r"(明显|清晰|浓重|深深|深刻|很深|粗重|密集)(的)?(皱纹|细纹|法令纹|抬头纹|鱼尾纹|川字纹|干纹)"),
        r"浅浅的\3",
    ),
    (re.compile(r"(皱纹|细纹|法令纹)(密布|遍布|纵横|深刻)"), r"几道浅浅的\1"),
    (
        re.compile(r"(明显|清晰|浓重|深深|大片|一片)(的)?(雀斑|色斑|晒斑|斑点|痘印|黑眼圈|眼袋|青黑)"),
        r"淡淡的\3",
    ),
    (re.compile(r"痘坑|痤疮|坑坑洼洼(的)?(皮肤|肌肤)?|皮肤坑洼"), "轻微的肤色不均"),
    (re.compile(r"(粗糙|干燥|干裂|蜡黄|暗沉|油腻)(的)?(皮肤|肌肤)"), "有自然纹理的皮肤"),
    (
        re.compile(
            r"\b(heavy|dense|pronounced|deep|many) (freckles|wrinkles|blemishes|acne scars)\b",
            re.I,
        ),
        r"a few faint \2",
    ),
    (re.compile(r"\bpockmarks?\b|\bacne\b", re.I), "slightly uneven skin tone"),
]

# 描述里有这些词之一，就算已经写了具体瑕疵
IMPERFECTION_MARKERS = (
    "毛孔", "雀斑", "斑", "疤", "皱纹", "细纹", "痘", "黑眼圈", "青黑", "晒斑", "色斑",
    "纹理", "粗糙", "瑕", "不对称", "不均", "绒毛", "胡茬", "干纹",
    "pore", "freckle", "blemish", "wrinkle", "scar", "uneven", "asymmetr", "stubble",
)

_DEFAULT_IMPERFECTIONS = {
    "subtle": (
        "面部真实质感：细看可见毛孔与极细的纹理，肤色自然略有深浅，整体干净好看；五官轻微不对称"
    ),
    "natural": (
        "面部真实质感：毛孔可见，鼻翼与颧骨处有细小斑点，肤色略不均匀，"
        "眼下有淡淡的细纹与暗沉，五官轻微不对称"
    ),
    "strong": (
        "面部真实质感：毛孔清晰，鼻翼与颧骨有较多雀斑与色斑，肤色不均，眼下有细纹与青黑，"
        "五官明显不对称"
    ),
}
DEFAULT_IMPERFECTIONS = _DEFAULT_IMPERFECTIONS[DEFAULT_LEVEL]


def sanitize_beauty(text: str) -> tuple[str, list[str]]:
    """把描述里的美化措辞换成真实感说法。返回 (新文本, 被替换的原词)。"""
    out = text or ""
    hits: list[str] = []
    for pat, repl in _BEAUTY_SWAPS:
        for m in pat.finditer(out):
            hits.append(m.group(0))
        out = pat.sub(repl, out)
    return out, hits


def soften_flaws(text: str) -> tuple[str, list[str]]:
    """把描述里写得过重的瑕疵压回轻微。返回 (新文本, 被压下去的原词)。"""
    out = text or ""
    hits: list[str] = []
    for pat, repl in _HEAVY_SWAPS:
        for m in pat.finditer(out):
            hits.append(m.group(0))
        out = pat.sub(repl, out)
    return out, hits


def has_imperfections(text: str) -> bool:
    low = (text or "").lower()
    return any(k in low for k in IMPERFECTION_MARKERS)


def ensure_imperfections(text: str, level: str = "") -> tuple[str, bool]:
    """描述里没写具体瑕疵就补一句默认的。返回 (文本, 是否补了)。"""
    if has_imperfections(text):
        return text, False
    sep = "" if not text or text.endswith(("。", "；", ";", "|", " ")) else "；"
    return f"{text}{sep}{_DEFAULT_IMPERFECTIONS[norm_level(level)]}", True


def _prepare_body(core: str, level: str) -> tuple[str, list[str]]:
    body, hits = sanitize_beauty(core)
    notes = [f"替换美化措辞：{'、'.join(dict.fromkeys(hits))}"] if hits else []
    if level == "subtle":
        body, heavy = soften_flaws(body)
        if heavy:
            notes.append(f"压轻过重的瑕疵：{'、'.join(dict.fromkeys(heavy))}")
    return body, notes


def person_image_prompt(
    core: str, retry: bool | str = False, level: str = "", minor: bool | None = None
) -> tuple[str, list[str]]:
    """人物生图的完整提示词：硬约束在前、描述居中、真实感段在后。

    retry：False/"" 首次；True 或 "smooth" = 上一版磨皮了；"heavy" = 做旧过头；"light" = 光太柔。
    minor：未成年角色（None = 按描述自动判）→ 儿童安全写法：不写皮肤瑕疵、不成人化。
    返回 (提示词, 处理记录)：清洗替换了哪些词、压轻了哪些瑕疵、有没有补锚点。
    """
    lv = norm_level(level)
    if minor is None:
        minor = is_minor(core)
    if minor:
        body, _ = sanitize_beauty(core)
        head = (_CHILD_RETRY + " " if retry else "") + _CHILD_HEAD
        note = "未成年角色：儿童安全写法（不写皮肤瑕疵、不成人化）"
        return f"{head}\n{body}\n{_CHILD_SUFFIX}", [note]
    body, notes = _prepare_body(core, lv)
    body, injected = ensure_imperfections(body, lv)
    if injected:
        notes.append("描述没写具体瑕疵，已补默认锚点")
    direction = "smooth" if retry is True else (retry if isinstance(retry, str) else "")
    boost = _RETRY.get(direction, {}).get(lv, "") if direction else ""
    head = (boost + " " if boost else "") + _HARD_HEAD[lv]
    return f"{head}\n{body}\n{image_suffix(lv)}", notes


def person_video_prompt(core: str, level: str = "", minor: bool = False) -> str:
    """人物视频提示词：画面描述在前，真实感段与硬约束尾巴在后。
    minor：这段里有未成年角色 → 儿童安全的尾巴（不强调皮肤瑕疵与硬光）。"""
    lv = norm_level(level)
    body, _ = _prepare_body(core, lv)
    if minor:
        return f"{body} {_CHILD_TAIL_VIDEO}"
    return f"{body} {video_suffix(lv)} {_HARD_TAIL_VIDEO[lv]}"


# ---------------------------------------------------------------- 生成后校验（视觉模型）

REALISM_ROLE = "realism_check"

_TARGET = {
    "subtle": (
        "目标是「真实但干净」：像状态很好的真人高清原片，细看有毛孔和细微纹理，"
        "但整体好看，不显老、不显脏，雀斑皱纹不明显。"
    ),
    "natural": "目标是「未修图的真人原片」：可见毛孔、少量雀斑或斑点、肤色略不均，侧光有明暗交界。",
    "strong": "目标是「明显未修饰、有岁月感的真人原片」：清晰毛孔、较多雀斑与细纹、硬光。",
}

_CHECK_PROMPT = (
    "这是一张 AI 生成的人物参考图。{target}\n"
    "请只看面部与皮肤，判断属于哪种情况（issue_type）：\n"
    "· smooth：明显磨皮/美颜/瓷感塑料感，看不到毛孔和纹理（不合格）\n"
    "· heavy：做旧过头 —— 雀斑、斑点、皱纹、痘印过于明显，显老、显脏或显丑（不合格）\n"
    "· light：影棚均匀柔光/蝴蝶光，脸上没有明暗层次（不合格）\n"
    "· ok：符合目标（合格）\n"
    "给 0–10 的分数（10 = 完全符合目标）。6 分以上算通过。\n"
    '只输出 JSON：{{"pass": true, "score": 7, "issue_type": "ok", "issues": ["问题一句话"]}}'
)


def realism_check_messages(
    image_url: str, level: str = "", minor: bool = False
) -> list[dict[str, Any]]:
    """给视觉模型的校验消息。image_url 可以是 data URL（本地文件）或 https 链接。"""
    target = _CHILD_TARGET if minor else _TARGET[norm_level(level)]
    text = _CHECK_PROMPT.format(target=target)
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
            ],
        }
    ]


@dataclass
class RealismVerdict:
    passed: bool = True
    score: int = -1
    issues: list[str] = field(default_factory=list)
    direction: str = ""  # smooth / heavy / light / ""（合格或判不出来）


def parse_realism_report(text: str) -> RealismVerdict:
    """解析校验结果，容忍 ```json 围栏与前后解释。解析不出来按通过处理 ——
    校验挂了不该把生图链路一起挂掉，但会在报告里标出来。"""
    raw = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if m:
        raw = m.group(1).strip()
    i, j = raw.find("{"), raw.rfind("}")
    if i < 0 or j <= i:
        return RealismVerdict(True, -1, ["校验输出不是 JSON"])
    try:
        data = json.loads(raw[i : j + 1])
    except json.JSONDecodeError:
        return RealismVerdict(True, -1, ["校验输出不是合法 JSON"])
    try:
        score = int(float(data.get("score", -1)))
    except (TypeError, ValueError):
        score = -1
    passed = data.get("pass")
    if not isinstance(passed, bool):
        passed = score >= 6 if score >= 0 else True
    issues = [str(x) for x in (data.get("issues") or []) if str(x).strip()]
    direction = str(data.get("issue_type") or "").strip().lower()
    if direction not in ("smooth", "heavy", "light"):
        # 老格式没有 issue_type：从问题描述里猜方向，猜不到按"磨皮"处理（早上那版的默认）
        joined = " ".join(issues)
        if any(k in joined for k in ("过重", "太明显", "显老", "显脏", "做旧", "heavy")):
            direction = "heavy"
        elif any(k in joined for k in ("柔光", "蝴蝶光", "三点光", "flat light")):
            direction = "light"
        else:
            direction = "smooth"
    if passed:
        direction = ""
    return RealismVerdict(passed, score, issues, direction)


def parse_realism_verdict(text: str) -> tuple[bool, int, list[str]]:
    v = parse_realism_report(text)
    return v.passed, v.score, v.issues


# 和真实感预设冲突的老规则 —— 开预设时用右边替换左边。
# 留在这里而不是直接删，是为了让"为什么改了你的提示词"这件事可追溯。
_SKIN_RULE = {
    # 2026-09-23 审查：之前三档都写「必须保留少量雀斑」，和 subtle 档「雀斑至多零星几点」打架
    "subtle": (
        "皮肤真实度：保留细看可见的毛孔与细微纹理，雀斑至多零星几点极淡，年轻角色不写皱纹。"
        "严禁磨皮、瓷化、塑料感。眼窝按人物年龄自然呈现，不刻意填平。未成年角色不写皮肤瑕疵。"
    ),
    "natural": (
        "皮肤真实度：**必须**保留毛孔、细微纹理、少量雀斑 (freckles) 与颗粒感。"
        "严禁磨皮、瓷化、塑料感。眼窝按人物年龄自然呈现，不刻意填平。未成年角色不写皮肤瑕疵。"
    ),
    "strong": (
        "皮肤真实度：**必须**保留清晰的毛孔、细纹、较多雀斑 (freckles) 与颗粒感。"
        "严禁磨皮、瓷化、塑料感。眼窝按人物年龄自然呈现。未成年角色不写皮肤瑕疵。"
    ),
}

_FLAWLESS_RULE = (
    '禁止项：严禁出现"皱纹 (wrinkles)"、"青筋 (veins)"、"血丝 (bloodshot)"、'
    '"痣 (moles/freckles)"以及"深陷的眼窝 (deep-set eyes)"。'
)

CONFLICTS: list[tuple[str, str]] = [
    (_FLAWLESS_RULE, _SKIN_RULE["natural"]),
    (
        '面部光影：强制使用"柔和的蝴蝶光 (Butterfly Lighting)"或"影棚级三点光 '
        '(Studio 3-point lighting)"，以消除面部多余的阴影坑洞。',
        "面部光影：以**侧光 (side lighting)** 与**局部光 (practical/spot light)** 为主，"
        "保留明确的明暗交界与阴影。部分镜头可用**硬光 (hard light)** 或"
        "**闪光灯直闪 (direct flash)**。不要均匀柔光 —— 阴影是真实感的来源，不是缺陷。",
    ),
    (
        '五官比例：强制加入"高度对称的五官 (Highly symmetrical features)"和'
        '"极其精细的皮肤纹理 (Ultra-fine skin texture)"，确保皮肤像剥壳鸡蛋一样有质感而非干瘪。',
        "五官比例：五官端正但**不要高度对称** —— 真人脸都有轻微不对称，完全对称是 AI 特征。"
        "皮肤要有**真实纹理**（毛孔、细纹、轻微不均匀），不是剥壳鸡蛋那种光滑。",
    ),
]


def apply_conflicts(template: str, level: str = "") -> str:
    """把和真实感预设冲突的老规则换掉（皮肤那条按档位换）。

    逐条替换而不是整段重写：原提示词里其他约束（族裔锁定、微表情注入、
    骨骼结构干预）都还要留着，那些和真实感不冲突。
    """
    out = template
    for old, new in CONFLICTS:
        if old == _FLAWLESS_RULE:
            new = _SKIN_RULE[norm_level(level)]
        if old in out:
            out = out.replace(old, new)
    return out

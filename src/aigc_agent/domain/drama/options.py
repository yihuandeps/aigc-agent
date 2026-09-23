"""短剧的两个必选项：面孔族裔 与 台词语言。

**为什么必须问、不能默认**：

族裔上，原始 system prompt 里写死了「禁止出现亚裔」，并强制 Caucasian /
Nordic / Latino / African-American。那是为出海短剧调的，对国内片就是错的 ——
而且一旦生成，角色形象、全部分镜视频都跟着走，返工成本极高。

语言上，seedance 是**按提示词里的台词原文发声**的。提示词里写英文就出英文，
写中文就出中文。默认"跟剧本走"看似安全，但用户经常拿中文剧本做出海内容，
静默按中文生成等于白跑一整条链。

所以这两项**没给就停下来问**，不猜。
"""

from __future__ import annotations

from dataclasses import dataclass

# 族裔选项。value 会被直接写进角色提示词的 [族裔] 位，所以用模型认得的英文术语。
ETHNICITIES: dict[str, tuple[str, str]] = {
    "asian": ("East Asian", "东亚面孔（中/日/韩）"),
    "chinese": ("Han Chinese", "中国面孔"),
    "caucasian": ("Caucasian", "欧美白人面孔"),
    "african": ("African-American", "非裔面孔"),
    "latino": ("Latino", "拉丁裔面孔"),
    "mixed": ("", "不限定，由剧本背景决定"),
}

LANGUAGES: dict[str, tuple[str, str]] = {
    "zh": ("中文", "台词用中文，画面里说中文"),
    "en": ("English", "台词用英文，画面里说英文"),
    "keep": ("", "跟剧本原文，不做转换"),
}


@dataclass(frozen=True)
class DramaOptions:
    ethnicity: str = ""  # ETHNICITIES 的 key
    language: str = ""  # LANGUAGES 的 key

    @property
    def ready(self) -> bool:
        return bool(self.ethnicity) and bool(self.language)

    @property
    def missing(self) -> list[str]:
        out = []
        if not self.ethnicity:
            out.append("ethnicity")
        if not self.language:
            out.append("language")
        return out

    @property
    def ethnic_term(self) -> str:
        return ETHNICITIES.get(self.ethnicity, ("", ""))[0]

    @property
    def lang_term(self) -> str:
        return LANGUAGES.get(self.language, ("", ""))[0]

    def brief(self) -> str:
        e = ETHNICITIES.get(self.ethnicity, ("", self.ethnicity))[1]
        ln = LANGUAGES.get(self.language, ("", self.language))[1]
        return f"{e} · {ln}"


def normalize(ethnicity: str, language: str) -> DramaOptions:
    """接受 key、英文术语或中文说法，统一成 key。"""
    return DramaOptions(_match(ethnicity, ETHNICITIES), _match(language, LANGUAGES))


def _match(raw: str, table: dict[str, tuple[str, str]]) -> str:
    v = (raw or "").strip().lower()
    if not v:
        return ""
    if v in table:
        return v
    for key, (term, note) in table.items():
        if v == term.lower() or v in note.lower() or (term and term.lower() in v):
            return key
    # 常见别名
    alias = {
        "亚洲": "asian", "亚裔": "asian", "东亚": "asian", "asia": "asian",
        "中国": "chinese", "华人": "chinese", "cn": "chinese",
        "欧美": "caucasian", "白人": "caucasian", "western": "caucasian", "white": "caucasian",
        "黑人": "african", "非裔": "african",
        "拉丁": "latino",
        "不限": "mixed", "auto": "mixed", "任意": "mixed",
        "中文": "zh", "chinese": "zh", "zh-cn": "zh", "普通话": "zh",
        "英文": "en", "english": "en", "英语": "en",
        "原文": "keep", "不变": "keep",
    }
    return alias.get(v, "")


def ask_text() -> str:
    """没给选项时返回给调用方/模型的提问文本。

    写得足够具体，让人能直接照着回答，而不是再来回问一轮。
    """
    e = "\n".join(f"    {k:<10} {note}" for k, (_, note) in ETHNICITIES.items())
    ln = "\n".join(f"    {k:<10} {note}" for k, (_, note) in LANGUAGES.items())
    return (
        "开始生成前需要先定两件事 —— 它们会贯穿角色形象和全部分镜视频，"
        "生成之后再改等于整条链重跑。\n\n"
        f"1. 画面里的面孔（ethnicity）：\n{e}\n\n"
        f"2. 台词语言（language）：\n{ln}\n\n"
        "请用户明确这两项后再调一次本工具。"
    )

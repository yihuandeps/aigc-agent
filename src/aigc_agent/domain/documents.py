"""Office 文档抽文本（2026-09-20）—— 用户的剧本、人物小传常是 .docx，之前 fs_read 只认纯文本。

纯标准库实现：docx / pptx / xlsx 都是「zip + XML」，不引入 python-docx 之类的依赖
（venv 里没有，装了也只是多一个出错点）。.doc / .pdf 这类抽不出来的，直接告诉用户
怎么转，不装作读到了。
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

DOC_EXT = {".docx", ".pptx", ".xlsx"}
UNSUPPORTED_EXT = {
    ".doc": "旧版 Word（.doc）",
    ".wps": "WPS 文字（.wps）",
    ".pdf": "PDF",
    ".ppt": "旧版 PowerPoint（.ppt）",
    ".xls": "旧版 Excel（.xls）",
}

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_SLIDE = re.compile(r"^ppt/slides/slide(\d+)\.xml$")
_SHEET = re.compile(r"^xl/worksheets/sheet(\d+)\.xml$")


def is_document(path: Path) -> bool:
    return path.suffix.lower() in DOC_EXT


def extract_text(path: Path) -> tuple[str, str]:
    """返回 (文本, 格式名)。读不了的格式抛 ValueError，信息可直接给用户看。"""
    ext = path.suffix.lower()
    if ext in UNSUPPORTED_EXT:
        raise ValueError(
            f"{path.name} 是{UNSUPPORTED_EXT[ext]}，抽不出文字：用 Word / WPS 另存为 .docx"
            "（PDF 先复制成 .txt / .md）再读"
        )
    if ext not in DOC_EXT:
        raise ValueError(f"{path.name} 不是支持的文档格式（支持 docx / pptx / xlsx）")
    try:
        with zipfile.ZipFile(path) as z:
            if ext == ".docx":
                return _docx(z), "docx"
            if ext == ".pptx":
                return _pptx(z), "pptx"
            return _xlsx(z), "xlsx"
    except (zipfile.BadZipFile, ET.ParseError, KeyError) as e:
        raise ValueError(f"{path.name} 打不开（文件损坏或不是真正的 {ext[1:]}）：{e}") from e


def _read_xml(z: zipfile.ZipFile, member: str) -> ET.Element | None:
    if member not in z.namelist():
        return None
    return ET.fromstring(z.read(member))


def _para_text(
    p: ET.Element, text_tag: str, tab_tag: str = "", br_tags: tuple[str, ...] = ()
) -> str:
    parts: list[str] = []
    for node in p.iter():
        if node.tag == text_tag:
            parts.append(node.text or "")
        elif tab_tag and node.tag == tab_tag:
            parts.append("\t")
        elif node.tag in br_tags:
            parts.append("\n")
    return "".join(parts)


def _docx(z: zipfile.ZipFile) -> str:
    root = _read_xml(z, "word/document.xml")
    if root is None:
        raise KeyError("缺 word/document.xml")
    lines = [
        _para_text(p, _W + "t", _W + "tab", (_W + "br", _W + "cr"))
        for p in root.iter(_W + "p")
    ]
    return _tidy("\n".join(lines))


def _pptx(z: zipfile.ZipFile) -> str:
    slides = sorted(
        ((int(m.group(1)), name) for name in z.namelist() if (m := _SLIDE.match(name))),
    )
    out: list[str] = []
    for no, name in slides:
        root = ET.fromstring(z.read(name))
        paras = [_para_text(p, _A + "t") for p in root.iter(_A + "p")]
        body = "\n".join(t for t in paras if t.strip())
        out.append(f"## 第 {no} 页\n{body}".rstrip())
    return _tidy("\n\n".join(out))


def _xlsx(z: zipfile.ZipFile) -> str:
    shared: list[str] = []
    ss = _read_xml(z, "xl/sharedStrings.xml")
    if ss is not None:
        for si in ss.iter(_S + "si"):
            shared.append("".join(t.text or "" for t in si.iter(_S + "t")))
    sheets = sorted(
        ((int(m.group(1)), name) for name in z.namelist() if (m := _SHEET.match(name))),
    )
    out: list[str] = []
    for no, name in sheets:
        root = ET.fromstring(z.read(name))
        rows: list[str] = []
        for row in root.iter(_S + "row"):
            cells: list[str] = []
            for c in row.iter(_S + "c"):
                kind = c.get("t") or ""
                if kind == "inlineStr":
                    cells.append("".join(t.text or "" for t in c.iter(_S + "t")))
                    continue
                v = c.find(_S + "v")
                raw = (v.text or "") if v is not None else ""
                if kind == "s" and raw.isdigit() and int(raw) < len(shared):
                    cells.append(shared[int(raw)])
                elif kind == "b":
                    cells.append("TRUE" if raw == "1" else "FALSE")
                else:
                    cells.append(raw)
            if any(cell.strip() for cell in cells):
                rows.append("\t".join(cells).rstrip())
        out.append(f"## 工作表 {no}\n" + "\n".join(rows))
    return _tidy("\n\n".join(out))


def _tidy(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

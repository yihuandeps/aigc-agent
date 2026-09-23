#!/usr/bin/env python3
"""
把 Markdown 分析报告渲染为美观 PDF。

用法:
    python3 render_pdf.py <input.md> <output.pdf>

依赖:
    - python3 markdown (pip3 install --user markdown)
    - Google Chrome (or Microsoft Edge / Chromium / Arc 任一)
"""
import sys
import os
import subprocess
import tempfile
import shutil
from pathlib import Path

import markdown

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


SKILL_DIR = Path(__file__).resolve().parent.parent
CSS_PATH = SKILL_DIR / "assets" / "report.css"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
{content}
</body>
</html>
"""


def find_chrome() -> str:
    """找一个可用的 Chromium 系浏览器。"""
    import os
    local = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        # Windows
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.join(local, r"Google\Chrome\Application\chrome.exe") if local else "",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        # macOS
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Arc.app/Contents/MacOS/Arc",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    ]
    for p in candidates:
        if p and Path(p).is_file():
            return p
    raise FileNotFoundError(
        "没找到 Chrome/Edge/Chromium/Arc/Brave。请安装其中之一，或自行指定路径。"
    )


def md_to_html(md_text: str) -> str:
    md = markdown.Markdown(
        extensions=[
            "tables",
            "fenced_code",
            "attr_list",
            "def_list",
            "smarty",
            "sane_lists",
        ],
        output_format="html5",
    )
    return md.convert(md_text)


def extract_title(md_text: str) -> str:
    for line in md_text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return "爆款视频分析报告"


def render(md_path: Path, pdf_path: Path) -> None:
    md_text = md_path.read_text(encoding="utf-8")
    title = extract_title(md_text)
    content = md_to_html(md_text)

    css_text = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""
    html = HTML_TEMPLATE.format(title=title, css=css_text, content=content)

    # 写到临时目录（Chrome 要绝对路径）
    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        f.write(html)
        html_path = Path(f.name)

    try:
        chrome = find_chrome()
        cmd = [
            chrome,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path.resolve()}",
            html_path.resolve().as_uri(),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            print("Chrome stderr:", result.stderr, file=sys.stderr)
            raise RuntimeError("PDF 渲染失败")
        print(
            f"✅ 已生成: {pdf_path}  ({pdf_path.stat().st_size / 1024:.1f} KB)"
        )
    finally:
        html_path.unlink(missing_ok=True)


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    md_path = Path(sys.argv[1]).expanduser().resolve()
    pdf_path = Path(sys.argv[2]).expanduser().resolve()
    if not md_path.exists():
        print(f"找不到输入文件: {md_path}", file=sys.stderr)
        sys.exit(1)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    render(md_path, pdf_path)


if __name__ == "__main__":
    main()

"""给视觉模型看的图，一律先转成 data URL（2026-09-29 审查 1.2）。

视觉模型拉不到不少图床的外链（9-21 一个会话就报了 8 次 unsupported image url），本地路径
更不能当链接发（面容审查因此从没真正比对过）。统一在这里转：
  · 已经是 data URL → 原样；
  · 本地文件 → 读出来转；
  · http(s) 链接 → 先走代理下载到内存，再转；
  · 都拿不到 → 返回原因。调用方按「没查成」算，不能当成「查过、通过」。
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

import httpx2 as httpx

from ...harness.model.media import default_proxy

DOWNLOAD_TIMEOUT = 60.0
# 再大的图不发：base64 之后请求体太大，视觉接口多半直接拒（4K 的 PNG 也就十来 MB）
MAX_BYTES = 20 * 1024 * 1024


def data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def guess_mime(data: bytes, name: str = "", header: str = "") -> str:
    """按文件名后缀 → 响应头 → 文件头猜图片类型；都猜不出按 jpeg。"""
    by_name = mimetypes.guess_type(name)[0] if name else None
    if by_name and by_name.startswith("image/"):
        return by_name
    ctype = header.split(";", 1)[0].strip().lower()
    if ctype.startswith("image/"):
        return ctype
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/jpeg"


def file_data_url(path: Path | str) -> tuple[str, str]:
    """本地图片 → (data URL, 读不了的原因)。"""
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as e:
        return "", f"读不了 {p.name}（{type(e).__name__}）"
    if not data:
        return "", f"{p.name} 是空文件"
    return data_url(data, guess_mime(data, p.name)), ""


async def fetch_image(url: str) -> tuple[bytes, str, str]:
    """下载一张图 → (字节, 图片类型, 失败原因)。走和生成接口一样的代理。"""
    try:
        async with httpx.AsyncClient(
            timeout=DOWNLOAD_TIMEOUT, proxy=default_proxy(), follow_redirects=True
        ) as c:
            resp = await c.get(url)
    except Exception as e:  # noqa: BLE001 — 下载失败只是这次没查成，不能把调用方挂掉
        return b"", "", f"{type(e).__name__}: {e}"[:120]
    if resp.status_code >= 400:
        return b"", "", f"HTTP {resp.status_code}"
    data = resp.content
    if not data:
        return b"", "", "下载到 0 字节"
    if len(data) > MAX_BYTES:
        return b"", "", f"图片太大（{len(data) // (1024 * 1024)}MB）"
    header = resp.headers.get("content-type", "")
    ctype = header.split(";", 1)[0].strip().lower()
    # 过期链接常常回 200 加一张 HTML / JSON 错误页，当成图发出去就是一次白付的视觉调用
    if ctype and not ctype.startswith(("image/", "application/octet-stream", "binary/")):
        return b"", "", f"链接返回的不是图片（{ctype}），可能已经过期"
    return data, guess_mime(data, Path(urlparse(url).path).name, header), ""


async def image_data_url(source: str) -> tuple[str, str]:
    """本地路径 / http 链接 / data URL → (data URL, 拿不到的原因)。"""
    s = (source or "").strip()
    if not s:
        return "", "没有图片（既没有本地副本也没有链接）"
    if s.startswith("data:"):
        return s, ""
    if s.startswith(("http://", "https://")):
        data, mime, why = await fetch_image(s)
        if why:
            return "", f"图片链接下载不下来（{why}）"
        return data_url(data, mime), ""
    return await asyncio.to_thread(_local_data_url, s)


def _local_data_url(s: str) -> tuple[str, str]:
    try:
        p = Path(s).expanduser()
        is_file = p.is_file()
    except (OSError, ValueError, RuntimeError):  # 路径里有非法字符 / 用户目录展不开
        is_file = False
    if is_file:
        return file_data_url(p)
    return "", f"不是本地文件也不是链接：{s[:60]}"

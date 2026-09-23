# ruff: noqa: ASYNC240 — 同 video_edit.py：落盘是小文件/分块写，紧邻网络 IO，量小
"""产物目录偏好 —— 生成内容（文本/图片/视频）的用户指定落盘文件夹。

默认是**用户打开 Agent 的那个文件夹**（进程当前工作目录，见 default_root）；
同名 session 记住，`/out` 随时改。所有消费点「缺省走 prefs，显式传参优先」。

下载器可注入：测试不碰网络，生产走 httpx + 代理（与国内网络环境适配，
和 media.py 的 default_proxy 同一个口径）。
"""

from __future__ import annotations

import os
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx2 as httpx

from ..harness.model.media import default_proxy

# (url, 目标路径) -> 是否成功
Downloader = Callable[[str, Path], Awaitable[bool]]


def default_root(workspace: Path, session_id: str = "", project_root: Path | None = None) -> Path:
    """默认产物目录 = **用户打开的那个文件夹**（进程的当前工作目录）。

    2026-09-22 用户定的规则：内容都在本地跑，生成的东西默认就落在他当前打开的文件夹里 ——
    不要再问一遍，也不要藏进 workspace 让他自己找。要放别处由他明说（/out 或对话里指定）。

    两种情况回落到 workspace/output/<session>：
      · 就在 Agent 自己的项目目录里启动 —— 那是工具的代码目录，不是内容目录，
        往里倒剧本和视频会把仓库搅乱
      · 当前目录取不到或不可写（权限、盘符没挂上）
    """
    fallback = workspace / "output" / (session_id or "default")
    try:
        cwd = Path.cwd().resolve()
    except OSError:
        return fallback
    if project_root is not None and cwd == Path(project_root).resolve():
        return fallback
    if not os.access(cwd, os.W_OK):
        return fallback
    return cwd


class OutputPrefs:
    """一场会话的产物根目录 + 下载器。"""

    def __init__(self, root: Path | str, downloader: Downloader | None = None) -> None:
        self.root = Path(root)
        self.downloader = downloader or _download_default

    def dir_for(self, kind: str) -> Path:
        """kind: texts | images | videos | downloads | exports"""
        return self.root / kind

    async def download(self, url: str, dest_dir: Path, name: str) -> Path | None:
        """best-effort 下载到产物目录。失败返回 None —— 本地副本不是主链路，
        远端 URL 还在资产上，不能因为下载挂了把生成也算成失败。"""
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / name
            ok = await self.downloader(url, dest)
        except Exception:  # noqa: BLE001
            return None
        return dest if ok else None


async def _download_default(url: str, dest: Path) -> bool:
    async with httpx.AsyncClient(
        proxy=default_proxy(), timeout=300, follow_redirects=True
    ) as client:
        async with client.stream("GET", url) as resp:
            if resp.status_code >= 400:
                return False
            with dest.open("wb") as f:  # 视频几十 MB，流式写不整读进内存
                async for chunk in resp.aiter_bytes(1 << 16):
                    f.write(chunk)
        return True


def slug(text: str, limit: int = 20) -> str:
    """文件名里的摘要段：去掉 Windows 非法字符，空白压成 -。"""
    s = re.sub(r'[\\/:*?"<>|\s]+', "-", (text or "").strip())[:limit].strip("-")
    return s or "untitled"

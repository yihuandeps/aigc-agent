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
    if project_root is not None:
        pr = Path(project_root).resolve()
        # 项目目录**及其子目录**（src、workspace…）都不是内容目录（之前只判相等）
        if cwd == pr or cwd.is_relative_to(pr):
            return fallback
    if _system_dir(cwd) or not writable(cwd):
        return fallback
    return cwd


def writable(d: Path) -> bool:
    """真的写一个临时文件试试。os.access 在 Windows 上对目录恒为真（System32 也是），
    判不出来（2026-09-23 审查实测）。"""
    try:
        import tempfile

        fd, name = tempfile.mkstemp(prefix=".aigc-probe-", dir=d)
        os.close(fd)
        os.unlink(name)
        return True
    except OSError:
        return False


def _system_dir(d: Path) -> bool:
    """系统目录 / 用户主目录本身 / AppData 下：不该当产物目录（往里倒剧本视频会搅乱系统
    或把主目录当垃圾场）。"""
    s = str(d).replace("\\", "/").lower().rstrip("/")
    if s.startswith(("c:/windows", "c:/program files", "c:/programdata")):
        return True
    # AppData 下是程序与配置（用户的 agent.cmd 就装在 AppData/Local/Programs 下，从快捷方式
    # 启动时当前目录就是它）—— 只放过临时目录
    if "/appdata/" in s + "/" and "/appdata/local/temp" not in s:
        return True
    try:
        return d == Path.home().resolve()
    except (OSError, RuntimeError):
        return False


class OutputPrefs:
    """一场会话的产物根目录 + 下载器。"""

    def __init__(self, root: Path | str, downloader: Downloader | None = None) -> None:
        self.root = Path(root)
        self.downloader = downloader or _download_default
        self.retry_delay = 2.0  # 网络抖动时下载重试的退避基数（秒）

    def dir_for(self, kind: str) -> Path:
        """kind: texts | images | videos | downloads | exports"""
        return self.root / kind

    async def download(
        self, url: str, dest_dir: Path, name: str, attempts: int = 3
    ) -> Path | None:
        """下载到产物目录，失败退避重试。都失败返回 None —— 远端 URL 还在资产上，
        不能因为下载挂了把生成也算成失败。

        但本地副本不只是方便：字幕门、镜头门、一致性门都要抽本地帧，远端链接约 24h 失效。
        之前一次失败就放弃，网络一抖这几道门就被静默跳过（2026-09-23 审查），所以重试。
        """
        import asyncio

        dest = dest_dir / name
        for k in range(max(1, attempts)):
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                if await self.downloader(url, dest):
                    return dest
                failed_for_good = True  # 服务端明确拒了（4xx）：几秒内不会自己好，不重试
            except Exception:  # noqa: BLE001 — 网络抖动：值得再试
                failed_for_good = False
            try:
                dest.unlink(missing_ok=True)  # 下载一半的残文件不能留着冒充副本
            except OSError:
                pass
            if failed_for_good:
                return None
            if k + 1 < attempts:
                await asyncio.sleep(self.retry_delay * (k + 1))
        return None


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

"""拒绝表只有一份（2026-09-29 优化审查 1.5）。

素材库 MCP server 是独立进程，之前照 functions/files.py 手抄了一份拒绝表，漏了
`**/rpa/profile/**` 和 `.config/gcloud` —— import_material 能把抖音、小红书的浏览器
登录态拷进素材库，主进程的 fs_read 再从素材库读出来发给模型。现在两边共用
domain/sensitive_paths.py 这一份，匹配也走同一个函数。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.functions.files import FileFunctions, FsPolicy
from aigc_agent.interfaces.mcp_servers import material_lib

# 相对一个假的用户目录：前一半该拦，后一半照常能碰
_DENIED = [
    ".ssh/id_ed25519",
    "proj/.env",
    "proj/.env.local",
    "keys/server.pem",
    ".config/gcloud/access_tokens.db",
    ".config/gcloud/configurations/config_default",
    "proj/workspace/rpa/profile/Default/Cookies",
    "proj/workspace/rpa/profile/Local State",
    "AppData/Roaming/app/config.json",
    "AppData/Local/Google/Chrome/User Data/Default/Login Data",
    "proj/.venv/Lib/site.py",
    "proj/.git/config",
    "proj/node_modules/x/index.js",
    "vault.kdbx",
    "my_credentials.txt",
]
_ALLOWED = [
    "素材/a.mp4",
    "AppData/Local/Temp/clip.mp4",
    "AppData/Local/Programs/aigc-agent/agent.cmd",
    "proj/workspace/rpa/hot_20260923.json",
    "照片/profile/头像.png",
    ".config/别的工具/settings.json",
]


def test_素材库拒绝RPA登录态和gcloud凭据(tmp_path: Path):
    home = tmp_path / "home"
    # gcloud 下的文件名特意不带 credential —— 带了会被 *credential* 那条顺手拦住，测不出漏没漏
    assert material_lib._sensitive(home / ".config" / "gcloud" / "access_tokens.db")
    assert material_lib._sensitive(home / "ws" / "rpa" / "profile" / "Default" / "Cookies")
    assert material_lib._sensitive(home / "ws" / "rpa" / "profile" / "Local State")
    # rpa 下 profile 以外的抓取数据照常能导
    assert not material_lib._sensitive(home / "ws" / "rpa" / "hot.json")


def test_素材库导入RPA登录态被拒且不落库(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    lib = tmp_path / "materials"
    monkeypatch.setenv("MATERIAL_LIB_ROOT", str(lib))
    cookies = tmp_path / "ws" / "rpa" / "profile" / "Default" / "Cookies"
    cookies.parent.mkdir(parents=True)
    cookies.write_bytes(b"SQLite format 3\x00")
    with pytest.raises(ToolError, match="拒绝导入"):
        material_lib.import_material(str(cookies))
    assert not [p for p in lib.rglob("*") if p.is_file()], "登录态被拷进了素材库"


def test_素材库和fs工具用的是同一份拒绝表(tmp_path: Path):
    from aigc_agent.domain import sensitive_paths

    assert material_lib._SENSITIVE is sensitive_paths.HARD_DENY
    assert set(sensitive_paths.HARD_DENY) <= set(FsPolicy().deny)
    # 配置写了自己的 deny 也挤不掉共用表
    cfg = tmp_path / "filesystem.yaml"
    cfg.write_text('deny: ["**/只拦这个/**"]\n', encoding="utf-8")
    assert set(sensitive_paths.HARD_DENY) <= set(FsPolicy.load(cfg).deny)
    for pat in ("**/rpa/profile/**", "**/.config/gcloud/**"):
        assert pat in sensitive_paths.HARD_DENY


def test_同一批路径两边判得一样(tmp_path: Path):
    home = tmp_path / "home"
    fns = FileFunctions(AssetStore(tmp_path / "a"), tmp_path / "ws", tmp_path / "out", FsPolicy())
    for rel in _DENIED + _ALLOWED:
        p = home / rel
        want = rel in _DENIED
        assert material_lib._sensitive(p) is want, f"素材库判错了：{rel}"
        assert fns.denied(p.resolve()) is want, f"fs 工具判错了：{rel}"


def test_拒绝表模块只用标准库():
    """素材库 server 是独立进程，本包里只引这一个文件 —— 它要是 import 了第三方库或本包
    别的模块，server 启动就得把那一串都带上。"""
    from aigc_agent.domain import sensitive_paths

    tree = ast.parse(Path(sensitive_paths.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, "不许相对 import 本包别的模块"
            names = [node.module or ""]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            continue
        for name in names:
            assert name.split(".")[0] in sys.stdlib_module_names, f"{name} 不是标准库"

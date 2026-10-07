"""永远不碰的本机路径 —— fs_* 工具和素材库 MCP server 共用这一份。

凭据（私钥、云账号、密码库）、浏览器数据（登录态、Cookie、保存的密码）、RPA 用的浏览器
profile、虚拟环境与系统目录。fs_read 读到的东西会原样发给文本模型（第三方），一旦读出来
就收不回；素材库的 import_material 拷进去的文件，主进程再从素材库读出来，等于绕过了
fs_* 的拒绝 —— 所以两边必须是同一份表。

2026-09-29 优化审查 1.5：素材库 server 之前照 functions/files.py 手抄了一份，漏了
`**/rpa/profile/**` 和 `.config/gcloud`，import_material 能把抖音、小红书的浏览器登录态
拷进素材库。现在两边都只引这里，加规则只加这一处。

**只许用标准库，也不许 import 本包别的模块**：素材库 server 是独立进程，本包里只引这一个文件。
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable
from pathlib import PurePath

# 写法：开头 **/ = 任意目录下；结尾 /** = 目录里的一切。匹配时路径和规则都统一成
# 小写、正斜杠（Windows 不分大小写）
HARD_DENY: tuple[str, ...] = (
    # 凭据文件、虚拟环境、仓库内部、系统目录
    "**/.env", "**/.env.*", "**/*.pem", "**/*.key", "**/id_rsa*", "**/*.pfx",
    "**/.venv/**", "**/.git/**", "**/__pycache__/**", "**/node_modules/**",
    "C:/Windows/**", "C:/Program Files/**", "C:/Program Files (x86)/**",
    # 私钥目录、云账号与包管理器凭据、密码库
    "**/.ssh/**", "**/.gnupg/**", "**/.aws/**", "**/.azure/**", "**/.kube/**",
    "**/.docker/**", "**/.config/gcloud/**", "**/.claude/**",
    "**/.git-credentials", "**/.netrc", "**/_netrc", "**/.npmrc", "**/.pypirc",
    "**/*credential*", "**/*.kdbx", "**/*.p12", "**/*.ppk",
    "**/id_ed25519*", "**/id_ecdsa*", "**/id_dsa*",
    # 应用配置与浏览器数据（登录态、Cookie、保存的密码）。AppData/Local/Temp 与 Programs 不拦 ——
    # 临时文件和装在那里的程序（用户的 agent.cmd 就在 Programs 下）是正常要碰的
    "**/AppData/Roaming/**", "**/AppData/LocalLow/**", "**/User Data/**",
    "**/AppData/Local/Microsoft/**", "**/AppData/Local/Google/**", "**/AppData/Local/Packages/**",
    # RPA 用的浏览器 profile（抖音、小红书登录态，Local State 里的加密主密钥）
    "**/rpa/profile/**",
)


def _canon(s: str) -> str:
    return s.replace("\\", "/").lower()


def denied(path: str | PurePath, patterns: Iterable[str] = HARD_DENY) -> bool:
    """路径命中任一条规则就返回 True。

    不解析路径：符号链接、`..`、8.3 短名要调用方先 resolve 再传进来。
    fnmatch 的 * 本来就跨目录，所以「**/x」除了按原样匹配（任意目录下的 x），
    还去掉开头的 **/ 再匹配一次（相对路径本身就是 x 的情况）。
    """
    s = _canon(str(path))
    for pat in patterns:
        q = _canon(pat)
        if fnmatch.fnmatchcase(s, q) or fnmatch.fnmatchcase(s, q.removeprefix("**/")):
            return True
    return False

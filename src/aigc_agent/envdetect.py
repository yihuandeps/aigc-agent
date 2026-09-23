"""环境自适应 —— 本机自动就位，换台机器则明确告诉你缺什么。

设计取舍：
  · **本机**：代理、ffmpeg、密钥能自动找到的都自动找，不让你重复配
  · **别的机器**：不猜、不静默降级，`agent setup` 逐条列出缺什么、去哪配

为什么要自动找代理：国内网络下多数外部 API 不走代理连不通，而这个失败
会伪装成「key 无效」「DNS 污染」，排查起来很费时间。本项目就踩过一次。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 运行时产物目录（资产库、台账、记忆、会话日志、回收站…）的唯一出处。
# 环境变量 AIGC_WORKSPACE 可以把它挪走 —— 测试就靠它指到临时目录。
WORKSPACE_ENV = "AIGC_WORKSPACE"


def workspace_root() -> Path:
    """当前进程该用的 workspace。

    2026-09-23 审查：之前 workspace 写死成 PROJECT_ROOT/workspace，测试一跑就往真实
    资产库、台账、会话日志里写东西 —— 108 份测试桩成了「最新的角色档案/创作方案」，
    项目卡每轮把它们当权威 pin 给模型。所以测试进程里**必须**显式指定别的目录，
    没指定就直接报错，而不是悄悄写进真实数据。
    """
    raw = os.environ.get(WORKSPACE_ENV, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    if "pytest" in sys.modules:
        raise RuntimeError(
            f"测试进程不许写真实 workspace（{PROJECT_ROOT / 'workspace'}）："
            f"设 {WORKSPACE_ENV} 指到临时目录（tests/conftest.py 已经这么做）"
        )
    return PROJECT_ROOT / "workspace"

# 按优先级找代理：项目 .env > 进程环境 > Claude Code 设置
_CLAUDE_SETTINGS = [
    Path.home() / ".claude" / "settings.json",
    Path.home() / ".claude" / "settings.local.json",
]

# 每个能力需要哪个密钥，以及缺了会怎样
REQUIREMENTS = [
    ("KIMI_API_KEY", "主 Agent 文本", "https://www.kimi.com/code/console", True),
    ("APIMART_API_KEY", "图像/视频/配音/转写", "https://apimart.ai", False),
    ("TIKHUB_API_KEY", "抖音公开数据与评论", "https://tikhub.io", False),
]


@dataclass
class EnvReport:
    dotenv_loaded: list[str] = field(default_factory=list)
    proxy: str | None = None
    proxy_source: str = ""
    ffmpeg: bool = False
    present: list[str] = field(default_factory=list)
    missing: list[tuple[str, str, str, bool]] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """必需项齐了就算能跑。可选项缺失只是少几个能力。"""
        return not any(required for _, _, _, required in self.missing)

    @property
    def blocking(self) -> list[tuple[str, str, str, bool]]:
        return [m for m in self.missing if m[3]]


def load_dotenv(path: Path | None = None) -> list[str]:
    """极简 .env 加载。已存在的环境变量优先，不覆盖。

    值里可能含 `=`（base64 密钥），所以只按第一个 `=` 切。
    """
    path = path or PROJECT_ROOT / ".env"
    loaded: list[str] = []
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def _proxy_from_claude() -> tuple[str | None, str]:
    """Claude Code 的 settings.json 里常配着代理，直接复用，省得重配一遍。"""
    for f in _CLAUDE_SETTINGS:
        if not f.exists():
            continue
        try:
            env = (json.loads(f.read_text(encoding="utf-8")) or {}).get("env") or {}
        except Exception:  # noqa: BLE001
            continue
        for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            if env.get(key):
                return env[key], f"{f.name} 的 env 段"
    return None, ""


def detect(auto_proxy: bool = True) -> EnvReport:
    """探测环境。auto_proxy=False 时不去翻 Claude 配置（换机部署时更可控）。"""
    r = EnvReport()
    r.dotenv_loaded = load_dotenv()

    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        if os.environ.get(key):
            r.proxy, r.proxy_source = os.environ[key], f"环境变量 {key}"
            break

    if not r.proxy and auto_proxy:
        proxy, src = _proxy_from_claude()
        if proxy:
            # 注入进程环境，各 provider 统一从这里取
            os.environ.setdefault("HTTPS_PROXY", proxy)
            os.environ.setdefault("HTTP_PROXY", proxy)
            os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
            r.proxy, r.proxy_source = proxy, src

    r.ffmpeg = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))

    for key, what, where, required in REQUIREMENTS:
        val = os.environ.get(key, "")
        if val and not val.startswith("sk-xxx") and "xxxx" not in val:
            r.present.append(key)
        else:
            r.missing.append((key, what, where, required))
    return r

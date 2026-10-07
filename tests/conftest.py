"""测试隔离（2026-09-23 审查后加）：整个测试进程的 workspace 指到临时目录。

之前测试直接用项目下的 workspace/：每跑一次全量测试，真实资产库就多几十份
「创作方案 / 角色档案 / 分集目录」测试桩，会话日志多几十份，台账多几行 ——
测试桩还成了「最新的角色档案」，被项目卡每轮 pin 给模型当权威。

envdetect.workspace_root() 在测试进程里没设 AIGC_WORKSPACE 会直接报错；
这里在任何测试模块 import 之前就把它设好（conftest 最先被加载）。
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest

_TMP_WORKSPACE = Path(tempfile.mkdtemp(prefix="aigc-test-ws-"))
os.environ["AIGC_WORKSPACE"] = str(_TMP_WORKSPACE)
atexit.register(shutil.rmtree, _TMP_WORKSPACE, ignore_errors=True)


@pytest.fixture(autouse=True)
def _cwd_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """兜底：每个测试都在自己的临时目录里跑（2026-09-29 审查 4.5）。

    当前目录默认是仓库根目录，哪段代码按相对路径写当前目录，就写进了仓库 ——
    没传 root 的资产库往 ./blobs 写，blobs/板卡照片_镜02.mp4 每跑一次全量测试被重写一遍。
    读 config/、skills/ 的地方都按 PROJECT_ROOT / __file__ 走绝对路径，不受影响。
    Agent.create 的默认产物目录取当前目录（default_root），所以装配层测试的产物也落在这里。
    """
    monkeypatch.chdir(tmp_path)

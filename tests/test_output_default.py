"""产物目录默认落在「用户当前打开的文件夹」（2026-09-22 用户定的规则）。

用户的原话：内容都在本地跑，开工前先读一遍当前文件夹有什么、没什么；
之后生成的一切都存进这个已经打开的文件夹，除非他明说要存到别处。

之前的行为是默认 workspace/output/<session>、开工前问一次 —— 生成的东西藏在
工具自己的目录里，用户得自己去翻。
"""

from __future__ import annotations

import os
from pathlib import Path

from aigc_agent.domain.local_materials import LocalMaterials
from aigc_agent.domain.output import OutputPrefs, default_root


def _cd(path: Path):
    """临时切工作目录。"""

    class _Ctx:
        def __enter__(self) -> None:
            self.old = os.getcwd()
            os.chdir(path)

        def __exit__(self, *a: object) -> None:
            os.chdir(self.old)

    return _Ctx()


# ---------------------------------------------------------------- 默认目录


def test_默认就是当前打开的文件夹(tmp_path: Path):
    here = tmp_path / "西游记"
    here.mkdir()
    ws = tmp_path / "ws"
    with _cd(here):
        assert default_root(ws, "default", tmp_path / "项目") == here.resolve()


def test_在项目目录里启动时不往源码目录里倒(tmp_path: Path):
    """Agent 自己的代码目录不是内容目录 —— 往里写剧本和视频会把仓库搅乱。"""
    project = tmp_path / "手搓Agent"
    project.mkdir()
    ws = project / "workspace"
    with _cd(project):
        assert default_root(ws, "abc", project) == ws / "output" / "abc"


def test_目录不可写就回落(tmp_path: Path, monkeypatch):
    """可写性真的写一个临时文件去试（2026-09-23）：os.access 在 Windows 上对目录恒为真，
    System32 也判成可写 —— 所以这里模拟的是「试写失败」。"""
    import aigc_agent.domain.output as out_mod

    here = tmp_path / "只读"
    here.mkdir()
    ws = tmp_path / "ws"
    monkeypatch.setattr(out_mod, "writable", lambda d: False)
    with _cd(here):
        assert default_root(ws, "s1", None) == ws / "output" / "s1"


def test_项目子目录_系统目录_程序目录都不当产物目录(tmp_path: Path):
    from aigc_agent.domain.output import _system_dir

    project = tmp_path / "手搓Agent"
    (project / "src").mkdir(parents=True)
    ws = project / "workspace"
    with _cd(project / "src"):  # 之前只判「等于项目目录」，子目录里启动照样往仓库倒
        assert default_root(ws, "s", project) == ws / "output" / "s"
    assert _system_dir(Path("C:/Windows/System32"))
    assert _system_dir(Path("C:/Users/x/AppData/Local/Programs/aigc-agent"))
    assert not _system_dir(Path("C:/Users/x/AppData/Local/Temp/work"))
    assert not _system_dir(Path("E:/西游记"))


def test_取不到当前目录也不炸(tmp_path: Path, monkeypatch):
    ws = tmp_path / "ws"

    def boom() -> Path:
        raise OSError("盘符没挂上")

    monkeypatch.setattr(Path, "cwd", staticmethod(boom))
    assert default_root(ws, "", None) == ws / "output" / "default"


def test_产物目录的子目录跟着走(tmp_path: Path):
    """落盘点都走 prefs.dir_for，根一换整条链跟着换。"""
    prefs = OutputPrefs(tmp_path / "我的剧")
    assert prefs.dir_for("videos") == tmp_path / "我的剧" / "videos"
    prefs.root = tmp_path / "别处"
    assert prefs.dir_for("images") == tmp_path / "别处" / "images"


# ---------------------------------------------------------------- 开工清点


def test_清点报有什么也报没什么(tmp_path: Path):
    out = tmp_path / "西游记"
    (out / "texts").mkdir(parents=True)
    (out / "texts" / "第1集.md").write_text("x", encoding="utf-8")
    (out / "texts" / "第2集.md").write_text("x", encoding="utf-8")
    (out / "images").mkdir()
    (out / "images" / "参考图-角色-01_阿蛛.png").write_bytes(b"p")
    text = LocalMaterials(out).index().inventory()
    assert "西游记" in text and "3 个文件" in text
    assert "文本 2（第 1-2 集）" in text or "文本 2（第 1–2 集）" in text
    assert "图片 1" in text
    assert "没有：" in text and "视频" in text, "缺什么要说出来"


def test_空文件夹也说清楚(tmp_path: Path):
    out = tmp_path / "新项目"
    out.mkdir()
    text = LocalMaterials(out).index().inventory()
    assert "空的" in text and "都会落在这里" in text


def test_清点不会因为目录不存在而炸(tmp_path: Path):
    text = LocalMaterials(tmp_path / "还没建").index().inventory()
    assert "空的" in text

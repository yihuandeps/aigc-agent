"""M20 真实链路验收：起一个**真的** MCP server（stdio），走官方 SDK 客户端。

之前 23 条 MCP 测试全用 FakeMcpClient —— SdkMcpClient 一次都没被跑过。
而装的 mcp 2.x 把 inputSchema / isError 改成了 snake_case，那条路其实是断的，
假客户端永远测不出来。这里连的是本项目自带的素材库 server
（src/aigc_agent/interfaces/mcp_servers/material_lib.py）。

顺带把 Windows + 中文路径这条老坑走一遍：中文文件名要能在 stdio 上往返。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from aigc_agent.capabilities.mcp_hub.config import McpConfig, ServerSpec  # noqa: E402
from aigc_agent.capabilities.mcp_hub.hub import McpHub  # noqa: E402
from aigc_agent.harness.events.bus import EventBus  # noqa: E402
from aigc_agent.harness.permission.gate import PermissionGate  # noqa: E402
from aigc_agent.harness.tools.builtin import builtin  # noqa: E402
from aigc_agent.harness.tools.provider import PermissionLevel  # noqa: E402
from aigc_agent.harness.tools.registry import ToolRegistry  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "mcp_servers.yaml"
SERVER_MODULE = "aigc_agent.interfaces.mcp_servers.material_lib"


def _spec(root: Path) -> ServerSpec:
    return ServerSpec(
        alias="material_lib",
        transport="stdio",
        command=sys.executable,
        args=["-m", SERVER_MODULE],
        env={"MATERIAL_LIB_ROOT": str(root), "PYTHONUTF8": "1"},
        tool_overrides={
            "list_folders": PermissionLevel.READ,
            "search_materials": PermissionLevel.READ,
            "get_material": PermissionLevel.READ,
            "import_material": PermissionLevel.WRITE,
        },
    )


@pytest.fixture
def lib(tmp_path: Path) -> Path:
    root = tmp_path / "素材库"
    (root / "bgm").mkdir(parents=True)
    (root / "旧素材").mkdir()
    (root / "海边日落.mp4").write_bytes(b"\x00" * 128)
    (root / "logo.png").write_bytes(b"\x89PNG" + b"\x00" * 32)
    (root / "bgm" / "轻快.mp3").write_bytes(b"ID3" + b"\x00" * 64)
    (root / "旧素材" / "old.txt").write_text("旧的", encoding="utf-8")
    return root


async def _connect(root: Path):
    bus = EventBus()
    hub = McpHub(McpConfig(servers=[_spec(root)]), bus)
    ok = await hub.connect_all()
    registry = ToolRegistry(bus)
    registry.register(builtin)
    hub.register_into(registry)
    await registry.refresh()
    return hub, registry, bus, ok


async def test_真实server连上并按命名空间注册(lib: Path):
    hub, registry, _, ok = await _connect(lib)
    try:
        assert ok == {"material_lib": True}
        names = {m.name for m in registry.catalog()}
        assert {
            "material_lib__list_folders",
            "material_lib__search_materials",
            "material_lib__get_material",
            "material_lib__import_material",
            "material_lib__remove_material",
        } <= names
        assert "now" in names, "内置工具与外部工具在同一个注册表里"

        perms = {m.name: m.permission for m in registry.catalog()}
        assert perms["material_lib__search_materials"] is PermissionLevel.READ
        # 没显式降级的保持最严
        assert perms["material_lib__remove_material"] is PermissionLevel.EXTERNAL

        schema = (await registry.schemas(["material_lib__search_materials"]))[0]
        assert "外部 Server" in schema["function"]["description"], "外部描述要包裹标注来源"
        # mcp 2.x 的 input_schema（snake_case）要读得到
        assert "query" in schema["function"]["parameters"]["properties"]
    finally:
        await hub.close_all()


async def test_中文文件名在stdio上往返(lib: Path):
    hub, registry, _, _ = await _connect(lib)
    try:
        r = await registry.invoke("material_lib__search_materials", {"query": "日落"})
        assert r.ok, r.error
        assert "海边日落.mp4" in r.content
        assert '"kind": "video"' in r.content or "video" in r.content
    finally:
        await hub.close_all()


async def test_按类型过滤(lib: Path):
    hub, registry, _, _ = await _connect(lib)
    try:
        r = await registry.invoke("material_lib__search_materials", {"kind": "audio"})
        assert r.ok, r.error
        assert "轻快.mp3" in r.content and "海边日落" not in r.content
    finally:
        await hub.close_all()


async def test_越界路径被server拒绝(lib: Path):
    hub, registry, _, _ = await _connect(lib)
    try:
        r = await registry.invoke("material_lib__get_material", {"path": "../../etc/passwd"})
        assert not r.ok
        assert "越界" in (r.error or "")
    finally:
        await hub.close_all()


async def test_破坏性工具默认要人确认(lib: Path):
    hub, registry, bus, _ = await _connect(lib)
    try:
        registry.gate = PermissionGate(bus)  # 无人可问
        r = await registry.invoke("material_lib__remove_material", {"path": "logo.png"})
        assert not r.ok and "需要人工确认" in r.error
        assert (lib / "logo.png").exists(), "被拒的调用不能真的执行"

        async def yes(meta, args):
            return True

        registry.gate = PermissionGate(bus, asker=yes)
        r = await registry.invoke("material_lib__remove_material", {"path": "logo.png"})
        assert r.ok, r.error
        assert not (lib / "logo.png").exists()
        assert list((lib / ".trash").glob("*logo.png")), "移进回收站而不是物理删除"
    finally:
        await hub.close_all()


async def test_导入外部文件进库(lib: Path, tmp_path: Path):
    src = tmp_path / "外部_封面.png"
    src.write_bytes(b"\x89PNG" + b"\x00" * 16)
    hub, registry, bus, _ = await _connect(lib)
    try:
        registry.gate = PermissionGate(bus)
        r = await registry.invoke(
            "material_lib__import_material", {"source": str(src), "folder": "imports"}
        )
        assert r.ok, r.error
        assert (lib / "imports" / "外部_封面.png").exists()
        assert src.exists(), "导入是复制不是移动"

        r = await registry.invoke("material_lib__search_materials", {"query": "封面"})
        assert r.ok and "imports/外部_封面.png" in r.content
    finally:
        await hub.close_all()


async def test_两级披露对真实server生效(lib: Path):
    hub, registry, _, _ = await _connect(lib)
    try:
        digest = registry.catalog_digest()
        assert "material_lib__search_materials*" in digest, "外部工具默认只上目录，标 * 待展开"
        sent = {s["function"]["name"] for s in await registry.schemas_for_context()}
        assert "material_lib__search_materials" not in sent
        assert "now" in sent

        ok, _ = registry.expand("material_lib__search_materials")
        assert ok
        sent = {s["function"]["name"] for s in await registry.schemas_for_context()}
        assert "material_lib__search_materials" in sent
    finally:
        await hub.close_all()


async def test_hub状态与审计(lib: Path):
    hub, _, _, _ = await _connect(lib)
    try:
        (s,) = await hub.status()
        assert s["alias"] == "material_lib" and s["ok"] and s["tools"] == 5
        assert s["externals"] == 1 and not s["circuit_open"]
        assert hub.audit_permissions() == ["material_lib__remove_material"]
    finally:
        await hub.close_all()


async def test_真实配置文件里声明的server能连上():
    """config/mcp_servers.yaml 里的 ${PYTHON} / ${PROJECT_ROOT} 解析后要真能起进程。"""
    bus = EventBus()
    hub = McpHub.from_file(CONFIG_PATH, bus)
    try:
        ok = await hub.connect_all()
        assert ok.get("material_lib") is True, [e.data for e in bus.history]
        registry = ToolRegistry(bus)
        hub.register_into(registry)
        await registry.refresh()
        assert "material_lib__search_materials" in {m.name for m in registry.catalog()}
    finally:
        await hub.close_all()

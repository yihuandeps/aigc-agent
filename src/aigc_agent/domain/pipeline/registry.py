"""M11 图注册表 —— 扫描 graphs/ 目录，加载并校验所有图定义。

图定义是数据不是代码：加一个内容形态 = 往 graphs/ 丢一个 yaml，不动引擎。
实现 L0 Router 需要的 GraphCatalog 协议。
"""

from __future__ import annotations

from pathlib import Path

from ...harness.execution.graph.models import GraphDef

GRAPHS_DIR = Path(__file__).parent / "graphs"
SUBGRAPHS_DIR = Path(__file__).parent / "subgraphs"


class GraphRegistry:
    def __init__(self, graphs_dir: Path | None = None) -> None:
        self.dir = graphs_dir or GRAPHS_DIR
        self._graphs: dict[str, GraphDef] = {}
        self.errors: dict[str, str] = {}

    def load_all(self) -> None:
        """加载全部图。单个图写错只记录错误，不影响其余图可用。"""
        self._graphs.clear()
        self.errors.clear()
        if not self.dir.exists():
            return
        for f in sorted(self.dir.glob("*.yaml")):
            try:
                g = GraphDef.load(f)
            except Exception as e:  # noqa: BLE001
                self.errors[f.stem] = str(e)
                continue
            self._graphs[g.id] = g

    def get(self, graph_id: str) -> GraphDef:
        if graph_id not in self._graphs:
            raise KeyError(
                f"未知的图 {graph_id!r}。已加载：{', '.join(sorted(self._graphs)) or '（无）'}"
            )
        return self._graphs[graph_id]

    def all(self) -> list[GraphDef]:
        return list(self._graphs.values())

    # ---- GraphCatalog 协议（供 L0 Router 用）----

    def triggers(self) -> dict[str, list[str]]:
        return {g.id: list(g.triggers) for g in self._graphs.values()}

    def has(self, graph_id: str) -> bool:
        return graph_id in self._graphs

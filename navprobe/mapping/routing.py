from __future__ import annotations


from navprobe.memory.graph.node import Node
from navprobe.memory.graph.graph import Graph


def _node_id_sort_key(node_id: str) -> tuple[int, str]:
    stripped = str(node_id).strip()
    if stripped.startswith("n") and stripped[1:].isdigit():
        return (int(stripped[1:]), stripped)
    digits = "".join(char for char in stripped if char.isdigit())
    if digits != "":
        return (int(digits), stripped)
    return (10**9, stripped)


def _place_nodes(graph: Graph) -> list[Node]:
    return [node for node in graph.iter_nodes() if node.node_kind == "place"]

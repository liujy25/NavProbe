from __future__ import annotations

from dataclasses import dataclass, field
import threading

from navprobe.memory.graph.edge import Edge, normalize_path_xy_points
from navprobe.memory.graph.node import Node

DEFAULT_FLOOR_ID = "floor_0"


@dataclass
class Floor:
    id: str
    height: float
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Floor":
        floor_id = str(payload.get("id", DEFAULT_FLOOR_ID))
        raw_nodes = payload.get("nodes", [])
        raw_edges = payload.get("edges", [])
        if not isinstance(raw_nodes, list):
            raise ValueError(f"floor.nodes must be list, got {type(raw_nodes).__name__}")
        if not isinstance(raw_edges, list):
            raise ValueError(f"floor.edges must be list, got {type(raw_edges).__name__}")
        floor = cls(
            id=floor_id,
            height=float(payload.get("height", 0.0)),
        )
        for raw_node in raw_nodes:
            if not isinstance(raw_node, dict):
                raise ValueError(f"floor node payload must be dict, got {type(raw_node).__name__}")
            node = Node.from_dict({**raw_node, "floor_id": raw_node.get("floor_id", floor_id)})
            node.floor_id = floor_id
            floor.nodes[str(node.id)] = node
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict):
                raise ValueError(f"floor edge payload must be dict, got {type(raw_edge).__name__}")
            floor.edges.append(Edge.from_dict(raw_edge))
        return floor

    def to_dict(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "height": float(self.height),
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
        }


class Graph:
    def __init__(self) -> None:
        self.floors: dict[str, Floor] = {
            DEFAULT_FLOOR_ID: Floor(id=DEFAULT_FLOOR_ID, height=0.0)
        }
        self.vertical_edges: list[Edge] = []
        self._node_index = 0
        self._edge_index = 0
        self._lock = threading.RLock()

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def __getitem__(self, floor_id: str) -> Floor:
        return self.get_floor(str(floor_id))

    def add_floor(self, floor_id: str | None = None, height: float = 0.0) -> Floor:
        with self._lock:
            normalized_floor_id = (
                f"floor_{len(self.floors)}"
                if floor_id is None or str(floor_id).strip() == ""
                else str(floor_id).strip()
            )
            existing = self.floors.get(normalized_floor_id)
            if existing is not None:
                existing.height = float(height)
                return existing
            floor = Floor(id=normalized_floor_id, height=float(height))
            self.floors[normalized_floor_id] = floor
            return floor

    def get_floor(self, floor_id: str) -> Floor:
        with self._lock:
            return self.floors[str(floor_id)]

    def set_floor_height(self, floor_id: str, height: float) -> Floor:
        with self._lock:
            floor_id_str = str(floor_id)
            if floor_id_str not in self.floors:
                return self.add_floor(floor_id=floor_id_str, height=float(height))
            floor = self.floors[floor_id_str]
            floor.height = float(height)
            return floor

    def iter_nodes(self, floor_id: str | None = None) -> list[Node]:
        with self._lock:
            if floor_id is not None:
                return list(self.floors[str(floor_id)].nodes.values())
            nodes: list[Node] = []
            for floor in self.floors.values():
                nodes.extend(floor.nodes.values())
            return nodes

    def iter_edges(self, floor_id: str | None = None, include_vertical: bool = True) -> list[Edge]:
        with self._lock:
            if floor_id is not None:
                edges = list(self.floors[str(floor_id)].edges)
            else:
                edges = []
                for floor in self.floors.values():
                    edges.extend(floor.edges)
            if include_vertical and floor_id is None:
                edges.extend(self.vertical_edges)
            return edges

    def add_node(
        self,
        position: tuple[float, float, float],
        yaw: float,
        obs_id: str,
        floor_id: str = DEFAULT_FLOOR_ID,
        node_kind: str = "",
        category: str = "",
        features: list[str] | None = None,
    ) -> Node:
        with self._lock:
            node_id = f"n{self._node_index}"
            self._node_index += 1
            node = Node(
                id=node_id,
                position=position,
                yaw=yaw,
                floor_id=str(floor_id),
                node_kind="place" if str(node_kind).strip() == "" else str(node_kind).strip(),
                obs_ids=[obs_id],
                category=category,
                features=[] if features is None else list(features),
            )
            floor = self.floors.get(str(floor_id))
            if floor is None:
                floor = self.add_floor(floor_id=str(floor_id), height=float(position[2]))
            floor.nodes[node_id] = node
            return node

    def add_edge(
        self,
        src_id: str,
        dst_id: str,
        relation: str = "",
        floor_id: str | None = None,
        path_length: float | None = None,
        rgb_history_obs_ids: list[str] | None = None,
        path_xy: list[list[float]] | list[tuple[float, float]] | None = None,
    ) -> Edge:
        with self._lock:
            rgb_history = [] if rgb_history_obs_ids is None else [str(obs_id) for obs_id in rgb_history_obs_ids]
            edge_path_xy = (
                [] if path_xy is None else normalize_path_xy_points(list(path_xy), context="edge path_xy")
            )
            edge_store = self._edge_store(src_id=str(src_id), dst_id=str(dst_id), relation=str(relation), floor_id=floor_id)
            for edge in edge_store:
                if edge.src_id == src_id and edge.dst_id == dst_id and edge.relation == relation:
                    edge.traversal_count += 1
                    if path_length is not None:
                        path_length_float = float(path_length)
                        if edge.path_length is None or path_length_float < float(edge.path_length):
                            edge.path_length = path_length_float
                    if edge.rgb_history_obs_ids == [] and rgb_history != []:
                        edge.rgb_history_obs_ids = list(rgb_history)
                    if edge.path_xy == [] and edge_path_xy != []:
                        edge.path_xy = list(edge_path_xy)
                    return edge

            edge_id = f"e{self._edge_index}"
            self._edge_index += 1
            edge = Edge(
                id=edge_id,
                src_id=src_id,
                dst_id=dst_id,
                relation=relation,
                path_length=None if path_length is None else float(path_length),
                rgb_history_obs_ids=rgb_history,
                path_xy=edge_path_xy,
            )
            edge_store.append(edge)
            return edge

    def _edge_store(
        self,
        *,
        src_id: str,
        dst_id: str,
        relation: str,
        floor_id: str | None,
    ) -> list[Edge]:
        if _is_vertical_relation(relation):
            return self.vertical_edges
        target_floor_id = floor_id
        if target_floor_id is None:
            src_node = self.get_node(src_id)
            dst_node = self.get_node(dst_id)
            if str(src_node.floor_id) != str(dst_node.floor_id):
                return self.vertical_edges
            target_floor_id = str(src_node.floor_id)
        return self.floors[str(target_floor_id)].edges

    def edit_node(self, node_id: str, patch: dict[str, object]) -> Node:
        with self._lock:
            node = self.get_node(node_id)
            if "position" in patch:
                position = patch["position"]
                if not isinstance(position, (list, tuple)) or len(position) != 3:
                    raise ValueError("edit_node patch.position must be a length-3 list/tuple")
                node.position = (float(position[0]), float(position[1]), float(position[2]))
            if "floor_id" in patch:
                new_floor_id = str(patch["floor_id"]).strip()
                if new_floor_id == "":
                    raise ValueError("edit_node patch.floor_id must be non-empty")
                if new_floor_id != str(node.floor_id):
                    self._move_node_to_floor(node, new_floor_id)
            if "node_kind" in patch:
                node.node_kind = str(patch["node_kind"]).strip()
            if "category" in patch:
                node.category = str(patch["category"])
            if "features" in patch:
                node.features = [str(feature) for feature in list(patch["features"])]
            if "obs_ids" in patch:
                node.obs_ids = list(patch["obs_ids"])
            if "semantic_anchor_xy" in patch:
                semantic_anchor_xy = patch["semantic_anchor_xy"]
                if semantic_anchor_xy is None:
                    node.semantic_anchor_xy = None
                else:
                    if not isinstance(semantic_anchor_xy, (list, tuple)) or len(semantic_anchor_xy) != 2:
                        raise ValueError("edit_node patch.semantic_anchor_xy must be a length-2 list/tuple or None")
                    node.semantic_anchor_xy = (float(semantic_anchor_xy[0]), float(semantic_anchor_xy[1]))
            if "nav_goal_xy" in patch:
                nav_goal_xy = patch["nav_goal_xy"]
                if nav_goal_xy is None:
                    node.nav_goal_xy = None
                else:
                    if not isinstance(nav_goal_xy, (list, tuple)) or len(nav_goal_xy) != 2:
                        raise ValueError("edit_node patch.nav_goal_xy must be a length-2 list/tuple or None")
                    node.nav_goal_xy = (float(nav_goal_xy[0]), float(nav_goal_xy[1]))
            return node

    def get_node(self, node_id: str) -> Node:
        with self._lock:
            node_id_str = str(node_id)
            for floor in self.floors.values():
                node = floor.nodes.get(node_id_str)
                if node is not None:
                    return node
            raise KeyError(node_id_str)

    def has_node(self, node_id: str) -> bool:
        with self._lock:
            node_id_str = str(node_id)
            return any(node_id_str in floor.nodes for floor in self.floors.values())

    def _move_node_to_floor(self, node: Node, floor_id: str) -> None:
        old_floor = self.floors.get(str(node.floor_id))
        if old_floor is not None:
            old_floor.nodes.pop(str(node.id), None)
        new_floor = self.floors.get(str(floor_id))
        if new_floor is None:
            new_floor = self.add_floor(floor_id=str(floor_id), height=float(node.position[2]))
        node.floor_id = str(floor_id)
        new_floor.nodes[str(node.id)] = node


    def to_dict(self) -> dict[str, object]:
        with self._lock:
            return {
                "floors": [floor.to_dict() for floor in self.floors.values()],
                "vertical_edges": [edge.to_dict() for edge in self.vertical_edges],
            }

    def snapshot(self) -> dict[str, object]:
        return self.to_dict()

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Graph":
        graph = cls()
        graph.floors = {}
        raw_floors = payload["floors"]
        if not isinstance(raw_floors, list):
            raise ValueError(f"graph.floors must be list, got {type(raw_floors).__name__}")
        for raw_floor in raw_floors:
            if not isinstance(raw_floor, dict):
                raise ValueError(f"graph floor payload must be dict, got {type(raw_floor).__name__}")
            floor = Floor.from_dict(raw_floor)
            graph.floors[str(floor.id)] = floor
        if graph.floors == {}:
            graph.floors[DEFAULT_FLOOR_ID] = Floor(id=DEFAULT_FLOOR_ID, height=0.0)
        raw_vertical_edges = payload.get("vertical_edges", [])
        if not isinstance(raw_vertical_edges, list):
            raise ValueError(f"graph.vertical_edges must be list, got {type(raw_vertical_edges).__name__}")
        parsed_vertical_edges = []
        for raw_edge in raw_vertical_edges:
            if not isinstance(raw_edge, dict):
                raise ValueError(
                    f"graph vertical edge payload must be dict, got {type(raw_edge).__name__}"
                )
            parsed_vertical_edges.append(Edge.from_dict(raw_edge))
        graph.vertical_edges = parsed_vertical_edges
        graph._node_index = _next_prefixed_index((node.id for node in graph.iter_nodes()), "n")
        graph._edge_index = _next_prefixed_index((edge.id for edge in graph.iter_edges()), "e")
        return graph


def _next_prefixed_index(ids, prefix: str) -> int:
    next_index = 0
    for raw_id in ids:
        item_id = str(raw_id)
        if not item_id.startswith(prefix):
            continue
        suffix = item_id[len(prefix):]
        if not suffix.isdigit():
            continue
        next_index = max(next_index, int(suffix) + 1)
    return next_index


def _is_vertical_relation(relation: str) -> bool:
    relation_text = str(relation).strip()
    return relation_text in {"stairs_up", "stairs_down", "vertical_transition"}

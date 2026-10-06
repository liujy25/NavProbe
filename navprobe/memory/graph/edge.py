from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from navprobe.memory.entity_knowledge import EntityKnowledge


@dataclass
class Edge:
    id: str
    src_id: str
    dst_id: str
    relation: str = ""
    traversal_count: int = 1
    path_length: float | None = None
    rgb_history_obs_ids: list[str] = field(default_factory=list)
    path_xy: list[tuple[float, float]] = field(default_factory=list)
    knowledge: list[EntityKnowledge] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Edge":
        path_length = payload.get("path_length")
        return cls(
            id=str(payload["id"]),
            src_id=str(payload["src_id"]),
            dst_id=str(payload["dst_id"]),
            relation=str(payload.get("relation", "")),
            traversal_count=int(payload.get("traversal_count", 1)),
            path_length=None if path_length is None else float(path_length),
            rgb_history_obs_ids=[
                str(obs_id)
                for obs_id in list(payload.get("rgb_history_obs_ids", []))
            ],
            path_xy=_path_xy_from_payload(payload.get("path_xy", [])),
            knowledge=[
                EntityKnowledge.from_dict(item)
                for item in list(payload.get("knowledge", []))
                if isinstance(item, dict)
            ],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "src_id": self.src_id,
            "dst_id": self.dst_id,
            "relation": self.relation,
            "traversal_count": self.traversal_count,
            "path_length": None if self.path_length is None else float(self.path_length),
            "rgb_history_obs_ids": [str(obs_id) for obs_id in self.rgb_history_obs_ids],
            "path_xy": [[float(x), float(y)] for x, y in self.path_xy],
            "knowledge": [item.to_dict() for item in self.knowledge],
        }


def _path_xy_from_payload(payload: object) -> list[tuple[float, float]]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError(f"edge.path_xy must be list when present: {payload!r}")
    return normalize_path_xy_points(payload, context="edge.path_xy")


def normalize_path_xy_points(points: Iterable[object], *, context: str) -> list[tuple[float, float]]:
    """Convert path points; callers retain their container and iteration policies."""
    normalized: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"{context} point must be length-2 list/tuple: {point!r}")
        normalized.append((float(point[0]), float(point[1])))
    return normalized

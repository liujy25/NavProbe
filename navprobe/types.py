from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field


@dataclass
class LocalmapOverlayRecord:
    overlay_id: str
    image_id: str
    source_node_id: str
    source_place_index: int
    frontier_ids: list[str]
    local_frontier_ids: list[str]
    view_overlay_ids: list[str]
    obs_ids: list[str]

    def to_dict(self, active_frontier_ids: list[str] | None = None) -> dict[str, object]:
        active_set = None if active_frontier_ids is None else {str(item) for item in active_frontier_ids}
        active_ids = list(self.frontier_ids) if active_set is None else [
            frontier_id for frontier_id in self.frontier_ids if frontier_id in active_set
        ]
        active_local_ids = [
            local_id
            for frontier_id, local_id in zip(self.frontier_ids, self.local_frontier_ids)
            if active_set is None or frontier_id in active_set
        ]
        return {
            "overlay_id": str(self.overlay_id),
            "image_id": str(self.image_id),
            "source_node_id": str(self.source_node_id),
            "source_place_index": int(self.source_place_index),
            "frontier_ids": list(self.frontier_ids),
            "local_frontier_ids": list(self.local_frontier_ids),
            "active_frontier_ids": active_ids,
            "active_local_frontier_ids": active_local_ids,
            "view_overlay_ids": list(self.view_overlay_ids),
            "obs_ids": list(self.obs_ids),
        }


@dataclass
class LocalmapFrontierRecord:
    frontier_id: str
    source_node_id: str
    source_place_index: int
    local_frontier_index: int
    overlay_id: str
    frontier_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    goal_yaw_deg: float
    path_xy: list[tuple[float, float]]
    path_length: float
    source_obs_ids: list[str]
    source_angle_degs: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LocalmapFrontierRecord":
        frontier_xy = payload.get("frontier_xy")
        if not isinstance(frontier_xy, (list, tuple)) or len(frontier_xy) != 2:
            raise ValueError(f"frontier_xy must be length-2 list/tuple: {frontier_xy!r}")
        goal_xy = payload.get("goal_xy")
        if not isinstance(goal_xy, (list, tuple)) or len(goal_xy) != 2:
            raise ValueError(f"goal_xy must be length-2 list/tuple: {goal_xy!r}")
        raw_path_xy = payload.get("path_xy", [])
        if not isinstance(raw_path_xy, list):
            raise ValueError(f"path_xy must be a list, got {type(raw_path_xy).__name__}")
        path_xy: list[tuple[float, float]] = []
        for point in raw_path_xy:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"path_xy point must be length-2 list/tuple: {point!r}")
            path_xy.append((float(point[0]), float(point[1])))
        return cls(
            frontier_id=str(payload["frontier_id"]),
            source_node_id=str(payload["source_node_id"]),
            source_place_index=int(payload["source_place_index"]),
            local_frontier_index=int(payload["local_frontier_index"]),
            overlay_id=str(payload.get("overlay_id", "")),
            frontier_xy=(float(frontier_xy[0]), float(frontier_xy[1])),
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            goal_yaw_deg=float(payload["goal_yaw_deg"]),
            path_xy=path_xy,
            path_length=float(payload["path_length"]),
            source_obs_ids=[str(obs_id) for obs_id in list(payload.get("source_obs_ids", []))],
            source_angle_degs=_normalize_angle_degs(payload.get("source_angle_degs", [])),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "frontier_id": str(self.frontier_id),
            "source_node_id": str(self.source_node_id),
            "source_place_index": int(self.source_place_index),
            "local_frontier_index": int(self.local_frontier_index),
            "overlay_id": str(self.overlay_id),
            "frontier_xy": [float(self.frontier_xy[0]), float(self.frontier_xy[1])],
            "goal_xy": [float(self.goal_xy[0]), float(self.goal_xy[1])],
            "goal_yaw_deg": float(self.goal_yaw_deg),
            "path_xy": [[float(x), float(y)] for x, y in self.path_xy],
            "path_length": float(self.path_length),
            "source_obs_ids": [str(obs_id) for obs_id in self.source_obs_ids],
            "source_angle_degs": [int(angle) for angle in self.source_angle_degs],
        }


def _normalize_angle_degs(value: list[int]) -> list[int]:
    angles: list[int] = []
    for item in value:
        angle = int(item)
        if angle not in angles:
            angles.append(angle)
    return angles


@dataclass
class LocalmapPlaceReuseState:
    place_node_id: str
    reason: str
    recovery: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "place_node_id": str(self.place_node_id),
            "reason": str(self.reason),
            "recovery": deepcopy(self.recovery),
        }

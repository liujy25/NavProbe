from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from navprobe.mapping.exploration.bev_map import FrontierCandidate


@dataclass
class FrontierEntry:
    id: str
    xy: tuple[float, float]
    goal_xy: tuple[float, float] | None = None
    goal_yaw_deg: float | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "frontier_id": self.id,
            "xy": [float(self.xy[0]), float(self.xy[1])],
        }
        if self.goal_xy is not None and self.goal_yaw_deg is not None:
            payload["goal_pose"] = {
                "x": float(self.goal_xy[0]),
                "y": float(self.goal_xy[1]),
                "yaw": float(self.goal_yaw_deg),
            }
        return payload


class FrontierBuffer:
    def __init__(self, match_distance_m: float = 0.75) -> None:
        self.match_distance_m = float(match_distance_m)
        self.entries: list[FrontierEntry] = []
        self._frontier_index = 0

    def clear(self) -> None:
        self.entries = []
        self._frontier_index = 0


    def update(self, frontiers_xy: np.ndarray) -> list[FrontierEntry]:
        previous_entries = list(self.entries)
        matched_previous: set[int] = set()
        updated_entries: list[FrontierEntry] = []

        for frontier_xy in frontiers_xy:
            frontier_xy = np.asarray(frontier_xy, dtype=np.float64)

            matched_index = None
            matched_distance = None
            for index, entry in enumerate(previous_entries):
                if index in matched_previous:
                    continue
                entry_xy = np.asarray(entry.xy, dtype=np.float64)

                distance = float(np.linalg.norm(frontier_xy - entry_xy))
                if distance > self.match_distance_m:
                    continue

                if matched_distance is None or distance < matched_distance:
                    matched_index = index
                    matched_distance = distance

            if matched_index is None:
                frontier_id = f"f{self._frontier_index}"
                self._frontier_index += 1
            else:
                previous_entry = previous_entries[matched_index]
                frontier_id = previous_entry.id
                matched_previous.add(matched_index)
            updated_entries.append(
                FrontierEntry(
                    id=frontier_id,
                    xy=(float(frontier_xy[0]), float(frontier_xy[1])),
                    goal_xy=None,
                    goal_yaw_deg=None,
                )
            )

        self.entries = updated_entries
        return list(self.entries)

    def update_navigation_goals(self, candidates: list[FrontierCandidate]) -> None:
        candidate_by_id = {str(candidate.frontier_id): candidate for candidate in candidates}

        for entry in self.entries:
            candidate = candidate_by_id.get(str(entry.id))
            if candidate is None:
                entry.goal_xy = None
                entry.goal_yaw_deg = None
                continue
            goal_xy_arr = np.asarray(candidate.goal_xy, dtype=np.float64).reshape(2)
            entry.goal_xy = (float(goal_xy_arr[0]), float(goal_xy_arr[1]))
            entry.goal_yaw_deg = float(candidate.goal_yaw_degrees())

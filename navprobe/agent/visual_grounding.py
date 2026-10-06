from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VisualWaypoint:
    obs_id: str
    angle_deg: int
    point_2d: tuple[float, float]
    point_pixel: tuple[float, float]
    raw_world_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    goal_yaw: float
    path_xy: list[tuple[float, float]] = field(default_factory=list)
    waypoint_target: str = ""
    target: str = ""
    depth_m: float = 0.0
    raw_world_z: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "obs_id": str(self.obs_id),
            "angle_deg": int(self.angle_deg),
            "point_2d": [float(self.point_2d[0]), float(self.point_2d[1])],
            "point_pixel": [float(self.point_pixel[0]), float(self.point_pixel[1])],
            "raw_world_xy": [float(self.raw_world_xy[0]), float(self.raw_world_xy[1])],
            "goal_xy": [float(self.goal_xy[0]), float(self.goal_xy[1])],
            "goal_yaw": float(self.goal_yaw),
            "path_xy": [[float(x), float(y)] for x, y in self.path_xy],
            "waypoint_target": str(self.waypoint_target),
            "target": str(self.target),
            "depth_m": float(self.depth_m),
            "raw_world_z": float(self.raw_world_z),
        }

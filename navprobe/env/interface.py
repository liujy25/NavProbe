from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Pose:
    x: float
    y: float
    z: float
    yaw: float

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "yaw": self.yaw}


@dataclass
class MoveResult:
    """Controller outcome and measured poses for one movement request."""

    success: bool
    start_pose: Pose
    pose: Pose
    nav_info: dict[str, Any] = field(default_factory=dict)

    @property
    def failure_reason(self) -> str:
        if self.success:
            return ""
        if self.nav_info.get("interrupt_reason"):
            return str(self.nav_info["interrupt_reason"])
        position = self.nav_info.get("position_nav_info", {})
        alignment = self.nav_info.get("orientation_align_info", {})
        if position.get("success") is True and alignment.get("success") is False:
            return "orientation_alignment_failed"
        return "controller_reported_failure"


@dataclass
class RawObservation:
    pose: Pose
    rgb: Any = None
    depth: Any = None
    intrinsics: list[list[float]] | None = None
    T_cam_odom: list[list[float]] | None = None
    T_odom_base: list[list[float]] | None = None
    text_hint: str = ""
    visible_objects: list[str] = field(default_factory=list)
    semantic_annotations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RawDetection:
    class_name: str
    bbox: list[float]
    score: float
    location: tuple[float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Detection3D:
    x: float
    y: float
    z: float

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


class EnvInterface(ABC):
    @abstractmethod
    def get_obs(self) -> RawObservation:
        pass

    @abstractmethod
    def turn(self, direction: str) -> Pose:
        pass

    @abstractmethod
    def move(self, x: float, y: float, yaw: float, z: float | None = None) -> MoveResult:
        pass

    @abstractmethod
    def pop_last_move_intermediate_observations(self) -> list[RawObservation]:
        pass

    @abstractmethod
    def evaluation_snapshot(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def stop(self) -> dict[str, Any]:
        pass

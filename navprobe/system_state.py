from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from navprobe.memory.task_state import NavProbeTaskStateMemory

if TYPE_CHECKING:
    from navprobe.memory.graph.graph import Graph
    from navprobe.perception.landmark_detection import LandmarkDetectionController
    from navprobe.runtime.cache import RuntimeCache


@dataclass
class GoalState:
    target: str

    def __post_init__(self) -> None:
        target = str(self.target).strip()
        if target == "":
            raise ValueError("GoalState.target must be non-empty")
        self.target = target

    def to_dict(self) -> dict[str, object]:
        return {"target": str(self.target)}


@dataclass
class SystemState:
    goal: GoalState
    graph: Graph
    landmark_controller: LandmarkDetectionController
    cache: RuntimeCache
    current_node_id: str | None = None
    current_floor_id: str = "floor_0"
    current_floor_height: float = 0.0
    task_state: NavProbeTaskStateMemory = field(default_factory=NavProbeTaskStateMemory)
    is_done: bool = False
    done_reason: str | None = None

    def set_current_node_id(self, node_id: str | None) -> None:
        if node_id is None:
            self.current_node_id = None
            return
        node_id_str = str(node_id)
        if not self.graph.has_node(node_id_str):
            raise ValueError(f"current_node_id must exist in graph: {node_id_str!r}")
        self.current_node_id = node_id_str

    def set_current_floor(self, floor_id: str, height: float | None = None) -> None:
        floor_id_str = str(floor_id).strip()
        if floor_id_str == "":
            raise ValueError("current_floor_id must be non-empty")
        if floor_id_str not in self.graph.floors:
            self.graph.add_floor(
                floor_id=floor_id_str,
                height=float(0.0 if height is None else height),
            )
        floor = self.graph.get_floor(floor_id_str)
        if height is not None:
            floor.height = float(height)
        self.current_floor_id = floor_id_str
        self.current_floor_height = float(floor.height)

    def mark_done(self, reason: str) -> None:
        reason_text = str(reason).strip()
        if reason_text == "":
            raise ValueError("done reason must be non-empty")
        self.is_done = True
        self.done_reason = reason_text

    def to_dict(self) -> dict[str, object]:
        return {
            "goal": self.goal.to_dict(),
            "current_node_id": self.current_node_id,
            "current_floor_id": self.current_floor_id,
            "current_floor_height": float(self.current_floor_height),
            "graph": self.graph.to_dict(),
            "landmark_buffer": self.landmark_controller.landmark_memory.to_context(),
            "memory": {"task_state": self.task_state.to_dict()},
            "is_done": bool(self.is_done),
            "done_reason": self.done_reason,
        }

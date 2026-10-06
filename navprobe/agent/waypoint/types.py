from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from navprobe.schemas import ActionCall

if TYPE_CHECKING:
    from navprobe.agent.state import NavProbeAgentState, NavProbeStepState
    from navprobe.agent.visual_action_context import VisualActionContext
    from navprobe.agent.navigation_decisions import NavProbeSkillDecision
    from navprobe.agent.landmark_context import NavProbeLandmarkContext
    from navprobe.memory.task_state import NavProbeTaskStateUpdateResult
    from navprobe.visualization.action_mode_overlays import BevOverlayTransform


FRONTIER_SKELETON_SAMPLE_WAYPOINT_POLICY = "frontier_skeleton_sample"
WAYPOINT_POLICY_NAMES = (
    FRONTIER_SKELETON_SAMPLE_WAYPOINT_POLICY,
)


def validate_waypoint_policy(
    *, waypoint_policy_name: str,
) -> None:
    if waypoint_policy_name not in WAYPOINT_POLICY_NAMES:
        raise ValueError(f"unsupported waypoint policy: {waypoint_policy_name!r}")


@dataclass(frozen=True)
class GroundedWaypointTarget:
    goal_xy: tuple[float, float]
    goal_yaw: float
    world_z: float | None
    policy_name: str
    source_type: str
    source_id: str = ""
    obs_id: str = ""
    angle_deg: int | None = None
    point_2d: tuple[float, float] | None = None
    raw_world_xy: tuple[float, float] | None = None

    def to_action_call(self) -> ActionCall:
        args: dict[str, object] = {
            "x": float(self.goal_xy[0]),
            "y": float(self.goal_xy[1]),
            "yaw": float(self.goal_yaw),
        }
        if self.world_z is not None:
            args["z"] = float(self.world_z)
        return ActionCall(action="move", args=args)

    def to_dict(self) -> dict[str, object]:
        return {
            "goal_xy": [float(self.goal_xy[0]), float(self.goal_xy[1])],
            "goal_yaw": float(self.goal_yaw),
            "world_z": None if self.world_z is None else float(self.world_z),
            "policy_name": str(self.policy_name),
            "source_type": str(self.source_type),
            "source_id": str(self.source_id),
            "obs_id": str(self.obs_id),
            "angle_deg": None if self.angle_deg is None else int(self.angle_deg),
            "point_2d": (
                None
                if self.point_2d is None
                else [float(self.point_2d[0]), float(self.point_2d[1])]
            ),
            "raw_world_xy": (
                None
                if self.raw_world_xy is None
                else [float(self.raw_world_xy[0]), float(self.raw_world_xy[1])]
            ),
        }


@dataclass(frozen=True)
class WaypointPolicyResult:
    policy_name: str
    target: GroundedWaypointTarget | None
    reasoning: str = ""
    failure_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_name": str(self.policy_name),
            "target": None if self.target is None else self.target.to_dict(),
            "reasoning": str(self.reasoning),
            "failure_reason": str(self.failure_reason),
        }



@dataclass(frozen=True)
class WaypointPlanningContext:
    state: NavProbeAgentState
    step: NavProbeStepState
    goal_kind: str
    visual_context: VisualActionContext
    task_state_initialization: dict[str, object]
    navigation_mode: NavProbeSkillDecision
    task_state_update_result: NavProbeTaskStateUpdateResult
    landmark_context: NavProbeLandmarkContext
    execution_visual_context: VisualActionContext | None = None
    inherited_agent_context_content: list[dict[str, object]] = field(default_factory=list)
    active_agenda_item: str = ""
    candidate_bev_landmark_markers: list[dict[str, object]] = field(default_factory=list)
    candidate_bev_base_image: object | None = None
    candidate_bev_transform: BevOverlayTransform | None = None
    candidate_bev_reference_node_marker: dict[str, object] | None = None
    candidate_bev_context_text: str = ""
    candidate_cache: dict | None = None

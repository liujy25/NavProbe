from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from navprobe.memory.graph.graph import Graph

if TYPE_CHECKING:
    from navprobe.agent.actions import AgentAction
    from navprobe.agent.node_summary import NodeSummaryDecision
    from navprobe.agent.state import NavProbeAgentState, NavProbeStepState
    from navprobe.agent.visual_action_context import VisualActionContext
    from navprobe.agent.visual_navigation import VisualNavigationDecision
    from navprobe.schemas import ActionCall, ActionResult


def _recent_edge_rgb_history_obs_ids(
    *, state: NavProbeAgentState, current_node_id: str,
) -> list[str]:
    for edge in reversed(state.graph.iter_edges(include_vertical=True)):
        if str(edge.relation) not in {"move", "stairs_up", "stairs_down"}:
            continue
        if str(edge.dst_id) == str(current_node_id):
            return [str(obs_id) for obs_id in edge.rgb_history_obs_ids]
    return []


def build_visual_context_record(
    *, state: NavProbeAgentState, context: dict[str, object],
) -> dict[str, object]:
    """Attach movement-observation references to the recorded visual context."""
    blocks: list[dict[str, object]] = []
    for item in state.node_move_history:
        if not isinstance(item, dict):
            continue
        obs_ids = [str(obs_id) for obs_id in item.get("rgb_history_obs_ids", [])]
        if obs_ids:
            blocks.append({
                "step_id": int(item.get("step_id", 0)),
                "src_node_id": str(item.get("from_node", "")),
                "dst_node_id": str(item.get("to_node", "")),
                "rgb_history_obs_ids": obs_ids,
            })
    return {
        **context,
        "recent_edge_rgb_history_obs_ids": _recent_edge_rgb_history_obs_ids(
            state=state, current_node_id=str(context["current_node_id"]),
        ),
        "edge_rgb_history_blocks": blocks,
    }


def build_visual_navigation_record(
    *, state: NavProbeAgentState, decision: VisualNavigationDecision,
) -> dict[str, object]:
    payload = decision.to_dict()
    payload["visual_context"] = build_visual_context_record(
        state=state, context=payload["visual_context"],
    )
    return payload


def build_node_summary_record(
    *, state: NavProbeAgentState, visual_context: VisualActionContext,
    decision: NodeSummaryDecision,
) -> dict[str, object]:
    return {
        "node_id": str(visual_context.current_node_id),
        "panorama_obs_ids": {
            str(view.angle_deg): str(view.obs_id) for view in visual_context.views
        },
        "edge_rgb_history_obs_ids": _recent_edge_rgb_history_obs_ids(
            state=state, current_node_id=visual_context.current_node_id,
        ),
        **decision.to_dict(),
    }


def record_agent_action(
    step: NavProbeStepState, *, action: AgentAction,
) -> None:
    """Record the selected action and the reasoning behind its selection."""
    step.agent_action = action.to_dict()


def record_waypoint_failure(
    step: NavProbeStepState, *, failure_reason: str, reason: str,
    selected_angle_deg: int | None = None,
) -> None:
    """Record a failed selection without constructing an executable action."""
    step.agent_action = {
        "action_type": "visual_waypoint",
        "args": {
            "selected_angle_deg": selected_angle_deg,
            "selected_obs_id": "",
            "target": "",
            "failure_reason": failure_reason,
        },
        "source": "visual_waypoint_policy",
        "reason": reason,
    }


def record_action_timing(
    step: NavProbeStepState, *, action_call: ActionCall, result: ActionResult,
    started_at: str, rgb_history_obs_ids: list[str], path_point_count: int,
) -> None:
    """Record the single execution segment for a Move or Stop action."""
    step.navigation_segment_timings = [{
        "segment_index": 0,
        "segment_type": str(action_call.action),
        "move_ok": bool(result.ok),
        "started_at": started_at,
        "timing": deepcopy(result.data.get("timing", {})),
        "rgb_history_obs_ids": rgb_history_obs_ids,
        "path_point_count": path_point_count,
    }]


def summarize_agent_state(
    *,
    graph: Graph,
    frontier_count: int,
    overlay_count: int,
    current_floor_id: str,
    current_place_node_id: str | None,
    previous_place_node_id: str | None,
    current_obs_id: str | None,
    finalize_reason: str,
) -> dict[str, Any]:
    place_node_count = sum(
        1
        for node in graph.iter_nodes()
        if node.node_kind == "place"
    )
    return {
        "current_floor_id": str(current_floor_id),
        "current_place_node_id": None if current_place_node_id is None else str(current_place_node_id),
        "previous_place_node_id": None if previous_place_node_id is None else str(previous_place_node_id),
        "current_obs_id": None if current_obs_id is None else str(current_obs_id),
        "place_node_count": int(place_node_count),
        "edge_count": int(len(graph.iter_edges())),
        "frontier_count": int(frontier_count),
        "overlay_count": int(overlay_count),
        "finalize_reason": str(finalize_reason),
    }


def build_agent_step_summary(
    *,
    state: "NavProbeAgentState",
    step: "NavProbeStepState",
    env_step_total: int,
) -> dict[str, object]:
    floor_id = str(state.system.current_floor_id)
    frontier_records = state.frontier_records_for_floor(floor_id)
    overlay_records = state.overlay_records_for_floor(floor_id)
    landmark_buffer = _landmark_buffer_context(state=state, include_observations=True)
    state_summary = summarize_agent_state(
        graph=state.graph,
        frontier_count=len(frontier_records),
        overlay_count=len(overlay_records),
        current_floor_id=floor_id,
        current_place_node_id=state.current_place_node_id,
        previous_place_node_id=state.previous_place_node_id,
        current_obs_id=step.current_obs_id_after,
        finalize_reason=state.finalize_reason,
    )
    decision_payload = _drop_empty(
        {
            "policy": deepcopy(step.policy_decision),
            "node_summary": deepcopy(step.node_summary),
            "agent_action": deepcopy(step.agent_action),
        }
    )
    payload = {
        "agent_step_index": int(step.place_step_index),
        "env_step_total": int(env_step_total),
        "status": "completed",
        "state": state_summary,
        "refresh_state": deepcopy(step.refresh_state_summary),
        "current_observation": {
            "current_place_node_id": None if step.current_place_node_id is None else str(step.current_place_node_id),
            "anchor_obs_id": None if step.anchor_obs_id is None else str(step.anchor_obs_id),
            "current_obs_id": None if step.current_obs_id is None else str(step.current_obs_id),
            "current_obs_id_after": None if step.current_obs_id_after is None else str(step.current_obs_id_after),
            "panorama_obs_ids": [str(obs_id) for obs_id in step.panorama_obs_ids],
            "angle_to_obs_id": {
                str(angle): str(obs_id)
                for angle, obs_id in dict(step.angle_to_obs_id).items()
            },
            "panorama_views": deepcopy(step.panorama_views),
            "panorama_angle_debug": deepcopy(step.panorama_angle_debug),
            "reused_place_obs_ids": [str(obs_id) for obs_id in step.reused_place_obs_ids],
        },
        "llm_ref": "llm/index.json",
        "decision": decision_payload,
        "map_update": {
            "created_frontier_ids": [str(frontier_id) for frontier_id in step.created_frontier_ids],
            "active_frontier_count": int(len(frontier_records)),
            "created_frontier_count": int(len(step.created_frontier_ids)),
        },
        "execution": {
            "executed_action": _executed_action_summary(step.executed_action),
            "result_count": int(len(step.results)),
            "results": _action_result_summaries(step.results),
            "navigation_segment_count": int(len(step.navigation_segment_timings)),
            "finalize_reason": str(state.finalize_reason),
        },
        "artifacts": {
            **deepcopy(step.artifact_summary),
            "maps": deepcopy(step.step_visualizations.step_visualization_images)
                if step.step_visualizations is not None else {},
            "waypoint": deepcopy(step.visual_waypoint_artifact_summary),
        },
        "timing": deepcopy(step.step_timing_payload),
    }
    if _landmark_buffer_has_content(landmark_buffer):
        payload["landmarks"] = {"buffer": landmark_buffer}
    return payload


def _is_empty_log_value(value: object) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _drop_empty(payload: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in payload.items()
        if not _is_empty_log_value(value)
    }


def _landmark_buffer_has_content(buffer: dict[str, object]) -> bool:
    return bool(buffer.get("confirmed_landmarks"))


def _landmark_buffer_context(
    *,
    state: "NavProbeAgentState",
    include_observations: bool = False,
) -> dict[str, object]:
    landmark_controller = getattr(state, "landmark_controller", None)
    if landmark_controller is None:
        return {
            "confirmed_landmarks": [],
        }
    return deepcopy(landmark_controller.landmark_memory.to_context(include_observations=include_observations))


def _executed_action_summary(executed_action: dict[str, object] | None) -> dict[str, object] | None:
    if executed_action is None:
        return None
    summary: dict[str, object] = {
        "type": str(executed_action.get("type", "")),
    }
    for key in (
        "route_completed",
        "execution",
        "origin_place_node_id",
    ):
        if key in executed_action:
            summary[key] = deepcopy(executed_action[key])
    if summary["type"] == "vertical_transition":
        for key in (
            "algorithm",
            "subgoal_id",
            "subgoal_attempt",
            "waypoint_target",
            "objective_completed",
            "executed_move_history",
            "direction",
            "before_node_id",
            "before_floor_id",
            "after_node_id",
            "after_floor_id",
            "edge_id",
            "attempts",
            "va_result_history",
            "step_decision",
            "verification",
            "failure_reason",
        ):
            if key in executed_action:
                summary[key] = deepcopy(executed_action[key])
    return summary


def _action_result_summaries(
    results: list[tuple["ActionCall", "ActionResult"]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for index, (call, result) in enumerate(results):
        step_observations = result.data.get("step_observations")
        step_observation_count = len(step_observations) if isinstance(step_observations, list) else 0
        rgb_history_obs_ids = result.data.get("rgb_history_obs_ids")
        rgb_history_obs_count = len(rgb_history_obs_ids) if isinstance(rgb_history_obs_ids, list) else 0
        summary = {
            "index": int(index),
            "call": call.to_dict(),
            "ok": bool(result.ok),
            "message": str(result.message),
            "obs_id": None if result.data.get("obs_id") is None else str(result.data.get("obs_id")),
            "step_observation_count": int(step_observation_count),
            "rgb_history_obs_count": int(rgb_history_obs_count),
            "data_keys": sorted(str(key) for key in result.data.keys()),
        }
        path_xy = _compact_path_xy(result.data.get("path_xy"))
        if path_xy != []:
            summary["path_xy"] = path_xy
        summaries.append(summary)
    return summaries


def _compact_path_xy(raw_path: object) -> list[list[float]]:
    if not isinstance(raw_path, list):
        return []
    return [
        [float(point[0]), float(point[1])]
        for point in raw_path
        if isinstance(point, (list, tuple)) and len(point) == 2
    ]

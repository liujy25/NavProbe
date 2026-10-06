"""Vertical navigation execution and floor-state updates using the support graph."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from navprobe.agent.node_moves import attach_backtrack_edge_knowledge
from navprobe.agent.visual_action_context import VisualActionContext, VisualViewContext
from navprobe.agent.vertical_transition_policy import VerticalTransitionStepDecision
from navprobe.env.episode_status import episode_end_reason
from navprobe.memory.graph.graph import Floor, Graph
from navprobe.mapping.place_graph_ops import _create_place_node
from navprobe.perception.panorama import PanoramaCaptureResult, PanoramaView, _capture_panorama
from navprobe.runtime.timing import WallStepTimer as WallStepTimer
from navprobe.schemas import ActionCall, ActionResult
from navprobe.types import LocalmapPlaceReuseState
from navprobe.agent.vertical_config import (
    VERTICAL_METHOD_PAPER_V8,
    normalize_vertical_method,
)

if TYPE_CHECKING:
    from navprobe.agent.state import NavProbeAgentContext, NavProbeAgentState, NavProbeStepState
    from navprobe.agent.stair_controller import StairController


DEFAULT_VERTICAL_FLOOR_MATCH_THRESHOLD_M = 0.8


def execute_vertical_transition_action(*, context, state, step, action):
    """Dispatch the configured VerticalMove implementation.

    The support-graph controller executes the transition. Selecting ``paper_v8``
    raises NotImplementedError because that method has no executor.
    """
    method = normalize_vertical_method(
        getattr(getattr(context, "args", None), "vertical_method", None)
    )
    if method == VERTICAL_METHOD_PAPER_V8:
        raise NotImplementedError(
            "vertical_method='paper_v8' has no HSGM/FSS executor; use "
            "'support_graph_2453f25' for the executable controller"
        )
    from navprobe.agent.stair_execution import execute_graph_stair_action

    direction = str(action.args.get("direction", "")).strip()
    if direction not in {"up", "down"}:
        raise ValueError(f"vertical_transition requires direction up or down, got {direction!r}")
    if not str(action.args.get("waypoint_target", "")).strip():
        raise ValueError("NavProbe VerticalMove requires waypoint_target")
    return execute_graph_stair_action(context=context, state=state, step=step, action=action)


def _finish_vertical_transition_action(
    *,
    context: "NavProbeAgentContext",
    state: "NavProbeAgentState",
    step: "NavProbeStepState",
    results: list[tuple[ActionCall, ActionResult]],
    segment_timings: list[dict[str, object]],
    attempts: list[dict[str, object]],
    executed_move_history: list[dict[str, object]],
    before_node_id: str,
    before_floor_id: str,
    before_floor_height: float,
    direction: str,
    floor_match_threshold_m: float,
    final_panorama: PanoramaCaptureResult,
    rgb_history_obs_ids: list[str],
    path_xy: list[tuple[float, float]],
    step_decision: VerticalTransitionStepDecision | None,
    planning_node_id: str,
    backtrack_contexts: list[dict[str, object]],
    destination_floor_reached: bool = True,
    subgoal_id: str | None = None,
    subgoal_attempt: int | None = None,
    failure_reason: str = "",
) -> None:
    completion = _complete_vertical_transition(
        graph=state.graph,
        cache=state.cache,
        before_node_id=before_node_id,
        before_floor_id=before_floor_id,
        before_floor_height=before_floor_height,
        direction=direction,
        floor_match_threshold_m=floor_match_threshold_m,
        final_panorama=final_panorama,
        rgb_history_obs_ids=rgb_history_obs_ids,
        path_xy=path_xy,
        destination_floor_reached=destination_floor_reached,
    )
    state.system.set_current_floor(
        str(completion["after_floor_id"]),
        height=float(completion["after_floor_height"]),
    )
    state.ensure_floor_runtime_state(
        str(completion["after_floor_id"]),
        context.global_bev_kwargs,
    )
    after_exploration = state.global_exploration_for_floor(str(completion["after_floor_id"]))
    state.action_executor.exploration = after_exploration
    state.current_place_node_id = str(completion["after_node_id"])
    state.previous_place_node_id = before_node_id
    state.current_obs_id = str(completion["current_obs_id_after"])
    state.reuse_current_place_next_step = LocalmapPlaceReuseState(
        place_node_id=str(completion["after_node_id"]),
        reason="vertical_transition_interrupted" if failure_reason else "vertical_transition_completed",
        recovery=deepcopy(completion),
    )
    step.current_place_node_id = str(completion["after_node_id"])
    step.current_obs_id_after = str(completion["current_obs_id_after"])
    _record_vertical_transition_node_move(
        state=state,
        step_id=int(step.place_step_index),
        before_node_id=before_node_id,
        after_node_id=str(completion["after_node_id"]),
        direction=direction,
        rgb_history_obs_ids=rgb_history_obs_ids,
        planning_node_id=planning_node_id,
        backtrack_contexts=backtrack_contexts,
        subgoal_id=subgoal_id, subgoal_attempt=subgoal_attempt,
    )
    if backtrack_contexts != []:
        edge = next(
            (
                item
                for item in state.graph.iter_edges()
                if str(item.id) == str(completion["edge_id"])
            ),
            None,
        )
        if edge is None:
            raise ValueError(
                "vertical transition edge missing after completion: "
                f"{completion['edge_id']}"
            )
        attach_backtrack_edge_knowledge(
            state=state,
            edge=edge,
            update_node_id=str(completion["after_node_id"]),
            backtrack_contexts=backtrack_contexts,
            fallback_trigger_node_id=before_node_id,
            fallback_reference_node_id=planning_node_id,
            from_node_id=before_node_id,
            to_node_id=str(completion["after_node_id"]),
        )

    completion_call = ActionCall(
        action="vertical_transition",
        args={"direction": direction, "subgoal_id": subgoal_id, "subgoal_attempt": subgoal_attempt},
    )
    status_payload = None if step_decision is None else step_decision.to_dict()
    completion_result = ActionResult(
        ok=not failure_reason,
        data={
            **completion,
            "planning_node_id": planning_node_id,
            "backtrack_contexts": [
                deepcopy(item) for item in backtrack_contexts
            ],
            "obs_id": str(completion["current_obs_id_after"]),
            "attempt_count": int(len(attempts)),
            "step_decision": status_payload,
            "failure_reason": failure_reason,
            "route_completed": not failure_reason,
            "objective_completed": not failure_reason,
            "destination_floor_reached": destination_floor_reached,
            "subgoal_id": subgoal_id,
            "subgoal_attempt": subgoal_attempt,
            "verification": None,
            "waypoint_plan": None,
            "executed_move_history": [dict(item) for item in executed_move_history],
            "va_result_history": [dict(item) for item in executed_move_history],
        },
        message=f"VerticalMove {direction}: {failure_reason or 'requested endpoint reached'}.",
    )
    results.append((completion_call, completion_result))
    step.results = results
    step.navigation_segment_timings = segment_timings
    step.executed_action = {
        "type": "vertical_transition",
        "route_completed": not failure_reason,
        "objective_completed": not failure_reason,
        "destination_floor_reached": destination_floor_reached,
        "subgoal_id": subgoal_id,
        "subgoal_attempt": subgoal_attempt,
        "failure_reason": failure_reason,
        "direction": direction,
        "before_node_id": before_node_id,
        "planning_node_id": planning_node_id,
        "before_floor_id": before_floor_id,
        "after_node_id": str(completion["after_node_id"]),
        "after_floor_id": str(completion["after_floor_id"]),
        "edge_id": str(completion["edge_id"]),
        "backtrack_contexts": [
            deepcopy(item) for item in backtrack_contexts
        ],
        "attempts": attempts,
        "executed_move_history": [dict(item) for item in executed_move_history],
        "va_result_history": [dict(item) for item in executed_move_history],
        "step_decision": status_payload,
        "verification": None,
        "waypoint_plan": None,
    }
    state.last_executed_action = step.executed_action


def _vertical_panorama_from_current_step(
    step: "NavProbeStepState",
) -> PanoramaCaptureResult | None:
    raw_views = list(getattr(step, "panorama_views", []))
    if raw_views == []:
        return None
    views: list[PanoramaView] = []
    for raw_view in raw_views:
        if not isinstance(raw_view, dict):
            return None
        try:
            view = PanoramaView(
                angle_deg=int(raw_view["angle_deg"]),
                obs_id=str(raw_view["obs_id"]),
                rgb_id=str(raw_view["rgb_id"]),
                depth_id=str(raw_view["depth_id"]),
                pose=dict(raw_view.get("pose", {})),
                yaw_debug=dict(raw_view.get("yaw_debug", {})),
            )
        except (KeyError, TypeError, ValueError):
            return None
        views.append(view)
    angle_to_obs_id = {
        int(angle): str(obs_id)
        for angle, obs_id in dict(getattr(step, "angle_to_obs_id", {})).items()
    }
    if angle_to_obs_id == {}:
        angle_to_obs_id = {int(view.angle_deg): str(view.obs_id) for view in views}
    anchor_obs_id = str(getattr(step, "anchor_obs_id", "") or "")
    if anchor_obs_id == "":
        anchor_obs_id = str(angle_to_obs_id.get(0, views[0].obs_id))
    current_obs_id = str(getattr(step, "current_obs_id", "") or anchor_obs_id)
    return PanoramaCaptureResult(
        views=views,
        angle_to_obs_id=angle_to_obs_id,
        anchor_obs_id=anchor_obs_id,
        current_obs_id=current_obs_id,
        angle_debug=dict(getattr(step, "panorama_angle_debug", {})),
    )


def _capture_vertical_panorama(
    *,
    context: "NavProbeAgentContext",
    state: "NavProbeAgentState",
) -> PanoramaCaptureResult:
    if episode_end_reason(context.env) is not None:
        # Keep the final physical observation without issuing turns after the
        # environment has exhausted its action budget.
        observation = context.env.get_obs()
        record = state.cache.store_observation(observation)
        view = PanoramaView(
            angle_deg=0, obs_id=record.id, rgb_id=f"{record.id}:rgb",
            depth_id=f"{record.id}:depth", pose=observation.pose.to_dict(),
        )
        return PanoramaCaptureResult(
            views=[view], angle_to_obs_id={0: record.id},
            anchor_obs_id=record.id, current_obs_id=record.id, angle_debug={},
        )
    return _capture_panorama(
        env=context.env,
        cache=state.cache,
        global_exploration=None,
        landmark_controller=None,
        panorama_config=context.panorama_config,
        stop_capture=lambda: episode_end_reason(context.env) is not None,
    )


def _visual_context_from_panorama(
    *,
    current_node_id: str,
    panorama: PanoramaCaptureResult,
) -> VisualActionContext:
    views = [
        VisualViewContext(
            angle_deg=int(view.angle_deg),
            obs_id=str(view.obs_id),
            rgb_id=str(view.rgb_id),
            depth_id=str(view.depth_id),
            pose=dict(view.pose),
        )
        for view in panorama.views
    ]
    return VisualActionContext(
        current_node_id=str(current_node_id),
        views=views,
    )


def _execute_vt_move(
    *,
    state: "NavProbeAgentState",
    move_call: ActionCall,
    vt_exploration: StairController,
) -> ActionResult:
    previous_exploration = state.action_executor.exploration
    previous_handler = state.action_executor.step_observation_handler
    previous_enabled_getter = state.action_executor.step_observation_enabled_getter
    try:
        state.action_executor.exploration = vt_exploration
        state.action_executor.step_observation_handler = None
        state.action_executor.step_observation_enabled_getter = None
        return state.action_executor.execute(move_call)
    finally:
        state.action_executor.exploration = previous_exploration
        state.action_executor.step_observation_handler = previous_handler
        state.action_executor.step_observation_enabled_getter = previous_enabled_getter


def _extend_unique_obs_ids(target: list[str], raw_obs_ids: object) -> None:
    if not isinstance(raw_obs_ids, list):
        return
    for obs_id in raw_obs_ids:
        obs_id_text = str(obs_id).strip()
        if obs_id_text == "":
            continue
        if obs_id_text not in target:
            target.append(obs_id_text)


def _extend_path_xy(target: list[tuple[float, float]], raw_path_xy: object) -> None:
    if not isinstance(raw_path_xy, list):
        return
    for point in raw_path_xy:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        normalized = (float(point[0]), float(point[1]))
        if target != [] and target[-1] == normalized:
            continue
        target.append(normalized)


def _record_vertical_transition_node_move(
    *,
    state: "NavProbeAgentState",
    step_id: int,
    before_node_id: str,
    after_node_id: str,
    direction: str,
    rgb_history_obs_ids: list[str],
    planning_node_id: str,
    backtrack_contexts: list[dict[str, object]],
    subgoal_id: str | None = None,
    subgoal_attempt: int | None = None,
) -> None:
    if str(before_node_id).strip() == "" or str(after_node_id).strip() == "":
        return
    if str(before_node_id) == str(after_node_id):
        return
    record = {
        "step_id": int(step_id),
        "from_node": str(before_node_id),
        "to_node": str(after_node_id),
        "move_mode": f"vertical_transition_{direction}",
        "reason": f"vertical_transition_{direction}",
        "rgb_history_obs_ids": [str(obs_id) for obs_id in rgb_history_obs_ids],
    }
    if subgoal_id is not None:
        record.update(subgoal_id=subgoal_id, subgoal_attempt=subgoal_attempt)
    if backtrack_contexts != []:
        record["backtrack_reference_node_id"] = str(planning_node_id)
        record["backtrack_contexts"] = [
            deepcopy(item) for item in backtrack_contexts
        ]
    history = list(getattr(state, "node_move_history", []))
    history.append(record)
    state.node_move_history = history


def _complete_vertical_transition(
    *,
    graph: Graph,
    cache,
    before_node_id: str,
    before_floor_id: str,
    before_floor_height: float,
    direction: str,
    floor_match_threshold_m: float,
    final_panorama: PanoramaCaptureResult,
    rgb_history_obs_ids: list[str],
    path_xy: list[tuple[float, float]],
    destination_floor_reached: bool = True,
) -> dict[str, object]:
    anchor_obs_id = str(final_panorama.anchor_obs_id)
    anchor_observation = cache.get_observation(anchor_obs_id).observation
    after_height = float(anchor_observation.pose.z)
    if destination_floor_reached:
        after_floor, created_floor = _resolve_vertical_transition_floor(
            graph=graph,
            before_floor_id=before_floor_id,
            after_height=after_height,
            floor_match_threshold_m=float(floor_match_threshold_m),
        )
    else:
        after_floor, created_floor = graph.floors[before_floor_id], False
    after_node_id = _create_place_node(
        graph=graph,
        cache=cache,
        anchor_obs_id=anchor_obs_id,
        obs_ids=final_panorama.obs_ids,
        floor_id=str(after_floor.id),
    )
    relation = "stairs_up" if direction == "up" else "stairs_down"
    edge = graph.add_edge(
        src_id=str(before_node_id),
        dst_id=str(after_node_id),
        relation=relation,
        rgb_history_obs_ids=[str(obs_id) for obs_id in rgb_history_obs_ids],
        path_xy=[(float(x), float(y)) for x, y in path_xy],
    )
    return {
        "before_node_id": str(before_node_id),
        "before_floor_id": str(before_floor_id),
        "before_floor_height": float(before_floor_height),
        "after_node_id": str(after_node_id),
        "after_floor_id": str(after_floor.id),
        "after_floor_height": float(after_floor.height),
        "created_floor": bool(created_floor),
        "destination_floor_reached": bool(destination_floor_reached),
        "edge_id": str(edge.id),
        "edge_relation": str(edge.relation),
        "current_obs_id_after": anchor_obs_id,
        "final_panorama_obs_ids": final_panorama.obs_ids,
        "rgb_history_obs_ids": [str(obs_id) for obs_id in rgb_history_obs_ids],
        "path_xy": [[float(x), float(y)] for x, y in path_xy],
        "register_frontiers_on_reuse": True,
    }


def _resolve_vertical_transition_floor(
    *,
    graph: Graph,
    before_floor_id: str,
    after_height: float,
    floor_match_threshold_m: float,
) -> tuple[Floor, bool]:
    threshold = float(floor_match_threshold_m)
    matches: list[tuple[float, str, Floor]] = []
    for floor_id, floor in graph.floors.items():
        if str(floor_id) == str(before_floor_id):
            continue
        floor_height = float(floor.height)
        distance = abs(floor_height - float(after_height))
        if distance <= threshold:
            matches.append((float(distance), str(floor.id), floor))
    if matches != []:
        matches.sort(key=lambda item: (item[0], item[1]))
        matched_floor = matches[0][2]
        matched_floor.height = float(after_height)
        return matched_floor, False
    return graph.add_floor(height=float(after_height)), True


def _current_episode_step(context: "NavProbeAgentContext") -> int:
    info = context.env.current_episode_info()
    return int(info.get("pointnav_step_total", 0))

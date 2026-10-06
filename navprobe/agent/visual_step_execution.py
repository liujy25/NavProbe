from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING

from navprobe.agent.actions import AgentAction, AgentActionType
from navprobe.agent.node_moves import set_pending_node_move
from navprobe.agent.visual_navigation import VisualNavigationDecision, plan_visual_navigation_action
from navprobe.agent.vertical_transition import execute_vertical_transition_action
from navprobe.logging.agent_logging import (
    build_visual_navigation_record,
    record_action_timing,
    record_agent_action,
    record_waypoint_failure,
)
from navprobe.schemas import ActionCall
from navprobe.types import LocalmapPlaceReuseState

if TYPE_CHECKING:
    from navprobe.agent.state import NavProbeAgentContext, NavProbeAgentState


def _execute_terminal_stop(
    *,
    state: "NavProbeAgentState",
    step,
    terminal_check: dict[str, object],
) -> None:
    """Execute Stop after the Executive confirms the terminal constraints."""
    action = AgentAction(
        action_type=AgentActionType.FINALIZE,
        args={"reason": "done"},
        source="task_state_terminal_check",
        reason=str(terminal_check.get("reason", "")),
    )
    record_agent_action(step, action=action)
    step.policy_decision = {
        "source": "task_state_terminal_check",
        "decision": {**step.visual_decision, "terminal_check": deepcopy(terminal_check)},
    }
    action_call = ActionCall(action="done", args={})
    segment_timer_started = datetime.now().isoformat()
    result = state.action_executor.execute(action_call)
    step.results = [(action_call, result)]
    record_action_timing(
        step, action_call=action_call, result=result,
        started_at=segment_timer_started, rgb_history_obs_ids=[], path_point_count=0,
    )
    stop_result = result.data.get("stop_result")
    if isinstance(stop_result, dict):
        state.stop_result = dict(stop_result)
        state.stop_result["agent_declared_done"] = True
    step.executed_action = {
        "type": "task_state_terminal_done",
        "pending_stop": {},
        "terminal_check": deepcopy(terminal_check),
        "done_call": action_call.to_dict(),
        "done_result": result.to_dict(),
    }
    state.last_executed_action = step.executed_action
    state.pending_terminal_check = None
    state.finalize_reason = "done"


def run_visual_waypoint_decision(
    *,
    context: "NavProbeAgentContext",
    state: "NavProbeAgentState",
    step,
) -> None:
    decision = plan_visual_navigation_action(
        state=state,
        step=step,
        goal=context.goal_spec,
    )
    decision_payload = build_visual_navigation_record(state=state, decision=decision)
    if step.episodic_retrieval:
        decision_payload["episodic_retrieval"] = deepcopy(step.episodic_retrieval)
    step.visual_decision = decision_payload
    step.policy_decision = {
        "source": "visual_waypoint_policy",
        "decision": decision_payload,
    }
    if decision.navigation_mode is None:
        if str(decision.failure_reason).strip() != "":
            failure_reason = str(decision.failure_reason)
            record_waypoint_failure(step, failure_reason=failure_reason, reason=failure_reason)
            state.finalize_reason = (
                f"visual_waypoint_failed:{failure_reason}"
            )
            return
        if (
            decision.action_call is None
            or decision.action_call.action != "done"
            or str(decision.terminal_check.get("decision", "")) != "done"
        ):
            raise ValueError("task-state terminal decision requires a done action")
        _execute_terminal_stop(
            state=state,
            step=step,
            terminal_check=decision.terminal_check,
        )
        return
    if (
        decision.navigation_mode.action_mode == "approach_to_stop"
        and decision.navigation_mode.approach_movement == "stay"
        and decision.action_call is None
    ):
        _mark_selected_subgoal_started(state=state, step=step, navigation_mode=decision.navigation_mode)
        current_node_id = str(step.current_place_node_id)
        reason = str(decision.navigation_mode.reasoning_action)
        pending_stop = {
            "requested_at_step": int(step.place_step_index),
            "approach_completed_at_step": int(step.place_step_index),
            "origin_place_node_id": current_node_id,
            "movement": "stay",
            "stop_objective": str(decision.navigation_mode.stop_objective),
            "reasoning": reason,
            "rgb_history_obs_ids": [],
        }
        action_payload = {
            "action_type": "approach_to_stop",
            "args": {"movement": "stay"},
            "source": "visual_waypoint_policy",
            "reason": reason,
        }
        step.agent_action = deepcopy(action_payload)
        step.results = []
        step.navigation_segment_timings = []
        step.executed_action = {
            "type": "vln_approach_to_stop_stay",
            "movement": "stay",
            "origin_place_node_id": current_node_id,
            "decision": deepcopy(decision_payload),
            "pending_terminal_check": deepcopy(pending_stop),
            "rgb_history_obs_ids": [],
            "path_xy": [],
        }
        state.pending_terminal_check = pending_stop
        state.reuse_current_place_next_step = LocalmapPlaceReuseState(
            place_node_id=current_node_id,
            reason="approach_to_stop_stay",
        )
        state.last_executed_action = step.executed_action
        return
    if decision.action_call is None:
        failure_reason = str(decision.failure_reason or "visual_waypoint_no_action")
        record_waypoint_failure(
            step, failure_reason=failure_reason,
            reason=str(decision.navigation_mode.reasoning_action),
            selected_angle_deg=(
                None if decision.local_move_plan is None else decision.local_move_plan.selected_angle_deg
            ),
        )
        state.finalize_reason = f"visual_waypoint_failed:{failure_reason}"
        return

    if decision.action_call.action == "vertical_transition":
        decision_payload["task_state_before_action"] = state.system.task_state.to_dict()
        direction = str(decision.action_call.args.get("direction", "")).strip()
        vertical_args: dict[str, object] = {"direction": direction}
        for key in ("waypoint_target", "subgoal_id", "subgoal_attempt", "task_context"):
            if key in decision.action_call.args:
                vertical_args[key] = decision.action_call.args[key]
        if str(decision.planning_current_node_id).strip() != "":
            vertical_args["planning_node_id"] = str(
                decision.planning_current_node_id
            )
        if decision.backtrack_contexts != []:
            vertical_args["backtrack_contexts"] = [
                deepcopy(item) for item in decision.backtrack_contexts
            ]
        agent_action = AgentAction(
            action_type=AgentActionType.VERTICAL_TRANSITION,
            args=vertical_args,
            source="visual_waypoint_policy",
            reason=str(decision.navigation_mode.reasoning_action),
        )
        record_agent_action(
            step, action=agent_action,
        )
        execute_vertical_transition_action(
            context=context,
            state=state,
            step=step,
            action=agent_action,
        )
        return

    if decision.grounded_waypoint_target is None:
        raise ValueError("visual waypoint action requires a grounded waypoint target")
    grounded_target = decision.grounded_waypoint_target
    if decision.local_move_plan is None or decision.local_move_plan.selected_angle_deg is None:
        raise ValueError("grounded waypoint requires local_move_plan.selected_angle_deg")
    stop_approach = (
        str(decision.navigation_mode.action_mode)
        == "approach_to_stop"
    )
    navigation_action_mode = str(decision.navigation_mode.action_mode)
    decision_type = (
        navigation_action_mode
        if stop_approach or navigation_action_mode == "backtrack"
        else "go_to_waypoint"
    )
    agent_action = AgentAction(
        action_type=AgentActionType.VISUAL_WAYPOINT,
        args={
            "selected_angle_deg": int(decision.local_move_plan.selected_angle_deg),
            "selected_obs_id": str(grounded_target.obs_id),
            "target": str(decision.local_move_plan.waypoint_target),
            "stop_approach": stop_approach,
            "decision_type": decision_type,
            "waypoint_policy": str(grounded_target.policy_name),
        },
        source="visual_waypoint_policy",
        reason=str(decision.navigation_mode.action_reason or decision.navigation_mode.reasoning_action),
    )
    record_agent_action(
        step, action=agent_action,
    )
    _execute_visual_action_call(
        state=state,
        step=step,
        action=agent_action,
        decision=decision,
        decision_payload=decision_payload,
    )


def _mark_selected_subgoal_started(*, state, step, navigation_mode) -> None:
    if navigation_mode.subgoal_id is None:
        return
    task_state = state.system.task_state
    if task_state.agenda_initialized:
        item = next((item for item in task_state.agenda if item.subgoal_id == navigation_mode.subgoal_id), None)
        if item is None or item.attempt != navigation_mode.subgoal_attempt:
            raise ValueError("cannot execute an inactive or stale subgoal attempt")
        task_state.mark_subgoal_started(
            navigation_mode.subgoal_id, str(step.current_place_node_id),
        )


def _execute_visual_action_call(
    *,
    state: "NavProbeAgentState",
    step,
    action: AgentAction,
    decision: VisualNavigationDecision,
    decision_payload: dict[str, object],
) -> None:
    """Execute the runtime decision; retain its serialized payload only for recording."""
    action_call = decision.action_call
    if action_call is None:
        raise ValueError("waypoint execution requires an action call")
    navigation_mode = decision.navigation_mode
    subgoal_id = None if navigation_mode is None else navigation_mode.subgoal_id
    subgoal_attempt = None if navigation_mode is None else navigation_mode.subgoal_attempt
    if subgoal_id is not None:
        task_state = state.system.task_state
        item = next((item for item in task_state.agenda if item.subgoal_id == subgoal_id), None)
        if item is None or item.attempt != subgoal_attempt:
            raise ValueError("cannot execute an inactive or stale subgoal attempt")
        task_state.mark_subgoal_started(subgoal_id, str(step.current_place_node_id))
        action.args.update({"subgoal_id": subgoal_id, "subgoal_attempt": subgoal_attempt})
        step.agent_action = action.to_dict()
    segment_timer_started = datetime.now().isoformat()
    result = state.action_executor.execute(action_call)
    step.results = [(action_call, result)]
    rgb_history_obs_ids = [str(obs_id) for obs_id in result.data.get("rgb_history_obs_ids", [])]
    path_xy = [[float(x), float(y)] for x, y in result.data["path_xy"]]
    record_action_timing(
        step, action_call=action_call, result=result,
        started_at=segment_timer_started, rgb_history_obs_ids=rgb_history_obs_ids,
        path_point_count=len(path_xy),
    )
    step.executed_action = {
        "type": "move_to_visual_waypoint",
        "route_completed": bool(result.ok),
        "reason": str(action.reason),
        **({"subgoal_id": subgoal_id, "subgoal_attempt": subgoal_attempt} if subgoal_id is not None else {}),
        "origin_place_node_id": str(step.current_place_node_id),
        "decision": deepcopy(decision_payload),
        "move_call": action_call.to_dict(),
        "move_result": result.to_dict(),
        "rgb_history_obs_ids": rgb_history_obs_ids,
        "path_xy": path_xy,
    }
    state.last_executed_action = step.executed_action
    # A partial failed move still physically connects its origin to the next
    # observed place. Do not leave a previous segment's origin in state.
    if result.data["position_changed"]:
        state.previous_place_node_id = str(step.current_place_node_id)
        decision_type = str(action.args.get("decision_type", "")).strip()
        move_mode = decision_type if decision_type != "" else "visual_waypoint"
        backtrack_reference_node_id = ""
        # The decision retains all attempted references for audit. Only the
        # final planning reference and its surviving contexts own this move.
        planning_node_id = str(decision.planning_current_node_id).strip()
        backtrack_contexts = []
        if planning_node_id and planning_node_id != str(step.current_place_node_id):
            move_mode = "backtrack"
            backtrack_reference_node_id = planning_node_id
            backtrack_contexts = [
                item for item in decision.backtrack_contexts
                if not (isinstance(item.get("result"), dict)
                        and item["result"].get("status") == "failed")
            ]
        set_pending_node_move(
            state=state,
            step_id=int(step.place_step_index),
            from_node_id=str(step.current_place_node_id),
            reason=str(action.reason),
            move_mode=move_mode,
            backtrack_reference_node_id=backtrack_reference_node_id,
            subgoal_id=subgoal_id,
            subgoal_attempt=subgoal_attempt,
            backtrack_contexts=backtrack_contexts,
            rgb_history_obs_ids=rgb_history_obs_ids,
            path_xy=path_xy,
            execution_outcome={
                "route_completed": False,
                "failure_reason": result.data["failure_reason"],
                "requested_target": action_call.args,
                "actual_pose": result.data["pose"],
            } if not result.ok else None,
        )
    else:
        # Arrival success may require zero motion when already within tolerance.
        # A fresh observation can create a node, but must not invent a move edge.
        state.previous_place_node_id = None
        state.pending_node_move = None

    if bool(action.args.get("stop_approach", False)) and result.ok:
        local_move_plan = decision.local_move_plan
        waypoint = decision.waypoint
        pending_stop = {
            "requested_at_step": int(step.place_step_index),
            "approach_completed_at_step": int(step.place_step_index),
            "origin_place_node_id": str(step.current_place_node_id),
            "stop_objective": (
                str(navigation_mode.stop_objective)
                if navigation_mode is not None
                else ""
            ),
            "waypoint_target": (
                str(local_move_plan.waypoint_target)
                if local_move_plan is not None
                else str(action.reason)
            ),
            "reasoning": (
                str(navigation_mode.reasoning_action)
                if navigation_mode is not None
                else ""
            ),
            "selected_angle_deg": action.args.get("selected_angle_deg"),
            "selected_obs_id": action.args.get("selected_obs_id"),
            "target": action.args.get("target"),
            "waypoint_obs_id": (
                str(waypoint.obs_id)
                if waypoint is not None
                else ""
            ),
            "waypoint_point_2d": (
                [float(value) for value in waypoint.point_2d]
                if waypoint is not None
                else None
            ),
            "waypoint_point_pixel": (
                [float(value) for value in waypoint.point_pixel]
                if waypoint is not None
                else None
            ),
            "grounded_waypoint_target": (
                str(waypoint.target)
                if waypoint is not None
                else ""
            ),
            "rgb_history_obs_ids": [str(obs_id) for obs_id in rgb_history_obs_ids],
        }
        pending_stop["movement"] = "move"
        state.pending_terminal_check = pending_stop

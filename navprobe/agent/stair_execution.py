"""Execute support-graph floor transitions for the active subgoal."""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from navprobe.agent.stair_controller import StairController
from navprobe.agent.vertical_config import NAVPROBE_VERTICAL_MAX_MOVES, vertical_execution_configuration
from navprobe.agent.vertical_transition_policy import VerticalTransitionStepDecision
from navprobe.schemas import ActionCall, ActionResult


def stair_request_error(state, subgoal_id, subgoal_attempt=None):
    task_state = state.system.task_state
    if subgoal_id is None:
        return "missing_stair_subgoal_id" if task_state.agenda else None
    if not isinstance(subgoal_id, str) or not subgoal_id:
        return "invalid_stair_subgoal_id"
    task_state.ensure_subgoal_ids()
    item = next((item for item in task_state.agenda if item.subgoal_id == subgoal_id), None)
    if item is None:
        return "invalid_stair_subgoal_id"
    if type(subgoal_attempt) is not int or subgoal_attempt != item.attempt:
        return "stale_stair_subgoal_attempt"
    if f"{subgoal_id}:{subgoal_attempt}" in state.completed_stair_items:
        return "already_completed"
    if item.status != "active":
        return "stair_subgoal_not_active"
    return None


def stair_task_context(state, action):
    subgoal_id = action.args.get("subgoal_id")
    if subgoal_id is None:
        heading = (
            f"Floor transition: {action.args['direction']} to the first destination floor.\n"
            f"Task objective: {action.args.get('waypoint_target', '')}\n"
            f"Skill selection reason: {action.args.get('reason', '')}\n"
        )
    else:
        heading = f"Selected single-floor stair subgoal: {subgoal_id}\n"
    return heading + state.system.task_state.format_for_prompt()


def restore_completed_stair_items(executed_actions):
    records = {}
    for action in executed_actions:
        if not isinstance(action, dict):
            continue
        subgoal_id, attempt = action.get("subgoal_id"), action.get("subgoal_attempt")
        if (
            isinstance(subgoal_id, str) and subgoal_id
            and type(attempt) is int and attempt > 0
            and action.get("route_completed") and action.get("edge_id")
            and action.get("type") == "vertical_transition"
        ):
            records[f"{subgoal_id}:{attempt}"] = {
                key: deepcopy(action.get(key))
                for key in (
                    "subgoal_id", "subgoal_attempt", "direction",
                    "before_node_id", "after_node_id", "before_floor_id",
                    "after_floor_id", "edge_id",
                )
            }
    return records


def execute_graph_stair_action(*, context, state, step, action):
    from navprobe.agent import vertical_transition as vt

    direction = str(action.args.get("direction", ""))
    if direction not in {"up", "down"}:
        raise ValueError("Stair direction must be up or down")
    execution_config = vertical_execution_configuration(
        getattr(getattr(context, "args", None), "vertical_method", None)
    )
    subgoal_id = action.args.get("subgoal_id")
    subgoal_attempt = action.args.get("subgoal_attempt")
    binding = {"subgoal_id": subgoal_id, "subgoal_attempt": subgoal_attempt}
    error = stair_request_error(state, subgoal_id, subgoal_attempt)
    if error is not None:
        already = error == "already_completed"
        data = {
            "type": "vertical_transition",
            "algorithm": "graph_memory_2453f25",
            "execution_config": execution_config,
            **binding,
            "direction": direction,
            "route_completed": already,
            "objective_completed": already,
            "destination_floor_reached": already,
            "executed": False,
            "status": error,
        }
        if already:
            data["completion_evidence"] = deepcopy(state.completed_stair_items[f"{subgoal_id}:{subgoal_attempt}"])
        step.executed_action = data
        step.results = [
            (
                ActionCall(action="vertical_transition", args=dict(action.args)),
                ActionResult(ok=already, data=data, message=error),
            )
        ]
        state.last_executed_action = data
        if not already:
            state.finalize_reason = f"vertical_transition_failed:{error}"
        return

    override = action.args.get(
        "max_steps", getattr(context.args, "max_vertical_transition_steps", None)
    )
    max_moves = NAVPROBE_VERTICAL_MAX_MOVES if override is None else int(override)
    if max_moves <= 0:
        raise ValueError("max_vertical_transition_steps must be positive")
    budget = int(getattr(context.env, "max_episode_steps", 0))
    before_node = str(step.current_place_node_id or state.current_place_node_id)
    before_floor = str(state.system.current_floor_id)
    before_height = float(state.system.current_floor_height)
    planning_node = str(action.args.get("planning_node_id", before_node))
    backtracks = deepcopy(action.args.get("backtrack_contexts", []))
    attempts, results, timings, histories, history_ids, path_xy = [], [], [], [], [], []
    policy = None
    failure = "vertical_transition_exhausted"
    step.policy_decision["vertical_transition"] = {
        "algorithm": "graph_memory_2453f25",
        "execution_config": execution_config,
        **binding,
        "direction": direction,
        "max_steps": max_moves,
        "before_node_id": before_node,
        "planning_node_id": planning_node,
    }

    def remaining():
        return budget - vt._current_episode_step(context) if budget > 0 else float("inf")

    def capture():
        turns = (
            context.panorama_config.observation_count
            * context.panorama_config.turns_per_observation
        )
        if remaining() < turns:
            raise ValueError("episode_over")
        return vt._capture_vertical_panorama(context=context, state=state)

    def move(path, yaw, receiver, *, short_goal=True):
        if remaining() <= 0:
            raise ValueError("episode_over")
        call = ActionCall(
            action="move",
            args={
                "x": float(path[-1][0]),
                "y": float(path[-1][1]),
                "yaw": float(yaw),
                "z": float(path[-1][2]),
            },
        )
        # Habitat exposes the per-goal budget through its local controller.
        controller = getattr(
            getattr(context.env, "client", None), "nav_controller", None
        )
        old_limit = None if controller is None else controller.max_steps
        if controller is not None:
            controller.max_steps = min(35 if short_goal else old_limit, remaining())
        timer = vt.WallStepTimer(env_step_start=vt._current_episode_step(context))
        try:
            result = vt._execute_vt_move(
                state=state, move_call=call, vt_exploration=receiver
            )
        finally:
            if controller is not None:
                controller.max_steps = old_limit
        results.append((call, result))
        timings.append(
            {
                "segment_index": len(histories),
                "segment_type": (
                    "stair_graph_move" if short_goal else "backtrack_to_stair_anchor"
                ),
                "timing": timer.finish(vt._current_episode_step(context)),
            }
        )
        vt._extend_unique_obs_ids(
            history_ids, result.data.get("rgb_history_obs_ids", [])
        )
        vt._extend_path_xy(path_xy, result.data.get("path_xy", []))
        histories.append({"move_call": call.to_dict(), "move_result": result.to_dict()})
        if not result.ok:
            raise ValueError("move_failed")
        return result

    try:
        if planning_node != before_node:
            node = state.graph.get_node(planning_node)
            if str(node.floor_id) != before_floor:
                raise ValueError("backtrack_anchor_not_on_source_floor")
            pose = context.env.get_obs().pose
            target = np.asarray(node.position, dtype=float).copy()
            if node.nav_goal_xy is not None:
                target[:2] = node.nav_goal_xy
            move(
                np.array([[pose.x, pose.y, pose.z], target]),
                node.yaw,
                state.global_exploration_for_floor(before_floor),
                short_goal=False,
            )
            pose = context.env.get_obs().pose
            if (
                np.linalg.norm(np.array([pose.x, pose.y]) - target[:2]) > 0.3
                or abs(pose.z - target[2]) > 0.2
            ):
                raise ValueError("backtrack_anchor_not_reached")
        panorama = (
            None
            if planning_node != before_node
            else vt._vertical_panorama_from_current_step(step)
        )
        if panorama is None:
            panorama = capture()
        observations = [
            state.cache.get_observation(v.obs_id).observation for v in panorama.views
        ]
        pose = state.cache.get_observation(panorama.anchor_obs_id).observation.pose
        task_context = stair_task_context(state, action)
        policy = StairController(direction, [pose.x, pose.y, pose.z], task_context)
        blocked = 0
        for cycle in range(max_moves + 1):
            end_reason = vt.episode_end_reason(context.env)
            if end_reason is not None:
                failure = end_reason
                break
            policy.observe(observations)
            visual = vt._visual_context_from_panorama(
                current_node_id=planning_node, panorama=panorama
            )
            selected = policy.decide(
                pose, state.llm_client, state.cache, visual
            )
            record = {
                "cycle": cycle,
                "pose": pose.to_dict(),
                **deepcopy(policy.last_decision),
            }
            attempts.append(record)
            if policy.failure_reason:
                failure = policy.failure_reason
                break
            if policy.arrived:
                decision = VerticalTransitionStepDecision(
                    transition_status="complete",
                    waypoint_target="",
                    selected_angle_deg=None,
                    reason=str(
                        (record.get("semantic") or {}).get(
                            "reason", "Confirmed destination surface"
                        )
                    ),
                )
                vt._finish_vertical_transition_action(
                    context=context,
                    state=state,
                    step=step,
                    results=results,
                    segment_timings=timings,
                    attempts=attempts,
                    executed_move_history=histories,
                    before_node_id=before_node,
                    before_floor_id=before_floor,
                    before_floor_height=before_height,
                    direction=direction,
                    floor_match_threshold_m=float(
                        getattr(context.args, "vertical_floor_match_threshold_m", 0.8)
                    ),
                    final_panorama=panorama,
                    rgb_history_obs_ids=history_ids,
                    path_xy=path_xy,
                    step_decision=decision,
                    planning_node_id=planning_node,
                    backtrack_contexts=backtracks,
                    **binding,
                )
                step.executed_action.update(
                    algorithm="graph_memory_2453f25",
                    execution_config=execution_config,
                    **binding,
                    transition_state=policy.state(),
                    trace_xyz=np.asarray(policy.trace).tolist(),
                )
                results[-1][1].data.update(
                    **binding, algorithm="graph_memory_2453f25"
                )
                state.completed_stair_items.update(
                    restore_completed_stair_items([step.executed_action])
                )
                return
            if cycle >= max_moves:
                break
            if selected is None:
                blocked += 1
                if blocked >= 2:
                    failure = "no_continuation"
                    break
            else:
                path = (
                    np.asarray(selected["path_xyz"])
                    if record.get("entrance_candidate_id") is not None
                    else policy.short_path(selected, pose)
                )
                before = np.array([pose.x, pose.y, pose.z])
                yaw = math.degrees(
                    math.atan2(path[-1, 1] - pose.y, path[-1, 0] - pose.x)
                )
                movement = move(path, yaw, policy)
                after_pose = context.env.get_obs().pose
                after = np.array([after_pose.x, after_pose.y, after_pose.z])
                policy.reached(selected, before, after)
                policy.observe_raw_observation(context.env.get_obs())
                record.update(
                    selected_id=selected["label"], movement=movement.to_dict()
                )
                blocked = (
                    blocked + 1 if np.linalg.norm(after[:2] - before[:2]) < 0.06 else 0
                )
                if blocked >= 3:
                    failure = "repeated_no_progress"
                    break
            panorama = capture()
            observations = [
                state.cache.get_observation(v.obs_id).observation
                for v in panorama.views
            ]
            pose = state.cache.get_observation(panorama.anchor_obs_id).observation.pose
    except ValueError as exc:
        failure = str(exc)
    data = {
        "type": "vertical_transition",
        "algorithm": "graph_memory_2453f25",
        "execution_config": execution_config,
        **binding,
        "direction": direction,
        "route_completed": False,
        "objective_completed": False,
        "destination_floor_reached": False,
        "failure_reason": failure,
        "before_node_id": before_node,
        "before_floor_id": before_floor,
        "planning_node_id": planning_node,
        "attempts": attempts,
        "executed_move_history": histories,
        "trace_xyz": [] if policy is None else np.asarray(policy.trace).tolist(),
        "transition_state": {} if policy is None else policy.state(),
    }
    results.append(
        (
            ActionCall(action="vertical_transition", args=dict(action.args)),
            ActionResult(ok=False, data=data, message=failure),
        )
    )
    step.results, step.navigation_segment_timings = results, timings
    step.executed_action = data
    state.last_executed_action = data
    state.finalize_reason = f"vertical_transition_failed:{failure}"

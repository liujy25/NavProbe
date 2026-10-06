"""Text and visual evidence supplied to navigation decisions.

The decision loop owns reference changes, budgets, and state commits. These
helpers read that state to construct model inputs without making new decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING

import numpy as np

from navprobe.agent.ablation import task_state_enabled
from navprobe.agent.visual_action_context import direction_for_angle
from navprobe.perception.goal import NAVPROBE_LANGUAGE_GOAL_KIND
from navprobe.visualization.action_mode_overlays import (
    BevOverlayTransform, _node_labels_by_id, render_task_state_bev_overlay_result,
)

if TYPE_CHECKING:
    from navprobe.agent.state import NavProbeAgentState
    from navprobe.agent.decision_session import NavProbeDecisionSession
    from navprobe.agent.navigation_decisions import NavProbeSkillDecision


@dataclass(frozen=True)
class _NavProbeWaypointCandidateBevBase:
    image: np.ndarray
    transform: BevOverlayTransform
    landmark_markers: list[dict[str, object]]
    reference_node_marker: dict[str, object] | None
    context_text: str


def _initial_orientation_note(*, goal_kind: str, step_index: int) -> str:
    if str(goal_kind) != NAVPROBE_LANGUAGE_GOAL_KIND or int(step_index) != 0:
        return ""
    return "Step 0 note: no movement has been executed yet; do not assume any route progress has already happened."


def _initial_task_state_note(*, goal_kind: str, step_index: int) -> str:
    if str(goal_kind) != NAVPROBE_LANGUAGE_GOAL_KIND or int(step_index) != 0:
        return ""
    return (
        "Step 0 note: no movement has been executed yet. Use the current observation "
        "to establish the starting state, and do not mark instruction actions such as "
        "exit, turn, walk, pass, or stop as completed."
    )


def _backtrack_context_text(
    backtrack_contexts: list[dict[str, object]],
) -> str:
    if backtrack_contexts == []:
        return ""
    lines = ["Backtrack context:"]
    for index, item in enumerate(backtrack_contexts):
        lines.extend(
            [
                f"Backtrack {int(index) + 1}:",
                "- Trigger planning reference node: "
                + str(item.get("trigger_planning_node_id", "")),
                "- Planning reference node: " + str(item.get("anchor_node_id", "")),
                "- Physical robot node: "
                + str(item.get("physical_robot_node_id", "")),
                "- Objective: " + str(item.get("objective", "")),
                "- Reason: " + str(item.get("reason", "")),
            ]
        )
        result = item.get("result")
        if isinstance(result, dict):
            lines.extend(
                [
                    "- result: " + str(result.get("status", "")),
                    "- resulting navigation action: "
                    + str(result.get("navigation_action_mode", "")),
                    "- result detail: " + str(result.get("detail", "")),
                ]
            )
    return "\n".join(lines)


def _waypoint_inherited_agent_context(
    *, navigation_mode: NavProbeSkillDecision,
) -> list[dict[str, object]]:
    """Build grounder input from the selected skill's intent and reasoning."""
    navigation_action: dict[str, object] = {
        "mode": str(navigation_mode.action_mode),
    }
    if str(navigation_mode.action_mode) == "approach_to_stop":
        navigation_action["movement"] = str(
            navigation_mode.approach_movement or "move"
        )
    direction = str(navigation_mode.direction).strip()
    if direction != "":
        navigation_action["direction"] = direction
    objective = str(navigation_mode.action_objective).strip()
    if objective != "":
        navigation_action["objective"] = objective
    navigation_reason = str(
        navigation_mode.action_reason or navigation_mode.reasoning_action
    ).strip()
    if navigation_reason != "":
        navigation_action["reason"] = navigation_reason
    if str(navigation_mode.action_mode) == "vertical_transition":
        navigation_action["vertical_direction"] = str(
            navigation_mode.vertical_direction
        )
    if navigation_mode.subgoal_id is not None:
        navigation_action["subgoal_id"] = navigation_mode.subgoal_id
        navigation_action["subgoal_attempt"] = navigation_mode.subgoal_attempt
    if navigation_mode.waypoint_target:
        navigation_action["waypoint_target"] = navigation_mode.waypoint_target
    return [{
        "type": "text",
        "text": "Selected skill and local intent:\n"
        + json.dumps(navigation_action, ensure_ascii=False),
    }]


def _current_active_agenda_item(task_state_memory, subgoal_id: str | None = None) -> str:
    for item in list(task_state_memory.agenda):
        if subgoal_id is not None and item.subgoal_id != subgoal_id:
            continue
        if str(item.status) == "active" and str(item.kind) == "task":
            content = str(item.content).strip()
            unconfirmed = [
                str(predicate.content).strip()
                for predicate in item.predicates
                if str(predicate.status) == "unconfirmed"
            ]
            if unconfirmed:
                return content + "\nUnconfirmed predicates:\n- " + "\n- ".join(unconfirmed)
            return content
    return ""


def _waypoint_bev_landmark_markers(
    *,
    step_index: int,
    floor_id: str,
    context_loop: NavProbeDecisionSession | None,
    landmark_context,
) -> list[dict[str, object]]:
    if landmark_context is None:
        return []
    markers = [marker.to_dict() for marker in landmark_context.bev_markers]
    if int(step_index) == 0:
        return markers
    if context_loop is None:
        return []
    workspace = context_loop.workspace
    selected_ids = set(
        workspace.selected_landmark_ids_by_floor.get(str(floor_id), set())
    )
    labels_by_id = workspace.landmark_display_labels_by_floor.get(
        str(floor_id), {}
    )
    selected_labels = {
        int(labels_by_id[landmark_id])
        for landmark_id in selected_ids
        if landmark_id in labels_by_id
    }
    return [
        marker
        for marker in markers
        if int(marker.get("label", -1)) in selected_labels
    ]


def _waypoint_candidate_bev_base(
    *,
    state: "NavProbeAgentState",
    step_index: int,
    current_node_id: str,
    context_loop: NavProbeDecisionSession | None,
    landmark_context,
    include_graph_context: bool,
) -> _NavProbeWaypointCandidateBevBase:
    floor_id = str(state.system.current_floor_id)
    global_exploration = state.global_exploration_for_floor(floor_id)
    landmark_markers = _waypoint_bev_landmark_markers(
        step_index=int(step_index),
        floor_id=floor_id,
        context_loop=context_loop,
        landmark_context=landmark_context,
    )
    floor_nodes = list(state.graph.iter_nodes(floor_id=floor_id))
    node_labels_by_id = _node_labels_by_id(floor_nodes)
    current_id = str(current_node_id)
    selected_node_ids: set[str] = set()
    selected_edge_ids: set[str] = set()
    base_image: np.ndarray | None = None
    base_transform: BevOverlayTransform | None = None
    if include_graph_context and context_loop is not None:
        workspace = context_loop.workspace
        base_image = workspace.shared_bev_by_floor.get(floor_id)
        base_transform = workspace.shared_bev_transform_by_floor.get(floor_id)
        selected_node_ids = set(
            workspace.selected_node_ids_by_floor.get(floor_id, set())
        )
        selected_edge_ids = set(
            workspace.selected_edge_ids_by_floor.get(floor_id, set())
        )
    if base_image is None or base_transform is None:
        selected_node_ids = {current_id} if include_graph_context else set()
        selected_edge_ids = set()
        render_result = render_task_state_bev_overlay_result(
            graph=state.graph,
            global_exploration=global_exploration,
            floor_id=floor_id,
            landmark_markers=landmark_markers,
            show_graph=include_graph_context,
            node_ids=selected_node_ids,
            edge_ids=selected_edge_ids,
        )
        base_image = render_result.image
        base_transform = render_result.transform

    reference_node_marker = None
    if include_graph_context and current_id not in selected_node_ids:
        current_node = state.graph.get_node(current_id)
        reference_node_marker = {
            "label": int(node_labels_by_id[current_id]),
            "xy": [float(current_node.position[0]), float(current_node.position[1])],
        }
        selected_node_ids.add(current_id)

    context_lines: list[str] = []
    if include_graph_context:
        node_text = ", ".join(
            f"{int(node_labels_by_id[node_id])}={node_id}"
            for node_id in sorted(
                selected_node_ids,
                key=lambda node_id: int(node_labels_by_id[node_id]),
            )
            if node_id in node_labels_by_id
        )
        context_lines.append(
            f"Blue numbered circles are graph nodes: {node_text or 'none'}."
        )
        if current_id in node_labels_by_id:
            context_lines.append(
                "Current reference node: "
                f"{int(node_labels_by_id[current_id])}={current_id}."
            )
        edges_by_id = {
            str(edge.id): edge
            for edge in state.graph.iter_edges(
                floor_id=floor_id,
                include_vertical=False,
            )
        }
        edge_text = ", ".join(
            f"{edge_id} ({edges_by_id[edge_id].src_id} -> "
            f"{edges_by_id[edge_id].dst_id})"
            for edge_id in sorted(selected_edge_ids)
            if edge_id in edges_by_id
        )
        if edge_text != "":
            context_lines.append(
                "Highlighted retrieved edge trajectories: " + edge_text + "."
            )
    return _NavProbeWaypointCandidateBevBase(
        image=np.asarray(base_image, dtype=np.uint8),
        transform=base_transform,
        landmark_markers=landmark_markers,
        reference_node_marker=reference_node_marker,
        context_text="\n".join(context_lines),
    )


def _previous_move_failure_context(state) -> str:
    action = getattr(state, "last_executed_action", None)
    if not isinstance(action, dict) or action.get("type") != "move_to_visual_waypoint":
        return ""
    if action.get("route_completed") is not False:
        return ""
    result = action.get("move_result", {})
    data = result.get("data", {})
    decision = action.get("decision", {})
    navigation = decision.get("navigation_mode", {})
    nav_info = data.get("nav_info", {})
    summary = {
        "origin_place_node_id": action.get("origin_place_node_id"),
        "planning_current_node_id": decision.get("planning_current_node_id"),
        "reason": action.get("reason", ""),
        "reasoning_action": navigation.get("reasoning_action", ""),
        "requested_target": action.get("move_call", {}).get("args", {}),
        "start_pose": data.get("start_pose"),
        "actual_pose": data.get("pose"),
        "position_changed": data.get("position_changed"),
        "route_completed": False,
        "failure_reason": data.get("failure_reason", result.get("message", "")),
        "position_navigation_success": nav_info.get("position_nav_info", {}).get("success"),
        "orientation_alignment_success": nav_info.get("orientation_align_info", {}).get("success"),
        "final_distance": nav_info.get("final_distance"),
    }
    if task_state_enabled(state.ablation_name):
        summary.update(subgoal_id=action.get("subgoal_id"), subgoal_attempt=action.get("subgoal_attempt"))
    return (
        "Previous move failed; execution stopped at the measured actual pose. "
        "No automatic return or retry was performed. Reassess the next action from the current "
        "physical observation. The previous reason describes the old planning reference; "
        "do not interpret its relative directions as directions from the current pose. "
        "A controller failure does not by itself prove that the route is blocked or the task is complete.\n"
        + json.dumps(summary, ensure_ascii=False)
    )


def _waypoint_selection_context_for_node(
    state: "NavProbeAgentState",
    planner_node_id: str,
) -> dict[str, object] | None:
    return state.waypoint_selection_context_by_node_id.get(str(planner_node_id))


def _waypoint_selection_text_for_node(
    *,
    state: "NavProbeAgentState",
    planner_node_id: str,
    backtrack_reason: str,
) -> str:
    context = _waypoint_selection_context_for_node(state, planner_node_id)
    if context is None:
        return f"Previous selection at reference_node_id {planner_node_id}: none."
    direction_text = direction_for_angle(context["selected_angle_deg"])
    kind = context["selection_kind"]
    selected_label = context["selected_label"]
    label_text = ""
    if selected_label is not None:
        label_text = f" label {selected_label}"
    selection = f"{direction_text} {kind}{label_text}".strip()
    return (
        f"previous_selection: {selection}; "
        f"replan_reason: {str(backtrack_reason).strip() or 'none'}"
    )


def _waypoint_selection_images_for_node(
    state: "NavProbeAgentState",
    planner_node_id: str,
    *,
    include_node_identity: bool = True,
) -> list[tuple[str, np.ndarray]]:
    context = _waypoint_selection_context_for_node(state, planner_node_id)
    if context is None:
        return []
    image = np.asarray(state.cache.get_image(context["image_id"]).image, dtype=np.uint8)
    label = context["selected_label"]
    label_text = ""
    if label is not None:
        label_text = f" The previous selected label was {label}."
    text = (
        f"Backtrack feedback RGB for reference_node_id {planner_node_id}: "
        "sampled labels from the previous decision are shown."
        f"{label_text}"
        if include_node_identity
        else (
            "Backtrack feedback RGB from the previous decision at the selected prior node: "
            f"sampled labels are shown.{label_text}"
        )
    )
    return [(text, image)]


def _waypoint_planner_context_text(
    *,
    state: "NavProbeAgentState",
    navigation_mode: NavProbeSkillDecision,
    current_node_id: str,
    planner_node_id: str,
    initial_orientation_note: str = "",
) -> str:
    orientation_note = str(initial_orientation_note).strip()
    orientation_lines = []
    if orientation_note != "":
        orientation_lines = [orientation_note]
    if str(planner_node_id) == str(current_node_id):
        lines = [
            f"reference_node_id: {current_node_id}",
            "Candidate waypoints are generated from the current robot panorama.",
            *orientation_lines,
        ]
        return "\n".join(lines).strip()
    lines = [
        "Waypoint reference:",
        f"- current_robot_node_id: {current_node_id}",
        f"- reference_node_id: {planner_node_id}",
        "- Candidate overlays are generated from the reference node panorama.",
        "- Execution starts from the current robot node and routes to the selected candidate.",
        f"- {_waypoint_selection_text_for_node(state=state, planner_node_id=planner_node_id, backtrack_reason=navigation_mode.backtrack_reason)}",
        *[f"- {line}" for line in orientation_lines],
    ]
    return "\n".join(lines).strip()

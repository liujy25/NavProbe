"""Navigation skill selection from the committed Executive assessment."""
from __future__ import annotations

from copy import deepcopy
import json
from typing import TYPE_CHECKING

import numpy as np

from navprobe.agent.decision_inputs import request_task_response, task_panorama_content, task_state_content
from navprobe.agent.navigation_decisions import NavProbeSkillDecision, NavProbeTaskStateDecision
from navprobe.perception.goal import NAVPROBE_LANGUAGE_GOAL_KIND
from navprobe.agent.visual_action_context import VisualActionContext
from navprobe.memory.task_state import NavProbeTaskStateMemory

if TYPE_CHECKING:
    from navprobe.llm.client import LLMClient
    from navprobe.runtime.cache import RuntimeCache


NAVPROBE_HORIZONTAL_DIRECTIONS = {"front", "back", "left", "right"}


_NAVPROBE_SKILL_TOOLS = {
    "backtrack",
    "go_to_waypoint",
    "approach_to_stop",
    "vertical_transition",
}


def _navprobe_skill_tool_schemas(
    *, allowed_backtrack_node_ids: set[str] | None,
    planning_reference_panorama: bool = False,
    has_agenda: bool = True,
) -> list[str]:
    schemas = ["""Action objective reference:
Set `subgoal_id` to the selected active objective's stable ID. When the agenda is empty, use JSON null and pursue the original task and its constraints. In the examples, replace `<active ID>` with null in that case, and replace null with the selected ID when an objective remains.""" if has_agenda else """Action objective reference:
This ablation has no persistent agenda. Set `subgoal_id` to JSON null and pursue the original instruction using the current Executive assessment."""]
    schemas.append("""Grounding handoff:
The grounder receives your selected action and candidate RGB/BEV overlays. It needs the local intent to be self-contained: in `reason`, identify the visible route or target region, the evidence that identifies it, and the route-order, target-validity, or stopping constraints that apply. Include any correction from retrieval directly; an ID or a reference to an earlier assessment is not enough to convey that evidence.""")
    if allowed_backtrack_node_ids:
        schemas.append("""### `backtrack`
Select a listed visited node when its stored view provides useful evidence or grounding options for a missed route or search region. This switches the planning reference without moving the robot. Executive and skill selection then run on that view; any subsequent movement starts at the actual physical pose, without requiring a switch back to the physical view first.
Available anchors: """ + ", ".join(sorted(allowed_backtrack_node_ids)) + """
JSON:
{"backtrack":{"subgoal_id":"<active ID>","anchor_node_id":"<available anchor>","objective":"<relation or region to reassess>","reason":"<evidence supporting recovery>"}}""")
    stay_rule = (
        "Here `stay` requests physical navigation to the historical planning node before terminal assessment. Choose it only when that node is a supported final stopping position you intend to reach. It is not a request for ordinary recovery or an intermediate revisit; the stored image cannot establish arrival. The resulting terminal context records movement=move."
        if planning_reference_panorama else
        "Here `stay` keeps the robot at its current pose without movement. Choose it only when current-pose evidence supports already occupying the final stopping region."
    )
    schemas.extend([
        """### `go_to_waypoint`
Advance toward a visible route or region that supports the next objective. The grounder selects an FSS waypoint for the local intent expressed in `reason`. Interpret direction in the supplied panorama's reference frame.
JSON:
{"go_to_waypoint":{"subgoal_id":"<active ID>","direction":"front|back|left|right", "reason":"<visible route and why it advances this objective>"}}""",
        """### `approach_to_stop`
Prepare the robot for final stopping. Choose `move` for a reachable local approach to the stopping region. """ + stay_rule + """ The Executive checks the original task and stopping requirements at the next decision step; this skill does not declare completion. Seeing the destination at a distance is insufficient evidence of arrival.
JSON for stay:
{"approach_to_stop":{"subgoal_id":null,"movement":"stay","stop_objective":"<original task's stopping relation>","reason":"<endpoint evidence>"}}
JSON for move:
{"approach_to_stop":{"subgoal_id":null,"movement":"move","direction":"front|back|left|right","stop_objective":"<original task's stopping relation>","reason":"<reachable approach evidence>"}}""",
        """### `vertical_transition`
Move along a visible, locally accessible staircase in the required direction to the first destination floor. The stair controller continues through connecting landings and confirms the destination walking surface before returning.
`waypoint_target` states the selected objective (the original task if no agenda objective exists) and its route constraints. This skill executes one complete floor transition.
JSON:
{"vertical_transition":{"subgoal_id":"<active ID>","vertical_direction":"up|down","waypoint_target":"<full selected objective including its required endpoint>","reason":"<visible accessible staircase and why it advances this objective>"}}""",
    ])
    if not has_agenda:
        schemas = [schema.replace('"<active ID>"', 'null') for schema in schemas]
    return schemas


def select_navigation_skill(
    *,
    client: "LLMClient",
    cache: "RuntimeCache",
    goal_kind: str = NAVPROBE_LANGUAGE_GOAL_KIND,
    visual_context: VisualActionContext,
    task_state: NavProbeTaskStateMemory,
    latest_task_state: NavProbeTaskStateDecision,
    retrieval_workspace_content: list[dict[str, object]],
    task_state_context_text: str = "",
    backtrack_context_text: str = "",
    landmark_panorama_views: dict[int, np.ndarray] | None = None,
    detected_landmarks_text: str = "",
    allowed_backtrack_node_ids: set[str] | None = None,
    planning_reference_panorama: bool = False,
    original_instruction: str | None = None,
    task_constraints: str = "",
    no_progress: bool = False,
) -> NavProbeSkillDecision:
    """Select a navigation skill from the committed Executive assessment."""
    if str(goal_kind).strip() != NAVPROBE_LANGUAGE_GOAL_KIND:
        raise ValueError(f"unsupported goal_kind: {goal_kind!r}")
    dynamic_agenda = not no_progress
    if not no_progress and not task_state.agenda_initialized:
        raise ValueError("NavProbe requires an initialized task agenda")
    if latest_task_state is None:
        raise ValueError("navigation action tools require current task state")
    tool_schemas = _navprobe_skill_tool_schemas(
        allowed_backtrack_node_ids=allowed_backtrack_node_ids,
        planning_reference_panorama=planning_reference_panorama,
        has_agenda=not no_progress,
    )
    task_state_context = str(task_state_context_text).strip()
    task_state_context_section = (
        f"\n\nCurrent-step context:\n{task_state_context}"
        if task_state_context != ""
        else ""
    )
    backtrack_context = str(backtrack_context_text).strip()
    backtrack_context_section = (
        f"Backtrack context:\n{backtrack_context}"
        if backtrack_context != ""
        and not backtrack_context.startswith("Backtrack context:")
        else backtrack_context
    )
    latest_assessment_section = ""
    latest_assessment_lines: list[str] = []
    if str(latest_task_state.task_state_assessment).strip() != "":
        latest_assessment_lines.extend(
            [
                "Latest Task Executive assessment:",
                latest_task_state.task_state_assessment,
            ]
        )
    if str(latest_task_state.terminal_check_decision).strip() != "":
        latest_assessment_lines.extend(
            [
                f"terminal_check: {latest_task_state.terminal_check_decision}",
                "terminal_missing_constraints: "
                + json.dumps(
                    latest_task_state.terminal_check_missing_constraints,
                    ensure_ascii=False,
                ),
            ]
        )
    if latest_assessment_lines:
        latest_assessment_section = "\n\n" + "\n".join(latest_assessment_lines)
    landmark_text = str(detected_landmarks_text).strip()
    navigation_state_semantics = """
Input meanings:
- The original instruction defines the task, route order, and stopping relation. The agenda gives pursuit priorities; helper objectives and historical references cannot add requirements such as exact pose matching, centering, or facing.
- The latest Executive assessment identifies progress, the next objective, and unresolved questions. Retrieval conclusions provide historical evidence for interpreting the route or target.
- The labeled panorama provides the reference frame for the local action. Historical planning views are explicitly identified when used.

Decision boundary:
Choose an action using the committed task state; this module does not revise it. Carry material uncertainty or contradictions into the action's `reason`. A confirmed predicate can be corrected by later evidence; an unconfirmed predicate is unknown, not false. The Executive judges task completion, including when the agenda is empty or a search has found no target.
""".strip()
    navigation_action_rules = """
Choose the next skill:
Follow the latest Executive assessment to advance the task or inspect a supported hypothesis. An intermediate visible route can be useful even when its destination is unknown. Preserve instructed turns, passages, and stopping boundaries, and leave unseen contents uncertain. The action contract below defines the local intent to pass to the grounder. If grounding rejects that intent, control returns to the Executive for reassessment before another skill is selected.
""".strip()
    system_prompt = """You are the Skill Router in NavProbe.
Turn the latest Executive assessment into one local skill and a self-contained intent for grounding and execution."""
    task_state_section = task_state_content(
        task_state=task_state,
        visual_context=visual_context,
        no_progress=no_progress,
        original_instruction=original_instruction,
        task_constraints=task_constraints,
    )
    response_protocol = (
        'Output contract:\nReturn only JSON matching exactly one of the action contracts below.\n\n'
        + '\n\n'.join(tool_schemas)
    )
    decision_text = '\n\n'.join(
        section for section in (
            navigation_state_semantics, navigation_action_rules, response_protocol,
        ) if section != ''
    )
    if no_progress:
        from navprobe.agent.no_progress_policy import no_progress_prompt
        (system_prompt, decision_text) = no_progress_prompt(
            navigation=True,
            tool_schemas=tool_schemas,
            allow_retrieve=False,
            require_retrieval_conclusion=False,
            planning_reference_panorama=planning_reference_panorama,
        )
    content: list[dict[str, object]] = []
    content.append({"type": "text", "text": task_state_section})
    if task_state_context_section != "":
        content.append({"type": "text", "text": task_state_context_section.strip()})
    if latest_assessment_section != '':
        content.append({"type": "text", "text": latest_assessment_section.strip()})
    observation_reference_lines: list[str] = []
    if planning_reference_panorama:
        observation_reference_lines.append(
            "The attached panorama is stored planning-reference evidence, not a fresh observation from the robot's physical pose."
        )
        if dynamic_agenda:
            observation_reference_lines.append(
                "Express the next intent in this stored view's frame; execution starts from the physical robot pose in the Backtrack context. "
                "The reference change requires neither switching back to the physical view nor visiting the anchor. "
                "A physical revisit needs support from the original route, actual path constraints, or a needed fresh observation, not a helper objective alone. "
                "Selecting a waypoint from this view does not guarantee passage through the anchor or establish a completed route event. "
                "The BEV reference marker identifies the planning anchor."
            )
    if landmark_text != "":
        observation_reference_lines.append(landmark_text)
    if observation_reference_lines:
        content.append(
            {"type": "text", "text": "\n".join(observation_reference_lines)}
        )
    content.extend(task_panorama_content(
        cache=cache,
        visual_context=visual_context,
        landmark_panorama_views=landmark_panorama_views,
        landmark_text=landmark_text,
        planning_reference_panorama=planning_reference_panorama,
    ))
    content.extend(deepcopy(retrieval_workspace_content))
    if backtrack_context_section != '':
        content.append({"type": "text", "text": backtrack_context_section})
    content.append({"type": "text", "text": decision_text})

    def normalize_response(parsed):
        return normalize_skill_decision(
            parsed,
            latest_task_state=latest_task_state,
            allowed_backtrack_node_ids=allowed_backtrack_node_ids,
            active_subgoal_ids={item.subgoal_id for item in task_state.agenda} if dynamic_agenda else set(),
        )
    return request_task_response(
        request=client.select_skill,
        system_prompt=system_prompt,
        content=content,
        normalize_response=normalize_response,
        retrieval_round=None,
        completed_retrieval_rounds=0,
    )


def normalize_skill_decision(
    payload: dict[str, object],
    *,
    latest_task_state: NavProbeTaskStateDecision,
    allowed_backtrack_node_ids: set[str] | None = None,
    active_subgoal_ids: set[str] | None = None,
) -> NavProbeSkillDecision:
    if not isinstance(payload, dict):
        raise ValueError("NavProbe response must be an object")
    module_name = 'Semantic Skill Selector'
    module_tools = _NAVPROBE_SKILL_TOOLS
    selected_tools = [name for name in module_tools if name in payload]
    if len(selected_tools) != 1:
        raise ValueError(
            f"{module_name} must select exactly one available operation: "
            f"{payload!r}"
        )
    selected_tool = selected_tools[0]
    expected_top_fields = set(selected_tools)
    if set(payload) != expected_top_fields:
        raise ValueError(
            f"{module_name} returned unexpected top-level fields: "
            f"{payload!r}"
        )
    raw_tool = payload.get(selected_tool)
    if not isinstance(raw_tool, dict):
        raise ValueError(f"{selected_tool} tool payload must be an object: {payload!r}")
    if latest_task_state is None:
        raise ValueError(f"{selected_tool} is not available before update_task_state: {payload!r}")
    if "subgoal_id" not in raw_tool:
        raise ValueError("navigation action requires subgoal_id")
    subgoal_id = raw_tool["subgoal_id"]
    if subgoal_id is None:
        if active_subgoal_ids:
            raise ValueError("subgoal_id=null is allowed only with an empty agenda")
    elif (
        not isinstance(subgoal_id, str) or not subgoal_id.strip()
        or (active_subgoal_ids is not None and subgoal_id not in active_subgoal_ids)
    ):
        raise ValueError("navigation action must reference an active subgoal_id")
    raw_tool = {key: value for key, value in raw_tool.items() if key != "subgoal_id"}
    if selected_tool == "backtrack":
        expected_fields = {"anchor_node_id", "objective", "reason"}
        if set(raw_tool) != expected_fields:
            raise ValueError(f"backtrack fields must be exactly {sorted(expected_fields)!r}: {payload!r}")
        anchor_node_id = str(raw_tool.get("anchor_node_id", "")).strip()
        objective = raw_tool["objective"]
        reason = raw_tool["reason"]
        if allowed_backtrack_node_ids is not None and anchor_node_id not in allowed_backtrack_node_ids:
            raise ValueError(f"backtrack selected unavailable anchor node: {payload!r}")
        if anchor_node_id == "" or any(
            not isinstance(text, str) or not text.strip() for text in (objective, reason)
        ):
            raise ValueError(f"backtrack requires anchor, objective, and reason: {payload!r}")
        objective, reason = objective.strip(), reason.strip()
        return NavProbeSkillDecision(
            action_mode="backtrack",
            stop_objective="",
            reasoning_action=objective,
            action_objective=objective,
            action_reason=reason,
            backtrack_reason=reason,
            backtrack_anchor_node_id=anchor_node_id,
            backtrack_objective=objective,
            subgoal_id=subgoal_id,
        )
    if selected_tool == "go_to_waypoint":
        expected_fields = {"direction", "reason"}
        if set(raw_tool) != expected_fields:
            raise ValueError(f"go_to_waypoint fields must be exactly {sorted(expected_fields)!r}: {payload!r}")
        direction = str(raw_tool.get("direction", "")).strip()
        reason = raw_tool["reason"]
        if (
            not isinstance(reason, str) or not reason.strip()
            or direction not in NAVPROBE_HORIZONTAL_DIRECTIONS
        ):
            raise ValueError(f"go_to_waypoint requires its shown fields: {payload!r}")
        reason = reason.strip()
        return NavProbeSkillDecision(
            action_mode="go_to_waypoint",
            stop_objective="",
            reasoning_action=reason,
            direction=direction,
            action_reason=reason,
            subgoal_id=subgoal_id,
        )
    if selected_tool == "approach_to_stop":
        movement = str(raw_tool.get("movement", "")).strip()
        expected_fields = {"movement", "stop_objective", "reason"}
        if movement == "move":
            expected_fields.add("direction")
        if set(raw_tool) != expected_fields:
            raise ValueError(
                "approach_to_stop fields must be exactly "
                f"{sorted(expected_fields)!r}: {payload!r}"
            )
        direction = str(raw_tool.get("direction", "")).strip()
        stop_objective = raw_tool["stop_objective"]
        reason = raw_tool["reason"]
        if movement not in {"move", "stay"}:
            raise ValueError(
                f"approach_to_stop movement must be move or stay: {payload!r}"
            )
        if any(not isinstance(text, str) or not text.strip() for text in (stop_objective, reason)):
            raise ValueError(
                f"approach_to_stop requires stop_objective and reason: {payload!r}"
            )
        stop_objective, reason = stop_objective.strip(), reason.strip()
        if movement == "move" and direction not in NAVPROBE_HORIZONTAL_DIRECTIONS:
            raise ValueError(
                f"moving approach_to_stop requires a horizontal direction: {payload!r}"
            )
        return NavProbeSkillDecision(
            action_mode="approach_to_stop",
            approach_movement=movement,
            stop_objective=stop_objective,
            reasoning_action=reason,
            direction=direction,
            action_objective=stop_objective,
            action_reason=reason,
            subgoal_id=subgoal_id,
        )
    if selected_tool == "vertical_transition":
        expected_fields = {"vertical_direction", "reason", "waypoint_target"}
        if set(raw_tool) != expected_fields:
            raise ValueError(f"vertical_transition fields must be exactly {sorted(expected_fields)!r}: {payload!r}")
        vertical_direction = str(raw_tool.get("vertical_direction", "")).strip()
        reason = str(raw_tool.get("reason", "")).strip()
        waypoint_target = str(raw_tool.get("waypoint_target", "")).strip()
        if vertical_direction not in {"up", "down"} or reason == "":
            raise ValueError(f"vertical_transition requires direction and reason: {payload!r}")
        if not waypoint_target:
            raise ValueError("vertical_transition requires waypoint_target with the full selected objective and endpoint")
        return NavProbeSkillDecision(
            action_mode="vertical_transition",
            stop_objective="",
            reasoning_action=reason,
            vertical_direction=vertical_direction,
            waypoint_target=waypoint_target,
            action_objective=waypoint_target or f"Take the {vertical_direction} vertical transition.",
            action_reason=reason,
            subgoal_id=subgoal_id,
        )
    raise ValueError(f"unsupported task-state/navigation tool: {payload!r}")

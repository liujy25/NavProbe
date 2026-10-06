from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NavProbeTaskStateDecision:
    task_state_assessment: str
    retrieval_conclusion: str = ""
    agenda_updates: list[dict[str, object]] = field(default_factory=list)
    predicate_updates: list[dict[str, object]] = field(default_factory=list)
    terminal_check_decision: str = ""
    terminal_check_reasoning: str = ""
    terminal_check_missing_constraints: list[str] = field(default_factory=list)
    # Retain an explicit empty update call without fabricating one for a no-op response.
    update_task_state_called: bool = True
    standalone_terminal_check: bool = False

    def to_dict(self) -> dict[str, object]:
        tool_calls: list[dict[str, object]] = []
        if self.predicate_updates:
            tool_calls.append(
                {
                    "name": "update_predicates",
                    "arguments": {
                        "predicate_updates": [
                            dict(item) for item in self.predicate_updates
                        ]
                    },
                }
            )
        task_state_arguments: dict[str, object] = {
            "agenda_updates": [dict(item) for item in self.agenda_updates]
        }
        if str(self.terminal_check_decision).strip() != "":
            task_state_arguments["terminal_check"] = {
                "decision": str(self.terminal_check_decision),
                "missing_constraints": [
                    str(item) for item in self.terminal_check_missing_constraints
                ],
            }
        if self.update_task_state_called or self.agenda_updates or (self.terminal_check_decision and not self.standalone_terminal_check):
            tool_calls.append(
                {"name": "update_task_state", "arguments": task_state_arguments}
            )
        payload: dict[str, object] = {"tool_calls": tool_calls}
        if self.standalone_terminal_check and self.terminal_check_decision:
            payload["terminal_check"] = task_state_arguments["terminal_check"]
        if str(self.task_state_assessment).strip() != "":
            payload = {"task_state_assessment": str(self.task_state_assessment), **payload}
        if str(self.retrieval_conclusion).strip() != "":
            payload = {
                "retrieval_conclusion": str(self.retrieval_conclusion),
                **payload,
            }
        return payload


@dataclass(frozen=True)
class NavProbeSkillDecision:
    action_mode: str
    stop_objective: str
    reasoning_action: str
    approach_movement: str = ""
    direction: str = ""
    vertical_direction: str = ""
    task_item_index: int | None = None
    subgoal_id: str | None = None
    subgoal_attempt: int | None = None
    waypoint_target: str = ""
    action_objective: str = ""
    action_reason: str = ""
    backtrack_reason: str = ""
    backtrack_anchor_node_id: str = ""
    backtrack_objective: str = ""

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "action_mode": str(self.action_mode),
            "stop_objective": str(self.stop_objective),
            "reasoning_action": str(self.reasoning_action),
            "direction": str(self.direction),
            "vertical_direction": str(self.vertical_direction),
        }
        if self.action_mode == "vertical_transition" and self.task_item_index is not None:
            payload["task_item_index"] = self.task_item_index
        if self.subgoal_id is not None:
            payload["subgoal_id"] = self.subgoal_id
            if self.subgoal_attempt is not None:
                payload["subgoal_attempt"] = self.subgoal_attempt
        if self.waypoint_target:
            payload["waypoint_target"] = self.waypoint_target
        if (
            str(self.action_mode).strip() == "approach_to_stop"
            and str(self.approach_movement).strip() != ""
        ):
            payload["approach_movement"] = str(self.approach_movement)
        if str(self.action_objective).strip() != "":
            payload["action_objective"] = str(self.action_objective)
        if str(self.action_reason).strip() != "":
            payload["action_reason"] = str(self.action_reason)
        if str(self.action_mode).strip() == "backtrack":
            payload.update({
                "backtrack_reason": str(self.backtrack_reason),
                "backtrack_anchor_node_id": str(self.backtrack_anchor_node_id),
                "backtrack_objective": str(self.backtrack_objective),
            })
        return payload


@dataclass(frozen=True)
class NavProbeWaypointDecision:
    selected_angle_deg: int | None
    waypoint_target: str
    reasoning: str
    failure_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_angle_deg": None if self.selected_angle_deg is None else int(self.selected_angle_deg),
            "waypoint_target": str(self.waypoint_target),
            "reasoning": str(self.reasoning),
            "failure_reason": str(self.failure_reason),
        }

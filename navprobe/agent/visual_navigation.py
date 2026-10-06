from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
import math
from typing import TYPE_CHECKING

import numpy as np

from navprobe.agent.navigation_context import (
    _initial_orientation_note,
    _initial_task_state_note,
    _backtrack_context_text,
    _waypoint_inherited_agent_context,
    _current_active_agenda_item,
    _waypoint_candidate_bev_base,
    _previous_move_failure_context,
    _waypoint_selection_images_for_node,
    _waypoint_planner_context_text,
)
from navprobe.agent.decision_session import NavProbeDecisionSession
from navprobe.agent.knowledge_consolidation import manage_retrieved_knowledge
from navprobe.agent.episodic_retrieval import MaterializedMemoryContext
from navprobe.agent.episodic_retrieval import RetrievalWorkspace
from navprobe.agent.episodic_retrieval import RetrieveRequest
from navprobe.agent.episodic_retrieval import build_memory_index
from navprobe.agent.episodic_retrieval import execute_retrieve_request
from navprobe.agent.episodic_retrieval import materialize_memory_context
from navprobe.agent.episodic_retrieval import memory_index_text
from navprobe.agent.node_summary import summarize_current_node
from navprobe.agent.ablation import effective_max_retrieve_rounds
from navprobe.agent.ablation import compact_memory_only_enabled
from navprobe.agent.ablation import passive_full_history_enabled
from navprobe.agent.ablation import task_state_enabled
from navprobe.agent.visual_action_context import VisualActionContext, build_visual_action_context
from navprobe.agent.visual_action_context import build_visual_action_context_for_node
from navprobe.agent.visual_grounding import VisualWaypoint
from navprobe.agent.skill_policy import select_navigation_skill
from navprobe.agent.task_executive import assess_task_state
from navprobe.agent.task_executive import ensure_task_state_memory
from navprobe.agent.navigation_decisions import NavProbeWaypointDecision
from navprobe.agent.navigation_decisions import NavProbeSkillDecision
from navprobe.agent.navigation_decisions import NavProbeTaskStateDecision
from navprobe.agent.landmark_context import build_landmark_context
from navprobe.agent.landmark_context import NavProbeLandmarkContext
from navprobe.agent.waypoint_grounding import ground_waypoint
from navprobe.agent.waypoint import GroundedWaypointTarget
from navprobe.agent.waypoint import WaypointPlanningContext
from navprobe.agent.waypoint import WaypointPolicyResult
from navprobe.agent.waypoint.types import FRONTIER_SKELETON_SAMPLE_WAYPOINT_POLICY
from navprobe.agent.waypoint.types import validate_waypoint_policy
from navprobe.memory.graph.node import add_node_observation_knowledge
from navprobe.memory.graph.node import has_node_observation_knowledge
from navprobe.logging.agent_logging import build_node_summary_record
from navprobe.memory.task_state import NavProbeTaskStateUpdateResult
from navprobe.perception.goal import NAVPROBE_LANGUAGE_GOAL_KIND
from navprobe.schemas import ActionCall
from navprobe.visualization.waypoint_overlay import draw_waypoint_overlay_rgb

if TYPE_CHECKING:
    from navprobe.agent.state import NavProbeAgentState, NavProbeStepState
    from navprobe.mapping.exploration.manager import ExplorationManager
    from navprobe.perception.goal import NavProbeGoalSpec


@dataclass(frozen=True)
class VisualWaypointAttempt:
    attempt_index: int
    selected_angle_deg: int
    selected_obs_id: str
    waypoint_target: str
    failure_reason: str = ""
    sampled_candidate: dict[str, object] = field(default_factory=dict)
    candidate_selection: dict[str, object] = field(default_factory=dict)
    candidate_count: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_index": int(self.attempt_index),
            "selected_angle_deg": int(self.selected_angle_deg),
            "selected_obs_id": str(self.selected_obs_id),
            "waypoint_target": str(self.waypoint_target),
            "failure_reason": str(self.failure_reason),
            "sampled_candidate": dict(self.sampled_candidate),
            "candidate_selection": dict(self.candidate_selection),
            "candidate_count": None if self.candidate_count is None else int(self.candidate_count),
        }


@dataclass(frozen=True)
class VisualNavigationDecision:
    visual_context: VisualActionContext
    task_state_initialization: dict[str, object]
    navigation_mode: NavProbeSkillDecision | None
    task_state_update_result: NavProbeTaskStateUpdateResult
    waypoint: VisualWaypoint | None
    action_call: ActionCall | None
    local_move_plan: NavProbeWaypointDecision | None = None
    waypoint_attempts: list[VisualWaypointAttempt] = field(default_factory=list)
    navigation_replan_feedback: list[dict[str, object]] = field(default_factory=list)
    failure_reason: str = ""
    waypoint_policy_name: str = ""
    grounded_waypoint_target: GroundedWaypointTarget | None = None
    terminal_check: dict[str, object] = field(default_factory=dict)
    physical_current_node_id: str = ""
    planning_current_node_id: str = ""
    backtrack_contexts: list[dict[str, object]] = field(default_factory=list)

    @property
    def waypoint_policy_result(self) -> WaypointPolicyResult | None:
        """Log projection; geometry and failures have one runtime owner."""
        if not self.waypoint_policy_name:
            return None
        return WaypointPolicyResult(
            policy_name=self.waypoint_policy_name, target=self.grounded_waypoint_target,
            reasoning="" if self.local_move_plan is None else str(self.local_move_plan.reasoning),
            failure_reason=self.failure_reason,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "visual_context": self.visual_context.to_dict(),
            "task_state_initialization": deepcopy(self.task_state_initialization),
            "navigation_mode": (
                None if self.navigation_mode is None else self.navigation_mode.to_dict()
            ),
            "task_state_update_result": self.task_state_update_result.to_dict(),
            "waypoint": None if self.waypoint is None else self.waypoint.to_dict(),
            "action_call": None if self.action_call is None else self.action_call.to_dict(),
            "local_move_plan": None if self.local_move_plan is None else self.local_move_plan.to_dict(),
            "waypoint_attempts": [attempt.to_dict() for attempt in self.waypoint_attempts],
            "navigation_replan_feedback": [dict(item) for item in self.navigation_replan_feedback],
            "failure_reason": str(self.failure_reason),
            "waypoint_policy_result": (
                None if self.waypoint_policy_result is None else self.waypoint_policy_result.to_dict()
            ),
            "grounded_waypoint_target": (
                None if self.grounded_waypoint_target is None else self.grounded_waypoint_target.to_dict()
            ),
            "terminal_check": deepcopy(self.terminal_check),
            "physical_current_node_id": str(self.physical_current_node_id),
            "planning_current_node_id": str(self.planning_current_node_id),
            "backtrack_contexts": [
                deepcopy(item) for item in self.backtrack_contexts
            ],
        }


def summarize_current_node_for_visual_policy(
    *,
    state: "NavProbeAgentState",
    step: "NavProbeStepState",
) -> dict[str, object]:
    visual_context = build_visual_action_context(
        state=state,
        step=step,
        include_graph_overlays=False,
    )
    decision = summarize_current_node(
        client=state.llm_client,
        cache=state.cache,
        visual_context=visual_context,
    )
    node_id = str(visual_context.current_node_id)
    node = state.graph.get_node(node_id)
    added_knowledge = add_node_observation_knowledge(
        node,
        node_summary=decision.node_summary,
        direction_summaries=decision.direction_summaries,
    )
    payload = build_node_summary_record(
        state=state,
        visual_context=visual_context,
        decision=decision,
    )
    payload["knowledge"] = [item.to_dict() for item in added_knowledge]
    step.node_summary = payload
    return payload


def ensure_current_node_summary_for_visual_policy(
    *,
    state: "NavProbeAgentState",
    step: "NavProbeStepState",
) -> dict[str, object]:
    node_id = str(step.current_place_node_id or state.current_place_node_id or "").strip()
    if node_id == "":
        raise ValueError("node summary requires a current place node")
    node = state.graph.get_node(node_id)
    if has_node_observation_knowledge(node):
        payload = {
            "node_id": node_id,
            "source": "existing",
            "knowledge": [item.to_dict() for item in node.knowledge],
        }
        step.node_summary = payload
        return payload
    return summarize_current_node_for_visual_policy(
        state=state,
        step=step,
    )


def _goal_kind(goal: "NavProbeGoalSpec | None") -> str:
    if goal is None:
        return ""
    return str(getattr(goal, "goal_kind", "") or "").strip()


def _with_replan_log(
    decision: VisualNavigationDecision,
    *,
    waypoint_attempts: list[VisualWaypointAttempt],
    navigation_replan_feedback: list[dict[str, object]],
) -> VisualNavigationDecision:
    if waypoint_attempts == [] and navigation_replan_feedback == []:
        return decision
    return replace(
        decision,
        waypoint_attempts=[*waypoint_attempts, *decision.waypoint_attempts],
        navigation_replan_feedback=[
            *[dict(item) for item in navigation_replan_feedback],
            *[dict(item) for item in decision.navigation_replan_feedback],
        ],
    )


def episodic_retrieval_enabled_for_goal(
    *, goal: "NavProbeGoalSpec | None",
) -> bool:
    return _goal_kind(goal) == NAVPROBE_LANGUAGE_GOAL_KIND


def _backtrack_node_ids(
    *,
    state: "NavProbeAgentState",
    current_node_id: str,
) -> set[str]:
    return {
        str(node.id)
        for node in state.graph.iter_nodes()
        if str(node.id) != str(current_node_id)
        and str(node.floor_id) == str(state.system.current_floor_id)
        and str(node.node_kind) == "place"
        and list(node.obs_ids) != []
    }


def _latest_incoming_edge_ref(
    *,
    state: "NavProbeAgentState",
    current_node_id: str,
) -> str:
    for edge in reversed(state.graph.iter_edges(include_vertical=True)):
        if str(edge.relation) not in {"move", "stairs_up", "stairs_down"}:
            continue
        if str(edge.dst_id) == str(current_node_id):
            return str(edge.id)
    return ""


def _backtrack_context(
    *,
    decision: NavProbeSkillDecision,
    planning_node_id: str,
    physical_node_id: str,
) -> dict[str, object]:
    return {
        "trigger_planning_node_id": str(planning_node_id),
        "anchor_node_id": str(decision.backtrack_anchor_node_id),
        "physical_robot_node_id": str(physical_node_id),
        "objective": str(decision.backtrack_objective or decision.action_objective),
        "reason": str(decision.backtrack_reason or decision.action_reason),
    }


def _bind_navigation_subgoal(decision: NavProbeSkillDecision, task_state_memory) -> NavProbeSkillDecision:
    if not task_state_memory.agenda_initialized or decision.subgoal_id is None:
        return decision
    item = next((item for item in task_state_memory.agenda if item.subgoal_id == decision.subgoal_id), None)
    if item is None:
        raise ValueError(f"navigation selected inactive subgoal: {decision.subgoal_id}")
    return replace(decision, subgoal_attempt=item.attempt)


def _previous_vertical_execution_context(state) -> str:
    action = getattr(state, "last_executed_action", None)
    if not isinstance(action, dict) or action.get("type") != "vertical_transition":
        return ""
    summary = {
        key: action.get(key)
        for key in ("subgoal_id", "subgoal_attempt", "waypoint_target", "direction",
                    "route_completed", "objective_completed", "failure_reason",
                    "before_node_id", "after_node_id", "before_floor_id", "after_floor_id")
    }
    summary["executed_local_moves"] = len(action.get("executed_move_history", []))
    return "Previous VerticalMove execution feedback (actual execution outcome):\n" + json.dumps(summary, ensure_ascii=False)


def run_episodic_retrieval_loop(
    *,
    state: "NavProbeAgentState",
    step: "NavProbeStepState",
    goal: "NavProbeGoalSpec | None",
    goal_text: str,
    visual_context: VisualActionContext,
    landmark_context,
    session: NavProbeDecisionSession,
) -> None:
    goal_kind = _goal_kind(goal)
    task_state_memory = state.system.task_state
    include_task_state = task_state_enabled(state.ablation_name)
    instruction_context = {
        "original_instruction": goal_text,
        "task_constraints": str(getattr(goal, "task_constraints", "") or ""),
        "no_progress": not include_task_state,
    }
    passive_full_history = passive_full_history_enabled(state.ablation_name)
    compact_memory_only = compact_memory_only_enabled(state.ablation_name)
    if not session.initialized:
        session.initialize_context(visual_context, landmark_context)
        session.pending_terminal_check = getattr(state, "pending_terminal_check", None)
        session.terminal_check_required = isinstance(session.pending_terminal_check, dict)
        pending_initialization = getattr(
            state,
            "pending_task_state_initialization",
            None,
        )
        if not include_task_state:
            session.task_state_initialization = {"initialized": False, "source": "ablation:no_progress"}
        elif pending_initialization is not None:
            session.task_state_initialization = dict(pending_initialization)
            setattr(state, "pending_task_state_initialization", None)
        else:
            session.task_state_initialization = ensure_task_state_memory(
                client=state.llm_client,
                task_state=task_state_memory,
                goal_text=goal_text,
                goal_kind=goal_kind,
            )
        session.workspace = RetrievalWorkspace(
            entries=build_memory_index(state),
            max_retrieve_rounds=effective_max_retrieve_rounds(
                state.ablation_name,
                int(state.max_retrieve_rounds),
            ),
        )
        session.initialized = True

    session.update_trace(state, step)
    while session.decision_count < session.max_decisions:
        session.decision_count += 1
        has_landmark_evidence = (
            session.planning_landmark_context is not None
            and bool(session.planning_landmark_context.evidences)
        )
        landmark_panorama_views = session.get_landmark_panorama_views(cache=state.cache)
        detected_landmarks_text = (
            str(session.planning_landmark_context.text)
            if has_landmark_evidence
            else ""
        )
        planning_node_ref = str(session.planning_visual_context.current_node_id)
        provided_fields_by_ref = {
            planning_node_ref: [
                "rgb",
                *(["landmarks"] if has_landmark_evidence else []),
            ]
        }
        if passive_full_history:
            if session.passive_preloaded_planning_node_id != planning_node_ref:
                session.passive_preloaded_context = materialize_memory_context(
                    state=state,
                    workspace=session.workspace,
                    entries=session.workspace.entries,
                    provided_fields_by_ref=provided_fields_by_ref,
                )
                session.passive_preloaded_planning_node_id = planning_node_ref
                session.passive_latest_incoming_edge_ref = _latest_incoming_edge_ref(
                    state=state,
                    current_node_id=planning_node_ref,
                )
                session.update_trace(state, step)
            retrieval_context_content = session.workspace.materialized_context_content()
        else:
            retrieval_context_content = (
                session.workspace.materialized_context_content()
                if not session.task_state_is_current
                else session.workspace.conclusion_context_content()
            ) if session.workspace.rounds != [] else []
        backtrack_context_text = _backtrack_context_text(
            session.backtrack_contexts
        )
        task_state_context_text = "\n\n".join(
            item
            for item in (
                _initial_task_state_note(
                    goal_kind=goal_kind,
                    step_index=int(step.place_step_index),
                ) if include_task_state else (
                    "Episode start: no instructed movement has been executed yet."
                    if int(step.place_step_index) == 0 else ""
                ),
                _previous_vertical_execution_context(state),
                _previous_move_failure_context(state),
            )
            if item != ""
        )
        planning_reference_panorama = (
            str(session.planning_visual_context.current_node_id)
            != str(session.physical_visual_context.current_node_id)
        )
        if session.task_state_is_current:
            if session.task_state_decision is None:
                raise ValueError(
                    "Skill selection requires the latest task-state assessment"
                )
            allowed_backtrack_node_ids = (
                _backtrack_node_ids(state=state, current_node_id=planning_node_ref)
                - session.rejected_backtrack_anchor_node_ids
                if int(step.place_step_index) > 0
                and len(session.backtrack_contexts) < int(session.max_execution_preparations)
                else set()
            )
            session.skill_count += 1
            decision = select_navigation_skill(
                client=state.llm_client,
                cache=state.cache,
                goal_kind=goal_kind,
                visual_context=session.planning_visual_context,
                task_state=task_state_memory,
                latest_task_state=session.task_state_decision,
                **instruction_context,
                retrieval_workspace_content=retrieval_context_content,
                task_state_context_text=task_state_context_text,
                backtrack_context_text=backtrack_context_text,
                landmark_panorama_views=landmark_panorama_views,
                detected_landmarks_text=detected_landmarks_text,
                allowed_backtrack_node_ids=allowed_backtrack_node_ids,
                planning_reference_panorama=planning_reference_panorama,
            )
        else:
            loop_fields_by_ref = session.workspace.fields_by_ref(
                provided_fields_by_ref=provided_fields_by_ref,
            )
            loop_index_text = memory_index_text(
                session.workspace.entries,
                provided_fields_by_ref=provided_fields_by_ref,
            )
            allow_retrieve = (
                not passive_full_history
                and not compact_memory_only
                and int(step.place_step_index) > 0
                and session.workspace.can_retrieve
                and any(loop_fields_by_ref.values())
                and session.decision_count < session.max_decisions
            )
            if passive_full_history:
                preloaded_context_text = (
                    "Complete episode graph memory (all text entries and all "
                    "historical multimodal fields are provided below; current "
                    "planning-node RGB and landmarks remain in the base observation; "
                    "the complete graph BEV is attached):\n"
                    + loop_index_text
                )
                task_state_context_text = "\n\n".join(
                    item for item in (task_state_context_text, preloaded_context_text) if item != ""
                )
            session.executive_inputs.append({
                "decision_index": session.decision_count,
                "event_count": len(session.working_memory),
                "planning_node_id": str(session.planning_visual_context.current_node_id),
            })
            session.executive_count += 1
            session.update_trace(state, step)
            decision = assess_task_state(
                client=state.llm_client,
                cache=state.cache,
                visual_context=session.planning_visual_context,
                task_state=task_state_memory,
                memory_index_text=loop_index_text,
                working_memory_text=session.working_memory_text(),
                retrieval_workspace_content=retrieval_context_content,
                retrieve_max_rounds=int(session.workspace.max_retrieve_rounds),
                retrieve_completed_rounds=int(session.workspace.retrieve_count),
                retrieve_fields_by_ref=(
                    loop_fields_by_ref if allow_retrieve else {}
                ),
                retrieve_provided_fields_by_ref=(
                    provided_fields_by_ref if allow_retrieve else {}
                ),
                allow_retrieve=allow_retrieve,
                allow_update_task_state=include_task_state,
                **instruction_context,
                require_retrieval_conclusion=session.workspace.has_pending_evidence,
                memory_index_context_only=compact_memory_only,
                task_state_context_text=task_state_context_text,
                backtrack_context_text=backtrack_context_text,
                landmark_panorama_views=landmark_panorama_views,
                detected_landmarks_text=detected_landmarks_text,
                terminal_check_context=(
                    session.pending_terminal_check if session.terminal_check_required else None
                ),
                planning_reference_panorama=planning_reference_panorama,
            )
        if isinstance(decision, (RetrieveRequest, NavProbeTaskStateDecision)):
            if not include_task_state and (decision.agenda_updates or decision.predicate_updates):
                raise ValueError("no_progress cannot write task-state memory")
            transaction = {
                "step_index": int(step.place_step_index),
                "decision_index": int(session.decision_count),
                "retrieval_round_index": int(session.workspace.retrieve_count),
                "request": decision.to_dict(),
                "status": "pending",
                "applied_updates": [],
                "skipped_updates": [],
            }
            request_index = getattr(state.llm_client, "_request_index", None)
            if type(request_index) is int:
                transaction["llm_request_index"] = request_index
            session.task_state_update_transactions.append(transaction)
            if session.workspace.has_pending_evidence:
                session.workspace.conclude_latest_retrieval(decision.retrieval_conclusion)
                session.record("retrieval_conclusion", state=state,
                               round_index=session.workspace.rounds[-1].round_index)
            session.record("task_state_assessment", state=state,
                           task_state_assessment=decision.task_state_assessment,
                           transaction_index=len(session.task_state_update_transactions)-1)
            if include_task_state:
                # apply_task_state_updates() already stages the whole response
                # internally and writes it atomically.  Commit this update
                # before retrieval: retrieve only supplies the next
                # Executive input and is not evidence for this response.
                try:
                    committed_updates = task_state_memory.apply_task_state_updates(
                        list(decision.agenda_updates),
                        list(decision.predicate_updates),
                        current_node_id=str(session.physical_visual_context.current_node_id),
                    )
                except Exception as exc:
                    transaction.update(
                        status="not_committed",
                        stage="agenda_update",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    session.record("state_update", state=state, transaction_index=len(session.task_state_update_transactions)-1)
                    session.failure_reason = "agenda_update_failed"
                    session.update_trace(state, step)
                    raise
                session.task_state_update_result.applied_updates.extend(committed_updates.applied_updates)
                session.task_state_update_result.skipped_updates.extend(committed_updates.skipped_updates)
                transaction.update(
                    status="agenda_committed",
                    **committed_updates.to_dict(),
                )
            session.task_state_update_count += 1
            if not include_task_state:
                transaction.update(status="agenda_committed", applied_updates=[], skipped_updates=[])
            session.record("state_update", state=state, transaction_index=len(session.task_state_update_transactions)-1)
            # Publish the memory state before any retrieval I/O.  If retrieval
            # fails, the trace and the in-memory agenda agree about what was
            # already applied.
            session.update_trace(state, step)
            if session.workspace.rounds != []:
                session.workspace.discard_concluded_raw_evidence()
            if isinstance(decision, RetrieveRequest):
                session.record("retrieve_action", state=state,
                               transaction_index=len(session.task_state_update_transactions)-1)
                session.update_trace(state, step)
                try:
                    execute_retrieve_request(
                        state=state,
                        workspace=session.workspace,
                        request=decision,
                    )
                except Exception as exc:
                    transaction.update(
                        status="retrieval_failed",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    session.failure_reason = "retrieval_failed"
                    session.update_trace(state, step)
                    raise
                transaction.update(status="committed", retrieval_round_committed=True,
                                   committed_round_index=session.workspace.rounds[-1].round_index)
                session.task_state_is_current = False
                session.update_trace(state, step)
                continue
            transaction.update(status="committed", retrieval_round_committed=False)
            session.task_state_decision = decision
            session.task_state_is_current = True
            if session.terminal_check_required or not include_task_state:
                session.terminal_check_payload = {
                    "decision": str(decision.terminal_check_decision),
                    "reason": str(decision.terminal_check_reasoning),
                    "missing_constraints": [
                        str(item)
                        for item in decision.terminal_check_missing_constraints
                    ],
                }
                has_active_task = any(
                    str(item.status) == "active"
                    for item in task_state_memory.agenda
                    if str(item.kind) == "task"
                )
                if include_task_state and decision.terminal_check_decision == "done" and has_active_task:
                    raise ValueError(
                        "terminal_check done requires all agenda items to be done"
                    )
                if (include_task_state and not task_state_memory.agenda_initialized
                        and decision.terminal_check_decision == "continue" and not has_active_task):
                    raise ValueError(
                        "terminal_check continue requires an active agenda item"
                    )
                state.pending_terminal_check = None
                if decision.terminal_check_decision == "done":
                    session.navigation_mode = None
                    session.update_trace(state, step, final_navigation_mode=None)
                    return
                session.terminal_check_required = False
                session.pending_terminal_check = None
                session.update_trace(state, step, terminal_check=session.terminal_check_payload)
            else:
                session.update_trace(state, step)
            continue
        if session.task_state_decision is None or not session.task_state_is_current:
            raise ValueError("navigation action preceded the required task-state update")
        decision = _bind_navigation_subgoal(decision, task_state_memory)
        session.navigation_mode = decision
        session.record("skill_selection", state=state, **session.skill_payload(decision))
        if str(decision.action_mode) == "backtrack":
            session.backtrack_contexts.append(
                _backtrack_context(
                    decision=decision,
                    planning_node_id=str(session.planning_visual_context.current_node_id),
                    physical_node_id=str(session.physical_visual_context.current_node_id),
                )
            )
            session.record("planning_reference_change", state=state, backtrack_index=len(session.backtrack_contexts)-1, execution_started=False,
                           **session.backtrack_contexts[-1])
            session.planning_visual_context = build_visual_action_context_for_node(
                state=state,
                node_id=str(decision.backtrack_anchor_node_id),
            )
            session.planning_landmark_context = build_landmark_context(
                landmark_controller=state.landmark_controller,
                visual_context=session.planning_visual_context,
            )
            session.task_state_is_current = False
            session.update_trace(state, step)
            continue
        session.update_trace(state, step, final_navigation_mode=session.navigation_mode)
        return
    session.failure_reason = "navigation_decision_budget_exhausted"
    session.navigation_mode = None
    session.update_trace(state, step, final_navigation_mode=None)


def plan_visual_navigation_action(
    *, state: "NavProbeAgentState", step: "NavProbeStepState", goal: "NavProbeGoalSpec | None",
) -> VisualNavigationDecision:
    session = NavProbeDecisionSession(max_execution_preparations=state.max_execution_preparations)
    try:
        decision = _plan_visual_navigation_session(state=state, step=step, goal=goal, session=session)
        return session.finish(state=state, step=step, decision=decision,
                              knowledge_manager=manage_retrieved_knowledge)
    except Exception as exc:
        session.abort(state=state, step=step, error=exc)
        raise


def _plan_visual_navigation_session(
    *,
    state: "NavProbeAgentState",
    step: "NavProbeStepState",
    goal: "NavProbeGoalSpec | None",
    session: NavProbeDecisionSession,
) -> VisualNavigationDecision:
    validate_waypoint_policy(
        waypoint_policy_name=state.waypoint_policy_name,
    )
    goal_kind = _goal_kind(goal)
    if goal_kind != NAVPROBE_LANGUAGE_GOAL_KIND:
        raise ValueError(f"unsupported NavProbe goal kind: {goal_kind!r}")
    goal_text = _goal_text(state=state, goal=goal)
    task_state_memory = state.system.task_state
    visual_context = build_visual_action_context(
        state=state,
        step=step,
        include_arrival_edge=True,
    )
    physical_visual_context = visual_context
    planning_visual_context = visual_context
    backtrack_contexts: list[dict[str, object]] = []
    landmark_context = build_landmark_context(
        landmark_controller=state.landmark_controller,
        visual_context=visual_context,
    )
    include_task_state = task_state_enabled(state.ablation_name)
    navigation_replan_feedback = session.navigation_replan_feedback
    waypoint_attempts = session.waypoint_attempts
    for navigation_attempt_index in range(session.max_execution_preparations):
        run_episodic_retrieval_loop(
            state=state,
            step=step,
            goal=goal,
            goal_text=goal_text,
            visual_context=visual_context,
            landmark_context=landmark_context,
            session=session,
        )
        episodic_context_loop = session
        task_state_initialization = session.task_state_initialization
        task_state_decision = session.task_state_decision
        task_state_update_result = session.task_state_update_result
        initial_navigation_mode = session.navigation_mode
        planning_visual_context = (
            session.planning_visual_context
            if session.planning_visual_context is not None
            else physical_visual_context
        )
        backtrack_contexts = session.backtrack_contexts
        landmark_context = session.planning_landmark_context
        if str(session.failure_reason).strip() != "":
            return VisualNavigationDecision(
                visual_context=planning_visual_context,
                task_state_initialization=task_state_initialization,
                navigation_mode=None,
                task_state_update_result=task_state_update_result,
                waypoint=None,
                action_call=None,
                failure_reason=str(session.failure_reason),
                physical_current_node_id=str(
                    physical_visual_context.current_node_id
                ),
                planning_current_node_id=str(
                    planning_visual_context.current_node_id
                ),
                backtrack_contexts=backtrack_contexts,
            )
        if initial_navigation_mode is None:
            terminal_check = {
                "decision": str(task_state_decision.terminal_check_decision),
                "reason": str(task_state_decision.terminal_check_reasoning),
                "missing_constraints": [
                    str(item)
                    for item in task_state_decision.terminal_check_missing_constraints
                ],
            }
            if terminal_check["decision"] != "done":
                raise ValueError(
                    "missing navigation mode requires a completed terminal check"
                )
            return VisualNavigationDecision(
                visual_context=visual_context,
                task_state_initialization=task_state_initialization,
                navigation_mode=None,
                task_state_update_result=task_state_update_result,
                waypoint=None,
                action_call=ActionCall(action="done", args={}),
                terminal_check=terminal_check,
                physical_current_node_id=str(
                    physical_visual_context.current_node_id
                ),
                planning_current_node_id=str(
                    planning_visual_context.current_node_id
                ),
                backtrack_contexts=backtrack_contexts,
            )
        navigation_mode = initial_navigation_mode
        session.preparation_count += 1
        navigation_mode = _bind_navigation_subgoal(navigation_mode, task_state_memory)
        if (
            str(navigation_mode.action_mode) == "approach_to_stop"
            and str(navigation_mode.approach_movement) == "stay"
        ):
            planning_node_id = str(planning_visual_context.current_node_id)
            physical_node_id = str(physical_visual_context.current_node_id)
            if planning_node_id != physical_node_id:
                planning_node = state.graph.get_node(planning_node_id)
                goal_xy = (
                    planning_node.nav_goal_xy
                    if planning_node.nav_goal_xy is not None
                    else (
                        float(planning_node.position[0]),
                        float(planning_node.position[1]),
                    )
                )
                target = GroundedWaypointTarget(
                    goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
                    goal_yaw=float(planning_node.yaw),
                    world_z=float(planning_node.position[2]),
                    policy_name=state.waypoint_policy_name,
                    source_type="planning_node",
                    source_id=planning_node_id,
                    obs_id=str(planning_visual_context.view_for_angle(0).obs_id),
                    angle_deg=0,
                )
                return VisualNavigationDecision(
                    visual_context=planning_visual_context,
                    task_state_initialization=task_state_initialization,
                    navigation_mode=navigation_mode,
                    task_state_update_result=task_state_update_result,
                    waypoint=None,
                    action_call=target.to_action_call(),
                    local_move_plan=NavProbeWaypointDecision(
                        selected_angle_deg=0,
                        waypoint_target=(
                            f"return to planning node {planning_node_id}"
                        ),
                        reasoning=str(navigation_mode.action_reason),
                    ),
                    grounded_waypoint_target=target,
                    waypoint_policy_name=state.waypoint_policy_name,
                    waypoint_attempts=waypoint_attempts,
                    navigation_replan_feedback=navigation_replan_feedback,
                    physical_current_node_id=physical_node_id,
                    planning_current_node_id=planning_node_id,
                    backtrack_contexts=backtrack_contexts,
                )
            return VisualNavigationDecision(
                visual_context=planning_visual_context,
                task_state_initialization=task_state_initialization,
                navigation_mode=navigation_mode,
                task_state_update_result=task_state_update_result,
                waypoint=None,
                action_call=None,
                waypoint_attempts=waypoint_attempts,
                navigation_replan_feedback=navigation_replan_feedback,
                physical_current_node_id=physical_node_id,
                planning_current_node_id=planning_node_id,
                backtrack_contexts=backtrack_contexts,
            )
        use_waypoint_stop = str(navigation_mode.action_mode) == "approach_to_stop"
        if (
            navigation_mode.action_mode == "go_to_waypoint"
            or use_waypoint_stop
        ):
            active_agenda_item = (
                _current_active_agenda_item(task_state_memory, navigation_mode.subgoal_id)
                if include_task_state and not use_waypoint_stop else ""
            )
            inherited_agent_context_content = _waypoint_inherited_agent_context(navigation_mode=navigation_mode)
            candidate_bev_base = _waypoint_candidate_bev_base(
                state=state,
                step_index=int(step.place_step_index),
                current_node_id=str(planning_visual_context.current_node_id),
                context_loop=episodic_context_loop,
                landmark_context=landmark_context,
                include_graph_context=visual_context.graph_context_visible,
            )
            policy_context = WaypointPlanningContext(
                state=state,
                step=step,
                candidate_cache=session.candidate_cache,
                goal_kind=goal_kind,
                visual_context=planning_visual_context,
                execution_visual_context=physical_visual_context,
                landmark_context=landmark_context,
                task_state_initialization=task_state_initialization,
                navigation_mode=navigation_mode,
                task_state_update_result=task_state_update_result,
                inherited_agent_context_content=inherited_agent_context_content,
                active_agenda_item=active_agenda_item,
                candidate_bev_landmark_markers=candidate_bev_base.landmark_markers,
                candidate_bev_base_image=candidate_bev_base.image,
                candidate_bev_transform=candidate_bev_base.transform,
                candidate_bev_reference_node_marker=candidate_bev_base.reference_node_marker,
                candidate_bev_context_text=candidate_bev_base.context_text,
            )
            if use_waypoint_stop:
                waypoint_decision = _plan_frontier_skeleton_sample_stop_policy(
                    policy_context
                )
            else:
                waypoint_decision = _plan_frontier_skeleton_sample_waypoint_policy(policy_context)
            waypoint_decision = replace(
                waypoint_decision,
                physical_current_node_id=str(
                    physical_visual_context.current_node_id
                ),
                planning_current_node_id=str(
                    planning_visual_context.current_node_id
                ),
                backtrack_contexts=[
                    deepcopy(item) for item in backtrack_contexts
                ],
            )
            failure_reason = str(waypoint_decision.failure_reason)
            failed_backtrack_index = None
            if failure_reason == "no_projected_sampled_waypoint_candidates":
                for index in range(len(backtrack_contexts) - 1, -1, -1):
                    context = backtrack_contexts[index]
                    if (str(context.get("anchor_node_id", "")) == str(planning_visual_context.current_node_id)
                            and not isinstance(context.get("result"), dict)):
                        failed_backtrack_index = index
                        break
            if failure_reason == "navigation_intent_has_no_matching_waypoint" or failed_backtrack_index is not None:
                failure = {
                    "failure_reason": failure_reason,
                    "execution_started": False,
                    "physical_pose": deepcopy(physical_visual_context.views[0].pose) if physical_visual_context.views else {},
                    "failed_planning_node_id": str(planning_visual_context.current_node_id),
                    "skill_event_index": next(event["event_index"] for event in reversed(session.working_memory)
                                              if event["event_type"] == "skill_selection"),
                    "grounding_feedback": deepcopy(waypoint_decision.navigation_replan_feedback),
                    "observation_ids": [view.obs_id for view in planning_visual_context.views],
                    "candidate_count": next((item["candidate_count"] for item in waypoint_decision.navigation_replan_feedback
                                             if "candidate_count" in item), 0 if failed_backtrack_index is not None else None),
                }
                if failed_backtrack_index is not None:
                    failed_backtrack = backtrack_contexts[failed_backtrack_index]
                    failure["backtrack_index"] = failed_backtrack_index
                    failure["failure_summary"] = (
                        "This planning reference currently has no usable/projectable waypoint candidates. "
                        "No return movement has started."
                    )
                    failed_backtrack["result"] = {
                        "status": "failed", "execution_started": False,
                        "navigation_action_mode": navigation_mode.action_mode,
                        "detail": failure["failure_summary"],
                    }
                    session.rejected_backtrack_anchor_node_ids.add(str(failed_backtrack["anchor_node_id"]))
                    trigger = str(failed_backtrack["trigger_planning_node_id"])
                    session.planning_visual_context = (
                        physical_visual_context if trigger == str(physical_visual_context.current_node_id)
                        else build_visual_action_context_for_node(state=state, node_id=trigger)
                    )
                    session.planning_landmark_context = build_landmark_context(
                        landmark_controller=state.landmark_controller, visual_context=session.planning_visual_context,
                    )
                failure["resumed_planning_node_id"] = str(session.planning_visual_context.current_node_id)
                session.last_failure = deepcopy(failure)
                session.record("grounding_failure", state=state, **failure)
                session.navigation_mode = None
                session.task_state_is_current = False
                navigation_replan_feedback.extend(waypoint_decision.navigation_replan_feedback)
                if not waypoint_decision.navigation_replan_feedback:
                    navigation_replan_feedback.append(deepcopy(failure))
                session.update_trace(state, step)
                if navigation_attempt_index + 1 < session.max_execution_preparations:
                    waypoint_attempts.extend(waypoint_decision.waypoint_attempts)
                    continue
                # Keep the failed attempt's visual/planning reference, but
                # include the failure just recorded in the session history.
                # last_failure identifies both failed and resumed references.
                return _with_replan_log(
                    replace(
                        waypoint_decision,
                        navigation_replan_feedback=[],
                        backtrack_contexts=deepcopy(session.backtrack_contexts),
                    ),
                    waypoint_attempts=waypoint_attempts,
                    navigation_replan_feedback=navigation_replan_feedback,
                )
            return _with_replan_log(
                waypoint_decision,
                waypoint_attempts=waypoint_attempts,
                navigation_replan_feedback=navigation_replan_feedback,
            )
        if navigation_mode.action_mode == "vertical_transition":
            request_error = (
                "vertical_transition_requires_waypoint_target"
                if not navigation_mode.waypoint_target else None
            )
            if request_error is not None:
                navigation_replan_feedback.append(
                    {
                        "action_mode": "vertical_transition",
                        "failure_reason": request_error,
                        "failure_summary": request_error,
                    }
                )
                if navigation_attempt_index + 1 < session.max_execution_preparations:
                    navigation_mode = None
                    continue
                return _visual_navigation_failure(
                    visual_context=planning_visual_context,
                    task_state_initialization=task_state_initialization,
                    navigation_mode=navigation_mode,
                    task_state_update_result=task_state_update_result,
                    failure_reason=request_error,
                    waypoint_attempts=waypoint_attempts,
                    navigation_replan_feedback=navigation_replan_feedback,
                )
            return VisualNavigationDecision(
                visual_context=planning_visual_context,
                task_state_initialization=task_state_initialization,
                navigation_mode=navigation_mode,
                task_state_update_result=task_state_update_result,
                waypoint=None,
                action_call=ActionCall(
                    action="vertical_transition",
                    args={
                        "direction": str(navigation_mode.vertical_direction),
                        **({
                            "waypoint_target": navigation_mode.waypoint_target,
                            "subgoal_id": navigation_mode.subgoal_id,
                            "subgoal_attempt": navigation_mode.subgoal_attempt,
                            "task_context": "\n".join(
                                str(block.get("text", ""))
                                for block in _waypoint_inherited_agent_context(navigation_mode=navigation_mode)
                            ),
                        }),
                    },
                ),
                waypoint_attempts=waypoint_attempts,
                navigation_replan_feedback=navigation_replan_feedback,
                physical_current_node_id=str(
                    physical_visual_context.current_node_id
                ),
                planning_current_node_id=str(
                    planning_visual_context.current_node_id
                ),
                backtrack_contexts=[
                    deepcopy(item) for item in backtrack_contexts
                ],
            )
        raise ValueError(f"unsupported navigation mode: {navigation_mode.action_mode!r}")
    return _visual_navigation_failure(
        visual_context=planning_visual_context,
        task_state_initialization=task_state_initialization,
        navigation_mode=navigation_mode,
        task_state_update_result=task_state_update_result,
        waypoint_attempts=waypoint_attempts,
        failure_reason="navigation_replan_exhausted",
        navigation_replan_feedback=navigation_replan_feedback,
    )


def _plan_sampled_waypoint_policy(
    context: WaypointPlanningContext,
    *,
    policy_name: str,
    source_type: str,
    selection_kind: str,
    no_action_failure: str,
    sample_spacing_m: float | None = None,
    max_distance_m: float | None = None,
    enable_node_dedup: bool = True,
    use_global_node_dedup: bool = False,
) -> VisualNavigationDecision:
    state = context.state
    navigation_mode = context.navigation_mode
    planner_visual_context = context.visual_context
    action_origin_visual_context = (
        planner_visual_context
        if context.execution_visual_context is None
        else context.execution_visual_context
    )
    # The decision session owns reference switching; grounding consumes it.
    planner_context_text = _waypoint_planner_context_text(
        state=state,
        navigation_mode=navigation_mode,
        current_node_id=str(action_origin_visual_context.current_node_id),
        planner_node_id=str(planner_visual_context.current_node_id),
        initial_orientation_note=_initial_orientation_note(
            goal_kind=context.goal_kind,
            step_index=int(context.step.place_step_index),
        ),
    )
    planner_landmark_context = context.landmark_context
    fss = state.fss_config
    sampling_kwargs: dict[str, object] = dict(
        sampling_config=fss, sample_spacing_m=fss.sample_spacing_m, max_distance_m=fss.max_distance_m)
    if sample_spacing_m is not None:
        sampling_kwargs["sample_spacing_m"] = float(sample_spacing_m)
    if max_distance_m is not None:
        sampling_kwargs["max_distance_m"] = float(max_distance_m)
    waypoint_context_kwargs: dict[str, object] = {
        "active_agenda_item": str(context.active_agenda_item),
        "candidate_bev_landmark_markers": list(
            context.candidate_bev_landmark_markers or []
        ),
        "candidate_bev_base_image": context.candidate_bev_base_image,
        "candidate_bev_transform": context.candidate_bev_transform,
        "candidate_bev_reference_node_marker": (
            context.candidate_bev_reference_node_marker
        ),
        "candidate_bev_context_text": str(context.candidate_bev_context_text),
    }
    waypoint_context_kwargs["node_dedup_radius_m"] = fss.node_dedup_radius_m
    if context.candidate_bev_base_image is not None:
        waypoint_context_kwargs["candidate_bev_coordinate_exploration"] = (
            state.global_exploration_for_floor(
                str(state.system.current_floor_id)
            )
        )
    if (
        enable_node_dedup
        and use_global_node_dedup
        and str(context.goal_kind) == NAVPROBE_LANGUAGE_GOAL_KIND
    ):
        waypoint_context_kwargs["node_dedup_map"] = state.global_exploration_for_floor(
            str(state.system.current_floor_id)
        ).map
        waypoint_context_kwargs["node_dedup_radius_m"] = float(
            fss.global_node_dedup_radius_m
        )
    if context.candidate_cache is not None:
        waypoint_context_kwargs["candidate_cache"] = context.candidate_cache
    result = ground_waypoint(
        client=state.llm_client,
        cache=state.cache,
        goal_text="",
        visual_context=planner_visual_context,
        task_state_assessment="",
        task_state_text="",
        exploration=_waypoint_local_exploration(
            state=state,
            node_id=str(planner_visual_context.current_node_id),
        ),
        floor_height_m=float(state.system.current_floor_height),
        planner_context_text=planner_context_text,
        planner_context_images=(
            _waypoint_selection_images_for_node(
                state,
                str(planner_visual_context.current_node_id),
                include_node_identity=planner_visual_context.graph_context_visible,
            )
            if str(planner_visual_context.current_node_id)
            != str(action_origin_visual_context.current_node_id)
            else []
        ),
        inherited_agent_context_content=context.inherited_agent_context_content,
        landmark_evidences=planner_landmark_context.evidences,
        frontier_records=state.frontier_records_for_floor(str(state.system.current_floor_id)),
        avoid_node_xys=(
            _waypoint_avoid_node_xys(
                state=state,
                planner_node_id=str(planner_visual_context.current_node_id),
                floor_id=str(state.system.current_floor_id),
            )
            if enable_node_dedup
            else []
        ),
        floor_id=str(state.system.current_floor_id),
        **waypoint_context_kwargs,
        **sampling_kwargs,
    )
    waypoint_attempts = _waypoint_attempts_from_records(result.attempt_records)
    if result.failure_reason != "" or result.waypoint is None:
        return replace(
            _visual_navigation_failure(
                visual_context=planner_visual_context,
                task_state_initialization=context.task_state_initialization,
                navigation_mode=navigation_mode,
                task_state_update_result=context.task_state_update_result,
                waypoint_attempts=waypoint_attempts,
                failure_reason=str(result.failure_reason or no_action_failure),
                local_move_plan=result.local_move_plan,
                navigation_replan_feedback=result.navigation_replan_feedback,
            ),
            waypoint_policy_name=policy_name,
        )
    waypoint = replace(
        result.waypoint,
        goal_yaw=_waypoint_start_to_goal_yaw(
            state=state,
            visual_context=planner_visual_context,
            waypoint=result.waypoint,
        ),
    )
    selected_label: int | None = None
    candidate_overlay: np.ndarray | None = None
    if result.attempt_records != []:
        record = result.attempt_records[-1]
        sampled_candidate = record.get("sampled_candidate", {})
        selected_label = sampled_candidate.get("label")
        overlay_images = record.get("overlay_images", {})
        if isinstance(overlay_images, dict) and isinstance(
            overlay_images.get("sampled_candidate_rgb_overlay"),
            np.ndarray,
        ):
            candidate_overlay = np.asarray(overlay_images["sampled_candidate_rgb_overlay"], dtype=np.uint8)
    _store_waypoint_selection_for_node(
        state=state,
        waypoint=waypoint,
        reasoning="" if result.local_move_plan is None else str(result.local_move_plan.reasoning),
        planner_node_id=str(planner_visual_context.current_node_id),
        selection_kind=selection_kind,
        selected_label=selected_label,
        overlay_image=candidate_overlay,
    )
    source_id = ""
    if waypoint_attempts:
        sampled_candidate = waypoint_attempts[-1].sampled_candidate
        if sampled_candidate.get("label") is not None:
            source_id = str(sampled_candidate["label"])
    target = GroundedWaypointTarget(
        goal_xy=(float(waypoint.goal_xy[0]), float(waypoint.goal_xy[1])),
        goal_yaw=float(waypoint.goal_yaw),
        world_z=float(waypoint.raw_world_z),
        policy_name=policy_name,
        source_type=source_type,
        source_id=source_id,
        obs_id=str(waypoint.obs_id),
        angle_deg=int(waypoint.angle_deg),
        point_2d=(float(waypoint.point_2d[0]), float(waypoint.point_2d[1])),
        raw_world_xy=(float(waypoint.raw_world_xy[0]), float(waypoint.raw_world_xy[1])),
    )
    return VisualNavigationDecision(
        visual_context=planner_visual_context,
        task_state_initialization=context.task_state_initialization,
        navigation_mode=navigation_mode,
        task_state_update_result=context.task_state_update_result,
        waypoint=waypoint,
        action_call=target.to_action_call(),
        local_move_plan=result.local_move_plan,
        waypoint_attempts=waypoint_attempts,
        navigation_replan_feedback=result.navigation_replan_feedback,
        waypoint_policy_name=policy_name,
        grounded_waypoint_target=target,
    )


def _store_waypoint_selection_for_node(
    *,
    state: "NavProbeAgentState",
    waypoint: VisualWaypoint,
    reasoning: str,
    planner_node_id: str,
    selection_kind: str = "visual_waypoint",
    selected_label: int | None = None,
    overlay_image: np.ndarray | None = None,
) -> None:
    if overlay_image is None:
        observation = state.cache.get_observation(str(waypoint.obs_id)).observation
        image = draw_waypoint_overlay_rgb(
            observation.rgb,
            point_pixel=(float(waypoint.point_pixel[0]), float(waypoint.point_pixel[1])),
        )
    else:
        image = np.asarray(overlay_image, dtype=np.uint8)
    image_record = state.cache.store_image(
        image,
        kind="last_waypoint_selection_overlay",
        metadata={
            "selection_kind": str(selection_kind),
            "obs_id": str(waypoint.obs_id),
            "angle_deg": int(waypoint.angle_deg),
            "planner_node_id": str(planner_node_id),
            "selected_label": None if selected_label is None else int(selected_label),
        },
    )
    state.waypoint_selection_context_by_node_id[str(planner_node_id)] = {
        "selection_kind": str(selection_kind),
        "selected_angle_deg": int(waypoint.angle_deg),
        "planner_node_id": str(planner_node_id),
        "selected_label": None if selected_label is None else int(selected_label),
        "reasoning": str(reasoning),
        "image_id": str(image_record.id),
    }


def _waypoint_start_to_goal_yaw(
    *,
    state: "NavProbeAgentState",
    visual_context: VisualActionContext,
    waypoint: VisualWaypoint,
) -> float:
    start_xy = _waypoint_start_xy(state=state, visual_context=visual_context)
    goal_xy = np.asarray(waypoint.goal_xy, dtype=np.float64).reshape(2)
    delta = goal_xy - start_xy
    if float(np.linalg.norm(delta)) <= 1e-6:
        return float(waypoint.goal_yaw)
    return float(math.degrees(math.atan2(float(delta[1]), float(delta[0]))))


def _waypoint_start_xy(
    *,
    state: "NavProbeAgentState",
    visual_context: VisualActionContext,
) -> np.ndarray:
    current_view = visual_context.view_for_angle(0)
    observation = state.cache.get_observation(str(current_view.obs_id)).observation
    if observation.T_odom_base is not None:
        transform = np.asarray(observation.T_odom_base, dtype=np.float64)
        return np.asarray([float(transform[0, 3]), float(transform[1, 3])], dtype=np.float64)
    return np.asarray([float(observation.pose.x), float(observation.pose.y)], dtype=np.float64)


def _waypoint_local_exploration(
    *,
    state: "NavProbeAgentState",
    node_id: str,
) -> "ExplorationManager":
    node_id_text = str(node_id).strip()
    if node_id_text == "":
        raise ValueError("VLN waypoint sampling requires a planner node id")
    exploration = state.graph.get_node(node_id_text).localmap
    if exploration is None:
        raise ValueError(f"missing local exploration for VLN waypoint node {node_id_text!r}")
    return exploration


def _waypoint_avoid_node_xys(
    *,
    state: "NavProbeAgentState",
    planner_node_id: str,
    floor_id: str,
) -> list[tuple[float, float]]:
    planner_node_id_text = str(planner_node_id).strip()
    floor_id_text = str(floor_id).strip()
    avoid_xys: list[tuple[float, float]] = []
    with state.graph.lock:
        for node in state.graph.iter_nodes(floor_id=floor_id_text):
            if str(node.node_kind) != "place":
                continue
            if str(node.id) == planner_node_id_text:
                continue
            avoid_xys.append((float(node.position[0]), float(node.position[1])))
    return avoid_xys


def _waypoint_attempts_from_records(records: list[dict[str, object]]) -> list[VisualWaypointAttempt]:
    attempts: list[VisualWaypointAttempt] = []
    for index, record in enumerate(records):
        attempts.append(
            VisualWaypointAttempt(
                attempt_index=int(index),
                selected_angle_deg=int(record.get("selected_angle_deg", 0)),
                selected_obs_id=str(record.get("selected_obs_id", "")),
                waypoint_target=str(record.get("waypoint_target", "")),
                failure_reason=str(record.get("failure_reason", "")),
                sampled_candidate=(
                    dict(record.get("sampled_candidate", {}))
                    if isinstance(record.get("sampled_candidate"), dict)
                    else {}
                ),
                candidate_selection=(
                    dict(record.get("candidate_selection", {}))
                    if isinstance(record.get("candidate_selection"), dict)
                    else {}
                ),
                candidate_count=(
                    int(record["candidate_count"])
                    if record.get("candidate_count") is not None
                    else None
                ),
            )
        )
    return attempts


def _plan_frontier_skeleton_sample_waypoint_policy(
    context: WaypointPlanningContext,
) -> VisualNavigationDecision:
    validate_waypoint_policy(waypoint_policy_name=context.state.waypoint_policy_name)
    return _plan_sampled_waypoint_policy(
        context,
        policy_name=FRONTIER_SKELETON_SAMPLE_WAYPOINT_POLICY,
        source_type="sample",
        selection_kind="sampled_waypoint_candidate",
        no_action_failure="vln_waypoint_no_action",
        use_global_node_dedup=True,
    )


def _plan_frontier_skeleton_sample_stop_policy(
    context: WaypointPlanningContext,
) -> VisualNavigationDecision:
    return _plan_sampled_waypoint_policy(
        context,
        policy_name=context.state.waypoint_policy_name,
        source_type="stop_sample",
        selection_kind="stop_sampled_waypoint_candidate",
        no_action_failure="vln_stop_waypoint_no_action",
        sample_spacing_m=context.state.fss_config.stop_sample_spacing_m,
        max_distance_m=context.state.fss_config.stop_max_distance_m,
        enable_node_dedup=False,
    )


def _visual_navigation_failure(
    *,
    visual_context: VisualActionContext,
    task_state_initialization: dict[str, object],
    navigation_mode: NavProbeSkillDecision,
    task_state_update_result: NavProbeTaskStateUpdateResult,
    waypoint_attempts: list[VisualWaypointAttempt],
    failure_reason: str,
    local_move_plan: NavProbeWaypointDecision | None = None,
    navigation_replan_feedback: list[dict[str, object]] | None = None,
) -> VisualNavigationDecision:
    return VisualNavigationDecision(
        visual_context=visual_context,
        task_state_initialization=task_state_initialization,
        navigation_mode=navigation_mode,
        task_state_update_result=task_state_update_result,
        waypoint=None,
        action_call=None,
        local_move_plan=local_move_plan,
        waypoint_attempts=waypoint_attempts,
        navigation_replan_feedback=[dict(item) for item in (navigation_replan_feedback or [])],
        failure_reason=str(failure_reason),
    )


def _goal_text(
    *,
    state: "NavProbeAgentState",
    goal: "NavProbeGoalSpec | None",
) -> str:
    if goal is not None:
        navigation_goal_text = getattr(goal, "navigation_goal_text", None)
        if callable(navigation_goal_text):
            goal_text = str(navigation_goal_text()).strip()
            if goal_text != "":
                return goal_text
        description = str(getattr(goal, "description", "")).strip()
        if description != "":
            return description
    return str(state.system.goal.target)

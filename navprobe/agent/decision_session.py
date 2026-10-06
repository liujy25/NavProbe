"""Transient state for one physical observation's Executive/grounding decision."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import TYPE_CHECKING

from navprobe.agent.ablation import compact_memory_only_enabled, knowledge_consolidation_enabled, passive_full_history_enabled
from navprobe.agent.episodic_retrieval import RetrievalWorkspace
from navprobe.agent.landmark_context import draw_landmark_panorama_views
from navprobe.memory.task_state import NavProbeTaskStateUpdateResult

if TYPE_CHECKING:
    import numpy as np
    from navprobe.agent.visual_action_context import VisualActionContext
    from navprobe.agent.navigation_decisions import NavProbeSkillDecision, NavProbeTaskStateDecision
    from navprobe.agent.landmark_context import NavProbeLandmarkContext



@dataclass
class NavProbeDecisionSession:
    max_execution_preparations: int
    initialized: bool = False
    physical_visual_context: VisualActionContext | None = None
    planning_visual_context: VisualActionContext | None = None
    planning_landmark_context: object = None
    workspace: RetrievalWorkspace | None = None
    task_state_initialization: dict = field(default_factory=dict)
    task_state_decision: NavProbeTaskStateDecision | None = None
    # One cumulative record of committed updates; consumers receive snapshots.
    task_state_update_result: NavProbeTaskStateUpdateResult = field(default_factory=NavProbeTaskStateUpdateResult)
    task_state_update_transactions: list = field(default_factory=list)
    task_state_update_count: int = 0
    task_state_is_current: bool = False
    pending_terminal_check: dict | None = None
    terminal_check_required: bool = False
    terminal_check_payload: dict | None = None
    passive_preloaded_planning_node_id: str = ""
    passive_preloaded_context: object = None
    passive_latest_incoming_edge_ref: str = ""
    backtrack_contexts: list = field(default_factory=list)
    rejected_backtrack_anchor_node_ids: set[str] = field(default_factory=set)
    working_memory: list[dict] = field(default_factory=list)
    executive_inputs: list[dict] = field(default_factory=list)
    navigation_replan_feedback: list[dict] = field(default_factory=list)
    waypoint_attempts: list = field(default_factory=list)
    candidate_cache: dict = field(default_factory=dict)
    decision_count: int = 0
    executive_count: int = 0
    skill_count: int = 0
    preparation_count: int = 0
    navigation_mode: NavProbeSkillDecision | None = None
    failure_reason: str = ""
    last_failure: dict = field(default_factory=dict)
    knowledge_management_complete: bool = False
    knowledge_management_payload: dict | None = None
    closed: bool = False
    _landmark_panorama_source: tuple[VisualActionContext, NavProbeLandmarkContext] | None = field(default=None, init=False, repr=False)
    _landmark_panorama_views: dict[int, np.ndarray] | None = field(default=None, init=False, repr=False)

    @property
    def max_decisions(self):
        return 2 * self.workspace.max_retrieve_rounds + 3 + 2 * self.max_execution_preparations

    def initialize_context(self, visual_context, landmark_context):
        self.physical_visual_context = visual_context
        self.planning_visual_context = visual_context
        self.planning_landmark_context = landmark_context

    def get_landmark_panorama_views(self, *, cache):
        """Reuse only the current reference's annotated panorama in this session."""
        visual_context = self.planning_visual_context
        landmark_context = self.planning_landmark_context
        if landmark_context is None or not landmark_context.evidences:
            self._landmark_panorama_source = None
            self._landmark_panorama_views = None
            return None
        source = self._landmark_panorama_source
        # Contexts stay unchanged during assessment/retrieval/skill selection.
        # Backtrack and recovery replace them, invalidating this single entry.
        if source is None or source[0] is not visual_context or source[1] is not landmark_context:
            self._landmark_panorama_source = None
            self._landmark_panorama_views = None
            self._landmark_panorama_views = draw_landmark_panorama_views(
                cache=cache, visual_context=visual_context, evidences=landmark_context.evidences,
            )
            self._landmark_panorama_source = (visual_context, landmark_context)
        return self._landmark_panorama_views

    def counters(self):
        return {
            "decisions": self.decision_count, "max_decisions": self.max_decisions,
            "executive": self.executive_count, "skills": self.skill_count,
            "execution_preparations": self.preparation_count,
            "max_execution_preparations": self.max_execution_preparations,
            "backtrack_switches": len(self.backtrack_contexts),
            "max_backtrack_switches": self.max_execution_preparations,
            "retrieve_count": self.workspace.retrieve_count,
            "max_retrieve_rounds": self.workspace.max_retrieve_rounds,
        }

    @staticmethod
    def skill_payload(decision):
        # Explicit compact snapshot of the selected skill and its reason.
        return {name: deepcopy(getattr(decision, name)) for name in (
            "action_mode", "direction", "action_objective", "action_reason", "reasoning_action",
            "waypoint_target", "stop_objective", "approach_movement", "subgoal_id", "subgoal_attempt",
            "backtrack_anchor_node_id", "backtrack_objective", "backtrack_reason",
        )}

    def record(self, event_type, *, state, **payload):
        event = {
            "event_index": len(self.working_memory), "decision_index": self.decision_count,
            "event_type": event_type,
            "physical_node_id": str(self.physical_visual_context.current_node_id),
            "planning_node_id": str(self.planning_visual_context.current_node_id),
            **deepcopy(payload),
        }
        request_index = getattr(state.llm_client, "_request_index", None)
        if type(request_index) is int:
            event["llm_request_index"] = request_index
        # Reject accidental image/object graphs rather than retaining raw evidence.
        json.dumps(event, ensure_ascii=False, allow_nan=False)
        self.working_memory.append(event)
        return event

    def serialized_events(self):
        events = deepcopy(self.working_memory)
        for event in events:
            kind = event["event_type"]
            if kind == "retrieval_conclusion":
                record = next(r for r in self.workspace.rounds if r.round_index == event["round_index"])
                event.update(conclusion=record.conclusion, source_obs_ids=list(record.source_obs_ids))
                if record.unavailable_fields:
                    event["unavailable_fields"] = dict(record.unavailable_fields)
            if "transaction_index" in event:
                transaction = self.task_state_update_transactions[event["transaction_index"]]
                if kind == "retrieve_action":
                    request = transaction["request"]
                    event.update(query=request["query"], items=deepcopy(request["items"]),
                                 status=transaction["status"],
                                 already_provided_items=deepcopy(request.get("already_provided_items", [])))
                    for name in ("error", "error_type", "committed_round_index"):
                        if name in transaction:
                            event[name] = transaction[name]
                elif kind == "state_update":
                    event.update(applied_updates=deepcopy(transaction["applied_updates"]),
                                 skipped_updates=deepcopy(transaction["skipped_updates"]),
                                 committed=transaction["status"] != "not_committed")
        return events

    def working_memory_text(self):
        if not self.working_memory:
            return ""
        # Keep the complete events in the trace. The model receives their
        # decision content in the same order, without request/debug metadata.
        diagnostic_fields = {
            "event_index", "decision_index", "event_type", "llm_request_index",
            "transaction_index", "skill_event_index", "backtrack_index",
            "physical_pose", "counters",
        }
        labels = {
            "task_state_assessment": "Earlier Executive assessment",
            "state_update": "Applied task-state update",
            "retrieve_action": "Retrieval request",
            "retrieval_conclusion": "Recorded retrieval conclusion",
            "skill_selection": "Selected skill and intent",
            "planning_reference_change": "Planning reference change",
            "grounding_failure": "Grounding feedback",
            "session_end": "Decision outcome",
        }
        sections = [
            "Current decision working memory (chronological):\n"
            "These earlier assessments and retrievals explain how this decision developed. "
            "Reconsider earlier judgments using the current task state and available evidence."
        ]
        for event in self.serialized_events():
            kind = event["event_type"]
            if (kind == "state_update" and event.get("committed")
                    and not event.get("applied_updates") and not event.get("skipped_updates")):
                continue
            payload = {key: value for key, value in event.items() if key not in diagnostic_fields}
            assessment = payload.pop("task_state_assessment", "") if kind == "task_state_assessment" else ""
            section = labels.get(kind, kind) + ":\n"
            if assessment:
                section += assessment + "\n"
            section += json.dumps(payload, ensure_ascii=False)
            if kind == "grounding_failure" and event.get("execution_started") is False:
                section += (
                    "\nNo movement started for this attempt. Reassess the next objective using "
                    "this feedback and the preceding skill intent. The feedback describes a local "
                    "waypoint-selection failure; it does not establish that the goal is absent, "
                    "the node is unreachable, or the objective has failed."
                )
            sections.append(section)
        sections.append("Current reference: " + json.dumps({
            "physical_node_id": self.physical_visual_context.current_node_id,
            "planning_node_id": self.planning_visual_context.current_node_id,
        }, ensure_ascii=False))
        return "\n\n".join(sections)

    def finalize_retrieval_handoff(self, *, state, knowledge_manager):
        """Consolidate retrieval evidence without closing the decision session.

        Called by :meth:`finish` before recording the final trace and releasing
        the session. Retrieval evidence remains available during consolidation.
        """
        if self.workspace is None:
            raise RuntimeError("retrieval handoff requires an initialized workspace")
        if self.workspace.has_pending_evidence:
            raise RuntimeError("cannot hand off a session with unexplained retrieval evidence")
        if self.workspace.rounds and not self.knowledge_management_complete:
            if knowledge_consolidation_enabled(state.ablation_name):
                self.knowledge_management_payload = knowledge_manager(
                    client=state.llm_client, state=state, workspace=self.workspace,
                    current_node_id=str(self.physical_visual_context.current_node_id),
                    task_state_updates=deepcopy(self.task_state_update_result.applied_updates),
                )
            else:
                self.knowledge_management_payload = {
                    "status": "skipped", "reason": "ablation:no_consolidation",
                    "retrieve_count": self.workspace.retrieve_count,
                    "applied_updates": [], "skipped_updates": [],
                }
            self.knowledge_management_complete = True
        return self.knowledge_management_payload

    def finish(self, *, state, step, decision, knowledge_manager):
        if self.closed:
            raise RuntimeError("decision session already closed")
        self.failure_reason = decision.failure_reason
        self.navigation_mode = None if decision.failure_reason else decision.navigation_mode
        self.finalize_retrieval_handoff(
            state=state,
            knowledge_manager=knowledge_manager,
        )
        self.record("session_end", state=state, status=("failed" if self.failure_reason else
                            "done" if decision.action_call is not None and decision.action_call.action == "done" else "handoff"),
                    failure_reason=self.failure_reason, last_failure=self.last_failure, counters=self.counters())
        self.update_trace(state, step, final_navigation_mode=self.navigation_mode)
        self.release()
        return decision

    def abort(self, *, state, step, error):
        # Never make an additional model request on an exception path.
        if self.workspace is not None:
            self.record("session_end", state=state, status="exception", error_type=type(error).__name__,
                        error=str(error), last_failure=self.last_failure, counters=self.counters())
            self.failure_reason = self.failure_reason or "decision_session_exception"
            self.update_trace(state, step)
        self.release()

    def release(self):
        self.candidate_cache.clear()
        self._landmark_panorama_source = None
        self._landmark_panorama_views = None
        if self.workspace is not None:
            self.workspace.clear_materialized_evidence()
        self.passive_preloaded_context = None
        self.planning_landmark_context = None
        self.physical_visual_context = None
        self.planning_visual_context = None
        self.closed = True

    def update_trace(
        self, state, step,
        *,
        final_navigation_mode: NavProbeSkillDecision | None = None,
        terminal_check: dict[str, object] | None = None,
    ) -> None:
        """Snapshot decision state without changing the session's control state."""
        task_state_memory = state.system.task_state
        passive_full_history = passive_full_history_enabled(state.ablation_name)
        consolidate_retrieved_knowledge = knowledge_consolidation_enabled(state.ablation_name)
        memory_context_mode = ("passive_full_history" if passive_full_history else
                               "text_index" if compact_memory_only_enabled(state.ablation_name) else "active_retrieval")
        payload = self.workspace.to_dict()
        payload["task_state_update_count"] = int(self.task_state_update_count)
        payload["task_state_memory"] = task_state_memory.to_dict()
        payload["task_state_update_result"] = self.task_state_update_result.to_dict()
        payload["task_state_update_transactions"] = deepcopy(self.task_state_update_transactions)
        payload["post_approach_terminal_check"] = bool(self.terminal_check_required)
        payload["physical_current_node_id"] = str(
            self.physical_visual_context.current_node_id
        )
        payload["planning_current_node_id"] = str(
            self.planning_visual_context.current_node_id
        )
        payload["backtrack_contexts"] = [
            deepcopy(item) for item in self.backtrack_contexts
        ]
        payload["memory_context_mode"] = memory_context_mode
        payload["knowledge_consolidation_enabled"] = bool(
            consolidate_retrieved_knowledge
        )
        payload["passive_full_history_preloaded"] = bool(
            passive_full_history and self.passive_preloaded_context is not None
        )
        if passive_full_history:
            preloaded_context_payload = (
                {"status": "pending"}
                if self.passive_preloaded_context is None
                else self.passive_preloaded_context.to_dict()
            )
            preloaded_context_payload["planning_node_id"] = str(
                self.planning_visual_context.current_node_id
            )
            preloaded_context_payload["latest_incoming_edge_ref"] = str(
                self.passive_latest_incoming_edge_ref
            )
            preloaded_context_payload["latest_incoming_edge_preloaded"] = bool(
                self.passive_latest_incoming_edge_ref != ""
            )
            payload["preloaded_context"] = preloaded_context_payload
        if self.task_state_decision is not None:
            payload["final_task_state_update"] = self.task_state_decision.to_dict()
        if self.task_state_update_result.applied_updates:
            payload["task_state_update_history"] = deepcopy(self.task_state_update_result.applied_updates)
        if final_navigation_mode is not None:
            payload["final_navigation_mode"] = final_navigation_mode.to_dict()
        resolved_terminal_check = (
            self.terminal_check_payload if terminal_check is None else terminal_check
        )
        if resolved_terminal_check is not None:
            payload["terminal_check"] = deepcopy(resolved_terminal_check)
        if self.knowledge_management_payload is not None:
            payload["knowledge_management"] = deepcopy(
                self.knowledge_management_payload
            )
        payload["working_memory"] = self.serialized_events()
        payload["executive_inputs"] = deepcopy(self.executive_inputs)
        payload["counters"] = self.counters()
        payload["rejected_backtrack_anchor_node_ids"] = sorted(self.rejected_backtrack_anchor_node_ids)
        if self.failure_reason:
            payload["failure_reason"] = self.failure_reason
        step.episodic_retrieval = payload

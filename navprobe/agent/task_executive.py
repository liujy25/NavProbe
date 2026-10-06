"""Task Executive assessment, agenda initialization and response validation."""
from __future__ import annotations

from copy import deepcopy
import json
from typing import TYPE_CHECKING

import numpy as np

from navprobe.agent.decision_inputs import request_task_response, task_panorama_content, task_state_content
from navprobe.agent.episodic_retrieval import RetrieveRequest, normalize_retrieve_request
from navprobe.agent.navigation_decisions import NavProbeTaskStateDecision
from navprobe.agent.visual_policy_prompt_images import image_content_for_movement_history_sheet
from navprobe.memory.task_state import NavProbeAgendaItem
from navprobe.perception.goal import NAVPROBE_LANGUAGE_GOAL_KIND
from navprobe.agent.visual_action_context import VisualActionContext
from navprobe.memory.task_state import NavProbeTaskStateMemory

if TYPE_CHECKING:
    from navprobe.llm.client import LLMClient
    from navprobe.runtime.cache import RuntimeCache


def ensure_task_state_memory(
    *,
    client: "LLMClient",
    task_state: NavProbeTaskStateMemory,
    goal_text: str,
    goal_kind: str = "",
) -> dict[str, object]:
    if task_state.agenda_initialized:
        task_state.initialize_agenda(goal_text, [])
        return {"initialized": False, "source": "existing", "task_state": task_state.to_dict()}
    if task_state.task_constraints:
        task_state.initialize_agenda(goal_text, [])
        return {"initialized": True, "source": "online_objectives", "task_state": task_state.to_dict()}
    if task_state.agenda != []:
        task_state.initialize_agenda(goal_text, task_state.agenda)
        return {"initialized": False, "source": "existing", "task_state": task_state.to_dict()}
    normalized_goal_kind = str(goal_kind).strip()
    system_prompt = """
You are the Initial Task Executive in NavProbe.
Turn the navigation instruction into an initial agenda for the online Task Executive.
""".strip()
    user_prompt = f"""
Original navigation instruction:
{goal_text}

The instruction defines the task; no observations are available at initialization.

Build the agenda:
- Create route-level objectives in instruction order. Keep each required movement with the landmarks and spatial relations that define it.
- Preserve negation, ordinal references, before/after dependencies, and all stated stopping constraints. Add no requirements beyond the instruction.
- When a phrase needs visual context to interpret, retain that ambiguity for the online Executive.

Output contract:
Return only JSON with a non-empty agenda. Each objective has non-empty content, status active, and an empty result:
{{"agenda":[{{"content":"<instruction-derived objective>","status":"active","result":""}}]}}
""".strip()
    initial_items = request_task_response(
        request=client.initialize_task_state,
        system_prompt=system_prompt,
        content=user_prompt,
        normalize_response=lambda parsed: _normalize_initial_agenda(
            parsed, goal_kind=normalized_goal_kind,
        ),
        retrieval_round=None,
        completed_retrieval_rounds=0,
    )
    task_state.initialize_agenda(goal_text, initial_items)
    return {"initialized": True, "source": "llm", "task_state": task_state.to_dict()}


def _normalize_initial_agenda(parsed: dict[str, object], *, goal_kind: str) -> list[NavProbeAgendaItem]:
    if not isinstance(parsed, dict) or set(parsed) != {"agenda"}:
        raise ValueError("NavProbe initialization requires only the agenda field")
    raw_items = parsed["agenda"]
    if not isinstance(raw_items, list) or raw_items == []:
        raise ValueError(f"initial agenda generator returned no agenda: {parsed!r}")
    for item in raw_items:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("content"), str)
            or not item["content"].strip()
        ):
            raise ValueError(f"initial agenda items require non-empty string content: {item!r}")
    if goal_kind == NAVPROBE_LANGUAGE_GOAL_KIND:
        expected_fields = {"content", "status", "result"}
        for item in raw_items:
            if set(item) != expected_fields:
                raise ValueError(
                    "VLN initial agenda items must contain exactly "
                    f"{sorted(expected_fields)!r}: {item!r}"
                )
            if (
                str(item.get("status", "")).strip() != "active"
                or str(item.get("result", "")).strip() != ""
            ):
                raise ValueError(
                    "VLN initial agenda items require non-empty content, "
                    f"status='active', and result='': {item!r}"
                )
    return [NavProbeAgendaItem.from_dict(item) for item in raw_items]



_NAVPROBE_TASK_STATE_TOOLS = {
    "retrieve",
    "update_task_state",
}


def _navprobe_executive_tool_schemas(
    *, allow_retrieve: bool, allow_update_task_state: bool,
    require_terminal_check: bool = False,
) -> list[str]:
    schemas = ["""### `update_predicates`
Apply evidence-backed predicate changes to an active subgoal using its stable `subgoal_id`.
Arguments: `predicate_updates`, a non-empty list of these operations:
- add: {"op":"add","subgoal_id":"sg0","content":"<evidence-checkable proposition>","status":"confirmed|unconfirmed"}
- update: {"op":"update","subgoal_id":"sg0","predicate_id":"pc0","status":"confirmed|unconfirmed"}
- rewrite: {"op":"rewrite","subgoal_id":"sg0","predicate_id":"pc0","content":"<revised proposition>","status":"confirmed|unconfirmed"}
- remove: {"op":"remove","subgoal_id":"sg0","predicate_id":"pc0"}
Predicate updates for active objectives are applied before agenda operations. You may also reference a historical objective that is reopened in this same response: its predicate edits apply to the new attempt immediately after reopening, while the historical snapshot is preserved. Newly added objectives receive IDs on commit and can be referenced in subsequent assessments.
JSON:
{"name":"update_predicates","arguments":{"predicate_updates":[{"op":"add","subgoal_id":"sg0","content":"<supported proposition>","status":"confirmed"}]}}"""]
    if allow_update_task_state:
        terminal_argument = ""
        agenda_example = '{"agenda_updates":[{"op":"complete","subgoal_id":"sg0","result":"<observed outcome>"}]}'
        if require_terminal_check:
            terminal_argument = """
Additional argument: `terminal_check`, an object with exactly `decision` (done|continue) and `missing_constraints` (a list of original-task constraints).
Use an empty list for done and a non-empty list for continue. Explain the decision in `task_state_assessment`, without a separate reason field.
"""
            terminal_argument += (
                "When calling retrieve, omit terminal_check from the entire response. Otherwise, call update_task_state with terminal_check, even if agenda_updates is empty.\n"
                if allow_retrieve else
                "Call update_task_state with terminal_check in this response, even if agenda_updates is empty.\n"
            )
            agenda_example = '{"agenda_updates":[],"terminal_check":{"decision":"continue","missing_constraints":["<unmet or unestablished original-task constraint>"]}}'
        schemas.append("""### `update_task_state`
Revise the agenda and preserve outcomes in execution history.
Arguments: `agenda_updates`, a list of these operations; an empty list is allowed:
- add: {"op":"add","content":"<new intermediate objective>","position":0}
- rewrite: {"op":"rewrite","subgoal_id":"sg0","content":"<refined objective>"}
- reorder: {"op":"reorder","subgoal_ids":["sg1","sg0"]}
- complete: {"op":"complete","subgoal_id":"sg0","result":"<execution summary and outcome>"}
- abandon: {"op":"abandon","subgoal_id":"sg0","result":"<attempt outcome and why it is no longer useful>"}
- reopen: {"op":"reopen","subgoal_id":"sg0","position":0}
`position` is the insertion index in the active agenda: 0 inserts first, and the current agenda length appends at the end. Each operation uses the agenda produced by the preceding operations. `reorder` lists every currently active ID exactly once. The system assigns new IDs on add. Complete and abandon remove the objective from the active agenda and archive this attempt with its evidence and spatial references. Reopen returns the same historical objective as a new attempt while retaining its previous outcome. Operations apply in list order as one validated transaction.
All agenda entries, including instruction-initialized objectives, support these operations.
""" + terminal_argument + 'JSON:\n{"name":"update_task_state","arguments":' + agenda_example + '}')
    if allow_retrieve:
        schemas.append("""### `retrieve`
Load historical entity fields to answer a question about task state or the next action.
Arguments: `query`, one decision-relevant question, and `items`, a non-empty list of entity refs and fields exposed by the memory index. Batch sufficient complementary fields for that question; exclude fields marked as provided.
The system loads the requested fields and calls Task Executive again to assess them. Omitting `retrieve` completes this assessment and passes it to the next stage.
JSON:
{"name":"retrieve","arguments":{"query":"<unresolved task-state question>","items":[{"ref":"<memory ref>","fields":["<available field>"]}]}}""")
    return schemas


def assess_task_state(
    *,
    client: "LLMClient",
    cache: "RuntimeCache",
    visual_context: VisualActionContext,
    task_state: NavProbeTaskStateMemory,
    memory_index_text: str,
    retrieval_workspace_content: list[dict[str, object]],
    retrieve_max_rounds: int,
    retrieve_completed_rounds: int,
    retrieve_fields_by_ref: dict[str, list[str]],
    allow_retrieve: bool,
    allow_update_task_state: bool,
    require_retrieval_conclusion: bool,
    retrieve_provided_fields_by_ref: dict[str, list[str]] | None = None,
    memory_index_context_only: bool = False,
    task_state_context_text: str = "",
    working_memory_text: str = "",
    backtrack_context_text: str = "",
    landmark_panorama_views: dict[int, np.ndarray] | None = None,
    detected_landmarks_text: str = "",
    terminal_check_context: dict[str, object] | None = None,
    planning_reference_panorama: bool = False,
    original_instruction: str | None = None,
    task_constraints: str = "",
    no_progress: bool = False,
) -> NavProbeTaskStateDecision | RetrieveRequest:
    """Assess task state and retrieve evidence using the Executive contract."""
    dynamic_agenda = not no_progress
    if not no_progress and not task_state.agenda_initialized:
        raise ValueError("NavProbe requires an initialized task agenda")
    if no_progress:
        allow_update_task_state = False
    if not no_progress and (not (allow_retrieve or allow_update_task_state)):
        raise ValueError("VLN task module has no available operation")
    terminal_check_required = isinstance(terminal_check_context, dict)
    if terminal_check_required and not allow_update_task_state and not no_progress:
        raise ValueError("post-approach terminal check requires `update_task_state`")
    tool_schemas = _navprobe_executive_tool_schemas(
        allow_retrieve=allow_retrieve,
        allow_update_task_state=allow_update_task_state,
        require_terminal_check=terminal_check_required,
    )
    retrieve_remaining = max(
        0,
        int(retrieve_max_rounds) - int(retrieve_completed_rounds),
    )
    retrieval_catalog_section = ""
    retrieval_semantics = ""
    if allow_retrieve or (dynamic_agenda and (not memory_index_context_only)):
        single_retrieval_instruction = (
            "\n\nSingle-retrieval requirement:\n"
            "This step permits one retrieval request. Include all entity fields needed "
            "for the historical evidence question together in its `items` list."
            if allow_retrieve and int(retrieve_max_rounds) == 1
            and int(retrieve_completed_rounds) == 0
            else ""
        )
        retrieval_catalog_section = f"""

Episodic-memory text index:
{memory_index_text if str(memory_index_text).strip() != "" else "none"}

Retrieval budget for this navigation step:
- maximum rounds: {int(retrieve_max_rounds)}
- completed rounds: {int(retrieve_completed_rounds)}
- remaining rounds: {int(retrieve_remaining)}
- `retrieve` is {"available" if allow_retrieve else "unavailable"} for this call.
{single_retrieval_instruction}
""".rstrip()
        retrieval_semantics = """
Memory field semantics:
- `node.rgb`: the node's stored multi-direction panorama.
- `node.landmarks`: all landmark detections annotated in that panorama.
- `edge.rgb`: the stored start view with visible segments of the smoothed executed path overlaid. The endpoint marker is drawn only when visible. Blue arrows show the executed travel direction. If no overlay can be supplied, unavailable_fields reports the missing RGB; this does not establish that the edge was not traversed.
- `edge.trajectory`: the executed edge trajectory highlighted on the shared-floor BEV.
- `edge.movement_rgb`: traversal RGB keyframes ordered chronologically from left to right and then top to bottom.
- `landmark.rgb`: a local historical crop around the landmark with its bounding box.
- `provided_fields` are already present in this context; `available_fields` may be requested.
- Landmark refs use `landmark_<global index>`; image boxes display only the numeric suffix.
""".strip()
    elif memory_index_context_only:
        retrieval_catalog_section = f"""

Episodic-memory text index:
{memory_index_text if str(memory_index_text).strip() != "" else "none"}

Only the compact text index is available in this condition. No stored RGB, trajectory,
landmark overlay, or graph BEV evidence is attached.
""".rstrip()
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
    landmark_text = str(detected_landmarks_text).strip()
    conclusion_rules = (
        """
Latest retrieval evidence:
The latest query and requested entity fields appear in working memory. Their returned evidence is attached separately for this response to interpret. Fields in `unavailable_fields` were not returned; claims that depend on them remain unresolved.
""".strip()
        if require_retrieval_conclusion
        else ""
    )
    task_state_semantics = """
Input meanings:
- The original instruction defines the required actions, route order, and stopping relation. The agenda is a revisable plan for satisfying those requirements; its order sets pursuit priority.
- Execution history records earlier objective attempts and outcomes.
- The labeled panorama supplies visual evidence; its captions identify the observation reference. The memory index identifies historical entities and fields; each returned field and its images are evidence about that entity.

Assess progress:
- Check movement requirements against executed trajectories and the robot's position relative to landmarks. Seeing a doorway establishes visibility; crossing it requires evidence of movement through it. Judge arrival from the actual position relative to the instructed destination.
- Retain established execution events unless new evidence contradicts them. Recheck claims about the current position against current observations.
- Verify detected categories in their images. Treat a promising route or place to inspect as a hypothesis until evidence confirms it.

Task-state updates:
- When requirements remain unmet or unverified, continue a suitable objective or add or revise one supported by the instruction and observations. Rewriting or abandoning an objective never removes an original requirement.
- Complete an objective when evidence supports all its requirements. An inspection can finish with a negative finding: no chair in this room resolves that inspection, while finding a chair still requires searching elsewhere. An empty agenda alone does not establish task completion.
- Predicates start empty and record individual evidence-checkable claims; they may cover only part of an objective. Mark a supported claim `confirmed` and an unresolved claim `unconfirmed`. Revise or remove claims when the evidence or their relevance changes.
""".strip()
    evidence_retrieval_policy = (
        "Evidence and retrieval policy:\n"
        "- Assess the task from the observations, execution history, and conclusions already supplied.\n"
        + (
            "- If a remaining question affects task interpretation, progress, or the next action and historical fields can answer it, request those fields together. Request another round only for a remaining decision-relevant question.\n"
            "- Reuse recorded retrieval conclusions. Request their original images again only to check a missing detail or resolve a contradiction.\n"
            "- Finish retrieving when the evidence supports an assessment and next objective, or when no available historical field can resolve the remaining question.\n"
            if allow_retrieve else
            "- Retrieval is unavailable for this call. Assess the supplied evidence and state any remaining uncertainty.\n"
        )
        + "- If a question requires movement or a fresh observation, explain what needs to be observed next. Verify a stopping requirement at the physical endpoint after movement."
    )
    terminal_check_section = ""
    if terminal_check_required:
        stop_objective = str(terminal_check_context.get("stop_objective", "")).strip()
        approach_movement = str(
            terminal_check_context.get("movement", "move")
        ).strip()
        approach_result = (
            "The `approach_to_stop` decision retained the current pose."
            if approach_movement == "stay"
            else "The latest `approach_to_stop` movement reached the current pose."
        )
        terminal_check_section = f"""

Post-approach terminal check:
{approach_result} Check whether the task is complete at this physical endpoint.
- Choose `done` only when evidence supports the original goal, required route events, and final stopping relation. Resolve the remaining agenda by completing supported objectives and abandoning obsolete helpers under the task-state update rules.
- Otherwise choose `continue` and identify unmet or unverified original requirements. Distinguish evidence of a violation from a lack of evidence, and add a supported next objective when one is clear.
- Use the instructed stopping relation, without adding exact pose matching, centering, orientation, or proximity requirements. Different node IDs can represent positions satisfying the same stopping relation.


Proposed approach objective (the preceding action's plan, not an extra task requirement):
{stop_objective}
""".rstrip()
    system_prompt = """You are the Task Executive in NavProbe.
Assess progress against the original instruction and revise the agenda and verification predicates when needed.
Your assessment guides the next skill selection; after approach_to_stop, it also determines whether the task is complete."""
    task_state_section = task_state_content(
        task_state=task_state,
        visual_context=visual_context,
        no_progress=no_progress,
        original_instruction=original_instruction,
        task_constraints=task_constraints,
    )
    response_example: dict[str, object] = {}
    if require_retrieval_conclusion:
        response_example["retrieval_conclusion"] = (
            "<direct answer to the latest retrieval query, supporting refs, and unresolved evidence>"
        )
    response_example["task_state_assessment"] = (
        "<supported progress, required updates, and any decision-relevant gap or needed action/observation>"
    )
    response_example["tool_calls"] = []
    conclusion_protocol = (
        "- Return a JSON object with exactly these fields, in this order: `retrieval_conclusion`, `task_state_assessment`, `tool_calls`.\n"
        "- `retrieval_conclusion`: answer the latest retrieval query using the returned evidence, cite supporting refs, and state what remains unknown. Complete this field even when the remaining retrieval budget is zero.\n"
        if require_retrieval_conclusion
        else "- Return a JSON object with exactly these fields, in this order: `task_state_assessment`, `tool_calls`.\n"
    )
    response_examples = "Response example for this call (no changes or retrieval needed):\n" + json.dumps(response_example)
    if terminal_check_required:
        terminal_example = deepcopy(response_example)
        terminal_example["tool_calls"] = [{
            "name": "update_task_state",
            "arguments": {
                "agenda_updates": [],
                "terminal_check": {
                    "decision": "continue",
                    "missing_constraints": ["<remaining endpoint requirement>"],
                },
            },
        }]
        response_examples = (
            "Example: terminal assessment without retrieval.\n"
            + json.dumps(terminal_example)
        )
        if allow_retrieve:
            retrieval_example = deepcopy(response_example)
            retrieval_example["tool_calls"] = [
                {"name": "update_task_state", "arguments": {"agenda_updates": []}},
                {"name": "retrieve", "arguments": {
                    "query": "<decision-relevant question answerable from historical records>",
                    "items": [{"ref": "<retrievable ref>", "fields": ["<available field>"]}],
                }},
            ]
            response_examples = (
                "Example: retrieve evidence before the terminal assessment.\n"
                + json.dumps(retrieval_example) + "\n\n" + response_examples
            )
    executive_tool_order = ["`update_predicates`"]
    if allow_update_task_state:
        executive_tool_order.append("`update_task_state`")
    if allow_retrieve:
        executive_tool_order.append("`retrieve`")
    response_protocol = (
        "Tool definitions:\n\n"
        + "\n\n".join(tool_schemas)
        + "\n\nResponse protocol:\n"
        + conclusion_protocol
        + "- `task_state_assessment` must be non-empty, including when `tool_calls` is empty. Summarize task progress, reasons for agenda and predicate updates, remaining questions, and the next objective or needed observation.\n"
        + f"- `tool_calls` contains zero to {len(executive_tool_order)} calls. Use each available tool at most once.\n"
        + "- Order calls as " + ", ".join(executive_tool_order) + ". Omit calls that are not needed.\n"
        + "- Return only changed agenda items and predicates. The system validates and commits these updates atomically before executing any retrieval; base them only on evidence already supplied.\n"
        + "- When no update, retrieval, or terminal check is needed, return `tool_calls: []`.\n"
        + "\n"
        + response_examples
    )
    executive_rules = "\n\n".join(
        section for section in (
            task_state_semantics, retrieval_semantics, evidence_retrieval_policy,
        ) if section
    )
    decision_text = '\n\n'.join(section for section in (conclusion_rules, response_protocol) if section)
    if no_progress:
        from navprobe.agent.no_progress_policy import no_progress_prompt
        (system_prompt, decision_text) = no_progress_prompt(
            navigation=False,
            tool_schemas=tool_schemas,
            allow_retrieve=allow_retrieve,
            require_retrieval_conclusion=require_retrieval_conclusion,
            planning_reference_panorama=planning_reference_panorama,
        )
        decision_text = retrieval_semantics + '\n\n' + decision_text
        terminal_check_section = (
            "Post-approach context:\n" + json.dumps(terminal_check_context)
            if terminal_check_required else ""
        )
    content: list[dict[str, object]] = []
    if not no_progress:
        content.append({"type": "text", "text": executive_rules})
    content.append({"type": "text", "text": task_state_section})
    if task_state_context_section != "":
        content.append({"type": "text", "text": task_state_context_section.strip()})
    if terminal_check_section != '':
        content.append({"type": "text", "text": terminal_check_section.strip()})
    if backtrack_context_section != '':
        content.append({"type": "text", "text": backtrack_context_section})
    observation_reference_lines: list[str] = []
    if planning_reference_panorama:
        observation_reference_lines.append(
            "The attached panorama is stored planning-reference evidence, not a fresh observation from the robot's physical pose."
        )
        if dynamic_agenda:
            observation_reference_lines.append(
                "Backtrack changed the planning reference without moving the robot. "
                "Use the stored panorama to reassess the route; the physical and planning node IDs are given in the Backtrack context. "
                "The reference change is not evidence of arrival or of an executed route event. "
                "A physical revisit needs support from the original route, actual path constraints, or a needed fresh observation; "
                "a helper objective or reference change alone does not require returning to the anchor. "
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
    if terminal_check_required:
        raw_movement_obs_ids = terminal_check_context.get("rgb_history_obs_ids", [])
        movement_obs_ids = (
            [str(obs_id) for obs_id in raw_movement_obs_ids]
            if isinstance(raw_movement_obs_ids, (list, tuple))
            else []
        )
        movement_history_content = image_content_for_movement_history_sheet(
            cache=cache,
            obs_ids=movement_obs_ids,
        )
        if movement_history_content is not None:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Post-approach movement RGB. Frames are ordered "
                        "chronologically from left to right and then top to bottom."
                    ),
                }
            )
            content.append(movement_history_content)
    if working_memory_text:
        content.append({"type": "text", "text": working_memory_text})
    content.extend(deepcopy(retrieval_workspace_content))
    if retrieval_catalog_section != "":
        content.append(
            {"type": "text", "text": retrieval_catalog_section.strip()}
        )
    content.append({"type": "text", "text": decision_text})

    def normalize_response(parsed):
        if no_progress:
            from navprobe.agent.no_progress_policy import normalize_no_progress_step
            return normalize_no_progress_step(
                parsed,
                retrieve_fields_by_ref=retrieve_fields_by_ref,
                retrieve_provided_fields_by_ref=retrieve_provided_fields_by_ref,
                allow_retrieve=allow_retrieve,
                require_retrieval_conclusion=require_retrieval_conclusion,
                planning_reference_panorama=planning_reference_panorama,
            )
        decision = normalize_task_state_decision(
            parsed,
            retrieve_fields_by_ref=retrieve_fields_by_ref,
            retrieve_provided_fields_by_ref=retrieve_provided_fields_by_ref,
            allow_retrieve=allow_retrieve,
            allow_update_task_state=allow_update_task_state,
            require_retrieval_conclusion=require_retrieval_conclusion,
            require_terminal_check=terminal_check_required,
        )
        if dynamic_agenda:
            projected_task_state = deepcopy(task_state)
            projected_task_state.apply_task_state_updates(
                list(decision.agenda_updates),
                list(decision.predicate_updates),
                current_node_id='' if planning_reference_panorama else visual_context.current_node_id,
            )
            if isinstance(decision, NavProbeTaskStateDecision) and decision.terminal_check_decision == 'done' and projected_task_state.agenda:
                raise ValueError('terminal_check done requires resolving or abandoning remaining agenda objectives')
        return decision
    return request_task_response(
        request=client.assess_task_state,
        system_prompt=system_prompt,
        content=content,
        normalize_response=normalize_response,
        retrieval_round=int(retrieve_completed_rounds) if require_retrieval_conclusion else None,
        completed_retrieval_rounds=int(retrieve_completed_rounds),
    )


def _normalize_predicate_updates(
    raw_updates: object,
) -> list[dict[str, object]]:
    if not isinstance(raw_updates, list):
        raise ValueError("predicate_updates must be a list")
    normalized: list[dict[str, object]] = []
    for raw_update in raw_updates:
        if not isinstance(raw_update, dict):
            raise ValueError(f"predicate update must be an object: {raw_update!r}")
        op = str(raw_update.get("op", "")).strip().lower()
        raw_id_field = "predicate_id"
        parent_field = "subgoal_id"
        expected_fields = {
            "add": {"op", parent_field, "content", "status"},
            "update": {"op", parent_field, raw_id_field, "status"},
            "rewrite": {
                "op",
                parent_field,
                raw_id_field,
                "content",
                "status",
            },
            "remove": {"op", parent_field, raw_id_field},
        }.get(op)
        if expected_fields is None or set(raw_update) != expected_fields:
            raise ValueError(f"invalid predicate operation; expected fields {expected_fields!r}: {raw_update!r}")
        parent_ref = raw_update.get(parent_field)
        if not isinstance(parent_ref, str) or not parent_ref.strip():
            raise ValueError("predicate requires a non-empty subgoal_id")
        item: dict[str, object] = {
            "op": op,
            parent_field: parent_ref.strip(),
        }
        if raw_id_field in expected_fields:
            predicate_id = str(raw_update.get(raw_id_field, "")).strip()
            if predicate_id == "":
                raise ValueError(f"{op} predicate update requires predicate_id")
            item["predicate_id"] = predicate_id
        if "content" in expected_fields:
            content = raw_update["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"{op} predicate update requires content")
            item["content"] = content.strip()
        if "status" in expected_fields:
            status = str(raw_update.get("status", "")).strip().lower()
            if status not in {"unconfirmed", "confirmed"}:
                raise ValueError(f"invalid predicate status: {raw_update!r}")
            item["status"] = status
        normalized.append(item)
    return normalized


def _normalize_agenda_updates(
    raw_updates: object,
) -> list[dict[str, object]]:
    if not isinstance(raw_updates, list):
        raise ValueError("update_task_state agenda_updates must be a list")
    normalized: list[dict[str, object]] = []
    for raw_update in raw_updates:
        if not isinstance(raw_update, dict):
            raise ValueError(f"agenda update must be an object: {raw_update!r}")
        op = str(raw_update.get("op", "")).strip().lower()
        expected_fields = {
            "add": {"op", "content", "position"},
            "rewrite": {"op", "subgoal_id", "content"},
            "reorder": {"op", "subgoal_ids"},
            "complete": {"op", "subgoal_id", "result"},
            "abandon": {"op", "subgoal_id", "result"},
            "reopen": {"op", "subgoal_id", "position"},
        }.get(op)
        if expected_fields is None or set(raw_update) != expected_fields:
            raise ValueError(f"invalid agenda operation fields: {raw_update!r}")
        item = {"op": op}
        for name in expected_fields - {"op"}:
            value = raw_update[name]
            if name == "position":
                if type(value) is not int or value < 0:
                    raise ValueError("agenda position must be a nonnegative integer")
            elif name == "subgoal_ids":
                if not isinstance(value, list) or any(not isinstance(ref, str) or not ref.strip() for ref in value):
                    raise ValueError("reorder requires a list of subgoal IDs")
                value = [ref.strip() for ref in value]
                if len(value) != len(set(value)):
                    raise ValueError("reorder cannot duplicate a subgoal ID")
            else:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{op} requires non-empty {name}")
                value = value.strip()
            item[name] = value
        normalized.append(item)
    return normalized


def normalize_task_state_decision(
    payload: dict[str, object],
    *,
    retrieve_fields_by_ref: dict[str, list[str]],
    allow_retrieve: bool,
    allow_update_task_state: bool,
    require_retrieval_conclusion: bool,
    retrieve_provided_fields_by_ref: dict[str, list[str]] | None = None,
    require_terminal_check: bool = False,
) -> NavProbeTaskStateDecision | RetrieveRequest:
    if not isinstance(payload, dict):
        raise ValueError("NavProbe response must be an object")
    module_name = 'Task Executive'
    module_tools = _NAVPROBE_TASK_STATE_TOOLS
    raw_analysis = payload.get('task_state_assessment')
    if not isinstance(raw_analysis, str) or raw_analysis.strip() == '':
        raise ValueError('Task Executive requires non-empty task_state_assessment')
    task_state_assessment = raw_analysis.strip()
    payload = _parse_executive_tool_calls(payload, require_retrieval_conclusion=require_retrieval_conclusion)
    selected_tools = [name for name in module_tools if name in payload]
    selected_tool = 'retrieve' if 'retrieve' in payload else 'update_task_state'
    expected_top_fields = set(selected_tools)
    expected_top_fields.add('task_state_assessment')
    if require_retrieval_conclusion:
        expected_top_fields.add("retrieval_conclusion")
    expected_top_fields.add('predicate_updates')
    if set(payload) != expected_top_fields:
        raise ValueError(
            f"{module_name} returned unexpected top-level fields: "
            f"{payload!r}"
        )
    retrieval_conclusion = payload.get("retrieval_conclusion", "")
    raw_tool = payload.get('update_task_state', {'agenda_updates': []})
    if not isinstance(raw_tool, dict):
        raise ValueError(f"{selected_tool} tool payload must be an object: {payload!r}")
    if not allow_update_task_state and ('update_task_state' in payload or selected_tool != 'retrieve'):
        raise ValueError(f'update_task_state is not available in this state: {payload!r}')
    expected_fields = {'agenda_updates'}
    if require_terminal_check and selected_tool != 'retrieve':
        expected_fields.add('terminal_check')
    if require_terminal_check:
        if selected_tool == 'retrieve' and 'terminal_check' in raw_tool:
            raise ValueError('Omit terminal_check when calling retrieve; assess the endpoint after the retrieved evidence returns.')
        if selected_tool != 'retrieve' and 'terminal_check' not in raw_tool:
            raise ValueError((
            'Missing required field: update_task_state.arguments.terminal_check. Without retrieve, call '
            'update_task_state with terminal_check even when agenda_updates is empty.'
        ))
    if set(raw_tool) != expected_fields:
        raise ValueError(f'update_task_state fields must be exactly {sorted(expected_fields)!r}: {payload!r}')
    try:
        agenda_updates = _normalize_agenda_updates(raw_tool.get('agenda_updates'))
        predicate_updates = _normalize_predicate_updates(payload.get('predicate_updates'))
    except ValueError as exc:
        raise ValueError(f'{exc}: {payload!r}') from exc
    if selected_tool == 'retrieve':
        if not allow_retrieve:
            raise ValueError(f'retrieve is not available in this state: {payload!r}')
        raw_retrieve = payload['retrieve']
        if not isinstance(raw_retrieve, dict) or set(raw_retrieve) != {'query', 'items'}:
            raise ValueError('retrieve arguments must contain exactly query and items')
        request = normalize_retrieve_request(
            payload,
            fields_by_ref=retrieve_fields_by_ref,
            provided_fields_by_ref=retrieve_provided_fields_by_ref,
            require_retrieval_conclusion=require_retrieval_conclusion,
        )
        return RetrieveRequest(
            query=request.query,
            items=request.items,
            already_provided_items=request.already_provided_items,
            retrieval_conclusion=request.retrieval_conclusion,
            predicate_updates=tuple(predicate_updates),
            agenda_updates=tuple(agenda_updates),
            task_state_assessment=task_state_assessment,
        )
    terminal_check_decision = ''
    terminal_check_reasoning = ''
    terminal_check_missing_constraints: list[str] = []
    if require_terminal_check:
        raw_terminal_check = raw_tool.get('terminal_check')
        if not isinstance(raw_terminal_check, dict):
            raise ValueError(f'update_task_state terminal_check must be an object: {payload!r}')
        expected_terminal_fields = {'decision', 'missing_constraints'}
        if set(raw_terminal_check) != expected_terminal_fields:
            raise ValueError(f'terminal_check fields must be exactly {sorted(expected_terminal_fields)!r}: {payload!r}')
        terminal_check_decision = str(raw_terminal_check.get('decision', '')).strip()
        terminal_check_reasoning = task_state_assessment
        raw_missing_constraints = raw_terminal_check.get('missing_constraints')
        if terminal_check_decision not in {'done', 'continue'}:
            raise ValueError(f'terminal_check decision must be done or continue: {payload!r}')
        if not isinstance(raw_missing_constraints, list) or any((not isinstance(item, str) or not item.strip() for item in raw_missing_constraints)):
            raise ValueError(f'terminal_check missing_constraints must be a list: {payload!r}')
        terminal_check_missing_constraints = [str(item).strip() for item in raw_missing_constraints if str(item).strip() != '']
        if terminal_check_decision == 'done' and terminal_check_missing_constraints:
            raise ValueError(f'done terminal_check cannot have missing constraints: {payload!r}')
        if terminal_check_decision == 'continue' and (not terminal_check_missing_constraints):
            raise ValueError(f'continue terminal_check requires missing constraints: {payload!r}')
    return NavProbeTaskStateDecision(
        task_state_assessment=task_state_assessment,
        retrieval_conclusion=retrieval_conclusion,
        agenda_updates=agenda_updates,
        predicate_updates=predicate_updates,
        terminal_check_decision=terminal_check_decision,
        terminal_check_reasoning=terminal_check_reasoning,
        terminal_check_missing_constraints=terminal_check_missing_constraints,
        update_task_state_called='update_task_state' in payload,
    )


def _parse_executive_tool_calls(
    payload: dict[str, object],
    *,
    require_retrieval_conclusion: bool,
) -> dict[str, object]:
    expected_top_fields = {"task_state_assessment", "tool_calls"}
    if require_retrieval_conclusion:
        expected_top_fields.add("retrieval_conclusion")
    missing_fields = expected_top_fields - set(payload)
    unexpected_fields = set(payload) - expected_top_fields
    if missing_fields or unexpected_fields:
        details = []
        if missing_fields:
            details.append(f"Missing required top-level fields: {sorted(missing_fields)!r}.")
        if "retrieval_conclusion" in missing_fields:
            details.append("Answer the latest retrieval query in retrieval_conclusion before issuing further tools.")
        if unexpected_fields:
            details.append(f"Unexpected top-level fields: {sorted(unexpected_fields)!r}.")
        if "retrieval_conclusion" in unexpected_fields:
            details.append("Omit retrieval_conclusion because no retrieval round awaits a conclusion.")
        raise ValueError("Task Executive: " + " ".join(details))
    if require_retrieval_conclusion:
        conclusion = payload["retrieval_conclusion"]
        if not isinstance(conclusion, str) or not conclusion.strip():
            raise ValueError(
                "Task Executive must conclude the latest retrieval "
                f"before its next tool: {payload!r}"
            )
    raw_calls = payload.get("tool_calls")
    if not isinstance(raw_calls, list) or len(raw_calls) > 3:
        raise ValueError("tool_calls must contain zero to three calls")
    call_order = {"update_predicates": 0, "update_task_state": 1, "retrieve": 2}
    previous_position = -1
    normalized: dict[str, object] = {
        "task_state_assessment": payload["task_state_assessment"],
        "predicate_updates": [],
    }
    if require_retrieval_conclusion:
        normalized["retrieval_conclusion"] = conclusion.strip()
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict) or set(raw_call) != {"name", "arguments"}:
            raise ValueError(f"invalid TPU tool call: {raw_call!r}")
        name = str(raw_call.get("name", "")).strip()
        arguments = raw_call.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError(f"TPU tool arguments must be an object: {raw_call!r}")
        if name not in call_order or call_order[name] <= previous_position:
            raise ValueError(
                "Task Executive tools may occur at most once, ordered as "
                "update_predicates, update_task_state, retrieve"
            )
        previous_position = call_order[name]
        if name == "update_predicates":
            if set(arguments) != {"predicate_updates"}:
                raise ValueError(
                    "update_predicates arguments must contain exactly predicate_updates"
                )
            predicate_updates = arguments["predicate_updates"]
            if not isinstance(predicate_updates, list) or not predicate_updates:
                raise ValueError("update_predicates requires non-empty predicate_updates")
            normalized["predicate_updates"] = predicate_updates
        else:
            normalized[name] = arguments
    return normalized

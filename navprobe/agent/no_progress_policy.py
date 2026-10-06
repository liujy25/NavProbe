"""Task Executive protocol for the no-persistent-task-progress comparison."""
from __future__ import annotations

from dataclasses import replace
import json

from navprobe.agent.episodic_retrieval import normalize_retrieve_request
from navprobe.agent.navigation_decisions import NavProbeTaskStateDecision


def no_progress_prompt(
    *, navigation: bool, tool_schemas: list[str], allow_retrieve: bool = True,
    require_retrieval_conclusion: bool = False, planning_reference_panorama: bool = False,
) -> tuple[str, str]:
    if navigation:
        contracts = "\n\n".join(tool_schemas)
        return (
            "You are the Skill Router in NavProbe. Turn the current Executive assessment "
            "into one local skill and a self-contained intent for grounding and execution.",
            "Input meanings:\n"
            "The original instruction defines the task and route order. The current Executive "
            "assessment and retrieval conclusions explain progress and remaining uncertainty; "
            "the supplied images show the routes available from the planning reference. "
            "No persistent agenda is supplied.\n\n"
            "Choose the next skill:\n"
            "Compare visible routes and select an action supported by the assessment and evidence. "
            "Preserve instructed turns, passages, and stopping boundaries without inventing unseen contents. "
            "Express the intended route and its supporting evidence in reason, using the supplied "
            "reference frame. Do not add pose matching, centering, or facing requirements beyond the instruction.\n\n"
            "Output contract:\nReturn only JSON matching exactly one of the action contracts below.\n\n" + contracts,
        )
    retrieval_tools = "\n\n".join(
        schema for schema in tool_schemas if schema.startswith("### `retrieve`")
    )
    example = {
        "task_state_assessment": "<evidence-backed route and endpoint assessment>",
        "tool_calls": [],
        "terminal_check": {"decision": "continue", "missing_constraints": ["<remaining requirement>"]},
    }
    if require_retrieval_conclusion:
        example = {"retrieval_conclusion": "<answer to the latest query, supporting refs, and remaining unknowns>", **example}
    retrieval_rule = (
        "If an unresolved route or endpoint question can be answered by available historical fields, "
        "request those fields together. Finish the assessment when evidence is sufficient or no "
        "available field can resolve the question."
        if allow_retrieve else
        "Retrieval is unavailable for this call. Assess the supplied evidence and identify what remains unknown."
    )
    conclusion_rule = (
        "- Include a non-empty retrieval_conclusion answering the latest query with supporting refs "
        "and remaining unknowns, before task_state_assessment. Interpret the returned evidence even "
        "when no retrieval budget remains.\n"
        if require_retrieval_conclusion else ""
    )
    tool_rule = (
        "- tool_calls contains one retrieve call or is empty. A retrieve response omits terminal_check; "
        "the system returns its evidence for the next assessment.\n"
        if allow_retrieve else "- tool_calls must be empty for this call.\n"
    )
    reference_rule = (
        " The stored planning-reference panorama does not establish the physical endpoint; "
        "completion cannot be declared from this historical-reference call."
        if planning_reference_panorama else ""
    )
    return (
        "You are the Task Executive in NavProbe, operating without persistent task-progress memory. "
        "Assess the instructed route and completion; your current assessment guides the Skill Router.",
        "Input meanings:\n"
        "The original instruction defines the actions, route order, and stopping relation. "
        "Current observations, executed history, entity knowledge, and retrieval conclusions "
        "provide evidence; no persistent agenda is supplied or updated.\n\n"
        "Assess the task:\n"
        "Check progress against the entire instruction. Seeing the target alone does not establish "
        "arrival or completion of the route. " + retrieval_rule + " If the remaining question needs "
        "movement or a fresh observation, identify that need in the assessment.\n"
        "Choose done only when evidence supports the entire required route and the final stopping "
        "relation at the physical robot pose. Otherwise choose continue and identify unmet or "
        "unconfirmed requirements for the next skill selection." + reference_rule + "\n\n"
        + ("Tool definition:\n" + retrieval_tools + "\n\n" if retrieval_tools else "")
        + "Output contract:\n"
        "- Return only JSON with non-empty task_state_assessment and tool_calls. "
        "Explain supported progress, uncertainty, and the next objective or needed observation in task_state_assessment.\n"
        + conclusion_rule + tool_rule
        + "- When tool_calls is empty, include top-level terminal_check with exactly decision and missing_constraints. "
        "Use decision=done with missing_constraints=[], or decision=continue with a non-empty list of unmet or unconfirmed requirements.\n"
        "- Include no other fields.\nExample without retrieval:\n" + json.dumps(example),
    )


def normalize_no_progress_step(
    payload, *, retrieve_fields_by_ref, retrieve_provided_fields_by_ref,
    allow_retrieve, require_retrieval_conclusion, planning_reference_panorama=False,
):
    fields = {"task_state_assessment", "tool_calls"}
    if require_retrieval_conclusion:
        fields.add("retrieval_conclusion")
    analysis = payload.get("task_state_assessment")
    if not isinstance(analysis, str) or not analysis.strip():
        raise ValueError("Task Executive requires non-empty task_state_assessment")
    conclusion = payload.get("retrieval_conclusion", "")
    if require_retrieval_conclusion and (not isinstance(conclusion, str) or not conclusion.strip()):
        raise ValueError("Task Executive must conclude the latest retrieval")
    calls = payload.get("tool_calls")
    if not isinstance(calls, list) or len(calls) > 1:
        raise ValueError("no_progress tool_calls must be empty or contain one retrieve")
    if calls:
        call = calls[0]
        if (not allow_retrieve or not isinstance(call, dict)
                or set(call) != {"name", "arguments"} or call["name"] != "retrieve"):
            raise ValueError("only retrieve is available in no_progress")
        if set(payload) != fields:
            raise ValueError("retrieve response has unexpected fields")
        arguments = call["arguments"]
        if not isinstance(arguments, dict) or set(arguments) != {"query", "items"}:
            raise ValueError("retrieve requires exactly query and items")
        request = normalize_retrieve_request(
            {"retrieve": arguments, "retrieval_conclusion": conclusion},
            fields_by_ref=retrieve_fields_by_ref,
            provided_fields_by_ref=retrieve_provided_fields_by_ref,
            require_retrieval_conclusion=require_retrieval_conclusion,
        )
        return replace(request, task_state_assessment=analysis.strip())
    if set(payload) != fields | {"terminal_check"}:
        raise ValueError("finished Task Executive response requires top-level terminal_check")
    terminal = payload["terminal_check"]
    if not isinstance(terminal, dict) or set(terminal) != {"decision", "missing_constraints"}:
        raise ValueError("terminal_check requires decision and missing_constraints")
    decision, missing = terminal["decision"], terminal["missing_constraints"]
    if decision not in {"done", "continue"} or not isinstance(missing, list):
        raise ValueError("invalid terminal_check")
    if any(not isinstance(item, str) or not item.strip() for item in missing):
        raise ValueError("missing_constraints must contain non-empty strings")
    if (decision == "done" and (missing or planning_reference_panorama)) or (decision == "continue" and not missing):
        raise ValueError("terminal decision conflicts with endpoint evidence or missing constraints")
    return NavProbeTaskStateDecision(
        task_state_assessment=analysis.strip(),
        retrieval_conclusion=conclusion.strip(), update_task_state_called=False,
        terminal_check_decision=decision, terminal_check_reasoning=analysis.strip(),
        terminal_check_missing_constraints=missing, standalone_terminal_check=True,
    )

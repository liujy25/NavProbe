from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING

from navprobe.memory.entity_knowledge import EntityKnowledge
from navprobe.memory.entity_knowledge import next_entity_knowledge_id

if TYPE_CHECKING:
    from navprobe.agent.episodic_retrieval import RetrievalWorkspace
    from navprobe.agent.state import NavProbeAgentState
    from navprobe.llm.client import LLMClient


@dataclass(frozen=True)
class EntityKnowledgeUpdate:
    op: str
    ref: str
    knowledge_id: str = ""
    content: str = ""

    def to_dict(self) -> dict[str, str]:
        payload = {"op": str(self.op), "ref": str(self.ref)}
        if self.knowledge_id != "":
            payload["knowledge_id"] = str(self.knowledge_id)
        if self.content != "":
            payload["content"] = str(self.content)
        return payload


def manage_retrieved_knowledge(
    *,
    client: "LLMClient",
    state: "NavProbeAgentState",
    workspace: "RetrievalWorkspace",
    current_node_id: str,
    task_state_updates: list[dict[str, object]],
) -> dict[str, object]:
    refs = _retrieved_refs(workspace)
    if refs == []:
        raise ValueError("knowledge management requires retrieved entities")
    existing = {
        ref: [item.to_dict() for item in _entity_knowledge(state, ref)]
        for ref in refs
    }
    retrieval_rounds = [
        {
            "round": int(round_record.round_index) + 1,
            "query": str(round_record.request.query),
            "request": [item.to_dict() for item in round_record.request.items],
            "conclusion": str(round_record.conclusion),
            **({"unavailable_fields": dict(round_record.unavailable_fields)}
               if round_record.unavailable_fields else {}),
        }
        for round_record in workspace.rounds
    ]
    system_prompt = """
You are the Entity Knowledge Manager (EKM) in NavProbe.
Preserve reusable facts from this decision's retrieval conclusions so later navigation decisions can recall knowledge about each entity.
""".strip()
    user_prompt = f"""
Input meanings:
- Each retrieval round identifies a question, the requested entity fields, and the Executive's conclusion. Raw images are not supplied here; use only details stated in those conclusions.
- Existing knowledge contains facts already stored on each node, edge, or landmark. Committed task-state updates explain why a finding mattered to the task, but are not independent evidence about the entity.
- A field in unavailable_fields supplied no evidence. Requesting it does not mean it was seen; an overlay visibility limitation does not establish a fact about the scene.

Update rules:
- Store a fact only on a listed retrieved entity whose requested evidence supports that fact in the conclusion. Keep it concise, specific to that entity, and useful beyond the current decision.
- Add a distinct fact; refine or correct an existing claim with update. Avoid duplicate or paraphrased copies.
- Remove a fact only when a retrieval conclusion establishes that it is invalid. Lack of new support is not a reason to remove it.
- Keep temporary queries, uncertainty, task predicates, action choices, waypoint candidates, and navigation advice out of entity knowledge.

Retrieved entity refs:
{json.dumps(refs, ensure_ascii=False)}

Retrieval trace:
{json.dumps(retrieval_rounds, ensure_ascii=False, indent=2)}

Committed task-state updates:
{json.dumps(task_state_updates, ensure_ascii=False, indent=2)}

Existing entity knowledge:
{json.dumps(existing, ensure_ascii=False, indent=2)}

Output contract:
Return only JSON. Use only the listed retrieved refs, and the entity's existing knowledge_id for update or remove. The entries below show the three operation formats; include only the changes needed, or an empty list when nothing changes.
{{
  "entity_knowledge_updates": [
    {{"op":"add","ref":"<retrieved ref>","content":"<compact entity-local fact>"}},
    {{"op":"update","ref":"<retrieved ref>","knowledge_id":"<existing id>","content":"<corrected or refined fact>"}},
    {{"op":"remove","ref":"<retrieved ref>","knowledge_id":"<existing id>"}}
  ]
}}
""".strip()
    parsed = client.manage_knowledge(system_prompt, user_prompt)
    try:
        updates = normalize_entity_knowledge_updates(parsed, allowed_refs=set(refs))
        result = apply_entity_knowledge_updates(
            state=state,
            updates=updates,
            current_node_id=current_node_id,
        )
    except ValueError as error:
        # The session abort trace records this error without another request.
        raise ValueError(
            f"entity knowledge proposal rejected: {error}; proposal={parsed!r}"
        ) from error
    return {
        "retrieved_refs": refs,
        "proposed_updates": [update.to_dict() for update in updates],
        **result,
        "knowledge_after": {
            ref: [item.to_dict() for item in _entity_knowledge(state, ref)]
            for ref in refs
        },
    }


def normalize_entity_knowledge_updates(
    payload: object,
    *,
    allowed_refs: set[str],
) -> list[EntityKnowledgeUpdate]:
    if not isinstance(payload, dict) or set(payload) != {"entity_knowledge_updates"}:
        raise ValueError(f"invalid entity knowledge output: {payload!r}")
    raw_updates = payload.get("entity_knowledge_updates")
    if not isinstance(raw_updates, list):
        raise ValueError("entity_knowledge_updates must be a list")
    updates: list[EntityKnowledgeUpdate] = []
    for raw_update in raw_updates:
        if not isinstance(raw_update, dict):
            raise ValueError(f"entity knowledge update must be an object: {raw_update!r}")
        op = str(raw_update.get("op", "")).strip()
        ref = str(raw_update.get("ref", "")).strip()
        if ref not in allowed_refs:
            raise ValueError(f"entity knowledge update uses an unretrieved ref: {ref!r}")
        expected_fields = {
            "add": {"op", "ref", "content"},
            "update": {"op", "ref", "knowledge_id", "content"},
            "remove": {"op", "ref", "knowledge_id"},
        }.get(op)
        if expected_fields is None or set(raw_update) != expected_fields:
            raise ValueError(f"invalid entity knowledge update fields: {raw_update!r}")
        knowledge_id = str(raw_update.get("knowledge_id", "")).strip()
        content = ""
        if op in {"update", "remove"} and knowledge_id == "":
            raise ValueError(f"{op} requires a knowledge_id")
        if op in {"add", "update"}:
            content = raw_update["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"{op} requires non-empty content")
            content = content.strip()
        updates.append(
            EntityKnowledgeUpdate(
                op=op,
                ref=ref,
                knowledge_id=knowledge_id,
                content=content,
            )
        )
    return updates


def apply_entity_knowledge_updates(
    *,
    state: "NavProbeAgentState",
    updates: list[EntityKnowledgeUpdate],
    current_node_id: str,
) -> dict[str, object]:
    node_id = str(current_node_id).strip()
    if node_id == "" or not state.graph.has_node(node_id):
        raise ValueError(f"knowledge update node must exist in graph: {node_id!r}")
    applied: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    staged: dict[int, tuple[list[EntityKnowledge], list[EntityKnowledge]]] = {}
    for update in updates:
        if update.op not in {"add", "update", "remove"}:
            raise ValueError(f"invalid entity knowledge operation: {update.op!r}")
        original = _entity_knowledge(state, update.ref)
        # Merged landmark refs can resolve to the same list. They must observe
        # preceding operations in this batch, including ID reuse and duplicates.
        key = id(original)
        if key not in staged:
            staged[key] = (original, list(original))
        knowledge = staged[key][1]
        if update.op == "add":
            if any(item.content.casefold() == update.content.casefold() for item in knowledge):
                skipped.append({**update.to_dict(), "reason": "duplicate_content"})
                continue
            item = EntityKnowledge(
                knowledge_id=next_entity_knowledge_id(knowledge),
                update_node_id=node_id,
                content=update.content,
            )
            knowledge.append(item)
            applied.append({"op": "add", "ref": update.ref, **item.to_dict()})
            continue
        item_index = next(
            (
                index
                for index, item in enumerate(knowledge)
                if str(item.knowledge_id) == str(update.knowledge_id)
            ),
            None,
        )
        if item_index is None:
            raise ValueError(
                f"unknown knowledge_id {update.knowledge_id!r} for {update.ref!r}"
            )
        if update.op == "update":
            # Unchanged records retain their identity; changed records remain
            # private until every operation in the proposal has succeeded.
            knowledge[item_index] = copy(knowledge[item_index])
            knowledge[item_index].content = str(update.content)
            knowledge[item_index].update_node_id = node_id
            applied.append({"op": "update", "ref": update.ref, **knowledge[item_index].to_dict()})
        else:
            removed = knowledge.pop(item_index)
            applied.append({"op": "remove", "ref": update.ref, **removed.to_dict()})
    for original, knowledge in staged.values():
        original[:] = knowledge
    return {"applied_updates": applied, "skipped_updates": skipped}


def _retrieved_refs(workspace: "RetrievalWorkspace") -> list[str]:
    refs: list[str] = []
    for round_record in workspace.rounds:
        for item in round_record.request.items:
            ref = str(item.ref)
            if ref not in refs:
                refs.append(ref)
    return refs


def _entity_knowledge(state: "NavProbeAgentState", ref: str) -> list[EntityKnowledge]:
    if state.graph.has_node(str(ref)):
        return state.graph.get_node(str(ref)).knowledge
    for edge in state.graph.iter_edges():
        if str(edge.id) == str(ref):
            return edge.knowledge
    candidate = state.landmark_controller.landmark_memory.find_record(str(ref))
    if candidate is not None:
        return candidate.knowledge
    raise ValueError(f"unknown knowledge entity ref: {ref!r}")

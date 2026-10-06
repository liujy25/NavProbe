"""Field-selective retrieval from episode graph memory."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
from typing import TYPE_CHECKING

import numpy as np

from navprobe.agent.retrieval_evidence import (
    _EvidencePanel,
    _render_shared_bev,
    _add_node_rgb_panel,
    _add_node_landmark_panel,
    _add_landmark_rgb_panel,
    _add_movement_panel,
    _add_edge_rgb_panel,
    _movement_keyframes,
    _reference_sort_key,
)
from navprobe.agent.visual_policy_prompt_images import image_content_for_array
from navprobe.visualization.action_mode_overlays import BevOverlayTransform

if TYPE_CHECKING:
    from navprobe.agent.state import NavProbeAgentState


@dataclass(frozen=True)
class MemoryIndexEntry:
    ref: str
    kind: str
    floor_id: str
    available_fields: dict[str, int]
    knowledge: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def to_index_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "ref": str(self.ref),
            "floor_id": str(self.floor_id),
            "knowledge": [str(item) for item in self.knowledge],
            "available_fields": {
                str(name): int(count)
                for name, count in self.available_fields.items()
                if int(count) > 0
            },
            "metadata": deepcopy(self.metadata),
        }
        return payload


@dataclass(frozen=True)
class RetrieveItem:
    ref: str
    fields: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"ref": str(self.ref), "fields": [str(name) for name in self.fields]}


@dataclass(frozen=True)
class RetrieveRequest:
    query: str
    items: tuple["RetrieveItem", ...]
    retrieval_conclusion: str = ""
    predicate_updates: tuple[dict[str, object], ...] = ()
    already_provided_items: tuple["RetrieveItem", ...] = ()
    task_state_assessment: str = ""
    agenda_updates: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "query": str(self.query),
            "items": [item.to_dict() for item in self.items],
            "predicate_updates": [
                deepcopy(item) for item in self.predicate_updates
            ],
        }
        if str(self.retrieval_conclusion).strip() != "":
            payload["retrieval_conclusion"] = str(self.retrieval_conclusion)
        if str(self.task_state_assessment).strip() != "":
            payload["task_state_assessment"] = str(self.task_state_assessment)
        if self.agenda_updates:
            payload["agenda_updates"] = [deepcopy(item) for item in self.agenda_updates]
        if self.already_provided_items:
            payload["already_provided_items"] = [
                item.to_dict() for item in self.already_provided_items
            ]
        return payload


@dataclass(frozen=True)
class RetrieveRound:
    round_index: int
    request: RetrieveRequest
    source_obs_ids: tuple[str, ...]
    conclusion: str = ""
    unavailable_fields: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "round_index": int(self.round_index),
            "query": str(self.request.query),
            "items": [item.to_dict() for item in self.request.items],
            "already_provided_items": [
                item.to_dict() for item in self.request.already_provided_items
            ],
            "source_obs_ids": [str(obs_id) for obs_id in self.source_obs_ids],
            "conclusion": str(self.conclusion),
            "predicate_updates": [
                deepcopy(item)
                for item in self.request.predicate_updates
            ],
        }
        if str(self.request.task_state_assessment).strip() != "":
            payload["task_state_assessment"] = str(self.request.task_state_assessment)
        if self.request.agenda_updates:
            payload["agenda_updates"] = [deepcopy(item) for item in self.request.agenda_updates]
        if self.unavailable_fields:
            payload["unavailable_fields"] = dict(self.unavailable_fields)
        return payload


@dataclass(frozen=True)
class MaterializedMemoryContext:
    loaded_fields_by_ref: dict[str, list[str]]
    skipped_fields_by_ref: dict[str, list[str]]
    missing_fields_by_ref: dict[str, list[str]]
    source_obs_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "materialized",
            "loaded_fields_by_ref": deepcopy(self.loaded_fields_by_ref),
            "skipped_fields_by_ref": deepcopy(self.skipped_fields_by_ref),
            "missing_fields_by_ref": deepcopy(self.missing_fields_by_ref),
            "source_obs_ids": [str(obs_id) for obs_id in self.source_obs_ids],
        }


@dataclass
class RetrievalWorkspace:
    entries: list[MemoryIndexEntry]
    max_retrieve_rounds: int
    rounds: list[RetrieveRound] = field(default_factory=list)
    text_evidence: dict[str, str] = field(default_factory=dict)
    image_panels: dict[str, _EvidencePanel] = field(default_factory=dict)
    unavailable_fields: dict[str, str] = field(default_factory=dict)
    selected_node_ids_by_floor: dict[str, set[str]] = field(default_factory=dict)
    selected_edge_ids_by_floor: dict[str, set[str]] = field(default_factory=dict)
    selected_landmark_ids_by_floor: dict[str, set[str]] = field(default_factory=dict)
    node_display_labels_by_floor: dict[str, dict[str, int]] = field(default_factory=dict)
    landmark_display_labels_by_floor: dict[str, dict[str, int]] = field(default_factory=dict)
    shared_bev_by_floor: dict[str, np.ndarray] = field(default_factory=dict)
    shared_bev_transform_by_floor: dict[str, BevOverlayTransform] = field(default_factory=dict)

    @property
    def retrieve_count(self) -> int:
        return len(self.rounds)

    @property
    def can_retrieve(self) -> bool:
        return self.retrieve_count < int(self.max_retrieve_rounds)

    @property
    def has_pending_evidence(self) -> bool:
        return self.rounds != [] and str(self.rounds[-1].conclusion).strip() == ""

    def conclude_latest_retrieval(self, conclusion: str) -> None:
        text = str(conclusion).strip()
        if text == "":
            raise ValueError("latest retrieval requires a non-empty conclusion")
        if not self.has_pending_evidence:
            raise ValueError("no unconcluded retrieval evidence is available")
        self.rounds[-1] = replace(self.rounds[-1], conclusion=text)

    def discard_concluded_raw_evidence(self) -> None:
        if self.rounds != [] and self.has_pending_evidence:
            raise ValueError("cannot discard raw evidence before concluding it")
        self.text_evidence.clear()
        self.image_panels.clear()
        self.unavailable_fields.clear()

    def clear_materialized_evidence(self) -> None:
        self.text_evidence.clear()
        self.image_panels.clear()
        self.unavailable_fields.clear()
        self.selected_node_ids_by_floor.clear()
        self.selected_edge_ids_by_floor.clear()
        self.selected_landmark_ids_by_floor.clear()
        self.node_display_labels_by_floor.clear()
        self.landmark_display_labels_by_floor.clear()
        self.shared_bev_by_floor.clear()
        self.shared_bev_transform_by_floor.clear()

    def fields_by_ref(
        self,
        *,
        provided_fields_by_ref: dict[str, list[str]] | None = None,
    ) -> dict[str, list[str]]:
        provided = provided_fields_by_ref or {}
        return {
            str(entry.ref): [
                str(name)
                for name, count in entry.available_fields.items()
                if int(count) > 0
                and str(name) not in set(provided.get(str(entry.ref), []))
            ]
            for entry in self.entries
        }

    def retrieval_log_text(self) -> str:
        if self.rounds == []:
            return "none"
        lines: list[str] = []
        for round_record in self.rounds:
            retrieved_refs = []
            for item in round_record.request.items:
                fields = [name for name in item.fields
                          if f"{item.ref}.{name}" not in round_record.unavailable_fields]
                if fields:
                    retrieved_refs.append(f"{item.ref}[{', '.join(fields)}]")
            refs = ", ".join(retrieved_refs)
            provided_refs = ", ".join(
                f"{item.ref}[{', '.join(item.fields)}]"
                for item in round_record.request.already_provided_items
            )
            provided_text = (
                f"already provided={provided_refs}; "
                if provided_refs != ""
                else ""
            )
            unavailable_text = (
                f"unavailable_fields={json.dumps(round_record.unavailable_fields, ensure_ascii=False)}; "
                if round_record.unavailable_fields else ""
            )
            pending_text = ("pending; available evidence attached below" if round_record.unavailable_fields
                            else "pending; raw evidence attached below")
            lines.append(
                f"round {round_record.round_index}: "
                f"query={round_record.request.query}; retrieved={refs or 'none'}; "
                f"{provided_text}"
                f"{unavailable_text}"
                f"conclusion={round_record.conclusion or pending_text}"
            )
        return "\n".join(lines)

    def materialized_context_content(self) -> list[dict[str, object]]:
        content = self._bev_context_content()
        content.extend(self._materialized_field_content())
        return content

    def _materialized_field_content(self) -> list[dict[str, object]]:
        content: list[dict[str, object]] = []
        if self.unavailable_fields:
            content.append({
                "type": "text",
                "text": "Requested fields unavailable (no image evidence supplied for these fields):\n"
                + json.dumps(self.unavailable_fields, ensure_ascii=False),
            })
        for key in sorted(self.text_evidence):
            content.append({"type": "text", "text": self.text_evidence[key]})
        for key in sorted(self.image_panels):
            panel = self.image_panels[key]
            content.append({"type": "text", "text": str(panel.label)})
            content.extend(deepcopy(list(panel.content)))
        return content

    def conclusion_context_content(self) -> list[dict[str, object]]:
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": "Retrieval rounds:\n" + self.retrieval_log_text(),
            }
        ]
        content.extend(self._bev_context_content())
        return content

    def _bev_context_content(self) -> list[dict[str, object]]:
        content: list[dict[str, object]] = []
        for floor_id in sorted(self.shared_bev_by_floor):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Retrieved BEV for {floor_id}:\n"
                        f"- nodes: {_inline_labeled_refs(self.selected_node_ids_by_floor.get(floor_id, set()), self.node_display_labels_by_floor.get(floor_id, {}))}\n"
                        f"- edges (highlighted executed trajectories): {_inline_edge_refs(self, floor_id)}\n"
                        f"- landmarks: {_inline_labeled_refs(self.selected_landmark_ids_by_floor.get(floor_id, set()), {})}\n"
                        "- circles are node display numbers; landmark squares show the numeric suffix of each landmark ref."
                    ),
                }
            )
            content.append(image_content_for_array(self.shared_bev_by_floor[floor_id]))
        return content

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": True,
            "max_retrieve_rounds": int(self.max_retrieve_rounds),
            "retrieve_count": int(self.retrieve_count),
            "memory_index": memory_index_dict(self.entries),
            "rounds": [round_record.to_dict() for round_record in self.rounds],
            "workspace": {
                "unavailable_fields": dict(self.unavailable_fields),
                "text_evidence_keys": sorted(self.text_evidence),
                "image_panels": [
                    {
                        "key": panel.key,
                        "label": panel.label,
                        "source_obs_ids": [str(obs_id) for obs_id in panel.source_obs_ids],
                    }
                    for panel in sorted(self.image_panels.values(), key=lambda item: item.key)
                ],
                "shared_bev_floors": sorted(self.shared_bev_by_floor),
                "shared_bev_transform_by_floor": {
                    str(floor_id): transform.to_dict()
                    for floor_id, transform in sorted(
                        self.shared_bev_transform_by_floor.items()
                    )
                },
                "selected_node_ids_by_floor": _sorted_set_mapping(
                    self.selected_node_ids_by_floor
                ),
                "selected_edge_ids_by_floor": _sorted_set_mapping(
                    self.selected_edge_ids_by_floor
                ),
                "selected_landmark_ids_by_floor": _sorted_set_mapping(
                    self.selected_landmark_ids_by_floor
                ),
                "node_display_labels_by_floor": _sorted_label_mapping(
                    self.node_display_labels_by_floor
                ),
                "landmark_display_labels_by_floor": _sorted_label_mapping(
                    self.landmark_display_labels_by_floor
                ),
            },
        }


def build_memory_index(
    state: "NavProbeAgentState",
) -> list[MemoryIndexEntry]:
    candidates = state.landmark_controller.landmark_memory.iter_records()
    landmark_ids_by_node = _landmark_ids_by_node(state=state)
    node_ids_by_landmark = _node_ids_by_landmark(landmark_ids_by_node)
    entries: list[MemoryIndexEntry] = []
    for node in sorted(state.graph.iter_nodes(), key=lambda item: _reference_sort_key(str(item.id))):
        landmark_ids = sorted(
            landmark_ids_by_node.get(str(node.id), set()),
            key=_reference_sort_key,
        )
        entries.append(
            MemoryIndexEntry(
                ref=str(node.id),
                kind="node",
                floor_id=str(node.floor_id),
                available_fields={
                    "rgb": len(node.obs_ids),
                    "landmarks": len(landmark_ids),
                },
                knowledge=tuple(str(item.content) for item in node.knowledge),
                metadata={"associated_landmark_refs": landmark_ids},
            )
        )
    for edge in sorted(state.graph.iter_edges(), key=lambda item: _reference_sort_key(str(item.id))):
        src_node = state.graph.get_node(str(edge.src_id))
        floor_id = str(src_node.floor_id)
        entries.append(
            MemoryIndexEntry(
                ref=str(edge.id),
                kind="edge",
                floor_id=floor_id,
                available_fields={
                    "rgb": int(
                        len(src_node.obs_ids) > 0
                        and len(edge.path_xy) >= 2
                    ),
                    "trajectory": int(len(edge.path_xy) >= 2),
                    "movement_rgb": len(edge.rgb_history_obs_ids),
                },
                knowledge=(
                    f"Actual {str(edge.relation or 'move')} edge from "
                    f"{edge.src_id} to {edge.dst_id}.",
                    *(str(item.content) for item in edge.knowledge),
                ),
                metadata={
                    "src_node_id": str(edge.src_id),
                    "dst_node_id": str(edge.dst_id),
                    "traversal_count": int(edge.traversal_count),
                },
            )
        )
    for candidate in candidates:
        landmark_id = str(candidate.landmark_id)
        associated_nodes = sorted(
            {
                str(node_id)
                for node_id in node_ids_by_landmark.get(landmark_id, set())
            },
            key=_reference_sort_key,
        )
        if not candidate.detections:
            continue
        floor_id = _landmark_floor_id(
            state=state,
            associated_node_ids=associated_nodes,
        )
        entries.append(
            MemoryIndexEntry(
                ref=landmark_id,
                kind="landmark",
                floor_id=floor_id,
                available_fields={"rgb": len(candidate.detections)},
                knowledge=(
                    str(candidate.class_name),
                    *(
                        str(item.content).strip()
                        for item in candidate.knowledge
                        if str(item.content).strip() != ""
                    ),
                ),
                metadata={"associated_node_refs": associated_nodes},
            )
        )
    return sorted(entries, key=lambda entry: (_kind_rank(entry.kind), _reference_sort_key(entry.ref)))


def memory_index_dict(
    entries: list[MemoryIndexEntry],
    *,
    provided_fields_by_ref: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    groups: dict[str, dict[str, object]] = {
        "nodes": {},
        "edges": {},
        "landmarks": {},
    }
    group_by_kind = {"node": "nodes", "edge": "edges", "landmark": "landmarks"}
    provided = provided_fields_by_ref or {}
    for entry in entries:
        payload = entry.to_index_dict()
        available_fields = dict(payload["available_fields"])
        provided_fields: dict[str, int] = {}
        for field_name in provided.get(str(entry.ref), []):
            count = available_fields.pop(str(field_name), None)
            if count is not None:
                provided_fields[str(field_name)] = int(count)
        payload["available_fields"] = available_fields
        if provided_fields:
            payload["provided_fields"] = provided_fields
        groups[group_by_kind[str(entry.kind)]][str(entry.ref)] = payload
    return groups


def memory_index_text(
    entries: list[MemoryIndexEntry],
    *,
    provided_fields_by_ref: dict[str, list[str]] | None = None,
) -> str:
    prompt_index = memory_index_dict(
        entries,
        provided_fields_by_ref=provided_fields_by_ref,
    )
    floor_ids = {str(entry.floor_id) for entry in entries}
    single_floor_id = next(iter(floor_ids)) if len(floor_ids) == 1 else ""
    for group_name, group in prompt_index.items():
        for payload in group.values():
            if single_floor_id != "":
                payload.pop("floor_id", None)
            if payload.get("knowledge") == []:
                payload.pop("knowledge", None)
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                if group_name == "edges":
                    metadata.pop("src_node_id", None)
                    metadata.pop("dst_node_id", None)
                elif group_name == "landmarks":
                    metadata.pop("associated_node_refs", None)
                if metadata == {}:
                    payload.pop("metadata", None)
    if single_floor_id != "":
        prompt_index = {"floor_id": single_floor_id, **prompt_index}
    return json.dumps(
        {"memory_index": prompt_index},
        ensure_ascii=False,
        indent=2,
    )


def normalize_retrieve_request(
    payload: dict[str, object],
    *,
    fields_by_ref: dict[str, list[str]],
    provided_fields_by_ref: dict[str, list[str]] | None = None,
    require_retrieval_conclusion: bool = False,
) -> RetrieveRequest:
    retrieval_conclusion = payload.get("retrieval_conclusion", "")
    if not isinstance(retrieval_conclusion, str):
        raise ValueError(f"retrieval_conclusion must be a string: {payload!r}")
    retrieval_conclusion = retrieval_conclusion.strip()
    if require_retrieval_conclusion and retrieval_conclusion == "":
        raise ValueError(
            "Task Executive must conclude the latest retrieval before "
            f"requesting another one: {payload!r}"
        )
    raw_retrieve = payload.get("retrieve")
    if not isinstance(raw_retrieve, dict):
        raise ValueError(f"retrieve decision requires a retrieve object: {payload!r}")
    query = raw_retrieve.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"retrieve query must be non-empty: {payload!r}")
    query = query.strip()
    raw_items = raw_retrieve.get("items")
    if not isinstance(raw_items, list) or raw_items == []:
        raise ValueError(f"retrieve items must be a non-empty list: {payload!r}")
    provided = provided_fields_by_ref or {}
    items: list[RetrieveItem] = []
    already_provided_items: list[RetrieveItem] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ValueError(f"retrieve item must be an object: {raw_item!r}")
        ref = _canonical_retrieve_ref(
            raw_item.get("ref", ""),
            fields_by_ref=fields_by_ref,
        )
        allowed_fields = fields_by_ref.get(ref)
        if allowed_fields is None:
            raise ValueError(f"unknown retrieve reference: {ref!r}")
        raw_fields = raw_item.get("fields")
        if not isinstance(raw_fields, list) or raw_fields == []:
            raise ValueError(f"retrieve fields must be a non-empty list: {raw_item!r}")
        fields: list[str] = []
        already_provided_fields: list[str] = []
        for raw_field in raw_fields:
            name = str(raw_field).strip()
            if name in allowed_fields:
                if name not in fields:
                    fields.append(name)
                continue
            if name in provided.get(ref, []):
                if name not in already_provided_fields:
                    already_provided_fields.append(name)
                continue
            raise ValueError(
                f"field {name!r} is unavailable for {ref!r}; "
                f"available={allowed_fields!r}; "
                f"provided={provided.get(ref, [])!r}"
            )
        if fields:
            items.append(RetrieveItem(ref=ref, fields=tuple(fields)))
        if already_provided_fields:
            already_provided_items.append(
                RetrieveItem(ref=ref, fields=tuple(already_provided_fields))
            )
    if items == []:
        raise ValueError(
            "all requested fields are already attached in the current planning "
            "observation; use that evidence directly or request an available "
            "historical field"
        )
    return RetrieveRequest(
        query=query,
        items=tuple(items),
        already_provided_items=tuple(already_provided_items),
        retrieval_conclusion=retrieval_conclusion,
    )


def _canonical_retrieve_ref(
    raw_ref: object,
    *,
    fields_by_ref: dict[str, list[str]],
) -> str:
    ref = str(raw_ref).strip()
    if ref in fields_by_ref:
        return ref
    for group_prefix, entity_prefix in (
        ("nodes.", "n"),
        ("edges.", "e"),
        ("landmarks.", "landmark_"),
    ):
        if not ref.startswith(group_prefix):
            continue
        candidate = ref[len(group_prefix) :]
        if candidate.startswith(entity_prefix) and candidate in fields_by_ref:
            return candidate
    return ref


def execute_retrieve_request(
    *,
    state: "NavProbeAgentState",
    workspace: RetrievalWorkspace,
    request: RetrieveRequest,
) -> RetrieveRound:
    if not workspace.can_retrieve:
        raise ValueError(
            "retrieve budget exhausted: "
            f"{workspace.retrieve_count}/{workspace.max_retrieve_rounds}"
        )
    source_obs_ids = _materialize_request(
        state=state,
        workspace=workspace,
        request=request,
    )
    _render_shared_bev(state=state, workspace=workspace)
    record = RetrieveRound(
        round_index=int(workspace.retrieve_count),
        request=RetrieveRequest(
            query=request.query,
            items=request.items,
            already_provided_items=request.already_provided_items,
            predicate_updates=request.predicate_updates,
            task_state_assessment=request.task_state_assessment,
            agenda_updates=request.agenda_updates,
        ),
        source_obs_ids=tuple(source_obs_ids),
        unavailable_fields={
            f"{item.ref}.{name}": workspace.unavailable_fields[f"{item.ref}.{name}"]
            for item in request.items for name in item.fields
            if f"{item.ref}.{name}" in workspace.unavailable_fields
        },
    )
    workspace.rounds.append(record)
    return record


def materialize_memory_context(
    *,
    state: "NavProbeAgentState",
    workspace: RetrievalWorkspace,
    entries: list[MemoryIndexEntry],
    provided_fields_by_ref: dict[str, list[str]] | None = None,
) -> MaterializedMemoryContext:
    entry_refs = {str(entry.ref) for entry in workspace.entries}
    unknown_refs = [
        str(entry.ref)
        for entry in entries
        if str(entry.ref) not in entry_refs
    ]
    if unknown_refs:
        raise ValueError(
            f"memory context entries are absent from workspace: {unknown_refs}"
        )
    provided = provided_fields_by_ref or {}
    loaded_items: list[RetrieveItem] = []
    skipped_items: list[RetrieveItem] = []
    for entry in entries:
        available_fields = [
            str(name)
            for name, count in entry.available_fields.items()
            if int(count) > 0
        ]
        provided_fields = set(provided.get(str(entry.ref), []))
        skipped_fields = [
            name
            for name in available_fields
            if name in provided_fields
        ]
        loaded_fields = [name for name in available_fields if name not in skipped_fields]
        loaded_items.append(
            RetrieveItem(ref=str(entry.ref), fields=tuple(loaded_fields))
        )
        if skipped_fields:
            skipped_items.append(
                RetrieveItem(ref=str(entry.ref), fields=tuple(skipped_fields))
            )

    workspace.clear_materialized_evidence()
    source_obs_ids = _materialize_request(
        state=state,
        workspace=workspace,
        request=RetrieveRequest(
            query="System-preloaded memory context.",
            items=tuple(loaded_items),
        ),
    )
    _render_shared_bev(state=state, workspace=workspace)
    actually_loaded: list[RetrieveItem] = []
    missing_items: list[RetrieveItem] = []
    for item in loaded_items:
        present_fields = [
            field_name
            for field_name in item.fields
            if f"{item.ref}.{field_name}" in workspace.text_evidence
            or f"{item.ref}.{field_name}" in workspace.image_panels
        ]
        missing_fields = [
            field_name for field_name in item.fields if field_name not in present_fields
        ]
        if present_fields:
            actually_loaded.append(
                RetrieveItem(ref=item.ref, fields=tuple(present_fields))
            )
        if missing_fields:
            missing_items.append(
                RetrieveItem(ref=item.ref, fields=tuple(missing_fields))
            )
    return MaterializedMemoryContext(
        loaded_fields_by_ref=_fields_by_ref_from_items(actually_loaded),
        skipped_fields_by_ref=_fields_by_ref_from_items(skipped_items),
        missing_fields_by_ref=_fields_by_ref_from_items(missing_items),
        source_obs_ids=tuple(source_obs_ids),
    )


def _fields_by_ref_from_items(
    items: list[RetrieveItem],
) -> dict[str, list[str]]:
    return {
        str(item.ref): [str(field_name) for field_name in item.fields]
        for item in items
        if item.fields
    }


def _materialize_request(
    *,
    state: "NavProbeAgentState",
    workspace: RetrievalWorkspace,
    request: RetrieveRequest,
) -> list[str]:
    entry_by_ref = {str(entry.ref): entry for entry in workspace.entries}
    source_obs_ids: list[str] = []
    for item in request.items:
        entry = entry_by_ref[str(item.ref)]
        if entry.kind == "node":
            node = state.graph.get_node(str(item.ref))
            _select_node(workspace, node_id=str(node.id), floor_id=str(node.floor_id))
            for field_name in item.fields:
                key = f"{item.ref}.{field_name}"
                if field_name == "rgb":
                    obs_ids = [str(obs_id) for obs_id in node.obs_ids]
                    _add_node_rgb_panel(
                        state=state,
                        workspace=workspace,
                        key=key,
                        node_id=str(node.id),
                        obs_ids=obs_ids,
                    )
                    _extend_unique(source_obs_ids, obs_ids)
                elif field_name == "landmarks":
                    obs_ids = _add_node_landmark_panel(
                        state=state,
                        workspace=workspace,
                        key=key,
                        node_id=str(node.id),
                        floor_id=str(node.floor_id),
                    )
                    _extend_unique(source_obs_ids, obs_ids)
        elif entry.kind == "edge":
            edge = _edge_by_id(state, str(item.ref))
            src_node = state.graph.get_node(str(edge.src_id))
            dst_node = state.graph.get_node(str(edge.dst_id))
            floor_id = str(src_node.floor_id)
            _select_node(workspace, node_id=str(src_node.id), floor_id=floor_id)
            _select_node(workspace, node_id=str(dst_node.id), floor_id=str(dst_node.floor_id))
            workspace.selected_edge_ids_by_floor.setdefault(floor_id, set()).add(str(edge.id))
            for field_name in item.fields:
                key = f"{item.ref}.{field_name}"
                if field_name == "trajectory":
                    workspace.text_evidence.setdefault(
                        key,
                        (
                            f"Edge {edge.id} actual trajectory is highlighted on the shared BEV; "
                            f"path_xy_points={len(edge.path_xy)}."
                        ),
                    )
                elif field_name == "rgb":
                    obs_ids = _add_edge_rgb_panel(
                        state=state,
                        workspace=workspace,
                        key=key,
                        edge=edge,
                        src_node=src_node,
                        dst_node=dst_node,
                    )
                    _extend_unique(source_obs_ids, obs_ids)
                elif field_name == "movement_rgb":
                    obs_ids = _movement_keyframes(
                        state=state,
                        obs_ids=[str(obs_id) for obs_id in edge.rgb_history_obs_ids],
                    )
                    _add_movement_panel(
                        state=state,
                        workspace=workspace,
                        key=key,
                        edge_id=str(edge.id),
                        obs_ids=obs_ids,
                    )
                    _extend_unique(source_obs_ids, obs_ids)
        elif entry.kind == "landmark":
            candidate = state.landmark_controller.landmark_memory.find_record(str(item.ref))
            if candidate is None:
                raise ValueError(f"missing landmark candidate for memory index ref {item.ref!r}")
            associated_nodes = list(entry.metadata.get("associated_node_refs", []))
            floor_id = str(entry.floor_id or state.system.current_floor_id)
            workspace.selected_landmark_ids_by_floor.setdefault(floor_id, set()).add(str(item.ref))
            for node_id in associated_nodes:
                node = state.graph.get_node(str(node_id))
                _select_node(workspace, node_id=str(node.id), floor_id=str(node.floor_id))
            for field_name in item.fields:
                key = f"{item.ref}.{field_name}"
                if field_name == "rgb":
                    obs_ids = _add_landmark_rgb_panel(
                        state=state,
                        workspace=workspace,
                        key=key,
                        candidate=candidate,
                    )
                    _extend_unique(source_obs_ids, obs_ids)
    return source_obs_ids


def _select_node(workspace: RetrievalWorkspace, *, node_id: str, floor_id: str) -> None:
    workspace.selected_node_ids_by_floor.setdefault(str(floor_id), set()).add(str(node_id))


def _edge_by_id(state: "NavProbeAgentState", edge_id: str):
    for edge in state.graph.iter_edges():
        if str(edge.id) == str(edge_id):
            return edge
    raise ValueError(f"unknown graph edge: {edge_id!r}")


def _landmark_ids_by_node(
    *, state: "NavProbeAgentState",
) -> dict[str, set[str]]:
    nodes = list(state.graph.iter_nodes())
    obs_ids = list(dict.fromkeys(str(obs_id) for node in nodes for obs_id in node.obs_ids))
    detections = state.landmark_controller.landmark_memory.detections_for_obs_ids(obs_ids)
    return {
        str(node.id): {
            str(item["landmark_id"])
            for obs_id in node.obs_ids
            for item in detections[str(obs_id)]
        }
        for node in nodes
    }


def _node_ids_by_landmark(by_node: dict[str, set[str]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for node_id, landmark_ids in by_node.items():
        for landmark_id in landmark_ids:
            result.setdefault(landmark_id, set()).add(node_id)
    return result


def _landmark_floor_id(
    *,
    state: "NavProbeAgentState",
    associated_node_ids: list[str],
) -> str:
    if associated_node_ids != []:
        return str(state.graph.get_node(str(associated_node_ids[0])).floor_id)
    return str(state.system.current_floor_id)


def _kind_rank(kind: str) -> int:
    return {"node": 0, "edge": 1, "landmark": 2}.get(str(kind), 3)


def _sorted_set_mapping(value: dict[str, set[str]]) -> dict[str, list[str]]:
    return {
        str(key): sorted(items, key=_reference_sort_key)
        for key, items in sorted(value.items())
    }


def _sorted_label_mapping(
    value: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    return {
        str(floor_id): {
            str(ref): int(label)
            for ref, label in sorted(
                labels.items(),
                key=lambda item: _reference_sort_key(str(item[0])),
            )
        }
        for floor_id, labels in sorted(value.items())
    }


def _inline_labeled_refs(values: set[str], labels_by_ref: dict[str, int]) -> str:
    ordered = sorted((str(value) for value in values), key=_reference_sort_key)
    if ordered == []:
        return "none"
    return ", ".join(
        (
            f"{labels_by_ref[ref]}={ref}"
            if ref in labels_by_ref
            else ref
        )
        for ref in ordered
    )


def _inline_edge_refs(workspace: RetrievalWorkspace, floor_id: str) -> str:
    edge_refs = sorted(
        workspace.selected_edge_ids_by_floor.get(str(floor_id), set()),
        key=_reference_sort_key,
    )
    if edge_refs == []:
        return "none"
    entries_by_ref = {entry.ref: entry for entry in workspace.entries}
    values: list[str] = []
    for ref in edge_refs:
        entry = entries_by_ref.get(str(ref))
        metadata = {} if entry is None else entry.metadata
        src = str(metadata.get("src_node_id", ""))
        dst = str(metadata.get("dst_node_id", ""))
        values.append(f"{ref} ({src} -> {dst})" if src != "" and dst != "" else str(ref))
    return ", ".join(values)


def _extend_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        text = str(value)
        if text != "" and text not in target:
            target.append(text)

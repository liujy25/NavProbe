from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field


NAVPROBE_AGENDA_STATUS_VALUES = {"active", "done"}
NAVPROBE_AGENDA_KIND_VALUES = {"task", "verify_candidate"}
NAVPROBE_PREDICATE_STATUS_VALUES = {"unconfirmed", "confirmed"}


@dataclass
class NavProbePredicate:
    predicate_id: str
    update_node_id: str
    content: str
    status: str = "unconfirmed"

    def __post_init__(self) -> None:
        predicate_id = str(self.predicate_id).strip()
        update_node_id = str(self.update_node_id).strip()
        content = str(self.content).strip()
        status = _normalize_predicate_status(self.status)
        if predicate_id == "":
            raise ValueError("predicate requires a predicate_id")
        if content == "":
            raise ValueError("predicate content must be non-empty")
        if status not in NAVPROBE_PREDICATE_STATUS_VALUES:
            raise ValueError(f"unsupported predicate status: {status!r}")
        self.predicate_id = predicate_id
        self.update_node_id = update_node_id
        self.content = content
        self.status = status

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "NavProbePredicate":
        return cls(
            predicate_id=str(
                payload.get("predicate_id", "")
            ),
            update_node_id=str(payload.get("update_node_id", "")),
            content=str(payload.get("content", "")),
            status=str(payload.get("status", "unconfirmed")),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "predicate_id": str(self.predicate_id),
            "update_node_id": str(self.update_node_id),
            "content": str(self.content),
            "status": str(self.status),
        }


@dataclass
class NavProbeAgendaItem:
    content: str
    status: str = "active"
    result: str = ""
    kind: str = "task"
    candidate: dict[str, object] = field(default_factory=dict)
    start_node_id: str = ""
    completion_node_id: str = ""
    predicates: list[NavProbePredicate] = field(default_factory=list)
    subgoal_id: str = ""
    attempt: int = 1

    def __post_init__(self) -> None:
        content = str(self.content).strip()
        status = _normalize_agenda_status(self.status)
        result = str(self.result).strip()
        if status == "active":
            result = ""
        kind = str(self.kind).strip().lower() or "task"
        candidate = deepcopy(self.candidate) if isinstance(self.candidate, dict) else {}
        predicates = [
            NavProbePredicate.from_dict(item.to_dict())
            if isinstance(item, NavProbePredicate)
            else NavProbePredicate.from_dict(item)
            for item in self.predicates
            if isinstance(item, (NavProbePredicate, dict))
        ]
        start_node_id = str(self.start_node_id).strip()
        completion_node_id = (
            str(self.completion_node_id).strip() if status == "done" else ""
        )
        if content == "":
            raise ValueError("NavProbeAgendaItem.content must be non-empty")
        if status not in NAVPROBE_AGENDA_STATUS_VALUES:
            raise ValueError(f"unsupported NavProbeAgendaItem.status: {status!r}")
        if kind not in NAVPROBE_AGENDA_KIND_VALUES:
            raise ValueError(f"unsupported NavProbeAgendaItem.kind: {kind!r}")
        if status == "done" and result == "":
            raise ValueError("done NavProbeAgendaItem requires non-empty result")
        if type(self.attempt) is not int or self.attempt < 1:
            raise ValueError("NavProbeAgendaItem.attempt must be a positive integer")
        self.content = content
        self.status = status
        self.result = result
        self.kind = kind
        self.candidate = candidate
        self.start_node_id = start_node_id
        self.completion_node_id = completion_node_id
        self.predicates = predicates
        self.subgoal_id = str(self.subgoal_id).strip()

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "NavProbeAgendaItem":
        if "conditions" in payload:
            raise ValueError("incompatible agenda item key 'conditions'; expected 'predicates'")
        candidate = payload.get("candidate", {})
        raw_predicates = payload.get("predicates", [])
        return cls(
            content=str(payload.get("content", "")),
            status=str(payload.get("status", "active")),
            result=str(payload.get("result", "")),
            kind=str(payload.get("kind", "task")),
            candidate=deepcopy(candidate) if isinstance(candidate, dict) else {},
            start_node_id=str(payload.get("start_node_id", "")),
            completion_node_id=str(payload.get("completion_node_id", "")),
            predicates=[
                NavProbePredicate.from_dict(item)
                for item in raw_predicates
                if isinstance(item, dict)
            ]
            if isinstance(raw_predicates, list)
            else [],
            subgoal_id=str(payload.get("subgoal_id", "")),
            attempt=payload.get("attempt", 1),
        )

    def to_dict(self) -> dict[str, object]:
        payload = {
            "content": str(self.content),
            "status": str(self.status),
            "result": str(self.result),
            "kind": str(self.kind),
            "candidate": deepcopy(self.candidate),
            "start_node_id": str(self.start_node_id),
            "completion_node_id": str(self.completion_node_id),
            "predicates": [item.to_dict() for item in self.predicates],
        }
        if self.subgoal_id:
            payload.update(subgoal_id=self.subgoal_id, attempt=self.attempt)
        return payload


@dataclass
class NavProbeTaskStateUpdateResult:
    applied_updates: list[dict[str, object]] = field(default_factory=list)
    skipped_updates: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "applied_updates": deepcopy(self.applied_updates),
            "skipped_updates": deepcopy(self.skipped_updates),
        }


@dataclass
class NavProbeTaskStateMemory:
    agenda: list[NavProbeAgendaItem] = field(default_factory=list)
    task_constraints: str = ""
    completed_stair_items: dict[int | str, dict[str, object]] = field(default_factory=dict)
    original_goal: str = ""
    agenda_initialized: bool = False
    history: list[dict[str, object]] = field(default_factory=list)
    next_subgoal_index: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "NavProbeTaskStateMemory":
        if any(key in payload for key in ("progress_items", "task_progress", "items", "todos")):
            raise ValueError("incompatible task-state record; expected the current 'agenda' format")
        if "agenda" not in payload:
            raise ValueError("task-state record requires 'agenda'")
        raw_agenda = payload["agenda"]
        if not isinstance(raw_agenda, list):
            raise ValueError("NavProbeTaskStateMemory.agenda must be a list")
        memory = cls(agenda=[NavProbeAgendaItem.from_dict(item) for item in raw_agenda if isinstance(item, dict)])
        memory.task_constraints = str(payload.get("task_constraints", ""))
        memory.completed_stair_items = {
            int(k) if str(k).isdigit() else str(k): deepcopy(v)
            for k, v in payload.get("completed_stair_items", {}).items()
        }
        memory.original_goal = str(payload.get("original_goal", ""))
        memory.agenda_initialized = bool(payload.get("agenda_initialized", False))
        history = payload.get("history", [])
        if any(isinstance(record, dict) and "conditions" in record for record in history):
            raise ValueError("incompatible history key 'conditions'; expected 'predicates'")
        memory.history = deepcopy(history)
        memory.next_subgoal_index = int(payload.get("next_subgoal_index", 0))
        if memory.agenda_initialized:
            memory.ensure_subgoal_ids()
        return memory

    def ensure_subgoal_ids(self) -> None:
        used = {str(record.get("subgoal_id", "")) for record in self.history}
        active_ids = [item.subgoal_id for item in self.agenda if item.subgoal_id]
        if len(active_ids) != len(set(active_ids)):
            raise ValueError("active subgoal IDs must be unique")
        used.update(active_ids)
        for item in self.agenda:
            if item.subgoal_id:
                continue
            while f"sg{self.next_subgoal_index}" in used:
                self.next_subgoal_index += 1
            item.subgoal_id = f"sg{self.next_subgoal_index}"
            used.add(item.subgoal_id)
            self.next_subgoal_index += 1

    def initialize_agenda(
        self, goal_text: str, agenda: list[NavProbeAgendaItem | dict[str, object]]
    ) -> None:
        goal = str(goal_text).strip()
        if not goal:
            raise ValueError("agenda requires the original task goal")
        if self.agenda_initialized:
            if self.original_goal != goal:
                raise ValueError("original task goal is immutable")
            return
        projected = deepcopy(self)
        projected.original_goal = goal
        projected.agenda = [
            NavProbeAgendaItem.from_dict(item.to_dict() if isinstance(item, NavProbeAgendaItem) else item)
            for item in agenda
        ]
        projected.ensure_subgoal_ids()
        for item in projected.agenda:
            if item.status == "done":
                projected.history.append({**item.to_dict(), "outcome": "completed"})
        projected.agenda = [item for item in projected.agenda if item.status == "active"]
        projected.agenda_initialized = True
        self.__dict__.update(projected.__dict__)

    def mark_subgoal_started(self, subgoal_id: str, current_node_id: str) -> None:
        item = self._active_subgoal(subgoal_id)
        if not item.start_node_id:
            item.start_node_id = str(current_node_id).strip()

    def _active_subgoal(self, subgoal_id: str) -> NavProbeAgendaItem:
        for item in self.agenda:
            if item.subgoal_id == subgoal_id:
                return item
        raise ValueError(f"unknown active subgoal: {subgoal_id!r}")

    def apply_task_state_updates(
        self,
        agenda_updates: list[dict[str, object]],
        predicate_updates: list[dict[str, object]],
        current_node_id: str = "",
    ) -> NavProbeTaskStateUpdateResult:
        """Apply one executive response atomically, preserving resolved attempts."""
        if not self.agenda_initialized:
            raise ValueError("agenda must be initialized before updates")
        if not isinstance(agenda_updates, list) or not isinstance(predicate_updates, list):
            raise ValueError("agenda and predicate updates must be lists")
        projected = deepcopy(self)
        result = NavProbeTaskStateUpdateResult()
        active_ids = {item.subgoal_id for item in projected.agenda}
        reopened_ids = {
            update["subgoal_id"] for update in agenda_updates
            if isinstance(update, dict) and update.get("op") == "reopen"
            and isinstance(update.get("subgoal_id"), str)
        }
        deferred_predicates: dict[str, list[dict[str, object]]] = {}
        for update in predicate_updates:
            subgoal_id = update.get("subgoal_id") if isinstance(update, dict) else None
            if isinstance(subgoal_id, str) and subgoal_id not in active_ids and subgoal_id in reopened_ids:
                deferred_predicates.setdefault(subgoal_id, []).append(update)
            else:
                projected._apply_predicate_update(update, result, current_node_id)
        for update in agenda_updates:
            projected._apply_agenda_operation(update, result, current_node_id)
            if update["op"] == "reopen":
                for predicate in deferred_predicates.pop(update["subgoal_id"], []):
                    projected._apply_predicate_update(predicate, result, current_node_id)
        self.__dict__.update(projected.__dict__)
        return result

    def _apply_predicate_update(
        self, update: dict[str, object], result: NavProbeTaskStateUpdateResult, current_node_id: str
    ) -> None:
        _validate_agenda_fields(update, {
            "add": {"op", "subgoal_id", "content", "status"},
            "update": {"op", "subgoal_id", "predicate_id", "status"},
            "rewrite": {"op", "subgoal_id", "predicate_id", "content", "status"},
            "remove": {"op", "subgoal_id", "predicate_id"},
        })
        item = self._active_subgoal(str(update["subgoal_id"]))
        _apply_predicate_operations(
            item.predicates, [update], current_node_id=str(current_node_id).strip()
        )
        result.applied_updates.append({
            **deepcopy(update), "predicates": [predicate.to_dict() for predicate in item.predicates]
        })

    def _apply_agenda_operation(
        self, update: dict[str, object], result: NavProbeTaskStateUpdateResult, current_node_id: str
    ) -> None:
        _validate_agenda_fields(update, {
            "add": {"op", "content", "position"},
            "rewrite": {"op", "subgoal_id", "content"},
            "reorder": {"op", "subgoal_ids"},
            "complete": {"op", "subgoal_id", "result"},
            "abandon": {"op", "subgoal_id", "result"},
            "reopen": {"op", "subgoal_id", "position"},
        })
        op = update["op"]
        applied = deepcopy(update)
        if op == "reorder":
            ids = update["subgoal_ids"]
            expected = {item.subgoal_id for item in self.agenda}
            if (
                not isinstance(ids, list) or not all(isinstance(item, str) for item in ids)
                or len(ids) != len(expected) or set(ids) != expected
            ):
                raise ValueError("reorder must enumerate every active subgoal exactly once")
            self.agenda = [self._active_subgoal(subgoal_id) for subgoal_id in ids]
        elif op in {"add", "reopen"}:
            position = update["position"]
            if type(position) is not int or not 0 <= position <= len(self.agenda):
                raise ValueError("agenda position must be an integer within the agenda")
            if op == "add":
                item = NavProbeAgendaItem(content=_agenda_text(update, "content"))
            else:
                subgoal_id = _agenda_text(update, "subgoal_id")
                if any(item.subgoal_id == subgoal_id for item in self.agenda):
                    raise ValueError(f"subgoal is already active: {subgoal_id!r}")
                previous = next(
                    (record for record in reversed(self.history) if record["subgoal_id"] == subgoal_id), None
                )
                if previous is None:
                    raise ValueError(f"unknown historical subgoal: {subgoal_id!r}")
                item = NavProbeAgendaItem.from_dict({
                    **previous, "status": "active", "result": "", "start_node_id": "",
                    "completion_node_id": "", "attempt": int(previous["attempt"]) + 1,
                })
            self.agenda.insert(position, item)
            self.ensure_subgoal_ids()
            applied["agenda_item"] = item.to_dict()
            applied["subgoal_id"] = item.subgoal_id
        else:
            item = self._active_subgoal(_agenda_text(update, "subgoal_id"))
            if op == "rewrite":
                item.content = _agenda_text(update, "content")
                applied["agenda_item"] = item.to_dict()
            else:
                record = {
                    **item.to_dict(),
                    "status": "done" if op == "complete" else "abandoned",
                    "outcome": "completed" if op == "complete" else "abandoned",
                    "result": _agenda_text(update, "result"),
                    "completion_node_id": str(current_node_id).strip(),
                }
                self.history.append(record)
                self.agenda.remove(item)
                applied["history_record"] = deepcopy(record)
        result.applied_updates.append(applied)

    def _format_agenda_for_prompt(
        self, *, include_node_bindings: bool, show_empty_predicates: bool
    ) -> str:
        lines = ["Original task goal:", self.original_goal]
        if self.task_constraints:
            lines.extend(["", "Task constraints:", self.task_constraints])
        lines.extend(["", "Active subgoal agenda (pursuit priority):"])
        for position, item in enumerate(self.agenda):
            lines.append(f"{position}. {item.subgoal_id} [active, attempt {item.attempt}] {item.content}")
            if include_node_bindings and item.start_node_id:
                lines.append(f"   start node: {item.start_node_id}")
            if show_empty_predicates and not item.predicates:
                lines.append("   predicates: empty")
            for predicate in item.predicates:
                node = f"; updated at {predicate.update_node_id}" if include_node_bindings and predicate.update_node_id else ""
                lines.append(f"   - {predicate.predicate_id} [{predicate.status}] {predicate.content}{node}")
        if not self.agenda:
            lines.append("empty")
        lines.extend(["", "Execution history:"])
        for record in self.history:
            lines.append(
                f"{record['subgoal_id']} [attempt {record['attempt']}, {record['outcome']}] "
                f"{record['content']}; result={record['result']}"
            )
            if include_node_bindings:
                lines.append(f"   node span: {record.get('start_node_id') or 'unrecorded'} -> {record.get('completion_node_id') or 'unrecorded'}")
            for predicate in record.get("predicates", []):
                lines.append(f"   - {predicate['predicate_id']} [{predicate['status']}] {predicate['content']}")
        if not self.history:
            lines.append("empty")
        if self.completed_stair_items:
            lines.extend(["", "Executed floor-transition evidence:"])
            for key, record in self.completed_stair_items.items():
                lines.append(f"{key}: {record}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "agenda": [item.to_dict() for item in self.agenda],
        }
        if self.task_constraints:
            payload["task_constraints"] = self.task_constraints
        if self.completed_stair_items:
            payload["completed_stair_items"] = {str(k): deepcopy(v) for k, v in self.completed_stair_items.items()}
        if self.agenda_initialized:
            payload.update(
                original_goal=self.original_goal,
                agenda_initialized=True,
                history=deepcopy(self.history),
                next_subgoal_index=self.next_subgoal_index,
            )
        return payload

    def format_for_prompt(
        self,
        *,
        include_node_bindings: bool = True,
        show_empty_predicates: bool = False,
    ) -> str:
        if not self.agenda_initialized:
            if self.agenda or self.task_constraints:
                raise ValueError("agenda must be initialized before rendering task state")
            return "empty"
        return self._format_agenda_for_prompt(
            include_node_bindings=include_node_bindings,
            show_empty_predicates=show_empty_predicates,
        )


def _validate_agenda_fields(
    update: object, fields_by_op: dict[str, set[str]]
) -> None:
    if not isinstance(update, dict):
        raise ValueError("agenda update must be an object")
    op = update.get("op")
    if not isinstance(op, str) or op not in fields_by_op or set(update) != fields_by_op[op]:
        raise ValueError(f"invalid agenda update fields: {update!r}")


def _agenda_text(update: dict[str, object], field_name: str) -> str:
    value = update[field_name]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"agenda {field_name} must be non-empty text")
    return value.strip()


def _apply_predicate_operations(
    predicates: list[NavProbePredicate],
    raw_updates: list[object],
    *,
    current_node_id: str,
) -> None:
    for raw_update in raw_updates:
        if not isinstance(raw_update, dict):
            raise ValueError("predicate update must be an object")
        op = str(raw_update.get("op", "")).strip().lower()
        if op == "add":
            content = str(raw_update.get("content", "")).strip()
            if any(item.content.casefold() == content.casefold() for item in predicates):
                raise ValueError("predicate content must be distinct")
            predicates.append(
                NavProbePredicate(
                    predicate_id=_next_predicate_id(predicates),
                    update_node_id=current_node_id,
                    content=content,
                    status=str(raw_update.get("status", "unconfirmed")),
                )
            )
            continue
        predicate_id = str(
            raw_update.get("predicate_id", "")
        ).strip()
        item = next(
            (
                candidate
                for candidate in predicates
                if candidate.predicate_id == predicate_id
            ),
            None,
        )
        if item is None:
            raise ValueError(f"unknown predicate id: {predicate_id!r}")
        if op == "update":
            status = _normalize_predicate_status(raw_update.get("status", ""))
            if status not in NAVPROBE_PREDICATE_STATUS_VALUES:
                raise ValueError(f"unsupported predicate status: {status!r}")
            if item.status != status:
                item.status = status
                item.update_node_id = current_node_id
        elif op == "rewrite":
            content = str(raw_update.get("content", "")).strip()
            if content == "":
                raise ValueError("predicate content must be non-empty")
            status = _normalize_predicate_status(raw_update.get("status", ""))
            if status not in NAVPROBE_PREDICATE_STATUS_VALUES:
                raise ValueError(f"unsupported predicate status: {status!r}")
            if any(
                candidate.predicate_id != predicate_id
                and candidate.content.casefold() == content.casefold()
                for candidate in predicates
            ):
                raise ValueError("predicate content must be distinct")
            if item.content != content or item.status != status:
                item.content = content
                item.status = status
                item.update_node_id = current_node_id
        elif op == "remove":
            predicates.remove(item)
        else:
            raise ValueError(f"unsupported predicate op: {op!r}")


def _normalize_agenda_status(value: object) -> str:
    status = str(value).strip().lower()
    return status


def _normalize_predicate_status(value: object) -> str:
    status = str(value).strip().lower()
    return status


def _next_predicate_id(
    predicates: list[NavProbePredicate],
) -> str:
    used = {str(item.predicate_id) for item in predicates}
    index = 0
    while f"pc{index}" in used:
        index += 1
    return f"pc{index}"

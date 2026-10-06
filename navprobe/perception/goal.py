from __future__ import annotations

from dataclasses import dataclass

# Persisted tag for the shared language goal representation across navigation tasks.
NAVPROBE_LANGUAGE_GOAL_KIND = "vln_instruction"


def _normalize_text(value: object, field_name: str) -> str:
    text = str(value).strip()
    if text == "":
        raise ValueError(f"{field_name} must be non-empty")
    return text


@dataclass
class NavProbeGoalSpec:
    description: str
    goal_kind: str = NAVPROBE_LANGUAGE_GOAL_KIND
    task_constraints: str = ""

    def __post_init__(self) -> None:
        self.goal_kind = str(self.goal_kind).strip()
        if self.goal_kind != NAVPROBE_LANGUAGE_GOAL_KIND:
            raise ValueError(f"unsupported goal_kind: {self.goal_kind!r}")
        self.description = _normalize_text(self.description, "description")

    def to_dict(self) -> dict[str, object]:
        payload = {
            "description": self.description,
            "goal_kind": self.goal_kind,
        }
        if self.task_constraints:
            payload["task_constraints"] = self.task_constraints
        return payload


    def navigation_goal_text(self) -> str:
        """Return the unified natural-language goal shown to navigation planners."""
        return self.description


def navigation_goal_spec(instruction_text: str, *, task_constraints: str = "") -> NavProbeGoalSpec:
    normalized_instruction = _normalize_text(instruction_text, "instruction_text")
    return NavProbeGoalSpec(
        description=normalized_instruction,
        goal_kind=NAVPROBE_LANGUAGE_GOAL_KIND,
        task_constraints=task_constraints,
    )

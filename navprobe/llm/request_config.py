from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


# Total attempts, including the first call, for model requests and response correction.
NAVPROBE_MODEL_MAX_ATTEMPTS = 3

@dataclass
class ModelRequestBudget:
    """One decision's shared transport, JSON and semantic-validation budget."""

    attempts: int = 0
    last_response_log: dict | None = None

    @property
    def remaining(self) -> int:
        return NAVPROBE_MODEL_MAX_ATTEMPTS - self.attempts

    def consume(self) -> int:
        if not self.remaining:
            raise RuntimeError("model request budget exhausted")
        self.attempts += 1
        return self.attempts

    def record_validation_error(self, error: ValueError) -> None:
        if self.last_response_log is not None:
            self.last_response_log.update(
                status="validation_error",
                error={"type": type(error).__name__, "message": str(error)},
            )


_REQUEST_BUDGET = ContextVar("navprobe_model_request_budget", default=None)


@contextmanager
def model_request_budget():
    """Nested response correction reuses the current request budget."""
    current = _REQUEST_BUDGET.get()
    if current is not None:
        yield current
        return
    budget = ModelRequestBudget()
    token = _REQUEST_BUDGET.set(budget)
    try:
        yield budget
    finally:
        _REQUEST_BUDGET.reset(token)

REASONING_EFFORT_CHOICES = ("none", "low", "medium", "high", "xhigh")
# Module order for positional reasoning-effort overrides.
NAVPROBE_REASONING_MODULES = (
    "task_executive", "skill_selector", "waypoint_grounder",
    "entity_knowledge_manager", "place_memory",
    "vertical_navigation", "landmark_perception",
)


def validate_module_reasoning_efforts(values):
    if values is None:
        return None
    if not isinstance(values, (list, tuple)) or len(values) != len(NAVPROBE_REASONING_MODULES):
        raise ValueError(f"module reasoning efforts require {len(NAVPROBE_REASONING_MODULES)} integers")
    if any(type(value) is not int or not 0 <= value <= 4 for value in values):
        raise ValueError("module reasoning efforts must be integers from 0 to 4")
    return list(values)


def reasoning_effort_by_module(default, values):
    values = validate_module_reasoning_efforts(values)
    return {module: (default if values is None else REASONING_EFFORT_CHOICES[values[index]])
            for index, module in enumerate(NAVPROBE_REASONING_MODULES)}


def _model_uses_temperature_parameter(model: str) -> bool:
    normalized = str(model).strip().lower()
    return not (
        normalized == "4o"
        or normalized.startswith("4o-")
        or normalized.startswith("gpt-")
        or normalized.startswith("chatgpt-")
        or normalized.startswith("o1")
        or normalized.startswith("o3")
        or normalized.startswith("o4")
    )

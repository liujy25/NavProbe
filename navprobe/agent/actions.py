from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentActionType(str, Enum):
    VISUAL_WAYPOINT = "visual_waypoint"
    VERTICAL_TRANSITION = "vertical_transition"
    FINALIZE = "finalize"


@dataclass(frozen=True)
class AgentAction:
    action_type: AgentActionType
    args: dict[str, Any] = field(default_factory=dict)
    source: str = "deterministic_policy"
    reason: str = ""

    def __post_init__(self) -> None:
        _validate_agent_action(self.action_type, self.args)

    def to_dict(self) -> dict[str, object]:
        return {
            "action_type": str(self.action_type.value),
            "args": dict(self.args),
            "source": str(self.source),
            "reason": str(self.reason),
        }


def _validate_agent_action(action_type: AgentActionType, args: dict[str, Any]) -> None:
    if action_type == AgentActionType.VISUAL_WAYPOINT:
        selected_angle_deg = args.get("selected_angle_deg")
        if selected_angle_deg is None:
            raise ValueError("visual_waypoint action requires args.selected_angle_deg")
    elif action_type == AgentActionType.VERTICAL_TRANSITION:
        direction = str(args.get("direction", "")).strip()
        if direction not in {"up", "down"}:
            raise ValueError("vertical_transition action requires direction up or down")
    elif action_type == AgentActionType.FINALIZE:
        reason = str(args.get("reason", "")).strip()
        if reason == "":
            raise ValueError("finalize action requires args.reason")
    else:
        raise ValueError(f"unsupported agent action type: {action_type!r}")

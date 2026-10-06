"""Destination decision recorded by the stair controller runtime adapter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VerticalTransitionStepDecision:
    transition_status: str
    waypoint_target: str
    selected_angle_deg: int | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "transition_status": str(self.transition_status),
            "waypoint_target": str(self.waypoint_target),
            "selected_angle_deg": (
                None if self.selected_angle_deg is None else int(self.selected_angle_deg)
            ),
            "reason": str(self.reason),
        }

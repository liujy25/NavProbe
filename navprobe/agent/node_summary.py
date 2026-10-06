from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from navprobe.agent.visual_action_context import angle_for_direction
from navprobe.agent.visual_action_context import direction_for_angle
from navprobe.agent.visual_action_context import ordered_panorama_angles
from navprobe.agent.visual_action_context import VISUAL_ACTION_ANGLES
from navprobe.agent.visual_policy_prompt_images import (
    image_content_for_current_panorama_views,
)

if TYPE_CHECKING:
    from navprobe.agent.visual_action_context import VisualActionContext
    from navprobe.llm.client import LLMClient
    from navprobe.runtime.cache import RuntimeCache


@dataclass(frozen=True)
class NodeSummaryDecision:
    node_summary: str
    direction_summaries: dict[int, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "node_summary": str(self.node_summary),
            "direction_summaries": {
                direction_for_angle(angle): str(self.direction_summaries.get(angle, ""))
                for angle in VISUAL_ACTION_ANGLES
                if angle in self.direction_summaries
            },
        }


NODE_SUMMARY_ANGLES = list(VISUAL_ACTION_ANGLES)


def summarize_current_node(
    *,
    client: "LLMClient",
    cache: "RuntimeCache",
    visual_context: "VisualActionContext",
) -> NodeSummaryDecision:
    allowed_angles = ordered_panorama_angles(
        [int(view.angle_deg) for view in visual_context.views]
    )
    system_prompt = """
You are the Place Memory module in NavProbe.
Describe the observed place so later navigation decisions can recognize it and find relevant visual evidence.
""".strip()
    text = f"""
Input:
The labeled views form a panorama of one place. Use only these images; no task or execution history is supplied.

Describe the place:
- Identify the current place from visible room boundaries and distinctive features. An object seen through a doorway may belong to the next room. Use a specific type such as kitchen only when the images establish it; otherwise use a description such as room, corridor, doorway area, junction, entrance, or unclear indoor place.
- Mention clearly visible objects, landmarks, passages, and floor areas. An open-looking passage does not establish that the robot can move through it. Leave unseen areas undescribed.
- Record observations, without inferring task progress, past events, or a navigation action. Omit image annotations and system identifiers such as boxes, coordinates, and candidate labels.

Output contract:
Return only JSON. Write one concise sentence in node_summary. In direction_summaries, describe the distinctive navigation-relevant cues in each labeled view; use an empty string when there are none or they only repeat node_summary.
{{
  "node_summary": "<one concise place-level sentence>",
  "direction_summaries": {_direction_summary_schema_text(allowed_angles)}
}}
""".strip()
    content = [{"type": "text", "text": text}]
    content.extend(
        image_content_for_current_panorama_views(
            cache=cache,
            views=visual_context.views,
            include_visited_nodes=False,
        )
    )
    parsed = client.summarize_node(system_prompt, content)
    return normalize_node_summary(parsed, allowed_angles=allowed_angles)


def normalize_node_summary(
    payload: dict[str, object],
    *,
    allowed_angles: list[int] | None = None,
) -> NodeSummaryDecision:
    valid_angles = (
        NODE_SUMMARY_ANGLES
        if allowed_angles is None
        else ordered_panorama_angles([int(angle) for angle in allowed_angles])
    )
    if not isinstance(payload, dict) or set(payload) != {"node_summary", "direction_summaries"}:
        raise ValueError("place memory requires node_summary and direction_summaries")
    if not isinstance(payload["node_summary"], str) or not payload["node_summary"].strip():
        raise ValueError(f"place memory returned empty or invalid node_summary: {payload!r}")
    direction_summaries = _normalize_direction_summaries(
        payload["direction_summaries"],
        allowed_angles=valid_angles,
    )
    return NodeSummaryDecision(
        node_summary=payload["node_summary"].strip(),
        direction_summaries=direction_summaries,
    )


def _normalize_direction_summaries(value: object, *, allowed_angles: list[int]) -> dict[int, str]:
    if not isinstance(value, dict):
        raise ValueError("direction_summaries must be an object keyed by panorama direction")
    summaries = {int(angle): "" for angle in allowed_angles}
    for direction, description in value.items():
        angle = angle_for_direction(direction)
        if not isinstance(description, str):
            raise ValueError("direction_summaries descriptions must be strings")
        if angle in summaries:
            summaries[angle] = description.strip()
    return summaries


def _direction_summary_schema_text(angles: list[int]) -> str:
    entries = [
        f'"{direction_for_angle(angle)}": "<visible content toward {direction_for_angle(angle)} or empty string>"'
        for angle in ordered_panorama_angles(angles)
    ]
    return "{ " + ", ".join(entries) + " }"

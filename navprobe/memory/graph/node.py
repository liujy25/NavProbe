from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from navprobe.agent.visual_action_context import direction_for_angle
from navprobe.agent.visual_action_context import ordered_panorama_angles
from navprobe.memory.entity_knowledge import EntityKnowledge

if TYPE_CHECKING:
    from navprobe.mapping.exploration.manager import ExplorationManager
    from navprobe.types import LocalmapFrontierRecord


@dataclass
class Node:
    id: str
    position: tuple[float, float, float]
    yaw: float
    floor_id: str = "floor_0"
    node_kind: str = "place"
    obs_ids: list[str] = field(default_factory=list)
    category: str = ""
    features: list[str] = field(default_factory=list)
    knowledge: list[EntityKnowledge] = field(default_factory=list)
    semantic_anchor_xy: tuple[float, float] | None = None
    nav_goal_xy: tuple[float, float] | None = None
    localmap: ExplorationManager | None = None
    frontiers: dict[str, LocalmapFrontierRecord] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Node":
        position = payload.get("position")
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError(f"node.position must be length-3 list/tuple: {position!r}")
        semantic_anchor_xy = payload.get("semantic_anchor_xy")
        if semantic_anchor_xy is None:
            semantic_anchor_xy_value = None
        else:
            if not isinstance(semantic_anchor_xy, (list, tuple)) or len(semantic_anchor_xy) != 2:
                raise ValueError(
                    "node.semantic_anchor_xy must be length-2 list/tuple or None: "
                    f"{semantic_anchor_xy!r}"
                )
            semantic_anchor_xy_value = (
                float(semantic_anchor_xy[0]),
                float(semantic_anchor_xy[1]),
            )
        nav_goal_xy = payload.get("nav_goal_xy")
        if nav_goal_xy is None:
            nav_goal_xy_value = None
        else:
            if not isinstance(nav_goal_xy, (list, tuple)) or len(nav_goal_xy) != 2:
                raise ValueError(f"node.nav_goal_xy must be length-2 list/tuple or None: {nav_goal_xy!r}")
            nav_goal_xy_value = (float(nav_goal_xy[0]), float(nav_goal_xy[1]))
        node = cls(
            id=str(payload["id"]),
            position=(float(position[0]), float(position[1]), float(position[2])),
            yaw=float(payload["yaw"]),
            floor_id=str(payload.get("floor_id", "floor_0")),
            node_kind=str(payload.get("node_kind", "place")),
            obs_ids=[str(obs_id) for obs_id in list(payload.get("obs_ids", []))],
            category=str(payload.get("category", "")),
            features=[str(feature) for feature in list(payload.get("features", []))],
            knowledge=[
                EntityKnowledge.from_dict(item)
                for item in list(payload.get("knowledge", []))
                if isinstance(item, dict)
            ],
            semantic_anchor_xy=semantic_anchor_xy_value,
            nav_goal_xy=nav_goal_xy_value,
        )
        raw_frontiers = payload.get("frontiers", [])
        if isinstance(raw_frontiers, dict):
            frontier_payloads = list(raw_frontiers.values())
        elif isinstance(raw_frontiers, list):
            frontier_payloads = raw_frontiers
        else:
            raise ValueError(f"node.frontiers must be list/dict when present: {raw_frontiers!r}")
        if frontier_payloads != []:
            from navprobe.types import LocalmapFrontierRecord

            for raw_frontier in frontier_payloads:
                if not isinstance(raw_frontier, dict):
                    raise ValueError(
                        "node.frontiers entries must be dict payloads: "
                        f"{type(raw_frontier).__name__}"
                    )
                frontier = LocalmapFrontierRecord.from_dict(raw_frontier)
                node.frontiers[str(frontier.frontier_id)] = frontier
        return node

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "position": self.position,
            "yaw": self.yaw,
            "floor_id": self.floor_id,
            "node_kind": self.node_kind,
            "obs_ids": list(self.obs_ids),
            "category": self.category,
            "features": list(self.features),
            "knowledge": [item.to_dict() for item in self.knowledge],
            "semantic_anchor_xy": None
            if self.semantic_anchor_xy is None
            else [float(self.semantic_anchor_xy[0]), float(self.semantic_anchor_xy[1])],
            "nav_goal_xy": None
            if self.nav_goal_xy is None
            else [float(self.nav_goal_xy[0]), float(self.nav_goal_xy[1])],
            "frontiers": [
                self.frontiers[frontier_id].to_dict()
                for frontier_id in sorted(self.frontiers)
            ],
        }


def _node_observation_knowledge_specs(
    *,
    node_summary: str,
    direction_summaries: dict[int, str],
) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    summary = str(node_summary).strip()
    if summary != "":
        specs.append(("observation_summary", summary))
    for angle in ordered_panorama_angles(list(direction_summaries)):
        description = direction_summaries[angle]
        text = str(description).strip()
        if text == "":
            continue
        angle_value = int(angle)
        specs.append(
            (
                f"observation_{direction_for_angle(angle_value)}",
                f"{direction_for_angle(angle_value)} visible: {text}",
            )
        )
    return specs


def add_node_observation_knowledge(
    node: Node,
    *,
    node_summary: str,
    direction_summaries: dict[int, str],
) -> list[EntityKnowledge]:
    existing_ids = {item.knowledge_id for item in node.knowledge}
    existing_contents = {item.content.casefold() for item in node.knowledge}
    added: list[EntityKnowledge] = []
    for knowledge_id, content in _node_observation_knowledge_specs(
        node_summary=node_summary,
        direction_summaries=direction_summaries,
    ):
        if knowledge_id in existing_ids or content.casefold() in existing_contents:
            continue
        item = EntityKnowledge(
            knowledge_id=knowledge_id,
            update_node_id=str(node.id),
            content=content,
        )
        node.knowledge.append(item)
        added.append(item)
        existing_ids.add(knowledge_id)
        existing_contents.add(content.casefold())
    return added


def has_node_observation_knowledge(node: Node) -> bool:
    return any(item.knowledge_id == "observation_summary" for item in node.knowledge)

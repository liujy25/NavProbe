from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EntityKnowledge:
    knowledge_id: str
    update_node_id: str
    content: str

    def __post_init__(self) -> None:
        self.knowledge_id = str(self.knowledge_id).strip()
        self.update_node_id = str(self.update_node_id).strip()
        self.content = str(self.content).strip()
        if self.knowledge_id == "":
            raise ValueError("entity knowledge requires a knowledge_id")
        if self.update_node_id == "":
            raise ValueError("entity knowledge requires an update_node_id")
        if self.content == "":
            raise ValueError("entity knowledge requires non-empty content")

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "EntityKnowledge":
        return cls(
            knowledge_id=str(payload.get("knowledge_id", "")),
            update_node_id=str(payload.get("update_node_id", "")),
            content=str(payload.get("content", "")),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "knowledge_id": str(self.knowledge_id),
            "update_node_id": str(self.update_node_id),
            "content": str(self.content),
        }


def next_entity_knowledge_id(items: list[EntityKnowledge]) -> str:
    used = {str(item.knowledge_id) for item in items}
    index = 0
    while f"k{index}" in used:
        index += 1
    return f"k{index}"


def merge_entity_knowledge(
    target: list[EntityKnowledge],
    source: list[EntityKnowledge],
) -> None:
    known_contents = {str(item.content).casefold() for item in target}
    used_ids = {str(item.knowledge_id) for item in target}
    for item in source:
        if str(item.content).casefold() in known_contents:
            continue
        knowledge_id = str(item.knowledge_id)
        if knowledge_id in used_ids:
            knowledge_id = next_entity_knowledge_id(target)
        target.append(
            EntityKnowledge(
                knowledge_id=knowledge_id,
                update_node_id=str(item.update_node_id),
                content=str(item.content),
            )
        )
        known_contents.add(str(item.content).casefold())
        used_ids.add(knowledge_id)

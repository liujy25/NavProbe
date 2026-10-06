"""Read serialized graph geometry for recordings and rendered artifacts."""

from __future__ import annotations

from typing import Any


def graph_floor_height(graph_payload: dict[str, Any], floor_id: str) -> float | None:
    floors = graph_payload.get("floors", [])
    if not isinstance(floors, list):
        return None
    for floor in floors:
        if not isinstance(floor, dict):
            continue
        if str(floor.get("id", "")) == str(floor_id):
            height = floor.get("height")
            if height is None:
                return None
            return float(height)
    return None


def graph_payload_edges(
    graph_payload: dict[str, Any],
    *,
    include_vertical: bool,
    floor_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return source edge records in payload order, preferring a flat edge list.

    Nested vertical edges are appended only for an unfiltered all-floor view.
    Flat edge lists are filtered by each edge's ``floor_id`` when specified.
    """
    floor_id_str = None if floor_id is None else str(floor_id)
    raw_edges = graph_payload.get("edges")
    if isinstance(raw_edges, list):
        edges = [edge for edge in raw_edges if isinstance(edge, dict)]
        if floor_id_str is None:
            return edges
        return [edge for edge in edges if str(edge.get("floor_id", "")) == floor_id_str]
    edges: list[dict[str, Any]] = []
    raw_floors = graph_payload.get("floors")
    if isinstance(raw_floors, list):
        for floor in raw_floors:
            if not isinstance(floor, dict):
                continue
            if floor_id_str is not None and str(floor.get("id", "")) != floor_id_str:
                continue
            raw_floor_edges = floor.get("edges", [])
            if isinstance(raw_floor_edges, list):
                edges.extend(edge for edge in raw_floor_edges if isinstance(edge, dict))
    if include_vertical and floor_id_str is None:
        raw_vertical_edges = graph_payload.get("vertical_edges", [])
        if isinstance(raw_vertical_edges, list):
            edges.extend(edge for edge in raw_vertical_edges if isinstance(edge, dict))
    return edges

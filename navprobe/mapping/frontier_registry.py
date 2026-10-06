from __future__ import annotations


from navprobe.mapping.exploration.bev_map import FrontierCandidate
from navprobe.memory.graph.graph import Graph
from navprobe.types import LocalmapFrontierRecord, LocalmapOverlayRecord


def _frontier_id_sort_key(frontier_id: str) -> tuple[int, int, str]:
    parts = str(frontier_id).split("-", 1)
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return (int(parts[0]), int(parts[1]), str(frontier_id))
    return (10**9, 10**9, str(frontier_id))


def _copy_frontier_candidate(candidate: FrontierCandidate, frontier_id: str) -> FrontierCandidate:
    return FrontierCandidate(
        frontier_id=str(frontier_id),
        xy=(float(candidate.xy[0]), float(candidate.xy[1])),
        goal_xy=(float(candidate.goal_xy[0]), float(candidate.goal_xy[1])),
        path_xy=[(float(x), float(y)) for x, y in candidate.path_xy],
        path_length=float(candidate.path_length),
    )


def _frontier_candidate_from_record(record: LocalmapFrontierRecord) -> FrontierCandidate:
    return FrontierCandidate(
        frontier_id=str(record.frontier_id),
        xy=(float(record.frontier_xy[0]), float(record.frontier_xy[1])),
        goal_xy=(float(record.goal_xy[0]), float(record.goal_xy[1])),
        path_xy=[(float(x), float(y)) for x, y in record.path_xy],
        path_length=float(record.path_length),
    )


def _active_frontier_ids_by_overlay(
    frontier_records: dict[str, LocalmapFrontierRecord],
) -> dict[str, list[str]]:
    active_by_overlay: dict[str, list[str]] = {}
    for frontier_id, record in frontier_records.items():
        overlay_id = str(record.overlay_id)
        if overlay_id == "":
            continue
        active_by_overlay.setdefault(overlay_id, []).append(str(frontier_id))
    for frontier_ids in active_by_overlay.values():
        frontier_ids.sort(key=_frontier_id_sort_key)
    return active_by_overlay


def _sync_frontier_overlay_state(
    frontier_records: dict[str, LocalmapFrontierRecord],
    overlay_records: dict[str, LocalmapOverlayRecord],
) -> None:
    active_overlays = {
        str(record.overlay_id)
        for record in frontier_records.values()
        if str(record.overlay_id) != ""
    }
    stale_overlay_ids = [
        overlay_id
        for overlay_id in overlay_records
        if overlay_id not in active_overlays
    ]
    for overlay_id in stale_overlay_ids:
        del overlay_records[overlay_id]


def _register_frontier_record(
    *,
    graph: Graph,
    frontier_records: dict[str, LocalmapFrontierRecord],
    record: LocalmapFrontierRecord,
) -> None:
    """Keep the ordered floor registry and Node serialization index in sync."""
    node = graph.get_node(str(record.source_node_id))
    frontier_records[str(record.frontier_id)] = record
    node.frontiers[str(record.frontier_id)] = record


def _remove_frontier_record(
    frontier_records: dict[str, LocalmapFrontierRecord],
    frontier_id: str,
    graph: Graph,
) -> None:
    record = frontier_records.get(str(frontier_id))
    if record is None:
        return
    node = graph.get_node(str(record.source_node_id))
    del frontier_records[str(frontier_id)]
    node.frontiers.pop(str(record.frontier_id), None)


def _remove_owned_frontier_records(
    frontier_records: dict[str, LocalmapFrontierRecord],
    source_node_id: str,
    graph: Graph,
) -> None:
    stale_frontier_ids = [
        frontier_id
        for frontier_id, record in frontier_records.items()
        if str(record.source_node_id) == str(source_node_id)
    ]
    for frontier_id in stale_frontier_ids:
        _remove_frontier_record(frontier_records, frontier_id, graph)

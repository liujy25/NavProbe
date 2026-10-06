from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from navprobe.mapping.exploration.bev_map import FrontierCandidate
from navprobe.mapping.exploration.manager import ExplorationManager
from navprobe.memory.graph.graph import Graph
from navprobe.mapping.frontier_generation import (
    _filter_local_candidate_proposals,
    _place_index_by_node_id,
    _rebuild_active_overlay_records,
    _register_place_frontiers,
    _remove_old_frontiers_visible_from_new_place,
)
from navprobe.mapping.frontier_registry import (
    _frontier_id_sort_key,
    _remove_frontier_record,
    _sync_frontier_overlay_state,
)
from navprobe.types import (
    LocalmapFrontierRecord,
    LocalmapOverlayRecord,
    LocalmapPlaceReuseState,
)
from navprobe.runtime.cache import RuntimeCache


GLOBAL_STALE_FRONTIER_UNKNOWN_RADIUS_PX = 10


@dataclass
class LocalmapFrontierUpdateResult:
    covered_frontier_ids_removed: list[str] = field(default_factory=list)
    place_coverage_pruned_frontier_debug: list[dict[str, object]] = field(default_factory=list)
    stale_frontier_ids_removed: list[str] = field(default_factory=list)
    stale_frontier_pruned_debug: list[dict[str, object]] = field(default_factory=list)
    filtered_candidates: list[FrontierCandidate] = field(default_factory=list)
    local_candidate_filter_debug: list[dict[str, object]] = field(default_factory=list)
    created_registered_frontier_ids: list[str] = field(default_factory=list)
    skipped_for_reuse: bool = False

    def timing_details(
        self,
        *,
        active_frontier_count: int,
    ) -> dict[str, object]:
        dilated_obstacle_removed_count = sum(
            1
            for item in self.stale_frontier_pruned_debug
            if str(item.get("reason", "")) == "frontier_xy_in_dilated_obstacle"
        )
        return {
            "filtered_candidate_count": len(self.filtered_candidates),
            "created_frontier_count": len(self.created_registered_frontier_ids),
            "covered_frontier_removed_count": len(self.covered_frontier_ids_removed),
            "stale_frontier_removed_count": len(self.stale_frontier_ids_removed),
            "dilated_obstacle_frontier_removed_count": int(dilated_obstacle_removed_count),
            "active_frontier_count": int(active_frontier_count),
            "skipped_for_reuse": bool(self.skipped_for_reuse),
        }


def update_localmap_place_frontiers(
    *,
    graph: Graph,
    cache: RuntimeCache,
    global_exploration: ExplorationManager,
    local_exploration: ExplorationManager | None,
    place_reuse_state: LocalmapPlaceReuseState | None,
    frontier_records: dict[str, LocalmapFrontierRecord],
    overlay_records: dict[str, LocalmapOverlayRecord],
    current_place_node_id: str,
    current_projection_obs_ids: list[str],
    reachable_candidates: list[FrontierCandidate],
    candidate_dedup_radius_m: float,
    current_projection_angle_to_obs_id: dict[int, str] | None = None,
) -> LocalmapFrontierUpdateResult:
    skip_frontier_update = (
        place_reuse_state is not None
        and not _reuse_should_register_frontiers(place_reuse_state)
    )
    result = LocalmapFrontierUpdateResult(skipped_for_reuse=skip_frontier_update)
    if not skip_frontier_update:
        if local_exploration is None:
            raise ValueError("localmap generation did not produce local_exploration")
        # Keep the records that existed before this place update.  They are
        # reusable for places whose frontier set did not change; the current
        # place may also install a freshly rendered record below, which is
        # merged into the reusable set after registration.
        previous_overlay_records = dict(overlay_records)
        (
            result.covered_frontier_ids_removed,
            result.place_coverage_pruned_frontier_debug,
        ) = _remove_old_frontiers_visible_from_new_place(
            graph=graph,
            local_exploration=local_exploration,
            frontier_records=frontier_records,
            place_node_id=str(current_place_node_id),
        )
        (
            result.stale_frontier_ids_removed,
            result.stale_frontier_pruned_debug,
        ) = _remove_stale_frontiers_without_unknown_patch(
            graph=graph,
            global_exploration=global_exploration,
            frontier_records=frontier_records,
        )
        first_stale_debug_by_frontier_id = {
            str(item["frontier_id"]): dict(item)
            for item in result.stale_frontier_pruned_debug
        }
        (
            result.filtered_candidates,
            result.local_candidate_filter_debug,
        ) = _filter_local_candidate_proposals(
            graph=graph,
            local_exploration=local_exploration,
            frontier_records=frontier_records,
            place_node_id=str(current_place_node_id),
            reachable_candidates=reachable_candidates,
            candidate_dedup_radius_m=float(candidate_dedup_radius_m),
        )
        result.created_registered_frontier_ids = _register_place_frontiers(
            graph=graph,
            cache=cache,
            local_exploration=local_exploration,
            frontier_records=frontier_records,
            overlay_records=overlay_records,
            place_node_id=str(current_place_node_id),
            projection_obs_ids=current_projection_obs_ids,
            projection_angle_to_obs_id=current_projection_angle_to_obs_id,
            filtered_candidates=result.filtered_candidates,
        )
        _annotate_registered_frontiers(
            graph=graph,
            frontier_records=frontier_records,
            current_place_node_id=str(current_place_node_id),
            local_candidate_filter_debug=result.local_candidate_filter_debug,
        )
        post_register_removed_ids, post_register_pruned_debug = (
            _remove_stale_frontiers_without_unknown_patch(
                graph=graph,
                global_exploration=global_exploration,
                frontier_records=frontier_records,
                reuse_debug_by_frontier_id=first_stale_debug_by_frontier_id,
                recheck_frontier_ids={
                    str(frontier_id)
                    for frontier_id in result.created_registered_frontier_ids
                },
            )
        )
        _append_global_pruned_frontiers(
            result=result,
            removed_frontier_ids=post_register_removed_ids,
            debug_records=post_register_pruned_debug,
        )
        if post_register_removed_ids != []:
            removed_set = {str(frontier_id) for frontier_id in post_register_removed_ids}
            result.created_registered_frontier_ids = [
                frontier_id
                for frontier_id in result.created_registered_frontier_ids
                if str(frontier_id) not in removed_set
            ]
        dirty_source_node_ids = _dirty_overlay_source_node_ids(
            removed_frontier_ids=[
                *result.covered_frontier_ids_removed,
                *result.stale_frontier_ids_removed,
            ],
            debug_records=[
                *result.place_coverage_pruned_frontier_debug,
                *result.stale_frontier_pruned_debug,
            ],
        )
        _sync_frontier_overlay_state(
            frontier_records=frontier_records,
            overlay_records=overlay_records,
        )
        # Prefer a freshly generated overlay for the current update, while
        # retaining old overlays for unchanged places.  The dirty set is
        # still passed through so callers of the helper remain conservative
        # when a source changed without installing a replacement overlay.
        reusable_overlay_records = dict(previous_overlay_records)
        reusable_overlay_records.update(overlay_records)
        _rebuild_active_overlay_records(
            cache=cache,
            graph=graph,
            frontier_records=frontier_records,
            overlay_records=overlay_records,
            floor_id=str(graph.get_node(str(current_place_node_id)).floor_id),
            reusable_overlay_records=reusable_overlay_records,
            dirty_source_node_ids=dirty_source_node_ids,
        )

    return result


def _dirty_overlay_source_node_ids(
    *,
    removed_frontier_ids: list[str],
    debug_records: list[dict[str, object]],
) -> set[str]:
    """Return source places whose active frontier set lost a record.

    The records are removed before the overlay rebuild, so the source node
    must be recovered from the diagnostic entries captured at removal time.
    """
    removed_ids = {str(frontier_id) for frontier_id in removed_frontier_ids}
    dirty_sources: set[str] = set()
    for item in debug_records:
        frontier_id = str(item["frontier_id"])
        if frontier_id not in removed_ids:
            continue
        dirty_sources.add(str(item["source_node_id"]))
    return dirty_sources


def _append_global_pruned_frontiers(
    *,
    result: LocalmapFrontierUpdateResult,
    removed_frontier_ids: list[str],
    debug_records: list[dict[str, object]],
) -> None:
    existing_removed = {str(frontier_id) for frontier_id in result.stale_frontier_ids_removed}
    result.stale_frontier_ids_removed.extend(
        str(frontier_id)
        for frontier_id in removed_frontier_ids
        if str(frontier_id) not in existing_removed
    )
    result.stale_frontier_ids_removed.sort(key=_frontier_id_sort_key)
    result.stale_frontier_pruned_debug.extend(debug_records)
    result.stale_frontier_pruned_debug.sort(
        key=lambda item: _frontier_id_sort_key(str(item["frontier_id"]))
    )


def _remove_stale_frontiers_without_unknown_patch(
    *,
    graph: Graph,
    global_exploration: ExplorationManager,
    frontier_records: dict[str, LocalmapFrontierRecord],
    reuse_debug_by_frontier_id: dict[str, dict[str, object]] | None = None,
    recheck_frontier_ids: set[str] | None = None,
) -> tuple[list[str], list[dict[str, object]]]:
    removed_frontier_ids: list[str] = []
    debug_records: list[dict[str, object]] = []
    recheck_ids = {str(frontier_id) for frontier_id in (recheck_frontier_ids or set())}
    for frontier_id, record in list(frontier_records.items()):
        frontier_id_text = str(frontier_id)
        if (
            reuse_debug_by_frontier_id is not None
            and frontier_id_text in reuse_debug_by_frontier_id
            and frontier_id_text not in recheck_ids
        ):
            # The global map is unchanged between the two scans in one place
            # update. Reuse the first scan's diagnostic record for survivors;
            # newly registered/replaced IDs are explicitly rechecked below.
            debug_records.append(dict(reuse_debug_by_frontier_id[frontier_id_text]))
            continue
        frontier_xy = (float(record.frontier_xy[0]), float(record.frontier_xy[1]))
        in_dilated_obstacle = _frontier_xy_in_dilated_obstacle(
            global_exploration=global_exploration,
            frontier_xy=frontier_xy,
        )
        patch_stats = _frontier_xy_unknown_patch_stats(
            global_exploration=global_exploration,
            frontier_xy=frontier_xy,
            radius_px=GLOBAL_STALE_FRONTIER_UNKNOWN_RADIUS_PX,
        )
        prune_reason = ""
        if in_dilated_obstacle:
            prune_reason = "frontier_xy_in_dilated_obstacle"
        elif bool(patch_stats["fully_known_patch"]):
            prune_reason = "frontier_patch_has_no_unknown"
        debug_record = {
            "frontier_id": frontier_id_text,
            "source_node_id": str(record.source_node_id),
            "source_place_index": int(record.source_place_index),
            "frontier_xy": [float(frontier_xy[0]), float(frontier_xy[1])],
            "unknown_radius_px": int(GLOBAL_STALE_FRONTIER_UNKNOWN_RADIUS_PX),
            "in_dilated_obstacle": bool(in_dilated_obstacle),
            "near_unknown_boundary": bool(patch_stats["near_unknown_boundary"]),
            "patch_in_bounds": bool(patch_stats["in_bounds"]),
            "patch_unknown_count": int(patch_stats["unknown_count"]),
            "patch_free_count": int(patch_stats["free_count"]),
            "fully_known_patch": bool(patch_stats["fully_known_patch"]),
            "pruned": prune_reason != "",
            "reason": prune_reason,
        }
        debug_records.append(debug_record)
        if prune_reason == "":
            continue
        _remove_frontier_record(
            frontier_records=frontier_records,
            frontier_id=frontier_id_text,
            graph=graph,
        )
        removed_frontier_ids.append(frontier_id_text)
    removed_frontier_ids.sort(key=_frontier_id_sort_key)
    debug_records.sort(key=lambda item: _frontier_id_sort_key(str(item["frontier_id"])))
    return removed_frontier_ids, debug_records


def _frontier_xy_in_dilated_obstacle(
    *,
    global_exploration: ExplorationManager,
    frontier_xy: tuple[float, float],
) -> bool:
    frontier_px = global_exploration.map.xy_to_px(
        np.asarray(frontier_xy, dtype=np.float64).reshape(1, 2)
    )[0].astype(np.int32)
    map_size = int(global_exploration.map.size)
    if (
        int(frontier_px[0]) < 0
        or int(frontier_px[0]) >= map_size
        or int(frontier_px[1]) < 0
        or int(frontier_px[1]) >= map_size
    ):
        return False
    dilated_obstacles = np.asarray(global_exploration.map.dilated_obstacles, dtype=bool)
    return bool(dilated_obstacles[int(frontier_px[1]), int(frontier_px[0])])


def _frontier_xy_unknown_patch_stats(
    *,
    global_exploration: ExplorationManager,
    frontier_xy: tuple[float, float],
    radius_px: int,
) -> dict[str, object]:
    frontier_px = global_exploration.map.xy_to_px(
        np.asarray(frontier_xy, dtype=np.float64).reshape(1, 2)
    )[0].astype(np.int32)
    map_size = int(global_exploration.map.size)
    if (
        int(frontier_px[0]) < 0
        or int(frontier_px[0]) >= map_size
        or int(frontier_px[1]) < 0
        or int(frontier_px[1]) >= map_size
    ):
        return {
            "in_bounds": False,
            "unknown_count": 0,
            "free_count": 0,
            "near_unknown_boundary": False,
            "fully_known_patch": False,
        }
    x = int(frontier_px[0])
    y = int(frontier_px[1])
    radius = int(radius_px)
    x0 = max(0, x - radius)
    x1 = min(map_size, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(map_size, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return {
            "in_bounds": False,
            "unknown_count": 0,
            "free_count": 0,
            "near_unknown_boundary": False,
            "fully_known_patch": False,
        }
    known_map = np.asarray(global_exploration.map.known_map, dtype=bool)
    free_map = np.asarray(global_exploration.map.free_map, dtype=bool)
    unknown_patch = ~known_map[y0:y1, x0:x1]
    free_patch = free_map[y0:y1, x0:x1]
    unknown_count = int(np.count_nonzero(unknown_patch))
    free_count = int(np.count_nonzero(free_patch))
    return {
        "in_bounds": True,
        "unknown_count": int(unknown_count),
        "free_count": int(free_count),
        "near_unknown_boundary": bool(unknown_count > 0 and free_count > 0),
        "fully_known_patch": bool(unknown_count == 0),
    }


def _reuse_should_register_frontiers(place_reuse_state: LocalmapPlaceReuseState | None) -> bool:
    if place_reuse_state is None:
        return False
    recovery = getattr(place_reuse_state, "recovery", {})
    return isinstance(recovery, dict) and bool(recovery.get("register_frontiers_on_reuse", False))


def _annotate_registered_frontiers(
    *,
    graph: Graph,
    frontier_records: dict[str, LocalmapFrontierRecord],
    current_place_node_id: str,
    local_candidate_filter_debug: list[dict[str, object]],
) -> None:
    place_node = graph.get_node(str(current_place_node_id))
    place_index = _place_index_by_node_id(graph, floor_id=str(place_node.floor_id)).get(str(current_place_node_id))
    if place_index is None:
        return
    for debug_record in local_candidate_filter_debug:
        if not bool(debug_record.get("accepted", False)):
            continue
        local_index = int(debug_record["filtered_candidate_index"])
        registered_frontier_id = f"{place_index}-{local_index}"
        record = frontier_records.get(registered_frontier_id)
        if record is None:
            debug_record["registered_frontier_id"] = None
            debug_record["registered_local_frontier_index"] = None
            debug_record["registered_overlay_id"] = None
            debug_record["registered_visible_in_rgb_overlay"] = False
            debug_record["registration_reject_reason"] = "not_visible_in_rgb_overlay"
            continue
        debug_record["registered_frontier_id"] = registered_frontier_id
        debug_record["registered_local_frontier_index"] = int(local_index)
        debug_record["registered_overlay_id"] = str(record.overlay_id)
        debug_record["registered_visible_in_rgb_overlay"] = True
        debug_record["registration_reject_reason"] = None

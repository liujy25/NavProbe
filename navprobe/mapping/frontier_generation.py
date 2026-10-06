from __future__ import annotations

import numpy as np

from navprobe.mapping.exploration.bev_map import FrontierCandidate, LocalFrontierBEVMap
from navprobe.mapping.exploration.manager import ExplorationManager
from navprobe.memory.graph.graph import Graph
from navprobe.llm.image_preprocessing import LLM_CAMERA_IMAGE_MAX_SIZE
from navprobe.mapping.frontier_registry import (
    _copy_frontier_candidate,
    _frontier_candidate_from_record,
    _frontier_id_sort_key,
    _register_frontier_record,
    _remove_frontier_record,
    _remove_owned_frontier_records,
    _sync_frontier_overlay_state,
)
from navprobe.logging.recording import _compose_contact_sheet
from navprobe.mapping.routing import _node_id_sort_key, _place_nodes
from navprobe.types import (
    LocalmapFrontierRecord,
    LocalmapOverlayRecord,
)
from navprobe.runtime.cache import RuntimeCache


LOCAL_CANDIDATE_MIN_DISTANCE_TO_OWNER_M = 0.3
LOCALMAP_FRONTIER_COVERAGE_MAX_DEPTH_RATIO = 0.8
LOCALMAP_FRONTIER_COVERAGE_MIN_KNOWN_FREE_RATIO = 0.9


def _distance_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _segment_known_free_enough(stats: dict[str, object] | None) -> bool:
    if stats is None:
        return False
    return (
        int(stats["obstacle_hit_count"]) == 0
        and float(stats["known_free_ratio"]) >= float(LOCALMAP_FRONTIER_COVERAGE_MIN_KNOWN_FREE_RATIO)
    )


def _place_index_by_node_id(graph: Graph, *, floor_id: str | None = None) -> dict[str, int]:
    places = (
        _place_nodes(graph)
        if floor_id is None
        else [node for node in graph.iter_nodes(floor_id=str(floor_id)) if node.node_kind == "place"]
    )
    ordered_places = sorted(places, key=lambda node: _node_id_sort_key(str(node.id)))
    return {
        str(node.id): index for index, node in enumerate(ordered_places)
    }


def _rebuild_active_overlay_records(
    *,
    cache: RuntimeCache,
    graph: Graph,
    frontier_records: dict[str, LocalmapFrontierRecord],
    overlay_records: dict[str, LocalmapOverlayRecord],
    floor_id: str,
    reusable_overlay_records: dict[str, LocalmapOverlayRecord] | None = None,
    dirty_source_node_ids: set[str] | None = None,
) -> None:
    previous_overlay_records = dict(
        overlay_records if reusable_overlay_records is None else reusable_overlay_records
    )
    reusable_by_source_node_id = {
        str(record.source_node_id): record
        for record in previous_overlay_records.values()
    }
    dirty_sources = {str(node_id) for node_id in (dirty_source_node_ids or set())}
    overlay_records.clear()
    place_index_by_node_id = _place_index_by_node_id(graph, floor_id=str(floor_id))
    active_records_by_place: dict[str, list[LocalmapFrontierRecord]] = {}
    for record in frontier_records.values():
        active_records_by_place.setdefault(str(record.source_node_id), []).append(record)

    for source_node_id, records in active_records_by_place.items():
        place_index = place_index_by_node_id.get(str(source_node_id))
        if place_index is None:
            for record in list(records):
                _remove_frontier_record(
                    frontier_records=frontier_records,
                    frontier_id=str(record.frontier_id),
                    graph=graph,
                )
            continue
        sorted_records = sorted(records, key=lambda item: int(item.local_frontier_index))
        reusable = reusable_by_source_node_id.get(str(source_node_id))
        active_frontier_ids = [str(record.frontier_id) for record in sorted_records]
        if (
            str(source_node_id) not in dirty_sources
            and reusable is not None
            and list(reusable.frontier_ids) == active_frontier_ids
            and str(reusable.image_id) in cache.images
            and all(str(image_id) in cache.images for image_id in reusable.view_overlay_ids)
        ):
            for record in sorted_records:
                record.overlay_id = str(reusable.overlay_id)
            overlay_records[str(reusable.overlay_id)] = reusable
            continue
        candidates = [_frontier_candidate_from_record(record) for record in sorted_records]
        frontier_label_map = {
            str(record.frontier_id): str(record.local_frontier_index)
            for record in sorted_records
        }
        source_obs_ids = [str(obs_id) for obs_id in sorted_records[0].source_obs_ids]
        source_node = graph.get_node(str(source_node_id))
        local_exploration = source_node.localmap
        if local_exploration is None:
            raise ValueError(f"missing local exploration map for source_node_id={source_node_id!r}")
        projection_base_z_m = float(graph.get_floor(str(source_node.floor_id)).height)
        final_views = local_exploration.render_frontier_annotated_views(
            cache=cache,
            obs_ids=source_obs_ids,
            candidates=candidates,
            projection_base_z_m=projection_base_z_m,
            frontier_label_map=frontier_label_map,
            image_max_size=LLM_CAMERA_IMAGE_MAX_SIZE,
        )
        visible_frontier_ids = {
            str(frontier_id)
            for view in final_views
            for frontier_id in view.frontier_ids
        }
        overlay_id = f"localmap_overlay_{place_index}"
        for record in list(sorted_records):
            if str(record.frontier_id) not in visible_frontier_ids:
                _remove_frontier_record(
                    frontier_records=frontier_records,
                    frontier_id=str(record.frontier_id),
                    graph=graph,
                )
                continue
            record.overlay_id = overlay_id
        if final_views == [] or visible_frontier_ids == set():
            continue
        overlay_image = _compose_contact_sheet(
            [cache.get_image(str(view.overlay_id)).image for view in final_views]
        )
        overlay_image_record = cache.store_image(overlay_image, kind="localmap_frontier_overlay")
        visible_overlay_frontier_ids = [
            str(record.frontier_id)
            for record in sorted_records
            if str(record.frontier_id) in visible_frontier_ids
        ]
        visible_overlay_local_ids = [
            str(record.local_frontier_index)
            for record in sorted_records
            if str(record.frontier_id) in visible_frontier_ids
        ]
        overlay_records[str(overlay_id)] = LocalmapOverlayRecord(
            overlay_id=overlay_id,
            image_id=str(overlay_image_record.id),
            source_node_id=str(source_node_id),
            source_place_index=int(place_index),
            frontier_ids=visible_overlay_frontier_ids,
            local_frontier_ids=visible_overlay_local_ids,
            view_overlay_ids=[str(view.overlay_id) for view in final_views],
            obs_ids=[str(view.obs_id) for view in final_views],
        )


def _remove_old_frontiers_visible_from_new_place(
    graph: Graph,
    local_exploration: ExplorationManager,
    frontier_records: dict[str, LocalmapFrontierRecord],
    place_node_id: str,
) -> tuple[list[str], list[dict[str, object]]]:
    place_node = graph.get_node(str(place_node_id))
    place_xy = (float(place_node.position[0]), float(place_node.position[1]))
    coverage_radius_m = (
        float(local_exploration.map.max_depth)
        * float(LOCALMAP_FRONTIER_COVERAGE_MAX_DEPTH_RATIO)
    )
    removed_frontier_ids: list[str] = []
    debug_records: list[dict[str, object]] = []
    for frontier_id, record in list(frontier_records.items()):
        frontier_xy = (float(record.frontier_xy[0]), float(record.frontier_xy[1]))
        distance_m = _distance_xy(place_xy, frontier_xy)
        debug_record: dict[str, object] = {
            "frontier_id": str(frontier_id),
            "source_node_id": str(record.source_node_id),
            "source_place_index": int(record.source_place_index),
            "new_place_node_id": str(place_node_id),
            "new_place_xy": [float(place_xy[0]), float(place_xy[1])],
            "frontier_xy": [float(frontier_xy[0]), float(frontier_xy[1])],
            "goal_xy": [float(record.goal_xy[0]), float(record.goal_xy[1])],
            "distance_to_new_place_m": float(distance_m),
            "coverage_radius_m": float(coverage_radius_m),
            "visibility_map_node_id": str(place_node_id),
            "pruned": False,
            "reason": "",
        }
        if str(record.source_node_id) == str(place_node_id):
            debug_record["reason"] = "same_source_place"
            debug_records.append(debug_record)
            continue
        if distance_m > coverage_radius_m:
            debug_record["reason"] = "out_of_coverage_radius"
            debug_records.append(debug_record)
            continue
        visibility_stats = local_exploration.map.segment_known_free_stats(
            start_xy=place_xy,
            end_xy=frontier_xy,
        )
        debug_record["visibility_line_in_bounds"] = visibility_stats is not None
        debug_record["visibility_min_known_free_ratio"] = float(
            LOCALMAP_FRONTIER_COVERAGE_MIN_KNOWN_FREE_RATIO
        )
        debug_record["visibility_sample_count"] = (
            None if visibility_stats is None else int(visibility_stats["sample_count"])
        )
        debug_record["visibility_obstacle_hit_count"] = (
            None if visibility_stats is None else int(visibility_stats["obstacle_hit_count"])
        )
        debug_record["visibility_known_free_count"] = (
            None if visibility_stats is None else int(visibility_stats["known_free_count"])
        )
        debug_record["visibility_known_free_ratio"] = (
            None if visibility_stats is None else float(visibility_stats["known_free_ratio"])
        )
        debug_record["visibility_known_free_enough"] = _segment_known_free_enough(visibility_stats)
        if visibility_stats is None:
            debug_record["reason"] = "line_out_of_bounds"
            debug_records.append(debug_record)
            continue
        if int(visibility_stats["obstacle_hit_count"]) > 0:
            debug_record["reason"] = "blocked_by_obstacle"
            debug_records.append(debug_record)
            continue
        if float(visibility_stats["known_free_ratio"]) < float(LOCALMAP_FRONTIER_COVERAGE_MIN_KNOWN_FREE_RATIO):
            debug_record["reason"] = "blocked_by_unknown"
            debug_records.append(debug_record)
            continue
        _remove_frontier_record(
            frontier_records=frontier_records,
            frontier_id=str(frontier_id),
            graph=graph,
        )
        removed_frontier_ids.append(str(frontier_id))
        debug_record["pruned"] = True
        debug_record["reason"] = "visible_from_new_place"
        debug_records.append(debug_record)
    removed_frontier_ids.sort(key=_frontier_id_sort_key)
    debug_records.sort(key=lambda item: _frontier_id_sort_key(str(item["frontier_id"])))
    return removed_frontier_ids, debug_records


def _build_local_candidate_proposals(
    cache: RuntimeCache,
    obs_ids: list[str],
    bev_map_kwargs: dict[str, object],
) -> tuple[ExplorationManager, list[FrontierCandidate]]:
    local_exploration = ExplorationManager(
        bev_map=LocalFrontierBEVMap(**dict(bev_map_kwargs)),
    )
    local_exploration.observe_observations(cache=cache, obs_ids=obs_ids)

    anchor_obs_id = str(obs_ids[-1])
    anchor_observation = cache.get_observation(anchor_obs_id).observation
    robot_xy = np.asarray([float(anchor_observation.pose.x), float(anchor_observation.pose.y)], dtype=np.float64)
    reachable_candidates = local_exploration.build_reachable_frontier_candidates(robot_xy)
    return local_exploration, reachable_candidates


def _filter_local_candidate_proposals(
    graph: Graph,
    local_exploration: ExplorationManager,
    frontier_records: dict[str, LocalmapFrontierRecord],
    place_node_id: str,
    reachable_candidates: list[FrontierCandidate],
    candidate_dedup_radius_m: float,
) -> tuple[list[FrontierCandidate], list[dict[str, object]]]:
    filtered_candidates: list[FrontierCandidate] = []
    debug_records: list[dict[str, object]] = []
    owner_node = graph.get_node(str(place_node_id))
    owner_floor_id = str(owner_node.floor_id)
    owner_xy = (float(owner_node.position[0]), float(owner_node.position[1]))
    coverage_radius_m = (
        float(local_exploration.map.max_depth)
        * float(LOCALMAP_FRONTIER_COVERAGE_MAX_DEPTH_RATIO)
    )
    other_places = [
        (str(node.id), (float(node.position[0]), float(node.position[1])))
        for node in graph.iter_nodes(floor_id=owner_floor_id)
        if node.node_kind == "place" and str(node.id) != str(place_node_id)
    ]
    for raw_index, candidate in enumerate(reachable_candidates):
        frontier_xy = (float(candidate.xy[0]), float(candidate.xy[1]))
        goal_xy = (float(candidate.goal_xy[0]), float(candidate.goal_xy[1]))
        owner_distance_m = _distance_xy(frontier_xy, owner_xy)
        debug_record: dict[str, object] = {
            "raw_candidate_index": int(raw_index),
            "raw_candidate_id": str(candidate.frontier_id),
            "source_node_id": str(place_node_id),
            "frontier_xy": [float(frontier_xy[0]), float(frontier_xy[1])],
            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "path_length": float(candidate.path_length),
            "accepted": False,
            "reject_reason": "",
            "owner_distance_m": float(owner_distance_m),
            "owner_min_distance_m": float(LOCAL_CANDIDATE_MIN_DISTANCE_TO_OWNER_M),
            "coverage_radius_m": float(coverage_radius_m),
            "candidate_dedup_radius_m": float(candidate_dedup_radius_m),
            "registered_frontier_id": None,
            "registered_local_frontier_index": None,
            "registered_overlay_id": None,
            "registered_visible_in_rgb_overlay": None,
        }
        if owner_distance_m <= float(LOCAL_CANDIDATE_MIN_DISTANCE_TO_OWNER_M):
            debug_record["reject_reason"] = "too_close_to_owner_place"
            debug_records.append(debug_record)
            continue

        visible_from_place: dict[str, object] | None = None
        visibility_checks: list[dict[str, object]] = []
        for node_id, node_xy in other_places:
            distance_m = _distance_xy(frontier_xy, node_xy)
            if distance_m > coverage_radius_m:
                continue
            visibility_exploration = graph.get_node(node_id).localmap
            if visibility_exploration is None:
                raise ValueError(f"missing local exploration map for place node {node_id!r}")
            visibility_stats = visibility_exploration.map.segment_known_free_stats(
                start_xy=node_xy,
                end_xy=frontier_xy,
            )
            known_free_enough = _segment_known_free_enough(visibility_stats)
            visibility_checks.append(
                {
                    "node_id": node_id,
                    "distance_m": float(distance_m),
                    "visibility_map_node_id": node_id,
                    "line_in_bounds": visibility_stats is not None,
                    "sample_count": None if visibility_stats is None else int(visibility_stats["sample_count"]),
                    "obstacle_hit_count": (
                        None if visibility_stats is None else int(visibility_stats["obstacle_hit_count"])
                    ),
                    "known_free_count": (
                        None if visibility_stats is None else int(visibility_stats["known_free_count"])
                    ),
                    "known_free_ratio": (
                        None if visibility_stats is None else float(visibility_stats["known_free_ratio"])
                    ),
                    "min_known_free_ratio": float(LOCALMAP_FRONTIER_COVERAGE_MIN_KNOWN_FREE_RATIO),
                    "obstacle_free": (
                        visibility_stats is not None
                        and int(visibility_stats["obstacle_hit_count"]) == 0
                    ),
                    "known_free_enough": bool(known_free_enough),
                }
            )
            if not known_free_enough:
                continue
            visible_from_place = {
                "node_id": node_id,
                "distance_m": float(distance_m),
                "visibility_map_node_id": node_id,
                "sample_count": int(visibility_stats["sample_count"]),
                "obstacle_hit_count": int(visibility_stats["obstacle_hit_count"]),
                "known_free_count": int(visibility_stats["known_free_count"]),
                "known_free_ratio": float(visibility_stats["known_free_ratio"]),
                "min_known_free_ratio": float(LOCALMAP_FRONTIER_COVERAGE_MIN_KNOWN_FREE_RATIO),
            }
            break
        if visibility_checks != []:
            debug_record["visibility_checks"] = visibility_checks
        if visible_from_place is not None:
            debug_record["reject_reason"] = "visible_from_existing_place"
            debug_record["visible_from_existing_place"] = visible_from_place
            debug_records.append(debug_record)
            continue

        nearest_frontier: dict[str, object] | None = None
        for frontier_id, record in frontier_records.items():
            existing_goal_xy = (float(record.goal_xy[0]), float(record.goal_xy[1]))
            distance_m = _distance_xy(goal_xy, existing_goal_xy)
            if distance_m > float(candidate_dedup_radius_m):
                continue
            if nearest_frontier is None or distance_m < float(nearest_frontier["distance_m"]):
                nearest_frontier = {
                    "frontier_id": str(frontier_id),
                    "source_node_id": str(record.source_node_id),
                    "distance_m": float(distance_m),
                }
        if nearest_frontier is not None:
            debug_record["reject_reason"] = "goal_too_close_to_existing_frontier"
            debug_record["nearest_frontier"] = nearest_frontier
            debug_records.append(debug_record)
            continue
        debug_record["accepted"] = True
        debug_record["reject_reason"] = None
        debug_record["filtered_candidate_index"] = int(len(filtered_candidates))
        debug_records.append(debug_record)
        filtered_candidates.append(candidate)
    return filtered_candidates, debug_records


def _register_place_frontiers(
    graph: Graph,
    cache: RuntimeCache,
    local_exploration: ExplorationManager,
    frontier_records: dict[str, LocalmapFrontierRecord],
    overlay_records: dict[str, LocalmapOverlayRecord],
    place_node_id: str,
    projection_obs_ids: list[str],
    filtered_candidates: list[FrontierCandidate],
    projection_angle_to_obs_id: dict[int, str] | None = None,
) -> list[str]:
    if filtered_candidates == []:
        return []
    place_node = graph.get_node(str(place_node_id))
    place_index = _place_index_by_node_id(graph, floor_id=str(place_node.floor_id)).get(str(place_node_id))
    if place_index is None:
        raise ValueError(f"missing place index for node_id={place_node_id!r}")

    final_candidates: list[FrontierCandidate] = []
    frontier_label_map: dict[str, str] = {}
    for local_index, candidate in enumerate(filtered_candidates):
        frontier_id = f"{place_index}-{local_index}"
        final_candidates.append(_copy_frontier_candidate(candidate, frontier_id=frontier_id))
        frontier_label_map[frontier_id] = str(local_index)

    final_views = local_exploration.render_frontier_annotated_views(
        cache=cache,
        obs_ids=list(projection_obs_ids),
        candidates=final_candidates,
        projection_base_z_m=float(graph.get_floor(str(place_node.floor_id)).height),
        frontier_label_map=frontier_label_map,
        image_max_size=LLM_CAMERA_IMAGE_MAX_SIZE,
    )
    visible_frontier_ids = {
        str(frontier_id)
        for view in final_views
        for frontier_id in view.frontier_ids
    }
    source_angle_degs_by_frontier_id = _source_angle_degs_by_frontier_id(
        final_views=final_views,
        angle_to_obs_id=projection_angle_to_obs_id,
    )
    overlay_id = ""
    if final_views != [] and visible_frontier_ids != set():
        overlay_image = _compose_contact_sheet(
            [cache.get_image(str(view.overlay_id)).image for view in final_views]
        )
        overlay_image_record = cache.store_image(overlay_image, kind="localmap_frontier_overlay")
        overlay_id = f"localmap_overlay_{place_index}"
        visible_overlay_frontier_ids = [
            str(candidate.frontier_id)
            for candidate in final_candidates
            if str(candidate.frontier_id) in visible_frontier_ids
        ]
        visible_overlay_local_ids = [
            frontier_label_map[frontier_id]
            for frontier_id in visible_overlay_frontier_ids
        ]
        overlay_record = LocalmapOverlayRecord(
            overlay_id=overlay_id,
            image_id=str(overlay_image_record.id),
            source_node_id=str(place_node_id),
            source_place_index=int(place_index),
            frontier_ids=visible_overlay_frontier_ids,
            local_frontier_ids=visible_overlay_local_ids,
            view_overlay_ids=[str(view.overlay_id) for view in final_views],
            obs_ids=[str(view.obs_id) for view in final_views],
        )
        overlay_records[str(overlay_id)] = overlay_record

    _remove_owned_frontier_records(
        frontier_records=frontier_records,
        source_node_id=str(place_node_id),
        graph=graph,
    )
    created_frontier_ids: list[str] = []
    for local_index, candidate in enumerate(final_candidates):
        if str(candidate.frontier_id) not in visible_frontier_ids:
            continue
        record = LocalmapFrontierRecord(
            frontier_id=str(candidate.frontier_id),
            source_node_id=str(place_node_id),
            source_place_index=int(place_index),
            local_frontier_index=int(local_index),
            overlay_id=str(overlay_id),
            frontier_xy=(float(candidate.xy[0]), float(candidate.xy[1])),
            goal_xy=(float(candidate.goal_xy[0]), float(candidate.goal_xy[1])),
            goal_yaw_deg=float(candidate.goal_yaw_degrees()),
            path_xy=[(float(x), float(y)) for x, y in candidate.path_xy],
            path_length=float(candidate.path_length),
            source_obs_ids=[str(obs_id) for obs_id in projection_obs_ids],
            source_angle_degs=source_angle_degs_by_frontier_id.get(str(candidate.frontier_id), []),
        )
        _register_frontier_record(graph=graph, frontier_records=frontier_records, record=record)
        created_frontier_ids.append(str(candidate.frontier_id))

    _sync_frontier_overlay_state(
        frontier_records=frontier_records,
        overlay_records=overlay_records,
    )
    return created_frontier_ids


def _source_angle_degs_by_frontier_id(
    *,
    final_views: list[object],
    angle_to_obs_id: dict[int, str] | None,
) -> dict[str, list[int]]:
    if not isinstance(angle_to_obs_id, dict) or angle_to_obs_id == {}:
        return {}
    obs_id_to_angle: dict[str, int] = {}
    for raw_angle, raw_obs_id in angle_to_obs_id.items():
        angle = int(raw_angle)
        obs_id = str(raw_obs_id).strip()
        if obs_id == "":
            continue
        obs_id_to_angle[obs_id] = angle
    frontier_angles: dict[str, set[int]] = {}
    for view in final_views:
        angle = obs_id_to_angle.get(str(view.obs_id))
        if angle is None:
            continue
        for frontier_id in view.frontier_ids:
            frontier_angles.setdefault(str(frontier_id), set()).add(int(angle))
    return {
        frontier_id: sorted(angles)
        for frontier_id, angles in frontier_angles.items()
    }

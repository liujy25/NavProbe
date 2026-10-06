from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from skimage.morphology import medial_axis

from navprobe.agent.visual_grounding import VisualWaypoint
from navprobe.mapping.exploration.bev_map import local_map_read_scope
from navprobe.mapping.exploration.bev_map import _GridPathSearch
from navprobe.mapping.exploration.bev_visuals import draw_numbered_circle_marker
from navprobe.mapping.exploration.bev_visuals import numbered_circle_marker_style
from navprobe.mapping.exploration.bev_visuals import place_node_marker_style
from navprobe.llm.image_preprocessing import LLM_CAMERA_IMAGE_MAX_SIZE
from navprobe.llm.image_preprocessing import resize_rgb_to_fit, resize_to_fit_metadata
from navprobe.llm.image_preprocessing import scale_xy
from navprobe.mapping.rgb_projection import project_visible_points
from navprobe.mapping.exploration.overlay_drawing import _clamp_frontier_circle_marker_center
from navprobe.mapping.exploration.overlay_drawing import _draw_frontier_circle_marker
from navprobe.visualization.action_mode_overlays import FRONTIER_MARKER_FILL
from navprobe.visualization.action_mode_overlays import BevOverlayTransform
from navprobe.visualization.action_mode_overlays import _render_scaled_overlay

if TYPE_CHECKING:
    from navprobe.agent.visual_action_context import VisualViewContext
    from navprobe.mapping.exploration.bev_map import GlobalBEVMap
    from navprobe.mapping.exploration.manager import ExplorationManager
    from navprobe.runtime.cache import RuntimeCache
    from navprobe.types import LocalmapFrontierRecord


def _normalized_projection_point(
    point_pixel: tuple[float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float]:
    """Convert a valid subpixel projection to the bounded visual schema.

    Projection validity uses half-open image bounds ``[0, width)``. The
    visual-action schema and its overlay helper use closed normalized bounds
    ``[0, 1]`` with ``width - 1`` as the pixel scale. A point in the final
    subpixel strip would otherwise serialize slightly above one and be rejected
    by the overlay logger. Clipping only affects that strip and keeps the
    existing pixel coordinate unchanged.
    """
    width_scale = float(max(1, int(image_width) - 1))
    height_scale = float(max(1, int(image_height) - 1))
    return (
        float(np.clip(float(point_pixel[0]) / width_scale, 0.0, 1.0)),
        float(np.clip(float(point_pixel[1]) / height_scale, 0.0, 1.0)),
    )


def _sampling_density_distance(
    limit_distance_m: float,
    sample_spacing_m: float,
) -> float:
    spacing = float(sample_spacing_m)
    if spacing <= 0.0:
        raise ValueError("VLN waypoint sample spacing must be positive")
    return min(float(limit_distance_m), spacing)


@dataclass(frozen=True)
class _NavProbeWaypointAnchor:
    xy: tuple[float, float]
    source: str
    score: float


@dataclass(frozen=True)
class NavProbeSampledWaypointCandidate:
    label: int
    goal_xy: tuple[float, float]
    path_xy: list[tuple[float, float]]
    point_pixel: tuple[float, float]
    point_2d: tuple[float, float]
    euclidean_distance_m: float
    path_length_m: float
    projected_depth_m: float

    def to_dict(self) -> dict[str, object]:
        return {
            "label": int(self.label),
            "goal_xy": [float(self.goal_xy[0]), float(self.goal_xy[1])],
            "path_xy": [[float(x), float(y)] for x, y in self.path_xy],
            "point_pixel": [float(self.point_pixel[0]), float(self.point_pixel[1])],
            "point_2d": [float(self.point_2d[0]), float(self.point_2d[1])],
            "euclidean_distance_m": float(self.euclidean_distance_m),
            "path_length_m": float(self.path_length_m),
            "projected_depth_m": float(self.projected_depth_m),
        }


@dataclass(frozen=True)
class NavProbeProjectedSampledWaypointCandidate:
    view: "VisualViewContext"
    candidate: NavProbeSampledWaypointCandidate
    source: str


def _project_waypoint_points(
    *,
    xy: np.ndarray,
    world_z: float,
    T_cam_odom: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    depth: np.ndarray,
    occlusion_tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(xy) == 0:
        return (
            np.zeros((0,), dtype=bool),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    z = np.full((len(xy), 1), float(world_z), dtype=np.float32)
    points_odom = np.concatenate([xy.astype(np.float32), z], axis=1)
    projected_points, projected_depths, visible = project_visible_points(
        points_odom=points_odom,
        T_cam_odom=T_cam_odom,
        intrinsics=intrinsics,
        image_width=image_width,
        image_height=image_height,
        depth_image=depth,
        depth_tolerance_m=float(occlusion_tolerance_m),
    )
    return visible, projected_points, projected_depths


def _nms_anchors(
    anchors: list[_NavProbeWaypointAnchor],
    *,
    min_distance_m: float,
    limit: int,
) -> list[_NavProbeWaypointAnchor]:
    selected: list[_NavProbeWaypointAnchor] = []
    for anchor in sorted(anchors, key=lambda item: (-float(item.score), str(item.source))):
        xy = np.asarray(anchor.xy, dtype=np.float64)
        if any(float(np.linalg.norm(xy - np.asarray(item.xy, dtype=np.float64))) < float(min_distance_m) for item in selected):
            continue
        selected.append(anchor)
        if len(selected) >= int(limit):
            break
    return selected


def _registered_frontier_anchors_unified(
    *,
    frontier_records: dict[str, "LocalmapFrontierRecord"] | None,
    robot_xy: np.ndarray,
    min_distance_m: float,
    anchor_spacing_m: float,
    limit: int,
) -> list[_NavProbeWaypointAnchor]:
    anchors: list[_NavProbeWaypointAnchor] = []
    for record in list((frontier_records or {}).values()):
        goal_xy = np.asarray(record.goal_xy, dtype=np.float64).reshape(2)
        if not bool(np.all(np.isfinite(goal_xy))):
            continue
        distance = float(np.linalg.norm(goal_xy - np.asarray(robot_xy, dtype=np.float64).reshape(2)))
        if distance < float(min_distance_m):
            continue
        anchors.append(
            _NavProbeWaypointAnchor(
                xy=(float(goal_xy[0]), float(goal_xy[1])),
                source="registered_frontier",
                score=1000.0,
            )
        )
    return _nms_anchors(
        anchors,
        min_distance_m=anchor_spacing_m,
        limit=limit,
    )


def _skeleton_anchors_unified(
    *,
    map_obj,
    robot_xy: np.ndarray,
    min_distance_m: float,
    anchor_spacing_m: float,
    limit: int,
    min_clearance_m: float,
) -> list[_NavProbeWaypointAnchor]:
    traversable = np.asarray(map_obj._traversable_map(), dtype=bool)
    traversable_ys, traversable_xs = np.nonzero(traversable)
    if len(traversable_xs) == 0:
        return []
    y_min = max(0, int(np.min(traversable_ys)) - 2)
    y_max = min(map_obj.size - 1, int(np.max(traversable_ys)) + 2)
    x_min = max(0, int(np.min(traversable_xs)) - 2)
    x_max = min(map_obj.size - 1, int(np.max(traversable_xs)) + 2)
    crop = traversable[y_min : y_max + 1, x_min : x_max + 1].copy()
    ys, xs = np.nonzero(crop)
    if len(xs) == 0:
        return []
    pixel_xy = np.stack([xs + x_min, ys + y_min], axis=1).astype(np.float64)
    xy = map_obj.px_to_xy(pixel_xy)
    distances = np.linalg.norm(xy - np.asarray(robot_xy, dtype=np.float64).reshape(1, 2), axis=1)
    keep = distances >= float(min_distance_m)
    filtered = np.zeros_like(crop, dtype=bool)
    filtered[ys[keep], xs[keep]] = True
    if not bool(np.any(filtered)):
        return []

    # Draw from the episode-seeded NumPy stream. The default
    # medial_axis RNG uses OS entropy and ignores np.random.seed().
    skeleton, distance = medial_axis(
        filtered, return_distance=True, rng=int(np.random.randint(0, 2**32, dtype=np.uint32)),
    )
    skeleton_ys, skeleton_xs = np.nonzero(skeleton)
    min_clearance_px = float(min_clearance_m) * float(map_obj.pixels_per_meter)
    anchors: list[_NavProbeWaypointAnchor] = []
    for x, y in zip(skeleton_xs, skeleton_ys):
        if float(distance[y, x]) < min_clearance_px:
            continue
        point_px = np.asarray([[x + x_min, y + y_min]], dtype=np.float64)
        point_xy = map_obj.px_to_xy(point_px)[0]
        anchors.append(
            _NavProbeWaypointAnchor(
                xy=(float(point_xy[0]), float(point_xy[1])),
                source="skeleton",
                score=500.0 + float(distance[y, x]) / float(map_obj.pixels_per_meter),
            )
        )
    return _nms_anchors(
        anchors,
        min_distance_m=anchor_spacing_m,
        limit=limit,
    )


def _path_length_m(path_xy: np.ndarray) -> float:
    path = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def _sample_path_point(path_xy: np.ndarray, distance_m: float) -> np.ndarray:
    path = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    if len(path) == 0:
        raise ValueError("path must not be empty")
    if len(path) == 1:
        return path[0]
    remaining = float(distance_m)
    for index in range(1, len(path)):
        start = path[index - 1]
        end = path[index]
        segment = end - start
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 1e-8:
            continue
        if remaining <= segment_length:
            return start + segment * (remaining / segment_length)
        remaining -= segment_length
    return path[-1]


def _path_prefix_to_distance(path_xy: np.ndarray, distance_m: float) -> np.ndarray:
    path = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    if len(path) <= 1:
        return path
    remaining = float(distance_m)
    pieces = [path[0]]
    for index in range(1, len(path)):
        start = path[index - 1]
        end = path[index]
        segment = end - start
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 1e-8:
            continue
        if remaining <= segment_length:
            pieces.append(start + segment * (remaining / segment_length))
            return np.asarray(pieces, dtype=np.float64)
        pieces.append(end)
        remaining -= segment_length
    return path


def _anchor_source_priority(source: str) -> int:
    if source == "registered_frontier":
        return 0
    if source == "skeleton":
        return 1
    return 9


def _is_near_avoid_node(
    xy: tuple[float, float] | np.ndarray,
    avoid_node_xys: list[tuple[float, float]] | None,
    radius_m: float,
    node_dedup_map: "GlobalBEVMap | None" = None,
) -> bool:
    nodes = [] if avoid_node_xys is None else list(avoid_node_xys)
    if nodes == []:
        return False
    point = np.asarray(xy, dtype=np.float64).reshape(2)
    for node_xy in nodes:
        node_point = np.asarray(node_xy, dtype=np.float64).reshape(2)
        distance_m = float(np.linalg.norm(point - node_point))
        if node_dedup_map is None:
            if distance_m < float(radius_m):
                return True
            continue
        if distance_m > float(radius_m):
            continue
        stats = node_dedup_map.segment_known_free_stats(
            start_xy=node_point,
            end_xy=point,
        )
        if stats is None:
            continue
        sample_count = int(stats["sample_count"])
        if (
            sample_count > 0
            and int(stats["known_free_count"]) == sample_count
            and int(stats["obstacle_hit_count"]) == 0
        ):
            return True
    return False


def _projection_view_score(
    *,
    point_pixel: tuple[float, float],
    image_width: int,
    image_height: int,
    projected_depth_m: float,
) -> float:
    x = float(point_pixel[0])
    y = float(point_pixel[1])
    border_margin = min(x, y, float(image_width - 1) - x, float(image_height - 1) - y)
    center_bonus = 1.0 - abs((x / max(1.0, float(image_width - 1))) - 0.5)
    lower_half_bonus = y / max(1.0, float(image_height - 1))
    depth_penalty = 0.01 * float(projected_depth_m)
    return float(border_margin * 0.01 + center_bonus + 0.5 * lower_half_bonus - depth_penalty)


@dataclass(frozen=True)
class _WaypointProjectionView:
    view: "VisualViewContext"
    image_height: int
    image_width: int
    depth: np.ndarray
    intrinsics: np.ndarray
    T_cam_odom: np.ndarray


class _WaypointProjector:
    """Projection metadata and results owned by one candidate-generation call."""

    def __init__(self, *, views, cache, world_z, occlusion_tolerance_m):
        self.views = views
        self.cache = cache
        self.world_z = float(world_z)
        self.occlusion_tolerance_m = float(occlusion_tolerance_m)
        self._results = {}

    @cached_property
    def prepared_views(self) -> list[_WaypointProjectionView]:
        prepared = []
        for view in self.views:
            observation = self.cache.get_observation(str(view.obs_id)).observation
            for name in ("rgb", "depth", "intrinsics", "T_cam_odom"):
                if getattr(observation, name) is None:
                    raise ValueError(f"unified sampled waypoint generation requires observation.{name}")
            height, width = np.asarray(observation.rgb).shape[:2]
            prepared.append(_WaypointProjectionView(
                view=view, image_height=int(height), image_width=int(width),
                depth=np.asarray(observation.depth),
                intrinsics=np.asarray(observation.intrinsics, dtype=np.float32),
                T_cam_odom=np.asarray(observation.T_cam_odom, dtype=np.float32),
            ))
        return prepared

    @cached_property
    def image_sizes(self) -> dict[str, tuple[int, int]]:
        return {str(item.view.obs_id): (item.image_height, item.image_width)
                for item in self.prepared_views}

    def project(self, waypoint_xy):
        # Use exact coordinates, not rounded bins: visibility at a boundary
        # must not be shared with a nearby but different waypoint.
        key = tuple(float(value) for value in waypoint_xy)
        if key not in self._results:
            self._results[key] = _projected_views_for_waypoint(
                waypoint_xy=waypoint_xy, projector=self,
            )
        return self._results[key]


def _projected_views_for_waypoint(
    *,
    waypoint_xy: np.ndarray,
    projector: _WaypointProjector,
) -> list[tuple[float, "VisualViewContext", tuple[float, float], float]]:
    projections: list[
        tuple[float, "VisualViewContext", tuple[float, float], float]
    ] = []
    for prepared in projector.prepared_views:
        view = prepared.view
        image_height, image_width = prepared.image_height, prepared.image_width
        visible, projected_points, projected_depths = _project_waypoint_points(
            xy=np.asarray(waypoint_xy, dtype=np.float64).reshape(1, 2),
            world_z=projector.world_z,
            T_cam_odom=prepared.T_cam_odom,
            intrinsics=prepared.intrinsics,
            image_width=int(image_width),
            image_height=int(image_height),
            depth=prepared.depth,
            occlusion_tolerance_m=projector.occlusion_tolerance_m,
        )
        if len(visible) == 0 or not bool(visible[0]):
            continue
        point_pixel = (float(projected_points[0, 0]), float(projected_points[0, 1]))
        projected_depth = float(projected_depths[0])
        score = _projection_view_score(
            point_pixel=point_pixel,
            image_width=int(image_width),
            image_height=int(image_height),
            projected_depth_m=projected_depth,
        )
        projections.append((score, view, point_pixel, projected_depth))
    return projections


def _best_projected_view_for_waypoint(
    *,
    waypoint_xy: np.ndarray,
    projector: _WaypointProjector,
) -> tuple["VisualViewContext", tuple[float, float], float] | None:
    projections = projector.project(waypoint_xy)
    if projections == []:
        return None
    _score, view, point_pixel, projected_depth = max(
        projections,
        key=lambda item: float(item[0]),
    )
    return view, point_pixel, projected_depth


def _candidate_from_anchor_best_view(
    *,
    map_obj,
    robot_xy: np.ndarray,
    anchor: _NavProbeWaypointAnchor,
    raw_label: int,
    cost_map: np.ndarray,
    projector: _WaypointProjector,
    min_distance_m: float,
    max_distance_m: float,
    path_search: _GridPathSearch,
    visible_backtrack_step_m: float,
) -> tuple[NavProbeSampledWaypointCandidate, "VisualViewContext"] | None:
    path_xy_arr = map_obj.compute_astar_path(
        robot_xy,
        np.asarray(anchor.xy, dtype=np.float64),
        cost_map=cost_map,
        path_search=path_search,
    )
    if path_xy_arr is None or len(path_xy_arr) < 2:
        return None
    path_length = _path_length_m(path_xy_arr)
    if path_length < float(min_distance_m):
        return None
    distance = min(path_length, float(max_distance_m))
    while distance >= float(min_distance_m):
        waypoint_xy = _sample_path_point(path_xy_arr, distance)
        best_projection = _best_projected_view_for_waypoint(
            waypoint_xy=waypoint_xy,
            projector=projector,
        )
        if best_projection is not None:
            view, point_pixel, projected_depth = best_projection
            image_height, image_width = projector.image_sizes[str(view.obs_id)]
            euclidean = float(np.linalg.norm(waypoint_xy - np.asarray(robot_xy, dtype=np.float64).reshape(2)))
            if euclidean < float(min_distance_m) or euclidean > float(max_distance_m):
                # A snapped start and curved paths separate path distance from
                # physical-pose distance. Continue the same bounded sampling
                # search for a point satisfying both distance and visibility.
                distance -= float(visible_backtrack_step_m)
                continue
            path_prefix = _path_prefix_to_distance(path_xy_arr, distance)
            candidate = NavProbeSampledWaypointCandidate(
                label=int(raw_label),
                goal_xy=(float(waypoint_xy[0]), float(waypoint_xy[1])),
                path_xy=[(float(item[0]), float(item[1])) for item in np.asarray(path_prefix, dtype=np.float64)],
                point_pixel=point_pixel,
                point_2d=_normalized_projection_point(
                    point_pixel,
                    image_width=image_width,
                    image_height=image_height,
                ),
                euclidean_distance_m=euclidean,
                path_length_m=float(distance),
                projected_depth_m=float(projected_depth),
            )
            return candidate, view
        distance -= float(visible_backtrack_step_m)
    return None


def generate_fss_waypoint_candidates(
    *,
    exploration: "ExplorationManager",
    cache: "RuntimeCache",
    views: list["VisualViewContext"],
    robot_xy: np.ndarray,
    world_z: float,
    frontier_records: dict[str, "LocalmapFrontierRecord"] | None = None,
    frontier_anchor_limit: int,
    skeleton_anchor_limit: int,
    combined_anchor_limit: int,
    skeleton_min_clearance_m: float,
    sample_spacing_m: float,
    min_distance_m: float,
    max_distance_m: float,
    max_candidates: int,
    occlusion_tolerance_m: float,
    anchor_nms_distance_m: float,
    anchor_combine_nms_distance_m: float,
    final_merge_distance_m: float,
    visible_backtrack_step_m: float,
    avoid_node_xys: list[tuple[float, float]] | None = None,
    node_dedup_radius_m: float,
    node_dedup_map: "GlobalBEVMap | None" = None,
) -> list[NavProbeProjectedSampledWaypointCandidate]:
    if views == []:
        raise ValueError("unified sampled waypoint generation requires at least one view")
    map_obj = exploration.map
    robot_xy = np.asarray(robot_xy, dtype=np.float64).reshape(2)
    anchor_spacing_m = _sampling_density_distance(
        anchor_nms_distance_m,
        sample_spacing_m,
    )
    combined_anchor_spacing_m = _sampling_density_distance(
        anchor_combine_nms_distance_m,
        sample_spacing_m,
    )
    final_merge_distance_m = _sampling_density_distance(
        final_merge_distance_m,
        sample_spacing_m,
    )
    anchors = _registered_frontier_anchors_unified(
        frontier_records=frontier_records,
        robot_xy=robot_xy,
        min_distance_m=min_distance_m,
        anchor_spacing_m=anchor_spacing_m,
        limit=frontier_anchor_limit,
    )
    anchors.extend(_skeleton_anchors_unified(
        map_obj=map_obj,
        robot_xy=robot_xy,
        min_distance_m=min_distance_m,
        anchor_spacing_m=anchor_spacing_m,
        limit=skeleton_anchor_limit,
        min_clearance_m=skeleton_min_clearance_m,
    ))
    anchors = _nms_anchors(
        anchors,
        min_distance_m=combined_anchor_spacing_m,
        limit=combined_anchor_limit,
    )
    if not anchors:
        return []
    cost_map = map_obj._build_astar_cost_map()
    path_search = _GridPathSearch(cost_map)
    projector = _WaypointProjector(
        views=views, cache=cache, world_z=world_z,
        occlusion_tolerance_m=occlusion_tolerance_m,
    )

    raw_candidates: list[tuple[NavProbeSampledWaypointCandidate, "VisualViewContext", str, float]] = []
    for raw_label, anchor in enumerate(anchors, start=1):
        projected = _candidate_from_anchor_best_view(
            map_obj=map_obj,
            robot_xy=robot_xy,
            anchor=anchor,
            raw_label=raw_label,
            cost_map=cost_map,
            path_search=path_search,
            visible_backtrack_step_m=visible_backtrack_step_m,
            projector=projector,
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
        )
        if projected is None:
            continue
        candidate, view = projected
        raw_candidates.append((candidate, view, str(anchor.source), float(anchor.score)))

    # The search workspace is much larger than the retained candidate paths.
    del path_search, cost_map

    selected: list[tuple[NavProbeSampledWaypointCandidate, "VisualViewContext", str, float]] = []
    for candidate, view, source, score in sorted(
        raw_candidates,
        key=lambda item: (
            _anchor_source_priority(item[2]),
            -float(item[3]),
            float(item[0].path_length_m),
            abs(float(item[0].point_2d[0]) - 0.5),
        ),
    ):
        goal_xy = np.asarray(candidate.goal_xy, dtype=np.float64)
        if _is_near_avoid_node(
            goal_xy,
            avoid_node_xys=avoid_node_xys,
            radius_m=float(node_dedup_radius_m),
            node_dedup_map=node_dedup_map,
        ):
            continue
        if any(
            float(np.linalg.norm(goal_xy - np.asarray(existing[0].goal_xy, dtype=np.float64)))
            < float(final_merge_distance_m)
            for existing in selected
        ):
            continue
        selected.append((candidate, view, source, score))
        if len(selected) >= int(max_candidates):
            break

    view_order = {int(view.angle_deg): index for index, view in enumerate(views)}
    selected = sorted(
        selected,
        key=lambda item: (
            view_order.get(int(item[1].angle_deg), 999),
            _candidate_label_sort_key(item[0]),
        ),
    )
    projected_selected: list[NavProbeProjectedSampledWaypointCandidate] = []
    for label, (candidate, _best_view, source, _score) in enumerate(
        selected,
        start=1,
    ):
        projections = projector.project(candidate.goal_xy)
        for _projection_score, view, point_pixel, projected_depth in projections:
            image_height, image_width = projector.image_sizes[str(view.obs_id)]
            projected_selected.append(
                NavProbeProjectedSampledWaypointCandidate(
                    view=view,
                    candidate=NavProbeSampledWaypointCandidate(
                        label=int(label),
                        goal_xy=candidate.goal_xy,
                        path_xy=candidate.path_xy,
                        point_pixel=point_pixel,
                        point_2d=_normalized_projection_point(
                            point_pixel,
                            image_width=image_width,
                            image_height=image_height,
                        ),
                        euclidean_distance_m=candidate.euclidean_distance_m,
                        path_length_m=candidate.path_length_m,
                        projected_depth_m=float(projected_depth),
                    ),
                    source=source,
                )
            )
    return sorted(
        projected_selected,
        key=lambda item: (
            view_order.get(int(item.view.angle_deg), 999),
            _candidate_label_sort_key(item.candidate),
        ),
    )


def draw_fss_waypoint_rgb_overlay(
    *,
    cache: "RuntimeCache",
    selected_view: "VisualViewContext",
    candidates: list[NavProbeSampledWaypointCandidate],
) -> np.ndarray:
    observation = cache.get_observation(str(selected_view.obs_id)).observation
    rgb = np.asarray(observation.rgb, dtype=np.uint8)
    node_overlay_image_id = str(selected_view.node_overlay_image_id).strip()
    if node_overlay_image_id != "":
        resize_metadata = resize_to_fit_metadata(rgb.shape, max_size=LLM_CAMERA_IMAGE_MAX_SIZE)
        render_rgb = np.asarray(cache.get_image(node_overlay_image_id).image, dtype=np.uint8)
    else:
        resized = resize_rgb_to_fit(rgb, max_size=LLM_CAMERA_IMAGE_MAX_SIZE)
        resize_metadata = resized.metadata
        render_rgb = np.asarray(resized.image, dtype=np.uint8)
    scale_x = float(resize_metadata["scale_x"])
    scale_y = float(resize_metadata["scale_y"])
    canvas = Image.fromarray(render_rgb.copy()).convert("RGBA")
    render_height, render_width = render_rgb.shape[:2]
    for candidate in candidates:
        label = str(candidate.label)
        scaled_xy = scale_xy(candidate.point_pixel, scale_x=scale_x, scale_y=scale_y)
        marker_xy = _clamp_frontier_circle_marker_center(
            center_xy=scaled_xy,
            image_size=(render_width, render_height),
        )
        _draw_frontier_circle_marker(
            center_xy=marker_xy,
            label=label,
            image=canvas,
        )
    return np.asarray(canvas.convert("RGB"), dtype=np.uint8)


def draw_fss_waypoint_bev_overlay(
    *,
    exploration: "ExplorationManager",
    robot_xy: np.ndarray,
    candidates: list[NavProbeSampledWaypointCandidate],
    landmark_markers: list[dict[str, object]] | None = None,
    base_overlay: np.ndarray | None = None,
    base_transform: BevOverlayTransform | None = None,
    coordinate_exploration: "ExplorationManager | None" = None,
    reference_node_marker: dict[str, object] | None = None,
) -> np.ndarray:
    if base_overlay is not None:
        if base_transform is None or coordinate_exploration is None:
            raise ValueError(
                "sampled waypoint base BEV requires its coordinate transform and exploration"
            )
        image = Image.fromarray(
            np.asarray(base_overlay, dtype=np.uint8).copy()
        ).convert("RGBA")
        image_width, image_height = int(image.size[0]), int(image.size[1])
        node_style = place_node_marker_style(
            image_width=image_width,
            image_height=image_height,
            mode="bev",
        )
        if reference_node_marker is not None:
            reference_xy = reference_node_marker.get("xy")
            if isinstance(reference_xy, (list, tuple)) and len(reference_xy) == 2:
                reference_px = coordinate_exploration.map.xy_to_px(
                    np.asarray(
                        [float(reference_xy[0]), float(reference_xy[1])],
                        dtype=np.float64,
                    ).reshape(1, 2)
                )[0]
                draw_numbered_circle_marker(
                    image,
                    center_xy=base_transform.transform_px(
                        (float(reference_px[0]), float(reference_px[1]))
                    ),
                    label=str(reference_node_marker.get("label", "")),
                    style=node_style,
                )
        candidate_style = numbered_circle_marker_style(
            image_width=image_width,
            image_height=image_height,
            mode="bev",
        )
        for candidate in candidates:
            goal_px = coordinate_exploration.map.xy_to_px(
                np.asarray(candidate.goal_xy, dtype=np.float64).reshape(1, 2)
            )[0]
            draw_numbered_circle_marker(
                image,
                center_xy=base_transform.transform_px(
                    (float(goal_px[0]), float(goal_px[1]))
                ),
                label=str(candidate.label),
                style=candidate_style,
            )
        rendered = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return rendered

    with local_map_read_scope(exploration.map):
        background = np.asarray(exploration.render_global_bev_background(), dtype=np.uint8)
        map_obj = exploration.map
        robot_px = map_obj.xy_to_px(np.asarray(robot_xy, dtype=np.float64).reshape(1, 2))[0]
        marker_specs: list[tuple[tuple[float, float], str, tuple[int, ...]]] = []
        for candidate in candidates:
            goal_px = map_obj.xy_to_px(np.asarray(candidate.goal_xy, dtype=np.float64).reshape(1, 2))[0]
            marker_specs.append(
                (
                    (float(goal_px[0]), float(goal_px[1])),
                    str(candidate.label),
                    FRONTIER_MARKER_FILL,
                )
            )
        landmark_marker_specs: list[tuple[tuple[float, float], str]] = []
        for marker in list(landmark_markers or []):
            xy = marker.get("xy")
            if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                continue
            landmark_px = map_obj.xy_to_px(
                np.asarray([float(xy[0]), float(xy[1])], dtype=np.float64).reshape(1, 2)
            )[0]
            landmark_marker_specs.append(
                (
                    (float(landmark_px[0]), float(landmark_px[1])),
                    str(marker.get("label", "")),
                )
            )
        rendered = _render_scaled_overlay(
            background=background,
            feasible_mask=np.asarray(map_obj.free_map, dtype=bool),
            marker_specs=marker_specs,
            robot_center=(float(robot_px[0]), float(robot_px[1])),
            landmark_marker_specs=landmark_marker_specs,
        )
        return rendered


def visual_waypoint_from_sampled_candidate(
    *,
    selected_view: "VisualViewContext",
    candidate: NavProbeSampledWaypointCandidate,
    target: str = "",
    raw_world_z: float = 0.0,
) -> VisualWaypoint:
    return VisualWaypoint(
        obs_id=str(selected_view.obs_id),
        angle_deg=int(selected_view.angle_deg),
        point_2d=(float(candidate.point_2d[0]), float(candidate.point_2d[1])),
        point_pixel=(float(candidate.point_pixel[0]), float(candidate.point_pixel[1])),
        raw_world_xy=(float(candidate.goal_xy[0]), float(candidate.goal_xy[1])),
        goal_xy=(float(candidate.goal_xy[0]), float(candidate.goal_xy[1])),
        goal_yaw=0.0,
        path_xy=[(float(x), float(y)) for x, y in candidate.path_xy],
        waypoint_target=str(target),
        target=str(target),
        depth_m=float(candidate.projected_depth_m),
        raw_world_z=float(raw_world_z),
    )


def _candidate_label_sort_key(candidate: NavProbeSampledWaypointCandidate) -> tuple[float, float, int]:
    return (
        float(candidate.point_2d[1]),
        float(candidate.point_2d[0]),
        int(candidate.label),
    )

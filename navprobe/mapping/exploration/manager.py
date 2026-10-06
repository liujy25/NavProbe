from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from PIL import ImageDraw

from navprobe.env.interface import RawObservation
from navprobe.memory.graph.graph import Graph
from navprobe.memory.graph.edge import Edge
from navprobe.llm.image_preprocessing import resize_rgb_to_fit, resize_to_fit_metadata
from navprobe.llm.image_preprocessing import scale_xy
from navprobe.runtime.cache import RuntimeCache
from navprobe.mapping.rgb_projection import _as_depth_meters, project_visible_points
from navprobe.visualization.trajectory import draw_rgb_trajectory
from navprobe.visualization.trajectory import edge_display_path_xyz

from navprobe.mapping.exploration.bev_map import local_map_read_scope
from navprobe.mapping.exploration.bev_map import FrontierCandidate, GlobalBEVMap
from navprobe.mapping.exploration.bev_visuals import (
    BEV_DILATED_OBSTACLE_COLOR,
    BEV_EXPLORED_FREE_COLOR,
    BEV_OBSTACLE_COLOR,
    BEV_UNKNOWN_BACKGROUND_COLOR,
    numbered_circle_marker_draw_center,
    place_node_marker_style,
)
from navprobe.mapping.exploration.frontier_buffer import FrontierBuffer
from navprobe.mapping.exploration.overlay_drawing import RGB_FRONTIER_BADGE_BACKOFF_M
from navprobe.mapping.exploration.overlay_drawing import _clamp_frontier_circle_marker_center
from navprobe.mapping.exploration.overlay_drawing import _draw_frontier_badge
from navprobe.mapping.exploration.overlay_drawing import _draw_frontier_circle_marker
from navprobe.mapping.exploration.overlay_drawing import _draw_place_node_circle
from navprobe.mapping.exploration.overlay_drawing import _frontier_label_text


RGB_OVERLAY_MAX_REMAINING_PATH_RATIO = 0.25
RGB_FRONTIER_PROJECTION_HEIGHTS_M = (0.03, 0.07, 0.11, 0.15)
RGB_PLACE_NODE_OCCLUSION_TOLERANCE_M = 0.10


def _node_id_sort_key(node_id: str) -> tuple[int, str]:
    stripped = str(node_id).strip()
    if stripped.startswith("n") and stripped[1:].isdigit():
        return (int(stripped[1:]), stripped)
    digits = "".join(char for char in stripped if char.isdigit())
    if digits != "":
        return (int(digits), stripped)
    return (10**9, stripped)


def _densify_path_xy(path_xy: list[tuple[float, float]], step_m: float = 0.05) -> np.ndarray:
    path = np.asarray(path_xy, dtype=np.float32)
    if len(path) <= 1:
        return path

    pieces: list[np.ndarray] = [path[0:1]]
    for index in range(1, len(path)):
        start = path[index - 1]
        end = path[index]
        delta = end - start
        distance = float(np.linalg.norm(delta))
        subdivisions = max(1, int(np.ceil(distance / step_m)))
        interpolation = np.linspace(start, end, subdivisions + 1, dtype=np.float32)[1:]
        pieces.append(interpolation)
    return np.concatenate(pieces, axis=0)


def _remaining_path_ratio_from_index(path_xy: np.ndarray, point_index: int) -> float:
    path = np.asarray(path_xy, dtype=np.float32)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError(f"path_xy must have shape Nx2, got {path.shape}")
    if len(path) <= 1:
        return 0.0
    clamped_index = int(np.clip(int(point_index), 0, len(path) - 1))
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total_length = float(segment_lengths.sum())
    if total_length <= 1e-6:
        return 0.0
    remaining_length = float(segment_lengths[clamped_index:].sum())
    return remaining_length / total_length


def _path_index_before_distance(
    path_xy: np.ndarray,
    visible_run: np.ndarray,
    anchor_index: int,
    distance_m: float,
) -> int:
    path = np.asarray(path_xy, dtype=np.float32)
    run = np.asarray(visible_run, dtype=np.int64)
    if len(run) == 0 or float(distance_m) <= 0.0:
        return int(anchor_index)
    start_index = int(run[0])
    current_index = int(np.clip(int(anchor_index), start_index, len(path) - 1))
    moved_m = 0.0
    while current_index > start_index and moved_m < float(distance_m):
        moved_m += float(np.linalg.norm(path[current_index] - path[current_index - 1]))
        current_index -= 1
    return int(current_index)


def _last_visible_run(valid_mask: np.ndarray) -> np.ndarray:
    visible_indices = np.flatnonzero(valid_mask)
    if len(visible_indices) == 0:
        return np.zeros((0,), dtype=np.int64)

    end_index = int(visible_indices[-1])
    start_index = end_index
    while start_index > 0 and bool(valid_mask[start_index - 1]):
        start_index -= 1
    return np.arange(start_index, end_index + 1, dtype=np.int64)


@dataclass
class ExploreView:
    obs_id: str
    overlay_id: str
    frontier_ids: list[str]
    frontier_sources: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "obs_id": self.obs_id,
            "overlay_id": self.overlay_id,
            "frontier_ids": list(self.frontier_ids),
            "frontier_sources": {
                str(frontier_id): str(source)
                for frontier_id, source in self.frontier_sources.items()
            },
        }


def project_visible_trajectory(
    *, observation, path_xyz: np.ndarray, image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Use the same camera and resize coordinates as the place-node overlay."""
    height, width = np.asarray(observation.rgb).shape[:2]
    projected, _depths, visible = project_visible_points(
        path_xyz,
        np.asarray(observation.T_cam_odom, dtype=np.float32),
        np.asarray(observation.intrinsics, dtype=np.float32),
        width, height, np.asarray(observation.depth),
        depth_tolerance_m=0.15,
    )
    scaled = projected.astype(np.float64) * [image_size[0] / width, image_size[1] / height]
    node_style = place_node_marker_style(
        image_width=image_size[0], image_height=image_size[1], mode="rgb",
    )
    for index in (0, len(scaled) - 1):
        if len(scaled) and visible[index]:
            scaled[index] = numbered_circle_marker_draw_center(
                tuple(scaled[index]), style=node_style, image_size=image_size,
            )
    return scaled, visible


@dataclass
class PlaceNodeOverlayView:
    obs_id: str
    overlay_id: str
    node_ids: list[str]
    edge_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "obs_id": str(self.obs_id),
            "overlay_id": str(self.overlay_id),
            "node_ids": [str(node_id) for node_id in self.node_ids],
            "edge_ids": [str(edge_id) for edge_id in self.edge_ids],
        }


@dataclass
class _ProjectedFrontierOverlay:
    obs_id: str
    frontier_id: str
    end_xy: tuple[float, float]
    center_distance_sq: float


@dataclass
class _ProjectedPlaceNodeOverlay:
    node_id: str
    end_xy: tuple[float, float]
    distance_m: float


class ExplorationManager:
    def __init__(self, bev_map: GlobalBEVMap) -> None:
        self.map = bev_map
        self.frontier_buffer = FrontierBuffer()
        self.integrated_obs_ids: set[str] = set()

    def reset(self) -> None:
        self.map.reset()
        self.frontier_buffer.clear()
        self.integrated_obs_ids = set()

    def observe_observation(self, cache: RuntimeCache, obs_id: str) -> dict[str, object]:
        if obs_id in self.integrated_obs_ids:
            return {
                "obs_id": obs_id,
                "integrated": False,
                "frontier_count": 0,
                "frontiers": [],
            }

        observation = cache.get_observation(obs_id).observation
        self.map.update_from_observation(observation)
        self.integrated_obs_ids.add(obs_id)
        return {
            "obs_id": obs_id,
            "integrated": True,
            "frontier_count": 0,
            "frontiers": [],
        }

    def observe_raw_observation(self, observation: RawObservation) -> None:
        self.map.update_from_observation(observation)

    def observe_observations(self, cache: RuntimeCache, obs_ids: list[str]) -> None:
        """Consume cached observations in order, retaining per-observation dedup."""
        def pending_observations():
            for obs_id in obs_ids:
                obs_id = str(obs_id)
                if obs_id in self.integrated_obs_ids:
                    continue
                yield cache.get_observation(obs_id).observation
                # The map requests the next item only after integrating this one.
                self.integrated_obs_ids.add(obs_id)

        self.map.update_from_observations(pending_observations())

    def merge_raw_layers_from_local_exploration(
        self,
        local_exploration: "ExplorationManager",
        *,
        obs_ids: list[str] | None = None,
    ) -> dict[str, object]:
        merge_method = getattr(self.map, "merge_raw_layers_from_local_map", None)
        if not callable(merge_method):
            raise TypeError(
                f"{type(self.map).__name__} does not support raw-layer localmap merge"
            )
        details = dict(merge_method(local_exploration.map))
        merged_obs_ids = [] if obs_ids is None else [str(obs_id) for obs_id in obs_ids]
        self.integrated_obs_ids.update(merged_obs_ids)
        details["merged_obs_count"] = int(len(merged_obs_ids))
        return details

    def refresh_frontiers(self) -> None:
        frontiers_xy = self.map.detect_global_frontiers()
        self.frontier_buffer.update(frontiers_xy)

    def build_reachable_frontier_candidates(
        self,
        robot_xy: np.ndarray,
    ) -> list[FrontierCandidate]:
        robot_xy = np.asarray(robot_xy, dtype=np.float64)
        self.refresh_frontiers()
        candidates = self.map.query_reachable_frontiers(
            robot_xy=robot_xy,
            frontier_entries=self.frontier_buffer.entries,
        )
        self.frontier_buffer.update_navigation_goals(candidates)
        return candidates


    def render_frontier_raw_overlay(
        self,
        robot_xy: np.ndarray,
    ) -> np.ndarray:
        with local_map_read_scope(self.map):
            robot_xy = np.asarray(robot_xy, dtype=np.float64).reshape(2)
            waypoints_px, raw_frontiers = self.map.detect_frontier_clusters_px()
            image = self._render_bev_background().copy()
            image[self.map.explored_area & self.map.navigable_map] = np.asarray(BEV_EXPLORED_FREE_COLOR, dtype=np.uint8)
            image[self.map.dilated_obstacles] = np.asarray(BEV_DILATED_OBSTACLE_COLOR, dtype=np.uint8)
            image[self.map.obstacle_map.astype(bool)] = np.asarray(BEV_OBSTACLE_COLOR, dtype=np.uint8)

            palette = [
                (255, 80, 80),
                (80, 200, 120),
                (80, 160, 255),
                (255, 180, 70),
                (220, 110, 255),
                (255, 255, 80),
                (80, 230, 230),
            ]
            frontier_entries = self.frontier_buffer.entries
            for index, frontier_px in enumerate(raw_frontiers):
                color = palette[index % len(palette)]
                frontier_px_int = np.asarray(frontier_px, dtype=np.int32)
                valid = (
                    (frontier_px_int[:, 0] >= 0)
                    & (frontier_px_int[:, 0] < self.map.size)
                    & (frontier_px_int[:, 1] >= 0)
                    & (frontier_px_int[:, 1] < self.map.size)
                )
                frontier_px_int = frontier_px_int[valid]
                if len(frontier_px_int) > 0:
                    image[frontier_px_int[:, 1], frontier_px_int[:, 0]] = np.asarray(color, dtype=np.uint8)

            rendered = Image.fromarray(image).convert("RGBA")
            draw = ImageDraw.Draw(rendered, "RGBA")
            robot_px = self.map.xy_to_px(robot_xy.reshape(1, 2))[0]
            draw.ellipse(
                (
                    float(robot_px[0]) - 8,
                    float(robot_px[1]) - 8,
                    float(robot_px[0]) + 8,
                    float(robot_px[1]) + 8,
                ),
                fill=(80, 160, 255),
                outline=(255, 255, 255),
                width=2,
            )
            for index, _frontier_px in enumerate(raw_frontiers):
                if index >= len(waypoints_px):
                    continue
                color = palette[index % len(palette)]
                frontier_id = f"raw_{index}"
                if index < len(frontier_entries):
                    frontier_id = str(frontier_entries[index].id)
                badge_center = (float(waypoints_px[index][0]), float(waypoints_px[index][1]))
                _draw_frontier_badge(badge_center, frontier_id, image=rendered)
            return np.asarray(rendered.convert("RGB"), dtype=np.uint8)

    def render_place_node_annotated_views(
        self,
        cache: RuntimeCache,
        graph: Graph,
        obs_ids: list[str],
        floor_id: str | None = None,
        image_max_size: tuple[int, int] | None = None,
        arrival_edge: Edge | None = None,
    ) -> list[PlaceNodeOverlayView]:
        floor_id_str = None if floor_id is None else str(floor_id)
        with graph.lock:
            place_nodes = [
                node
                for node in sorted(
                    graph.iter_nodes(floor_id=floor_id_str),
                    key=lambda item: _node_id_sort_key(str(item.id)),
                )
                if node.node_kind == "place"
            ]
        if place_nodes == []:
            return []
        node_points = np.asarray([node.position for node in place_nodes], dtype=np.float32)
        arrival_path = None
        if arrival_edge is not None and len(arrival_edge.path_xy) >= 2:
            arrival_path = edge_display_path_xyz(
                edge=arrival_edge,
                src_node=graph.get_node(arrival_edge.src_id),
                dst_node=graph.get_node(arrival_edge.dst_id),
            )

        views: list[PlaceNodeOverlayView] = []
        for obs_id in obs_ids:
            observation = cache.get_observation(str(obs_id)).observation
            if (
                observation.rgb is None
                or observation.depth is None
                or observation.intrinsics is None
                or observation.T_cam_odom is None
            ):
                continue

            rgb = np.asarray(observation.rgb, dtype=np.uint8)
            depth = _as_depth_meters(observation.depth)
            intrinsics = np.asarray(observation.intrinsics, dtype=np.float32)
            T_cam_odom = np.asarray(observation.T_cam_odom, dtype=np.float32)
            image_height, image_width = rgb.shape[:2]
            robot_xy = self._robot_xy_from_observation(observation)
            resize_metadata = resize_to_fit_metadata(rgb.shape, max_size=image_max_size)
            image_size = (int(resize_metadata["width"]), int(resize_metadata["height"]))
            scale_x = float(resize_metadata["scale_x"])
            scale_y = float(resize_metadata["scale_y"])

            edge_ids: list[str] = []
            if arrival_path is not None:
                projected_path, path_visible = project_visible_trajectory(
                    observation=observation, path_xyz=arrival_path, image_size=image_size,
                )
                # draw_rgb_trajectory draws precisely the contiguous runs of
                # at least two visible points. An isolated point draws nothing.
                if bool(np.any(path_visible[:-1] & path_visible[1:])):
                    edge_ids.append(str(arrival_edge.id))

            projected, _depths, unoccluded = project_visible_points(
                points_odom=node_points,
                T_cam_odom=T_cam_odom,
                intrinsics=intrinsics,
                image_width=image_width,
                image_height=image_height,
                depth_image=depth,
                depth_tolerance_m=RGB_PLACE_NODE_OCCLUSION_TOLERANCE_M,
            )
            projected_nodes: list[_ProjectedPlaceNodeOverlay] = []
            for index in np.flatnonzero(unoccluded):
                node = place_nodes[index]
                node_xy = (float(node.position[0]), float(node.position[1]))
                best_point_xy = (float(projected[index, 0]), float(projected[index, 1]))
                projected_nodes.append(
                    _ProjectedPlaceNodeOverlay(
                        node_id=str(node.id),
                        end_xy=best_point_xy,
                        distance_m=float(np.linalg.norm(np.asarray(node_xy, dtype=np.float64) - robot_xy)),
                    )
                )
            if projected_nodes == [] and edge_ids == []:
                continue

            resized = resize_rgb_to_fit(rgb, max_size=image_max_size)
            render_rgb = np.asarray(resized.image, dtype=np.uint8)
            image = Image.fromarray(render_rgb.copy()).convert("RGBA")
            if edge_ids:
                draw_rgb_trajectory(
                    draw=ImageDraw.Draw(image, "RGBA"), projected=projected_path, valid=path_visible,
                )

            for projected_node in sorted(
                projected_nodes,
                key=lambda item: (-float(item.distance_m), _node_id_sort_key(str(item.node_id))),
            ):
                _draw_place_node_circle(
                    center_xy=scale_xy(projected_node.end_xy, scale_x=scale_x, scale_y=scale_y),
                    label=projected_node.node_id,
                    image=image,
                )

            node_ids = sorted(
                {str(projected_node.node_id) for projected_node in projected_nodes},
                key=_node_id_sort_key,
            )
            overlay_record = cache.store_image(
                np.asarray(image.convert("RGB"), dtype=np.uint8),
                kind="place_node_overlay",
                metadata={
                    "obs_id": str(obs_id),
                    "floor_id": floor_id_str,
                    "node_ids": list(node_ids),
                    "edge_ids": list(edge_ids),
                },
            )
            views.append(
                PlaceNodeOverlayView(
                    obs_id=str(obs_id),
                    overlay_id=str(overlay_record.id),
                    node_ids=list(node_ids),
                    edge_ids=list(edge_ids),
                )
            )
        return views


    def render_global_bev_background(self) -> np.ndarray:
        return self._render_bev_background()

    def render_bev_graph_background_raw_obstacles(self) -> np.ndarray:
        return self._render_bev_background()

    def _render_bev_background(self) -> np.ndarray:
        with local_map_read_scope(self.map):
            base = np.full(
                (self.map.size, self.map.size, 3),
                BEV_UNKNOWN_BACKGROUND_COLOR,
                dtype=np.uint8,
            )
            base[self.map.explored_area & self.map.navigable_map] = np.asarray(BEV_EXPLORED_FREE_COLOR, dtype=np.uint8)
            base[self.map.dilated_obstacles] = np.asarray(BEV_DILATED_OBSTACLE_COLOR, dtype=np.uint8)
            base[self.map.obstacle_map] = np.asarray(BEV_OBSTACLE_COLOR, dtype=np.uint8)
            return base

    def _robot_xy_from_observation(self, observation) -> np.ndarray:
        if observation.T_odom_base is not None:
            T_odom_base = np.asarray(observation.T_odom_base, dtype=np.float64)
            return T_odom_base[:2, 3].copy()
        return np.asarray([observation.pose.x, observation.pose.y], dtype=np.float64)

    def render_frontier_annotated_views(
        self,
        cache: RuntimeCache,
        obs_ids: list[str],
        candidates: list[FrontierCandidate],
        projection_base_z_m: float,
        frontier_label_map: dict[str, str] | None = None,
        image_max_size: tuple[int, int] | None = None,
    ) -> list[ExploreView]:
        # Paths do not depend on the view. Keep only one densified path per
        # candidate, and retain only the best marker projection across views.
        dense_paths = [_densify_path_xy(candidate.path_xy) for candidate in candidates]
        best_by_frontier: dict[str, _ProjectedFrontierOverlay] = {}
        for obs_id in obs_ids:
            observation = cache.get_observation(obs_id).observation
            if observation.rgb is None:
                raise ValueError("explore requires observation.rgb")
            if observation.depth is None:
                raise ValueError("explore requires observation.depth")
            if observation.intrinsics is None:
                raise ValueError("explore requires observation.intrinsics")
            if observation.T_cam_odom is None:
                raise ValueError("explore requires observation.T_cam_odom")

            rgb = np.asarray(observation.rgb, dtype=np.uint8)
            depth = _as_depth_meters(observation.depth)
            intrinsics = np.asarray(observation.intrinsics, dtype=np.float32)
            T_cam_odom = np.asarray(observation.T_cam_odom, dtype=np.float32)
            image_height, image_width = rgb.shape[:2]
            image_center_x = (float(image_width) - 1.0) / 2.0
            image_center_y = (float(image_height) - 1.0) / 2.0

            for candidate, dense_path_xy in zip(candidates, dense_paths):
                selected_projected_path: np.ndarray | None = None
                selected_visible_run: np.ndarray | None = None
                selected_anchor_index: int | None = None
                for projection_height_m in RGB_FRONTIER_PROJECTION_HEIGHTS_M:
                    world_z = float(projection_base_z_m) + float(projection_height_m)
                    path_points = np.empty((len(dense_path_xy), 3), dtype=np.float32)
                    path_points[:, :2] = dense_path_xy.reshape(-1, 2)
                    path_points[:, 2] = world_z
                    projected_path, _depths, unoccluded_path = project_visible_points(
                        points_odom=path_points,
                        T_cam_odom=T_cam_odom,
                        intrinsics=intrinsics,
                        image_width=image_width,
                        image_height=image_height,
                        depth_image=depth,
                    )
                    visible_indices = np.flatnonzero(unoccluded_path)
                    if len(visible_indices) == 0:
                        continue

                    anchor_index = int(visible_indices[-1])
                    remaining_ratio = _remaining_path_ratio_from_index(
                        path_xy=dense_path_xy,
                        point_index=anchor_index,
                    )
                    if remaining_ratio >= RGB_OVERLAY_MAX_REMAINING_PATH_RATIO:
                        continue

                    visible_run = _last_visible_run(unoccluded_path)
                    if len(visible_run) == 0:
                        continue
                    selected_projected_path = np.asarray(projected_path, dtype=np.float32)
                    selected_visible_run = np.asarray(visible_run, dtype=np.int64)
                    selected_anchor_index = int(anchor_index)
                    break
                if selected_projected_path is None or selected_visible_run is None or selected_anchor_index is None:
                    continue

                badge_index = _path_index_before_distance(
                    path_xy=dense_path_xy,
                    visible_run=selected_visible_run,
                    anchor_index=selected_anchor_index,
                    distance_m=RGB_FRONTIER_BADGE_BACKOFF_M,
                )
                end_xy = (
                    float(selected_projected_path[badge_index][0]),
                    float(selected_projected_path[badge_index][1]),
                )
                center_distance_sq = float(
                    (float(end_xy[0]) - image_center_x) ** 2 + (float(end_xy[1]) - image_center_y) ** 2
                )
                frontier_id = str(candidate.frontier_id)
                current = best_by_frontier.get(frontier_id)
                if current is None or center_distance_sq < current.center_distance_sq:
                    best_by_frontier[frontier_id] = _ProjectedFrontierOverlay(
                        obs_id=str(obs_id), frontier_id=frontier_id,
                        end_xy=end_xy, center_distance_sq=center_distance_sq,
                    )

        selected_by_obs: dict[str, list[_ProjectedFrontierOverlay]] = {}
        for candidate in candidates:
            selected = best_by_frontier.get(str(candidate.frontier_id))
            if selected is not None:
                selected_by_obs.setdefault(selected.obs_id, []).append(selected)

        views: list[ExploreView] = []
        for obs_id, obs_frontiers in selected_by_obs.items():
            observation = cache.get_observation(obs_id).observation
            if observation.rgb is None:
                raise ValueError("explore requires observation.rgb")
            rgb = np.asarray(observation.rgb, dtype=np.uint8)
            image_height, image_width = rgb.shape[:2]
            resized = resize_rgb_to_fit(rgb, max_size=image_max_size)
            render_rgb = np.asarray(resized.image, dtype=np.uint8)
            render_height, render_width = render_rgb.shape[:2]
            scale_x = float(resized.metadata["scale_x"])
            scale_y = float(resized.metadata["scale_y"])
            image = Image.fromarray(render_rgb.copy()).convert("RGBA")
            visible_frontier_ids: list[str] = []
            frontier_sources: dict[str, str] = {}
            for projected in obs_frontiers:
                label_text = None
                if frontier_label_map is not None:
                    label_text = frontier_label_map.get(str(projected.frontier_id))
                label = _frontier_label_text(str(projected.frontier_id)) if label_text is None else str(label_text)
                scaled_end_xy = scale_xy(projected.end_xy, scale_x=scale_x, scale_y=scale_y)
                marker_xy = _clamp_frontier_circle_marker_center(
                    center_xy=scaled_end_xy,
                    image_size=(render_width, render_height),
                )
                _draw_frontier_circle_marker(
                    center_xy=marker_xy,
                    label=label,
                    image=image,
                )
                visible_frontier_ids.append(str(projected.frontier_id))
                frontier_sources[str(projected.frontier_id)] = "current"

            overlay_record = cache.store_image(np.asarray(image.convert("RGB"), dtype=np.uint8), kind="overlay")
            views.append(
                ExploreView(
                    obs_id=str(obs_id),
                    overlay_id=overlay_record.id,
                    frontier_ids=visible_frontier_ids,
                    frontier_sources=frontier_sources,
                )
            )
        return views

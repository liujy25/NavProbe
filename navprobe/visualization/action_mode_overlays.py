from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
import math
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from PIL import ImageDraw

from navprobe.mapping.exploration.bev_visuals import (
    BEV_PLACE_NODE_FILL,
    BEV_UNKNOWN_BACKGROUND_COLOR,
    draw_numbered_circle_marker,
    load_visual_font,
    numbered_circle_marker_style,
    place_node_marker_style,
)
from navprobe.types import LocalmapFrontierRecord
from navprobe.visualization.trajectory import edge_display_path_xy as _edge_display_path_xy

if TYPE_CHECKING:
    from navprobe.memory.graph.graph import Graph
    from navprobe.mapping.exploration.manager import ExplorationManager


ACTION_MODE_OVERLAY_CROP_MARGIN_PX = 28
ACTION_MODE_OVERLAY_TARGET_SHORT_SIDE_PX = 520
ACTION_MODE_OVERLAY_MIN_SCALE = 1.0
ACTION_MODE_OVERLAY_MAX_SCALE = 1.85
FRONTIER_MARKER_FILL = (255, 190, 30, 255)
MARKER_OUTLINE = (255, 255, 255, 255)
NODE_DISPLAY_DEDUP_RADIUS_PX = 18.0
NAVPROBE_TASK_STATE_MEMORY_EDGE_COLOR = (217, 70, 239, 255)
NAVPROBE_TASK_STATE_MEMORY_EDGE_WIDTH_PX = 2
NAVPROBE_TASK_STATE_MEMORY_NODE_FILL = (26, 112, 220, 230)
NAVPROBE_TASK_STATE_MEMORY_NODE_OUTLINE = (255, 255, 255, 255)
NAVPROBE_TASK_STATE_MEMORY_NODE_TEXT = (255, 255, 255, 255)
NAVPROBE_TASK_STATE_MEMORY_LANDMARK_FILL = (255, 255, 255, 255)
NAVPROBE_TASK_STATE_MEMORY_LANDMARK_OUTLINE = (35, 35, 35, 150)
NAVPROBE_TASK_STATE_MEMORY_LANDMARK_TEXT = (18, 18, 18, 255)


@dataclass(frozen=True)
class ActionModeOverlayLabelIndex:
    node_id_by_label: dict[int, str] = field(default_factory=dict)
    node_label_by_id: dict[str, int] = field(default_factory=dict)
    frontier_id_by_label: dict[int, str] = field(default_factory=dict)
    frontier_label_by_id: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id_by_label": {str(label): node_id for label, node_id in self.node_id_by_label.items()},
            "node_label_by_id": {node_id: int(label) for node_id, label in self.node_label_by_id.items()},
            "frontier_id_by_label": {
                str(label): frontier_id for label, frontier_id in self.frontier_id_by_label.items()
            },
            "frontier_label_by_id": {
                frontier_id: int(label) for frontier_id, label in self.frontier_label_by_id.items()
            },
            "node_label_range": _label_range_text(sorted(self.node_id_by_label)),
            "frontier_label_range": _label_range_text(sorted(self.frontier_id_by_label)),
        }


@dataclass(frozen=True)
class ActionModeOverlayRenderResult:
    image: np.ndarray
    labels: ActionModeOverlayLabelIndex


@dataclass(frozen=True)
class BevOverlayTransform:
    crop_x0: int
    crop_y0: int
    scale: float

    def transform_px(self, point: tuple[float, float]) -> tuple[float, float]:
        return (
            (float(point[0]) - float(self.crop_x0)) * float(self.scale),
            (float(point[1]) - float(self.crop_y0)) * float(self.scale),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "crop_x0": int(self.crop_x0),
            "crop_y0": int(self.crop_y0),
            "scale": float(self.scale),
        }


@dataclass(frozen=True)
class NavProbeTaskStateBevOverlayRenderResult:
    image: np.ndarray
    transform: BevOverlayTransform


def _action_overlay_background(
    *,
    global_exploration: "ExplorationManager",
) -> np.ndarray:
    background = np.asarray(
        global_exploration.render_global_bev_background(),
        dtype=np.uint8,
    )
    background = background.copy()
    dilated_only = np.asarray(
        global_exploration.map.dilated_obstacles,
        dtype=bool,
    ) & ~np.asarray(global_exploration.map.obstacle_map, dtype=bool)
    background[dilated_only] = np.asarray(
        BEV_UNKNOWN_BACKGROUND_COLOR,
        dtype=np.uint8,
    )
    return background


def render_explore_bev_frontier_overlay(
    *,
    graph: "Graph",
    global_exploration: "ExplorationManager",
    robot_xy: tuple[float, float] | np.ndarray,
    current_node_id: str,
    floor_id: str,
    frontier_records: dict[str, LocalmapFrontierRecord],
) -> ActionModeOverlayRenderResult:
    """Render a diagnostic BEV showing the robot, visited nodes, and frontier anchors."""
    background = _action_overlay_background(
        global_exploration=global_exploration,
    )
    index = _build_label_index(
        graph=graph,
        global_exploration=global_exploration,
        current_node_id=str(current_node_id),
        floor_id=str(floor_id),
        frontier_records=frontier_records,
    )
    marker_specs = _place_node_marker_specs(
        graph=graph,
        global_exploration=global_exploration,
        labels=index,
    )
    for label, frontier_id in sorted(index.frontier_id_by_label.items()):
        record = frontier_records[str(frontier_id)]
        frontier_center = _xy_to_px(global_exploration, record.goal_xy)
        marker_specs.append((frontier_center, str(label), FRONTIER_MARKER_FILL))
    robot_center = _xy_to_px(global_exploration, robot_xy)
    rendered = _render_scaled_overlay(
        background=background,
        feasible_mask=np.asarray(global_exploration.map.free_map, dtype=bool),
        marker_specs=marker_specs,
        robot_center=robot_center,
    )
    return ActionModeOverlayRenderResult(image=rendered, labels=index)


def render_task_state_bev_overlay_result(
    *,
    graph: "Graph",
    global_exploration: "ExplorationManager",
    floor_id: str,
    landmark_markers: list[dict[str, object]] | None = None,
    show_graph: bool = True,
    node_ids: set[str] | None = None,
    edge_ids: set[str] | None = None,
) -> NavProbeTaskStateBevOverlayRenderResult:
    """Render a BEV with selected graph nodes, edges, and landmark markers."""
    background = _action_overlay_background(
        global_exploration=global_exploration,
    )
    with graph.lock:
        all_floor_nodes = (
            list(graph.iter_nodes(floor_id=str(floor_id)))
            if show_graph
            else []
        )
        nodes = [
            node
            for node in all_floor_nodes
            if node_ids is None or str(node.id) in node_ids
        ]
        edges = (
            [
                edge
                for edge in graph.iter_edges(
                    floor_id=str(floor_id),
                    include_vertical=False,
                )
                if edge_ids is None or str(edge.id) in edge_ids
            ]
            if show_graph
            else []
        )
    nodes_by_id = {str(node.id): node for node in nodes}
    labels_by_id = _node_labels_by_id(all_floor_nodes)

    node_px_by_id: dict[str, tuple[float, float]] = {
        str(node.id): _xy_to_px(global_exploration, node.position[:2])
        for node in nodes
    }
    edge_paths_px: list[list[tuple[float, float]]] = []
    for edge in edges:
        src_id = str(edge.src_id)
        dst_id = str(edge.dst_id)
        if src_id not in nodes_by_id or dst_id not in nodes_by_id:
            continue
        path_xy = _edge_display_path_xy(
            edge=edge,
            src_xy=nodes_by_id[src_id].position[:2],
            dst_xy=nodes_by_id[dst_id].position[:2],
        )
        edge_paths_px.append([
            _xy_to_px(global_exploration, point)
            for point in path_xy
        ])

    landmark_marker_specs: list[tuple[tuple[float, float], str]] = []
    for marker in list(landmark_markers or []):
        if not isinstance(marker, dict):
            continue
        xy = marker.get("xy")
        if not isinstance(xy, (list, tuple)) or len(xy) != 2:
            continue
        center_px = _xy_to_px(global_exploration, (float(xy[0]), float(xy[1])))
        landmark_marker_specs.append((center_px, str(marker.get("label", ""))))
    crop_x0, crop_y0, crop_x1, crop_y1 = _crop_bbox(
        image=background,
        feasible_mask=np.asarray(global_exploration.map.free_map, dtype=bool),
    )
    cropped = np.asarray(background[crop_y0:crop_y1, crop_x0:crop_x1], dtype=np.uint8).copy()
    if cropped.size == 0:
        cropped = np.asarray(background, dtype=np.uint8).copy()
        crop_x0, crop_y0 = 0, 0
    scale = _overlay_scale(cropped)
    output_width = max(1, int(round(float(cropped.shape[1]) * scale)))
    output_height = max(1, int(round(float(cropped.shape[0]) * scale)))
    image = Image.fromarray(cropped).resize(
        (output_width, output_height),
        resample=Image.Resampling.NEAREST,
    ).convert("RGBA")
    base_size = (int(output_width), int(output_height))
    rotation_deg = None
    output_width, output_height = int(image.size[0]), int(image.size[1])
    draw = ImageDraw.Draw(image, "RGBA")

    for edge_path in edge_paths_px:
        if len(edge_path) < 2:
            continue
        transformed = [
            _transform_task_state_memory_px(
                point,
                crop_x0=crop_x0,
                crop_y0=crop_y0,
                scale=scale,
                base_size=base_size,
                output_size=(output_width, output_height),
                rotation_deg=rotation_deg,
            )
            for point in edge_path
        ]
        draw.line(
            transformed,
            fill=NAVPROBE_TASK_STATE_MEMORY_EDGE_COLOR,
            width=NAVPROBE_TASK_STATE_MEMORY_EDGE_WIDTH_PX,
            joint="curve",
        )

    marker_style = replace(
        numbered_circle_marker_style(
            image_width=output_width,
            image_height=output_height,
            mode="bev",
        ),
        fill=NAVPROBE_TASK_STATE_MEMORY_NODE_FILL,
        outline=NAVPROBE_TASK_STATE_MEMORY_NODE_OUTLINE,
        text_fill=NAVPROBE_TASK_STATE_MEMORY_NODE_TEXT,
    )
    for node in sorted(nodes, key=lambda item: _node_id_sort_key(str(item.id))):
        node_id = str(node.id)
        label = labels_by_id.get(node_id)
        if label is None:
            continue
        center = _transform_task_state_memory_px(
            node_px_by_id[node_id],
            crop_x0=crop_x0,
            crop_y0=crop_y0,
            scale=scale,
            base_size=base_size,
            output_size=(output_width, output_height),
            rotation_deg=rotation_deg,
        )
        draw_numbered_circle_marker(
            image,
            center_xy=center,
            label=str(label),
            style=marker_style,
        )
    landmark_style = replace(
        marker_style,
        fill=NAVPROBE_TASK_STATE_MEMORY_LANDMARK_FILL,
        outline=NAVPROBE_TASK_STATE_MEMORY_LANDMARK_OUTLINE,
        text_fill=NAVPROBE_TASK_STATE_MEMORY_LANDMARK_TEXT,
    )
    for point_px, label in sorted(landmark_marker_specs, key=lambda item: int(item[1])):
        center = _transform_task_state_memory_px(
            point_px,
            crop_x0=crop_x0,
            crop_y0=crop_y0,
            scale=scale,
            base_size=base_size,
            output_size=(output_width, output_height),
            rotation_deg=rotation_deg,
        )
        _draw_numbered_square_marker(
            image,
            center_xy=center,
            label=str(label),
            style=landmark_style,
        )
    return NavProbeTaskStateBevOverlayRenderResult(
        image=np.asarray(image.convert("RGB"), dtype=np.uint8),
        transform=BevOverlayTransform(
            crop_x0=int(crop_x0),
            crop_y0=int(crop_y0),
            scale=float(scale),
        ),
    )


def _build_label_index(
    *,
    graph: "Graph",
    global_exploration: "ExplorationManager",
    current_node_id: str,
    floor_id: str,
    frontier_records: dict[str, LocalmapFrontierRecord],
) -> ActionModeOverlayLabelIndex:
    with graph.lock:
        all_place_nodes = [
            node
            for node in graph.iter_nodes(floor_id=str(floor_id))
            if node.node_kind == "place"
        ]
    node_label_by_all_id = _node_labels_by_id(all_place_nodes)
    sorted_nodes = [
        node
        for node in sorted(all_place_nodes, key=lambda item: _node_id_sort_key(str(item.id)))
        if str(node.id) != str(current_node_id)
    ]
    required_node_ids = {
        str(record.source_node_id)
        for record in frontier_records.values()
        if str(record.source_node_id) != str(current_node_id)
    }
    display_nodes = _deduplicate_display_nodes(
        sorted_nodes=sorted_nodes,
        global_exploration=global_exploration,
        required_node_ids=required_node_ids,
    )
    node_id_by_label: dict[int, str] = {}
    node_label_by_id: dict[str, int] = {}
    for node in display_nodes:
        node_id = str(node.id)
        label = node_label_by_all_id.get(node_id)
        if label is None:
            continue
        node_id_by_label[int(label)] = node_id
        node_label_by_id[node_id] = int(label)

    frontier_start = 0 if node_label_by_all_id == {} else max(node_label_by_all_id.values()) + 1
    sorted_frontiers = sorted(
        frontier_records.values(),
        key=_frontier_record_sort_key,
    )
    frontier_id_by_label: dict[int, str] = {}
    frontier_label_by_id: dict[str, int] = {}
    for offset, record in enumerate(sorted_frontiers):
        label = int(frontier_start + offset)
        frontier_id = str(record.frontier_id)
        frontier_id_by_label[label] = frontier_id
        frontier_label_by_id[frontier_id] = label
    return ActionModeOverlayLabelIndex(
        node_id_by_label=node_id_by_label,
        node_label_by_id=node_label_by_id,
        frontier_id_by_label=frontier_id_by_label,
        frontier_label_by_id=frontier_label_by_id,
    )


def _node_labels_by_id(nodes: list[object]) -> dict[str, int]:
    sorted_nodes = sorted(nodes, key=lambda item: _node_id_sort_key(str(item.id)))
    node_ids = [str(node.id) for node in sorted_nodes]
    suffix_labels = [_node_id_numeric_suffix(node_id) for node_id in node_ids]
    if all(label is not None for label in suffix_labels) and len(set(suffix_labels)) == len(suffix_labels):
        return {node_id: int(label) for node_id, label in zip(node_ids, suffix_labels) if label is not None}
    return {node_id: int(index) for index, node_id in enumerate(node_ids)}


def _deduplicate_display_nodes(
    *,
    sorted_nodes: list[object],
    global_exploration: "ExplorationManager",
    required_node_ids: set[str],
) -> list[object]:
    selected: list[object] = []
    selected_px: list[np.ndarray] = []
    for node in sorted_nodes:
        node_id = str(node.id)
        px = np.asarray(_xy_to_px(global_exploration, node.position[:2]), dtype=np.float64)
        too_close = any(
            float(np.linalg.norm(px - previous_px)) < float(NODE_DISPLAY_DEDUP_RADIUS_PX)
            for previous_px in selected_px
        )
        if too_close and node_id not in required_node_ids:
            continue
        selected.append(node)
        selected_px.append(px)
    return selected


def _place_node_marker_specs(
    *,
    graph: "Graph",
    global_exploration: "ExplorationManager",
    labels: ActionModeOverlayLabelIndex,
) -> list[tuple[tuple[float, float], str, tuple[int, ...]]]:
    specs: list[tuple[tuple[float, float], str, tuple[int, ...]]] = []
    with graph.lock:
        nodes_by_id = {str(node.id): node for node in graph.iter_nodes()}
    for label, node_id in sorted(labels.node_id_by_label.items()):
        node = nodes_by_id[str(node_id)]
        center = _xy_to_px(global_exploration, node.position[:2])
        specs.append((center, str(label), BEV_PLACE_NODE_FILL))
    return specs


def _render_scaled_overlay(
    *,
    background: np.ndarray,
    feasible_mask: np.ndarray,
    marker_specs: list[tuple[tuple[float, float], str, tuple[int, ...]]],
    robot_center: tuple[float, float],
    landmark_marker_specs: list[tuple[tuple[float, float], str]] | None = None,
) -> np.ndarray:
    landmark_specs = list(landmark_marker_specs or [])
    crop_bbox = _crop_bbox(
        image=background,
        feasible_mask=feasible_mask,
    )
    crop_x0, crop_y0, crop_x1, crop_y1 = crop_bbox
    cropped = np.asarray(background[crop_y0:crop_y1, crop_x0:crop_x1], dtype=np.uint8).copy()
    if cropped.size == 0:
        cropped = np.asarray(background, dtype=np.uint8).copy()
        crop_x0, crop_y0 = 0, 0
    scale = _overlay_scale(cropped)
    output_width = max(1, int(round(float(cropped.shape[1]) * scale)))
    output_height = max(1, int(round(float(cropped.shape[0]) * scale)))
    image = Image.fromarray(cropped).resize(
        (output_width, output_height),
        resample=Image.Resampling.NEAREST,
    ).convert("RGBA")
    short_side = max(1, min(output_width, output_height))
    robot_size_px = int(np.clip(round(float(short_side) * 0.045), 14, 24))
    marker_style = numbered_circle_marker_style(
        image_width=output_width,
        image_height=output_height,
        mode="bev",
    )
    node_marker_style = place_node_marker_style(
        image_width=output_width,
        image_height=output_height,
        mode="bev",
    )
    for point, label, fill in marker_specs:
        center = _transform_px(point, crop_x0=crop_x0, crop_y0=crop_y0, scale=scale)
        draw_numbered_circle_marker(
            image,
            center_xy=center,
            label=label,
            style=node_marker_style if tuple(fill) == tuple(BEV_PLACE_NODE_FILL) else marker_style,
        )
    landmark_style = replace(
        marker_style,
        fill=NAVPROBE_TASK_STATE_MEMORY_LANDMARK_FILL,
        outline=NAVPROBE_TASK_STATE_MEMORY_LANDMARK_OUTLINE,
        text_fill=NAVPROBE_TASK_STATE_MEMORY_LANDMARK_TEXT,
    )
    for point, label in landmark_specs:
        center = _transform_px(point, crop_x0=crop_x0, crop_y0=crop_y0, scale=scale)
        _draw_numbered_square_marker(
            image,
            center_xy=center,
            label=str(label),
            style=landmark_style,
        )
    transformed_robot = _transform_px(robot_center, crop_x0=crop_x0, crop_y0=crop_y0, scale=scale)
    _draw_robot_marker(image=image, center_xy=transformed_robot, size_px=robot_size_px)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _crop_bbox(
    *,
    image: np.ndarray,
    feasible_mask: np.ndarray,
) -> tuple[int, int, int, int]:
    image_height, image_width = int(image.shape[0]), int(image.shape[1])
    feasible = np.asarray(feasible_mask, dtype=bool)
    if feasible.shape != (image_height, image_width):
        raise ValueError("crop feasible_mask shape must match image shape")
    if not np.any(feasible):
        return (0, 0, image_width, image_height)
    ys, xs = np.nonzero(feasible)
    margin = int(ACTION_MODE_OVERLAY_CROP_MARGIN_PX)
    min_x = max(0, int(np.min(xs)) - margin)
    max_x = min(image_width, int(np.max(xs)) + margin + 1)
    min_y = max(0, int(np.min(ys)) - margin)
    max_y = min(image_height, int(np.max(ys)) + margin + 1)
    if min_x >= max_x or min_y >= max_y:
        return (0, 0, image_width, image_height)
    return (min_x, min_y, max_x, max_y)


def _overlay_scale(image: np.ndarray) -> float:
    short_side = max(1, min(int(image.shape[0]), int(image.shape[1])))
    raw_scale = float(ACTION_MODE_OVERLAY_TARGET_SHORT_SIDE_PX) / float(short_side)
    return float(np.clip(raw_scale, ACTION_MODE_OVERLAY_MIN_SCALE, ACTION_MODE_OVERLAY_MAX_SCALE))


def _transform_px(
    px: tuple[float, float],
    *,
    crop_x0: int,
    crop_y0: int,
    scale: float,
) -> tuple[float, float]:
    return (
        (float(px[0]) - float(crop_x0)) * float(scale),
        (float(px[1]) - float(crop_y0)) * float(scale),
    )


def _transform_task_state_memory_px(
    px: tuple[float, float],
    *,
    crop_x0: int,
    crop_y0: int,
    scale: float,
    base_size: tuple[int, int],
    output_size: tuple[int, int],
    rotation_deg: float | None,
) -> tuple[float, float]:
    transformed = _transform_px(px, crop_x0=crop_x0, crop_y0=crop_y0, scale=scale)
    if rotation_deg is None:
        return transformed
    return _rotate_point_px(
        transformed,
        source_size=base_size,
        output_size=output_size,
        angle_deg=float(rotation_deg),
    )


def _rotate_point_px(
    point: tuple[float, float],
    *,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
    angle_deg: float,
) -> tuple[float, float]:
    source_width, source_height = int(source_size[0]), int(source_size[1])
    output_width, output_height = int(output_size[0]), int(output_size[1])
    theta = math.radians(float(angle_deg))
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    source_cx = (float(source_width) - 1.0) / 2.0
    source_cy = (float(source_height) - 1.0) / 2.0
    output_cx = (float(output_width) - 1.0) / 2.0
    output_cy = (float(output_height) - 1.0) / 2.0
    dx = float(point[0]) - source_cx
    dy = float(point[1]) - source_cy
    return (
        cos_theta * dx + sin_theta * dy + output_cx,
        -sin_theta * dx + cos_theta * dy + output_cy,
    )


def _draw_marker(
    *,
    draw: ImageDraw.ImageDraw,
    center_xy: tuple[float, float],
    fill: tuple[int, ...],
    radius_px: int,
) -> None:
    radius = float(radius_px)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=fill,
        outline=MARKER_OUTLINE,
        width=2,
    )


def _draw_numbered_square_marker(
    image: Image.Image,
    *,
    center_xy: tuple[float, float],
    label: str,
    style,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    radius = float(style.radius_px)
    cx = min(max(float(center_xy[0]), radius), max(radius, float(image.size[0]) - radius - 1.0))
    cy = min(max(float(center_xy[1]), radius), max(radius, float(image.size[1]) - radius - 1.0))
    box = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.rectangle(
        box,
        fill=style.fill,
        outline=style.outline,
        width=int(style.outline_width_px),
    )
    font = load_visual_font(int(style.font_size_px))
    draw.text((cx, cy), str(label), fill=style.text_fill, font=font, anchor="mm")


def _draw_robot_marker(
    *,
    image: Image.Image,
    center_xy: tuple[float, float],
    size_px: int,
) -> None:
    icon = _robot_icon(int(size_px))
    left = int(round(float(center_xy[0]) - float(icon.size[0]) / 2.0))
    top = int(round(float(center_xy[1]) - float(icon.size[1]) / 2.0))
    image.paste(icon, (left, top), icon)


@lru_cache(maxsize=4)
def _robot_icon(size_px: int) -> Image.Image:
    from pathlib import Path

    icon_path = Path(__file__).resolve().parents[1] / "assets" / "robot_icon.png"
    if not icon_path.exists():
        raise FileNotFoundError(f"BEV overlay icon does not exist: {icon_path}")
    return Image.open(icon_path).convert("RGBA").resize(
        (int(size_px), int(size_px)),
        resample=Image.Resampling.LANCZOS,
    )


def _xy_to_px(
    global_exploration: "ExplorationManager",
    xy: tuple[float, float] | np.ndarray,
) -> tuple[float, float]:
    point = np.asarray(xy, dtype=np.float64).reshape(1, 2)
    px = global_exploration.map.xy_to_px(point)[0]
    return (float(px[0]), float(px[1]))


def _node_id_sort_key(node_id: str) -> tuple[int, int, str]:
    suffix = _node_id_numeric_suffix(node_id)
    if suffix is not None:
        return (0, int(suffix), str(node_id))
    digits = "".join(char for char in str(node_id) if char.isdigit())
    if digits != "":
        return (1, int(digits), str(node_id))
    return (2, 10**9, str(node_id))


def _node_id_numeric_suffix(node_id: str) -> int | None:
    text = str(node_id).strip()
    if text.startswith("n") and text[1:].isdigit():
        return int(text[1:])
    return None


def _label_range_text(labels: list[int]) -> str:
    if labels == []:
        return "none"
    sorted_labels = sorted({int(label) for label in labels})
    ranges: list[str] = []
    start = sorted_labels[0]
    previous = sorted_labels[0]
    for label in sorted_labels[1:]:
        if int(label) == previous + 1:
            previous = int(label)
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = int(label)
        previous = int(label)
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _frontier_id_sort_key(frontier_id: str) -> tuple[int, int, str]:
    head, sep, tail = str(frontier_id).partition("-")
    if sep != "" and head.isdigit() and tail.isdigit():
        return (int(head), int(tail), str(frontier_id))
    return (10**9, 10**9, str(frontier_id))


def _frontier_record_sort_key(record: LocalmapFrontierRecord) -> tuple[tuple[int, int, str], int, tuple[int, int, str]]:
    return (
        _node_id_sort_key(str(record.source_node_id)),
        int(record.local_frontier_index),
        _frontier_id_sort_key(str(record.frontier_id)),
    )

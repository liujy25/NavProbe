from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
from PIL import ImageDraw

from navprobe.mapping.exploration.bev_visuals import (
    BEV_GRAPH_EDGE_COLOR,
    draw_numbered_circle_marker,
    numbered_circle_marker_style,
    place_node_marker_style,
)
from navprobe.mapping.routing import _node_id_sort_key
from navprobe.visualization.graph_payload import graph_payload_edges


def _normalize_rgb(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError("rgb image must be HxWxC")
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        return np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.clip(array, 0, 255).astype(np.uint8)


def _graph_payload_nodes(
    graph_payload: dict[str, Any],
    floor_id: str | None = None,
) -> list[dict[str, Any]]:
    floor_id_str = None if floor_id is None else str(floor_id)
    raw_nodes = graph_payload.get("nodes")
    if isinstance(raw_nodes, list):
        nodes = [node for node in raw_nodes if isinstance(node, dict)]
        if floor_id_str is None:
            return nodes
        return [node for node in nodes if str(node.get("floor_id", "")) == floor_id_str]
    raw_floors = graph_payload.get("floors")
    if not isinstance(raw_floors, list):
        return []
    nodes: list[dict[str, Any]] = []
    for floor in raw_floors:
        if not isinstance(floor, dict):
            continue
        floor_id = str(floor.get("id", ""))
        if floor_id_str is not None and floor_id != floor_id_str:
            continue
        raw_floor_nodes = floor.get("nodes", [])
        if not isinstance(raw_floor_nodes, list):
            continue
        for node in raw_floor_nodes:
            if not isinstance(node, dict):
                continue
            node_payload = dict(node)
            if floor_id != "" and "floor_id" not in node_payload:
                node_payload["floor_id"] = floor_id
            nodes.append(node_payload)
    return nodes


def _render_bev_graph_image(
    graph_payload: dict[str, Any],
    bev_background: np.ndarray,
    xy_to_px,
    explored_mask: np.ndarray | None = None,
    point_markers: list[dict[str, Any]] | None = None,
    crop: bool = True,
    floor_id: str | None = None,
) -> np.ndarray:
    image = Image.fromarray(np.asarray(bev_background, dtype=np.uint8).copy()).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    marker_style = numbered_circle_marker_style(
        image_width=int(image.size[0]),
        image_height=int(image.size[1]),
        mode="bev",
    )
    node_marker_style = place_node_marker_style(
        image_width=int(image.size[0]),
        image_height=int(image.size[1]),
        mode="bev",
    )

    nodes = _graph_payload_nodes(graph_payload, floor_id=floor_id)
    markers = [] if point_markers is None else list(point_markers)
    if nodes == [] and markers == []:
        draw.text((40, 40), "empty graph", fill=(0, 0, 0))
        return np.asarray(image, dtype=np.uint8)

    node_positions: dict[str, tuple[float, float]] = {}
    for node in nodes:
        node_id = str(node["id"])
        position_xy = np.asarray([[float(node["position"][0]), float(node["position"][1])]], dtype=np.float64)
        node_px = xy_to_px(position_xy)[0]
        node_positions[node_id] = (float(node_px[0]), float(node_px[1]))

    def draw_node_badge(
        px: float,
        py: float,
        label: str,
    ) -> None:
        draw_numbered_circle_marker(
            image,
            (float(px), float(py)),
            str(label),
            style=node_marker_style,
        )

    crop_points: list[np.ndarray] = []
    if explored_mask is not None and np.any(explored_mask):
        explored_y, explored_x = np.nonzero(explored_mask)
        crop_points.append(np.stack([explored_x, explored_y], axis=1).astype(np.float64))

    edges = graph_payload_edges(graph_payload, include_vertical=False, floor_id=floor_id)
    for edge in edges:
        src_id = str(edge.get("src_id", ""))
        dst_id = str(edge.get("dst_id", ""))
        if src_id not in node_positions or dst_id not in node_positions:
            continue
        src_xy = node_positions[src_id]
        dst_xy = node_positions[dst_id]
        draw.line([src_xy, dst_xy], fill=BEV_GRAPH_EDGE_COLOR, width=6)
        crop_points.append(np.asarray([[src_xy[0], src_xy[1]], [dst_xy[0], dst_xy[1]]], dtype=np.float64))

    ordered_nodes = sorted(nodes, key=lambda node: _node_id_sort_key(str(node["id"])))
    for index, node in enumerate(ordered_nodes):
        node_id = str(node["id"])
        px, py = node_positions[node_id]
        draw_node_badge(
            px=px,
            py=py,
            label=str(index),
        )
        crop_points.append(np.asarray([[px, py]], dtype=np.float64))

    for marker in markers:
        x = float(marker["x"])
        y = float(marker["y"])
        label = str(marker.get("label", ""))
        color_raw = marker.get("color", (255, 0, 255))
        if not isinstance(color_raw, (list, tuple)) or len(color_raw) != 3:
            raise ValueError(f"point marker color must be a length-3 list/tuple, got {color_raw!r}")
        color = (int(color_raw[0]), int(color_raw[1]), int(color_raw[2]))
        radius = int(marker.get("radius", 10))
        marker_px_array = xy_to_px(np.asarray([[x, y]], dtype=np.float64))[0]
        marker_px = (float(marker_px_array[0]), float(marker_px_array[1]))
        if label.isdigit() and len(label) <= 2:
            draw_numbered_circle_marker(
                image,
                marker_px,
                label,
                style=marker_style,
            )
            crop_points.append(np.asarray([[marker_px[0], marker_px[1]]], dtype=np.float64))
            continue
        draw.ellipse(
            (
                marker_px[0] - radius,
                marker_px[1] - radius,
                marker_px[0] + radius,
                marker_px[1] + radius,
            ),
            fill=color,
            outline=(255, 255, 255),
            width=2,
        )
        if label != "":
            draw.text(
                (marker_px[0] + radius + 4.0, marker_px[1] - radius - 4.0),
                label,
                fill=(255, 255, 255),
            )
        crop_points.append(np.asarray([[marker_px[0], marker_px[1]]], dtype=np.float64))

    rendered = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if crop_points == [] or not bool(crop):
        return rendered

    stacked = np.concatenate(crop_points, axis=0)
    margin = 120
    min_x = max(0, int(np.floor(np.min(stacked[:, 0]))) - margin)
    max_x = min(rendered.shape[1], int(np.ceil(np.max(stacked[:, 0]))) + margin + 1)
    min_y = max(0, int(np.floor(np.min(stacked[:, 1]))) - margin)
    max_y = min(rendered.shape[0], int(np.ceil(np.max(stacked[:, 1]))) + margin + 1)
    return rendered[min_y:max_y, min_x:max_x]


@dataclass
class CacheSnapshot:
    observation_ids: set[str]
    detection_ids: set[str]


@dataclass
class StepArtifactObservationGroups:
    panorama_obs_ids: set[str]
    step_obs_ids: set[str]

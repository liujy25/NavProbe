from __future__ import annotations

import numpy as np
from PIL import Image

from navprobe.mapping.exploration.bev_visuals import (
    clamp_numbered_circle_marker_center,
    draw_numbered_circle_marker,
    numbered_circle_marker_style,
    place_node_marker_style,
)


RGB_FRONTIER_BADGE_BACKOFF_M = 0.05
BEV_SELECTION_CROP_MARGIN_PX = 120


def _frontier_label_text(frontier_id: str) -> str:
    if frontier_id.startswith("f") and frontier_id[1:].isdigit():
        return frontier_id[1:]
    return frontier_id


def _draw_frontier_badge(
    center_xy: tuple[float, float],
    frontier_id: str,
    *,
    image: Image.Image,
) -> None:
    draw_numbered_circle_marker(
        image,
        center_xy,
        _frontier_label_text(frontier_id),
        style=numbered_circle_marker_style(
            image_width=int(image.size[0]),
            image_height=int(image.size[1]),
            mode="bev",
        ),
    )


def _clamp_frontier_circle_marker_center(
    *,
    center_xy: tuple[float, float],
    image_size: tuple[int, int],
) -> tuple[float, float]:
    style = numbered_circle_marker_style(
        image_width=int(image_size[0]),
        image_height=int(image_size[1]),
        mode="rgb",
    )
    return clamp_numbered_circle_marker_center(
        center_xy=center_xy,
        style=style,
        image_size=image_size,
    )


def _draw_frontier_circle_marker(
    *,
    center_xy: tuple[float, float],
    label: str,
    image: Image.Image,
) -> None:
    draw_numbered_circle_marker(
        image,
        center_xy,
        str(label),
        style=numbered_circle_marker_style(
            image_width=int(image.size[0]),
            image_height=int(image.size[1]),
            mode="rgb",
        ),
    )


def _crop_bev_overlay(
    image: np.ndarray,
    explored_mask: np.ndarray | None,
    extra_points_px: list[tuple[float, float]] | None = None,
    margin_px: int = BEV_SELECTION_CROP_MARGIN_PX,
) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    crop_points: list[np.ndarray] = []
    if explored_mask is not None:
        explored = np.asarray(explored_mask, dtype=bool)
        if explored.shape != (image_height, image_width):
            raise ValueError("crop explored_mask shape must match image shape")
        if np.any(explored):
            explored_y, explored_x = np.nonzero(explored)
            crop_points.append(np.stack([explored_x, explored_y], axis=1).astype(np.float64))
    if extra_points_px is not None and extra_points_px != []:
        crop_points.append(np.asarray(extra_points_px, dtype=np.float64).reshape(-1, 2))
    if crop_points == []:
        return image
    stacked = np.concatenate(crop_points, axis=0)
    min_x = max(0, int(np.floor(np.min(stacked[:, 0]))) - int(margin_px))
    max_x = min(image.shape[1], int(np.ceil(np.max(stacked[:, 0]))) + int(margin_px) + 1)
    min_y = max(0, int(np.floor(np.min(stacked[:, 1]))) - int(margin_px))
    max_y = min(image.shape[0], int(np.ceil(np.max(stacked[:, 1]))) + int(margin_px) + 1)
    return image[min_y:max_y, min_x:max_x]


def _draw_place_node_circle(
    *,
    center_xy: tuple[float, float],
    label: str,
    image: Image.Image,
) -> None:
    draw_numbered_circle_marker(
        image,
        center_xy,
        _place_node_display_label(label),
        style=place_node_marker_style(
            image_width=int(image.size[0]),
            image_height=int(image.size[1]),
            mode="rgb",
        ),
    )


def _place_node_display_label(label: object) -> str:
    text = str(label).strip()
    if len(text) >= 2 and text[0].lower() == "n" and text[1:].isdigit():
        return text[1:]
    return text

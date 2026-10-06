from __future__ import annotations

from typing import Any

import numpy as np

from navprobe.visualization.rendering import _normalize_rgb


def draw_waypoint_overlay_rgb(
    image: Any,
    *,
    point_pixel: tuple[float, float],
) -> np.ndarray:
    canvas = _normalize_rgb(image).copy()
    height, width = int(canvas.shape[0]), int(canvas.shape[1])
    cx = int(round(float(point_pixel[0])))
    cy = int(round(float(point_pixel[1])))
    cx = max(0, min(width - 1, cx))
    cy = max(0, min(height - 1, cy))
    radius = max(5, int(round(float(min(width, height)) * 0.018)))
    outline_radius = radius + 2
    yy, xx = np.ogrid[:height, :width]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    canvas[dist2 <= outline_radius**2] = np.array([0, 0, 0], dtype=np.uint8)
    canvas[dist2 <= radius**2] = np.array([255, 0, 0], dtype=np.uint8)
    return np.asarray(canvas, dtype=np.uint8)

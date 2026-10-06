"""Shared world-to-RGB projection and depth visibility for points and paths."""
from __future__ import annotations

import numpy as np


def _as_depth_meters(depth: np.ndarray) -> np.ndarray:
    depth_array = np.asarray(depth)
    if np.issubdtype(depth_array.dtype, np.integer):
        depth_array = depth_array.astype(np.float32)
        if float(np.nanmax(depth_array)) > 100.0:
            depth_array = depth_array / 1000.0
        return depth_array
    return depth_array.astype(np.float32, copy=False)


def _project_points(
    points_odom: np.ndarray,
    T_cam_odom: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points_odom) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )

    homo_points = np.empty((points_odom.shape[0], 4), dtype=np.float32)
    homo_points[:, :3] = points_odom
    homo_points[:, 3] = 1.0
    points_cam = (T_cam_odom.astype(np.float32, copy=False) @ homo_points.T).T[:, :3]
    del homo_points
    # A view here would keep all four transformed coordinates alive after
    # returning, even though callers only need the camera depth.
    depths = points_cam[:, 2].copy()
    valid = depths > 1e-3

    projected = np.zeros((points_cam.shape[0], 2), dtype=np.float32)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    safe_depths = np.clip(depths, 1e-6, None)
    projected[:, 0] = fx * points_cam[:, 0] / safe_depths + cx
    projected[:, 1] = fy * points_cam[:, 1] / safe_depths + cy
    del points_cam, safe_depths

    valid &= projected[:, 0] >= 0.0
    valid &= projected[:, 0] < float(image_width)
    valid &= projected[:, 1] >= 0.0
    valid &= projected[:, 1] < float(image_height)
    return projected, depths.astype(np.float32, copy=False), valid


def _compute_unoccluded_mask(
    projected_points: np.ndarray,
    projected_depths: np.ndarray,
    valid_projection: np.ndarray,
    depth_image: np.ndarray,
    depth_tolerance_m: float = 0.0,
) -> np.ndarray:
    """Compare with the finite positive median of each clipped 3x3 window.

    Only sample requested pixels; bounded batches avoid a full-frame median
    map or an unbounded path-by-window temporary. Border samples outside the
    image are excluded, rather than replicated into the median.
    """
    unoccluded = np.zeros((len(projected_points),), dtype=bool)
    valid_indices = np.flatnonzero(valid_projection)
    if len(valid_indices) == 0:
        return unoccluded

    depth_m = _as_depth_meters(depth_image)
    image_height, image_width = depth_m.shape[:2]
    offsets = np.array([-1, 0, 1], dtype=np.int64)
    for start in range(0, len(valid_indices), 4096):
        indices = valid_indices[start:start + 4096]
        pixels = np.rint(projected_points[indices]).astype(np.int64)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, image_width - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, image_height - 1)
        xs = pixels[:, 0, None] + offsets
        ys = pixels[:, 1, None] + offsets
        inside = ((ys[:, :, None] >= 0) & (ys[:, :, None] < image_height)
                  & (xs[:, None, :] >= 0) & (xs[:, None, :] < image_width))
        windows = depth_m[
            np.clip(ys, 0, image_height - 1)[:, :, None],
            np.clip(xs, 0, image_width - 1)[:, None, :],
        ].reshape(-1, 9)
        usable = inside.reshape(-1, 9) & np.isfinite(windows) & (windows > 1e-3)
        counts = usable.sum(axis=1)
        windows[~usable] = np.inf
        windows.sort(axis=1)
        rows = np.flatnonzero(counts)
        counts = counts[rows]
        medians = windows[rows, counts // 2].copy()
        even = counts % 2 == 0
        medians[even] = (
            windows[rows[even], counts[even] // 2 - 1] + medians[even]
        ) / np.float32(2.0)
        # Compare depth and tolerance in float64 to retain boundary precision.
        unoccluded[indices[rows]] = (
            projected_depths[indices[rows]].astype(np.float64)
            <= medians.astype(np.float64) + float(depth_tolerance_m)
        )
    return unoccluded


def project_visible_points(
    points_odom: np.ndarray,
    T_cam_odom: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    depth_image: np.ndarray,
    depth_tolerance_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pixel coordinates, camera depths and the visibility mask."""
    projected, depths, valid = _project_points(
        points_odom, T_cam_odom, intrinsics, image_width, image_height,
    )
    return projected, depths, _compute_unoccluded_mask(
        projected, depths, valid, depth_image, depth_tolerance_m,
    )

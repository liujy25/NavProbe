"""RGB-D stair geometry for the support-graph controller."""

from __future__ import annotations

import numpy as np


def _backproject(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    T_cam_odom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    ys, xs = np.indices((height, width), dtype=np.float32)
    points_cam = np.stack(
        [
            (xs - float(intrinsics[0, 2])) * depth / float(intrinsics[0, 0]),
            (ys - float(intrinsics[1, 2])) * depth / float(intrinsics[1, 1]),
            depth,
        ],
        axis=-1,
    )
    T_odom_cam = np.linalg.inv(T_cam_odom)
    points_odom = points_cam @ T_odom_cam[:3, :3].T + T_odom_cam[:3, 3]
    return points_cam, points_odom


def _horizontal_support_mask(
    points_cam: np.ndarray,
    T_cam_odom: np.ndarray,
    valid_depth: np.ndarray,
) -> np.ndarray:
    delta_x = points_cam[:, 2:, :] - points_cam[:, :-2, :]
    delta_y = points_cam[2:, :, :] - points_cam[:-2, :, :]
    normals_cam = np.cross(delta_x[1:-1, :, :], delta_y[:, 1:-1, :])
    normal_norm = np.linalg.norm(normals_cam, axis=2)
    normals_odom = normals_cam @ np.linalg.inv(T_cam_odom)[:3, :3].T
    horizontal = np.zeros(valid_depth.shape, dtype=bool)
    horizontal[1:-1, 1:-1] = (
        (normal_norm > 1e-6)
        & (
            np.abs(normals_odom[:, :, 2]) / np.clip(normal_norm, 1e-6, None)
            >= np.cos(np.deg2rad(35.0))
        )
        & valid_depth[1:-1, 1:-1]
        & valid_depth[1:-1, :-2]
        & valid_depth[1:-1, 2:]
        & valid_depth[:-2, 1:-1]
        & valid_depth[2:, 1:-1]
        & (np.linalg.norm(delta_x[1:-1, :, :], axis=2) < 0.30)
        & (np.linalg.norm(delta_y[:, 1:-1, :], axis=2) < 0.30)
    )
    return horizontal

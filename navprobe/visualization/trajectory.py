"""Display-only trajectory geometry and RGB styling; never used for execution."""

from __future__ import annotations

import numpy as np
from PIL import ImageDraw


RGB_EXECUTED_PATH_COLOR = (20, 124, 255, 255)


def densify_path(path_xy, step_m: float = 0.05) -> np.ndarray:
    path = np.asarray(path_xy, dtype=np.float32)
    if len(path) <= 1:
        return path.copy()
    pieces = [path[0:1]]
    for start, end in zip(path[:-1], path[1:]):
        delta = end - start
        subdivisions = max(1, int(np.ceil(float(np.linalg.norm(delta)) / step_m)))
        interpolation = np.linspace(0, 1, subdivisions + 1, dtype=np.float32)[1:]
        pieces.append(start + interpolation[:, None] * delta)
    return np.concatenate(pieces, axis=0)


def smooth_display_path(path_xy) -> np.ndarray:
    """Round corners in world coordinates, keeping both recorded endpoints fixed."""
    path = np.asarray(path_xy, dtype=np.float32)
    if len(path) < 3:
        return path.copy()
    path = path[np.r_[True, np.any(np.diff(path, axis=0) != 0, axis=1)]]
    if len(path) < 3:
        return path.copy()
    # Limit corner rounding even when the recorded trajectory has sparse samples.
    rounded = densify_path(path, step_m=0.25)
    for _ in range(3):
        quarter = 0.75 * rounded[:-1] + 0.25 * rounded[1:]
        three_quarter = 0.25 * rounded[:-1] + 0.75 * rounded[1:]
        rounded = np.vstack([
            path[0],
            np.stack([quarter, three_quarter], axis=1).reshape(-1, path.shape[1]),
            path[-1],
        ])
    return rounded


def edge_display_path_xy(*, edge, src_xy, dst_xy) -> list[tuple[float, float]]:
    path = [(float(x), float(y)) for x, y in edge.path_xy]
    src, dst = tuple(src_xy[:2]), tuple(dst_xy[:2])
    if not path:
        return [src, dst]
    if not np.allclose(path[0], src, rtol=0, atol=1e-4):
        path.insert(0, src)
    if not np.allclose(path[-1], dst, rtol=0, atol=1e-4):
        path.append(dst)
    path[0], path[-1] = src, dst
    return [tuple(point) for point in smooth_display_path(path)]


def interpolated_edge_path_z(*, path_xy, src_z: float, dst_z: float) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    cumulative = np.r_[np.float32(0), np.cumsum(lengths, dtype=np.float32)]
    if float(cumulative[-1]) <= 1e-6:
        return np.full(len(path_xy), src_z, dtype=np.float32)
    return np.asarray(src_z + cumulative / cumulative[-1] * (dst_z - src_z), dtype=np.float32)


def edge_display_path_xyz(*, edge, src_node, dst_node) -> np.ndarray:
    path = edge_display_path_xy(
        edge=edge, src_xy=src_node.position[:2], dst_xy=dst_node.position[:2],
    )
    xy = densify_path(path, step_m=0.01)
    z = interpolated_edge_path_z(
        path_xy=xy, src_z=src_node.position[2], dst_z=dst_node.position[2],
    )
    xyz = np.column_stack([xy, z]).astype(np.float32)
    # Match the node renderer exactly, including Z; no display-height offset.
    xyz[0], xyz[-1] = src_node.position, dst_node.position
    return xyz


def contiguous_projected_runs(*, projected, valid) -> list[list[tuple[float, float]]]:
    indices = np.flatnonzero(valid)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    return [[tuple(point) for point in projected[run]] for run in runs if len(run) >= 2]


def draw_rgb_trajectory(*, draw: ImageDraw.ImageDraw, projected, valid) -> bool:
    """Draw visible runs in travel order, with blue arrows and no outline."""
    runs = contiguous_projected_runs(projected=projected, valid=valid)
    for run in runs:
        draw.line(run, fill=RGB_EXECUTED_PATH_COLOR, width=3, joint="curve")
        points = np.asarray(run, dtype=np.float64)
        cumulative = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        for distance in np.arange(45, cumulative[-1] - 10, 80):
            index = int(np.searchsorted(cumulative, distance))
            tangent = points[min(index + 2, len(points) - 1)] - points[max(index - 2, 0)]
            length = float(np.linalg.norm(tangent))
            if length < 1e-5:
                continue
            tangent /= length
            normal = np.array([-tangent[1], tangent[0]])
            center = points[index]
            draw.polygon([
                tuple(center + 9 * tangent),
                tuple(center - 6 * tangent + 6 * normal),
                tuple(center - 6 * tangent - 6 * normal),
            ], fill=RGB_EXECUTED_PATH_COLOR)
    return bool(runs)

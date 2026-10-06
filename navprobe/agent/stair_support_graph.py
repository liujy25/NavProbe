"""Rolling multilayer support graph built only from observed RGB-D."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from navprobe.agent.vertical_fss import _backproject, _horizontal_support_mask


class SurfaceGraph:
    def __init__(self):
        self.support = {}
        self.occupied = {}

    def update(self, observations):
        for obs in observations:
            depth = np.asarray(obs.depth, dtype=np.float32)
            valid = np.isfinite(depth) & (depth > 0.15) & (depth < 4.9)
            cam, world = _backproject(
                depth, np.asarray(obs.intrinsics), np.asarray(obs.T_cam_odom)
            )
            mask = _horizontal_support_mask(cam, np.asarray(obs.T_cam_odom), valid)
            # Retain different support levels at the same XY, and fuse observations.
            for points, target, scale in [
                (world[mask][::2], self.support, np.array([0.10, 0.10, 0.08])),
                (world[valid][::4], self.occupied, np.array([0.10, 0.10, 0.10])),
            ]:
                keys = np.floor(points / scale).astype(int)
                _, ids = np.unique(keys, axis=0, return_index=True)
                for i in ids:
                    key = tuple(keys[i])
                    target[key] = points[i]
        pose = observations[-1].pose
        xy = np.array([pose.x, pose.y])
        for target in (self.support, self.occupied):
            stale = [k for k, p in target.items() if np.linalg.norm(p[:2] - xy) > 5.0]
            for k in stale:
                del target[k]

    def candidates(self, pose, direction):
        start = np.array([pose.x, pose.y, pose.z])
        sign = 1 if direction == "up" else -1
        points = np.asarray(list(self.support.values()))
        if len(points) == 0:
            return [], points
        points = points[
            (np.linalg.norm(points[:, :2] - start[:2], axis=1) < 4.0)
            & (np.abs(points[:, 2] - start[2]) < 1.6)
        ]
        occupied = np.asarray(list(self.occupied.values()))
        if len(points) == 0:
            return [], points
        # Check a body column above each support, including overhead surfaces.
        tree = cKDTree(occupied[:, :2])
        neighborhoods = tree.query_ball_point(points[:, :2], r=0.105)
        clear = np.array(
            [
                not np.any(
                    (occupied[n, 2] - p[2] > 0.26) & (occupied[n, 2] - p[2] < 1.15)
                )
                for p, n in zip(points, neighborhoods)
            ]
        )
        points = points[clear]
        if len(points) == 0:
            return [], points
        seed_scores = np.linalg.norm(points[:, :2] - start[:2], axis=1)
        # Use the observed support surface as the local elevation reference.
        # NavMesh agent poses can sit above the rendered floor by a voxel offset.
        seed_scores[
            (points[:, 2] < start[2] - 0.35) | (points[:, 2] > start[2] + 0.18)
        ] = np.inf
        seed = int(np.argmin(seed_scores))
        if seed_scores[seed] > 0.65:
            return [], points
        pairs = cKDTree(points[:, :2]).query_pairs(0.155, output_type="ndarray")
        if len(pairs) == 0:
            return [], points
        delta = points[pairs[:, 1]] - points[pairs[:, 0]]
        valid = (np.abs(delta[:, 2]) <= 0.24) & (
            np.linalg.norm(delta[:, :2], axis=1) > 0.025
        )
        pairs, delta = pairs[valid], delta[valid]
        weights = np.linalg.norm(delta, axis=1) + 0.2 * np.abs(delta[:, 2])
        graph = coo_matrix(
            (
                np.r_[weights, weights],
                (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]]),
            ),
            shape=(len(points), len(points)),
        ).tocsr()
        dist, parents = dijkstra(
            graph, directed=False, indices=seed, return_predecessors=True
        )
        planar = np.linalg.norm(points[:, :2] - start[:2], axis=1)
        support_z = float(points[seed, 2])
        progress = sign * (points[:, 2] - support_z)
        eligible = np.flatnonzero(
            np.isfinite(dist)
            & (planar >= 0.40)
            & (planar <= 2.0)
            & (progress >= -0.15)
            & (dist < 3.5)
        )
        # Preserve both stair-continuing and horizontal landing routes.
        rankings = [
            eligible[
                np.argsort(
                    -(
                        progress[eligible] * 2
                        + planar[eligible] * 0.3
                        - dist[eligible] * 0.1
                    )
                )
            ],
            eligible[np.argsort(-planar[eligible])],
        ]
        chosen = []
        for order in rankings:
            added = 0
            for j in order:
                if any(np.linalg.norm(points[j] - points[k]) < 0.55 for k in chosen):
                    continue
                chosen.append(int(j))
                added += 1
                if added >= 4:
                    break
        result = []
        for j in chosen:
            path = [j]
            while path[-1] != seed:
                parent = int(parents[path[-1]])
                if parent < 0:
                    break
                path.append(parent)
            path.reverse()
            xyz = np.vstack([start, points[path]])
            result.append(
                {
                    "label": len(result) + 1,
                    "goal_xyz": points[j].tolist(),
                    "path_xyz": xyz.tolist(),
                    "height_delta_m": float(points[j, 2] - support_z),
                    "agent_relative_height_delta_m": float(points[j, 2] - start[2]),
                    "path_length_m": float(dist[j]),
                }
            )
        return result, points[np.isfinite(dist)]

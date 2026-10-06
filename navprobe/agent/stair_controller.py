"""Observation-only stair routing and destination-surface evidence."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw

from navprobe.agent.stair_support_graph import SurfaceGraph
from navprobe.agent.visual_action_context import (
    direction_for_angle,
    ordered_visual_views,
)
from navprobe.agent.visual_policy_prompt_images import (
    image_content_for_array,
    image_content_for_vertical_transition_panorama_views,
)


class StairController:
    def __init__(self, direction, start, task_context):
        self.direction = direction
        self.sign = 1 if direction == "up" else -1
        self.start = np.asarray(start, dtype=float)
        self.source_support_height = float(self.start[2])
        self.task_context = task_context
        self.map = SurfaceGraph()
        self.trace = [self.start.copy()]
        self.visited = []
        self.failed = []
        self.last_heading = None
        self.phase = "entrance"
        self.arrived = False
        self.exit_level = None
        self.exit_start = None
        self.exit_trace_index = None
        self.failure_reason = None
        self.last_decision = {}

    def observe_raw_observation(self, observation):
        self.observe([observation])

    def observe(self, observations):
        self.map.update(observations)
        for observation in observations:
            pose = observation.pose
            xyz = np.array([pose.x, pose.y, pose.z], dtype=float)
            if np.linalg.norm(xyz - self.trace[-1]) > 0.01:
                self.trace.append(xyz)

    def flat_distance(self):
        points = np.asarray(self.trace)
        height = points[-1, 2]
        distance = 0.0
        for i in range(len(points) - 1, 0, -1):
            if max(abs(points[i, 2] - height), abs(points[i - 1, 2] - height)) > 0.12:
                break
            distance += np.linalg.norm(points[i, :2] - points[i - 1, :2])
        return float(distance)

    def state(self):
        return {
            "phase": self.phase,
            "floor_arrived": self.arrived,
            "height_progress": float(self.sign * (self.trace[-1][2] - self.start[2])),
            "flat_distance": self.flat_distance(),
            "exit_confirmation_height": self.exit_level,
            "failure_reason": self.failure_reason,
            "exit_confirmation_displacement": (
                0.0
                if self.exit_start is None
                else float(np.linalg.norm(self.trace[-1][:2] - self.exit_start[:2]))
            ),
        }

    def support_evidence(self, points, *, center=None):
        current = self.trace[-1] if center is None else np.asarray(center, dtype=float)
        area, width = 0.0, 0.0
        height = None
        margin = [False] * 5
        if len(points):
            local = points[
                (np.linalg.norm(points[:, :2] - current[:2], axis=1) < 0.9)
                & (points[:, 2] > current[2] - 0.35)
                & (points[:, 2] < current[2] + 0.08)
            ]
            if len(local) >= 20:
                bins, counts = np.unique(
                    np.round(local[:, 2] / 0.05), return_counts=True
                )
                height = bins[counts.argmax()] * 0.05
                plane = local[abs(local[:, 2] - height) < 0.075]
                cells = np.unique(np.floor(plane[:, :2] / 0.10), axis=0)
                area = len(cells) * 0.01
                if len(cells) >= 20:
                    _, _, axes = np.linalg.svd(
                        plane[:, :2] - plane[:, :2].mean(0), full_matrices=False
                    )
                    width = float(np.ptp(plane[:, :2] @ axes.T, axis=0).min())
                # points are body-clear supports connected to the current seed.
                seed = local[
                    np.argmin(np.linalg.norm(local[:, :2] - current[:2], axis=1))
                ]
                level = points[abs(points[:, 2] - seed[2]) <= 0.12]
                # Include body radius beyond the desired 0.20 m stopping margin.
                for i, delta in enumerate(
                    [(0, 0), (0.30, 0), (-0.30, 0), (0, 0.30), (0, -0.30)]
                ):
                    margin[i] = bool(
                        len(level)
                        and np.any(
                            np.linalg.norm(level[:, :2] - (current[:2] + delta), axis=1)
                            <= 0.10
                        )
                    )
        return {
            "broad": bool(area >= 0.35 and width >= 0.45),
            "area": round(area, 3),
            "width": round(width, 3),
            "support_height": None if height is None else float(height),
            "flat_distance": self.flat_distance(),
            "exit_margin_probes": margin,
            "exit_margin_clear": all(margin),
            # The outer probes describe unoccupied space, not the robot's
            # footprint. Walls and occlusions can legitimately hide a probe.
            "current_support_observed": margin[0],
        }

    def candidate_score(self, candidate, pose):
        goal = np.asarray(candidate["goal_xyz"])
        start = np.array([pose.x, pose.y, pose.z])
        visited = sum(np.linalg.norm(goal - p) < 0.65 for p in self.visited[-12:])
        failed = sum(np.linalg.norm(goal - p) < 0.65 for p in self.failed)
        distance = np.linalg.norm(goal[:2] - start[:2])
        heading = (goal[:2] - start[:2]) / max(distance, 1e-6)
        consistency = (
            0 if self.last_heading is None else float(heading @ self.last_heading)
        )
        return float(
            3.5 * self.sign * candidate["height_delta_m"]
            + 0.20 * distance
            - 0.18 * candidate["path_length_m"]
            - 0.75 * visited
            - 2.0 * failed
            + 0.15 * consistency
        )

    def rank(self, candidates, pose):
        usable = [
            c
            for c in candidates
            if not any(
                np.linalg.norm(np.asarray(c["goal_xyz"]) - p) < 0.4 for p in self.failed
            )
        ]
        directed = [c for c in usable if self.sign * c["height_delta_m"] > 0.22]
        return sorted(
            directed or usable,
            key=lambda c: self.candidate_score(c, pose),
            reverse=True,
        )

    def classify_surface(self, client, cache, visual_context, evidence):
        measured = {
            key: self.state()[key] for key in ("height_progress", "flat_distance")
        }
        support = {key: evidence[key] for key in ("area", "width")}
        prompt = f"""Current transition: go {self.direction} to the first destination floor.

Input meanings:
The four labeled views show the robot's current surroundings. Movement measurements describe its actual progress; local support measurements describe the observed surface nearby.

Classify the surface occupied now:
- flight: the robot is on stair treads.
- landing: the robot is on a platform where the same staircase continues in the requested direction.
- exit: the robot is physically in an ordinary room or corridor on the destination floor, beyond the final tread.

Check all views for continuing treads. A side doorway, window, or outdoor opening does not turn a landing into an exit. A room seen through an opening or beyond remaining treads has not necessarily been entered. A staircase reached through the destination-floor room or corridor belongs to a later transition.

Measured actual movement: {json.dumps(measured)}
Observed local support: {json.dumps(support)}

Output contract:
Return only JSON: {{"surface":"flight|landing|exit","reason":"current-surface evidence"}}."""
        content = [{"type": "text", "text": prompt}]
        content.extend(
            image_content_for_vertical_transition_panorama_views(
                cache=cache, views=visual_context.views
            )
        )
        result = client._create_visual_json_completion(
            call_name="vertical_navigation.surface",
            system_prompt="You classify the robot's current stair surface for NavProbe's stair controller, which uses the result to assess transition progress.",
            user_prompt=content,
            max_new_tokens=1200,
            token_field="max_completion_tokens",
        )
        if result.get("surface") not in {"flight", "landing", "exit"}:
            raise ValueError("Invalid stair surface classification")
        return result

    def choose_entrance(self, client, cache, visual_context, candidates):
        images, visible = [], set()
        for view in ordered_visual_views(visual_context.views):
            obs = cache.get_observation(view.obs_id).observation
            picture = Image.fromarray(obs.rgb).copy()
            draw = ImageDraw.Draw(picture)
            transform, intrinsics = np.asarray(obs.T_cam_odom), np.asarray(
                obs.intrinsics
            )
            for candidate in candidates:
                camera = transform[:3, :3] @ candidate["goal_xyz"] + transform[:3, 3]
                if camera[2] <= 0.1:
                    continue
                pixel = intrinsics @ camera
                u, v = pixel[:2] / pixel[2]
                if not (14 <= u < picture.width - 14 and 14 <= v < picture.height - 14):
                    continue
                depth = float(obs.depth[int(v), int(u)])
                if depth <= 0 or camera[2] > depth + 0.2:
                    continue
                label = candidate["label"]
                visible.add(label)
                draw.ellipse(
                    (u - 14, v - 14, u + 14, v + 14),
                    fill="yellow",
                    outline="black",
                    width=2,
                )
                draw.text((u - 5, v - 6), str(label), fill="black")
            images.extend(
                [
                    {
                        "type": "text",
                        "text": f"Current {direction_for_angle(view.angle_deg)} view:",
                    },
                    image_content_for_array(np.asarray(picture)),
                ]
            )
        if not visible:
            return None
        prompt = f"""Required staircase direction: {self.direction}.

Input meanings:
The numbered candidates in the four labeled views have observed connected support paths. A stair flight in the requested direction is not yet connected to those paths.

Choose an entrance approach:
Select a visible candidate that approaches the required staircase entrance. If none is supported by the images, report no selection.

Available candidate IDs: {sorted(visible)}.

Output contract:
Return only JSON: {{"candidate_id":integer or null,"reason":"visible entrance and approach route"}}.
Use an available integer ID for a supported approach, or null when no candidate visibly approaches the required staircase."""
        decision = client._create_visual_json_completion(
            call_name="vertical_navigation.entrance",
            system_prompt="You select a local approach to the required stair entrance for NavProbe's stair controller.",
            user_prompt=[{"type": "text", "text": prompt}] + images,
            max_new_tokens=1200,
            token_field="max_completion_tokens",
        )
        label = decision.get("candidate_id")
        if label is None:
            return None
        if type(label) is not int or label not in visible:
            raise ValueError("Invalid stair entrance candidate")
        return next(c for c in candidates if c["label"] == label)

    def decide(self, pose, client, cache, visual_context):
        if self.arrived:
            return None
        candidates, support = self.map.candidates(pose, self.direction)
        ranked = self.rank(candidates, pose)
        evidence = self.support_evidence(support)
        if abs(self.trace[-1][2] - self.start[2]) < 0.10 and evidence["support_height"] is not None:
            self.source_support_height = evidence["support_height"]
        # This threshold establishes executed stair progress, not floor height.
        support_progress = (
            0.0 if evidence["support_height"] is None else
            self.sign * (evidence["support_height"] - self.source_support_height)
        )
        evidence["support_height_progress"] = float(support_progress)
        possible_exit = (
            self.state()["height_progress"] > 0.20
            and support_progress > 0.20 and evidence["broad"]
        )
        target_evidence = (
            {
                c["label"]: self.support_evidence(support, center=c["goal_xyz"])
                for c in candidates
            }
            if possible_exit
            else {}
        )
        semantic = None
        if possible_exit:
            semantic = self.classify_surface(
                client, cache, visual_context, evidence
            )
            self.phase = semantic["surface"]
            if self.phase == "exit" and self.exit_level is None:
                self.exit_level = float(pose.z)
                self.exit_start = self.trace[-1].copy()
                self.exit_trace_index = len(self.trace) - 1
            elif self.phase != "exit":
                # A provisional exit must not permanently forbid the next
                # flight after fresh observations establish a turning landing.
                self.exit_level = None
                self.exit_start = None
                self.exit_trace_index = None
            self.arrived = bool(
                self.phase == "exit"
                and evidence["flat_distance"] >= 0.35
                and evidence["current_support_observed"]
            )
        elif self.state()["height_progress"] > 0.20:
            self.phase = "flight"
        if self.exit_level is not None and not self.arrived:
            ranked = self.exit_confirmation_candidates(candidates, support, pose)
        if (
            possible_exit
            and self.phase == "flight"
            and self.exit_level is None
            and not self.arrived
        ):
            # Probe a new wide surface before passing it for another flight.
            # Classify it from the reached pose; visit memory allows continuing
            # through a connecting platform after that surface has been explored.
            confirmation = []
            for candidate in candidates:
                if abs(candidate["goal_xyz"][2] - pose.z) >= 0.30:
                    continue
                if any(
                    np.linalg.norm(np.asarray(candidate["goal_xyz"]) - p) < 0.65
                    for p in self.visited
                ):
                    continue
                if any(
                    np.linalg.norm(np.asarray(candidate["goal_xyz"]) - p) < 0.4
                    for p in self.failed
                ):
                    continue
                target_support = target_evidence[candidate["label"]]
                if target_support["broad"] and target_support["current_support_observed"]:
                    confirmation.append(candidate)
            if confirmation:
                ranked = sorted(
                    confirmation,
                    key=lambda c: self.candidate_score(c, pose),
                    reverse=True,
                )
        self.last_decision = {
            "candidates": candidates,
            "ranked_ids": [c["label"] for c in ranked],
            "evidence": evidence,
            "semantic": semantic,
            "state": self.state(),
        }
        if (
            ranked
            and self.state()["height_progress"] < 0.25
            and not any(self.sign * c["height_delta_m"] > 0.22 for c in ranked)
        ):
            selected = self.choose_entrance(client, cache, visual_context, ranked)
            self.last_decision["entrance_candidate_id"] = (
                None if selected is None else selected["label"]
            )
            return selected
        return None if self.arrived or not ranked else ranked[0]

    def exit_confirmation_candidates(self, candidates, support, pose):
        """Confirm within the observed 0.9 m exit neighborhood, never explore rooms."""
        trace = np.asarray(self.trace[self.exit_trace_index:])
        travelled = float(np.linalg.norm(np.diff(trace[:, :2], axis=0), axis=1).sum())
        remaining = 0.9 - travelled
        if remaining < 0.10:
            self.failure_reason = "exit_confirmation_exhausted"
            return []
        ranked = []
        for candidate in candidates:
            # Reuse connected paths, but execute only a short local prefix.
            path = self.path_prefix(candidate["path_xyz"], min(0.4, remaining))
            if len(path) < 2:
                continue
            if (
                np.max(np.abs(path[:, 2] - pose.z)) >= 0.30
                or np.max(np.abs(path[:, 2] - self.exit_level)) >= 0.30
                or np.max(np.linalg.norm(path[:, :2] - self.exit_start[:2], axis=1)) > 0.9
                or any(np.linalg.norm(path[-1] - p) < 0.20 for p in self.failed)
            ):
                continue
            evidence = self.support_evidence(support, center=path[-1])
            if not evidence["broad"] or not evidence["current_support_observed"]:
                continue
            delta = path[-1, :2] - [pose.x, pose.y]
            distance = np.linalg.norm(delta)
            heading = delta / max(distance, 1e-6)
            alignment = 0.0 if self.last_heading is None else float(heading @ self.last_heading)
            # No reward for distance, novel rooms, or vertical progress.
            score = alignment - 3.5 * abs(path[-1, 2] - pose.z)
            ranked.append((score, {**candidate, "goal_xyz": path[-1].tolist(),
                                  "path_xyz": path.tolist(), "exit_confirmation": True}))
        return [candidate for _, candidate in sorted(ranked, key=lambda item: item[0], reverse=True)]

    @staticmethod
    def path_prefix(points, max_distance):
        path = np.asarray(points, dtype=float)
        prefix = [path[0]]
        remaining = max_distance
        for point in path[1:]:
            distance = np.linalg.norm(point[:2] - prefix[-1][:2])
            if distance > remaining:
                prefix.append(prefix[-1] + (point - prefix[-1]) * remaining / distance)
                break
            prefix.append(point)
            remaining -= distance
            if remaining <= 1e-6:
                break
        return np.asarray(prefix)

    def reached(self, selected, before, after):
        moved = np.linalg.norm(after[:2] - before[:2])
        if moved < 0.06:
            self.failed.append(np.asarray(selected["goal_xyz"]))
        else:
            self.visited.append(after.copy())
            self.last_heading = (after[:2] - before[:2]) / moved

    @staticmethod
    def short_path(selected, pose):
        path = np.asarray(selected["path_xyz"])
        distance = np.linalg.norm(path[:, :2] - [pose.x, pose.y], axis=1)
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
        ids = np.flatnonzero((distance >= 0.7) & (arc >= 0.8))
        index = int(ids[0]) if len(ids) else len(path) - 1
        return path[: index + 1]

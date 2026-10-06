"""Shared local Habitat observations and actions; datasets own selection/reset."""
from __future__ import annotations

import traceback
from typing import Any

import numpy as np

from navprobe.env.habitat_utils import (
    _ros_quaternion_from_yaw_degrees, _serialize_metric_value, _yaw_degrees_from_rotation,
)
from navprobe.env.interface import EnvInterface, MoveResult, Pose, RawObservation


def close_environment(env: Any, *, label: str) -> None:
    """Release an owned environment without replacing its result or exception."""
    if env is not None:
        try:
            env.close()
        except Exception:
            print(f"[cleanup] {label} environment close failed:")
            traceback.print_exc()


def _pose_from_base_transform(t_odom_base: np.ndarray) -> Pose:
    return Pose(
        x=float(t_odom_base[0, 3]),
        y=float(t_odom_base[1, 3]),
        z=float(t_odom_base[2, 3]),
        yaw=_yaw_degrees_from_rotation(t_odom_base[:3, :3]),
    )


class NavProbeHabitatEnv(EnvInterface):
    """One local simulator protocol, with task-specific metadata and metrics."""

    @property
    def max_episode_steps(self) -> int:
        """The action limit enforced by Habitat itself (zero means unlimited)."""
        return int(self.env._max_episode_steps)

    def close(self) -> None:
        env, self.env = self.env, None
        try:
            if env is not None:
                env.close()
        finally:
            self.client = None
            self.target_episode = None
            self.latest_obs = None
            self._last_move_intermediate_observations = []
            self._last_turn_observation = None

    def _metrics_payload(self) -> dict[str, Any]:
        metrics = self.env.get_metrics()
        return {
            str(key): _serialize_metric_value(value)
            for key, value in metrics.items()
        }

    def _episode_success_flag(self, metrics: dict[str, Any]) -> bool:
        success_metric = metrics["success"]
        if isinstance(success_metric, dict):
            raise ValueError("Local VLNCE/HM3D success metric must be scalar; dictionaries are unsupported")
        return bool(success_metric)

    def _current_obs(self) -> dict[str, Any]:
        if self.latest_obs is None:
            self.latest_obs = self.env.sim.get_sensor_observations()
        return self.latest_obs

    def _pose_from_observation(self, obs: dict[str, Any] | None = None) -> Pose:
        target_obs = self._current_obs() if obs is None else obs
        _, t_odom_base = self.client._get_transform_matrices(target_obs)
        return _pose_from_base_transform(t_odom_base)

    def _raw_observation_from_habitat_obs(self, obs: dict[str, Any]) -> RawObservation:
        rgb, depth = self.client._extract_images(obs)
        T_cam_odom, T_odom_base = self.client._get_transform_matrices(obs)
        return RawObservation(
            pose=_pose_from_base_transform(T_odom_base),
            rgb=np.asarray(rgb, dtype=np.uint8),
            depth=np.asarray(depth, dtype=np.float32),
            intrinsics=np.asarray(self.client.intrinsic, dtype=np.float32).tolist(),
            T_cam_odom=np.asarray(T_cam_odom, dtype=np.float32).tolist(),
            T_odom_base=np.asarray(T_odom_base, dtype=np.float32).tolist(),
            text_hint="",
            visible_objects=[],
        )

    def current_episode_info(self) -> dict[str, Any]:
        payload = self._episode_metadata_payload()
        payload.update(
            {
                "episode_over": bool(self.env.episode_over),
                "max_episode_steps": self.max_episode_steps,
                "pointnav_step_total": int(self.client.pointnav_step_total),
            }
        )
        return payload

    def get_obs(self) -> RawObservation:
        return self._raw_observation_from_habitat_obs(self._current_obs())

    def turn(self, direction: str) -> Pose:
        if direction not in {"left", "right"}:
            raise ValueError(f"Unsupported turn direction: {direction}")
        self._last_turn_observation = None
        if self.env.episode_over:
            return self._pose_from_observation()
        action_type = "turn_left" if direction == "left" else "turn_right"
        _, observations, nav_info = self.client.execute_discrete_action(action_type)
        self.client.pointnav_step_total += int(nav_info["num_steps"])
        self.latest_obs = observations[-1]
        self._last_turn_observation = self._raw_observation_from_habitat_obs(self.latest_obs)
        return self._last_turn_observation.pose

    def move(self, x: float, y: float, yaw: float, z: float | None = None) -> MoveResult:
        self._last_move_intermediate_observations = []
        start_pose = self._pose_from_observation()
        if self.env.episode_over:
            return MoveResult(False, start_pose, start_pose, {
                "success": False, "interrupt_reason": "episode_over", "num_steps": 0,
            })
        target_z = float(start_pose.z if z is None else z)
        goal_pose = {
            "pose": {
                "position": {
                    "x": float(x),
                    "y": float(y),
                    "z": target_z,
                },
                "orientation": _ros_quaternion_from_yaw_degrees(float(yaw)),
            }
        }
        step_observations: list[RawObservation] = []

        def collect_step_observation(step_obs: dict[str, Any], *_args: Any) -> None:
            step_observations.append(self._raw_observation_from_habitat_obs(step_obs))

        success, observations, nav_info = self.client.execute_goal_pose(
            goal_pose,
            max_steps=self.client.nav_controller.max_steps,
            step_callback=collect_step_observation,
        )
        self.client.pointnav_step_total += int(nav_info["num_steps"])
        if observations != []:
            self.latest_obs = observations[-1]
        else:
            self.latest_obs = self.env.sim.get_sensor_observations()
        if observations != [] and len(step_observations) == len(observations):
            step_observations = step_observations[:-1]
        self._last_move_intermediate_observations = list(step_observations)
        return MoveResult(bool(success), start_pose, self._pose_from_observation(self.latest_obs), nav_info)

    def pop_last_move_intermediate_observations(self) -> list[RawObservation]:
        observations = list(self._last_move_intermediate_observations)
        self._last_move_intermediate_observations = []
        return observations

    def pop_last_turn_observation(self) -> RawObservation | None:
        observation = self._last_turn_observation
        self._last_turn_observation = None
        return observation

    def evaluation_snapshot(self) -> dict[str, Any]:
        metrics = self._metrics_payload()
        return {
            "pose": self._pose_from_observation().to_dict(),
            "success": True,
            "episode_success": self._episode_success_flag(metrics),
            "pointnav_step_total": int(self.client.pointnav_step_total),
            "episode_over": bool(self.env.episode_over),
            "metrics": metrics,
            "nav_info": {
                "num_steps": 0,
                "actions": [],
                "success": True,
                "interrupted": False,
                "interrupt_reason": None,
                "mode": "evaluation_snapshot",
            },
        }

    def stop(self) -> dict[str, Any]:
        if self.env.episode_over:
            metrics = self._metrics_payload()
            return {
                "pose": self._pose_from_observation().to_dict(),
                "success": True,
                "episode_success": self._episode_success_flag(metrics),
                "pointnav_step_total": int(self.client.pointnav_step_total),
                "episode_over": True,
                "metrics": metrics,
                "nav_info": {
                    "num_steps": 0,
                    "actions": [],
                    "success": True,
                    "interrupted": False,
                    "interrupt_reason": None,
                    "mode": "stop_skipped_episode_over",
                },
            }
        success, observations, nav_info = self.client.execute_discrete_action("stop")
        self.latest_obs = observations[-1]
        self.client.pointnav_step_total += int(nav_info["num_steps"])
        metrics = self._metrics_payload()
        return {
            "pose": self._pose_from_observation(self.latest_obs).to_dict(),
            "success": bool(success),
            "episode_success": self._episode_success_flag(metrics),
            "pointnav_step_total": int(self.client.pointnav_step_total),
            "episode_over": bool(self.env.episode_over),
            "metrics": metrics,
            "nav_info": nav_info,
        }

from __future__ import annotations

import time
from dataclasses import replace
from typing import Callable

from navprobe.schemas import ActionCall, ActionResult
from navprobe.env.interface import EnvInterface
from navprobe.env.interface import RawObservation
from navprobe.perception import geometry as perception
from navprobe.runtime.cache import RuntimeCache


def _capture_step_observation_enabled(
    step_observation_handler: Callable[[RawObservation, str], dict[str, object] | None] | None,
    step_observation_enabled: Callable[[], bool] | None,
) -> bool:
    if step_observation_handler is None:
        return False
    if step_observation_enabled is None:
        return True
    return bool(step_observation_enabled())


def _apply_step_observation(
    observation: RawObservation,
    source: str,
    step_observation_handler: Callable[[RawObservation, str], dict[str, object] | None] | None,
) -> dict[str, object] | None:
    if step_observation_handler is None:
        return None
    return step_observation_handler(observation, source)


def _apply_intermediate_observations_to_bev(
    env: EnvInterface,
    exploration,
    step_observation_handler: Callable[[RawObservation, str], dict[str, object] | None] | None = None,
    step_observation_enabled: Callable[[], bool] | None = None,
    rgb_history_observation_handler: Callable[[RawObservation, str], dict[str, object] | None] | None = None,
    rgb_history_sample_interval: int = 2,
) -> tuple[list[dict[str, object]], list[str], list[list[float]], dict[str, object]]:
    total_start = time.perf_counter()
    observations = env.pop_last_move_intermediate_observations()
    step_payloads: list[dict[str, object]] = []
    rgb_history_obs_ids: list[str] = []
    observed_path_xy: list[list[float]] = []
    capture_step_observations = _capture_step_observation_enabled(
        step_observation_handler=step_observation_handler,
        step_observation_enabled=step_observation_enabled,
    )
    sample_interval = max(1, int(rgb_history_sample_interval))
    timing = {
        "intermediate_observation_count": int(len(observations)),
        "capture_step_observations": bool(capture_step_observations),
        "rgb_history_sample_interval": int(sample_interval),
        "bev_update_seconds": 0.0,
        "step_observation_handler_seconds": 0.0,
        "rgb_history_handler_seconds": 0.0,
        "step_observation_payload_count": 0,
        "rgb_history_obs_count": 0,
    }
    for move_obs_index, observation in enumerate(observations):
        _append_observation_path_xy(observed_path_xy, observation)
        payload: dict[str, object] | None = None
        bev_start = time.perf_counter()
        exploration.observe_raw_observation(observation)
        timing["bev_update_seconds"] = float(timing["bev_update_seconds"]) + (
            time.perf_counter() - bev_start
        )
        if capture_step_observations:
            handler_start = time.perf_counter()
            payload = _apply_step_observation(
                observation=observation,
                source="move_intermediate",
                step_observation_handler=step_observation_handler,
            )
            timing["step_observation_handler_seconds"] = float(
                timing["step_observation_handler_seconds"]
            ) + (time.perf_counter() - handler_start)
            if payload is not None:
                step_payloads.append(payload)
        if rgb_history_observation_handler is not None and int(move_obs_index) % sample_interval == 0:
            existing_obs_id = None if payload is None else payload.get("obs_id")
            if existing_obs_id is not None:
                rgb_history_obs_ids.append(str(existing_obs_id))
            else:
                history_start = time.perf_counter()
                history_payload = rgb_history_observation_handler(observation, "move_rgb_history")
                timing["rgb_history_handler_seconds"] = float(
                    timing["rgb_history_handler_seconds"]
                ) + (time.perf_counter() - history_start)
                if isinstance(history_payload, dict) and history_payload.get("obs_id") is not None:
                    rgb_history_obs_ids.append(str(history_payload["obs_id"]))
    timing["step_observation_payload_count"] = int(len(step_payloads))
    timing["rgb_history_obs_count"] = int(len(rgb_history_obs_ids))
    timing["elapsed_seconds"] = float(time.perf_counter() - total_start)
    return step_payloads, rgb_history_obs_ids, observed_path_xy, timing


def _append_observation_path_xy(path_xy: list[list[float]], observation: RawObservation) -> None:
    point = [float(observation.pose.x), float(observation.pose.y)]
    if path_xy != [] and path_xy[-1] == point:
        return
    path_xy.append(point)


def _pose_xy_list(pose) -> list[float]:
    return [float(pose.x), float(pose.y)]


def move(
    env: EnvInterface,
    exploration,
    step_observation_handler: Callable[[RawObservation, str], dict[str, object] | None] | None,
    step_observation_enabled: Callable[[], bool] | None,
    x: float,
    y: float,
    yaw: float,
    z: float | None = None,
    rgb_history_observation_handler: Callable[[RawObservation, str], dict[str, object] | None] | None = None,
) -> ActionResult:
    total_start = time.perf_counter()
    env_move_start = time.perf_counter()
    movement = env.move(x=x, y=y, yaw=yaw, z=z)
    env_move_seconds = time.perf_counter() - env_move_start
    step_payloads, rgb_history_obs_ids, observed_path_xy, intermediate_timing = _apply_intermediate_observations_to_bev(
        env,
        exploration,
        step_observation_handler=step_observation_handler,
        step_observation_enabled=step_observation_enabled,
        rgb_history_observation_handler=rgb_history_observation_handler,
    )
    move_timing: dict[str, object] = {
        "elapsed_seconds": float(time.perf_counter() - total_start),
        "env_move_call_seconds": float(env_move_seconds),
        "intermediate_processing": intermediate_timing,
    }
    data: dict[str, object] = {
        "pose": movement.pose.to_dict(),
        "start_pose": movement.start_pose.to_dict(),
        "nav_info": movement.nav_info,
        "failure_reason": movement.failure_reason,
        "timing": move_timing,
    }
    if step_payloads != []:
        data["step_observations"] = step_payloads
    if rgb_history_obs_ids != []:
        data["rgb_history_obs_ids"] = [str(obs_id) for obs_id in rgb_history_obs_ids]
    # Include measured endpoints even when no intermediate frame was emitted.
    # Planned targets never contribute to this trajectory.
    path_xy = [_pose_xy_list(movement.start_pose)]
    for point in [*observed_path_xy, _pose_xy_list(movement.pose)]:
        if point != path_xy[-1]:
            path_xy.append(point)
    data["path_xy"] = path_xy
    data["position_changed"] = (
        len(path_xy) > 1 or movement.pose.z != movement.start_pose.z
    )
    return ActionResult(
        ok=bool(movement.success),
        data=data,
        message=(
            f"Move completed at {movement.pose.to_dict()}." if movement.success else
            f"Move failed ({movement.failure_reason}); stopped at {movement.pose.to_dict()}."
        ),
    )


def done(env: EnvInterface) -> ActionResult:
    stop_result = env.stop()
    return ActionResult(
        ok=True,
        data={
            "done": True,
            "stop_result": stop_result,
        },
        message="Stop requested.",
    )


class LocalmapActionExecutor:
    """Execute navigation actions and capture observations for mapping."""

    def __init__(
        self,
        *,
        env: EnvInterface,
        cache: RuntimeCache,
        exploration,
    ) -> None:
        self.env = env
        self.cache = cache
        self.exploration = exploration
        self.post_execute_handler = None
        self.step_observation_handler = None
        self.step_observation_enabled_getter = None

    def should_capture_step_observations(self) -> bool:
        if self.step_observation_handler is None:
            return False
        if self.step_observation_enabled_getter is None:
            return True
        return bool(self.step_observation_enabled_getter())

    def handle_step_observation(
        self,
        observation: RawObservation,
        source: str,
    ) -> dict[str, object] | None:
        if self.step_observation_handler is None:
            return None
        return self.step_observation_handler(observation, source)

    def handle_rgb_history_observation(
        self,
        observation: RawObservation,
        source: str,
    ) -> dict[str, object]:
        # Mapping has consumed the original frame. Movement retrieval needs
        # only RGB and pose; keep node/panorama observations and the original
        # environment frame intact for their geometry consumers.
        record = self.cache.store_observation(replace(observation, depth=None))
        return {
            "obs_id": str(record.id),
            "rgb_id": f"{record.id}:rgb",
            "pose": observation.pose.to_dict(),
            "source": str(source),
        }

    def execute(self, call: ActionCall) -> ActionResult:
        action_name = str(call.action)
        args = dict(call.args)
        if action_name == "move":
            result = move(
                self.env,
                self.exploration,
                self.handle_step_observation,
                self.should_capture_step_observations,
                rgb_history_observation_handler=self.handle_rgb_history_observation,
                **args,
            )
        elif action_name == "get_obs":
            result = perception.get_obs(self.env, self.cache, self.exploration)
        elif action_name == "done":
            result = done(self.env)
        else:
            raise ValueError(f"unsupported localmap action: {action_name!r}")
        if self.post_execute_handler is not None:
            self.post_execute_handler(call, result)
        return result

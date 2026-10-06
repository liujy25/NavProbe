from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable

from navprobe.perception.landmark_detection import LandmarkDetectionController
from navprobe.env.interface import EnvInterface
from navprobe.mapping.exploration.manager import ExplorationManager
from navprobe.runtime.cache import RuntimeCache
from navprobe.runtime.panorama_config import PanoramaConfig, localmap_panorama_config


PANORAMA_YAW_WARNING_DEGREES = 15.0


@dataclass(frozen=True)
class PanoramaView:
    angle_deg: int
    obs_id: str
    rgb_id: str
    depth_id: str
    pose: dict[str, float]
    yaw_debug: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "angle_deg": int(self.angle_deg),
            "obs_id": str(self.obs_id),
            "rgb_id": str(self.rgb_id),
            "depth_id": str(self.depth_id),
            "pose": dict(self.pose),
            "yaw_debug": dict(self.yaw_debug),
        }


@dataclass(frozen=True)
class PanoramaCaptureResult:
    views: list[PanoramaView]
    angle_to_obs_id: dict[int, str]
    anchor_obs_id: str
    current_obs_id: str
    angle_debug: dict[str, object]

    @property
    def obs_ids(self) -> list[str]:
        return [str(view.obs_id) for view in self.views]

    def to_dict(self) -> dict[str, object]:
        return {
            "views": [view.to_dict() for view in self.views],
            "angle_to_obs_id": {str(angle): str(obs_id) for angle, obs_id in self.angle_to_obs_id.items()},
            "anchor_obs_id": str(self.anchor_obs_id),
            "current_obs_id": str(self.current_obs_id),
            "angle_debug": dict(self.angle_debug),
        }


def _capture_panorama(
    env: EnvInterface,
    cache: RuntimeCache,
    global_exploration: ExplorationManager | None,
    landmark_controller: LandmarkDetectionController | None = None,
    panorama_config: PanoramaConfig | None = None,
    stop_capture: Callable[[], bool] | None = None,
) -> PanoramaCaptureResult:
    if panorama_config is None:
        panorama_config = localmap_panorama_config()
    views: list[PanoramaView] = []
    angle_to_obs_id: dict[int, str] = {}
    reference_yaw: float | None = None
    target_angles = _panorama_target_angles(panorama_config)
    warnings: list[dict[str, object]] = []
    final_turn_pose = None
    for target_angle in target_angles:
        observation = env.get_obs()
        if reference_yaw is None:
            reference_yaw = float(observation.pose.yaw)
        record = cache.store_observation(observation)
        if global_exploration is not None:
            global_exploration.observe_observation(cache=cache, obs_id=record.id)
        if landmark_controller is not None and landmark_controller.is_active():
            landmark_controller.update_for_obs_id(obs_id=record.id, source="panorama_obs")
        yaw_debug = _view_yaw_debug(
            reference_yaw=float(reference_yaw),
            actual_yaw=float(observation.pose.yaw),
            target_angle=int(target_angle),
        )
        if abs(float(yaw_debug["error_degrees"])) > float(PANORAMA_YAW_WARNING_DEGREES):
            warnings.append(
                {
                    "kind": "view_yaw_error",
                    "target_angle": int(target_angle),
                    **yaw_debug,
                }
            )
        view = PanoramaView(
            angle_deg=int(target_angle),
            obs_id=str(record.id),
            rgb_id=f"{record.id}:rgb",
            depth_id=f"{record.id}:depth",
            pose=observation.pose.to_dict(),
            yaw_debug=yaw_debug,
        )
        views.append(view)
        angle_to_obs_id[int(target_angle)] = str(record.id)
        if stop_capture is None:
            final_turn_pose = _turn_to_next_panorama_view(env=env, panorama_config=panorama_config)
        else:
            for _ in range(int(panorama_config.turns_per_observation)):
                if stop_capture():
                    break
                final_turn_pose = env.turn(panorama_config.turn_direction)
            if stop_capture():
                break
    if views == []:
        raise ValueError("panorama capture produced no observations")
    anchor_obs_id = str(angle_to_obs_id.get(0, views[0].obs_id))
    final_yaw_debug = None
    if final_turn_pose is not None and reference_yaw is not None:
        final_yaw_debug = {
            "reference_yaw": float(reference_yaw),
            "final_yaw": float(final_turn_pose.yaw),
            "return_error_degrees": _signed_angle_delta_degrees(
                float(final_turn_pose.yaw),
                float(reference_yaw),
            ),
        }
        if abs(float(final_yaw_debug["return_error_degrees"])) > float(PANORAMA_YAW_WARNING_DEGREES):
            warnings.append({"kind": "final_return_yaw_error", **final_yaw_debug})
    return PanoramaCaptureResult(
        views=views,
        angle_to_obs_id=angle_to_obs_id,
        anchor_obs_id=anchor_obs_id,
        current_obs_id=anchor_obs_id,
        angle_debug={
            "target_angles": [int(angle) for angle in target_angles],
            "reference_yaw": reference_yaw,
            "final_yaw": None if final_turn_pose is None else float(final_turn_pose.yaw),
            "final_return": final_yaw_debug,
            "warnings": warnings,
        },
    )


def _panorama_target_angles(panorama_config: PanoramaConfig) -> list[int]:
    spacing = float(panorama_config.observation_spacing_degrees)
    return [
        int(round((float(index) * spacing) % 360.0))
        for index in range(int(panorama_config.observation_count))
    ]


def _turn_to_next_panorama_view(
    *,
    env: EnvInterface,
    panorama_config: PanoramaConfig,
):
    final_pose = None
    for _ in range(int(panorama_config.turns_per_observation)):
        final_pose = env.turn(panorama_config.turn_direction)
    return final_pose


def _view_yaw_debug(
    *,
    reference_yaw: float,
    actual_yaw: float,
    target_angle: int,
) -> dict[str, object]:
    actual_relative = _normalize_angle_360(float(actual_yaw) - float(reference_yaw))
    target_relative = float(target_angle) % 360.0
    return {
        "reference_yaw": float(reference_yaw),
        "actual_yaw": float(actual_yaw),
        "actual_relative_degrees": float(actual_relative),
        "target_relative_degrees": float(target_relative),
        "error_degrees": _signed_angle_delta_degrees(actual_relative, target_relative),
    }


def _normalize_angle_360(angle: float) -> float:
    value = math.fmod(float(angle), 360.0)
    if value < 0.0:
        value += 360.0
    return value


def _signed_angle_delta_degrees(value: float, reference: float) -> float:
    return ((float(value) - float(reference) + 180.0) % 360.0) - 180.0

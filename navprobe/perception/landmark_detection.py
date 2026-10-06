from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np

from navprobe.env.interface import RawObservation
from navprobe.perception import geometry as perception
from navprobe.perception.goal import NAVPROBE_LANGUAGE_GOAL_KIND
from navprobe.memory.landmarks import (
    NavProbeLandmarkMemory,
    landmark_position_distance_3d,
)
from navprobe.schemas import ActionCall, ActionResult
from navprobe.runtime.cache import RuntimeCache

if TYPE_CHECKING:
    from navprobe.llm.client import LLMClient
    from navprobe.perception.detectors.interface import DetectorInterface
    from navprobe.perception.goal import NavProbeGoalSpec


LANDMARK_GEOMETRY_BBOX_CENTER = "bbox_center"
SYSTEM_DOOR_CLASS_NAME = "door"


def configure_landmark_detection(
    *,
    controller: LandmarkDetectionController,
    client: LLMClient,
    goal_spec: NavProbeGoalSpec | None,
    goal_target: str = "",
    detector_threshold: float,
) -> dict[str, object]:
    if str(getattr(goal_spec, "goal_kind", "") or "").strip() != NAVPROBE_LANGUAGE_GOAL_KIND:
        controller.enabled = False
        return {"enabled": False, "reason": "unsupported_goal_kind"}
    navigation_goal_text = getattr(goal_spec, "navigation_goal_text", None)
    instruction = (
        str(navigation_goal_text()).strip()
        if callable(navigation_goal_text)
        else str(
            getattr(goal_spec, "description", "") or goal_target
        ).strip()
    )
    if instruction == "":
        controller.enabled = False
        return {"enabled": False, "reason": "empty_instruction"}
    response = client._create_visual_json_completion(
        call_name="landmark_perception.select_categories",
        system_prompt=(
            "You are the Landmark Category Generator in NavProbe.\n"
            "Select object and fixture categories for the detector to recognize "
            "landmarks relevant to the navigation instruction."
        ),
        user_prompt=f"""
Navigation instruction:
{instruction}

This request provides text only. The categories configure what to look for; they do not establish which objects are present.

Select categories:
- Start with objects and fixtures named in the instruction. Include those with clear physical boundaries that a detector can localize with a stable bounding box and that help recognize the instructed scene or place.
- For each distinct place type mentioned, add at most one typical object category when it is visually distinctive, detectable, and useful for recognizing that place.
- Leave place types, passage and circulation structures, routes, directions, and spatial relations to the navigation modules, which interpret them from images.
- Use short category nouns and keep only distinct, useful categories; merge duplicates and near-synonyms.

Output contract:
Return only JSON. Use an empty list if no category meets the selection rules:
{{
  "landmark_categories": ["<category>"]
}}
""".strip(),
        max_new_tokens=client.request_limits["landmark_perception"]["max_tokens"],
        token_field="max_completion_tokens",
    )
    categories = _normalize_landmark_categories(response.get("landmark_categories", []))
    if categories == []:
        controller.enabled = False
        return {"enabled": False, "reason": "empty_detection_list", "landmark_categories": []}
    class_map = {
        category: {
            "class_name": category,
            "threshold": float(detector_threshold),
            "merge_distance_m": float(controller.configuration["merge_distance_m"]),
        }
        for category in categories
    }
    controller.configure_landmark_class_map(
        class_map,
        reset_landmark_memory=True,
    )
    controller.enabled = True
    return {
        "enabled": True,
        "landmark_categories": list(categories),
    }


def _normalize_landmark_categories(value: object) -> list[str]:
    raw_items = list(value) if isinstance(value, list) else []
    categories: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        category = str(item).strip().lower().replace("_", " ")
        if category == "" or category in seen:
            continue
        seen.add(category)
        categories.append(category)
    return categories


class LandmarkDetectionController:
    def __init__(
        self,
        *,
        detector: DetectorInterface,
        configuration: dict[str, object],
        cache_provider: Callable[[], RuntimeCache | None],
        landmark_class_map: dict[str, object] | None = None,
    ) -> None:
        self.detector = detector
        self.configuration = dict(configuration)
        self._cache_provider = cache_provider
        # The caller sets the navigation step before capturing observations.
        self.step_index = 0
        self.landmark_class_map = _normalize_landmark_class_map(landmark_class_map, configuration=self.configuration)
        self.landmark_memory = NavProbeLandmarkMemory(
            merge_distance_m=self.configuration["merge_distance_m"],
            max_detections_per_entity=self.configuration["max_detections_per_entity"],
            merge_distance_by_class=_merge_distance_by_system_class(self.landmark_class_map)
        )
        self.enabled = True

    def is_active(self) -> bool:
        return bool(self.enabled) and self.landmark_class_map != {}

    def configure_landmark_class_map(
        self,
        landmark_class_map: dict[str, object],
        *,
        reset_landmark_memory: bool = True,
    ) -> None:
        self.landmark_class_map = _normalize_landmark_class_map(landmark_class_map, configuration=self.configuration)
        if bool(reset_landmark_memory):
            self.landmark_memory = NavProbeLandmarkMemory(
                merge_distance_m=self.configuration["merge_distance_m"],
                max_detections_per_entity=self.configuration["max_detections_per_entity"],
                merge_distance_by_class=_merge_distance_by_system_class(self.landmark_class_map),
            )

    def detections_for_obs_ids(self, obs_ids: list[str]) -> dict[str, list[dict[str, object]]]:
        return self.landmark_memory.detections_for_obs_ids(obs_ids)

    def _require_cache(self) -> RuntimeCache:
        cache = self._cache_provider()
        if cache is None:
            raise ValueError("landmark detection requires runtime cache")
        return cache

    def update_for_obs_id(self, obs_id: str, source: str) -> dict[str, object] | None:
        if not self.is_active():
            return None
        cache = self._require_cache()
        query_class_names = list(self.landmark_class_map.keys())
        detect_result = perception.detect(
            self.detector,
            cache,
            class_names=query_class_names,
            canonical_class_map={
                query_class_name: str(config["class_name"])
                for query_class_name, config in self.landmark_class_map.items()
            },
            obs_id=str(obs_id),
            min_score=_min_landmark_threshold(self.landmark_class_map),
            agnostic_nms=False,
            max_results=int(self.configuration["max_results"]),
        )
        return self.update_from_detect_result(detect_result=detect_result, source=source)

    def update_from_detect_result(
        self,
        *,
        detect_result: ActionResult,
        source: str,
    ) -> dict[str, object]:
        cache = self._require_cache()
        obs_id = detect_result.data.get("obs_id")
        if obs_id is None:
            raise ValueError("landmark detection update requires detect_result.data.obs_id")
        raw_detections = detect_result.data.get("detections", [])
        if not isinstance(raw_detections, list):
            raise ValueError("landmark detection update requires list data.detections")

        skipped_detections: list[dict[str, object]] = []
        mapped_detections: list[dict[str, object]] = []
        for raw_detection in raw_detections:
            if not isinstance(raw_detection, dict):
                continue
            detection = dict(raw_detection)
            detection["obs_id"] = str(obs_id)
            detector_class_name = _normalize_landmark_class_name(
                detection.get("detector_class_name", detection.get("class_name", ""))
            )
            class_config = self.landmark_class_map.get(detector_class_name)
            if class_config is None:
                continue
            if float(detection.get("score", 0.0)) < float(class_config["threshold"]):
                continue
            canonical_class_name = str(class_config["class_name"])
            detection["class_name"] = canonical_class_name
            mapped_detections.append(detection)

        door_detections = [
            item
            for item in mapped_detections
            if str(item.get("class_name", "")) == SYSTEM_DOOR_CLASS_NAME
        ]
        non_door_detections = [
            item
            for item in mapped_detections
            if str(item.get("class_name", "")) != SYSTEM_DOOR_CLASS_NAME
        ]

        valid_detections: list[dict[str, object]] = []
        duplicate_detections: list[dict[str, object]] = []

        # Preserve door-first insertion/ID order used by the current detector path.
        buffered_detection_count = 0
        for detections in (door_detections, non_door_detections):
            valid = self._detections_with_geometry(
                detections=detections,
                cache=cache,
                obs_id=str(obs_id),
                skipped_detections=skipped_detections,
            )
            valid_detections.extend(valid)
            kept, duplicates = _dedupe_same_frame_landmarks(
                valid, landmark_memory=self.landmark_memory,
                same_frame_iou_threshold=float(self.configuration["same_frame_iou_threshold"]),
            )
            duplicate_detections.extend(duplicates)
            buffered_detection_count += len(kept)
            for detection in kept:
                self.landmark_memory.add_detection(
                    detection=detection,
                    source=source,
                    step_index=self.step_index,
                )
        return {
            "source": str(source),
            "obs_id": str(obs_id),
            "landmark_class_map": {key: dict(value) for key, value in self.landmark_class_map.items()},
            "query_classes": list(self.landmark_class_map.keys()),
            "canonical_class_names": sorted(
                {str(config["class_name"]) for config in self.landmark_class_map.values()}
            ),
            "min_score": float(_min_landmark_threshold(self.landmark_class_map)),
            "raw_detection_count": int(len(raw_detections)),
            "valid_detection_count": int(len(valid_detections)),
            "buffered_detection_count": buffered_detection_count,
            "skipped_detections": skipped_detections,
            "same_frame_duplicate_detections": duplicate_detections,
        }


    def _detections_with_geometry(
        self,
        *,
        detections: list[dict[str, object]],
        cache: RuntimeCache,
        obs_id: str,
        skipped_detections: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        valid_detections: list[dict[str, object]] = []
        for detection in detections:
            geometry, skip_reason = _landmark_center_geometry_from_detection(
                cache=cache,
                obs_id=obs_id,
                det_id=str(detection.get("det_id", "") or ""),
                min_depth_m=float(self.configuration["min_depth_m"]),
            )
            if geometry is None:
                skipped_detections.append(
                    {
                        "det_id": str(detection.get("det_id", "") or ""),
                        "reason": str(skip_reason),
                    }
                )
                continue
            detection.update(geometry)
            valid_detections.append(detection)
        return valid_detections

    def handle_post_action_execute(self, call: ActionCall, result: ActionResult) -> None:
        if call.action != "get_obs":
            return
        obs_id = result.data.get("obs_id")
        if obs_id is None:
            return
        update = self.update_for_obs_id(obs_id=str(obs_id), source="get_obs")
        if update is not None:
            result.data["landmark_detection_buffer_update"] = update

def _normalize_landmark_class_name(value: object) -> str:
    return str(value).strip().replace("_", " ")


def _normalize_landmark_class_map(
    landmark_class_map: dict[str, object] | None,
    *, configuration: dict[str, object],
) -> dict[str, dict[str, object]]:
    normalized: dict[str, dict[str, object]] = {}
    for query_class_name, raw_config in (landmark_class_map or {}).items():
        query = _normalize_landmark_class_name(query_class_name)
        if isinstance(raw_config, str):
            raw_config = {"class_name": raw_config}
        normalized[query] = {
            "class_name": _normalize_landmark_class_name(raw_config.get("class_name", "")),
            "threshold": float(raw_config.get("threshold", configuration["threshold"])),
            "merge_distance_m": float(raw_config.get("merge_distance_m", configuration["merge_distance_m"])),
        }
    return normalized


def _min_landmark_threshold(landmark_class_map: dict[str, dict[str, object]]) -> float:
    return min(float(config["threshold"]) for config in landmark_class_map.values())


def _merge_distance_by_system_class(
    landmark_class_map: dict[str, dict[str, object]],
) -> dict[str, float]:
    return {
        str(config["class_name"]): float(config["merge_distance_m"])
        for config in landmark_class_map.values()
    }


def _landmark_center_geometry_from_detection(
    *,
    cache: RuntimeCache,
    obs_id: str,
    det_id: str,
    min_depth_m: float,
) -> tuple[dict[str, object] | None, str | None]:
    observation = cache.get_observation(str(obs_id)).observation
    detection_record = cache.get_detection(str(det_id))
    bbox = [float(value) for value in detection_record.detection.bbox]
    center_point = _project_bbox_center(observation=observation, bbox=bbox, min_depth_m=min_depth_m)
    if center_point is None:
        return None, "invalid_center_depth"
    return {
        "position": dict(center_point),
        "geometry": LANDMARK_GEOMETRY_BBOX_CENTER,
    }, None


def _project_bbox_center(
    *,
    observation: RawObservation,
    bbox: list[float],
    min_depth_m: float,
) -> dict[str, float] | None:
    depth, intrinsics, T_odom_cam = _projection_inputs(observation)
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return _project_pixel_to_odom(
        depth=depth,
        intrinsics=intrinsics,
        T_odom_cam=T_odom_cam,
        u=(x0 + x1) / 2.0,
        v=(y0 + y1) / 2.0,
        min_depth_m=min_depth_m,
    )


def _projection_inputs(observation: RawObservation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray(observation.depth, dtype=np.float32)
    intrinsics = np.asarray(observation.intrinsics, dtype=np.float32)
    T_cam_odom = np.asarray(observation.T_cam_odom, dtype=np.float32)
    T_odom_cam = np.linalg.inv(T_cam_odom)
    return depth, intrinsics, T_odom_cam


def _project_pixel_to_odom(
    *,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    T_odom_cam: np.ndarray,
    u: float,
    v: float,
    min_depth_m: float,
) -> dict[str, float] | None:
    depth_m = _pixel_depth_m(depth=depth, u=float(u), v=float(v), min_depth_m=min_depth_m)
    if depth_m is None:
        return None
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    x_cam = (float(u) - cx) * depth_m / fx
    y_cam = (float(v) - cy) * depth_m / fy
    point_cam = np.asarray([x_cam, y_cam, depth_m, 1.0], dtype=np.float32)
    point_odom = T_odom_cam @ point_cam
    return {
        "x": float(point_odom[0]),
        "y": float(point_odom[1]),
        "z": float(point_odom[2]),
    }


def _pixel_depth_m(*, depth: np.ndarray, u: float, v: float, min_depth_m: float) -> float | None:
    height, width = depth.shape
    px = int(round(float(u)))
    py = int(round(float(v)))
    if px < 0 or py < 0 or px >= width or py >= height:
        return None
    depth_m = float(depth[py, px])
    if not math.isfinite(depth_m) or depth_m < float(min_depth_m):
        return None
    return depth_m


def _dedupe_same_frame_landmarks(
    detections: list[dict[str, object]],
    *,
    landmark_memory: NavProbeLandmarkMemory,
    same_frame_iou_threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sorted_detections = sorted(
        detections,
        key=lambda item: (-float(item.get("score", 0.0)), str(item.get("det_id", ""))),
    )
    kept: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for detection in sorted_detections:
        duplicate_of = _same_frame_duplicate_of(
            detection=detection,
            kept=kept,
            landmark_memory=landmark_memory,
            same_frame_iou_threshold=same_frame_iou_threshold,
        )
        if duplicate_of is not None:
            skipped.append(
                {
                    "det_id": str(detection.get("det_id", "") or ""),
                    "duplicate_of_det_id": str(duplicate_of.get("det_id", "") or ""),
                    "reason": "same_frame_duplicate",
                }
            )
            continue
        kept.append(detection)
    return kept, skipped


def _same_frame_duplicate_of(
    *,
    detection: dict[str, object],
    kept: list[dict[str, object]],
    landmark_memory: NavProbeLandmarkMemory,
    same_frame_iou_threshold: float,
) -> dict[str, object] | None:
    bbox = [float(value) for value in list(detection.get("bbox", []))]
    position = dict(detection.get("position", {}))
    class_name = str(detection.get("class_name", "") or "")
    merge_distance_m = landmark_memory.merge_distance_for_class(class_name)
    for existing in kept:
        if str(existing.get("class_name", "") or "") != class_name:
            continue
        existing_bbox = [float(value) for value in list(existing.get("bbox", []))]
        if _bbox_iou(bbox, existing_bbox) >= same_frame_iou_threshold:
            return existing
        existing_position = dict(existing.get("position", {}))
        if landmark_position_distance_3d(position, existing_position) <= merge_distance_m:
            return existing
    return None


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(value) for value in a]
    bx0, by0, bx1, by1 = [float(value) for value in b]
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)

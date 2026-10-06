from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from navprobe.schemas import ActionResult
from navprobe.env.interface import RawDetection
from navprobe.runtime.cache import RuntimeCache

if TYPE_CHECKING:
    from navprobe.perception.detectors.interface import DetectorInterface
    from navprobe.env.interface import EnvInterface
    from navprobe.mapping.exploration.manager import ExplorationManager

DETECT_MAX_RESULTS_PER_CLASS = 2


def _normalize_detection_class_name(class_name: object) -> str:
    return str(class_name).strip().replace("_", " ")


def _normalize_detection_classes(class_names: list[str]) -> list[str]:
    query_classes: list[str] = []
    seen: set[str] = set()
    for item in class_names:
        normalized = _normalize_detection_class_name(item)
        if normalized == "" or normalized in seen:
            continue
        query_classes.append(normalized)
        seen.add(normalized)
    return query_classes


def get_obs(env: EnvInterface, cache: RuntimeCache, exploration: ExplorationManager) -> ActionResult:
    record = cache.store_observation(env.get_obs())
    frontier_update = exploration.observe_observation(cache=cache, obs_id=record.id)
    pose = record.observation.pose
    return ActionResult(
        ok=True,
        data={
            "obs_id": record.id,
            "pose": pose.to_dict(),
            "rgb_id": f"{record.id}:rgb",
            "depth_id": f"{record.id}:depth",
            "text_hint": record.observation.text_hint,
            "frontier_count": frontier_update["frontier_count"],
            "frontiers": frontier_update["frontiers"],
        },
        message=f"Captured observation {record.id}.",
    )


def detect(
    detector: DetectorInterface,
    cache: RuntimeCache,
    class_names: list[str],
    obs_id: str | None = None,
    min_score: float | None = None,
    canonical_class_map: dict[str, str] | None = None,
    agnostic_nms: bool | None = None,
    max_results: int | None = None,
) -> ActionResult:
    if obs_id is None:
        raise ValueError("detect requires obs_id")
    observation = cache.get_observation(obs_id).observation
    if observation.rgb is None:
        raise ValueError("detect requires observation.rgb")

    image_rgb = np.asarray(observation.rgb)
    query_classes = _normalize_detection_classes(class_names)
    if query_classes == []:
        raise ValueError("detect requires at least one non-empty class name")
    accepted_classes = set(query_classes)
    normalized_canonical_class_map = {
        _normalize_detection_class_name(key): _normalize_detection_class_name(value)
        for key, value in dict(canonical_class_map or {}).items()
    }

    def output_class_name(detector_class_name: object) -> str:
        normalized = _normalize_detection_class_name(detector_class_name)
        return normalized_canonical_class_map.get(normalized, normalized)

    detector_outputs = detector.detect_classes(
        image_rgb=image_rgb,
        class_names=query_classes,
        agnostic_nms=(len(query_classes) > 1) if agnostic_nms is None else bool(agnostic_nms),
    )
    detector_outputs = [
        item
        for item in detector_outputs
        if _normalize_detection_class_name(item.class_name) in accepted_classes
    ]
    if min_score is not None:
        threshold = float(min_score)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("detect min_score must be in [0, 1]")
        detector_outputs = [
            item
            for item in detector_outputs
            if float(item.score) >= threshold
        ]
    result_limit = int(DETECT_MAX_RESULTS_PER_CLASS) if max_results is None else int(max_results)
    detector_outputs = detector_outputs[:result_limit]
    raw_detections = [
        RawDetection(
            class_name=output_class_name(item.class_name),
            bbox=list(item.bbox),
            score=item.score,
            metadata={"detector_class_name": str(item.class_name)},
        )
        for item in detector_outputs
    ]
    records = cache.store_detections(obs_id=obs_id, detections=raw_detections)
    detection_payloads = []
    for record in records:
        payload = {
            "det_id": record.id,
            "class_name": record.detection.class_name,
            "detector_class_name": str(
                record.detection.metadata.get("detector_class_name", record.detection.class_name)
            ),
            "bbox": record.detection.bbox,
            "score": record.detection.score,
        }
        detection_payloads.append(payload)
    data = {
        "obs_id": obs_id,
        "detections": detection_payloads,
    }
    return ActionResult(
        ok=True,
        data=data,
        message=f"Detected {len(records)} objects for classes {query_classes}.",
    )

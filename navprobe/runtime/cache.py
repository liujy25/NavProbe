from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from navprobe.env.interface import RawDetection, RawObservation


@dataclass
class ObservationRecord:
    id: str
    observation: RawObservation


@dataclass
class DetectionRecord:
    id: str
    obs_id: str
    detection: RawDetection


@dataclass
class ImageRecord:
    id: str
    kind: str
    image: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class RuntimeCache:
    def __init__(self) -> None:
        self._obs_index = 0
        self._det_index = 0
        self._image_index = 0
        self.observations: dict[str, ObservationRecord] = {}
        self.detections: dict[str, DetectionRecord] = {}
        self.images: dict[str, ImageRecord] = {}
        self._lock = threading.RLock()

    def store_observation(self, observation: RawObservation) -> ObservationRecord:
        with self._lock:
            obs_id = f"obs_{self._obs_index}"
            self._obs_index += 1
            record = ObservationRecord(id=obs_id, observation=observation)
            self.observations[obs_id] = record
            return record

    def get_observation(self, obs_id: str) -> ObservationRecord:
        with self._lock:
            return self.observations[obs_id]

    def store_detections(
        self, obs_id: str, detections: list[RawDetection]
    ) -> list[DetectionRecord]:
        with self._lock:
            records: list[DetectionRecord] = []
            for detection in detections:
                det_id = f"det_{self._det_index}"
                self._det_index += 1
                record = DetectionRecord(id=det_id, obs_id=obs_id, detection=detection)
                self.detections[det_id] = record
                records.append(record)
            return records

    def get_detection(self, det_id: str) -> DetectionRecord:
        with self._lock:
            return self.detections[det_id]

    def store_image(
        self,
        image: Any,
        kind: str = "image",
        metadata: dict[str, Any] | None = None,
    ) -> ImageRecord:
        with self._lock:
            image_id = f"{kind}_{self._image_index}"
            self._image_index += 1
            record = ImageRecord(
                id=image_id,
                kind=kind,
                image=image,
                metadata=dict(metadata or {}),
            )
            self.images[image_id] = record
            return record

    def get_image(self, image_id: str) -> ImageRecord:
        with self._lock:
            return self.images[image_id]

    def discard_unreferenced_images(self, *, retained_ids: set[str], kinds: set[str]) -> None:
        """Drop only explicitly eligible images; never recycle their IDs."""
        with self._lock:
            stale_ids = [
                image_id for image_id, record in self.images.items()
                if record.kind in kinds and image_id not in retained_ids
            ]
            for image_id in stale_ids:
                del self.images[image_id]

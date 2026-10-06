from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection2D:
    class_name: str
    bbox: list[float]
    score: float


class DetectorInterface(ABC):
    @abstractmethod
    def detect(self, image_rgb: np.ndarray, class_name: str) -> list[Detection2D]:
        pass

    def detect_classes(
        self,
        image_rgb: np.ndarray,
        class_names: list[str],
        agnostic_nms: bool = False,
    ) -> list[Detection2D]:
        del agnostic_nms
        detections: list[Detection2D] = []
        seen: set[str] = set()
        for class_name in class_names:
            normalized = str(class_name).strip()
            if normalized == "" or normalized in seen:
                continue
            seen.add(normalized)
            detections.extend(self.detect(image_rgb=image_rgb, class_name=normalized))
        return detections

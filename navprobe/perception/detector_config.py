"""Detector selection, construction, and configuration metadata."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from navprobe.perception.detectors.interface import DetectorInterface


DETECTOR_YOLO_WORLD = "yolo_world"
DETECTOR_GROUNDING_DINO = "groundingdino"
DETECTOR_NAMES = (DETECTOR_YOLO_WORLD, DETECTOR_GROUNDING_DINO)


def create_landmark_detector(
    name: str, *, threshold: float, configuration: dict[str, Any],
) -> DetectorInterface:
    """Load only the configured detector implementation."""
    options = configuration
    if name == DETECTOR_YOLO_WORLD:
        from navprobe.perception.detectors import YOLOWorldLocalDetector

        return YOLOWorldLocalDetector(conf_threshold=threshold, **options["yolo_world"])
    if name == DETECTOR_GROUNDING_DINO:
        from navprobe.perception.detectors import GroundingDINOLocalDetector

        return GroundingDINOLocalDetector(box_threshold=threshold, **options["groundingdino"])
    raise ValueError(f"unsupported detector={name!r}")


def detector_configuration_from_args(args: argparse.Namespace) -> dict[str, Any]:
    landmark = args.landmark_detector
    landmark_threshold = float(
        args.landmark_detector_threshold
    )
    if not 0.0 <= landmark_threshold <= 1.0:
        raise ValueError(f"landmark_detector_threshold must be in [0, 1], got {landmark_threshold}")
    return {
        "landmark": {
            "name": landmark,
            "threshold": landmark_threshold,
            "enabled": True,
        },
    }

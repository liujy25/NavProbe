"""Detector interfaces and lazily loaded local implementations.

Model adapters load lazily so importing the interfaces does not load detector
dependencies. The selected adapter loads its dependencies when requested.
"""

from navprobe.perception.detectors.interface import Detection2D, DetectorInterface

__all__ = [
    "Detection2D",
    "DetectorInterface",
    "GroundingDINOLocalDetector",
    "YOLOWorldLocalDetector",
]


def __getattr__(name: str):
    if name == "GroundingDINOLocalDetector":
        from navprobe.perception.detectors.groundingdino_local import GroundingDINOLocalDetector

        return GroundingDINOLocalDetector
    if name == "YOLOWorldLocalDetector":
        from navprobe.perception.detectors.yoloworld_local import YOLOWorldLocalDetector

        return YOLOWorldLocalDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

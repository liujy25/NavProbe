"""Numeric and serialization helpers shared by local Habitat tasks."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


NAVPROBE_EPISODE_VECTOR_ATOL = 1e-4


def _yaw_degrees_from_rotation(rotation: np.ndarray) -> float:
    return float(math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))))


def _ros_quaternion_from_yaw_degrees(yaw: float) -> dict[str, float]:
    half = math.radians(float(yaw)) / 2.0
    return {
        "x": 0.0,
        "y": 0.0,
        "z": float(math.sin(half)),
        "w": float(math.cos(half)),
    }


def _serialize_metric_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _serialize_metric_value(metric_value)
            for key, metric_value in value.items()
        }
    if isinstance(value, list):
        return [_serialize_metric_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_metric_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _float_vector(value: Any, *, field_name: str, length: int) -> np.ndarray:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(
            f"{field_name} must be a list, tuple, or ndarray, got {type(value).__name__}"
        )
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(vector) != length:
        raise ValueError(f"{field_name} must have length {length}, got {len(vector)}")
    return vector


def _rotation_matches(loaded_rotation: Any, selected_rotation: Any) -> bool:
    loaded = _float_vector(loaded_rotation, field_name="loaded start_rotation", length=4)
    selected = _float_vector(selected_rotation, field_name="selected start_rotation", length=4)
    return bool(
        np.allclose(loaded, selected, rtol=0.0, atol=NAVPROBE_EPISODE_VECTOR_ATOL)
        or np.allclose(loaded, -selected, rtol=0.0, atol=NAVPROBE_EPISODE_VECTOR_ATOL)
    )


def _position_matches(loaded_position: Any, selected_position: Any) -> bool:
    loaded = _float_vector(loaded_position, field_name="loaded start_position", length=3)
    selected = _float_vector(selected_position, field_name="selected start_position", length=3)
    return bool(np.allclose(loaded, selected, rtol=0.0, atol=NAVPROBE_EPISODE_VECTOR_ATOL))


def _debug_sequence(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return value

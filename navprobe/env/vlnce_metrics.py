from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SUCCESS_DISTANCE_M = 3.0
NDTW_MEASURE_TYPE = "NavProbeNDTW"
ORACLE_SUCCESS_MEASURE_TYPE = "NavProbeOracleSuccess"
SDTW_MEASURE_TYPE = "NavProbeSDTW"


def resolve_official_vlnce_gt_path(dataset_file: str | Path) -> Path:
    dataset_path = Path(dataset_file).expanduser().resolve()
    if not dataset_path.name.endswith(".json.gz"):
        raise ValueError(f"VLN-CE dataset file must end with .json.gz: {dataset_path}")

    direct_path = dataset_path.with_name(f"{dataset_path.name[:-8]}_gt.json.gz")
    candidates = [direct_path]

    dataset_root = dataset_path.parent.parent
    if "r2r" in dataset_root.name.lower():
        preprocessed_name = dataset_root.name
        if not preprocessed_name.endswith("_preprocessed"):
            preprocessed_name = f"{preprocessed_name}_preprocessed"
        candidates.append(
            dataset_root.with_name(preprocessed_name)
            / dataset_path.parent.name
            / f"{dataset_path.parent.name}_gt.json.gz"
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Official dense VLN-CE ground-truth file is required for online nDTW. "
        f"dataset={dataset_path}, expected_one_of={[str(path) for path in candidates]}"
    )


def normalized_dtw(
    agent_locations: list[list[float]] | np.ndarray,
    reference_locations: list[list[float]] | np.ndarray,
    *,
    success_distance_m: float = SUCCESS_DISTANCE_M,
) -> float:
    agent = np.asarray(agent_locations, dtype=np.float64)
    reference = np.asarray(reference_locations, dtype=np.float64)
    if agent.ndim != 2 or agent.shape[1] != 3 or len(agent) == 0:
        raise ValueError("agent_locations must be a non-empty Nx3 trajectory")
    if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) == 0:
        raise ValueError("reference_locations must be a non-empty Nx3 trajectory")
    if success_distance_m <= 0.0:
        raise ValueError("success_distance_m must be positive")

    # FastDTW defines the alignment distance for nDTW.
    # Changing the alignment algorithm can change the reported scores.
    from fastdtw import fastdtw

    dtw_distance = float(fastdtw(
        agent, reference,
        dist=lambda first, second: float(np.linalg.norm(first - second)),
    )[0])
    return float(math.exp(-dtw_distance / (len(reference) * success_distance_m)))


@lru_cache(maxsize=2)
def _decode_dense_gt(contents: bytes) -> dict[str, Any]:
    # Cache decompression/parsing by the same file contents used for experiment
    # identity, not by a path that can be reused for a different GT file.
    payload = json.loads(gzip.decompress(contents))
    if not isinstance(payload, dict):
        raise ValueError("VLN-CE dense ground truth must be a JSON object")
    return payload


def _load_dense_gt(gt_path: str) -> dict[str, Any]:
    # Called once when constructing a measure, never on metric updates.
    return _decode_dense_gt(Path(gt_path).read_bytes())


def register_vlnce_metric_measures() -> None:
    from habitat.core.embodied_task import Measure
    from habitat.core.registry import registry

    if registry.get_measure(NDTW_MEASURE_TYPE) is None:

        class NavProbeNDTW(Measure):
            cls_uuid = "ndtw"

            def __init__(self, sim: Any, config: Any, *args: Any, **kwargs: Any) -> None:
                self._sim = sim
                self._success_distance_m = float(config.success_distance)
                self._gt_path = str(Path(config.gt_path).expanduser().resolve())
                self._ground_truth = _load_dense_gt(self._gt_path)
                self._reference_locations: list[list[float]] = []
                self._agent_locations: list[list[float]] = []
                super().__init__()

            def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
                return self.cls_uuid

            def reset_metric(self, episode: Any, *args: Any, **kwargs: Any) -> None:
                episode_id = str(episode.episode_id)
                record = self._ground_truth.get(episode_id)
                if not isinstance(record, dict) or not isinstance(record.get("locations"), list):
                    raise KeyError(
                        f"Episode {episode_id!r} has no locations in VLN-CE dense GT {self._gt_path}"
                    )
                self._reference_locations = record["locations"]
                self._agent_locations = []
                self.update_metric(episode=episode, *args, **kwargs)

            def update_metric(self, *args: Any, **kwargs: Any) -> None:
                current_position = [
                    float(value) for value in self._sim.get_agent_state().position
                ]
                if self._agent_locations != [] and self._agent_locations[-1] == current_position:
                    return
                self._agent_locations.append(current_position)
                self._metric = normalized_dtw(
                    self._agent_locations,
                    self._reference_locations,
                    success_distance_m=self._success_distance_m,
                )

        registry.register_measure(NavProbeNDTW, name=NDTW_MEASURE_TYPE)

    if registry.get_measure(ORACLE_SUCCESS_MEASURE_TYPE) is None:

        class NavProbeOracleSuccess(Measure):
            cls_uuid = "oracle_success"

            def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
                self._success_distance_m = float(config.success_distance)
                super().__init__()

            def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
                return self.cls_uuid

            def reset_metric(self, episode: Any, task: Any, *args: Any, **kwargs: Any) -> None:
                task.measurements.check_measure_dependencies(
                    self.uuid,
                    ["distance_to_goal"],
                )
                self._metric = 0.0
                self.update_metric(episode=episode, task=task, *args, **kwargs)

            def update_metric(self, episode: Any, task: Any, *args: Any, **kwargs: Any) -> None:
                distance_to_goal = task.measurements.measures[
                    "distance_to_goal"
                ].get_metric()
                self._metric = float(
                    bool(self._metric)
                    or float(distance_to_goal) < self._success_distance_m
                )

        registry.register_measure(
            NavProbeOracleSuccess,
            name=ORACLE_SUCCESS_MEASURE_TYPE,
        )

    if registry.get_measure(SDTW_MEASURE_TYPE) is None:

        class NavProbeSDTW(Measure):
            cls_uuid = "sdtw"

            def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
                return self.cls_uuid

            def reset_metric(self, episode: Any, task: Any, *args: Any, **kwargs: Any) -> None:
                task.measurements.check_measure_dependencies(self.uuid, ["success", "ndtw"])
                self.update_metric(task=task)

            def update_metric(self, task: Any, *args: Any, **kwargs: Any) -> None:
                measures = task.measurements.measures
                self._metric = float(measures["success"].get_metric()) * float(
                    measures["ndtw"].get_metric()
                )

        registry.register_measure(NavProbeSDTW, name=SDTW_MEASURE_TYPE)


def configure_vlnce_metrics(config: Any, *, dataset_file: str | Path) -> Path:
    import habitat
    from habitat.config.default_structured_configs import MeasurementConfig
    from omegaconf import open_dict

    @dataclass
    class NDTWMeasurementConfig(MeasurementConfig):
        type: str = NDTW_MEASURE_TYPE
        gt_path: str = ""
        success_distance: float = SUCCESS_DISTANCE_M

    @dataclass
    class OracleSuccessMeasurementConfig(MeasurementConfig):
        type: str = ORACLE_SUCCESS_MEASURE_TYPE
        success_distance: float = SUCCESS_DISTANCE_M

    @dataclass
    class SDTWMeasurementConfig(MeasurementConfig):
        type: str = SDTW_MEASURE_TYPE

    # Fail before loading a simulator, with an install command for the fixed
    # metric dependency rather than changing the evaluation algorithm.
    try:
        import fastdtw  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install NavProbe metric dependencies: pip install -r requirements-habitat.txt") from exc
    gt_path = resolve_official_vlnce_gt_path(dataset_file)
    register_vlnce_metric_measures()
    # Habitat's structured config marks the measurement dictionary as closed.
    # Opening only this mapping keeps the patch usable both inside and outside
    # a caller's ``habitat.config.read_write`` context.
    with habitat.config.read_write(config):
        with open_dict(config.habitat.task.measurements):
            config.habitat.task.measurements["ndtw"] = NDTWMeasurementConfig(
                gt_path=str(gt_path)
            )
            config.habitat.task.measurements[
                "oracle_success"
            ] = OracleSuccessMeasurementConfig()
            config.habitat.task.measurements["sdtw"] = SDTWMeasurementConfig()
    return gt_path


def add_vln_metric_aliases(metrics: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    if "success" in result:
        result["sr"] = result["success"]
    if "distance_to_goal" in result:
        result["ne"] = result["distance_to_goal"]
    if "oracle_success" in result:
        result["osr"] = result["oracle_success"]
    return result

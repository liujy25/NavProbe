"""Configuration-scoped batch results shared by VLNCE and HM3D runners."""
from __future__ import annotations

import argparse
from datetime import datetime
import time
import hashlib
import json
import math
import os
import platform
import traceback
import warnings
from pathlib import Path
from statistics import mean
from typing import Any

from navprobe.agent.vertical_config import DEFAULT_VERTICAL_METHOD, vertical_execution_configuration
from navprobe.perception.detector_config import detector_configuration_from_args
from navprobe.runtime.source_identity import source_digest, runtime_dependency_identity
from navprobe.config.settings import configuration_for_args
from navprobe.config.episode_config import _sanitize_name


def _identity(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def experiment_configuration(
    args: argparse.Namespace,
    *,
    runner: str,
    episode_payloads: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config = configuration_for_args(args)
    configuration = {
        "configuration_version": 4,
        "runner": runner,
        "source_digest": source_digest(),
        "vertical_execution": vertical_execution_configuration(
            getattr(args, "vertical_method", DEFAULT_VERTICAL_METHOD)
        ),
        "detectors": detector_configuration_from_args(args),
    }
    if episode_payloads is not None:
        # Hash the selected decoded inputs, including shared dataset metadata.
        # Formatting/compression changes do not change episode behavior.
        configuration["dataset_episodes"] = {
            str(index): _identity(payload) for index, payload in sorted(episode_payloads.items())
        }
    effective = config.to_dict()
    effective.pop("output")
    effective.pop("runtime")
    # Endpoint identity excludes potentially credential-bearing URLs.
    effective["model"]["base_url"] = {"sha256": _identity(config.model.base_url)}
    configuration["navprobe"] = effective
    inputs = {"habitat_config": effective["environment"]["habitat_config"],
              "episode_file": effective["dataset"]["episode_file"]}
    selection = {key: getattr(args, key) for key in (
        "goal", "episode_id", "dataset_index", "scene", "scene_id",
        "scene_name", "episode_index",
    ) if getattr(args, key, None) is not None}
    if selection:
        configuration["episode_selection"] = selection
    perception = config.perception
    detector = perception["detector"]
    configuration["runtime_dependencies"] = runtime_dependency_identity(detector)
    for key, value in perception[detector].items():
        if key.endswith("path"):
            inputs[f"{detector}.{key}"] = value
    configuration["resources"] = {name: _resource_identity(value) for name, value in inputs.items()}
    if effective["dataset"]["type"] == "R2RVLN-v1":
        from navprobe.env.vlnce_metrics import resolve_official_vlnce_gt_path

        dataset_file = effective["dataset"]["data_path"].format(split=effective["dataset"]["split"])
        gt_path = resolve_official_vlnce_gt_path(dataset_file)
        configuration["resources"]["vlnce_dense_gt"] = _resource_identity(gt_path)
    return {
        "configuration_id": _identity(configuration),
        "resolved_configuration": configuration,
        "runtime_platform": {"system": platform.system(), "machine": platform.machine()},
    }


def _resource_identity(value):
    path = Path(value)
    digest = hashlib.sha256()
    if not path.is_file():
        return {"path": str(path), "available": False}
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def record_configuration(
    result: dict[str, Any], configuration: dict[str, Any], request: dict[str, Any]
) -> None:
    result["configuration_id"] = configuration["configuration_id"]
    result["batch_request"] = dict(request)
    result["batch_episode_id"] = _identity(request)


def write_result_json(path: Path, result: dict[str, Any]) -> None:
    """Publish only complete JSON; interrupted writes never enter aggregation."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


_SUMMARY_METRICS = ("spl", "soft_spl", "distance_to_goal", "ndtw", "sdtw", "oracle_success")


class NavProbeBatchResults:
    """Selected results for one batch writer; reload disk state on each new run.

    Cache only the latest completed or failed attempt for each episode.
    Timestamped filenames order attempts across both result directories;
    historical records remain on disk without contributing stale outcomes.
    """

    def __init__(self, results_dir: Path, *, configuration: dict[str, Any],
                 requests: list[dict[str, Any]]) -> None:
        self.results_dir = results_dir
        self.configuration = configuration
        self.selected = {_identity(request) for request in requests}
        self._latest: dict[str, tuple[Path, dict[str, Any]]] = {}
        self.invalid_records: list[dict[str, str]] = []
        self._episode_prefixes = {
            _identity(request): _episode_log_path(results_dir, request, "").stem
            for request in requests if "scene_key" in request
        }
        for directory, include in ((results_dir, self._include_result),
                                   (results_dir / "errors", self._include_error)):
            for path in sorted(directory.glob("*.json")):
                if directory == results_dir and path.name == "summary.json":
                    continue
                try:
                    result = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    self._invalid_record(path, str(exc))
                    continue
                if not isinstance(result, dict):
                    self._invalid_record(path, "expected a result object")
                    continue
                if result.get("configuration_id") != self.configuration["configuration_id"]:
                    continue
                try:
                    include(path, result)
                except (KeyError, TypeError, ValueError) as exc:
                    self._invalid_record(path, str(exc), episode_id=result.get("batch_episode_id"))

    def _invalid_record(self, path: Path, message: str, *, episode_id: str | None = None) -> None:
        diagnostic = {"path": str(path), "error": message}
        self.invalid_records.append(diagnostic)
        warnings.warn(f"Invalid batch record {path}: {message}; rerun its episode", RuntimeWarning)
        affected = ([episode_id] if isinstance(episode_id, str) and episode_id in self.selected else [
            identity for identity, prefix in self._episode_prefixes.items()
            if path.stem.startswith(prefix)
        ])
        # An unreadable latest attempt supersedes an older completion. Files
        # with no identifiable episode are diagnosed without guessing an identity.
        for identity in affected:
            self._include_attempt(identity, path, {
                "status": "error", "error_type": "InvalidResultRecord",
                "error": message, "error_log": str(path),
            })

    def _include_result(self, path: Path, result: dict[str, Any]) -> None:
        # Reuse results only when their configuration matches this batch.
        if result.get("configuration_id") != self.configuration["configuration_id"]:
            return
        episode_id = result.get("batch_episode_id")
        if episode_id not in self.selected:
            return
        metrics = result["metrics"]
        if not isinstance(metrics, dict):
            raise ValueError("metrics must be an object")
        if type(result["episode_success"]) is not bool:
            raise ValueError("episode_success must be a boolean")
        for key in ("num_steps", "num_place_steps"):
            if type(result[key]) is not int or result[key] < 0:
                raise ValueError(f"{key} must be a non-negative integer")
        for key in _SUMMARY_METRICS:
            if metrics.get(key) is not None and type(metrics[key]) not in (int, float):
                raise ValueError(f"metrics.{key} must be numeric or null")
        self._include_attempt(episode_id, path, {
            "status": "completed",
            "episode_success": result["episode_success"],
            "num_steps": result["num_steps"],
            "num_place_steps": result["num_place_steps"],
            "metrics": {key: metrics.get(key) for key in _SUMMARY_METRICS},
        })

    def _include_error(self, path: Path, result: dict[str, Any]) -> None:
        if result.get("configuration_id") != self.configuration["configuration_id"]:
            return
        request = result["batch_request"]
        if not isinstance(request, dict):
            raise ValueError("batch_request must be an object")
        episode_id = _identity(request)
        if episode_id in self.selected:
            self._include_attempt(episode_id, path, {
                "status": "error",
                "batch_request": dict(request),
                "error_type": result.get("error_type"),
                "error": result.get("error"),
                "error_log": str(path),
            })

    def _include_attempt(self, episode_id: str, path: Path, result: dict[str, Any]) -> None:
        previous = self._latest.get(episode_id)
        # Compare filenames, not parent directories. If an attempt has both
        # records, its error must not be hidden by the completion record.
        order = (path.name, result["status"] == "error")
        if previous is None or order >= (previous[0].name, previous[1]["status"] == "error"):
            self._latest[episode_id] = (path, result)

    def matching_episode_logs(self, request: dict[str, Any]) -> list[Path]:
        latest = self._latest.get(_identity(request))
        return [latest[0]] if latest is not None and latest[1]["status"] == "completed" else []

    def write_episode_result(self, result_path: Path, result: dict[str, Any], *,
                             request: dict[str, Any]) -> None:
        record_configuration(result, self.configuration, request)
        # Publish the detailed result before its independent batch record. A failed
        # write cannot make a partial episode resumable or enter aggregation.
        episode_path = Path(str(result["run_dir"])) / "summary.json"
        write_result_json(episode_path, result)
        write_result_json(result_path, {
            "status": "completed",
            "attempt_id": result_path.stem,
            "configuration_id": result["configuration_id"],
            "batch_episode_id": result["batch_episode_id"],
            "batch_request": result["batch_request"],
            "episode_success": result["episode_success"],
            "num_steps": result["num_steps"],
            "num_place_steps": result["num_place_steps"],
            "metrics": dict(result["metrics"]),
            "result_ref": os.path.relpath(episode_path, result_path.parent),
        })
        self._include_result(result_path, result)

    def write_error(self, result_path: Path, error: Exception, *,
                    request: dict[str, Any], metadata: dict[str, Any] | None = None) -> None:
        result = {
            **(metadata or {}),
            "status": "error",
            "attempt_id": result_path.stem,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        if isinstance(getattr(error, "navprobe_step_error", None), dict):
            result["interrupted_step"] = error.navprobe_step_error
        record_configuration(result, self.configuration, request)
        error_dir = result_path.parent / "errors"
        error_dir.mkdir(parents=True, exist_ok=True)
        error_path = error_dir / result_path.name
        write_result_json(error_path, result)
        self._include_error(error_path, result)

    def write_summary(self, start_time: str, end_time: str, elapsed_seconds: float) -> None:
        attempts = [result for _, result in sorted(self._latest.values(), key=lambda item: item[0].name)]
        results = [result for result in attempts if result["status"] == "completed"]
        errors = [result for result in attempts if result["status"] == "error"]
        summary = {
            **self.configuration,
            "start_time": start_time,
            "end_time": end_time,
            "elapsed_seconds": float(elapsed_seconds),
            "num_episodes": len(results),
            "num_requested_episodes": len(self.selected),
            "num_error_episodes": len(errors),
            "error_episodes": errors,
            "invalid_records": self.invalid_records,
            "episode_success_rate": mean(bool(item["episode_success"]) for item in results) if results else 0.0,
            "avg_num_steps": mean(int(item["num_steps"]) for item in results) if results else 0.0,
            "avg_num_place_steps": mean(int(item["num_place_steps"]) for item in results) if results else 0.0,
        }
        for key in _SUMMARY_METRICS:
            values = [float(item["metrics"][key]) for item in results if item["metrics"].get(key) is not None]
            summary[f"avg_{key}"] = mean(values) if values else None
        # Keep incremental means, but expose their populations and an explicit
        # completion condition. Errors/missing values are never scored as zero.
        required = ["spl"]
        if self.configuration["resolved_configuration"]["navprobe"]["dataset"]["type"] == "R2RVLN-v1":
            required += ["distance_to_goal", "oracle_success", "ndtw"]
        coverage = {
            key: sum(item["metrics"].get(key) is not None and math.isfinite(float(item["metrics"][key]))
                     for item in results)
            for key in _SUMMARY_METRICS
        }
        summary.update(
            num_missing_episodes=len(self.selected) - len(results) - len(errors),
            metric_coverage=coverage,
            required_metrics=["success", *required],
            evaluation_complete=bool(self.selected) and len(results) == len(self.selected)
                and all(coverage[key] == len(self.selected) for key in required),
        )
        coverage["success"] = len(results)
        write_result_json(self.results_dir / "summary.json", summary)


class NavProbeBatchTimer:
    """One batch's wall-clock labels and monotonic elapsed time."""

    def __init__(self) -> None:
        self.start_time = datetime.now().isoformat()
        self.start_perf = time.perf_counter()

    def write_summary(self, results: NavProbeBatchResults) -> None:
        results.write_summary(
            start_time=self.start_time, end_time=datetime.now().isoformat(),
            elapsed_seconds=time.perf_counter() - self.start_perf,
        )


def _run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _episode_log_path(results_dir: Path, request: dict[str, Any], run_stamp: str) -> Path:
    scene_key = _sanitize_name(str(request["scene_key"]))
    episode_id = str(request["episode_id"])
    episode_id = f"{int(episode_id):04d}" if episode_id.isdigit() else _sanitize_name(episode_id)
    return results_dir / f"episode_{episode_id}_{scene_key}_{run_stamp}.json"

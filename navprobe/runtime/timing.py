from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import time
from typing import Any


@dataclass
class TimingStageRecord:
    name: str
    elapsed_seconds: float
    env_step_before: int
    env_step_after: int
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def env_step_delta(self) -> int:
        return int(self.env_step_after) - int(self.env_step_before)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "elapsed_seconds": float(self.elapsed_seconds),
            "env_step_before": int(self.env_step_before),
            "env_step_after": int(self.env_step_after),
            "env_step_delta": int(self.env_step_delta),
        }
        if self.details != {}:
            payload["details"] = dict(self.details)
        return payload


class StepTimingRecorder:
    def __init__(self, env_step_start: int) -> None:
        self.env_step_start = int(env_step_start)
        self._step_wall_start = time.perf_counter()
        self._active_name: str | None = None
        self._active_wall_start: float | None = None
        self._active_env_step_start: int | None = None
        self._stages: list[TimingStageRecord] = []

    def start(self, name: str, env_step: int) -> None:
        if self._active_name is not None:
            raise ValueError(f"Timing stage already active: {self._active_name}")
        self._active_name = str(name)
        self._active_wall_start = time.perf_counter()
        self._active_env_step_start = int(env_step)

    def stop(self, name: str, env_step: int, details: dict[str, Any] | None = None) -> TimingStageRecord:
        if self._active_name is None:
            raise ValueError(f"Timing stage was not started: {name}")
        if self._active_name != str(name):
            raise ValueError(f"Stopping timing stage {name}, but active stage is {self._active_name}")
        if self._active_wall_start is None or self._active_env_step_start is None:
            raise ValueError(f"Timing stage has incomplete active state: {name}")
        record = TimingStageRecord(
            name=str(name),
            elapsed_seconds=time.perf_counter() - float(self._active_wall_start),
            env_step_before=int(self._active_env_step_start),
            env_step_after=int(env_step),
            details={} if details is None else dict(details),
        )
        self._stages.append(record)
        self._active_name = None
        self._active_wall_start = None
        self._active_env_step_start = None
        return record

    def to_dict(self, env_step_end: int) -> dict[str, object]:
        if self._active_name is not None:
            raise ValueError(f"Timing stage still active: {self._active_name}")
        env_step_end_int = int(env_step_end)
        return {
            "step_total": {
                "elapsed_seconds": float(time.perf_counter() - self._step_wall_start),
                "env_step_before": int(self.env_step_start),
                "env_step_after": env_step_end_int,
                "env_step_delta": env_step_end_int - int(self.env_step_start),
            },
            "stages": [stage.to_dict() for stage in self._stages],
        }


class WallStepTimer:
    def __init__(self, env_step_start: int) -> None:
        self.env_step_start = int(env_step_start)
        self.wall_start = time.perf_counter()

    def finish(self, env_step_end: int) -> dict[str, object]:
        env_step_end_int = int(env_step_end)
        return {
            "elapsed_seconds": float(time.perf_counter() - self.wall_start),
            "env_step_before": int(self.env_step_start),
            "env_step_after": env_step_end_int,
            "env_step_delta": env_step_end_int - int(self.env_step_start),
        }


def summarize_step_timings(step_summaries: Iterable[dict[str, object]]) -> dict[str, object]:
    stage_totals: dict[str, dict[str, float]] = {}
    step_total_elapsed = 0.0
    step_total_env_delta = 0
    step_total_count = 0
    for step in step_summaries:
        timing = step.get("timing")
        if not isinstance(timing, dict):
            continue
        step_total = timing.get("step_total")
        if isinstance(step_total, dict):
            step_total_count += 1
            step_total_elapsed += float(step_total.get("elapsed_seconds", 0.0))
            step_total_env_delta += int(step_total["env_step_delta"])
        stages = timing.get("stages")
        if not isinstance(stages, list):
            continue
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            name = str(stage.get("name", "unknown"))
            if name not in stage_totals:
                stage_totals[name] = {
                    "count": 0.0,
                    "total_elapsed_seconds": 0.0,
                    "total_env_step_delta": 0.0,
                }
            stage_totals[name]["count"] += 1.0
            stage_totals[name]["total_elapsed_seconds"] += float(stage.get("elapsed_seconds", 0.0))
            stage_totals[name]["total_env_step_delta"] += float(
                stage["env_step_delta"]
            )

    stages_payload: dict[str, object] = {}
    for name, totals in stage_totals.items():
        count = int(totals["count"])
        total_elapsed = float(totals["total_elapsed_seconds"])
        total_env_delta = int(totals["total_env_step_delta"])
        stages_payload[name] = {
            "count": count,
            "total_elapsed_seconds": total_elapsed,
            "avg_elapsed_seconds": total_elapsed / count if count > 0 else 0.0,
            "total_env_step_delta": total_env_delta,
            "avg_env_step_delta": total_env_delta / count if count > 0 else 0.0,
        }
    return {
        "step_total": {
            "count": int(step_total_count),
            "total_elapsed_seconds": float(step_total_elapsed),
            "avg_elapsed_seconds": step_total_elapsed / step_total_count if step_total_count > 0 else 0.0,
            "total_env_step_delta": int(step_total_env_delta),
            "avg_env_step_delta": (
                step_total_env_delta / step_total_count if step_total_count > 0 else 0.0
            ),
        },
        "stages": stages_payload,
    }

"""Read completed steps for timing and trajectory aggregation."""
from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path


def iter_step_summaries(steps_dir: Path, count: int) -> Iterator[dict]:
    """Read the first ``count`` completed step summaries in step-index order."""
    for index in range(count):
        with (steps_dir / f"{index:04d}" / "step.json").open(encoding="utf-8") as stream:
            yield json.load(stream)

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any


def _resolve_reset_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    reset_kwargs: dict[str, Any] = {}
    if args.goal is not None:
        reset_kwargs["goal"] = args.goal
    if args.episode_id is not None:
        reset_kwargs["episode_id"] = int(args.episode_id)
    if args.dataset_index is not None:
        reset_kwargs["dataset_index"] = int(args.dataset_index)
    if args.scene is not None:
        reset_kwargs["scene_name"] = str(args.scene)
    if getattr(args, "scene_id", None) is not None:
        reset_kwargs["scene_id"] = str(args.scene_id)
    return reset_kwargs


def _run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _sanitize_name(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value).strip())
    return sanitized.strip("_") or "run"

from __future__ import annotations

import argparse
from navprobe.runners.batch_results import experiment_configuration
from pathlib import Path
from typing import Any

from navprobe.env.hm3dv2_dataset import resolve_hm3dv2_episode_selection
from navprobe.env.hm3dv2_env import HM3Dv2Env
from navprobe.env.habitat_env import close_environment
from navprobe.runners.episode_resources import _release_episode_memory
from navprobe.runners.preflight import validate_runtime_resources
from navprobe.agent.runner import run_navprobe_agent_episode
from navprobe.runtime.panorama_config import hm3dv2_panorama_config


def _agent_runner_args(args: argparse.Namespace) -> argparse.Namespace:
    runner_args = argparse.Namespace(**vars(args))
    runner_args.episode_id = None
    runner_args.dataset_index = None
    runner_args.scene = None
    runner_args.scene_id = None
    runner_args.record_dir = str(Path(str(args.record_dir)).expanduser().resolve())
    # Reset supplies the selected dataset's target category.
    runner_args.goal = None
    return runner_args


def run_hm3d_agent(
    args: argparse.Namespace, *, episode_payload: dict[str, Any] | None = None,
    global_episode_id: int | None = None,
) -> dict[str, Any]:
    selection = resolve_hm3dv2_episode_selection(
        data_path=args.data_path,
        scenes_dir=getattr(args, "scenes_dir", None),
        split=getattr(args, "split", "val"),
        scene_name=args.scene_name,
        episode_index=args.episode_index,
        episode_id=args.episode_id,
        episode_payload=episode_payload,
        global_episode_id=global_episode_id,
    )
    if getattr(args, "dry_run", False):
        preview = {"dry_run": True, "episode_id": str(selection.global_episode_id),
                   "scene_key": selection.scene_key, "episode_index": selection.episode_index,
                   "source_episode_id": selection.source_episode_id,
                   "goal": selection.goal}
        print(f"episode_id={preview['episode_id']} scene={preview['scene_key']} "
              f"local_episode_index={preview['episode_index']} goal={preview['goal']}")
        return preview
    configuration = getattr(args, "experiment_configuration", None)
    if configuration is None:
        validate_runtime_resources(args)
        configuration = experiment_configuration(args, runner="hm3d", episode_payloads={
            selection.global_episode_id: {**selection.dataset_metadata, "episodes": [selection.episode]},
        })
    env = None
    try:
        env = HM3Dv2Env(
            seed=args.seed,
            selection=selection,
            habitat_config=args.habitat_config, dataset_type=args.dataset_type,
            scenes_dir=getattr(args, "scenes_dir", None),
        )
        runner_args = _agent_runner_args(args)
        runner_args.experiment_configuration = configuration
        return run_navprobe_agent_episode(
            runner_args,
            env=env,
            panorama_config=hm3dv2_panorama_config(),
        )
    finally:
        close_environment(env, label=f"episode={selection.global_episode_id}")
        env = None
        _release_episode_memory(str(selection.global_episode_id))

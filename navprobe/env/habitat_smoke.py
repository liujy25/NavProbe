"""Check Habitat integration without a model service or detector weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from navprobe.env.habitat_adapter import (
    NavProbeHabitatAdapter,
    configure_habitat_scene_paths,
    create_configured_habitat_env,
)
from navprobe.env.vlnce_metrics import (
    NDTW_MEASURE_TYPE, ORACLE_SUCCESS_MEASURE_TYPE, SDTW_MEASURE_TYPE,
    configure_vlnce_metrics, normalized_dtw, register_vlnce_metric_measures,
)


# This module lives inside ``navprobe/env``; the package root already contains
# the shipped ``config`` and ``data`` directories.
CONFIG_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-install", action="store_true", help="Check imports, shipped configs and metric registration; no assets or GPU context needed")
    parser.add_argument("--config", default=None, help="Optional Habitat YAML override")
    parser.add_argument("--dataset-file", help="VLNCE split or ObjectNav split/content .json.gz")
    parser.add_argument("--scenes-dir", default="data/scene_datasets")
    parser.add_argument("--split", default=None)
    parser.add_argument("--scene", default=None, help="ObjectNav content scene key; avoids loading every scene")
    parser.add_argument("--task", choices=["vlnce", "objectnav"], default="vlnce")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--gpu-device-id", type=int, default=None)
    return parser


def check_install() -> dict[str, Any]:
    import habitat
    import habitat_sim
    from habitat.core.registry import registry
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    register_vlnce_metric_measures()
    names = (NDTW_MEASURE_TYPE, ORACLE_SUCCESS_MEASURE_TYPE, SDTW_MEASURE_TYPE)
    for name in names:
        if registry.get_measure(name) is None:
            raise RuntimeError(f"Unregistered metric: {name}")
    for relative in ("config/habitat_r2r_vln_navprobe.yaml", "config/habitat_hm3d_objectnav.yaml"):
        config = habitat.get_config(str(CONFIG_ROOT / relative))
        for measurement in config.habitat.task.measurements.values():
            if registry.get_measure(measurement.type) is None:
                raise RuntimeError(f"Unknown Habitat measure: {measurement.type}")
    if normalized_dtw([[0, 0, 0]], [[0, 0, 0]]) != 1.0:
        raise RuntimeError("nDTW self-trajectory check failed")
    return {
        "check": "installation_only",
        "habitat_lab": habitat.__version__, "habitat_sim": habitat_sim.__version__,
        "planner": ShortestPathFollower.__name__, "registered_metrics": names,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.check_install:
        return check_install()
    if args.dataset_file is None:
        raise ValueError("Supply --dataset-file for a real scene test, or use --check-install")
    if args.episodes < 1 or args.episode_index < 0:
        raise ValueError("--episodes must be positive; --episode-index must be nonnegative")

    import habitat
    from navprobe.env.habitat_utils import _serialize_metric_value
    from navprobe.env.vlnce_dataset import habitat_compatible_vlnce_dataset_file

    default = "config/habitat_r2r_vln_navprobe.yaml" if args.task == "vlnce" else "config/habitat_hm3d_objectnav.yaml"
    config_path = Path(args.config).expanduser().resolve() if args.config else CONFIG_ROOT / default
    config = habitat.get_config(str(config_path))
    dataset_file = Path(args.dataset_file).expanduser().resolve()
    with habitat.config.read_write(config):
        config.habitat.dataset.data_path = str(
            habitat_compatible_vlnce_dataset_file(dataset_file)
            if args.task == "vlnce" else dataset_file
        )
        if args.split is not None:
            config.habitat.dataset.split = args.split
        configure_habitat_scene_paths(config, args.scenes_dir)
        if args.task == "vlnce":
            configure_vlnce_metrics(config, dataset_file=dataset_file)
        if args.gpu_device_id is not None:
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.gpu_device_id
        config.habitat.environment.iterator_options.shuffle = False
        if args.scene is not None:
            from habitat.core.registry import registry
            from navprobe.env.hm3dv2_dataset import normalize_scene_name

            dataset_type = registry.get_dataset(config.habitat.dataset.type)
            # Split files use content-file keys; standalone scene files use
            # Habitat's scene stem (e.g. HM3D's "scene.basis").
            scene_keys = dataset_type.get_scenes_to_load(config.habitat.dataset)
            selected_keys = [key for key in scene_keys
                             if normalize_scene_name(key) == normalize_scene_name(args.scene)]
            if not selected_keys:
                raise ValueError(f"Scene {args.scene!r} is not present in {dataset_file}")
            config.habitat.dataset.content_scenes = selected_keys

    dataset = habitat.make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    selected = dataset.episodes[args.episode_index:args.episode_index + args.episodes]
    if len(selected) != args.episodes:
        raise ValueError(f"Requested {args.episodes} episodes at index {args.episode_index}, dataset has {len(dataset.episodes)}")
    dataset.episodes = selected
    for episode in selected:
        if not Path(episode.scene_id).is_file():
            raise FileNotFoundError(f"Scene asset missing: {episode.scene_id}")

    env = create_configured_habitat_env(config, dataset=dataset)
    results = []
    try:
        adapter = NavProbeHabitatAdapter(env=env)
        for _ in range(args.episodes):
            obs = env.reset()
            adapter.ground_height_offset = float(env.sim.get_agent_state().position[1])
            rgb, depth = adapter._extract_images(obs)
            camera, base = adapter._get_transform_matrices(obs)
            if not (np.isfinite(camera).all() and np.isfinite(base).all() and rgb.ndim == 3 and depth.ndim == 2):
                raise RuntimeError("Invalid RGB-D or camera pose")
            _, _, turn_info = adapter.execute_discrete_action("turn_left")
            # A short forward waypoint exercises the same planner and frame
            # conversion as NavProbe. Navigation success is not a smoke criterion.
            _, base = adapter._get_transform_matrices()
            target = base[:3, 3] + base[:3, 0] * 0.5
            goal_pose = {"pose": {"position": dict(zip(("x", "y", "z"), map(float, target)))}}
            _, _, nav_info = adapter.execute_goal_pose(goal_pose, max_steps=8)
            if env.episode_over:
                raise RuntimeError("Episode ended before the smoke STOP action")
            adapter.execute_discrete_action("stop")
            metrics = _serialize_metric_value(env.get_metrics())
            expected = {"distance_to_goal", "success", "spl"}
            if args.task == "vlnce":
                expected |= {"ndtw", "oracle_success", "sdtw"}
            if not expected.issubset(metrics) or not all(np.isfinite(metrics[name]) for name in expected):
                raise RuntimeError(f"Missing or non-finite metrics: {metrics}")
            if not env.episode_over:
                raise RuntimeError("STOP did not terminate the episode")
            results.append({
                "episode_id": str(env.current_episode.episode_id), "scene": env.current_episode.scene_id,
                "rgb_shape": list(rgb.shape), "depth_shape": list(depth.shape),
                "actions": turn_info["num_steps"] + nav_info["num_steps"] + 1,
                "metrics": metrics,
            })
    finally:
        env.close()
    return {"check": "real_scene_smoke", "episodes": results}


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2))
    print("Habitat smoke test passed.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from navprobe.config.settings import build_runner_parser
from pathlib import Path
import traceback
from typing import Any

from navprobe.runners.batch_results import (
    NavProbeBatchResults, NavProbeBatchTimer, _episode_log_path, _run_timestamp,
    experiment_configuration,
)
from navprobe.env.vlnce_dataset import filter_vlnce_requests_by_infos
from navprobe.env.vlnce_dataset import list_vlnce_episode_requests
from navprobe.env.vlnce_dataset import load_vlnce_episode_infos
from navprobe.env.vlnce_dataset import resolve_vlnce_episode_selection
from navprobe.env.vlnce_env import VLNCEEnv
from navprobe.env.habitat_env import close_environment
from navprobe.runners.episode_resources import _release_episode_memory
from navprobe.runners.preflight import is_global_model_error, validate_runtime_resources
from navprobe.agent.runner import run_navprobe_agent_episode
from navprobe.runtime.panorama_config import vlnce_panorama_config


def build_parser() -> argparse.ArgumentParser:
    return build_runner_parser("vlnce")


def _build_requests(
    args: argparse.Namespace, *, episode_payloads: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    requests = list_vlnce_episode_requests(
        data_path=args.data_path, split=args.split,
        episode_payloads=episode_payloads,
    )
    requests = filter_vlnce_requests_by_infos(requests, load_vlnce_episode_infos(args.episode_file))
    if episode_payloads is not None:
        selected_indices = {request.dataset_index for request in requests}
        for index in list(episode_payloads):
            if index not in selected_indices:
                del episode_payloads[index]
    return [request.to_dict() for request in requests]


def _episode_args_from_request(args: argparse.Namespace, request: dict[str, Any]) -> argparse.Namespace:
    episode_id = str(request["episode_id"])
    values = vars(args).copy()
    values.update(goal=str(request["instruction"]), task_type="vln",
                  episode_id=int(episode_id) if episode_id.isdigit() else None,
                  dataset_index=int(request["dataset_index"]), scene=None, scene_id=str(request["scene_id"]),
                  record_dir=str(Path(args.record_dir).expanduser().resolve()))
    return argparse.Namespace(**values)


def _env_from_request(
    args: argparse.Namespace, request: dict[str, Any], *,
    episode_payload: dict[str, Any] | None = None,
) -> VLNCEEnv:
    selection = resolve_vlnce_episode_selection(
        data_path=args.data_path,
        split=args.split,
        episode_id=str(request["episode_id"]),
        scene_id=str(request["scene_id"]),
        episode_payload=episode_payload,
        dataset_index=request.get("dataset_index"),
    )
    env = VLNCEEnv(
        seed=args.seed,
        selection=selection,
        habitat_config=args.habitat_config, dataset_type=args.dataset_type,
        scenes_dir=args.scenes_dir,
    )
    return env


def run_vlnce_agent(args: argparse.Namespace) -> list[dict[str, Any]]:
    # Batch data is kept outside args/configuration and contains
    # only the selected episodes. Native Habitat objects are fresh per run.
    episode_payloads: dict[int, dict[str, Any]] = {}
    requests = _build_requests(args, episode_payloads=None if args.dry_run else episode_payloads)
    total_available = len(requests)
    if bool(args.dry_run):
        for index, request in enumerate(requests, start=1):
            print(
                f"[{index}/{len(requests)}] episode_id={request['episode_id']} "
                f"scene={request['scene_key']} instruction={request['instruction']}"
            )
        return []

    timer = NavProbeBatchTimer()
    results_root = Path(args.results_dir).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    batch_run_stamp = _run_timestamp()
    configuration = experiment_configuration(args, runner="vlnce", episode_payloads=episode_payloads)
    args.experiment_configuration = configuration
    batch_results = NavProbeBatchResults(results_root, configuration=configuration, requests=requests)

    print(f"Loaded {total_available} VLNCE episodes.")
    episode_results: list[dict[str, Any]] = []
    execution_error_count = 0

    def update_summary() -> None:
        timer.write_summary(batch_results)

    update_summary()
    if any(args.no_resume or not batch_results.matching_episode_logs(request) for request in requests):
        validate_runtime_resources(args)
    for index, request in enumerate(requests, start=1):
        episode_payload = episode_payloads.pop(int(request.get("dataset_index", -1)), None)
        existing_result_paths = batch_results.matching_episode_logs(request)
        if existing_result_paths != [] and not args.no_resume:
            print(
                f"[{index}/{len(requests)}] Skip episode_id={request['episode_id']} "
                f"scene={request['scene_key']}"
            )
            continue

        result_path = _episode_log_path(results_root, request, batch_run_stamp)
        print(
            f"[{index}/{len(requests)}] Episode {request['episode_id']} "
            f"scene={request['scene_key']} instruction={request['instruction']}"
        )
        env = None
        try:
            env = _env_from_request(args, request, episode_payload=episode_payload)
            episode_result = run_navprobe_agent_episode(
                _episode_args_from_request(args, request),
                env=env,
                panorama_config=vlnce_panorama_config(),
            )
        except Exception as exc:
            execution_error_count += 1
            error_traceback = traceback.format_exc()
            batch_results.write_error(
                result_path, exc, request=request,
                metadata={"vlnce_batch_request": dict(request), "vlnce_batch_index": int(index)},
            )
            update_summary()
            if is_global_model_error(exc):
                raise
            print(error_traceback, end="")
            print(f"  error={type(exc).__name__}: {exc}; continuing to next episode")
            continue
        finally:
            close_environment(env, label=f"episode={request['episode_id']}")
            env = None
            _release_episode_memory(str(request["episode_id"]))
        batch_results.write_episode_result(
            result_path, episode_result, request=request,
        )
        episode_results.append(episode_result)
        update_summary()
        print(
            f"  success={episode_result['episode_success']} "
            f"steps={episode_result['num_steps']} "
            f"place_steps={episode_result['num_place_steps']} "
            f"run_dir={episode_result['run_dir']}"
        )

    update_summary()
    print(f"Saved episode logs to: {results_root}")
    print(f"Saved summary to: {results_root / 'summary.json'}")
    if execution_error_count:
        raise RuntimeError(
            f"VLNCE batch finished with {execution_error_count} episode execution error(s); "
            f"see {results_root / 'errors'}"
        )
    return episode_results


def main() -> None:
    args = build_parser().parse_args()
    run_vlnce_agent(args)


if __name__ == "__main__":
    main()

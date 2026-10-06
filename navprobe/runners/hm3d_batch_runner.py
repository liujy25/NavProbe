from __future__ import annotations

import argparse
from navprobe.config.settings import build_runner_parser
import json
from pathlib import Path
from typing import Any

from navprobe.runners.batch_results import (
    NavProbeBatchResults, NavProbeBatchTimer, _episode_log_path, _run_timestamp,
    experiment_configuration,
)
from navprobe.env.hm3dv2_dataset import HM3Dv2EpisodeRequest
from navprobe.env.hm3dv2_dataset import list_hm3dv2_episode_requests
from navprobe.env.hm3dv2_dataset import normalize_scene_name
from navprobe.runners.hm3d_runner import run_hm3d_agent
from navprobe.runners.preflight import validate_runtime_resources


def _episode_id_string(value: Any, *, field_name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{field_name} must be a string or integer episode id")
    episode_id = str(value).strip()
    if episode_id == "":
        raise ValueError(f"{field_name} must not be empty")
    return episode_id


def _filter_requests_by_manifest(
    requests: list[dict[str, Any]], payload: Any,
) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("episode_file must contain a JSON list")
    # Episode manifests use the same {id, scene_id} format as R2R/RxR.
    # Validate the scene as well as the global ID to catch changed content order.
    by_id = {str(request["episode_id"]): request for request in requests}
    if len(by_id) != len(requests):
        raise ValueError("HM3Dv2 dataset contains duplicate global episode ids")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"episode_file[{index}] must be an object with id and scene_id")
        episode_id = _episode_id_string(item.get("id"), field_name=f"episode_file[{index}].id")
        scene_id = item.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError(f"episode_file[{index}].scene_id must be a non-empty string")
        if episode_id in seen:
            raise ValueError(f"Duplicate HM3Dv2 episode id in episode_file: {episode_id}")
        seen.add(episode_id)
        if episode_id not in by_id:
            raise ValueError(f"episode_file contains id not present in HM3Dv2 dataset: {episode_id}")
        request = by_id[episode_id]
        if normalize_scene_name(scene_id) != str(request["scene_key"]):
            raise ValueError(
                f"HM3Dv2 episode_file scene mismatch for id={episode_id}: "
                f"manifest={scene_id!r}, dataset={request['scene_key']!r}"
            )
        selected.append(request)
    return selected


def _request_from_dataset(request: HM3Dv2EpisodeRequest) -> dict[str, Any]:
    return {
        "scene_name": request.scene_key,
        "scene_key": request.scene_key,
        "global_episode_id": str(request.global_episode_id),
        "episode_id": str(request.global_episode_id),
        "local_episode_index": int(request.episode_index),
        "episode_index": int(request.episode_index),
        "scene_episode_id": str(request.source_episode_id),
        "source_episode_id": str(request.source_episode_id),
        "goal": str(request.goal),
    }


def build_parser() -> argparse.ArgumentParser:
    return build_runner_parser("hm3d_batch")


def _build_requests(
    args: argparse.Namespace, *, episode_payloads: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    manifest = json.loads(Path(args.episode_file).read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("episode_file must contain a JSON list")
    payload_episode_ids = set()
    for index, item in enumerate(manifest):
        if not isinstance(item, dict):
            raise ValueError(f"episode_file[{index}] must be an object with id and scene_id")
        payload_episode_ids.add(
            _episode_id_string(item.get("id"), field_name=f"episode_file[{index}].id")
        )
    requests = [
        _request_from_dataset(request)
        for request in list_hm3dv2_episode_requests(
            data_path=args.data_path,
            scenes_dir=getattr(args, "scenes_dir", None),
            split=getattr(args, "split", "val"),
            episode_payloads=episode_payloads,
            payload_episode_ids=payload_episode_ids,
        )
    ]
    return _filter_requests_by_manifest(requests, manifest)


def _episode_args_from_request(args: argparse.Namespace, request: dict[str, Any]) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(scene_name=str(request["scene_name"]), task_type="objectnav",
                  episode_index=None if request.get("episode_index") is None else int(request["episode_index"]),
                  episode_id=None,
                  dataset_index=None, scene=None, scene_id=None,
                  record_dir=str(Path(args.record_dir).expanduser().resolve()))
    return argparse.Namespace(**values)


def main() -> None:
    args = build_parser().parse_args()
    timer = NavProbeBatchTimer()

    episode_payloads: dict[int, dict[str, Any]] = {}
    requests = _build_requests(args, episode_payloads=None if args.dry_run else episode_payloads)
    total_available = len(requests)
    total = len(requests)

    if args.dry_run:
        for index, request in enumerate(requests, start=1):
            print(f"[{index}/{total}] episode_id={request['episode_id']} "
                  f"scene={request['scene_key']} local_episode_index={request['episode_index']} goal={request['goal']}")
        return

    results_root = Path(args.results_dir).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    batch_run_stamp = _run_timestamp()
    configuration = experiment_configuration(args, runner="hm3d_batch", episode_payloads=episode_payloads)
    args.experiment_configuration = configuration
    batch_results = NavProbeBatchResults(results_root, configuration=configuration, requests=requests)

    print(f"Loaded {total_available} HM3Dv2 episodes.")

    def update_summary() -> None:
        timer.write_summary(batch_results)

    update_summary()
    if any(args.no_resume or not batch_results.matching_episode_logs(request) for request in requests):
        validate_runtime_resources(args)
    for index, request in enumerate(requests, start=1):
        global_episode_id = int(request["episode_id"])
        episode_payload = episode_payloads.pop(global_episode_id, None)
        existing_result_paths = batch_results.matching_episode_logs(request)
        if existing_result_paths != [] and not args.no_resume:
            print(
                f"[{index}/{total}] Skip episode_id={request.get('episode_id')} "
                f"scene={request['scene_key']} local_episode_index={request.get('episode_index')}"
            )
            continue

        result_path = _episode_log_path(results_root, request, batch_run_stamp)
        print(
            f"[{index}/{total}] Episode {request.get('episode_id')} "
            f"scene={request['scene_key']} local_episode_index={request.get('episode_index')} "
            f"goal={request.get('goal')}"
        )
        try:
            episode_result = run_hm3d_agent(
                _episode_args_from_request(args, request),
                episode_payload=episode_payload, global_episode_id=global_episode_id,
            )
        except Exception as exc:
            batch_results.write_error(result_path, exc, request=request)
            update_summary()
            raise
        batch_results.write_episode_result(
            result_path, episode_result, request=request,
        )
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

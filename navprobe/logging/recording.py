from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image
from PIL import ImageDraw

from navprobe.env.episode_status import current_episode_step
from navprobe.schemas import ActionCall, ActionResult
from navprobe.mapping.routing import _place_nodes
from navprobe.types import (
    LocalmapFrontierRecord,
)
from navprobe.visualization.graph_payload import graph_floor_height
from navprobe.visualization.rendering import (
    CacheSnapshot,
    StepArtifactObservationGroups,
    _render_bev_graph_image,
)
from navprobe.visualization.episode_result_bev import (
    EPISODE_RESULT_BEV_FILENAME,
    GOAL_MAX_HEIGHT_ABOVE_FLOOR_M,
    GOAL_MIN_HEIGHT_BELOW_FLOOR_M,
    render_episode_result_scene_bev,
)
from navprobe.runtime.timing import summarize_step_timings
from navprobe.logging.step_records import iter_step_summaries

if TYPE_CHECKING:
    from navprobe.env.interface import EnvInterface
    from navprobe.mapping.exploration.manager import ExplorationManager
    from navprobe.memory.graph.graph import Graph
    from navprobe.perception.landmark_detection import LandmarkDetectionController
    from navprobe.runtime.cache import RuntimeCache


def _compose_contact_sheet(images: list[np.ndarray], gap_px: int = 8) -> np.ndarray:
    if images == []:
        raise ValueError("contact sheet requires at least one image")
    normalized = [np.asarray(image, dtype=np.uint8) for image in images]
    if len(normalized) == 1:
        return normalized[0]
    max_height = max(int(image.shape[0]) for image in normalized)
    max_width = max(int(image.shape[1]) for image in normalized)
    columns = min(3, len(normalized))
    rows = int(math.ceil(len(normalized) / float(columns)))
    canvas_height = rows * max_height + max(0, rows - 1) * int(gap_px)
    canvas_width = columns * max_width + max(0, columns - 1) * int(gap_px)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    for index, image in enumerate(normalized):
        row = index // columns
        col = index % columns
        y0 = row * (max_height + int(gap_px))
        x0 = col * (max_width + int(gap_px))
        image_height = int(image.shape[0])
        image_width = int(image.shape[1])
        canvas[y0:y0 + image_height, x0:x0 + image_width] = image
    return canvas


def _write_json(path: Path, payload: Any) -> None:
    # Publish completed records atomically, including replacement after interruption.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _cache_snapshot(cache: RuntimeCache) -> CacheSnapshot:
    return CacheSnapshot(
        observation_ids=set(cache.observations.keys()),
        detection_ids=set(cache.detections.keys()),
    )


def _step_artifact_observation_groups(
    panorama_obs_ids: list[str],
    results: list[tuple[ActionCall, ActionResult]],
) -> StepArtifactObservationGroups:
    step_obs_ids: set[str] = set()
    for _, result in results:
        data = result.data
        step_observations = data.get("step_observations")
        if isinstance(step_observations, list):
            for payload in step_observations:
                if not isinstance(payload, dict):
                    continue
                obs_id = payload.get("obs_id")
                if obs_id is not None:
                    step_obs_ids.add(str(obs_id))
        rgb_history_obs_ids = data.get("rgb_history_obs_ids")
        if isinstance(rgb_history_obs_ids, list):
            for obs_id in rgb_history_obs_ids:
                step_obs_ids.add(str(obs_id))
        step_observation = data.get("step_observation")
        if isinstance(step_observation, dict):
            obs_id = step_observation.get("obs_id")
            if obs_id is not None:
                step_obs_ids.add(str(obs_id))
    return StepArtifactObservationGroups(
        panorama_obs_ids={str(obs_id) for obs_id in panorama_obs_ids},
        step_obs_ids=step_obs_ids,
    )


def _write_cache_delta_artifacts(
    *,
    cache: RuntimeCache,
    cache_before: CacheSnapshot,
    cache_after: CacheSnapshot,
    observation_groups: StepArtifactObservationGroups,
) -> dict[str, Any]:
    new_observation_ids = sorted(cache_after.observation_ids - cache_before.observation_ids)
    new_detection_ids = sorted(cache_after.detection_ids - cache_before.detection_ids)

    observation_index: dict[str, Any] = {}
    for obs_id in new_observation_ids:
        observation = cache.get_observation(obs_id).observation
        if obs_id in observation_groups.panorama_obs_ids:
            obs_group = "panorama_obs"
        elif obs_id in observation_groups.step_obs_ids:
            obs_group = "step_obs"
        else:
            obs_group = "observations"
        observation_index[obs_id] = {
            "artifact_group": obs_group,
            "pose": observation.pose.to_dict(),
            "text_hint": observation.text_hint,
            "visible_objects": list(observation.visible_objects),
            "semantic_annotations": [dict(item) for item in observation.semantic_annotations],
        }

    detection_index: dict[str, Any] = {}
    for det_id in new_detection_ids:
        record = cache.get_detection(det_id)
        detection_index[det_id] = {
            "obs_id": record.obs_id,
            "class_name": record.detection.class_name,
            "metadata": dict(record.detection.metadata),
            "bbox": list(record.detection.bbox),
            "score": float(record.detection.score),
            "location": None if record.detection.location is None else list(record.detection.location),
        }

    return {
        "observations": observation_index,
        "detections": detection_index,
    }


def write_localmap_replay_data(
    *,
    step_dir: Path,
    cache: RuntimeCache,
    obs_ids: list[str],
    angle_to_obs_id: dict[int, str],
    bev_map_kwargs: dict[str, object],
    metadata: dict[str, object],
) -> dict[str, Any]:
    output_dir = step_dir / "localmap_replay"
    observations_dir = output_dir / "observations"
    observations_dir.mkdir(parents=True, exist_ok=True)

    angle_by_obs_id = {
        str(obs_id): int(angle)
        for angle, obs_id in dict(angle_to_obs_id).items()
    }
    observation_records: list[dict[str, object]] = []
    for obs_id in [str(item) for item in obs_ids]:
        observation = cache.get_observation(obs_id).observation
        record: dict[str, object] = {
            "obs_id": obs_id,
            "angle_deg": angle_by_obs_id.get(obs_id),
            "pose": observation.pose.to_dict(),
            "intrinsics": _json_safe_array(observation.intrinsics),
            "T_cam_odom": _json_safe_array(observation.T_cam_odom),
            "T_odom_base": _json_safe_array(observation.T_odom_base),
            "text_hint": str(observation.text_hint),
            "visible_objects": [str(item) for item in observation.visible_objects],
        }
        if observation.rgb is not None:
            rgb_path = save_observation_rgb(step_dir=step_dir, cache=cache, obs_id=obs_id)
            record["rgb_path"] = "../" + rgb_path
            record["rgb_shape"] = list(_rgb_image_array(observation.rgb).shape)
        if observation.depth is not None:
            depth = np.asarray(observation.depth, dtype=np.float32)
            depth_path = observations_dir / f"{obs_id}_depth.npy"
            np.save(depth_path, depth)
            record["depth_path"] = str(depth_path.relative_to(output_dir))
            record["depth_shape"] = [int(dim) for dim in depth.shape]
            record["depth_dtype"] = "float32"
        observation_records.append(record)

    payload = {
        "format": "navprobe_localmap_replay_v1",
        "metadata": _json_safe_array(metadata),
        "bev_map_kwargs": _json_safe_array(bev_map_kwargs),
        "current_projection_obs_ids": [str(item) for item in obs_ids],
        "angle_to_obs_id": {
            str(int(angle)): str(obs_id)
            for angle, obs_id in sorted(dict(angle_to_obs_id).items(), key=lambda item: int(item[0]))
        },
        "observations": observation_records,
    }
    metadata_path = output_dir / "metadata.json"
    _write_json(metadata_path, payload)
    return {
        "dir": str(output_dir.relative_to(step_dir)),
        "metadata_path": str(metadata_path.relative_to(step_dir)),
        "observation_count": len(observation_records),
    }


def _rgb_image_array(rgb: object) -> np.ndarray:
    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"rgb observation must have shape HxWx3 or HxWx4, got {image.shape!r}")
    return np.ascontiguousarray(image)


def _json_safe_array(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe_array(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_array(item) for item in value]
    return value


def _runtime_snapshot(
    *,
    env: EnvInterface,
    graph: Graph,
    current_floor_id: str,
    frontier_records: dict[str, LocalmapFrontierRecord],
    current_place_node_id: str | None,
    current_obs_id: str | None,
    previous_place_node_id: str | None,
    landmark_controller: LandmarkDetectionController,
) -> dict[str, Any]:
    landmark_context = landmark_controller.landmark_memory.to_context()
    return {
        "episode_step_count": int(current_episode_step(env)),
        "current_floor_id": str(current_floor_id),
        "current_node_id": None if current_place_node_id is None else str(current_place_node_id),
        "current_obs_id": None if current_obs_id is None else str(current_obs_id),
        "previous_place_node_id": None if previous_place_node_id is None else str(previous_place_node_id),
        "num_place_nodes": int(len(_place_nodes(graph))),
        "num_frontiers": int(len(frontier_records)),
        "landmark_detection_active": bool(landmark_controller.is_active()),
        "confirmed_landmark_count": len(list(landmark_context.get("confirmed_landmarks", []))),
    }


def _goal_markers_from_reset_payload(
    reset_payload: dict[str, Any],
    floor_height: float | None = None,
    max_delta_z_above_agent: float = GOAL_MAX_HEIGHT_ABOVE_FLOOR_M,
) -> list[dict[str, Any]]:
    goals = reset_payload.get("goals", [])
    if not isinstance(goals, list):
        raise ValueError("reset_payload.goals must be a list")
    if goals == []:
        return []
    if floor_height is None and "ground_height_offset" not in reset_payload:
        raise ValueError("reset_payload missing ground_height_offset")
    agent_initial_height = (
        float(reset_payload["ground_height_offset"])
        if floor_height is None
        else float(floor_height)
    )
    min_goal_height = agent_initial_height - GOAL_MIN_HEIGHT_BELOW_FLOOR_M
    max_goal_height = agent_initial_height + float(max_delta_z_above_agent)
    markers: list[dict[str, Any]] = []
    for goal_index, goal in enumerate(goals):
        if not isinstance(goal, dict):
            raise ValueError(f"reset_payload.goals[{goal_index}] must be a dict")
        position = goal.get("position")
        if not isinstance(position, list) or len(position) < 3:
            raise ValueError(f"reset_payload.goals[{goal_index}].position must be a list of length >= 3")
        habitat_x = float(position[0])
        habitat_y = float(position[1])
        habitat_z = float(position[2])
        if not (min_goal_height <= habitat_y <= max_goal_height):
            continue
        markers.append(
            {
                "x": -habitat_z,
                "y": -habitat_x,
                "label": f"G{goal_index}",
                "color": (255, 0, 255),
                "radius": 9,
            }
        )
    return markers


def _goal_marker_floor_height_base(reset_payload: dict[str, Any]) -> float | None:
    ground_height = reset_payload.get("ground_height_offset")
    if ground_height is not None:
        return float(ground_height)
    start_position = reset_payload.get("start_position")
    if isinstance(start_position, (list, tuple)) and len(start_position) >= 2:
        return float(start_position[1])
    return None


def _goal_markers_by_floor_from_reset_payload(
    *,
    reset_payload: dict[str, Any],
    graph_payload: dict[str, Any],
    floor_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    base_height = _goal_marker_floor_height_base(reset_payload)
    if base_height is None:
        return {}
    markers_by_floor: dict[str, list[dict[str, Any]]] = {}
    for floor_id in floor_ids:
        internal_height = graph_floor_height(graph_payload, floor_id)
        if internal_height is None:
            continue
        floor_markers = _goal_markers_from_reset_payload(
            reset_payload,
            floor_height=base_height + float(internal_height),
        )
        if floor_markers != []:
            markers_by_floor[str(floor_id)] = floor_markers
    return markers_by_floor


def _final_agent_marker_from_stop_result(stop_result: dict[str, Any]) -> dict[str, Any] | None:
    pose = stop_result.get("pose")
    if not isinstance(pose, dict):
        return None
    return {
        "x": float(pose["x"]),
        "y": float(pose["y"]),
        "label": "agent_final",
        "color": (255, 80, 80),
        "radius": 10,
    }


def _habitat_pathfinder_from_env(env: EnvInterface) -> Any | None:
    habitat_env = getattr(env, "env", None)
    sim = getattr(habitat_env, "sim", None)
    return getattr(sim, "pathfinder", None)


def _label_floor_panel(image: np.ndarray, floor_id: str) -> np.ndarray:
    rendered = Image.fromarray(np.asarray(image, dtype=np.uint8).copy())
    draw = ImageDraw.Draw(rendered, "RGBA")
    text = f"floor: {floor_id}"
    width = max(120, 8 * len(text) + 24)
    draw.rectangle((8, 8, width, 36), fill=(255, 255, 255, 220))
    draw.text((16, 14), text, fill=(0, 0, 0))
    return np.asarray(rendered, dtype=np.uint8)


def _render_all_floor_bev_graph_image(
    *,
    graph_payload: dict[str, Any],
    explorations_by_floor: dict[str, ExplorationManager],
    point_markers_by_floor: dict[str, list[dict[str, Any]]] | None = None,
) -> np.ndarray:
    panels: list[np.ndarray] = []
    markers_by_floor = {} if point_markers_by_floor is None else dict(point_markers_by_floor)
    for floor_id in sorted(str(floor_id) for floor_id in explorations_by_floor):
        exploration = explorations_by_floor[floor_id]
        panel = _render_bev_graph_image(
            graph_payload=graph_payload,
            bev_background=exploration.render_bev_graph_background_raw_obstacles(),
            xy_to_px=exploration.map.xy_to_px,
            explored_mask=exploration.map.explored_area,
            point_markers=markers_by_floor.get(floor_id, []),
            floor_id=floor_id,
        )
        panels.append(_label_floor_panel(panel, floor_id))
    if panels == []:
        raise ValueError("final BEV rendering requires at least one floor exploration")
    return _compose_contact_sheet(panels)


def _write_episode_result_bev(
    env: EnvInterface,
    run_dir: Path,
    explorations_by_floor: dict[str, ExplorationManager],
    graph_payload: dict[str, Any],
    current_floor_id: str,
    reset_payload: dict[str, Any],
    stop_result: dict[str, Any],
) -> Path:
    output_path = run_dir / EPISODE_RESULT_BEV_FILENAME
    scene_bev = render_episode_result_scene_bev(
        run_dir=run_dir,
        reset_payload=reset_payload,
        graph_payload=graph_payload,
        stop_result=stop_result,
        current_floor_id=current_floor_id,
        pathfinder=_habitat_pathfinder_from_env(env),
    )
    if scene_bev is not None:
        Image.fromarray(scene_bev).save(output_path)
        return output_path

    point_markers_by_floor = _goal_markers_by_floor_from_reset_payload(
        reset_payload=reset_payload,
        graph_payload=graph_payload,
        floor_ids=sorted(str(floor_id) for floor_id in explorations_by_floor),
    )
    if point_markers_by_floor == {}:
        point_markers_by_floor = {str(current_floor_id): _goal_markers_from_reset_payload(reset_payload)}
    final_agent_marker = _final_agent_marker_from_stop_result(stop_result)
    if final_agent_marker is not None:
        point_markers_by_floor.setdefault(str(current_floor_id), []).append(final_agent_marker)
    bev_graph = _render_all_floor_bev_graph_image(
        graph_payload=graph_payload,
        explorations_by_floor=explorations_by_floor,
        point_markers_by_floor=point_markers_by_floor,
    )
    Image.fromarray(bev_graph).save(output_path)
    return output_path


def write_interrupted_retrieval_state(
    *,
    step_dir: Path,
    step_index: int,
    error: Exception,
    task_state_memory: dict[str, object],
    episodic_retrieval: dict[str, object],
) -> dict[str, object]:
    """Persist the committed agenda and failed retrieval request for audit.

    This diagnostic record is not applied to a later run. Failed episodes
    restart from their dataset start state.
    """
    payload = {
        "status": "error",
        "error_type": type(error).__name__,
        "error": str(error),
        "step_index": int(step_index),
        "step_error_path": str(step_dir / "step_error.json"),
        "task_state_memory": deepcopy(task_state_memory),
        "episodic_retrieval": deepcopy(episodic_retrieval),
        "llm_artifacts": str(step_dir / "llm" / "index.json"),
    }
    # Keep the exact snapshot available to batch error consumers even
    # if the filesystem write itself fails.
    error.navprobe_step_error = payload
    step_dir.mkdir(parents=True, exist_ok=True)
    temporary = step_dir / "step_error.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(step_dir / "step_error.json")
    error.navprobe_step_error = {"step_error_path": str(step_dir / "step_error.json")}
    return payload


def write_final_run_outputs(
    *,
    env: EnvInterface,
    run_dir: Path,
    run_metadata: dict[str, object],
    reset_payload: dict[str, Any],
    graph: Graph,
    explorations_by_floor: dict[str, ExplorationManager],
    frontier_records: dict[str, LocalmapFrontierRecord],
    current_floor_id: str,
    current_place_node_id: str | None,
    current_obs_id: str | None,
    previous_place_node_id: str | None,
    landmark_controller: LandmarkDetectionController,
    recorded_step_count: int,
    finalize_reason: str,
    stop_result: dict[str, Any],
    llm_usage_payload: dict[str, Any],
    recorder_episode_id: int,
    recorder_scene_id: str,
    recorder_scene_name: str,
    run_goal: str,
    visualize: bool = True,
) -> dict[str, Any]:
    final_runtime = _runtime_snapshot(
        env=env,
        graph=graph,
        current_floor_id=current_floor_id,
        frontier_records=frontier_records,
        current_place_node_id=current_place_node_id,
        current_obs_id=current_obs_id,
        previous_place_node_id=previous_place_node_id,
        landmark_controller=landmark_controller,
    )
    run_json = deepcopy(run_metadata)
    run_json["end_time"] = datetime.now().isoformat()
    _write_json(run_dir / "run.json", run_json)
    graph_payload = graph.to_dict()
    _write_json(run_dir / "graph_final.json", graph_payload)
    final_bev_path = None
    if visualize:
        final_bev_path = _write_episode_result_bev(
            env=env, run_dir=run_dir, explorations_by_floor=explorations_by_floor,
            graph_payload=graph_payload, current_floor_id=current_floor_id,
            reset_payload=reset_payload, stop_result=stop_result,
        ).relative_to(run_dir).as_posix()
    timing_summary = summarize_step_timings(
        iter_step_summaries(run_dir / "steps", recorded_step_count)
    )
    result = {
        "dataset_index": reset_payload.get("dataset_index"),
        "episode_id": int(recorder_episode_id),
        "scene_id": recorder_scene_id,
        "scene_name": recorder_scene_name,
        "goal": run_goal,
        "num_steps": int(stop_result.get("pointnav_step_total", 0)),
        "num_place_steps": int(recorded_step_count),
        "episode_success": bool(stop_result.get("episode_success", False)),
        "agent_declared_done": bool(stop_result.get("agent_declared_done", False)),
        "episode_over": bool(stop_result.get("episode_over", False)),
        "metrics": stop_result.get("metrics", {}),
        "run_dir": str(run_dir),
        "finalize_reason": str(finalize_reason),
        "final_pose": stop_result.get("pose"),
        "final_state": final_runtime,
        "landmarks": landmark_controller.landmark_memory.to_context(),
        "llm_usage": llm_usage_payload,
        "timing_summary": timing_summary,
        "artifacts": {
            "run": "run.json", "graph": "graph_final.json", "steps": "steps",
            **({"bev": final_bev_path} if final_bev_path is not None else {}),
        },
        "configuration_id": run_metadata["configuration_id"],
    }
    _write_json(run_dir / "summary.json", result)
    return result


def save_observation_rgb(*, step_dir: Path, cache: RuntimeCache, obs_id: str) -> str:
    path = step_dir / "observations" / f"{obs_id}.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb = _rgb_image_array(cache.get_observation(obs_id).observation.rgb)
        Image.fromarray(rgb).save(path)
    return path.relative_to(step_dir).as_posix()

"""Episode metadata and artifacts, called at explicit navigation boundaries."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from navprobe.env.episode_status import current_episode_step
from navprobe.llm.request_artifacts import write_llm_request_artifacts
from navprobe.logging.agent_logging import build_agent_step_summary
from navprobe.logging.recording import (
    _cache_snapshot,
    _step_artifact_observation_groups,
    _write_cache_delta_artifacts,
    _write_json,
    write_final_run_outputs,
    write_interrupted_retrieval_state,
    write_localmap_replay_data,
)
from navprobe.logging.visual_waypoint_artifacts import write_visual_waypoint_artifacts
from navprobe.mapping.frontier_registry import _frontier_id_sort_key
from navprobe.visualization.step_visualization import (
    StepVisualizationImageResult,
    render_step_visualization_images,
    should_skip_step_visualization_for_reuse,
)

if TYPE_CHECKING:
    from navprobe.agent.state import NavProbeAgentContext, NavProbeAgentState, NavProbeStepState
    from navprobe.mapping.exploration.bev_map import GlobalBEVMap
    from navprobe.mapping.exploration.manager import ExplorationManager
    from navprobe.runtime.panorama_config import PanoramaConfig


def _global_bev_metadata(bev_map: GlobalBEVMap) -> dict[str, object]:
    return {
        "size": int(bev_map.size),
        "pixels_per_meter": int(bev_map.pixels_per_meter),
        "min_height": float(bev_map.min_height),
        "max_height": float(bev_map.max_height),
        "agent_radius": float(bev_map.agent_radius),
        "min_depth": float(bev_map.min_depth),
        "max_depth": float(bev_map.max_depth),
        "astar_clearance_radius": float(bev_map.astar_clearance_radius),
        "astar_clearance_weight": float(bev_map.astar_clearance_weight),
        "depth_is_normalized": bool(bev_map.depth_is_normalized),
        "virtual_max_range_blocker": bool(bev_map.virtual_max_range_blocker),
        "virtual_max_range_pixel_stride": int(bev_map.virtual_max_range_pixel_stride),
        "virtual_max_range_blocker_dilate_px": int(bev_map.virtual_max_range_blocker_dilate_px),
    }


def _panorama_metadata(panorama_config: PanoramaConfig) -> dict[str, object]:
    return {
        "observation_count": int(panorama_config.observation_count),
        "turns_per_observation": int(panorama_config.turns_per_observation),
        "turn_direction": str(panorama_config.turn_direction),
        "turn_angle_degrees": float(panorama_config.turn_angle_degrees),
        "observation_spacing_degrees": float(panorama_config.observation_spacing_degrees),
    }


def build_run_metadata(
    *, run_name: str, run_goal: str, reset_payload: dict[str, Any],
    recorder_episode_id: int, recorder_scene_id: str, recorder_scene_name: str,
    global_exploration: ExplorationManager, panorama_config: PanoramaConfig,
) -> dict[str, Any]:
    """Record identity and resolved geometry, without copying dataset goal arrays."""
    return {
        "run_name": run_name, "goal": run_goal,
        "start_time": datetime.now().isoformat(),
        "metadata": {
            "episode_id": recorder_episode_id, "scene_id": recorder_scene_id,
            "scene_name": recorder_scene_name,
            "dataset_index": reset_payload.get("dataset_index"),
            "global_bev": _global_bev_metadata(global_exploration.map),
            "panorama": _panorama_metadata(panorama_config),
        },
        "episode": {key: reset_payload[key] for key in (
            "episode_id", "scene_id", "split", "start_position", "start_rotation",
            "ground_height_offset",
        ) if key in reset_payload},
    }


def render_step_outputs(
    *,
    context: NavProbeAgentContext,
    state: NavProbeAgentState,
    step: NavProbeStepState,
) -> None:
    if step.anchor_obs_id is None or step.current_place_node_id is None:
        raise ValueError("render_step_outputs requires observe_place")
    floor_id = str(state.system.current_floor_id)
    global_exploration = state.global_exploration_for_floor(floor_id)
    frontier_records = state.frontier_records_for_floor(floor_id)
    skipped_for_reuse = should_skip_step_visualization_for_reuse(
        step.place_reuse_state,
    )
    step.timing.start("render_visualizations", current_episode_step(context.env))
    step_visualizations = render_step_visualization_images(
        step_dir=step.step_dir,
        cache=state.cache,
        graph=state.graph,
        global_exploration=global_exploration,
        place_reuse_state=step.place_reuse_state,
        anchor_obs_id=str(step.anchor_obs_id),
        current_place_node_id=str(step.current_place_node_id),
        floor_id=floor_id,
        frontier_records=frontier_records,
    ) if getattr(context.args, "visualize", False) else StepVisualizationImageResult()
    step.timing.stop(
        "render_visualizations",
        current_episode_step(context.env),
        details={
            **step_visualizations.timing_details(),
            "skipped_for_reuse": skipped_for_reuse,
        },
    )
    step.step_visualizations = step_visualizations


def _current_obs_id_after_results(
    *,
    anchor_obs_id: str,
    results,
    reuse_current_place_next_step,
) -> str:
    current_obs_id_after = str(anchor_obs_id)
    for _, result in results:
        obs_id_after = result.data.get("obs_id")
        if obs_id_after is not None:
            current_obs_id_after = str(obs_id_after)
    if reuse_current_place_next_step is not None:
        recovery = getattr(reuse_current_place_next_step, "recovery", {})
        if isinstance(recovery, dict) and recovery.get("current_obs_id_after") is not None:
            return str(recovery["current_obs_id_after"])
        current_obs_id_after = str(anchor_obs_id)
    return current_obs_id_after


def _created_frontier_ids_for_current_place(
    *,
    frontier_records,
    current_place_node_id: str,
    created_registered_frontier_ids: set[str],
) -> list[str]:
    return [
        record.frontier_id
        for frontier_id in sorted(created_registered_frontier_ids, key=_frontier_id_sort_key)
        if (record := frontier_records.get(frontier_id)) is not None
        and str(record.source_node_id) == str(current_place_node_id)
    ]


def record_step(
    *,
    context: NavProbeAgentContext,
    state: NavProbeAgentState,
    step: NavProbeStepState,
) -> None:
    if step.anchor_obs_id is None or step.current_place_node_id is None:
        raise ValueError("record_step requires observe_place")
    if step.frontier_update is None:
        raise ValueError("record_step requires update_frontiers")
    if step.step_visualizations is None:
        raise ValueError("record_step requires render_step_outputs")

    step.created_frontier_ids = _created_frontier_ids_for_current_place(
        frontier_records=state.frontier_records_for_floor(str(state.system.current_floor_id)),
        current_place_node_id=str(step.current_place_node_id),
        created_registered_frontier_ids=set(step.frontier_update.created_registered_frontier_ids),
    )
    step.current_obs_id_after = _current_obs_id_after_results(
        anchor_obs_id=str(step.anchor_obs_id),
        results=step.results,
        reuse_current_place_next_step=state.reuse_current_place_next_step,
    )
    step.timing.start("write_artifacts", current_episode_step(context.env))
    cache_after = _cache_snapshot(state.cache)
    visualize = getattr(context.args, "visualize", False)

    step.artifact_summary = _write_cache_delta_artifacts(
        cache=state.cache,
        cache_before=step.cache_before,
        cache_after=cache_after,
        observation_groups=_step_artifact_observation_groups(
            panorama_obs_ids=[str(obs_id) for obs_id in step.panorama_obs_ids],
            results=step.results,
        ),
    )
    step.artifact_summary["visualize"] = visualize
    if visualize:
        step.artifact_summary["localmap_replay"] = write_localmap_replay_data(
            step_dir=step.step_dir,
            cache=state.cache,
            obs_ids=[str(obs_id) for obs_id in step.current_projection_obs_ids],
            angle_to_obs_id=dict(step.angle_to_obs_id),
            bev_map_kwargs=dict(context.global_bev_kwargs),
            metadata={
                "current_floor_id": str(state.system.current_floor_id),
                "current_place_node_id": str(step.current_place_node_id),
                "anchor_obs_id": str(step.anchor_obs_id),
                "current_obs_id": str(step.current_obs_id),
                "panorama_obs_ids": [str(obs_id) for obs_id in step.panorama_obs_ids],
                "reused_place_obs_ids": [str(obs_id) for obs_id in step.reused_place_obs_ids],
                "candidate_dedup_radius_m": float(context.args.candidate_dedup_radius),
            },
        )

    llm_index = write_llm_request_artifacts(
        step_dir=step.step_dir,
        logs=state.llm_client.get_llm_logs_since(step.llm_log_start_index),
        save_images=visualize,
    )
    step.visual_waypoint_artifact_summary = write_visual_waypoint_artifacts(
        step_dir=step.step_dir,
        cache=state.cache,
        visual_decision=step.visual_decision,
    ) if visualize else {}
    step.timing.stop(
        "write_artifacts",
        current_episode_step(context.env),
        details={
            "new_observation_count": len(step.artifact_summary["observations"]),
            "new_detection_count": len(step.artifact_summary["detections"]),
            "llm_request_count": state.llm_client.get_llm_log_count() - int(step.llm_log_start_index),
            "llm_prompt_image_count": int(llm_index["image_count"]),
        },
    )
    step.step_timing_payload = step.timing.to_dict(current_episode_step(context.env))
    if step.navigation_segment_timings != []:
        step.step_timing_payload["navigation_segments"] = step.navigation_segment_timings

    env_step_total = current_episode_step(context.env)
    summary = build_agent_step_summary(
        state=state,
        step=step,
        env_step_total=env_step_total,
    )
    _write_json(step.step_dir / "step.json", summary)
    state.recorded_step_count = step.place_step_index + 1


def record_episode_result(
    *,
    context: NavProbeAgentContext,
    state: NavProbeAgentState,
) -> dict[str, Any]:
    """Export a result after the agent has settled its termination state."""
    return write_final_run_outputs(
        env=context.env,
        run_dir=context.run_dir,
        run_metadata=context.run_metadata,
        reset_payload=context.reset_payload,
        graph=state.graph,
        explorations_by_floor=state.global_explorations_by_floor,
        frontier_records=state.frontier_records_for_floor(str(state.system.current_floor_id)),
        current_floor_id=str(state.system.current_floor_id),
        current_place_node_id=state.current_place_node_id,
        current_obs_id=state.current_obs_id,
        previous_place_node_id=state.previous_place_node_id,
        landmark_controller=state.landmark_controller,
        recorded_step_count=state.recorded_step_count,
        finalize_reason=str(state.finalize_reason),
        stop_result=state.stop_result,
        llm_usage_payload=state.llm_client.get_usage_summary(),
        recorder_episode_id=int(context.recorder_episode_id),
        recorder_scene_id=str(context.recorder_scene_id),
        recorder_scene_name=str(context.recorder_scene_name),
        run_goal=str(context.run_goal),
        visualize=getattr(context.args, "visualize", False),
    )


def record_interrupted_step(
    *,
    state: NavProbeAgentState,
    step: NavProbeStepState,
    error: BaseException,
    visualize: bool = True,
) -> None:
    """Try both artifact exports without replacing the navigation exception."""
    if step.episodic_retrieval:
        try:
            write_interrupted_retrieval_state(
                step_dir=step.step_dir,
                step_index=step.place_step_index,
                error=error,
                task_state_memory=state.system.task_state.to_dict(),
                episodic_retrieval=step.episodic_retrieval,
            )
        except Exception as logging_error:
            print(
                f"[logging] failed to save interrupted retrieval state: {logging_error}"
            )
    try:
        write_llm_request_artifacts(
            step_dir=step.step_dir,
            logs=state.llm_client.get_llm_logs_since(step.llm_log_start_index),
            save_images=visualize,
        )
    except Exception as logging_error:
        print(f"[logging] failed to save interrupted-step LLM artifacts: {logging_error}")

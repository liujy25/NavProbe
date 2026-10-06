from __future__ import annotations

from typing import Any

from navprobe.agent.state import NavProbeAgentContext, NavProbeAgentState, NavProbeStepState
from navprobe.env.episode_status import current_episode_step, episode_end_reason, finalize_evaluation_result
from navprobe.logging.episode_recording import record_episode_result
from navprobe.logging.recording import _cache_snapshot
from navprobe.mapping.frontier_update import update_localmap_place_frontiers
from navprobe.mapping.exploration.bev_map import LocalFrontierBEVMap
from navprobe.mapping.place_step import prepare_localmap_place_step
from navprobe.runtime.timing import StepTimingRecorder


def release_step_resources(state: NavProbeAgentState) -> None:
    """Release finished-step artifacts before the next observation."""
    state.release_unused_overlay_images()
    for node in state.graph.iter_nodes():
        exploration = node.localmap
        if exploration is not None and isinstance(exploration.map, LocalFrontierBEVMap):
            exploration.map.compact()


def begin_agent_step(
    *,
    context: NavProbeAgentContext,
    state: NavProbeAgentState,
    place_step_index: int,
) -> NavProbeStepState:
    step_dir = context.steps_dir / f"{place_step_index:04d}"
    step_dir.mkdir(parents=True, exist_ok=False)
    log_start_index = state.llm_client.get_llm_log_count()
    # Reaching a new step means previous records completed.
    # Their request artifacts are on disk; keep only this step's live payloads.
    state.llm_client.discard_llm_logs_before(log_start_index)
    place_reuse_state = state.reuse_current_place_next_step
    state.reuse_current_place_next_step = None
    state.landmark_controller.step_index = int(place_step_index)
    return NavProbeStepState(
        place_step_index=int(place_step_index),
        step_dir=step_dir,
        cache_before=_cache_snapshot(state.cache),
        llm_log_start_index=log_start_index,
        timing=StepTimingRecorder(env_step_start=current_episode_step(context.env)),
        place_reuse_state=place_reuse_state,
    )


def observe_place(
    *,
    context: NavProbeAgentContext,
    state: NavProbeAgentState,
    step: NavProbeStepState,
) -> None:
    floor_id = str(state.system.current_floor_id)
    state.ensure_floor_runtime_state(floor_id, context.global_bev_kwargs)
    global_exploration = state.global_exploration_for_floor(floor_id)
    frontier_filter_exploration = state.frontier_filter_exploration_for_floor(floor_id)
    state.action_executor.exploration = global_exploration
    place_step = prepare_localmap_place_step(
        env=context.env,
        cache=state.cache,
        graph=state.graph,
        global_exploration=global_exploration,
        landmark_controller=state.landmark_controller,
        bev_map_kwargs=context.global_bev_kwargs,
        panorama_config=context.panorama_config,
        current_place_node_id=state.current_place_node_id,
        current_floor_id=floor_id,
        previous_place_node_id=state.previous_place_node_id,
        place_reuse_state=step.place_reuse_state,
        timing=step.timing,
        current_step_count=lambda: current_episode_step(context.env),
    )
    state.current_place_node_id = str(place_step.current_place_node_id)
    step.current_place_node_id = str(place_step.current_place_node_id)
    step.anchor_obs_id = str(place_step.anchor_obs_id)
    state.current_obs_id = str(place_step.current_obs_id)
    step.current_obs_id = str(place_step.current_obs_id)
    step.panorama_obs_ids = list(place_step.panorama_obs_ids)
    step.angle_to_obs_id = dict(place_step.angle_to_obs_id)
    step.panorama_views = [view.to_dict() for view in list(place_step.panorama_views)]
    step.panorama_angle_debug = dict(place_step.panorama_angle_debug)
    step.reused_place_obs_ids = list(place_step.reused_place_obs_ids)
    step.current_projection_obs_ids = list(place_step.current_projection_obs_ids)
    step.local_exploration = place_step.local_exploration
    step.reachable_candidates = list(place_step.reachable_candidates)
    step.timing.start("update_frontier_filter_globalmap", current_episode_step(context.env))
    if step.local_exploration is None:
        raise ValueError("cannot update frontier filter globalmap without local_exploration")
    frontier_filter_merge_details = frontier_filter_exploration.merge_raw_layers_from_local_exploration(
        step.local_exploration,
        obs_ids=step.current_projection_obs_ids,
    )
    step.timing.stop(
        "update_frontier_filter_globalmap",
        current_episode_step(context.env),
        details={
            "current_place_node_id": str(step.current_place_node_id),
            "registered_obs_count": len(step.current_projection_obs_ids),
            "merge_details": frontier_filter_merge_details,
        },
    )


def update_frontiers(
    *,
    context: NavProbeAgentContext,
    state: NavProbeAgentState,
    step: NavProbeStepState,
) -> None:
    if step.current_place_node_id is None:
        raise ValueError("update_frontiers requires observe_place to set current_place_node_id")
    floor_id = str(state.system.current_floor_id)
    state.ensure_floor_runtime_state(floor_id, context.global_bev_kwargs)
    frontier_filter_exploration = state.frontier_filter_exploration_for_floor(floor_id)
    frontier_records = state.frontier_records_for_floor(floor_id)
    overlay_records = state.overlay_records_for_floor(floor_id)
    step.timing.start("update_place_graph_frontiers", current_episode_step(context.env))
    frontier_update = update_localmap_place_frontiers(
        graph=state.graph,
        cache=state.cache,
        global_exploration=frontier_filter_exploration,
        local_exploration=step.local_exploration,
        place_reuse_state=step.place_reuse_state,
        frontier_records=frontier_records,
        overlay_records=overlay_records,
        current_place_node_id=str(step.current_place_node_id),
        current_projection_obs_ids=step.current_projection_obs_ids,
        current_projection_angle_to_obs_id=step.angle_to_obs_id,
        reachable_candidates=step.reachable_candidates,
        candidate_dedup_radius_m=float(context.args.candidate_dedup_radius),
    )
    step.timing.stop(
        "update_place_graph_frontiers",
        current_episode_step(context.env),
        details=frontier_update.timing_details(
            active_frontier_count=len(frontier_records),
        ),
    )
    step.frontier_update = frontier_update


def finalize_run(
    *,
    context: NavProbeAgentContext,
    state: NavProbeAgentState,
) -> dict[str, Any]:
    end_reason = episode_end_reason(context.env)
    if (
        str(state.finalize_reason) != "done"
        and not str(state.finalize_reason).startswith("exception:")
        and end_reason is not None
    ):
        state.finalize_reason = end_reason
    if not state.system.is_done:
        state.system.mark_done(str(state.finalize_reason))
    state.stop_result = finalize_evaluation_result(
        env=context.env,
        existing_result=state.stop_result,
        # Stop may have completed before a subsequent artifact/hook failure.
        # Keep that fact separate from why the Python run ended.
        agent_declared_done=(
            str(state.finalize_reason) == "done"
            or bool((state.stop_result or {}).get("agent_declared_done", False))
        ),
    )

    return record_episode_result(context=context, state=state)

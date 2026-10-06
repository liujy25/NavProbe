from __future__ import annotations

from navprobe.agent.node_moves import complete_pending_node_move
from navprobe.agent.state import NavProbeAgentContext, NavProbeAgentState, NavProbeStepState
from navprobe.agent.tools import observe_place, update_frontiers


def refresh_state(
    *,
    context: NavProbeAgentContext,
    state: NavProbeAgentState,
    step: NavProbeStepState,
) -> dict[str, object]:
    """Refresh system-maintained navigation state after init or a world action."""
    observe_place(context=context, state=state, step=step)
    complete_pending_node_move(state=state)
    update_frontiers(context=context, state=state, step=step)
    summary = _refresh_state_summary(state=state, step=step)
    step.refresh_state_summary = summary
    return summary


def _refresh_state_summary(
    *,
    state: NavProbeAgentState,
    step: NavProbeStepState,
) -> dict[str, object]:
    floor_id = str(state.system.current_floor_id)
    frontier_update = step.frontier_update
    return {
        "current_place_node_id": None if step.current_place_node_id is None else str(step.current_place_node_id),
        "current_obs_id": None if step.current_obs_id is None else str(step.current_obs_id),
        "panorama_obs_count": len(step.panorama_obs_ids),
        "reused_place_obs_count": len(step.reused_place_obs_ids),
        "current_floor_id": floor_id,
        "active_frontier_count": len(state.frontier_records_for_floor(floor_id)),
        "frontier_filtering": {}
        if frontier_update is None
        else {
            "covered_frontier_removed_count": len(frontier_update.covered_frontier_ids_removed),
            "covered_frontier_ids_removed": list(frontier_update.covered_frontier_ids_removed),
            "stale_frontier_removed_count": len(frontier_update.stale_frontier_ids_removed),
            "stale_frontier_ids_removed": list(frontier_update.stale_frontier_ids_removed),
            "dilated_obstacle_frontier_removed_count": sum(
                1
                for item in frontier_update.stale_frontier_pruned_debug
                if str(item.get("reason", "")) == "frontier_xy_in_dilated_obstacle"
            ),
            "stale_frontier_pruned_debug": [
                dict(item)
                for item in frontier_update.stale_frontier_pruned_debug
                if bool(item.get("pruned", False))
            ],
            "filtered_candidate_count": len(frontier_update.filtered_candidates),
            "created_frontier_count": len(frontier_update.created_registered_frontier_ids),
            "place_coverage_pruned_frontier_count": len(frontier_update.place_coverage_pruned_frontier_debug),
        },
    }

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from navprobe.agent.waypoint.types import validate_waypoint_policy
from navprobe.agent.ablation import effective_max_retrieve_rounds
from navprobe.agent.ablation import normalize_ablation_name

if TYPE_CHECKING:
    from navprobe.config.settings import FSSConfig
    from navprobe.env.interface import EnvInterface
    from navprobe.mapping.exploration.bev_map import FrontierCandidate
    from navprobe.mapping.exploration.manager import ExplorationManager
    from navprobe.memory.graph.graph import Graph
    from navprobe.llm.client import LLMClient
    from navprobe.agent.action_execution import LocalmapActionExecutor
    from navprobe.mapping.frontier_update import LocalmapFrontierUpdateResult
    from navprobe.perception.goal import NavProbeGoalSpec
    from navprobe.perception.landmark_detection import LandmarkDetectionController
    from navprobe.schemas import ActionCall, ActionResult
    from navprobe.visualization.rendering import CacheSnapshot
    from navprobe.visualization.step_visualization import (
        StepVisualizationImageResult,
    )
    from navprobe.types import (
        LocalmapFrontierRecord,
        LocalmapOverlayRecord,
        LocalmapPlaceReuseState,
    )
    from navprobe.system_state import SystemState
    from navprobe.runtime.panorama_config import PanoramaConfig
    from navprobe.runtime.timing import StepTimingRecorder


@dataclass
class NavProbeAgentContext:
    args: argparse.Namespace
    env: EnvInterface
    panorama_config: PanoramaConfig
    global_bev_kwargs: dict[str, Any]
    run_dir: Path
    steps_dir: Path
    reset_payload: dict[str, Any]
    run_metadata: dict[str, Any]
    run_goal: str
    recorder_episode_id: int
    recorder_scene_id: str
    recorder_scene_name: str
    goal_spec: NavProbeGoalSpec | None


@dataclass
class NavProbeAgentState:
    system: SystemState
    llm_client: LLMClient
    action_executor: LocalmapActionExecutor
    waypoint_policy_name: str
    max_retrieve_rounds: int
    max_execution_preparations: int
    fss_config: "FSSConfig"
    ablation_name: str = "none"
    global_explorations_by_floor: dict[str, ExplorationManager] = field(default_factory=dict)
    frontier_filter_explorations_by_floor: dict[str, ExplorationManager] = field(default_factory=dict)
    frontier_records_by_floor: dict[str, dict[str, LocalmapFrontierRecord]] = field(default_factory=dict)
    overlay_records_by_floor: dict[str, dict[str, LocalmapOverlayRecord]] = field(default_factory=dict)
    recorded_step_count: int = 0
    _finalize_reason: str = "terminated"
    previous_place_node_id: str | None = None
    current_obs_id: str | None = None
    last_executed_action: dict[str, object] | None = None
    stop_result: dict[str, Any] | None = None
    reuse_current_place_next_step: LocalmapPlaceReuseState | None = None
    pending_node_move: dict[str, object] | None = None
    node_move_history: list[dict[str, object]] = field(default_factory=list)
    waypoint_selection_context_by_node_id: dict[str, dict[str, object]] = field(default_factory=dict)
    pending_terminal_check: dict[str, object] | None = None
    pending_task_state_initialization: dict[str, object] | None = None

    @property
    def completed_stair_items(self):
        return self.system.task_state.completed_stair_items

    def __post_init__(self) -> None:
        self.ablation_name = normalize_ablation_name(self.ablation_name)
        if int(self.max_retrieve_rounds) < 0:
            raise ValueError("max_retrieve_rounds must be non-negative")
        self.max_retrieve_rounds = effective_max_retrieve_rounds(
            self.ablation_name,
            self.max_retrieve_rounds,
        )
        validate_waypoint_policy(
            waypoint_policy_name=self.waypoint_policy_name,
        )

    @property
    def graph(self) -> Graph:
        return self.system.graph

    @property
    def cache(self):
        return self.system.cache

    def release_unused_overlay_images(self) -> None:
        """Release unreferenced overlays after the completed step is recorded."""
        retained_ids = set()
        for overlays in self.overlay_records_by_floor.values():
            for overlay in overlays.values():
                retained_ids.add(overlay.image_id)
                retained_ids.update(overlay.view_overlay_ids)
        for context in self.waypoint_selection_context_by_node_id.values():
            image_id = context.get("image_id")
            if image_id:
                retained_ids.add(str(image_id))
        # Place-node views belong to a single decision step and are rebuilt from
        # observations next time. Frontier views and Backtrack selections instead
        # remain live for as long as a floor/node references them.
        self.cache.discard_unreferenced_images(
            retained_ids=retained_ids,
            kinds={"overlay", "localmap_frontier_overlay", "place_node_overlay",
                   "last_waypoint_selection_overlay"},
        )


    @property
    def landmark_controller(self) -> LandmarkDetectionController:
        return self.system.landmark_controller

    @property
    def current_place_node_id(self) -> str | None:
        return self.system.current_node_id

    @current_place_node_id.setter
    def current_place_node_id(self, node_id: str | None) -> None:
        self.system.set_current_node_id(node_id)

    def ensure_floor_runtime_state(self, floor_id: str, bev_map_kwargs: dict[str, Any]) -> None:
        floor_id_str = str(floor_id).strip()
        if floor_id_str == "":
            raise ValueError("floor_id must be non-empty")
        self.frontier_records_by_floor.setdefault(floor_id_str, {})
        self.overlay_records_by_floor.setdefault(floor_id_str, {})
        if (
            floor_id_str not in self.global_explorations_by_floor
            or floor_id_str not in self.frontier_filter_explorations_by_floor
        ):
            from navprobe.mapping.exploration.bev_map import Map1Map2BEVMap
            from navprobe.mapping.exploration.manager import ExplorationManager

        if floor_id_str not in self.global_explorations_by_floor:
            self.global_explorations_by_floor[floor_id_str] = ExplorationManager(
                bev_map=Map1Map2BEVMap(**dict(bev_map_kwargs))
            )
        if floor_id_str not in self.frontier_filter_explorations_by_floor:
            self.frontier_filter_explorations_by_floor[floor_id_str] = ExplorationManager(
                bev_map=Map1Map2BEVMap(**dict(bev_map_kwargs))
            )

    def global_exploration_for_floor(self, floor_id: str) -> ExplorationManager:
        return self.global_explorations_by_floor[str(floor_id)]

    def frontier_filter_exploration_for_floor(self, floor_id: str) -> ExplorationManager:
        return self.frontier_filter_explorations_by_floor[str(floor_id)]

    def frontier_records_for_floor(self, floor_id: str) -> dict[str, LocalmapFrontierRecord]:
        return self.frontier_records_by_floor[str(floor_id)]

    def overlay_records_for_floor(self, floor_id: str) -> dict[str, LocalmapOverlayRecord]:
        return self.overlay_records_by_floor[str(floor_id)]


    @property
    def finalize_reason(self) -> str:
        return self._finalize_reason

    @finalize_reason.setter
    def finalize_reason(self, reason: str) -> None:
        reason_text = str(reason)
        self._finalize_reason = reason_text
        if reason_text != "terminated":
            self.system.mark_done(reason_text)


@dataclass
class NavProbeStepState:
    place_step_index: int
    step_dir: Path
    cache_before: CacheSnapshot
    llm_log_start_index: int
    timing: StepTimingRecorder
    policy_decision: dict[str, object] = field(default_factory=dict)
    visual_decision: dict[str, object] = field(default_factory=dict)
    node_summary: dict[str, object] = field(default_factory=dict)
    episodic_retrieval: dict[str, object] = field(default_factory=dict)
    agent_action: dict[str, object] = field(default_factory=dict)
    refresh_state_summary: dict[str, object] = field(default_factory=dict)
    place_reuse_state: LocalmapPlaceReuseState | None = None
    current_place_node_id: str | None = None
    anchor_obs_id: str | None = None
    current_obs_id: str | None = None
    panorama_obs_ids: list[str] = field(default_factory=list)
    angle_to_obs_id: dict[int, str] = field(default_factory=dict)
    panorama_views: list[dict[str, object]] = field(default_factory=list)
    panorama_angle_debug: dict[str, object] = field(default_factory=dict)
    reused_place_obs_ids: list[str] = field(default_factory=list)
    current_projection_obs_ids: list[str] = field(default_factory=list)
    local_exploration: ExplorationManager | None = None
    reachable_candidates: list[FrontierCandidate] = field(default_factory=list)
    frontier_update: LocalmapFrontierUpdateResult | None = None
    step_visualizations: StepVisualizationImageResult | None = None
    executed_action: dict[str, object] | None = None
    results: list[tuple[ActionCall, ActionResult]] = field(default_factory=list)
    navigation_segment_timings: list[dict[str, object]] = field(default_factory=list)
    current_obs_id_after: str | None = None
    created_frontier_ids: list[str] = field(default_factory=list)
    artifact_summary: dict[str, Any] = field(default_factory=dict)
    visual_waypoint_artifact_summary: dict[str, object] = field(default_factory=dict)
    step_timing_payload: dict[str, object] = field(default_factory=dict)

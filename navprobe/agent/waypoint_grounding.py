from __future__ import annotations

import hashlib

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import TYPE_CHECKING

import numpy as np

from navprobe.agent.visual_action_context import angle_for_direction
from navprobe.agent.visual_action_context import direction_for_angle
from navprobe.agent.visual_action_context import ordered_panorama_angles
from navprobe.agent.visual_action_context import VisualActionContext, VisualViewContext
from navprobe.agent.visual_grounding import VisualWaypoint
from navprobe.agent.navigation_decisions import NavProbeWaypointDecision
from navprobe.agent.visual_policy_prompt_images import image_content_for_array
from navprobe.agent.landmark_context import draw_landmarks_on_view_image
from navprobe.agent.landmark_context import NavProbeLandmarkEvidence
from navprobe.agent.fss_waypoint_sampling import draw_fss_waypoint_bev_overlay
from navprobe.agent.fss_waypoint_sampling import draw_fss_waypoint_rgb_overlay
from navprobe.agent.fss_waypoint_sampling import generate_fss_waypoint_candidates
from navprobe.agent.fss_waypoint_sampling import visual_waypoint_from_sampled_candidate
from navprobe.agent.fss_waypoint_sampling import NavProbeSampledWaypointCandidate
from navprobe.mapping.exploration.bev_map import local_map_read_scope
from navprobe.llm.request_config import NAVPROBE_MODEL_MAX_ATTEMPTS, model_request_budget

if TYPE_CHECKING:
    from navprobe.config.settings import FSSConfig
    from navprobe.llm.client import LLMClient
    from navprobe.mapping.exploration.manager import ExplorationManager
    from navprobe.mapping.exploration.bev_map import GlobalBEVMap
    from navprobe.runtime.cache import RuntimeCache
    from navprobe.types import LocalmapFrontierRecord
    from navprobe.visualization.action_mode_overlays import BevOverlayTransform


# The session caches candidate geometry by its preparation inputs.
# Model selection retries reuse these candidates; changed inputs get a new set.
NAVPROBE_WAYPOINT_CANDIDATE_SELECTOR_SYSTEM_PROMPT = """
You are the Waypoint Grounder in NavProbe.
Select a provided waypoint that advances the active objective in the supplied task progress, and identify the image supporting that choice.
""".strip()

NAVPROBE_WAYPOINT_GROUNDING_SYSTEM_PROMPT = """
You are the Waypoint Grounder in NavProbe.
Match the selected skill's local intent to a provided FSS waypoint for execution, or report that no candidate matches.
""".strip()


@dataclass(frozen=True)
class NavProbeWaypointLoopResult:
    local_move_plan: NavProbeWaypointDecision | None
    selected_view: VisualViewContext | None
    waypoint: VisualWaypoint | None
    attempt_records: list[dict[str, object]] = field(default_factory=list)
    navigation_replan_feedback: list[dict[str, object]] = field(default_factory=list)
    failure_reason: str = ""


@dataclass(frozen=True)
class _NavProbeViewCandidateSet:
    view: VisualViewContext
    candidates: list[NavProbeSampledWaypointCandidate]
    rgb_overlay: np.ndarray


@dataclass(frozen=True)
class NavProbePreparedWaypointCandidates:
    visual_context: VisualActionContext
    view_candidate_sets: list[_NavProbeViewCandidateSet]
    evidences: list[NavProbeLandmarkEvidence]
    candidate_bev_overlay: np.ndarray | None

    @property
    def candidates(self) -> list[NavProbeSampledWaypointCandidate]:
        return _all_view_candidates(self.view_candidate_sets)


    def selected(
        self,
        *,
        angle_deg: int,
        candidate_label: int,
    ) -> tuple[VisualViewContext, NavProbeSampledWaypointCandidate]:
        for item in self.view_candidate_sets:
            if int(item.view.angle_deg) != int(angle_deg):
                continue
            for candidate in item.candidates:
                if int(candidate.label) == int(candidate_label):
                    return item.view, candidate
        raise ValueError(
            "missing sampled waypoint candidate "
            f"direction={direction_for_angle(angle_deg)} label={candidate_label}"
        )


    def prompt_text(self, *, node_id: str) -> str:
        context_text = _waypoint_rgb_context_text(
            self.view_candidate_sets,
            evidences=self.evidences,
            include_graph_context=self.visual_context.graph_context_visible,
        )
        return "\n".join(
            [
                f"Sampled waypoint candidates for reference_node_id {node_id}:",
                context_text,
            ]
        )

    def prompt_images(self) -> list[tuple[str, np.ndarray]]:
        return [
            (
                _view_overlay_prompt_text(
                    item,
                    evidences=self.evidences,
                    include_graph_context=self.visual_context.graph_context_visible,
                ),
                np.asarray(item.rgb_overlay, dtype=np.uint8),
            )
            for item in _ordered_view_candidate_sets(self.view_candidate_sets)
        ]


SampledCandidateSetBuilder = Callable[
    ...,
    tuple[list[_NavProbeViewCandidateSet], list[dict[str, object]]],
]


def _build_sampled_candidate_sets(
    *,
    candidate_set_builder: SampledCandidateSetBuilder | None,
    exploration: "ExplorationManager",
    cache: "RuntimeCache",
    visual_context: VisualActionContext,
    robot_xy: np.ndarray,
    world_z: float,
    evidences: list[NavProbeLandmarkEvidence],
    frontier_records: dict[str, "LocalmapFrontierRecord"] | None,
    avoid_node_xys: list[tuple[float, float]] | None,
    node_dedup_map: "GlobalBEVMap | None" = None,
    node_dedup_radius_m: float,
    sampling_config: "FSSConfig",
    sample_spacing_m: float,
    max_distance_m: float,
) -> tuple[list[_NavProbeViewCandidateSet], list[dict[str, object]]]:
    with local_map_read_scope(getattr(exploration, "map", None)):
        common_kwargs = {
            "exploration": exploration,
            "cache": cache,
            "visual_context": visual_context,
            "robot_xy": robot_xy,
            "world_z": float(world_z),
            "evidences": evidences,
            "avoid_node_xys": avoid_node_xys,
        }
        if candidate_set_builder is not None:
            return candidate_set_builder(**common_kwargs)
        return _generate_all_view_sampled_candidates(
            **common_kwargs,
            frontier_records=frontier_records,
            node_dedup_map=node_dedup_map,
            node_dedup_radius_m=float(node_dedup_radius_m),
            sampling_config=sampling_config,
            sample_spacing_m=float(sample_spacing_m),
            max_distance_m=float(max_distance_m),
        )


def prepare_waypoint_candidates(
    *,
    cache: "RuntimeCache",
    visual_context: VisualActionContext,
    exploration: "ExplorationManager",
    floor_height_m: float,
    landmark_evidences: list[NavProbeLandmarkEvidence] | None = None,
    frontier_records: dict[str, "LocalmapFrontierRecord"] | None = None,
    avoid_node_xys: list[tuple[float, float]] | None = None,
    node_dedup_map: "GlobalBEVMap | None" = None,
    node_dedup_radius_m: float,
    candidate_set_builder: SampledCandidateSetBuilder | None = None,
    sampling_config: "FSSConfig",
    sample_spacing_m: float,
    max_distance_m: float,
) -> NavProbePreparedWaypointCandidates:
    evidences = list(landmark_evidences or [])
    robot_xy = _robot_xy_from_visual_context(cache=cache, visual_context=visual_context)
    view_candidate_sets, _ = _build_sampled_candidate_sets(
        candidate_set_builder=candidate_set_builder,
        exploration=exploration,
        cache=cache,
        visual_context=visual_context,
        robot_xy=robot_xy,
        world_z=float(floor_height_m),
        evidences=evidences,
        frontier_records=frontier_records,
        avoid_node_xys=avoid_node_xys,
        node_dedup_map=node_dedup_map,
        node_dedup_radius_m=float(node_dedup_radius_m),
        sampling_config=sampling_config,
        sample_spacing_m=float(sample_spacing_m),
        max_distance_m=float(max_distance_m),
    )
    all_candidates = _all_view_candidates(view_candidate_sets)
    candidate_bev_overlay = None
    if all_candidates != []:
        candidate_bev_overlay = draw_fss_waypoint_bev_overlay(
            exploration=exploration,
            robot_xy=robot_xy,
            candidates=all_candidates,
        )
    return NavProbePreparedWaypointCandidates(
        visual_context=visual_context,
        view_candidate_sets=view_candidate_sets,
        evidences=evidences,
        candidate_bev_overlay=candidate_bev_overlay,
    )


def _candidate_input_key(**inputs) -> str:
    """Fingerprint only candidate preparation inputs; keep payloads out of the key."""
    digest = hashlib.sha256()

    def add(value):
        if isinstance(value, np.ndarray):
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                add(key)
                add(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                add(item)
        elif hasattr(value, "to_dict"):
            add(value.to_dict())
        else:
            digest.update(repr(value).encode())
        digest.update(b"\0")

    def add_map(map_obj):
        add(id(map_obj))
        if map_obj is None:
            return
        # Both local traversability and global node-dedup visibility participate.
        for name in (
            "size", "pixels_per_meter", "episode_origin", "agent_radius",
            "astar_clearance_radius", "astar_clearance_weight", "depth_is_normalized",
            "obstacle_map", "dilated_obstacles", "navigable_map", "explored_area",
            "known_map", "free_map", "map1_free_map", "map2_free_map", "raw_free_map",
            "frontiers_px", "frontier_clusters_px",
        ):
            add(getattr(map_obj, name, None))

    add_map(getattr(inputs["exploration"], "map", None))
    with local_map_read_scope(inputs["node_dedup_map"]):
        add_map(inputs["node_dedup_map"])
    visual_context = inputs["visual_context"]
    add(visual_context.to_dict())
    cache = inputs["cache"]
    for view in visual_context.views:
        observation = cache.get_observation(str(view.obs_id)).observation
        for name in ("rgb", "depth", "intrinsics", "T_cam_odom", "T_odom_base"):
            add(getattr(observation, name, None))
        if view.node_overlay_image_id:
            add(cache.get_image(view.node_overlay_image_id).image)
    for name in ("robot_xy", "world_z", "evidences", "frontier_records", "avoid_node_xys",
                 "node_dedup_radius_m", "sample_spacing_m", "max_distance_m", "floor_id"):
        add(inputs[name])
    add(inputs.get("sampling_config"))
    add(id(inputs["candidate_set_builder"]))
    return digest.hexdigest()


def ground_waypoint(
    *,
    client: "LLMClient",
    cache: "RuntimeCache",
    goal_text: str,
    visual_context: VisualActionContext,
    task_state_assessment: str,
    task_state_text: str,
    exploration: "ExplorationManager",
    floor_height_m: float,
    planner_context_text: str = "",
    planner_context_images: list[tuple[str, np.ndarray]] | None = None,
    inherited_agent_context_content: list[dict[str, object]] | None = None,
    active_agenda_item: str = "",
    candidate_bev_landmark_markers: list[dict[str, object]] | None = None,
    candidate_bev_base_image: object | None = None,
    candidate_bev_transform: "BevOverlayTransform | None" = None,
    candidate_bev_coordinate_exploration: "ExplorationManager | None" = None,
    candidate_bev_reference_node_marker: dict[str, object] | None = None,
    candidate_bev_context_text: str = "",
    landmark_evidences: list[NavProbeLandmarkEvidence] | None = None,
    frontier_records: dict[str, "LocalmapFrontierRecord"] | None = None,
    avoid_node_xys: list[tuple[float, float]] | None = None,
    node_dedup_map: "GlobalBEVMap | None" = None,
    node_dedup_radius_m: float,
    candidate_set_builder: SampledCandidateSetBuilder | None = None,
    sampling_config: "FSSConfig",
    sample_spacing_m: float,
    max_distance_m: float,
    candidate_cache: dict | None = None,
    floor_id: str = "",
) -> NavProbeWaypointLoopResult:
    evidences: list[NavProbeLandmarkEvidence] = list(landmark_evidences or [])
    loop_events: list[dict[str, object]] = []
    attempt_records: list[dict[str, object]] = []
    navigation_replan_feedback: list[dict[str, object]] = []

    # Prepare geometry, projections, and candidate overlays before model selection.
    # Validation retries reuse these inputs.
    robot_xy = _robot_xy_from_visual_context(cache=cache, visual_context=visual_context)
    geometry_kwargs = dict(
        candidate_set_builder=candidate_set_builder, exploration=exploration, cache=cache,
        visual_context=visual_context, robot_xy=robot_xy, world_z=float(floor_height_m),
        evidences=evidences, frontier_records=frontier_records, avoid_node_xys=avoid_node_xys,
        node_dedup_map=node_dedup_map, node_dedup_radius_m=float(node_dedup_radius_m),
        sampling_config=sampling_config,
        sample_spacing_m=float(sample_spacing_m), max_distance_m=float(max_distance_m),
        floor_id=str(floor_id),
    )
    # The session owns this cache. Hash input contents (including mutable map/observation
    # arrays), never just a node ID. Display BEV is deliberately rendered below each time.
    with local_map_read_scope(getattr(exploration, "map", None)):
        cache_key = _candidate_input_key(**geometry_kwargs) if candidate_cache is not None else None
        if candidate_cache is not None and cache_key in candidate_cache:
            view_candidate_sets, candidate_generation_failures = candidate_cache[cache_key]
        else:
            builder_kwargs = dict(geometry_kwargs)
            builder_kwargs.pop("floor_id", None)
            view_candidate_sets, candidate_generation_failures = _build_sampled_candidate_sets(**builder_kwargs)
            if candidate_cache is not None:
                candidate_cache[cache_key] = (view_candidate_sets, candidate_generation_failures)
    all_candidates = _all_view_candidates(view_candidate_sets)
    if all_candidates == []:
        return NavProbeWaypointLoopResult(
            local_move_plan=None,
            selected_view=None,
            waypoint=None,
            attempt_records=attempt_records,
            navigation_replan_feedback=[{
                "failure_reason": "no_projected_sampled_waypoint_candidates",
                "candidate_count": 0, "candidate_generation_failures": deepcopy(candidate_generation_failures),
            }],
            failure_reason="no_projected_sampled_waypoint_candidates",
        )

    candidate_bev_overlay = draw_fss_waypoint_bev_overlay(
        exploration=exploration,
        robot_xy=robot_xy,
        candidates=all_candidates,
        landmark_markers=candidate_bev_landmark_markers,
        base_overlay=(
            None
            if candidate_bev_base_image is None
            else np.asarray(candidate_bev_base_image, dtype=np.uint8)
        ),
        base_transform=candidate_bev_transform,
        coordinate_exploration=candidate_bev_coordinate_exploration,
        reference_node_marker=candidate_bev_reference_node_marker,
    )

    with model_request_budget() as budget:
        for _ in range(NAVPROBE_MODEL_MAX_ATTEMPTS):
            if not budget.remaining:
                break
            candidate_response = _run_waypoint_candidate_selector(
                client=client,
                goal_text=goal_text,
                task_state_assessment=task_state_assessment,
                task_state_text=task_state_text,
                evidences=evidences,
                loop_events=loop_events,
                planner_context_text=planner_context_text,
                planner_context_images=planner_context_images,
                inherited_agent_context_content=inherited_agent_context_content,
                active_agenda_item=active_agenda_item,
                candidate_bev_landmark_markers=candidate_bev_landmark_markers,
                candidate_bev_context_text=candidate_bev_context_text,
                view_candidate_sets=view_candidate_sets,
                candidate_bev_overlay=candidate_bev_overlay,
                include_graph_context=visual_context.graph_context_visible,
            )
            if (
                "candidate_label" in candidate_response
                and candidate_response["candidate_label"] is None
                and str(candidate_response.get("reasoning", "")).strip()
            ):
                failure_reason = "navigation_intent_has_no_matching_waypoint"
                navigation_replan_feedback.append({
                    "failure_reason": failure_reason,
                    "failure_summary": str(candidate_response.get("failure_reason") or candidate_response["reasoning"]),
                    "grounding_response": deepcopy(candidate_response),
                    "candidate_count": len(all_candidates),
                    "candidate_labels_by_direction": {
                        direction_for_angle(item.view.angle_deg): [candidate.label for candidate in item.candidates]
                        for item in view_candidate_sets
                    },
                })
                return NavProbeWaypointLoopResult(
                    local_move_plan=None, selected_view=None, waypoint=None,
                    attempt_records=attempt_records,
                    navigation_replan_feedback=navigation_replan_feedback,
                    failure_reason=failure_reason,
                )
            selected_candidate, selected_view, candidate_failure_reason = _selected_sampled_candidate(
                response=candidate_response,
                view_candidate_sets=view_candidate_sets,
            )
            if selected_candidate is None or selected_view is None:
                budget.record_validation_error(ValueError(candidate_failure_reason))
                loop_events.append(
                    {
                        "feedback": {
                            "candidate_count": len(all_candidates),
                            "failure_reason": candidate_failure_reason,
                            "view_failures": candidate_generation_failures,
                        },
                        "overlay_images": {
                            "sampled_candidate_bev_overlay": candidate_bev_overlay,
                        },
                    }
                )
                continue
            candidate_rgb_overlay = _rgb_overlay_for_view(
                view_candidate_sets=view_candidate_sets,
                selected_view=selected_view,
            )
            waypoint_target = f"sampled candidate label {int(selected_candidate.label)}"
            local_move_plan = NavProbeWaypointDecision(
                selected_angle_deg=int(selected_view.angle_deg),
                waypoint_target=waypoint_target,
                reasoning=str(candidate_response.get("reasoning", "")).strip(),
                failure_reason="",
            )
            waypoint = visual_waypoint_from_sampled_candidate(
                selected_view=selected_view,
                candidate=selected_candidate,
                target=waypoint_target,
                raw_world_z=float(floor_height_m),
            )
            attempt_record = _attempt_record(
                decision=deepcopy(candidate_response),
                local_move_plan=local_move_plan,
                selected_view=selected_view,
                failure_reason="",
                overlay_images={
                    "sampled_candidate_rgb_overlay": candidate_rgb_overlay,
                    "sampled_candidate_bev_overlay": candidate_bev_overlay,
                },
                sampled_candidate=selected_candidate.to_dict(),
                candidate_selection=candidate_response,
                candidate_count=len(all_candidates),
            )
            attempt_records.append(attempt_record)
            return NavProbeWaypointLoopResult(
                local_move_plan=local_move_plan,
                selected_view=selected_view,
                waypoint=waypoint,
                attempt_records=attempt_records,
                navigation_replan_feedback=navigation_replan_feedback,
            )

    return NavProbeWaypointLoopResult(
        local_move_plan=None,
        selected_view=None,
        waypoint=None,
        attempt_records=attempt_records,
        navigation_replan_feedback=navigation_replan_feedback,
        failure_reason="vln_waypoint_loop_exhausted",
    )


def _generate_all_view_sampled_candidates(
    *,
    exploration: "ExplorationManager",
    cache: "RuntimeCache",
    visual_context: VisualActionContext,
    robot_xy: np.ndarray,
    world_z: float,
    evidences: list[NavProbeLandmarkEvidence],
    frontier_records: dict[str, "LocalmapFrontierRecord"] | None,
    avoid_node_xys: list[tuple[float, float]] | None,
    node_dedup_map: "GlobalBEVMap | None" = None,
    node_dedup_radius_m: float,
    sampling_config: "FSSConfig",
    sample_spacing_m: float,
    max_distance_m: float,
) -> tuple[list[_NavProbeViewCandidateSet], list[dict[str, object]]]:
    view_candidate_sets: list[_NavProbeViewCandidateSet] = []
    failures: list[dict[str, object]] = []
    views = [
        visual_context.view_for_angle(int(angle))
        for angle in _waypoint_overlay_angles(visual_context.available_angles)
    ]
    projected_candidates = generate_fss_waypoint_candidates(
        exploration=exploration,
        cache=cache,
        views=views,
        robot_xy=robot_xy,
        world_z=float(world_z),
        frontier_records=frontier_records,
        avoid_node_xys=avoid_node_xys,
        node_dedup_map=node_dedup_map,
        node_dedup_radius_m=float(node_dedup_radius_m),
        **sampling_config.candidate_kwargs(),
        sample_spacing_m=float(sample_spacing_m),
        max_distance_m=float(max_distance_m),
    )
    candidates_by_angle: dict[int, list[NavProbeSampledWaypointCandidate]] = {}
    for item in projected_candidates:
        candidates_by_angle.setdefault(int(item.view.angle_deg), []).append(item.candidate)
    for selected_view in views:
        candidates = candidates_by_angle.get(int(selected_view.angle_deg), [])
        if candidates == []:
            failures.append(
                {
                    "selected_direction": direction_for_angle(selected_view.angle_deg),
                    "selected_obs_id": str(selected_view.obs_id),
                    "failure_reason": "no_projected_sampled_waypoint_candidates_for_view",
                }
            )
        candidate_rgb_overlay = draw_fss_waypoint_rgb_overlay(
            cache=cache,
            selected_view=selected_view,
            candidates=candidates,
        )
        candidate_rgb_overlay = draw_landmarks_on_view_image(
            image=candidate_rgb_overlay,
            cache=cache,
            view=selected_view,
            evidences=evidences,
        )
        view_candidate_sets.append(
            _NavProbeViewCandidateSet(
                view=selected_view,
                candidates=candidates,
                rgb_overlay=candidate_rgb_overlay,
            )
        )
    return view_candidate_sets, failures


def _all_view_candidates(view_candidate_sets: list[_NavProbeViewCandidateSet]) -> list[NavProbeSampledWaypointCandidate]:
    candidates_by_label: dict[int, NavProbeSampledWaypointCandidate] = {}
    for item in view_candidate_sets:
        for candidate in item.candidates:
            candidates_by_label.setdefault(int(candidate.label), candidate)
    return [
        candidates_by_label[label]
        for label in sorted(candidates_by_label)
    ]


def _rgb_overlay_for_view(
    *,
    view_candidate_sets: list[_NavProbeViewCandidateSet],
    selected_view: VisualViewContext,
) -> np.ndarray:
    for item in view_candidate_sets:
        if int(item.view.angle_deg) == int(selected_view.angle_deg):
            return np.asarray(item.rgb_overlay, dtype=np.uint8)
    raise ValueError(
        "missing sampled waypoint overlay for "
        f"{direction_for_angle(selected_view.angle_deg)}"
    )


def _run_waypoint_candidate_selector(
    *,
    client: "LLMClient",
    goal_text: str,
    task_state_assessment: str,
    task_state_text: str,
    evidences: list[NavProbeLandmarkEvidence],
    loop_events: list[dict[str, object]],
    planner_context_text: str,
    planner_context_images: list[tuple[str, np.ndarray]] | None,
    inherited_agent_context_content: list[dict[str, object]] | None,
    active_agenda_item: str = "",
    candidate_bev_landmark_markers: list[dict[str, object]] | None = None,
    candidate_bev_context_text: str = "",
    view_candidate_sets: list[_NavProbeViewCandidateSet],
    candidate_bev_overlay: np.ndarray,
    include_graph_context: bool,
) -> dict[str, object]:
    active_item = str(active_agenda_item).strip()
    if active_item or inherited_agent_context_content:
        prompt_text = _build_inherited_waypoint_grounding_prompt(
            loop_events=loop_events,
        )
    else:
        prompt_text = _build_waypoint_candidate_selector_prompt(
            goal_text=goal_text,
            task_state_assessment=task_state_assessment,
            task_state_text=task_state_text,
            loop_events=loop_events,
            planner_context_text=planner_context_text,
            view_candidate_sets=view_candidate_sets,
            include_graph_context=include_graph_context,
        )
    user_prompt: list[dict[str, object]] = []
    if not inherited_agent_context_content:
        user_prompt.append({"type": "text", "text": prompt_text})
    user_prompt.extend(deepcopy(list(inherited_agent_context_content or [])))
    _append_planner_context_blocks(
        user_prompt=user_prompt,
        planner_context_text=planner_context_text,
        planner_context_images=planner_context_images,
    )
    candidate_bev_content = [
        {
            "type": "text",
            "text": _candidate_bev_prompt_text(
                candidate_bev_landmark_markers,
                candidate_bev_context_text=candidate_bev_context_text,
            ),
        },
        image_content_for_array(candidate_bev_overlay),
    ]
    if inherited_agent_context_content:
        user_prompt.extend(candidate_bev_content)
    user_prompt.append(
        {
            "type": "text",
            "text": _waypoint_rgb_context_text(
                view_candidate_sets,
                evidences=evidences,
                include_graph_context=include_graph_context,
            ),
        }
    )
    for item in _ordered_view_candidate_sets(view_candidate_sets):
        user_prompt.append(
            {
                "type": "text",
                "text": _view_overlay_prompt_text(
                    item,
                    evidences=evidences,
                    include_graph_context=include_graph_context,
                ),
            }
        )
        user_prompt.append(
            image_content_for_array(np.asarray(item.rgb_overlay, dtype=np.uint8))
        )
    if not inherited_agent_context_content:
        user_prompt.extend(candidate_bev_content)
    if not inherited_agent_context_content:
        for block in _loop_context_blocks(loop_events):
            text = str(block.get("text", "")).strip()
            if text != "":
                user_prompt.append({"type": "text", "text": text})
            for label, image in list(block.get("images", [])):
                user_prompt.append({"type": "text", "text": str(label)})
                user_prompt.append(image_content_for_array(np.asarray(image, dtype=np.uint8)))
    if inherited_agent_context_content:
        user_prompt.append({"type": "text", "text": prompt_text})
    if active_item or inherited_agent_context_content:
        system_prompt = NAVPROBE_WAYPOINT_GROUNDING_SYSTEM_PROMPT
    else:
        system_prompt = NAVPROBE_WAYPOINT_CANDIDATE_SELECTOR_SYSTEM_PROMPT
    if active_item == "" and not inherited_agent_context_content and not (
        str(task_state_assessment).strip() or str(task_state_text).strip()
    ):
        system_prompt = """
You are the Waypoint Grounder in NavProbe.
Select a provided local waypoint that advances the navigation instruction, and identify the image supporting that choice.
""".strip()
    return client._create_visual_json_completion(
        call_name="waypoint_grounder.select",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=client.request_limits["waypoint_grounder"]["max_tokens"],
        token_field="max_completion_tokens",
    )


def _append_planner_context_blocks(
    *,
    user_prompt: list[dict[str, object]],
    planner_context_text: str,
    planner_context_images: list[tuple[str, np.ndarray]] | None,
) -> None:
    context_text = str(planner_context_text).strip()
    images = list(planner_context_images or [])
    if context_text == "" and images == []:
        return
    if context_text != "":
        user_prompt.append({"type": "text", "text": f"Waypoint reference context:\n{context_text}"})
    for label, image in images:
        user_prompt.append({"type": "text", "text": str(label)})
        user_prompt.append(image_content_for_array(np.asarray(image, dtype=np.uint8)))


def _planner_context_reference_text(planner_context_text: str) -> str:
    if str(planner_context_text).strip() == "":
        return ""
    return "See the additional route context attached after this prompt text."


def _landmark_marker_labels_text(
    landmark_markers: list[dict[str, object]] | None,
) -> str:
    items = [
        f"{int(marker['label'])} {str(marker.get('class_name', '')).strip()}".strip()
        for marker in list(landmark_markers or [])
        if marker.get("label") is not None
    ]
    return ", ".join(items)


def _candidate_bev_prompt_text(
    landmark_markers: list[dict[str, object]] | None,
    *,
    candidate_bev_context_text: str = "",
) -> str:
    text = (
        "Candidate endpoint BEV. White numbered circles are the selectable waypoint "
        "endpoints and use the same labels as the RGB overlays."
    )
    graph_context = str(candidate_bev_context_text).strip()
    if graph_context != "":
        text += "\n" + graph_context
    landmark_text = _landmark_marker_labels_text(landmark_markers)
    if landmark_text != "":
        text += f"\nWhite numbered squares are landmarks: {landmark_text}."
    return text


def _build_inherited_waypoint_grounding_prompt(
    *,
    loop_events: list[dict[str, object]],
) -> str:
    retry_summary = _loop_context_summary(loop_events)
    retry_section = (
        f"\n\nValidation feedback from the previous response:\n{retry_summary}\n"
        "Return one corrected response using the same output contract."
        if retry_summary != "none"
        else ""
    )
    selection_rules = """
Selection rules:
- Read the local target, direction, constraints, and evidence from the selected skill's `reason`, `objective`, or `waypoint_target`. Match this intent to the candidates; do not replace it with a different navigation objective.
- Compare the RGB projections and BEV geometry. Select a candidate on visible connected floor, a doorway, a corridor, a stair tread or landing when relevant, or other safe free space. Nearby overlay labels alone do not establish spatial proximity.
- Reject projections onto walls, ceilings, furniture, object surfaces, clutter, windows, mirrors, railings, or door leaves. For an object or fixture target, choose nearby reachable floor.
- Check detector labels against the visible scene and the skill's evidence about the target.
""".strip()
    return f"""
Input meanings:
- The selected skill supplies the local intent. Candidate labels identify world-space waypoints; repeated labels across views refer to the same waypoint.
- RGB overlays show where candidates project onto visible surfaces; the BEV shows their geometry. The reference context identifies the view's relationship to the physical robot pose.
- The skill's `direction` constrains the world-space route in that reference frame. `selected_direction` identifies the RGB view that best supports the chosen waypoint; it may differ when the same candidate is clearer in another view.

{selection_rules}

Output contract:
Return only JSON. For a supported match, select one available label visible in selected_direction and explain its visual and geometric support:
{{
  "reasoning": "<concise geometric and visual evidence for the local intent>",
  "selected_direction": "front|back|left|right",
  "candidate_label": 1
}}
If no candidate matches the local intent on a supported walking surface, return `candidate_label=null`, `selected_direction=null`, and explain the mismatch in `reasoning` and `failure_reason`.{retry_section}
""".strip()


def _build_waypoint_candidate_selector_prompt(
    *,
    goal_text: str,
    task_state_assessment: str,
    task_state_text: str,
    loop_events: list[dict[str, object]],
    planner_context_text: str,
    view_candidate_sets: list[_NavProbeViewCandidateSet],
    include_graph_context: bool,
) -> str:
    planner_context_reference = _planner_context_reference_text(planner_context_text)
    planner_context_section = ""
    if planner_context_reference != "":
        planner_context_section = f"""

Additional route context:
{planner_context_reference}
""".rstrip()
    has_task_state = bool(
        str(task_state_assessment).strip() or str(task_state_text).strip()
    )
    task_state_section = ""
    if has_task_state:
        task_state_section = f"""
Task-state assessment:
{task_state_assessment}

Updated task-state memory:
{task_state_text}
""".rstrip()
    overlay_contents = "view direction, available waypoint labels, and visible landmarks"
    if include_graph_context:
        overlay_contents = (
            "view direction, available waypoint labels, visible visited nodes, and visible landmarks"
        )
    evidence_description = f"""
Input meanings:
The {len(view_candidate_sets)} labeled RGB overlays show candidate projections on visible surfaces; the BEV shows the same endpoints in world space. Each RGB overlay has a caption giving its {overlay_contents}. Repeated labels across views identify one waypoint, not separate choices.
""".strip()
    if has_task_state:
        objective_rule = (
            "Compare candidates against the current active agenda item using the supplied assessment, "
            "updated task-state memory, and images. Later agenda items may break a tie only between "
            "candidates that already support the active item."
        )
        agenda_evidence_schema = (
            "<visible RGB overlay evidence that supports the current active agenda item>"
        )
        reasoning_schema = (
            "<identify the current active agenda item, compare candidate support for that "
            "item across the provided views, and explain why the selected waypoint best advances it>"
        )
    else:
        objective_rule = (
            "Compare candidates against the navigation instruction using the supplied route context "
            "and images. Base the choice on visible support for the task."
        )
        agenda_evidence_schema = (
            "<visible RGB overlay evidence that supports the navigation task>"
        )
        reasoning_schema = (
            "<compare candidate support for the navigation task across the provided views "
            "and explain why the selected waypoint best advances it>"
        )
    retry_summary = _loop_context_summary(loop_events)
    retry_section = (
        f"\n\nValidation feedback from the previous response:\n{retry_summary}"
        if retry_summary != "none" else ""
    )
    return f"""
Navigation task:
{goal_text}
{task_state_section}
{planner_context_section}

{evidence_description}

Select a waypoint:
- {objective_rule}
- Compare the candidates across views and in the BEV. Choose visible connected floor, a stair tread or landing, a doorway, a corridor, or other safe free space.
- Reject projections onto walls, ceilings, furniture or object surfaces, clutter, windows, mirrors, or door leaves. Distance to the robot alone is not a reason to choose a candidate.
- Use the RGB view with the clearest task-relevant evidence for the selected waypoint. The label determines the world-space endpoint; the view supplies its visual grounding.
{retry_section}

Output contract:
Return only JSON. Choose one provided candidate_label visible in selected_direction; leave failure_reason empty after a valid selection. The example below shows the field structure:
{{
  "agenda_alignment_evidence": "{agenda_evidence_schema}",
  "reasoning": "{reasoning_schema}",
  "selected_direction": "front",
  "candidate_label": 1,
  "failure_reason": ""
}}
""".strip()


def _waypoint_overlay_angles(available_angles: list[int]) -> list[int]:
    return ordered_panorama_angles(available_angles)


def _ordered_view_candidate_sets(
    view_candidate_sets: list[_NavProbeViewCandidateSet],
) -> list[_NavProbeViewCandidateSet]:
    by_angle = {int(item.view.angle_deg): item for item in view_candidate_sets}
    return [
        by_angle[angle]
        for angle in ordered_panorama_angles(list(by_angle))
    ]


def _waypoint_rgb_context_text(
    view_candidate_sets: list[_NavProbeViewCandidateSet],
    *,
    evidences: list[NavProbeLandmarkEvidence],
    include_graph_context: bool,
) -> str:
    lines = [
        f"Waypoint RGB overlays: {len(view_candidate_sets)} directional views.",
        "- White numbered circles are selectable waypoint candidates.",
        "- The same label in multiple views is one world-space waypoint.",
    ]
    if include_graph_context:
        lines.append(
            "- Blue numbered circles are visited place-node overlays, not physical objects or selectable waypoints."
        )
    if evidences != []:
        lines.append(
            "- Landmark boxes display the numeric suffix of each canonical landmark ref and are not waypoint surfaces."
        )
    return "\n".join(lines)


def _robot_xy_from_visual_context(
    *,
    cache: "RuntimeCache",
    visual_context: VisualActionContext,
) -> np.ndarray:
    current_view = visual_context.view_for_angle(0)
    observation = cache.get_observation(str(current_view.obs_id)).observation
    if observation.T_odom_base is not None:
        transform = np.asarray(observation.T_odom_base, dtype=np.float64)
        return np.asarray([float(transform[0, 3]), float(transform[1, 3])], dtype=np.float64)
    return np.asarray([float(observation.pose.x), float(observation.pose.y)], dtype=np.float64)


def _candidate_labels_text(candidates: list[NavProbeSampledWaypointCandidate]) -> str:
    labels = [str(int(candidate.label)) for candidate in candidates]
    return ", ".join(labels) if labels != [] else "none"


def _visible_node_labels_text(view: "VisualViewContext") -> str:
    items: list[str] = []
    for node_id in view.visible_visited_nodes:
        node_id_text = str(node_id).strip()
        if node_id_text == "":
            continue
        items.append(node_id_text)
    return ", ".join(items) if items != [] else "none"


def _view_overlay_prompt_text(
    view_candidate_set: _NavProbeViewCandidateSet,
    *,
    evidences: list[NavProbeLandmarkEvidence],
    include_graph_context: bool,
) -> str:
    angle = int(view_candidate_set.view.angle_deg)
    lines = [
        f"Sampled waypoint RGB overlay for {direction_for_angle(angle)}.",
        f"- Available waypoint labels: {_candidate_labels_text(view_candidate_set.candidates)}.",
    ]
    if include_graph_context:
        visible_nodes = _visible_node_labels_text(view_candidate_set.view)
        if visible_nodes != "none":
            lines.append(f"- Visible visited nodes: {visible_nodes}.")
        if view_candidate_set.view.visible_arrival_edge_ids:
            lines.append(
                "- The blue curve and arrows show the last executed movement toward "
                "the current position, smoothed for display."
            )
    visible_landmarks = _landmark_labels_text(evidences, angle_deg=angle)
    if visible_landmarks != "none":
        lines.append(f"- Visible landmarks: {visible_landmarks}.")
    return "\n".join(lines)


def _landmark_labels_text(evidences: list[NavProbeLandmarkEvidence], *, angle_deg: int) -> str:
    items: list[str] = []
    for evidence in evidences:
        if int(evidence.angle_deg) != int(angle_deg):
            continue
        class_name = str(evidence.class_name).strip()
        landmark_id = str(evidence.landmark_id).strip()
        if class_name == "" or landmark_id == "":
            continue
        item = f"{landmark_id} {class_name} confidence={float(evidence.score):.2f}"
        if item not in items:
            items.append(item)
    return ", ".join(items) if items != [] else "none"


def _selected_sampled_candidate(
    *,
    response: dict[str, object],
    view_candidate_sets: list[_NavProbeViewCandidateSet],
) -> tuple[NavProbeSampledWaypointCandidate | None, VisualViewContext | None, str]:
    reasoning = str(response.get("reasoning", "")).strip()
    if reasoning == "":
        return None, None, "candidate_selector_missing_reasoning"
    raw_label = response.get("candidate_label")
    if raw_label is None:
        return None, None, str(response.get("failure_reason", "candidate_selector_returned_null_label")).strip()
    raw_direction = response.get("selected_direction")
    if raw_direction is None:
        return None, None, "candidate_selector_missing_selected_direction"
    try:
        selected_angle = angle_for_direction(raw_direction)
    except ValueError:
        return None, None, f"candidate_selector_invalid_direction:{raw_direction!r}"
    try:
        label = int(raw_label)
    except (TypeError, ValueError):
        return None, None, f"candidate_selector_invalid_label:{raw_label!r}"
    available_angles: list[int] = []
    for item in view_candidate_sets:
        for candidate in item.candidates:
            if int(candidate.label) != label:
                continue
            angle = int(item.view.angle_deg)
            available_angles.append(angle)
            if angle == selected_angle:
                return candidate, item.view, ""
    if available_angles != []:
        directions_text = "_".join(
            direction_for_angle(angle) for angle in available_angles
        )
        return None, None, (
            f"candidate_selector_direction_label_mismatch:"
            f"{direction_for_angle(selected_angle)}_label_{label}_"
            f"available_at_{directions_text}"
        )
    return None, None, f"candidate_selector_unavailable_label:{label}"


def _attempt_record(
    *,
    decision: dict[str, object],
    local_move_plan: NavProbeWaypointDecision,
    selected_view: VisualViewContext,
    failure_reason: str,
    overlay_images: dict[str, np.ndarray] | None = None,
    sampled_candidate: dict[str, object] | None = None,
    candidate_selection: dict[str, object] | None = None,
    candidate_count: int | None = None,
) -> dict[str, object]:
    return {
        "decision": deepcopy(decision),
        "selected_angle_deg": int(selected_view.angle_deg),
        "selected_obs_id": str(selected_view.obs_id),
        "waypoint_target": str(local_move_plan.waypoint_target),
        "failure_reason": str(failure_reason),
        "overlay_images": {} if overlay_images is None else dict(overlay_images),
        "sampled_candidate": {} if sampled_candidate is None else deepcopy(sampled_candidate),
        "candidate_selection": {} if candidate_selection is None else deepcopy(candidate_selection),
        "candidate_count": None if candidate_count is None else int(candidate_count),
    }


def _loop_context_summary(loop_events: list[dict[str, object]]) -> str:
    if loop_events == []:
        return "none"
    lines: list[str] = []
    for index, event in enumerate(loop_events, start=1):
        feedback = event["feedback"]
        lines.append(
            f"Previous planner response {index}: the response did not identify a valid sampled waypoint candidate. "
            f"Feedback: {json.dumps(feedback, ensure_ascii=False)}"
        )
    return "\n".join(lines)


def _loop_context_blocks(loop_events: list[dict[str, object]]) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    for index, event in enumerate(loop_events, start=1):
        text = _loop_context_summary([event])
        images: list[tuple[str, np.ndarray]] = []
        image = event["overlay_images"]["sampled_candidate_bev_overlay"]
        if isinstance(image, np.ndarray):
            images.append((f"Loop context event {index} sampled_candidate_bev_overlay", image))
        blocks.append({"text": text, "images": images})
    return blocks

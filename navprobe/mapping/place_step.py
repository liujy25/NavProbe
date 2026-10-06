from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from navprobe.env.interface import EnvInterface
from navprobe.env.episode_status import episode_over
from navprobe.mapping.exploration.bev_map import FrontierCandidate
from navprobe.mapping.exploration.manager import ExplorationManager
from navprobe.memory.graph.graph import Graph
from navprobe.mapping.frontier_generation import _build_local_candidate_proposals
from navprobe.perception.landmark_detection import LandmarkDetectionController
from navprobe.perception.panorama import _capture_panorama
from navprobe.perception.panorama import PanoramaView
from navprobe.mapping.place_graph_ops import _create_place_node
from navprobe.runtime.panorama_config import PanoramaConfig
from navprobe.types import LocalmapPlaceReuseState
from navprobe.runtime.cache import RuntimeCache
from navprobe.runtime.timing import StepTimingRecorder


@dataclass
class LocalmapPlaceStepResult:
    current_place_node_id: str
    anchor_obs_id: str
    current_obs_id: str
    panorama_obs_ids: list[str] = field(default_factory=list)
    panorama_views: list[PanoramaView] = field(default_factory=list)
    angle_to_obs_id: dict[int, str] = field(default_factory=dict)
    panorama_angle_debug: dict[str, object] = field(default_factory=dict)
    reused_place_obs_ids: list[str] = field(default_factory=list)
    current_projection_obs_ids: list[str] = field(default_factory=list)
    local_exploration: ExplorationManager | None = None
    reachable_candidates: list[FrontierCandidate] = field(default_factory=list)


def prepare_localmap_place_step(
    *,
    env: EnvInterface,
    cache: RuntimeCache,
    graph: Graph,
    global_exploration: ExplorationManager,
    landmark_controller: LandmarkDetectionController,
    bev_map_kwargs: dict[str, object],
    panorama_config: PanoramaConfig | None,
    current_place_node_id: str | None,
    current_floor_id: str,
    previous_place_node_id: str | None,
    place_reuse_state: LocalmapPlaceReuseState | None,
    timing: StepTimingRecorder,
    current_step_count: Callable[[], int],
) -> LocalmapPlaceStepResult:
    if place_reuse_state is not None:
        result = _reuse_current_place(
            graph=graph,
            current_place_node_id=current_place_node_id,
            place_reuse_state=place_reuse_state,
            panorama_config=panorama_config,
            timing=timing,
            current_step_count=current_step_count,
        )
    else:
        result = _capture_and_create_place(
            env=env,
            cache=cache,
            graph=graph,
            landmark_controller=landmark_controller,
            panorama_config=panorama_config,
            current_floor_id=current_floor_id,
            timing=timing,
            current_step_count=current_step_count,
        )
        if previous_place_node_id is not None and previous_place_node_id != result.current_place_node_id:
            graph.add_edge(
                src_id=str(previous_place_node_id),
                dst_id=str(result.current_place_node_id),
                relation="move",
            )

    _prepare_localmap_candidates(
        cache=cache,
        graph=graph,
        bev_map_kwargs=bev_map_kwargs,
        result=result,
        place_reuse_state=place_reuse_state,
        timing=timing,
        current_step_count=current_step_count,
    )
    _register_place_localmap_globalmap(
        global_exploration=global_exploration,
        result=result,
        place_reuse_state=place_reuse_state,
        timing=timing,
        current_step_count=current_step_count,
    )
    return result


def _register_place_localmap_globalmap(
    *,
    global_exploration: ExplorationManager,
    result: LocalmapPlaceStepResult,
    place_reuse_state: LocalmapPlaceReuseState | None,
    timing: StepTimingRecorder,
    current_step_count: Callable[[], int],
) -> None:
    should_register = (
        place_reuse_state is None
        or str(place_reuse_state.reason) == "vertical_transition_completed"
    )
    if not should_register:
        return
    if result.local_exploration is None:
        raise ValueError("cannot register place globalmap without local_exploration")
    timing.start("register_place_localmap_globalmap", current_step_count())
    merge_details = global_exploration.merge_raw_layers_from_local_exploration(
        result.local_exploration,
        obs_ids=result.current_projection_obs_ids,
    )
    timing.stop(
        "register_place_localmap_globalmap",
        current_step_count(),
        details={
            "current_place_node_id": str(result.current_place_node_id),
            "registered_obs_count": len(result.current_projection_obs_ids),
            "reason": "new_place" if place_reuse_state is None else str(place_reuse_state.reason),
            "merge_details": merge_details,
        },
    )


def _reuse_current_place(
    *,
    graph: Graph,
    current_place_node_id: str | None,
    place_reuse_state: LocalmapPlaceReuseState,
    panorama_config: PanoramaConfig | None,
    timing: StepTimingRecorder,
    current_step_count: Callable[[], int],
) -> LocalmapPlaceStepResult:
    timing.start("reuse_current_place", current_step_count())
    if current_place_node_id is None:
        raise ValueError("cannot reuse current place before any place node exists")
    current_place_node_id = str(place_reuse_state.place_node_id)
    current_place_node = graph.get_node(str(current_place_node_id))
    reused_place_obs_ids = [str(obs_id) for obs_id in current_place_node.obs_ids]
    if reused_place_obs_ids == []:
        raise ValueError(f"cannot reuse place node {current_place_node_id!r} without obs_ids")
    anchor_obs_id = str(reused_place_obs_ids[0])
    current_obs_id = str(anchor_obs_id)
    current_projection_obs_ids = list(reused_place_obs_ids)
    angle_to_obs_id = _angle_to_obs_id_from_ordered_obs_ids(reused_place_obs_ids, panorama_config=panorama_config)
    timing.stop(
        "reuse_current_place",
        current_step_count(),
        details={
            "current_place_node_id": str(current_place_node_id),
            "reason": str(place_reuse_state.reason),
            "reused_place_obs_count": len(reused_place_obs_ids),
            "current_projection_obs_count": len(current_projection_obs_ids),
        },
    )
    return LocalmapPlaceStepResult(
        current_place_node_id=str(current_place_node_id),
        anchor_obs_id=anchor_obs_id,
        current_obs_id=current_obs_id,
        panorama_obs_ids=[],
        panorama_views=[],
        angle_to_obs_id=angle_to_obs_id,
        panorama_angle_debug={"reused_place": True},
        reused_place_obs_ids=reused_place_obs_ids,
        current_projection_obs_ids=current_projection_obs_ids,
    )


def _capture_and_create_place(
    *,
    env: EnvInterface,
    cache: RuntimeCache,
    graph: Graph,
    landmark_controller: LandmarkDetectionController,
    panorama_config: PanoramaConfig | None,
    current_floor_id: str,
    timing: StepTimingRecorder,
    current_step_count: Callable[[], int],
) -> LocalmapPlaceStepResult:
    timing.start("capture_panorama", current_step_count())
    panorama = _capture_panorama(
        env=env,
        cache=cache,
        global_exploration=None,
        landmark_controller=landmark_controller,
        panorama_config=panorama_config,
        stop_capture=lambda: episode_over(env),
    )
    timing.stop(
        "capture_panorama",
        current_step_count(),
        details={
            "panorama_obs_count": len(panorama.obs_ids),
            "angle_to_obs_id": {str(angle): obs_id for angle, obs_id in panorama.angle_to_obs_id.items()},
            "panorama_warning_count": len(list(panorama.angle_debug.get("warnings", []))),
        },
    )

    timing.start("create_place_node", current_step_count())
    anchor_obs_id = str(panorama.anchor_obs_id)
    current_obs_id = str(panorama.current_obs_id)
    current_place_node_id = _create_place_node(
        graph=graph,
        cache=cache,
        anchor_obs_id=anchor_obs_id,
        obs_ids=panorama.obs_ids,
        floor_id=str(current_floor_id),
    )
    timing.stop(
        "create_place_node",
        current_step_count(),
        details={"current_place_node_id": str(current_place_node_id)},
    )

    current_projection_obs_ids = [str(obs_id) for obs_id in panorama.obs_ids]
    return LocalmapPlaceStepResult(
        current_place_node_id=str(current_place_node_id),
        anchor_obs_id=anchor_obs_id,
        current_obs_id=current_obs_id,
        panorama_obs_ids=[str(obs_id) for obs_id in panorama.obs_ids],
        panorama_views=list(panorama.views),
        angle_to_obs_id=dict(panorama.angle_to_obs_id),
        panorama_angle_debug=dict(panorama.angle_debug),
        current_projection_obs_ids=current_projection_obs_ids,
    )


def _prepare_localmap_candidates(
    *,
    cache: RuntimeCache,
    graph: Graph,
    bev_map_kwargs: dict[str, object],
    result: LocalmapPlaceStepResult,
    place_reuse_state: LocalmapPlaceReuseState | None,
    timing: StepTimingRecorder,
    current_step_count: Callable[[], int],
) -> None:
    if place_reuse_state is not None and _reuse_should_register_frontiers(place_reuse_state):
        timing.start("generate_reused_localmap", current_step_count())
        result.local_exploration, result.reachable_candidates = _build_local_candidate_proposals(
            cache=cache,
            obs_ids=result.current_projection_obs_ids,
            bev_map_kwargs=bev_map_kwargs,
        )
        current_place_node = graph.get_node(str(result.current_place_node_id))
        current_place_node.localmap = result.local_exploration
        timing.stop(
            "generate_reused_localmap",
            current_step_count(),
            details={
                "current_place_node_id": str(result.current_place_node_id),
                "reachable_candidate_count": len(result.reachable_candidates),
                "reason": str(place_reuse_state.reason),
            },
        )
        return

    if place_reuse_state is not None:
        timing.start("reuse_localmap_state", current_step_count())
        current_place_node = graph.get_node(str(result.current_place_node_id))
        result.local_exploration = current_place_node.localmap
        if result.local_exploration is None:
            raise ValueError(
                f"missing local exploration map for reused place node {result.current_place_node_id!r}"
            )
        result.reachable_candidates = []
        timing.stop(
            "reuse_localmap_state",
            current_step_count(),
            details={
                "current_place_node_id": str(result.current_place_node_id),
                "skipped_generate_localmap": True,
            },
        )
        return

    timing.start("generate_localmap", current_step_count())
    result.local_exploration, result.reachable_candidates = _build_local_candidate_proposals(
        cache=cache,
        obs_ids=result.panorama_obs_ids,
        bev_map_kwargs=bev_map_kwargs,
    )
    current_place_node = graph.get_node(str(result.current_place_node_id))
    current_place_node.localmap = result.local_exploration
    timing.stop(
        "generate_localmap",
        current_step_count(),
        details={"reachable_candidate_count": len(result.reachable_candidates)},
    )


def _reuse_should_register_frontiers(place_reuse_state: LocalmapPlaceReuseState) -> bool:
    recovery = getattr(place_reuse_state, "recovery", {})
    return isinstance(recovery, dict) and bool(recovery.get("register_frontiers_on_reuse", False))


def _angle_to_obs_id_from_ordered_obs_ids(
    obs_ids: list[str],
    *,
    panorama_config: PanoramaConfig | None,
) -> dict[int, str]:
    if panorama_config is not None:
        angles = [
            int(round((float(index) * float(panorama_config.observation_spacing_degrees)) % 360.0))
            for index in range(int(panorama_config.observation_count))
        ]
    else:
        count = len(obs_ids)
        angles = [
            int(round((float(index) * 360.0 / float(count)) % 360.0))
            for index in range(count)
        ]
    return {
        int(angle): str(obs_id)
        for angle, obs_id in zip(angles, [str(item) for item in obs_ids])
    }

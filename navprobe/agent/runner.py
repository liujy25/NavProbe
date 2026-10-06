from __future__ import annotations

import argparse
from navprobe.config.settings import configuration_for_args
from pathlib import Path
import sys
from typing import Any

from navprobe.perception.detector_config import (
    create_landmark_detector,
    detector_configuration_from_args,
)
from navprobe.env.interface import EnvInterface
from navprobe.env.episode_status import episode_end_reason
from navprobe.mapping.exploration.bev_map import Map1Map2BEVMap
from navprobe.mapping.exploration.manager import ExplorationManager
from navprobe.memory.graph.graph import Graph
from navprobe.llm.client import LLMClient
from navprobe.agent.action_execution import LocalmapActionExecutor
from navprobe.agent.ablation import normalize_ablation_name
from navprobe.agent.ablation import task_state_enabled
from navprobe.agent.ablation import validate_ablation_configuration
from navprobe.agent.refresh import refresh_state
from navprobe.agent.state import NavProbeAgentContext, NavProbeAgentState
from navprobe.agent.tools import (
    begin_agent_step,
    finalize_run,
    release_step_resources,
)
from navprobe.agent.visual_navigation import (
    ensure_current_node_summary_for_visual_policy,
    episodic_retrieval_enabled_for_goal,
)
from navprobe.agent.visual_step_execution import run_visual_waypoint_decision
from navprobe.agent.task_executive import ensure_task_state_memory
from navprobe.config.episode_config import (
    _resolve_reset_kwargs,
    _run_timestamp,
    _sanitize_name,
)
from navprobe.perception.goal import navigation_goal_spec
from navprobe.agent.objnav_prompt import OBJNAV_TASK_CONSTRAINTS
from navprobe.perception.goal import NAVPROBE_LANGUAGE_GOAL_KIND
from navprobe.perception.landmark_detection import (
    LandmarkDetectionController,
    configure_landmark_detection,
)
from navprobe.logging.recording import _write_json
from navprobe.logging.episode_recording import (
    build_run_metadata,
    record_interrupted_step,
    record_step,
    render_step_outputs,
)
from navprobe.system_state import GoalState, SystemState
from navprobe.runtime.cache import RuntimeCache
from navprobe.runtime.panorama_config import PanoramaConfig, localmap_panorama_config


GLOBAL_BEV_AGENT_RADIUS_M = 0.18


def _agent_radius_from_reset_payload(reset_payload: dict[str, Any] | None) -> float | None:
    if reset_payload is None:
        return None
    for key in ("agent_radius_m", "habitat_agent_radius_m", "agent_radius"):
        value = reset_payload.get(key)
        if value is not None:
            return float(value)
    return None


def _global_bev_kwargs(
    global_bev_kwargs: dict[str, Any] | None,
    *,
    reset_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs = dict(global_bev_kwargs or {})
    reset_agent_radius = _agent_radius_from_reset_payload(reset_payload)
    if "agent_radius" not in kwargs and reset_agent_radius is not None:
        kwargs["agent_radius"] = float(reset_agent_radius)
    kwargs.setdefault("agent_radius", float(GLOBAL_BEV_AGENT_RADIUS_M))
    return kwargs


def _validate_directed_reset(
    *,
    reset_kwargs: dict[str, Any],
    reset_payload: dict[str, Any],
) -> None:
    requested_episode_id = reset_kwargs.get("episode_id")
    requested_scene_id = reset_kwargs.get("scene_id")
    requested_dataset_index = reset_kwargs.get("dataset_index")
    if requested_episode_id is not None and int(reset_payload.get("episode_id", -1)) != int(requested_episode_id):
        raise ValueError(
            "Directed reset returned mismatched episode_id: "
            f"requested={requested_episode_id}, loaded={reset_payload.get('episode_id')}"
        )
    if requested_scene_id is not None and str(reset_payload.get("scene_id", "")) != str(requested_scene_id):
        raise ValueError(
            "Directed reset returned mismatched scene_id: "
            f"requested={requested_scene_id!r}, loaded={reset_payload.get('scene_id')!r}"
        )
    if requested_dataset_index is not None and int(reset_payload.get("dataset_index", -1)) != int(requested_dataset_index):
        raise ValueError(
            "Directed reset returned mismatched dataset_index: "
            f"requested={requested_dataset_index}, loaded={reset_payload.get('dataset_index')}"
        )


def _recorder_scene_name(reset_payload: dict[str, Any], recorder_scene_id: str) -> str:
    recorder_scene_name = str(reset_payload.get("scene_name", "") or "")
    if recorder_scene_name == "":
        recorder_scene_name = str(Path(recorder_scene_id).stem)
    if recorder_scene_name == "":
        raise ValueError(
            "NavProbe episode requires non-empty scene_name or scene_id in reset payload: "
            f"{reset_payload!r}"
        )
    return recorder_scene_name


def _prepare_runner_goal_spec(
    *,
    args: argparse.Namespace,
    reset_kwargs: dict[str, Any],
    reset_payload: dict[str, Any],
) -> object | None:
    raw_goal_text = str(reset_kwargs.get("goal") or reset_payload.get("goal", "") or "").strip()
    task_type = str(args.task_type)
    if task_type == "vln":
        if raw_goal_text == "":
            raw_goal_text = str(reset_payload.get("instruction", "") or "").strip()
        return navigation_goal_spec(raw_goal_text)
    if task_type != "objectnav":
        raise ValueError(f"unsupported task type: {task_type!r}")
    if raw_goal_text == "":
        return None
    return navigation_goal_spec(
        f"Find a {raw_goal_text}, stop close to it.",
        task_constraints=OBJNAV_TASK_CONSTRAINTS,
    )


def _initial_floor_height_from_reset(reset_payload: dict[str, Any]) -> float:
    pose = reset_payload.get("pose")
    if isinstance(pose, dict) and pose.get("z") is not None:
        return float(pose["z"])
    return 0.0


def _build_agent_state(
    *,
    args: argparse.Namespace,
    env: EnvInterface,
    global_exploration: ExplorationManager,
    goal_target: str,
    initial_floor_height: float = 0.0,
) -> NavProbeAgentState:
    graph = Graph()
    graph.set_floor_height("floor_0", float(initial_floor_height))
    cache = RuntimeCache()
    config = configuration_for_args(args)
    args.configuration = config
    llm_client = LLMClient(**config.model.client_kwargs())
    llm_client.reset_usage()
    landmark_detector_name = args.landmark_detector
    landmark_detector_threshold = float(args.landmark_detector_threshold)
    landmark_detector = create_landmark_detector(
        landmark_detector_name, threshold=landmark_detector_threshold,
        configuration=config.perception,
    )
    action_executor = LocalmapActionExecutor(
        env=env,
        cache=cache,
        exploration=global_exploration,
    )
    landmark_controller = LandmarkDetectionController(
        detector=landmark_detector,
        configuration=config.perception,
        cache_provider=lambda: cache,
    )
    action_executor.post_execute_handler = landmark_controller.handle_post_action_execute
    action_executor.step_observation_handler = _step_observation_detection_handler(
        cache=cache,
        landmark_controller=landmark_controller,
    )
    action_executor.step_observation_enabled_getter = lambda: False
    system = SystemState(
        goal=GoalState(target=str(goal_target)),
        graph=graph,
        landmark_controller=landmark_controller,
        cache=cache,
        current_floor_id="floor_0",
        current_floor_height=float(initial_floor_height),
    )
    state = NavProbeAgentState(
        system=system,
        llm_client=llm_client,
        action_executor=action_executor,
        waypoint_policy_name=str(args.waypoint_policy),
        ablation_name=normalize_ablation_name(getattr(args, "ablation", "none")),
        max_retrieve_rounds=int(config.values["entity_retrieval"]["max_rounds"]),
        max_execution_preparations=int(config.values["task_executive"]["max_execution_preparations"]),
        global_explorations_by_floor={"floor_0": global_exploration},
        frontier_records_by_floor={"floor_0": {}},
        overlay_records_by_floor={"floor_0": {}},
        fss_config=config.fss,
    )
    return state


def _step_observation_detection_handler(
    *,
    cache: RuntimeCache,
    landmark_controller: LandmarkDetectionController,
):
    def _handler(observation, source: str) -> dict[str, object]:
        record = cache.store_observation(observation)
        payload: dict[str, object] = {
            "obs_id": record.id,
            "rgb_id": f"{record.id}:rgb",
            "depth_id": f"{record.id}:depth",
            "pose": observation.pose.to_dict(),
            "source": str(source),
        }
        if landmark_controller.is_active() and str(source) != "move_intermediate":
            landmark_update = landmark_controller.update_for_obs_id(obs_id=record.id, source=str(source))
            if landmark_update is not None:
                payload["landmark_detection_buffer_update"] = landmark_update
        return payload

    return _handler


def run_navprobe_agent_episode(
    args: argparse.Namespace,
    env: EnvInterface,
    *,
    panorama_config: PanoramaConfig | None = None,
    global_bev_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if panorama_config is None:
        panorama_config = localmap_panorama_config()
    env_local = env
    reset_kwargs = _resolve_reset_kwargs(args)
    reset_payload = env_local.reset(**reset_kwargs)
    _validate_directed_reset(reset_kwargs=reset_kwargs, reset_payload=reset_payload)
    config = configuration_for_args(args)
    args.configuration = config
    configured_bev = dict(config.mapping["bev"])
    configured_bev.update(global_bev_kwargs or {})
    resolved_global_bev_kwargs = _global_bev_kwargs(
        configured_bev,
        reset_payload=reset_payload,
    )

    recorder_episode_id = int(reset_payload.get("episode_id"))
    recorder_scene_id = str(reset_payload.get("scene_id", "") or "")
    recorder_scene_name = _recorder_scene_name(reset_payload, recorder_scene_id)
    raw_goal_text = str(reset_kwargs.get("goal") or reset_payload.get("goal", "") or "").strip()
    run_goal = raw_goal_text if raw_goal_text != "" else "navprobe"
    run_name = (
        f"episode_{int(recorder_episode_id):04d}_{_sanitize_name(recorder_scene_name)}"
        f"_agent_{_run_timestamp()}"
    )
    run_dir = Path(args.record_dir).resolve() / run_name
    steps_dir = run_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=False)

    global_exploration = ExplorationManager(
        bev_map=Map1Map2BEVMap(**resolved_global_bev_kwargs)
    )
    detector_config = detector_configuration_from_args(args)
    state = _build_agent_state(
        args=args,
        env=env_local,
        global_exploration=global_exploration,
        goal_target=run_goal,
        initial_floor_height=_initial_floor_height_from_reset(reset_payload),
    )
    goal_spec = _prepare_runner_goal_spec(
        args=args,
        reset_kwargs=reset_kwargs,
        reset_payload=reset_payload,
    )
    state.system.task_state.task_constraints = str(
        getattr(goal_spec, "task_constraints", "")
    )
    validate_ablation_configuration(
        ablation_name=state.ablation_name,
        goal_kind=str(getattr(goal_spec, "goal_kind", "") or ""),
        waypoint_policy_name=state.waypoint_policy_name,
    )
    if goal_spec is not None:
        state.system.goal.target = str(goal_spec.description)
    active_retrieval_enabled = episodic_retrieval_enabled_for_goal(
        goal=goal_spec,
    )
    initialize_task_state_before_refresh = (
        active_retrieval_enabled and task_state_enabled(state.ablation_name)
    )
    landmark_detection = {"enabled": False, "reason": "pending_initial_configuration"}
    run_metadata = build_run_metadata(
        run_name=run_name,
        run_goal=run_goal,
        reset_payload=reset_payload,
        recorder_episode_id=recorder_episode_id,
        recorder_scene_id=recorder_scene_id,
        recorder_scene_name=recorder_scene_name,
        global_exploration=global_exploration,
        panorama_config=panorama_config,
    )
    run_metadata["metadata"]["configuration_sources"] = {
        "algorithm": config.source, "dataset_run": config.dataset_source,
    }
    from navprobe.runners.batch_results import experiment_configuration
    recorded = getattr(args, "experiment_configuration", None)
    run_metadata.update(recorded if recorded is not None else experiment_configuration(args, runner="episode"))
    run_metadata["metadata"]["vln_landmark_detection"] = dict(landmark_detection)
    _write_json(run_dir / "run.json", run_metadata)
    context = NavProbeAgentContext(
        args=args,
        env=env_local,
        panorama_config=panorama_config,
        global_bev_kwargs=resolved_global_bev_kwargs,
        run_dir=run_dir,
        steps_dir=steps_dir,
        reset_payload=reset_payload,
        run_metadata=run_metadata,
        run_goal=run_goal,
        recorder_episode_id=recorder_episode_id,
        recorder_scene_id=recorder_scene_id,
        recorder_scene_name=recorder_scene_name,
        goal_spec=goal_spec,
    )

    step = None
    try:
        for place_step_index in range(int(args.max_place_steps)):
            step = None
            end_reason = episode_end_reason(env_local)
            if end_reason is not None:
                state.finalize_reason = end_reason
                break
            release_step_resources(state)
            step = begin_agent_step(
                context=context,
                state=state,
                place_step_index=place_step_index,
            )
            if place_step_index == 0:
                if initialize_task_state_before_refresh:
                    initialization = ensure_task_state_memory(
                        client=state.llm_client,
                        task_state=state.system.task_state,
                        goal_text=str(goal_spec.navigation_goal_text()),
                        goal_kind=NAVPROBE_LANGUAGE_GOAL_KIND,
                    )
                    state.pending_task_state_initialization = dict(initialization)
                # Configure before observation, inside this step's log/error
                # boundary. Later steps reuse this initialized state.
                landmark_detection = configure_landmark_detection(
                    controller=state.landmark_controller, client=state.llm_client,
                    goal_target=state.system.goal.target,
                    goal_spec=goal_spec,
                    detector_threshold=float(detector_config["landmark"]["threshold"]),
                )
                run_metadata["metadata"]["vln_landmark_detection"] = dict(
                    landmark_detection
                )
                _write_json(run_dir / "run.json", run_metadata)
            refresh_state(context=context, state=state, step=step)
            end_reason = episode_end_reason(env_local)
            if end_reason is None and episodic_retrieval_enabled_for_goal(goal=goal_spec):
                ensure_current_node_summary_for_visual_policy(
                    state=state,
                    step=step,
                )
            render_step_outputs(context=context, state=state, step=step)

            if end_reason is not None:
                # Keep partial observations and the arrival edge in the usual
                # artifacts, without making a decision for an ended episode.
                state.finalize_reason = end_reason
                step.policy_decision = {
                    "source": "environment_termination",
                    "reason": end_reason,
                    "stage": "observe_place",
                }
            else:
                run_visual_waypoint_decision(
                    context=context,
                    state=state,
                    step=step,
                )
            step.policy_decision.setdefault("ablation", str(state.ablation_name))
            record_step(context=context, state=state, step=step)

            if state.system.is_done or state.finalize_reason in {"no_active_candidate", "episode_over", "done"}:
                break
        else:
            state.finalize_reason = "max_place_steps"
    finally:
        active_exception = sys.exc_info()[1]
        if active_exception is not None:
            if step is not None:
                record_interrupted_step(
                    state=state, step=step, error=active_exception,
                    visualize=getattr(context.args, "visualize", False),
                )
            state.finalize_reason = f"exception:{type(active_exception).__name__}"
        try:
            episode_result = finalize_run(context=context, state=state)
        except Exception as finalize_error:
            if active_exception is None:
                raise
            # Report the secondary failure while the original exception keeps
            # propagating to the batch runner and its error record.
            print(
                f"[finalize] failed during {type(active_exception).__name__}: "
                f"{type(finalize_error).__name__}: {finalize_error}"
            )
        finally:
            # Release the executor's observation callbacks.
            state.action_executor.post_execute_handler = None
            state.action_executor.step_observation_handler = None
            state.action_executor.step_observation_enabled_getter = None

    return episode_result

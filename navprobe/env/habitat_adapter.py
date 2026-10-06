"""Local Habitat environment setup, coordinate conversion and waypoint execution.

Habitat is an optional runtime dependency, imported when creating an environment.
"""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from navprobe.env.habitat_agent_config import _get_attr_or_key


NAVPROBE_NAVMESH_SNAP_MAX_HEIGHT_DELTA = 0.5


def configure_habitat_scene_paths(config: Any, scenes_dir: str | Path | None) -> None:
    """Resolve asset paths inside a caller's Habitat read_write context."""
    if scenes_dir is None:
        return
    root = Path(scenes_dir).expanduser().resolve()
    config.habitat.dataset.scenes_dir = str(root)
    config.habitat.simulator.scene_dataset = resolve_scene_dataset_path(
        str(config.habitat.simulator.scene_dataset), root
    )


def resolve_scene_dataset_path(value: str, scenes_dir: str | Path) -> str:
    if value in {"", "default"} or Path(value).is_absolute():
        return value
    relative = str(Path(value))
    for marker in ("data/scene_datasets/", "scene_datasets/"):
        if relative.startswith(marker):
            return str(Path(scenes_dir).expanduser().resolve() / relative[len(marker):])
    # Custom configurations may use paths relative to cwd instead of a scene
    # root. Resolve those only when an actual file exists under the new root.
    candidate = Path(scenes_dir).expanduser().resolve() / relative
    return str(candidate) if candidate.is_file() else value


def create_configured_habitat_env(config: Any, *, dataset: Any = None) -> Any:
    """Use public Habitat APIs, resolving per-episode scene metadata too.

    Habitat Env replaces simulator.scene_dataset with the first episode's
    scene_dataset_config, so fixing just the YAML path is insufficient.
    Only in-memory path fields are changed; source datasets remain untouched.
    """
    import habitat
    from navprobe.runtime.randomness import seed_local_randomness
    from navprobe.env.habitat_env import close_environment

    seed = int(config.habitat.seed)
    seed_local_randomness(seed)
    with habitat.config.read_write(config):
        config.habitat.simulator.seed = seed

    if dataset is None:
        dataset = habitat.make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    for episode in dataset.episodes:
        value = getattr(episode, "scene_dataset_config", "default")
        if value:
            episode.scene_dataset_config = resolve_scene_dataset_path(
                value, config.habitat.dataset.scenes_dir
            )
    env = habitat.Env(config=config, dataset=dataset)
    try:
        # Config alone does not seed Habitat's task, simulator or Numba RNGs.
        env.seed(seed)
    except BaseException:
        close_environment(env, label="Habitat initialization")
        raise
    return env


def make_habitat_episode_dataset(
    config: Any, payload: dict[str, Any], *, original_episode_id: str | None = None,
) -> Any:
    """Parse one selected episode without reloading the source split or scene.

    Parsing creates fresh native objects, so simulator caches cannot mutate the
    batch's source records. ObjectNav's loader renumbers episodes within each
    input file; retain its original scene-local ID when passing a single record.
    """
    import habitat

    if len(payload["episodes"]) != 1:
        raise ValueError("A local episode environment requires exactly one selected episode")
    if config.habitat.dataset.type == "R2RVLN-v1":
        from navprobe.env.vlnce_dataset import habitat_compatible_vlnce_payload
        payload = habitat_compatible_vlnce_payload(payload)
    dataset = habitat.make_dataset(config.habitat.dataset.type)
    dataset.from_json(json.dumps(payload), scenes_dir=str(config.habitat.dataset.scenes_dir))
    dataset.episodes = list(filter(dataset.build_content_scenes_filter(config.habitat.dataset), dataset.episodes))
    if len(dataset.episodes) != 1:
        raise ValueError("The selected episode is excluded by Habitat content_scenes")
    if original_episode_id is not None:
        dataset.episodes[0].episode_id = str(original_episode_id)
    return dataset


def _quat_xyzw(rotation: Any) -> np.ndarray:
    if rotation is None:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if all(hasattr(rotation, name) for name in ("x", "y", "z", "w")):
        return np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float64)
    values = np.asarray(rotation, dtype=np.float64).reshape(-1)
    if len(values) != 4:
        raise ValueError(f"rotation must contain four quaternion values, got {rotation!r}")
    return values


class NavProbeHabitatNavigationController:
    """Execute local goals with Habitat ShortestPathFollower."""

    def __init__(
        self,
        env: Any,
        *,
        stop_radius: float = 0.3,
        max_steps: int = 100,
    ) -> None:
        self.env = env
        self.stop_radius = float(stop_radius)
        self.max_steps = int(max_steps)
        self._stop_action_id = 0
        from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

        self.shortest_path_follower = ShortestPathFollower(
            env.sim,
            self.stop_radius,
            return_one_hot=False,
            stop_on_error=True,
        )

    def navigate_to_pose(
        self,
        goal_position: np.ndarray,
        *,
        max_steps: int | None = None,
        step_callback: Callable[..., Any] | None = None,
    ) -> tuple[bool, list[Any], dict[str, Any]]:
        max_steps = self.max_steps if max_steps is None else int(max_steps)
        goal = np.asarray(goal_position, dtype=np.float32).reshape(3)
        observations: list[Any] = []
        actions: list[int] = []
        interrupted = False
        interrupt_reason = None

        def distance() -> float:
            position = np.asarray(self.env.sim.get_agent_state().position, dtype=np.float32)
            return float(np.linalg.norm((goal - position)[[0, 2]]))

        final_distance = distance()
        if final_distance < self.stop_radius:
            return True, observations, {
                "num_steps": 0,
                "actions": [],
                "final_distance": final_distance,
                "success": True,
                "interrupted": False,
                "interrupt_reason": None,
                "step_budget": int(max_steps),
                "mode": "planner",
            }

        if max_steps <= 0:
            return False, observations, {
                "num_steps": 0,
                "actions": [],
                "final_distance": final_distance,
                "success": False,
                "interrupted": True,
                "interrupt_reason": "step_budget_exhausted",
                "step_budget": int(max_steps),
                "mode": "planner",
            }

        success = False
        for _ in range(int(max_steps)):
            action = self.shortest_path_follower.get_next_action(goal)
            if action is None:
                final_distance = distance()
                success = final_distance < self.stop_radius
                interrupt_reason = None if success else "planner_no_action"
                break
            action_id = int(action)
            if action_id == self._stop_action_id:
                final_distance = distance()
                success = final_distance < self.stop_radius
                break
            obs = self.env.step(action_id)
            observations.append(obs)
            actions.append(action_id)
            final_distance = distance()
            # Capture while the simulator still has this frame's pose. The
            # arrival frame can become intermediate when yaw alignment follows.
            callback_interrupt = step_callback is not None and bool(
                step_callback(obs, len(actions), final_distance, 0.0, action_id)
            )
            if final_distance < self.stop_radius:
                success = True
                break
            if callback_interrupt:
                interrupted = True
                interrupt_reason = "step_callback_interrupt"
                break
            if bool(getattr(self.env, "episode_over", False)):
                interrupted = True
                interrupt_reason = "episode_over"
                break
        else:
            interrupted = True
            interrupt_reason = "max_steps_reached"

        return success, observations, {
            "num_steps": len(actions),
            "actions": actions,
            "final_distance": final_distance,
            "success": bool(success),
            "interrupted": bool(interrupted),
            "interrupt_reason": interrupt_reason,
            "step_budget": int(max_steps),
            "mode": "planner",
        }


class NavProbeHabitatAdapter:
    """Adapt local Habitat observations and goal execution for NavProbe.

    Convert goal poses to Habitat coordinates and execute movement through
    ShortestPathFollower and ``env.step``.
    """

    def __init__(
        self,
        *,
        env: Any,
        goal_text: str | None = None,
    ) -> None:
        self.env = env
        self.goal_text = goal_text
        self.pointnav_step_total = 0
        action_name_to_id = {name: index for index, name in enumerate(env.task.actions)}
        self.nav_action_ids = {
            name: action_name_to_id[name]
            for name in ("stop", "move_forward", "turn_left", "turn_right")
        }
        self.nav_controller = NavProbeHabitatNavigationController(
            env,
            stop_radius=0.3,
            # Each local goal has a budget of 100 actions.
            # Habitat enforces the episode-wide limit from its configuration.
            max_steps=100,
        )
        depth_sensor = next(
            (sensor for sensor in env.sim.sensor_suite.sensors.values()
             if sensor.sensor_type.name == "DEPTH"),
            None,
        )
        if depth_sensor is None:
            raise ValueError("NavProbe requires a Habitat depth sensor")
        self._depth_sensor_uuid = depth_sensor.uuid
        self.intrinsic = self._compute_camera_intrinsic(depth_sensor.config)
        self.ground_height_offset: float | None = None

    @staticmethod
    def _compute_camera_intrinsic(sensor: Any) -> np.ndarray:
        width, height, hfov = float(sensor.width), float(sensor.height), float(sensor.hfov)
        focal = (width / 2.0) / np.tan(np.radians(hfov / 2.0))
        return np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]], dtype=np.float32)

    def _get_transform_matrices(self, _obs: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
        from scipy.spatial.transform import Rotation

        state = self.env.sim.get_agent_state()
        position = np.asarray(state.position, dtype=np.float64).reshape(3)
        agent_rotation = Rotation.from_quat(_quat_xyzw(state.rotation))
        ground_offset = float(self.ground_height_offset or 0.0)
        position_adjusted = position.copy()
        position_adjusted[1] -= ground_offset

        habitat_to_ros = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)
        rotation_ros = habitat_to_ros @ agent_rotation.as_matrix() @ habitat_to_ros.T
        position_ros = habitat_to_ros @ position_adjusted
        t_odom_base = np.eye(4, dtype=np.float32)
        t_odom_base[:3, :3] = rotation_ros.astype(np.float32)
        t_odom_base[:3, 3] = position_ros.astype(np.float32)

        # Habitat's runtime sensor UUID differs from its YAML entry name.
        sensor_state = state.sensor_states[self._depth_sensor_uuid]
        cam_position = np.array(sensor_state.position, dtype=np.float64).reshape(3)
        cam_rotation = Rotation.from_quat(_quat_xyzw(sensor_state.rotation)).as_matrix()

        cam_position[1] -= ground_offset
        cam_position_ros = habitat_to_ros @ cam_position
        cam_rotation_ros = habitat_to_ros @ cam_rotation @ habitat_to_ros.T
        t_odom_cam = np.eye(4, dtype=np.float64)
        t_odom_cam[:3, :3] = cam_rotation_ros
        t_odom_cam[:3, 3] = cam_position_ros
        ros_to_optical = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
        t_odom_cam_optical = t_odom_cam @ ros_to_optical.T
        return np.linalg.inv(t_odom_cam_optical).astype(np.float32), t_odom_base

    def _extract_images(self, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        rgb = next((value for key, value in obs.items() if "rgb" in str(key).lower()), None)
        depth = next((value for key, value in obs.items() if "depth" in str(key).lower()), None)
        if rgb is None or depth is None:
            raise ValueError(f"RGB or depth observation is missing; keys={list(obs)}")
        rgb_array = np.asarray(rgb)
        if np.issubdtype(rgb_array.dtype, np.floating) and float(np.nanmax(rgb_array)) <= 1.0:
            rgb_array = (rgb_array * 255.0).astype(np.uint8)
        else:
            rgb_array = rgb_array.astype(np.uint8)
        if rgb_array.ndim == 3 and rgb_array.shape[-1] == 4:
            rgb_array = rgb_array[..., :3]
        depth_array = np.array(depth, dtype=np.float32)
        if depth_array.ndim == 3:
            depth_array = depth_array[..., 0]
        depth_array[~np.isfinite(depth_array)] = 0.0
        return rgb_array, depth_array

    def _project_goal_to_navmesh(self, goal_position_habitat: np.ndarray) -> np.ndarray:
        goal = np.asarray(goal_position_habitat, dtype=np.float32).reshape(3)
        pathfinder = getattr(self.env.sim, "pathfinder", None)
        if pathfinder is None or not bool(getattr(pathfinder, "is_loaded", False)):
            return goal
        agent_position = np.asarray(self.env.sim.get_agent_state().position, dtype=np.float32).reshape(3)
        if bool(pathfinder.is_navigable(goal, max_y_delta=NAVPROBE_NAVMESH_SNAP_MAX_HEIGHT_DELTA)):
            return goal
        island = int(pathfinder.get_island(agent_position))
        if island < 0:
            return goal
        snapped = np.asarray(pathfinder.snap_point(goal, island_index=island), dtype=np.float32).reshape(3)
        if not np.all(np.isfinite(snapped)) or abs(float(snapped[1] - agent_position[1])) > NAVPROBE_NAVMESH_SNAP_MAX_HEIGHT_DELTA:
            return goal
        return snapped

    def execute_goal_pose(
        self,
        goal_pose_dict: dict[str, Any],
        max_steps: int | None,
        step_callback: Callable[..., Any] | None = None,
    ) -> tuple[bool, list[Any], dict[str, Any]]:
        pose = goal_pose_dict["pose"]
        ros_position = np.asarray([pose["position"][key] for key in ("x", "y", "z")], dtype=np.float32)
        ros_to_habitat = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float32)
        goal_position = ros_to_habitat @ ros_position
        goal_position[1] += float(self.ground_height_offset or 0.0)
        goal_position = self._project_goal_to_navmesh(goal_position)
        success, observations, nav_info = self.nav_controller.navigate_to_pose(
            goal_position, max_steps=max_steps, step_callback=step_callback
        )

        align_success = True
        align_observations: list[Any] = []
        align_info: dict[str, Any] = {"num_steps": 0, "actions": [], "success": True, "mode": "orientation_align_skipped"}
        if pose.get("orientation") is not None and not self.env.episode_over:
            budget = self.nav_controller.max_steps if max_steps is None else int(max_steps)
            remaining = max(0, budget - int(nav_info["num_steps"]))
            # Zero remaining actions still permits checking an already-correct
            # heading; the aligner performs no turn when max_turn_steps is zero.
            align_success, align_observations, align_info = self.align_to_goal_orientation(
                goal_pose_dict, step_callback=step_callback, max_turn_steps=remaining
            )
        success = bool(success and align_success)
        merged = {
            "num_steps": int(nav_info["num_steps"]) + int(align_info["num_steps"]),
            "actions": list(nav_info["actions"]) + list(align_info["actions"]),
            "final_distance": float(nav_info["final_distance"]),
            "success": success,
            "interrupted": bool(nav_info["interrupted"]),
            "interrupt_reason": nav_info["interrupt_reason"],
            "step_budget": self.nav_controller.max_steps if max_steps is None else int(max_steps),
            "mode": f"{nav_info['mode']}_with_align",
            "position_nav_info": nav_info,
            "orientation_align_info": align_info,
        }
        return success, list(observations) + list(align_observations), merged

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _quat_dict_to_yaw(q: dict[str, Any]) -> float:
        return math.atan2(2.0 * (q["w"] * q["z"] + q["x"] * q["y"]), 1.0 - 2.0 * (q["y"] ** 2 + q["z"] ** 2))

    def _get_current_yaw_ros(self) -> float:
        _camera, base = self._get_transform_matrices({})
        return float(math.atan2(base[1, 0], base[0, 0]))

    def align_to_goal_orientation(
        self,
        goal_pose_dict: dict[str, Any],
        *,
        step_callback: Callable[..., Any] | None = None,
        max_turn_steps: int = 24,
    ) -> tuple[bool, list[Any], dict[str, Any]]:
        orientation = goal_pose_dict.get("pose", {}).get("orientation")
        if orientation is None or self.env.episode_over:
            return True, [], {"num_steps": 0, "actions": [], "success": True, "mode": "orientation_align_skipped"}
        target_yaw = self._quat_dict_to_yaw(orientation)
        turn_step = math.radians(float(_get_attr_or_key(
            _get_attr_or_key(self.env.sim, "habitat_config"), "turn_angle", 30.0,
        )))
        observations: list[Any] = []
        actions: list[int] = []
        for index in range(int(max_turn_steps)):
            if self.env.episode_over:
                break
            difference = self._wrap_angle(target_yaw - self._get_current_yaw_ros())
            if abs(difference) <= turn_step / 2.0:
                break
            action_name = "turn_left" if difference > 0 else "turn_right"
            action_id = self.nav_action_ids[action_name]
            obs = self.env.step(action_id)
            observations.append(obs)
            actions.append(action_id)
            if step_callback is not None:
                step_callback(obs, index, action_id)
        final_difference = abs(self._wrap_angle(target_yaw - self._get_current_yaw_ros()))
        return final_difference <= turn_step / 2.0, observations, {
            "num_steps": len(actions),
            "actions": actions,
            "success": final_difference <= turn_step / 2.0,
            "mode": "orientation_align",
        }

    def execute_discrete_action(self, action_type: str) -> tuple[bool, list[Any], dict[str, Any]]:
        if action_type not in self.nav_action_ids:
            raise ValueError(f"Unsupported direct action_type: {action_type}")
        action_id = int(self.nav_action_ids[action_type])
        obs = self.env.step(action_id)
        return True, [obs], {
            "num_steps": 1,
            "actions": [action_id],
            "final_distance": -1.0,
            "success": True,
            "interrupted": False,
            "interrupt_reason": None,
            "step_budget": 1,
            "mode": "direct_action",
            "action_type": action_type,
        }


def create_habitat_env(
    config_path: str | Path,
    scene_path: str | Path | None = None,
    scenes_dir: str | Path | None = None,
    *,
    seed: int,
    dataset_type: str | None = None,
    episode_payload: dict[str, Any] | None = None,
    original_episode_id: str | None = None,
) -> Any:
    """Create a Habitat environment using the installed Habitat-Lab package."""
    try:
        import habitat
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Habitat-Lab is required for this entry point. Install habitat-lab and habitat-sim, "
            "then rerun the NavProbe command."
        ) from exc

    config = habitat.get_config(str(Path(config_path).expanduser()))
    with habitat.config.read_write(config):
        config.habitat.seed = seed
        if dataset_type is not None:
            config.habitat.dataset.type = dataset_type
        if scene_path is not None:
            config.habitat.dataset.data_path = str(Path(scene_path).expanduser())
        configure_habitat_scene_paths(config, scenes_dir)
        iterator_options = getattr(config.habitat.environment, "iterator_options", None)
        if iterator_options is not None and hasattr(iterator_options, "shuffle"):
            iterator_options.shuffle = True
        if config.habitat.dataset.type == "R2RVLN-v1":
            from navprobe.env.vlnce_dataset import habitat_compatible_vlnce_dataset_file
            from navprobe.env.vlnce_metrics import configure_vlnce_metrics

            source = Path(config.habitat.dataset.data_path.format(split=config.habitat.dataset.split))
            configure_vlnce_metrics(config, dataset_file=source)
            if episode_payload is None:
                config.habitat.dataset.data_path = str(habitat_compatible_vlnce_dataset_file(source))
    dataset = None if episode_payload is None else make_habitat_episode_dataset(
        config, episode_payload, original_episode_id=original_episode_id,
    )
    return create_configured_habitat_env(config, dataset=dataset)

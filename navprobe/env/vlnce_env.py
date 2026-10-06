from __future__ import annotations

from pathlib import Path
from typing import Any

from navprobe.env.habitat_agent_config import habitat_agent_radius_m
from navprobe.env.habitat_env import NavProbeHabitatEnv, close_environment
from navprobe.env.habitat_utils import _debug_sequence
from navprobe.env.habitat_utils import _position_matches
from navprobe.env.habitat_utils import _rotation_matches
from navprobe.env.interface import RawObservation
from navprobe.env.vlnce_dataset import DEFAULT_VLNCE_SCENES_DIR
from navprobe.env.vlnce_dataset import VLNCEEpisodeSelection
from navprobe.env.vlnce_metrics import add_vln_metric_aliases
from navprobe.env.vlnce_metrics import configure_vlnce_metrics
from navprobe.env.habitat_adapter import configure_habitat_scene_paths
from navprobe.env.habitat_adapter import create_configured_habitat_env
from navprobe.env.habitat_adapter import make_habitat_episode_dataset


def _scene_id_matches(loaded_scene_id: Any, selected_scene_id: Any) -> bool:
    loaded = str(loaded_scene_id or "")
    selected = str(selected_scene_id or "")
    if loaded == selected:
        return True
    if loaded.endswith(selected) or selected.endswith(loaded):
        return True
    return Path(loaded).stem == Path(selected).stem


def _episode_matches_selection(episode: Any, selection: VLNCEEpisodeSelection) -> bool:
    selected_episode = selection.episode
    if not _scene_id_matches(getattr(episode, "scene_id", None), selection.scene_id):
        return False
    if str(getattr(episode, "episode_id", "")) != str(selection.episode_id):
        return False
    return (
        _position_matches(
            getattr(episode, "start_position", None),
            selected_episode.get("start_position"),
        )
        and _rotation_matches(
            getattr(episode, "start_rotation", None),
            selected_episode.get("start_rotation"),
        )
    )


def _episode_debug_payload(episode: Any, index: int) -> dict[str, Any]:
    return {
        "dataset_position": int(index),
        "episode_id": str(getattr(episode, "episode_id", "")),
        "scene_id": str(getattr(episode, "scene_id", "")),
        "start_position": _debug_sequence(getattr(episode, "start_position", None)),
        "start_rotation": _debug_sequence(getattr(episode, "start_rotation", None)),
    }


def _find_habitat_episode_by_selection(episodes: list[Any], selection: VLNCEEpisodeSelection) -> Any:
    matches = [
        (index, episode)
        for index, episode in enumerate(episodes)
        if _episode_matches_selection(episode, selection)
    ]
    if len(matches) == 1:
        return matches[0][1]

    id_matches = [
        _episode_debug_payload(episode, index)
        for index, episode in enumerate(episodes)
        if str(getattr(episode, "episode_id", "")) == str(selection.episode_id)
    ]
    raise ValueError(
        "Could not uniquely bind VLNCE selection to a Habitat episode: "
        f"split={selection.split!r}, episode_id={selection.episode_id!r}, "
        f"scene_id={selection.scene_id!r}, strict_match_count={len(matches)}, "
        f"id_match_count={len(id_matches)}, id_matches={id_matches[:5]}"
    )


def _validate_habitat_episode_matches_selection(episode: Any, selection: VLNCEEpisodeSelection) -> None:
    if not _episode_matches_selection(episode, selection):
        raise ValueError(
            "Loaded Habitat episode does not match VLNCE selection: "
            f"expected_episode_id={selection.episode_id!r}, "
            f"expected_scene_id={selection.scene_id!r}, "
            f"expected_start_position={selection.episode.get('start_position')!r}, "
            f"expected_start_rotation={selection.episode.get('start_rotation')!r}, "
            f"loaded_episode_id={getattr(episode, 'episode_id', None)!r}, "
            f"loaded_scene_id={getattr(episode, 'scene_id', None)!r}, "
            f"loaded_start_position={getattr(episode, 'start_position', None)!r}, "
            f"loaded_start_rotation={getattr(episode, 'start_rotation', None)!r}"
        )


def _episode_id_payload(episode_id: str) -> int | str:
    text = str(episode_id)
    return int(text) if text.isdigit() else text


def _create_vlnce_habitat_env(
    *,
    seed: int,
    config_path: Path,
    selection: VLNCEEpisodeSelection,
    scenes_dir: Path,
    dataset_type: str = "R2RVLN-v1",
):
    import habitat

    config = habitat.get_config(str(config_path))
    with habitat.config.read_write(config):
        config.habitat.seed = seed
        config.habitat.dataset.type = dataset_type
        config.habitat.dataset.data_path = str(selection.dataset_file)
        config.habitat.dataset.scenes_dir = str(scenes_dir)
        config.habitat.dataset.split = str(selection.split)
        configure_habitat_scene_paths(config, scenes_dir)
        configure_vlnce_metrics(config, dataset_file=selection.dataset_file)
        iterator_options = getattr(config.habitat.environment, "iterator_options", None)
        if iterator_options is not None and hasattr(iterator_options, "shuffle"):
            iterator_options.shuffle = False
    dataset = make_habitat_episode_dataset(
        config, {**selection.dataset_metadata, "episodes": [selection.episode]},
    )
    _find_habitat_episode_by_selection(dataset.episodes, selection)
    return create_configured_habitat_env(config, dataset=dataset)


class VLNCEEnv(NavProbeHabitatEnv):
    def __init__(
        self,
        *,
        seed: int,
        selection: VLNCEEpisodeSelection,
        habitat_config: str | Path,
        dataset_type: str = "R2RVLN-v1",
        scenes_dir: str | Path = DEFAULT_VLNCE_SCENES_DIR,
    ) -> None:
        self.selection = selection
        self.habitat_config = Path(habitat_config).expanduser().resolve()
        self.scenes_dir = Path(scenes_dir).expanduser().resolve()
        if not self.scenes_dir.is_dir():
            raise FileNotFoundError(f"VLNCE scenes_dir not found: {self.scenes_dir}")
        self.latest_obs: dict[str, Any] | None = None
        self._last_move_intermediate_observations: list[RawObservation] = []
        self._last_turn_observation: RawObservation | None = None

        from navprobe.env.habitat_adapter import NavProbeHabitatAdapter

        self.env = _create_vlnce_habitat_env(
            seed=seed,
            config_path=self.habitat_config, dataset_type=dataset_type,
            selection=self.selection,
            scenes_dir=self.scenes_dir,
        )
        try:
            self.target_episode = _find_habitat_episode_by_selection(
                list(self.env._dataset.episodes),
                self.selection,
            )
            self.client = NavProbeHabitatAdapter(
                env=self.env,
                goal_text=self.selection.instruction,
            )
        except BaseException:
            close_environment(self, label=type(self).__name__)
            raise

    def _reset_to_target_episode(self) -> dict[str, Any]:
        target_episode_id = getattr(self.target_episode, "episode_id", None)
        target_scene_id = getattr(self.target_episode, "scene_id", None)
        self.env.current_episode = self.target_episode
        self.env.episode_iterator = iter([self.target_episode])
        obs = self.env.reset()
        loaded_episode_id = getattr(self.env.current_episode, "episode_id", None)
        loaded_scene_id = getattr(self.env.current_episode, "scene_id", None)
        if (loaded_episode_id, loaded_scene_id) != (target_episode_id, target_scene_id):
            raise ValueError(
                "Episode mismatch after VLNCE reset: "
                f"requested_episode_id={target_episode_id}, requested_scene_id={target_scene_id}, "
                f"loaded_episode_id={loaded_episode_id}, loaded_scene_id={loaded_scene_id}"
            )
        _validate_habitat_episode_matches_selection(self.env.current_episode, self.selection)
        return obs

    def _metrics_payload(self) -> dict[str, Any]:
        return add_vln_metric_aliases(super()._metrics_payload())

    def _episode_metadata_payload(self) -> dict[str, Any]:
        habitat_episode = self.env.current_episode
        agent_radius_m = habitat_agent_radius_m(self.env)
        if agent_radius_m is None:
            raise ValueError("Could not read Habitat agent radius from VLNCE env config")
        return {
            "dataset_index": int(self.selection.dataset_index),
            "episode_id": _episode_id_payload(self.selection.episode_id),
            "agent_radius_m": float(agent_radius_m),
            "scene_episode_id": str(self.selection.episode_id),
            "source_episode_id": str(self.selection.episode_id),
            "trajectory_id": str(self.selection.trajectory_id),
            "scene_id": str(self.selection.scene_id),
            "scene_name": str(self.selection.scene_key),
            "scene_key": str(self.selection.scene_key),
            "habitat_scene_id": str(getattr(habitat_episode, "scene_id", "") or ""),
            "split": str(self.selection.split),
            "dataset_file": str(self.selection.dataset_file),
            "scenes_dir": str(self.scenes_dir),
            "goal": str(self.client.goal_text or self.selection.instruction),
            "instruction": str(self.client.goal_text or self.selection.instruction),
            "goals": [dict(goal) for goal in self.selection.goals],
            "reference_path": list(self.selection.reference_path),
            "start_position": list(self.selection.episode.get("start_position", [])),
            "start_rotation": list(self.selection.episode.get("start_rotation", [])),
            "ground_height_offset": None if self.client.ground_height_offset is None else float(
                self.client.ground_height_offset
            ),
        }

    def reset(
        self,
        goal: str | None = None,
        episode_id: int | None = None,
        scene_id: str | None = None,
        scene_name: str | None = None,
        dataset_index: int | None = None,
    ) -> dict[str, Any]:
        del episode_id, scene_id, scene_name, dataset_index
        self._last_move_intermediate_observations = []
        self._last_turn_observation = None
        self.client.goal_text = str(goal).strip() if goal is not None else self.selection.instruction
        obs = self._reset_to_target_episode()
        self.latest_obs = obs
        # Use the reset pose as the height origin without moving the agent.
        self.client.ground_height_offset = float(self.env.sim.get_agent_state().position[1])
        self.client.pointnav_step_total = 0
        observation = self._raw_observation_from_habitat_obs(obs)
        payload = self._episode_metadata_payload()
        payload.update(
            {
                "pose": observation.pose.to_dict(),
                "episode_over": bool(self.env.episode_over),
                "max_episode_steps": self.max_episode_steps,
                "pointnav_step_total": int(self.client.pointnav_step_total),
            }
        )
        return payload

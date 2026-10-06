from __future__ import annotations

from pathlib import Path
from typing import Any

from navprobe.env.habitat_agent_config import habitat_agent_radius_m
from navprobe.env.habitat_env import NavProbeHabitatEnv, close_environment
from navprobe.env.habitat_utils import _debug_sequence, _position_matches, _rotation_matches
from navprobe.env.hm3dv2_dataset import HM3Dv2EpisodeSelection
from navprobe.env.interface import RawObservation


def _scene_id_matches(loaded_scene_id: Any, selected_scene_id: Any) -> bool:
    loaded = str(loaded_scene_id or "")
    selected = str(selected_scene_id or "")
    return loaded == selected or loaded.endswith(selected)


def _episode_matches_selection(episode: Any, selection: HM3Dv2EpisodeSelection) -> bool:
    selected_episode = selection.episode
    selected_scene_id = selected_episode.get("scene_id")
    if selected_scene_id is not None and not _scene_id_matches(
        getattr(episode, "scene_id", None),
        selected_scene_id,
    ):
        return False
    if str(getattr(episode, "episode_id", "")) != str(selection.episode_index):
        return False
    if str(getattr(episode, "object_category", "") or "") != str(selection.goal):
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
        "object_category": str(getattr(episode, "object_category", "") or ""),
        "start_position": _debug_sequence(getattr(episode, "start_position", None)),
        "start_rotation": _debug_sequence(getattr(episode, "start_rotation", None)),
    }


def _find_habitat_episode_by_selection(episodes: list[Any], selection: HM3Dv2EpisodeSelection) -> Any:
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
        if str(getattr(episode, "episode_id", "")) == str(selection.episode_index)
    ]
    raise ValueError(
        "Could not uniquely bind HM3Dv2 selection to a Habitat episode: "
        f"scene_key={selection.scene_key!r}, episode_index={selection.episode_index}, "
        f"source_episode_id={selection.source_episode_id!r}, goal={selection.goal!r}, "
        f"strict_match_count={len(matches)}, id_match_count={len(id_matches)}, "
        f"id_matches={id_matches[:5]}"
    )


def _validate_habitat_episode_matches_selection(episode: Any, selection: HM3Dv2EpisodeSelection) -> None:
    if not _episode_matches_selection(episode, selection):
        raise ValueError(
            "Loaded Habitat episode does not match HM3Dv2 selection: "
            f"expected_episode_index={selection.episode_index}, "
            f"expected_source_episode_id={selection.source_episode_id!r}, "
            f"expected_goal={selection.goal!r}, "
            f"expected_scene_id={selection.episode.get('scene_id')!r}, "
            f"expected_start_position={selection.episode.get('start_position')!r}, "
            f"expected_start_rotation={selection.episode.get('start_rotation')!r}, "
            f"loaded_episode_id={getattr(episode, 'episode_id', None)!r}, "
            f"loaded_scene_id={getattr(episode, 'scene_id', None)!r}, "
            f"loaded_object_category={getattr(episode, 'object_category', None)!r}, "
            f"loaded_start_position={getattr(episode, 'start_position', None)!r}, "
            f"loaded_start_rotation={getattr(episode, 'start_rotation', None)!r}"
        )


class HM3Dv2Env(NavProbeHabitatEnv):
    def __init__(
        self,
        *,
        seed: int,
        selection: HM3Dv2EpisodeSelection,
        habitat_config: str | Path,
        dataset_type: str = "ObjectNav-v1",
        scenes_dir: str | Path | None = None,
    ) -> None:
        self.selection = selection
        self.habitat_config = Path(habitat_config).expanduser().resolve()
        self.scenes_dir = (
            self.selection.scene_dir.parents[2] if scenes_dir is None
            else Path(scenes_dir).expanduser().resolve()
        )
        self.latest_obs: dict[str, Any] | None = None
        self._last_move_intermediate_observations: list[RawObservation] = []
        self._last_turn_observation: RawObservation | None = None

        from navprobe.env.habitat_adapter import NavProbeHabitatAdapter
        from navprobe.env.habitat_adapter import create_habitat_env

        self.env = create_habitat_env(
            str(self.habitat_config), dataset_type=dataset_type, seed=seed,
            scene_path=str(self.selection.scene_data_file),
            scenes_dir=self.scenes_dir,
            episode_payload={**self.selection.dataset_metadata, "episodes": [self.selection.episode]},
            original_episode_id=str(self.selection.episode_index),
        )
        try:
            self.target_episode = _find_habitat_episode_by_selection(
                list(self.env._dataset.episodes),
                self.selection,
            )
            self.client = NavProbeHabitatAdapter(
                env=self.env,
                goal_text=self.selection.goal,
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
                "Episode mismatch after HM3Dv2 reset: "
                f"requested_episode_id={target_episode_id}, requested_scene_id={target_scene_id}, "
                f"loaded_episode_id={loaded_episode_id}, loaded_scene_id={loaded_scene_id}"
            )
        _validate_habitat_episode_matches_selection(self.env.current_episode, self.selection)
        return obs

    def _episode_metadata_payload(self) -> dict[str, Any]:
        habitat_episode = self.env.current_episode
        agent_radius_m = habitat_agent_radius_m(self.env)
        if agent_radius_m is None:
            raise ValueError("Could not read Habitat agent radius from HM3Dv2 env config")
        return {
            "dataset_index": int(self.selection.global_episode_id),
            "episode_id": int(self.selection.global_episode_id),
            "agent_radius_m": float(agent_radius_m),
            "global_episode_id": int(self.selection.global_episode_id),
            "local_episode_index": int(self.selection.episode_index),
            "scene_episode_id": self.selection.source_episode_id,
            "source_episode_id": self.selection.source_episode_id,
            "scene_id": self.selection.scene_id,
            "scene_name": self.selection.scene_id,
            "scene_key": self.selection.scene_key,
            "habitat_scene_id": str(getattr(habitat_episode, "scene_id", "") or ""),
            "scene_data_file": str(self.selection.scene_data_file),
            "scene_dir": str(self.selection.scene_dir),
            "goal": str(self.client.goal_text or self.selection.goal),
            "goal_category": str(self.client.goal_text or self.selection.goal),
            "goals": [dict(goal) for goal in self.selection.goals],
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
        _ = episode_id, scene_id, scene_name, dataset_index
        self._last_move_intermediate_observations = []
        self._last_turn_observation = None
        self.client.goal_text = str(goal).strip() if goal is not None else self.selection.goal
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

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_HM3DV2_SCENES_DIR = "data/scene_datasets"


@dataclass(frozen=True)
class HM3Dv2EpisodeSelection:
    scene_key: str
    scene_id: str
    scene_dir: Path
    scene_data_file: Path
    global_episode_id: int
    episode_index: int
    source_episode_id: str
    episode: dict[str, Any]
    goal: str
    goals: list[dict[str, Any]]
    dataset_metadata: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class HM3Dv2EpisodeRequest:
    global_episode_id: int
    scene_key: str
    scene_data_file: Path
    episode_index: int
    source_episode_id: str
    goal: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_name": self.scene_key,
            "scene_key": self.scene_key,
            "scene_data_file": str(self.scene_data_file),
            "global_episode_id": str(self.global_episode_id),
            "episode_id": str(self.global_episode_id),
            "local_episode_index": int(self.episode_index),
            "episode_index": int(self.episode_index),
            "scene_episode_id": str(self.source_episode_id),
            "source_episode_id": str(self.source_episode_id),
            "goal": str(self.goal),
        }


def normalize_scene_name(scene_name: str | None) -> str | None:
    if scene_name is None:
        return None
    normalized = os.path.basename(str(scene_name).strip())
    if normalized == "":
        return None
    for suffix in (".json.gz", ".json", ".basis.glb", ".basis", ".glb"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    if "-" in normalized:
        prefix, remainder = normalized.split("-", 1)
        if prefix.isdigit():
            normalized = remainder
    return normalized


def _read_json(path: Path) -> Any:
    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _scene_data_files(test_data_dir: Path) -> list[Path]:
    if not test_data_dir.is_dir():
        raise FileNotFoundError(f"HM3Dv2 test_data_dir not found: {test_data_dir}")
    files = sorted([*test_data_dir.glob("*.json"), *test_data_dir.glob("*.json.gz")])
    scene_keys: dict[str, Path] = {}
    for path in files:
        scene_key = normalize_scene_name(path.name)
        if scene_key is None:
            continue
        existing = scene_keys.get(scene_key)
        if existing is not None:
            raise ValueError(
                f"Scene {scene_key!r} is ambiguous under {test_data_dir}: "
                f"{existing} and {path}"
            )
        scene_keys[scene_key] = path.resolve()
    return [scene_keys[key] for key in sorted(scene_keys)]


def hm3dv2_data_paths(
    *,
    data_path: str | Path,
    scenes_dir: str | Path | None = None,
    split: str = "val",
) -> tuple[Path, Path]:
    """Resolve the standard Habitat dataset and scene layout."""
    dataset_file = Path(str(data_path).format(split=split)).expanduser().resolve()
    scene_root = Path(scenes_dir or DEFAULT_HM3DV2_SCENES_DIR).expanduser().resolve()
    return dataset_file.parent / "content", scene_root / "hm3d_v0.2"


def _find_scene_data_file(test_data_dir: Path, scene_key: str) -> Path:
    if not test_data_dir.is_dir():
        raise FileNotFoundError(f"HM3Dv2 test_data_dir not found: {test_data_dir}")
    matches = [
        path
        for path in sorted([*test_data_dir.glob("*.json"), *test_data_dir.glob("*.json.gz")])
        if normalize_scene_name(path.name) == scene_key
    ]
    if matches == []:
        raise ValueError(f"Scene {scene_key!r} not found under {test_data_dir}")
    if len(matches) > 1:
        raise ValueError(f"Scene {scene_key!r} is ambiguous under {test_data_dir}: {matches}")
    return matches[0].resolve()


def _find_scene_dir(scene_data_path: Path, scene_key: str) -> tuple[str, Path]:
    split_dirs = [
        scene_data_path / split
        for split in ("train", "val")
        if (scene_data_path / split).is_dir()
    ]
    if split_dirs == []:
        raise FileNotFoundError(f"No train/val scene split directories found under {scene_data_path}")

    matches: list[Path] = []
    for split_dir in split_dirs:
        matches.extend(
            path
            for path in sorted(split_dir.iterdir())
            if path.is_dir() and normalize_scene_name(path.name) == scene_key
        )
    if matches == []:
        raise ValueError(f"Scene {scene_key!r} not found under {scene_data_path}/{{train,val}}")
    if len(matches) > 1:
        raise ValueError(f"Scene {scene_key!r} is ambiguous under {scene_data_path}: {matches}")
    scene_dir = matches[0].resolve()
    return scene_dir.name, scene_dir


def _select_episode_index(
    episodes: list[Any],
    episode_index: int | None,
    episode_id: int | str | None,
    scene_key: str,
) -> int:
    if episode_index is not None and episode_id is not None:
        raise ValueError("Use only one of --episode-index and --episode-id")
    if episode_index is not None:
        selected_index = int(episode_index)
        if not 0 <= selected_index < len(episodes):
            raise IndexError(
                f"Episode index {selected_index} is out of range for {scene_key} "
                f"with {len(episodes)} episodes"
            )
        return selected_index
    if episode_id is not None:
        matched_indices = [
            idx
            for idx, episode in enumerate(episodes)
            if isinstance(episode, dict) and str(episode.get("episode_id")) == str(episode_id)
        ]
        if matched_indices == []:
            raise ValueError(f"Episode id {episode_id} not found in {scene_key}")
        if len(matched_indices) > 1:
            raise ValueError(
                f"Episode id {episode_id} is ambiguous in {scene_key}; "
                f"matching indices: {matched_indices}. Use --episode-index."
            )
        return matched_indices[0]
    raise ValueError("HM3Dv2 local run requires --episode-index or --episode-id")


def _goals_for_category(scene_payload: dict[str, Any], object_category: str) -> list[dict[str, Any]]:
    goals_by_category = scene_payload.get("goals_by_category", {})
    if not isinstance(goals_by_category, dict):
        return []
    for key, value in goals_by_category.items():
        if str(key).split("glb_")[-1] != str(object_category):
            continue
        if not isinstance(value, list):
            raise ValueError(f"goals_by_category[{key!r}] must be a list")
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def list_hm3dv2_episode_requests(
    *,
    scene_names: list[str] | None = None,
    data_path: str | Path,
    scenes_dir: str | Path | None = None,
    split: str = "val",
    episode_payloads: dict[int, dict[str, Any]] | None = None,
    payload_episode_ids: set[str] | None = None,
) -> list[HM3Dv2EpisodeRequest]:
    test_data_dir, _ = hm3dv2_data_paths(
        data_path=data_path, scenes_dir=scenes_dir, split=split,
    )
    requested_scene_keys: set[str] | None = None
    if scene_names is not None and scene_names != []:
        requested_scene_keys = set()
        for scene_name in scene_names:
            scene_key = normalize_scene_name(scene_name)
            if scene_key is None:
                raise ValueError(f"Invalid scene name in batch request: {scene_name!r}")
            requested_scene_keys.add(scene_key)
        for scene_key in sorted(requested_scene_keys):
            _find_scene_data_file(test_data_dir, scene_key)

    requests: list[HM3Dv2EpisodeRequest] = []
    global_episode_id = 0
    for scene_data_file in _scene_data_files(test_data_dir):
        scene_key = normalize_scene_name(scene_data_file.name)
        if scene_key is None:
            raise ValueError(f"Cannot infer scene key from dataset file: {scene_data_file}")
        scene_payload = _read_json(scene_data_file)
        if not isinstance(scene_payload, dict):
            raise ValueError(f"Scene dataset file must contain a JSON object: {scene_data_file}")
        raw_episodes = scene_payload.get("episodes")
        if not isinstance(raw_episodes, list):
            raise ValueError(f"Scene dataset file missing episodes list: {scene_data_file}")
        for episode_index, episode in enumerate(raw_episodes):
            if not isinstance(episode, dict):
                raise ValueError(f"{scene_data_file} episodes[{episode_index}] must be a JSON object")
            goal = str(episode.get("object_category", "") or "").strip()
            if goal == "":
                raise ValueError(f"{scene_data_file} episodes[{episode_index}] missing object_category")
            if requested_scene_keys is None or scene_key in requested_scene_keys:
                requests.append(
                    HM3Dv2EpisodeRequest(
                        global_episode_id=int(global_episode_id),
                        scene_key=scene_key,
                        scene_data_file=scene_data_file.resolve(),
                        episode_index=int(episode_index),
                        source_episode_id=str(episode.get("episode_id", episode_index)),
                        goal=goal,
                    )
                )
                if episode_payloads is not None and (
                    payload_episode_ids is None or str(global_episode_id) in payload_episode_ids
                ):
                    episode_payloads[global_episode_id] = _single_episode_payload(scene_payload, episode)
            global_episode_id += 1
    return requests


def _global_episode_id_for_scene_episode(
    *,
    test_data_dir: Path,
    scene_key: str,
    episode_index: int,
) -> int:
    global_episode_id = 0
    for scene_data_file in _scene_data_files(test_data_dir):
        current_scene_key = normalize_scene_name(scene_data_file.name)
        if current_scene_key is None:
            raise ValueError(f"Cannot infer scene key from dataset file: {scene_data_file}")
        scene_payload = _read_json(scene_data_file)
        if not isinstance(scene_payload, dict):
            raise ValueError(f"Scene dataset file must contain a JSON object: {scene_data_file}")
        raw_episodes = scene_payload.get("episodes")
        if not isinstance(raw_episodes, list):
            raise ValueError(f"Scene dataset file missing episodes list: {scene_data_file}")
        if current_scene_key == scene_key:
            if not 0 <= int(episode_index) < len(raw_episodes):
                raise IndexError(
                    f"Episode index {episode_index} is out of range for {scene_key} "
                    f"with {len(raw_episodes)} episodes"
                )
            return global_episode_id + int(episode_index)
        global_episode_id += len(raw_episodes)
    raise ValueError(f"Scene {scene_key!r} not found under {test_data_dir}")


def _single_episode_payload(scene_payload: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in scene_payload.items() if key != "episodes"}
    payload["episodes"] = [episode]
    if "goals_by_category" in payload:
        # Goal lists are shared by episodes of the same scene/category. Do not
        # retain unrelated categories or copy their potentially large view lists.
        payload["goals_by_category"] = {
            key: value for key, value in payload["goals_by_category"].items()
            if str(key).split("glb_")[-1] == str(episode["object_category"])
        }
    return payload


def resolve_hm3dv2_episode_selection(
    *,
    scene_name: str | None,
    episode_index: int | None,
    episode_id: int | str | None = None,
    data_path: str | Path,
    scenes_dir: str | Path | None = None,
    split: str = "val",
    episode_payload: dict[str, Any] | None = None,
    global_episode_id: int | None = None,
) -> HM3Dv2EpisodeSelection:
    test_data_dir, scene_data_path = hm3dv2_data_paths(
        data_path=data_path, scenes_dir=scenes_dir, split=split,
    )

    scene_key = normalize_scene_name(scene_name)
    if scene_key is None:
        raise ValueError("HM3Dv2 local run requires --scene-name or --scene-id")

    scene_data_file = _find_scene_data_file(test_data_dir, scene_key)
    scene_id, scene_dir = _find_scene_dir(scene_data_path, scene_key)
    scene_payload = _read_json(scene_data_file) if episode_payload is None else episode_payload
    if not isinstance(scene_payload, dict):
        raise ValueError(f"Scene dataset file must contain a JSON object: {scene_data_file}")
    raw_episodes = scene_payload.get("episodes")
    if not isinstance(raw_episodes, list):
        raise ValueError(f"Scene dataset file missing episodes list: {scene_data_file}")

    if episode_payload is None:
        selected_index = _select_episode_index(
            episodes=raw_episodes, episode_index=episode_index, episode_id=episode_id,
            scene_key=scene_key,
        )
        selected_episode = raw_episodes[selected_index]
        global_episode_id = _global_episode_id_for_scene_episode(
            test_data_dir=test_data_dir, scene_key=scene_key, episode_index=selected_index,
        )
    else:
        if episode_index is None or global_episode_id is None or episode_id is not None:
            raise ValueError("A prepared HM3D episode requires its original local and global indices")
        selected_index = int(episode_index)
        selected_episode, = raw_episodes
    if not isinstance(selected_episode, dict):
        raise ValueError(f"episodes[{selected_index}] must be a JSON object")

    goal = str(selected_episode.get("object_category", "") or "").strip()
    if goal == "":
        raise ValueError(f"episodes[{selected_index}] missing object_category")
    source_episode_id = str(selected_episode.get("episode_id", selected_index))
    payload = _single_episode_payload(scene_payload, selected_episode)
    return HM3Dv2EpisodeSelection(
        scene_key=scene_key,
        scene_id=scene_id,
        scene_dir=scene_dir,
        scene_data_file=scene_data_file,
        global_episode_id=global_episode_id,
        episode_index=selected_index,
        source_episode_id=source_episode_id,
        episode=dict(selected_episode),
        goal=goal,
        goals=_goals_for_category(scene_payload, goal),
        dataset_metadata={key: value for key, value in payload.items() if key != "episodes"},
    )

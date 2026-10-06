from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


DEFAULT_VLNCE_SCENES_DIR = "data/scene_datasets"


def habitat_compatible_vlnce_dataset_file(dataset_file: str | Path) -> Path:
    source = Path(dataset_file).expanduser().resolve()
    stat = source.stat()
    cache_key = hashlib.sha256(
        f"{source}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "navprobe_vlnce_datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    compatible_marker = cache_dir / f"{cache_key}.compatible"
    normalized_file = cache_dir / f"{cache_key}.json.gz"
    if compatible_marker.is_file():
        return source
    if normalized_file.is_file():
        return normalized_file

    with gzip.open(source, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    episodes = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(episodes, list):
        raise ValueError(f"VLNCE dataset missing episodes list: {source}")

    instruction_keys = {"instruction_text", "instruction_tokens"}
    requires_normalization = "instruction_vocab" not in payload or any(
        isinstance(episode, dict)
        and isinstance(episode.get("instruction"), dict)
        and not set(episode["instruction"]).issubset(instruction_keys)
        for episode in episodes
    )
    if not requires_normalization:
        compatible_marker.touch()
        return source

    normalized_payload = habitat_compatible_vlnce_payload(payload)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
        with gzip.open(temporary_path, "wt", encoding="utf-8") as handle:
            json.dump(normalized_payload, handle)
        os.replace(temporary_path, normalized_file)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return normalized_file


def habitat_compatible_vlnce_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep source RxR metadata intact; adapt only the input to Habitat's VLN loader."""
    normalized = dict(payload)
    normalized.setdefault("instruction_vocab", {"word_list": []})
    normalized["episodes"] = []
    for episode in payload["episodes"]:
        if not isinstance(episode, dict) or not isinstance(episode.get("instruction"), dict):
            normalized["episodes"].append(episode)
            continue
        instruction = episode["instruction"]
        normalized["episodes"].append({
            **episode,
            "instruction": {key: instruction[key] for key in ("instruction_text", "instruction_tokens")
                            if key in instruction},
        })
    return normalized


@dataclass(frozen=True)
class VLNCEEpisodeInfo:
    episode_id: str
    scene_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": str(self.episode_id),
            "scene_id": str(self.scene_id),
        }


@dataclass(frozen=True)
class VLNCEEpisodeRequest:
    dataset_index: int
    episode_id: str
    scene_id: str
    scene_key: str
    instruction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_index": int(self.dataset_index),
            "episode_id": str(self.episode_id),
            "id": str(self.episode_id),
            "scene_id": str(self.scene_id),
            "scene_key": str(self.scene_key),
            "instruction": str(self.instruction),
            "goal": str(self.instruction),
        }


@dataclass(frozen=True)
class VLNCEEpisodeSelection:
    split: str
    dataset_file: Path
    dataset_index: int
    episode_id: str
    trajectory_id: str
    scene_id: str
    scene_key: str
    instruction: str
    episode: dict[str, Any]
    goals: list[dict[str, Any]]
    reference_path: list[Any]
    dataset_metadata: dict[str, Any] = field(default_factory=dict, repr=False)


def vlnce_dataset_file(*, data_path: str | Path, split: str) -> Path:
    """Resolve the Habitat-style data_path template for the requested split."""
    dataset_path = Path(str(data_path).format(split=split)).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"VLNCE dataset file not found: {dataset_path}")
    return dataset_path


def load_vlnce_episode_infos(path: str | Path) -> list[VLNCEEpisodeInfo]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("VLNCE episode file must contain a JSON list")
    infos: list[VLNCEEpisodeInfo] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"VLNCE episode file item {index} must be an object with id and scene_id")
        episode_id = str(item.get("id", item.get("episode_id", ""))).strip()
        scene_id = str(item.get("scene_id", "")).strip()
        if episode_id == "" or scene_id == "":
            raise ValueError(f"VLNCE episode file item {index} requires non-empty id and scene_id")
        key = (episode_id, scene_id)
        if key in seen:
            raise ValueError(f"Duplicate VLNCE episode info: id={episode_id!r}, scene_id={scene_id!r}")
        seen.add(key)
        infos.append(VLNCEEpisodeInfo(episode_id=episode_id, scene_id=scene_id))
    return infos


def list_vlnce_episode_requests(
    *,
    data_path: str | Path,
    split: str,
    episode_payloads: dict[int, dict[str, Any]] | None = None,
) -> list[VLNCEEpisodeRequest]:
    dataset_file = vlnce_dataset_file(
        data_path=data_path,
        split=split,
    )
    payload = _read_dataset(dataset_file)
    episodes = payload["episodes"]
    metadata = {key: value for key, value in payload.items() if key != "episodes"}
    requests: list[VLNCEEpisodeRequest] = []
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ValueError(f"{dataset_file} episodes[{index}] must be a JSON object")
        instruction = _instruction_text(episode)
        requests.append(
            VLNCEEpisodeRequest(
                dataset_index=int(index),
                episode_id=str(episode.get("episode_id", "")),
                scene_id=str(episode.get("scene_id", "")),
                scene_key=_scene_key(str(episode.get("scene_id", ""))),
                instruction=instruction,
            )
        )
        if episode_payloads is not None:
            episode_payloads[index] = {**metadata, "episodes": [episode]}
    return requests


def filter_vlnce_requests_by_infos(
    requests: list[VLNCEEpisodeRequest],
    infos: list[VLNCEEpisodeInfo],
) -> list[VLNCEEpisodeRequest]:
    by_key: dict[tuple[str, str], VLNCEEpisodeRequest] = {}
    for request in requests:
        by_key[(str(request.episode_id), str(request.scene_id))] = request
    selected: list[VLNCEEpisodeRequest] = []
    missing: list[dict[str, str]] = []
    for info in infos:
        request = by_key.get((str(info.episode_id), str(info.scene_id)))
        if request is None:
            request = _find_scene_suffix_match(requests=requests, info=info)
        if request is None:
            missing.append(info.to_dict())
            continue
        selected.append(request)
    if missing != []:
        raise ValueError(f"VLNCE episode file contains episodes not present in dataset: {missing}")
    return selected


def resolve_vlnce_episode_selection(
    *,
    data_path: str | Path,
    split: str,
    episode_id: str,
    scene_id: str,
    episode_payload: dict[str, Any] | None = None,
    dataset_index: int | None = None,
) -> VLNCEEpisodeSelection:
    dataset_file = vlnce_dataset_file(
        data_path=data_path,
        split=split,
    )
    payload = _read_dataset(dataset_file) if episode_payload is None else episode_payload
    episodes = payload["episodes"]
    matches = [
        (index, episode)
        for index, episode in enumerate(episodes)
        if isinstance(episode, dict)
        and str(episode.get("episode_id", "")) == str(episode_id)
        and _scene_id_matches(str(episode.get("scene_id", "")), str(scene_id))
    ]
    if len(matches) != 1:
        raise ValueError(
            "Could not uniquely resolve VLNCE episode: "
            f"split={split!r}, episode_id={episode_id!r}, scene_id={scene_id!r}, "
            f"match_count={len(matches)}"
        )
    matched_index, episode = matches[0]
    if episode_payload is None:
        dataset_index = matched_index
    elif dataset_index is None:
        raise ValueError("A prepared VLNCE episode requires its original dataset index")
    instruction = _instruction_text(episode)
    return VLNCEEpisodeSelection(
        split=str(split),
        dataset_file=dataset_file,
        dataset_index=int(dataset_index),
        episode_id=str(episode.get("episode_id", "")),
        trajectory_id=str(episode.get("trajectory_id", "")),
        scene_id=str(episode.get("scene_id", "")),
        scene_key=_scene_key(str(episode.get("scene_id", ""))),
        instruction=instruction,
        episode=dict(episode),
        goals=[dict(goal) for goal in list(episode.get("goals", [])) if isinstance(goal, dict)],
        reference_path=list(episode.get("reference_path", [])),
        dataset_metadata={key: value for key, value in payload.items() if key != "episodes"},
    )


def _read_dataset(dataset_file: Path) -> dict[str, Any]:
    with gzip.open(dataset_file, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"VLNCE dataset must contain a JSON object: {dataset_file}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"VLNCE dataset missing episodes list: {dataset_file}")
    return payload


def _instruction_text(episode: dict[str, Any]) -> str:
    instruction = episode.get("instruction")
    if not isinstance(instruction, dict):
        raise ValueError(f"VLNCE episode missing instruction object: {episode!r}")
    text = str(instruction.get("instruction_text", "")).strip()
    if text == "":
        raise ValueError(f"VLNCE episode missing instruction_text: {episode!r}")
    return text


def _scene_key(scene_id: str) -> str:
    text = str(scene_id).strip()
    if text == "":
        raise ValueError("VLNCE scene_id must be non-empty")
    return Path(text).stem


def _scene_id_matches(loaded_scene_id: str, selected_scene_id: str) -> bool:
    loaded = str(loaded_scene_id).strip()
    selected = str(selected_scene_id).strip()
    if loaded == selected:
        return True
    if loaded.endswith(selected) or selected.endswith(loaded):
        return True
    return Path(loaded).stem == Path(selected).stem


def _find_scene_suffix_match(
    *,
    requests: list[VLNCEEpisodeRequest],
    info: VLNCEEpisodeInfo,
) -> VLNCEEpisodeRequest | None:
    matches = [
        request
        for request in requests
        if str(request.episode_id) == str(info.episode_id)
        and _scene_id_matches(str(request.scene_id), str(info.scene_id))
    ]
    if len(matches) == 1:
        return matches[0]
    return None

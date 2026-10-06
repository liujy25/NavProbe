"""Compose disjoint algorithm and dataset-run YAMLs at the runner boundary."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from importlib import resources
import math
import os
from string import Formatter
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from navprobe.agent.ablation import ABLATION_NAMES
from navprobe.agent.vertical_config import VERTICAL_METHOD_NAMES
from navprobe.llm.request_config import NAVPROBE_REASONING_MODULES, REASONING_EFFORT_CHOICES


@dataclass(frozen=True)
class FSSConfig:
    method: str
    sample_spacing_m: float
    min_distance_m: float
    max_distance_m: float
    max_candidates: int
    occlusion_tolerance_m: float
    frontier_anchor_limit: int
    skeleton_anchor_limit: int
    combined_anchor_limit: int
    skeleton_min_clearance_m: float
    node_dedup_radius_m: float
    global_node_dedup_radius_m: float
    stop_sample_spacing_m: float
    stop_max_distance_m: float
    anchor_nms_distance_m: float
    anchor_combine_nms_distance_m: float
    final_merge_distance_m: float
    visible_backtrack_step_m: float

    def candidate_kwargs(self):
        return {name: getattr(self, name) for name in (
            "min_distance_m", "max_candidates", "occlusion_tolerance_m", "frontier_anchor_limit",
            "skeleton_anchor_limit", "combined_anchor_limit", "skeleton_min_clearance_m",
            "anchor_nms_distance_m", "anchor_combine_nms_distance_m",
            "final_merge_distance_m", "visible_backtrack_step_m")}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    base_url: str
    api_key_env: str
    reasoning: dict
    min_completion_tokens: int
    request_timeout_seconds: float
    requests: dict

    def client_kwargs(self):
        api_key = os.environ.get(self.api_key_env, "").strip() if self.api_key_env else ""
        if self.api_key_env and not api_key:
            raise ValueError(f"Set {self.api_key_env} before starting model-backed navigation")
        return {"model": self.name, "base_url": self.base_url,
                "api_key": api_key,
                "reasoning_effort": self.reasoning["default"],
                "reasoning_by_module": self.reasoning["modules"],
                "request_timeout_seconds": self.request_timeout_seconds,
                "min_completion_tokens": self.min_completion_tokens,
                "request_limits": deepcopy(self.requests)}


@dataclass(frozen=True)
class NavProbeConfig:
    values: dict
    fss: FSSConfig
    model: ModelConfig
    source: str
    dataset_source: str = "recorded"

    @property
    def mapping(self):
        return self.values["mapping"]

    @property
    def perception(self):
        return self.values["landmark_perception"]

    def to_dict(self):
        return deepcopy(self.values)


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError(f"configuration keys must be strings at line {key_node.start_mark.line + 1}")
        if key in result:
            raise ValueError(f"duplicate configuration key {key!r} at line {key_node.start_mark.line + 1}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def resource_path(value: str) -> str:
    """Resolve an installed package resource, never relative to a checkout."""
    prefix = "package://navprobe/"
    if not value.startswith(prefix):
        raise ValueError(f"unsupported package resource: {value}")
    relative = value[len(prefix):]
    if ".." in Path(relative).parts or relative.startswith("/"):
        raise ValueError(f"invalid package resource: {value}")
    # Wheels installed by pip are unpacked. Do not write assets into the package.
    path = resources.files("navprobe").joinpath(relative)
    if not path.is_file():
        raise ValueError(f"missing package resource: {value}")
    return str(path)


def _path(value, base):
    if value is None or value == "":
        return value
    if value.startswith("package://"):
        return resource_path(value)
    path = Path(value).expanduser()
    return str((base / path).resolve()) if not path.is_absolute() else str(path.resolve())


# Schema entries express types/ranges only; they are not another set of defaults.
S = (str, None, None, False)
B = (bool, None, None, False)
N = (float, 0, None, False)
POS = (float, 0, None, False, "positive")
I = (int, 1, None, False)
Z = (int, 0, None, False)
RATIO = (float, 0, 1, False)


def enum(*values):
    return (str, None, None, False, values)


SCHEMA = {
    "schema_version": (int, 2, 2, False),
    "seed": (int, 0, 2**32 - 1, False),
    "dataset": {"type": enum("R2RVLN-v1", "ObjectNav-v1"), "data_path": S,
                "scenes_dir": S, "split": S, "episode_file": S},
    "environment": {"habitat_config": S},
    "ablation": enum(*ABLATION_NAMES),
    "task_executive": {"max_place_steps": I, "max_execution_preparations": I}, "entity_retrieval": {"max_rounds": Z},
    "waypoint_sampling": {"method": enum("frontier_skeleton_sample"),
        "sample_spacing_m": POS, "min_distance_m": N, "max_distance_m": POS, "max_candidates": I,
        "occlusion_tolerance_m": N, "frontier_anchor_limit": I, "skeleton_anchor_limit": I,
        "combined_anchor_limit": I, "skeleton_min_clearance_m": N, "node_dedup_radius_m": N,
        "global_node_dedup_radius_m": N, "stop_sample_spacing_m": POS, "stop_max_distance_m": POS,
        "anchor_nms_distance_m": POS, "anchor_combine_nms_distance_m": POS,
        "final_merge_distance_m": POS, "visible_backtrack_step_m": POS},
    "mapping": {"candidate_dedup_radius_m": POS, "bev": {"size": I, "pixels_per_meter": I,
        "min_height": N, "max_height": POS, "min_depth": N, "max_depth": POS,
        "area_thresh": N, "hole_area_thresh": Z, "astar_clearance_radius": N,
        "astar_clearance_weight": N, "depth_is_normalized": B, "virtual_max_range_blocker": B,
        "virtual_max_range_pixel_stride": I, "virtual_max_range_blocker_dilate_px": I}},
    "landmark_perception": {"detector": enum("groundingdino", "yolo_world"),
        "threshold": RATIO, "max_results": I, "same_frame_iou_threshold": RATIO,
        "min_depth_m": N, "merge_distance_m": POS, "max_detections_per_entity": I, "groundingdino": {"model_config_path": S, "model_checkpoint_path": S,
            "device": S, "text_threshold": RATIO}, "yolo_world": {"model_path": S, "device": S}},
    "vertical_navigation": {"method": enum(*VERTICAL_METHOD_NAMES), "floor_match_threshold_m": POS},
    "model": {"name": S, "base_url": S, "api_key_env": S,
        "reasoning": {"default": enum(*REASONING_EFFORT_CHOICES), "modules": {}},
        "min_completion_tokens": Z, "request_timeout_seconds": POS,
        "requests": {name: {"max_tokens": I} for name in (
            "task_executive_initialize", "task_executive_assess", "skill_selector", "waypoint_grounder",
            "entity_knowledge_manager", "place_memory", "landmark_perception")}},
    "output": {"visualize": B, "record_dir": S, "results_dir": S},
    "runtime": {"resume_completed": B, "dry_run": B},
}
RUN_GROUPS = ("dataset", "environment", "output", "runtime")
ALGORITHM_SCHEMA = {key: value for key, value in SCHEMA.items() if key not in RUN_GROUPS}
RUN_SCHEMA = {key: SCHEMA[key] for key in RUN_GROUPS}
DATASETS = ("r2r", "rxr", "hm3dv2")
PATHS = ("dataset.data_path", "dataset.scenes_dir", "dataset.episode_file", "environment.habitat_config",
         "landmark_perception.groundingdino.model_config_path",
         "landmark_perception.groundingdino.model_checkpoint_path", "landmark_perception.yolo_world.model_path",
         "output.record_dir", "output.results_dir")


def _get(data, path):
    for part in path.split("."):
        data = data[part]
    return data


def _set(data, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        data = data[part]
    data[parts[-1]] = value


def _validate(value, schema=SCHEMA, path="configuration"):
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected mapping")
        if path == "configuration.model.reasoning.modules":
            if set(value) - set(NAVPROBE_REASONING_MODULES):
                raise ValueError(f"{path}: unknown module keys {sorted(set(value) - set(NAVPROBE_REASONING_MODULES))}")
            for name, effort in value.items():
                _validate(effort, enum(*REASONING_EFFORT_CHOICES), f"{path}.{name}")
            return
        if set(value) - set(schema):
            raise ValueError(f"{path}: unknown keys {sorted(set(value) - set(schema))}")
        if set(schema) - set(value):
            raise ValueError(f"{path}: missing keys {sorted(set(schema) - set(value))}")
        for key, rule in schema.items():
            _validate(value[key], rule, f"{path}.{key}")
        return
    kind, minimum, maximum, nullable, *extra = schema
    if value is None and nullable:
        return
    valid = type(value) is kind or (kind is float and type(value) is int)
    if not valid:
        raise ValueError(f"{path}: expected {kind.__name__}")
    if kind in (int, float):
        if not math.isfinite(value) or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
            raise ValueError(f"{path}: value outside valid range")
        if extra and extra[0] == "positive" and value <= 0:
            raise ValueError(f"{path}: must be positive")
    if extra and isinstance(extra[0], tuple) and value not in extra[0]:
        raise ValueError(f"{path}: expected one of {extra[0]}")


def validate_config(data):
    _validate(data)
    for group, low, high in ((data["mapping"]["bev"], "min_height", "max_height"),
                             (data["mapping"]["bev"], "min_depth", "max_depth"),
                             (data["waypoint_sampling"], "min_distance_m", "max_distance_m")):
        if group[low] >= group[high]:
            raise ValueError(f"{low} must be less than {high}")
    for key in ("name", "base_url"):
        if not data["model"][key].strip():
            raise ValueError(f"model.{key} must be filled in the algorithm YAML")
    if data["model"]["api_key_env"] and not data["model"]["api_key_env"].strip():
        raise ValueError("model.api_key_env must name a variable or be empty for an unauthenticated service")
    for key in ("data_path", "split", "scenes_dir", "episode_file"):
        if not data["dataset"][key].strip():
            raise ValueError(f"dataset.{key} must be filled in the dataset YAML")
    fields = Formatter().parse(data["dataset"]["data_path"])
    if any(field not in (None, "split") or spec or conversion for _, field, spec, conversion in fields):
        raise ValueError("dataset.data_path supports only the {split} placeholder")
    if data["model"]["base_url"]:
        url = urlsplit(data["model"]["base_url"])
        if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password:
            raise ValueError("model.base_url must be an HTTP(S) endpoint without embedded credentials")


def config_from_values(data, *, source="recorded", dataset_source="recorded"):
    validate_config(data)
    return NavProbeConfig(deepcopy(data), FSSConfig(**data["waypoint_sampling"]),
                          ModelConfig(**data["model"]), source, dataset_source)


def _read_config_file(path, *, resource, schema):
    default_source = Path(resource_path(resource)).resolve()
    source = default_source if path is None else Path(path).expanduser().resolve()
    data = yaml.load(source.read_text(), Loader=_UniqueLoader)
    _validate(data, schema)
    # Explicitly selecting the default resource preserves its launch-directory
    # paths. A custom file owns its relative resources and output directories.
    base = Path.cwd() if source == default_source else source.parent
    for key in PATHS:
        if key.split(".")[0] in data:
            _set(data, key, _path(_get(data, key), base))
    return data, str(source)


def load_config(path=None, *, dataset="r2r", dataset_config=None, overrides=None, runner="vlnce"):
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    if (runner == "vlnce") != (dataset in ("r2r", "rxr")):
        raise ValueError(f"dataset={dataset} does not match {runner} entry")
    algorithm, source = _read_config_file(path,
        resource="package://navprobe/config/navprobe.yaml", schema=ALGORITHM_SCHEMA)
    run, run_source = _read_config_file(dataset_config,
        resource=f"package://navprobe/config/datasets/{dataset}.yaml", schema=RUN_SCHEMA)
    data = {**algorithm, **run}
    for key, value in (overrides or {}).items():
        _set(data, key, _path(value, Path.cwd()) if key in PATHS else value)
    expected_type = "ObjectNav-v1" if dataset == "hm3dv2" else "R2RVLN-v1"
    if data["dataset"]["type"] != expected_type:
        raise ValueError(f"dataset.type must be {expected_type} for {dataset}")
    return config_from_values(data, source=source, dataset_source=run_source)


# Explicit command-line overrides project into the same resolved configuration.
# Model identity always comes from the algorithm YAML; only its key uses the environment.
CLI_FIELDS = {
    "seed": "seed",
    "data_path": "dataset.data_path", "scenes_dir": "dataset.scenes_dir",
    "split": "dataset.split",
    "episode_file": "dataset.episode_file",
    "habitat_config": "environment.habitat_config",
    "ablation": "ablation", "waypoint_policy": "waypoint_sampling.method",
    "candidate_dedup_radius": "mapping.candidate_dedup_radius_m", "max_place_steps": "task_executive.max_place_steps",
    "max_retrieve_rounds": "entity_retrieval.max_rounds", "vertical_method": "vertical_navigation.method",
    "vertical_floor_match_threshold_m": "vertical_navigation.floor_match_threshold_m",
    "landmark_detector": "landmark_perception.detector", "landmark_detector_threshold": "landmark_perception.threshold",
    "reasoning_effort": "model.reasoning.default", "min_completion_tokens": "model.min_completion_tokens",
    "record_dir": "output.record_dir", "results_dir": "output.results_dir", "visualize": "output.visualize",
    "dry_run": "runtime.dry_run",
}


def _module_reasoning_levels(reasoning):
    modules = reasoning["modules"]
    return None if not modules else [
        REASONING_EFFORT_CHOICES.index(modules.get(name, reasoning["default"]))
        for name in NAVPROBE_REASONING_MODULES
    ]


def namespace_from_config(config):
    data = config.values
    values = {name: _get(data, path) for name, path in CLI_FIELDS.items()}
    values.update(configuration=config, task_type="vln" if data["dataset"]["type"] == "R2RVLN-v1" else "objectnav",
                  dataset_type=data["dataset"]["type"],
                  dataset_index=None, scene=None, scene_id=None, goal=None,
                  scene_name=None, episode_index=None, episode_id=None,
                  no_resume=not data["runtime"]["resume_completed"],
                  module_reasoning_efforts=_module_reasoning_levels(data["model"]["reasoning"]))
    return argparse.Namespace(**values)


def configuration_for_args(args):
    """Apply runner argument overrides to the resolved configuration.

    Episode selection/output projections do not replace the batch configuration.
    This adapter is at the runner boundary; algorithm modules never parse flags.
    """
    config = args.configuration
    data = config.to_dict()
    original_levels = _module_reasoning_levels(data["model"]["reasoning"])
    for name, path in CLI_FIELDS.items():
        if path.split(".")[0] in {"dataset", "environment", "output", "runtime"} and path != "output.visualize":
            continue
        if hasattr(args, name):
            _set(data, path, getattr(args, name))
    levels = getattr(args, "module_reasoning_efforts", None)
    # Compare against the source projection, before changing the fallback.
    # Otherwise inherited values become mistaken for explicit module overrides.
    if levels is not None and list(levels) != original_levels:
        data["model"]["reasoning"]["modules"] = dict(zip(NAVPROBE_REASONING_MODULES, (REASONING_EFFORT_CHOICES[v] for v in levels)))
    resolved = config if data == config.values else config_from_values(data, source=config.source, dataset_source=config.dataset_source)
    args.configuration = resolved
    args.module_reasoning_efforts = _module_reasoning_levels(resolved.model.reasoning)
    return resolved


class _ConfigParser(argparse.ArgumentParser):
    def __init__(self, runner):
        super().__init__(description=f"Run NavProbe {runner} with shared algorithm and dataset-run YAMLs.", argument_default=argparse.SUPPRESS)
        self.runner = runner
        self.add_argument("--dataset", choices=DATASETS, default="r2r" if runner == "vlnce" else "hm3dv2")
        self.add_argument("--config", help="Shared algorithm YAML; defaults to the packaged navprobe.yaml")
        self.add_argument("--dataset-config", help="Dataset-run YAML; defaults to the selected dataset preset")
        for name, path in CLI_FIELDS.items():
            rule = _get(SCHEMA, path)
            kind = rule[0]
            kwargs = {"type": kind}
            if kind is bool:
                kwargs = {"action": "store_true"}
            if name == "visualize":
                kwargs = {"action": argparse.BooleanOptionalAction,
                          "help": "Save RGB-D, model input images and display visualizations"}
            if len(rule) > 4 and isinstance(rule[4], tuple):
                kwargs["choices"] = rule[4]
            self.add_argument("--" + name.replace("_", "-"), **kwargs)
        self.add_argument("--no-resume", action="store_true")
        self.add_argument("--module-reasoning-efforts", type=int, choices=range(5), nargs=len(NAVPROBE_REASONING_MODULES))

    def parse_args(self, args=None, namespace=None):
        raw = vars(super().parse_args(args, namespace))
        overrides = {CLI_FIELDS[k]: v for k, v in raw.items() if k in CLI_FIELDS}

        if "no_resume" in raw:
            overrides["runtime.resume_completed"] = not raw["no_resume"]
        if "module_reasoning_efforts" in raw:
            overrides["model.reasoning.modules"] = dict(zip(NAVPROBE_REASONING_MODULES,
                (REASONING_EFFORT_CHOICES[v] for v in raw["module_reasoning_efforts"])))
        try:
            config = load_config(raw.get("config"), dataset=raw["dataset"],
                dataset_config=raw.get("dataset_config"), overrides=overrides, runner=self.runner)
            result = namespace_from_config(config)
            return result
        except (ValueError, OSError, yaml.YAMLError) as exc:
            self.error(str(exc))


def build_runner_parser(runner):
    return _ConfigParser(runner)

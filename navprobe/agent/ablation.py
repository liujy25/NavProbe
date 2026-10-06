from __future__ import annotations

from navprobe.agent.waypoint.types import FRONTIER_SKELETON_SAMPLE_WAYPOINT_POLICY
from navprobe.perception.goal import NAVPROBE_LANGUAGE_GOAL_KIND


NO_ABLATION = "none"
NO_PROGRESS_ABLATION = "no_progress"
PASSIVE_FULL_HISTORY_ABLATION = "passive_full_history"
NO_CONSOLIDATION_ABLATION = "no_consolidation"
NO_CONSOLIDATION_NO_RETRIEVAL_ABLATION = "no_consolidation_no_retrieval"
ABLATION_NAMES = (
    NO_ABLATION,
    NO_PROGRESS_ABLATION,
    PASSIVE_FULL_HISTORY_ABLATION,
    NO_CONSOLIDATION_ABLATION,
    NO_CONSOLIDATION_NO_RETRIEVAL_ABLATION,
)


def normalize_ablation_name(value: object) -> str:
    name = str(value or NO_ABLATION).strip().lower()
    if name not in ABLATION_NAMES:
        raise ValueError(f"unsupported ablation: {value!r}; expected one of {ABLATION_NAMES!r}")
    return name


def task_state_enabled(ablation_name: str) -> bool:
    return normalize_ablation_name(ablation_name) != NO_PROGRESS_ABLATION


def passive_full_history_enabled(ablation_name: str) -> bool:
    return normalize_ablation_name(ablation_name) == PASSIVE_FULL_HISTORY_ABLATION


def knowledge_consolidation_enabled(ablation_name: str) -> bool:
    return normalize_ablation_name(ablation_name) not in {
        PASSIVE_FULL_HISTORY_ABLATION,
        NO_CONSOLIDATION_ABLATION,
        NO_CONSOLIDATION_NO_RETRIEVAL_ABLATION,
    }


def compact_memory_only_enabled(ablation_name: str) -> bool:
    return normalize_ablation_name(ablation_name) == NO_CONSOLIDATION_NO_RETRIEVAL_ABLATION


def effective_max_retrieve_rounds(
    ablation_name: str,
    configured_max_retrieve_rounds: int,
) -> int:
    if passive_full_history_enabled(ablation_name) or compact_memory_only_enabled(ablation_name):
        return 0
    return int(configured_max_retrieve_rounds)


def validate_ablation_configuration(
    *,
    ablation_name: str,
    goal_kind: str,
    waypoint_policy_name: str,
) -> None:
    name = normalize_ablation_name(ablation_name)
    if name == NO_ABLATION:
        return
    if name in {
        PASSIVE_FULL_HISTORY_ABLATION,
        NO_CONSOLIDATION_ABLATION,
        NO_CONSOLIDATION_NO_RETRIEVAL_ABLATION,
    }:
        if str(goal_kind) != NAVPROBE_LANGUAGE_GOAL_KIND:
            raise ValueError(
                f"ablation {name!r} requires goal_kind={NAVPROBE_LANGUAGE_GOAL_KIND!r}"
            )
    if str(goal_kind) != NAVPROBE_LANGUAGE_GOAL_KIND:
        raise ValueError(f"ablation {name!r} does not support goal_kind={goal_kind!r}")
    if str(waypoint_policy_name) != FRONTIER_SKELETON_SAMPLE_WAYPOINT_POLICY:
        raise ValueError(f"unsupported waypoint policy: {waypoint_policy_name!r}")

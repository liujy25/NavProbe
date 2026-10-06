from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from navprobe.env.interface import EnvInterface


def episode_end_reason(env: Any) -> str | None:
    info = env.current_episode_info()
    if bool(info.get("episode_over", False)):
        return "episode_over"
    return None


def episode_over(env: Any) -> bool:
    return episode_end_reason(env) is not None


def current_episode_step(env: EnvInterface) -> int:
    info = env.current_episode_info()
    return int(info.get("pointnav_step_total", 0))


def _failure_metrics(metrics: object) -> dict[str, Any]:
    normalized = deepcopy(metrics) if isinstance(metrics, dict) else {}
    normalized["success"] = 0.0
    if "sr" in normalized:
        normalized["sr"] = 0.0
    for metric_name in ("spl", "sdtw"):
        if metric_name in normalized:
            normalized[metric_name] = 0.0
    return normalized


def finalize_evaluation_result(
    *,
    env: EnvInterface,
    existing_result: dict[str, Any] | None,
    agent_declared_done: bool,
) -> dict[str, Any]:
    evaluation_result = existing_result
    if evaluation_result is None:
        evaluation_result = env.stop() if agent_declared_done else env.evaluation_snapshot()
    normalized = dict(evaluation_result)
    normalized["pointnav_step_total"] = int(normalized.get("pointnav_step_total", current_episode_step(env)))
    normalized["episode_over"] = bool(
        normalized.get("episode_over", False) or episode_over(env)
    )
    normalized["agent_declared_done"] = bool(agent_declared_done)
    normalized["episode_success"] = bool(
        agent_declared_done and normalized.get("episode_success", False)
    )
    if not agent_declared_done:
        normalized["metrics"] = _failure_metrics(normalized.get("metrics"))
    return normalized

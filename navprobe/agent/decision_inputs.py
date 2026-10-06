"""Shared decision inputs and semantic request validation for navigation roles."""
from __future__ import annotations

import json
from typing import Callable, TYPE_CHECKING, TypeVar

import numpy as np

from navprobe.agent.visual_policy_prompt_images import current_panorama_prompt_text
from navprobe.agent.visual_policy_prompt_images import image_content_for_current_panorama_views
from navprobe.llm.request_context import request_log_context
from navprobe.llm.request_config import NAVPROBE_MODEL_MAX_ATTEMPTS, model_request_budget
from navprobe.agent.visual_action_context import VisualActionContext
from navprobe.memory.task_state import NavProbeTaskStateMemory

if TYPE_CHECKING:
    from navprobe.runtime.cache import RuntimeCache




def task_panorama_content(
    *,
    cache: "RuntimeCache",
    visual_context: VisualActionContext,
    landmark_panorama_views: dict[int, np.ndarray] | None,
    landmark_text: str,
    planning_reference_panorama: bool,
) -> list[dict[str, object]]:
    content: list[dict[str, object]] = []
    panorama_text = current_panorama_prompt_text(
        visual_context,
        include_visited_nodes=visual_context.graph_context_visible,
        planning_reference=bool(planning_reference_panorama),
    )
    if landmark_text != "":
        panorama_text += (
            "\n- Detected landmark boxes display the numeric suffix of each canonical "
            "landmark ref."
        )
    content.append({"type": "text", "text": panorama_text})
    content.extend(
        image_content_for_current_panorama_views(
            cache=cache,
            views=visual_context.views,
            include_visited_nodes=visual_context.graph_context_visible,
            image_overrides_by_angle=landmark_panorama_views,
            planning_reference=bool(planning_reference_panorama),
        )
    )
    return content


def task_state_content(
    *,
    task_state: NavProbeTaskStateMemory,
    visual_context: VisualActionContext,
    no_progress: bool,
    original_instruction: str | None,
    task_constraints: str,
) -> str:
    task_state_section = (
        "Original navigation instruction:\n" + str(original_instruction or "")
        if no_progress else
        "Task state (agenda and execution history):"
        + "\n"
        + task_state.format_for_prompt(
            include_node_bindings=visual_context.graph_context_visible,
            show_empty_predicates=True,
        )
    )
    if no_progress and task_constraints:
        task_state_section += "\n\nTask constraints:\n" + task_constraints
    return task_state_section


_Response = TypeVar("_Response")


def request_task_response(
    *,
    request: Callable[[str, str | list[dict[str, object]]], dict[str, object]],
    system_prompt: str,
    content: str | list[dict[str, object]],
    normalize_response: Callable[[dict[str, object]], _Response],
    retrieval_round: int | None,
    completed_retrieval_rounds: int,
) -> _Response:
    """Correct responses within the same three attempts as transport and JSON parsing."""
    retry_feedback = ""
    with model_request_budget() as budget:
        for attempt_index in range(NAVPROBE_MODEL_MAX_ATTEMPTS):
            request_content = content if isinstance(content, str) else list(content)
            if retry_feedback:
                if isinstance(request_content, str):
                    request_content += "\n\n" + retry_feedback
                else:
                    request_content.append({"type": "text", "text": retry_feedback})
            with request_log_context(
                retrieval_round=retrieval_round,
                completed_retrieval_rounds=completed_retrieval_rounds,
                semantic_attempt=attempt_index + 1,
            ):
                parsed = request(system_prompt, request_content)
            try:
                return normalize_response(parsed)
            except ValueError as exc:
                budget.record_validation_error(exc)
                if (
                    attempt_index + 1
                    >= NAVPROBE_MODEL_MAX_ATTEMPTS or not budget.remaining
                ):
                    raise
                retry_feedback = (
                    "Validation feedback from the previous response:\n"
                    f"{json.dumps(parsed, ensure_ascii=False, indent=2)}\n\n"
                    "Validation error:\n"
                    f"{exc}\n\n"
                    "Return one corrected response using the same output contract."
                )

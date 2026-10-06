"""Method identifiers and movement budget for vertical navigation.

``support_graph_2453f25`` selects the executable support-graph controller.
``paper_v8`` identifies the paper's HSGM/FSS method but has no executor;
selecting it raises NotImplementedError.
"""

from __future__ import annotations

import argparse


VERTICAL_METHOD_PAPER_V8 = "paper_v8"
VERTICAL_METHOD_SUPPORT_GRAPH_2453F25 = "support_graph_2453f25"
VERTICAL_METHOD_NAMES = (
    VERTICAL_METHOD_PAPER_V8,
    VERTICAL_METHOD_SUPPORT_GRAPH_2453F25,
)
DEFAULT_VERTICAL_METHOD = VERTICAL_METHOD_SUPPORT_GRAPH_2453F25


def normalize_vertical_method(value: object | None) -> str:
    method = str(value or DEFAULT_VERTICAL_METHOD).strip().lower()
    if method not in VERTICAL_METHOD_NAMES:
        raise ValueError(
            f"unsupported vertical_method={method!r}; "
            f"choose one of {', '.join(VERTICAL_METHOD_NAMES)}"
        )
    return method


def add_vertical_method_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--vertical-method",
        choices=VERTICAL_METHOD_NAMES,
        default=DEFAULT_VERTICAL_METHOD,
        help=(
            "VerticalMove implementation. paper_v8 is the paper HSGM/FSS "
            "route placeholder; support_graph_2453f25 is the current executable controller."
        ),
    )

NAVPROBE_VERTICAL_MAX_MOVES = 16


def vertical_execution_configuration(method: object | None = None) -> dict[str, object]:
    resolved_method = normalize_vertical_method(method)
    if resolved_method == VERTICAL_METHOD_PAPER_V8:
        return {
            "method": VERTICAL_METHOD_PAPER_V8,
            "lifecycle": "paper_hsgm_visual_waypoint_fss",
            "source_commit": "navprobev8",
            "controller": "paper_hsgm_visual_waypoint_fss",
            "implementation_status": "declared_not_wired",
            "stair_detector": "temporary_region_traversability",
            "waypoint_policy": "frontier_skeleton_sample",
            "max_moves": NAVPROBE_VERTICAL_MAX_MOVES,
        }
    return {
        "method": VERTICAL_METHOD_SUPPORT_GRAPH_2453F25,
        "lifecycle": "single_floor_support_controller_v2",
        "source_commit": "2453f25",
        "controller": "rgbd_support_graph_2453f25",
        "implementation_status": "executable",
        "subgoal_binding": "optional_native_id",
        "stair_detector": "not_used",
        "max_moves": NAVPROBE_VERTICAL_MAX_MOVES,
        "exit_completion": "current_support_and_executed_flat_travel",
        "exit_confirmation_radius_m": 0.9,
        "exit_confirmation_step_m": 0.4,
        "surface_input": "current_four_views_and_local_measurements",
        "surface_view_labels": "current_direction_view",
        "local_prompt_revision": "compact_surface_and_entrance_v1",
        "progress_evidence": "actual_pose_and_support_plane_height",
        "failure_policy": "terminate_episode",
    }

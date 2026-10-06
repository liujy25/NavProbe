from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from navprobe.mapping.exploration.manager import ExplorationManager
from navprobe.memory.graph.graph import Graph
from navprobe.visualization.action_mode_overlays import render_explore_bev_frontier_overlay
from navprobe.types import (
    LocalmapFrontierRecord,
    LocalmapPlaceReuseState,
)
from navprobe.runtime.cache import RuntimeCache


@dataclass
class StepVisualizationImageResult:
    active_bev_frontier_count: int = 0
    step_visualization_images: dict[str, str] = field(default_factory=dict)

    def timing_details(self) -> dict[str, object]:
        return {
            "active_bev_frontier_count": self.active_bev_frontier_count,
            "image_count": len(self.step_visualization_images),
        }


def should_skip_step_visualization_for_reuse(
    place_reuse_state: LocalmapPlaceReuseState | None,
) -> bool:
    return (
        place_reuse_state is not None
        and str(place_reuse_state.reason) != "vertical_transition_completed"
    )


def render_step_visualization_images(
    *,
    step_dir: Path,
    cache: RuntimeCache,
    graph: Graph,
    global_exploration: ExplorationManager,
    place_reuse_state: LocalmapPlaceReuseState | None,
    anchor_obs_id: str,
    current_place_node_id: str,
    floor_id: str,
    frontier_records: dict[str, LocalmapFrontierRecord],
) -> StepVisualizationImageResult:
    if should_skip_step_visualization_for_reuse(place_reuse_state):
        return StepVisualizationImageResult()

    anchor_observation = cache.get_observation(str(anchor_obs_id)).observation
    robot_xy = np.asarray(
        [float(anchor_observation.pose.x), float(anchor_observation.pose.y)],
        dtype=np.float64,
    )
    explore_bev_frontier_overlay = render_explore_bev_frontier_overlay(
        graph=graph,
        global_exploration=global_exploration,
        robot_xy=robot_xy,
        current_node_id=str(current_place_node_id),
        floor_id=str(floor_id),
        frontier_records=frontier_records,
    )

    maps_dir = step_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    explore_path = maps_dir / "explore_bev_frontier_overlay.png"
    Image.fromarray(explore_bev_frontier_overlay.image).save(explore_path)
    return StepVisualizationImageResult(
        active_bev_frontier_count=len(frontier_records),
        step_visualization_images={
            "explore_bev_frontier_overlay": str(explore_path.relative_to(step_dir)),
        },
    )

"""Waypoint displays reference the step's shared observation images."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from navprobe.agent.visual_action_context import direction_for_angle, ordered_panorama_angles
from navprobe.logging.recording import save_observation_rgb
from navprobe.visualization.waypoint_overlay import draw_waypoint_overlay_rgb

if TYPE_CHECKING:
    from navprobe.runtime.cache import RuntimeCache


def write_visual_waypoint_artifacts(
    *, step_dir: Path, cache: RuntimeCache, visual_decision: dict[str, object],
) -> dict[str, object]:
    if not visual_decision:
        return {}
    context = visual_decision.get("visual_context", {})
    views = {int(view["angle_deg"]): view for view in context.get("views", [])}
    result = {"panorama": {
        direction_for_angle(angle): save_observation_rgb(
            step_dir=step_dir, cache=cache, obs_id=str(views[angle]["obs_id"]),
        ) for angle in ordered_panorama_angles(list(views))
    }}
    waypoint = visual_decision.get("waypoint")
    if isinstance(waypoint, dict) and waypoint.get("obs_id"):
        obs_id = str(waypoint["obs_id"])
        result["selected_view"] = save_observation_rgb(step_dir=step_dir, cache=cache, obs_id=obs_id)
        point = waypoint.get("point_pixel")
        if point is not None:
            output = step_dir / "selected_waypoint_overlay.png"
            overlay = draw_waypoint_overlay_rgb(
                cache.get_observation(obs_id).observation.rgb,
                point_pixel=(float(point[0]), float(point[1])),
            )
            Image.fromarray(overlay).save(output)
            result["selected_waypoint_overlay"] = output.relative_to(step_dir).as_posix()
    return result

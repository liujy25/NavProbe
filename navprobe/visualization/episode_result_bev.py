from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from navprobe.visualization.graph_payload import graph_floor_height, graph_payload_edges
from navprobe.visualization.trajectory import smooth_display_path


EPISODE_RESULT_BEV_FILENAME = "episode_result_bev.png"
GOAL_MIN_HEIGHT_BELOW_FLOOR_M = 0.2
GOAL_MAX_HEIGHT_ABOVE_FLOOR_M = 2.0
GT_PATH_COLOR = (37, 99, 235)
ACTUAL_PATH_COLOR = (217, 70, 239)


def render_episode_result_scene_bev(
    *,
    run_dir: Path,
    reset_payload: dict[str, Any],
    graph_payload: dict[str, Any],
    stop_result: dict[str, Any],
    current_floor_id: str | None = None,
    pathfinder: Any | None = None,
    resolution: int = 2048,
) -> np.ndarray | None:
    """Render a scene-level final BEV overlay for Habitat episodes.

    NavProbe internal xy is converted to Habitat xz by (-y, -x), then projected
    through Habitat-Lab's topdown-map utilities.
    """
    owned_sim = None
    if pathfinder is None:
        owned_sim = _open_scene_simulator(reset_payload)
        if owned_sim is None:
            return None
        pathfinder = owned_sim.pathfinder

    try:
        maps = _import_habitat_maps()
        if maps is None:
            return None
        map_height = _map_height_from_reset(reset_payload)
        meters_per_pixel = maps.calculate_meters_per_pixel(int(resolution), pathfinder=pathfinder)
        topdown = maps.get_topdown_map(
            pathfinder,
            height=map_height,
            map_resolution=int(resolution),
            draw_border=True,
            meters_per_pixel=meters_per_pixel,
        )
        image = Image.fromarray(maps.colorize_topdown_map(topdown)).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        grid_shape = topdown.shape
        label_font = _font(max(14, int(min(image.size) * 0.018)))
        legend_font = _font(max(16, int(min(image.size) * 0.022)))
        line_width = max(5, int(min(image.size) * 0.006))

        reference_path = _reference_path(reset_payload)
        reference_px = [
            _habitat_xz_to_px(
                *position,
                grid_shape=grid_shape,
                pathfinder=pathfinder,
                maps=maps,
            )
            for position in smooth_display_path([
                _habitat_position_to_xz(position) for position in reference_path
            ])
        ]

        actual_internal_path = _actual_internal_path(run_dir=run_dir, graph_payload=graph_payload)
        start_internal = _internal_start_xy(reset_payload)
        if start_internal is not None and (
            actual_internal_path == [] or not _same_xy(start_internal, actual_internal_path[0])
        ):
            actual_internal_path = [start_internal, *actual_internal_path]
        final_internal = _internal_stop_xy(stop_result)
        if final_internal is not None:
            actual_internal_path = _append_unique_point(actual_internal_path, final_internal)
        actual_px = [
            _habitat_xz_to_px(
                *_internal_xy_to_habitat_xz(point),
                grid_shape=grid_shape,
                pathfinder=pathfinder,
                maps=maps,
            )
            for point in smooth_display_path(actual_internal_path)
        ]

        goal_items = _goal_items_for_current_floor(
            reset_payload=reset_payload,
            graph_payload=graph_payload,
            current_floor_id=current_floor_id,
        )
        goal_marker_specs: list[tuple[tuple[int, int], str]] = []
        for goal_index, goal in enumerate(goal_items):
            goal_px = _habitat_xz_to_px(
                *_habitat_position_to_xz(goal["position"]),
                grid_shape=grid_shape,
                pathfinder=pathfinder,
                maps=maps,
            )
            radius_m = _goal_success_radius_m(goal, reset_payload)
            radius_px = int(round(radius_m / float(meters_per_pixel)))
            _draw_success_radius(draw, goal_px, radius_px)
            goal_marker_specs.append(
                (goal_px, "goal" if len(goal_items) == 1 else f"goal {goal_index}")
            )

        _draw_polyline(draw, reference_px, fill=GT_PATH_COLOR, width=line_width)
        _draw_polyline(draw, actual_px, fill=ACTUAL_PATH_COLOR, width=line_width)

        for goal_px, goal_label in goal_marker_specs:
            _draw_marker(
                draw,
                goal_px,
                fill=(220, 38, 38),
                label=goal_label,
                radius=max(8, int(min(image.size) * 0.010)),
                font=label_font,
            )

        start_px = _start_marker_px(
            reset_payload=reset_payload,
            reference_px=reference_px,
            grid_shape=grid_shape,
            pathfinder=pathfinder,
            maps=maps,
        )
        if start_px is not None:
            _draw_marker(
                draw,
                start_px,
                fill=(22, 163, 74),
                label="start",
                radius=max(7, int(min(image.size) * 0.009)),
                font=label_font,
            )
        if actual_px:
            _draw_marker(
                draw,
                actual_px[-1],
                fill=(126, 34, 206),
                label="agent final",
                radius=max(7, int(min(image.size) * 0.009)),
                font=label_font,
            )

        _draw_legend(
            draw,
            image_size=image.size,
            font=legend_font,
            title=_episode_title(reset_payload),
            has_reference_path=bool(reference_px),
            goal_radius_label=_goal_radius_legend(goal_items, reset_payload),
            line_width=line_width,
        )

        image.alpha_composite(overlay)
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    finally:
        if owned_sim is not None:
            owned_sim.close()


def _import_habitat_maps():
    try:
        from habitat.utils.visualizations import maps
    except ModuleNotFoundError as exc:
        if exc.name != "habitat":
            raise
        return None
    return maps


def _open_scene_simulator(reset_payload: dict[str, Any]):
    scene_path = _scene_path_from_reset(reset_payload)
    if scene_path is None:
        return None
    try:
        import habitat_sim
    except ModuleNotFoundError as exc:
        if exc.name != "habitat_sim":
            raise
        return None

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_path)
    sim_cfg.enable_physics = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def _scene_path_from_reset(reset_payload: dict[str, Any]) -> Path | None:
    raw_candidates: list[object] = [
        reset_payload.get("habitat_scene_id"),
        reset_payload.get("scene_id"),
    ]
    scenes_dir = reset_payload.get("scenes_dir")
    scene_id = reset_payload.get("scene_id")
    if scenes_dir is not None and scene_id is not None:
        raw_candidates.insert(0, Path(str(scenes_dir)).expanduser() / str(scene_id))
    scene_dir = reset_payload.get("scene_dir")
    if scene_dir is not None:
        raw_candidates.append(scene_dir)

    for raw in raw_candidates:
        if raw is None:
            continue
        path = Path(str(raw)).expanduser()
        if path.is_file():
            return path.resolve()
        if path.is_dir():
            for pattern in ("*.basis.glb", "*.glb", "*.basis"):
                matches = sorted(path.glob(pattern))
                if matches:
                    return matches[0].resolve()
    return None


def _map_height_from_reset(reset_payload: dict[str, Any]) -> float:
    reference_path = _reference_path(reset_payload)
    if reference_path:
        return float(reference_path[0][1])
    start_position = reset_payload.get("start_position")
    if isinstance(start_position, (list, tuple)) and len(start_position) >= 2:
        return float(start_position[1])
    ground_height = reset_payload.get("ground_height_offset")
    if ground_height is not None:
        return float(ground_height)
    pose = reset_payload.get("pose")
    if isinstance(pose, dict) and pose.get("z") is not None:
        return float(pose["z"])
    return 0.0


def _reference_path(reset_payload: dict[str, Any]) -> list[Any]:
    raw_path = reset_payload.get("reference_path", [])
    return list(raw_path) if isinstance(raw_path, list) else []


def _goal_items(reset_payload: dict[str, Any]) -> list[dict[str, Any]]:
    goals = reset_payload.get("goals", [])
    if not isinstance(goals, list):
        return []
    items: list[dict[str, Any]] = []
    for goal in goals:
        if not isinstance(goal, dict):
            continue
        position = goal.get("position")
        if isinstance(position, (list, tuple)) and len(position) >= 3:
            items.append(goal)
    return items


def _goal_items_for_current_floor(
    *,
    reset_payload: dict[str, Any],
    graph_payload: dict[str, Any],
    current_floor_id: str | None,
) -> list[dict[str, Any]]:
    goal_items = _goal_items(reset_payload)
    floor_height = _habitat_floor_height_for_graph_floor(
        reset_payload=reset_payload,
        graph_payload=graph_payload,
        floor_id=current_floor_id,
    )
    if floor_height is None:
        return goal_items
    min_height = float(floor_height) - GOAL_MIN_HEIGHT_BELOW_FLOOR_M
    max_height = float(floor_height) + GOAL_MAX_HEIGHT_ABOVE_FLOOR_M
    return [
        goal
        for goal in goal_items
        if min_height <= float(goal["position"][1]) <= max_height
    ]


def _habitat_floor_height_for_graph_floor(
    *,
    reset_payload: dict[str, Any],
    graph_payload: dict[str, Any],
    floor_id: str | None,
) -> float | None:
    if floor_id is None:
        return None
    internal_height = graph_floor_height(graph_payload=graph_payload, floor_id=str(floor_id))
    if internal_height is None:
        return None
    return _habitat_floor_height_base(reset_payload) + float(internal_height)


def _habitat_floor_height_base(reset_payload: dict[str, Any]) -> float:
    ground_height = reset_payload.get("ground_height_offset")
    if ground_height is not None:
        return float(ground_height)
    return _map_height_from_reset(reset_payload)


def _goal_success_radius_m(goal: dict[str, Any], reset_payload: dict[str, Any]) -> float:
    radius = goal.get("radius")
    if radius is not None:
        return float(radius)
    if _reference_path(reset_payload):
        return 3.0
    return 1.0


def _goal_radius_legend(goal_items: list[dict[str, Any]], reset_payload: dict[str, Any]) -> str:
    if goal_items == []:
        return "goal success radius"
    radii = sorted({_goal_success_radius_m(goal, reset_payload) for goal in goal_items})
    if len(radii) == 1:
        return f"goal success radius: {radii[0]:g} m"
    return "goal success radius"


def _internal_xy_to_habitat_xz(point: tuple[float, float]) -> tuple[float, float]:
    return -float(point[1]), -float(point[0])


def _habitat_position_to_xz(position: Any) -> tuple[float, float]:
    return float(position[0]), float(position[2])


def _habitat_xz_to_px(
    x: float,
    z: float,
    *,
    grid_shape: tuple[int, int],
    pathfinder: Any,
    maps: Any,
) -> tuple[int, int]:
    row, col = maps.to_grid(z, x, grid_shape, pathfinder=pathfinder)
    return int(col), int(row)


def _internal_start_xy(reset_payload: dict[str, Any]) -> tuple[float, float] | None:
    pose = reset_payload.get("pose")
    if isinstance(pose, dict) and pose.get("x") is not None and pose.get("y") is not None:
        return float(pose["x"]), float(pose["y"])
    start_position = reset_payload.get("start_position")
    if isinstance(start_position, (list, tuple)) and len(start_position) >= 3:
        habitat_x = float(start_position[0])
        habitat_z = float(start_position[2])
        return -habitat_z, -habitat_x
    return None


def _internal_stop_xy(stop_result: dict[str, Any]) -> tuple[float, float] | None:
    pose = stop_result.get("pose")
    if not isinstance(pose, dict):
        return None
    if pose.get("x") is None or pose.get("y") is None:
        return None
    return float(pose["x"]), float(pose["y"])


def _start_marker_px(
    *,
    reset_payload: dict[str, Any],
    reference_px: list[tuple[int, int]],
    grid_shape: tuple[int, int],
    pathfinder: Any,
    maps: Any,
) -> tuple[int, int] | None:
    if reference_px:
        return reference_px[0]
    start_internal = _internal_start_xy(reset_payload)
    if start_internal is not None:
        return _habitat_xz_to_px(
            *_internal_xy_to_habitat_xz(start_internal),
            grid_shape=grid_shape,
            pathfinder=pathfinder,
            maps=maps,
        )
    return None


def _actual_internal_path(
    *,
    run_dir: Path,
    graph_payload: dict[str, Any],
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    def append_point(raw: Any) -> None:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            return
        point = (float(raw[0]), float(raw[1]))
        if points and _same_xy(points[-1], point):
            return
        points.append(point)

    def append_path(raw_path: Any) -> None:
        if not isinstance(raw_path, list):
            return
        for raw_point in raw_path:
            append_point(raw_point)

    for step_path in sorted((run_dir / "steps").glob("*/step.json")):
        step = json.loads(step_path.read_text(encoding="utf-8"))
        for result in step.get("execution", {}).get("results", []):
            if not isinstance(result, dict):
                continue
            append_path(result.get("path_xy"))

    if points:
        return points

    for edge in graph_payload_edges(graph_payload, include_vertical=True):
        append_path(edge.get("path_xy"))
    return points


def _append_unique_point(
    points: list[tuple[float, float]],
    point: tuple[float, float],
) -> list[tuple[float, float]]:
    if points and _same_xy(points[-1], point):
        return points
    return [*points, point]


def _same_xy(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return (
        round(float(left[0]), 4) == round(float(right[0]), 4)
        and round(float(left[1]), 4) == round(float(right[1]), 4)
    )


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        candidate = Path(path)
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=int(size))
    return ImageFont.load_default()


def _draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    fill: tuple[int, int, int],
    width: int,
) -> None:
    if len(points) < 2:
        return
    draw.line(points, fill=(255, 255, 255), width=int(width) + 4, joint="curve")
    draw.line(points, fill=fill, width=int(width), joint="curve")


def _draw_success_radius(
    draw: ImageDraw.ImageDraw,
    center_xy: tuple[int, int],
    radius_px: int,
) -> None:
    x, y = center_xy
    radius = max(1, int(radius_px))
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=(220, 38, 38, 45),
        outline=(220, 38, 38, 230),
        width=max(3, int(radius * 0.04)),
    )


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    *,
    fill: tuple[int, int, int],
    label: str,
    radius: int,
    font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    r = int(radius)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=(255, 255, 255), width=3)
    if label != "":
        draw.text(
            (x + r + 6, y - r - 2),
            label,
            fill=(0, 0, 0),
            font=font,
            stroke_width=3,
            stroke_fill=(255, 255, 255),
        )


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    *,
    image_size: tuple[int, int],
    font: ImageFont.ImageFont,
    title: str,
    has_reference_path: bool,
    goal_radius_label: str,
    line_width: int,
) -> None:
    legend_x = 18
    legend_y = 18
    line_h = int(font.size * 1.55)
    legend_items = []
    if has_reference_path:
        legend_items.append(("GT reference path", GT_PATH_COLOR))
    legend_items.extend(
        [
            ("actual executed path", ACTUAL_PATH_COLOR),
            (goal_radius_label, (220, 38, 38)),
        ]
    )
    legend_w = int(max(390, min(image_size) * 0.42))
    legend_h = line_h * (len(legend_items) + 1)
    draw.rounded_rectangle(
        (legend_x - 10, legend_y - 10, legend_x + legend_w, legend_y + legend_h),
        radius=10,
        fill=(255, 255, 255, 210),
        outline=(0, 0, 0, 60),
        width=1,
    )
    draw.text((legend_x, legend_y), title, fill=(0, 0, 0), font=font)
    for index, (text, color) in enumerate(legend_items, start=1):
        y = legend_y + index * line_h
        draw.line(
            (legend_x, y + line_h // 2, legend_x + 48, y + line_h // 2),
            fill=color,
            width=int(line_width),
        )
        draw.text((legend_x + 62, y), text, fill=(0, 0, 0), font=font)


def _episode_title(reset_payload: dict[str, Any]) -> str:
    episode_id = reset_payload.get("episode_id", "")
    scene = reset_payload.get("scene_name") or reset_payload.get("scene_id", "")
    if episode_id == "" and scene == "":
        return "episode result"
    return f"episode {episode_id}, scene {scene}"

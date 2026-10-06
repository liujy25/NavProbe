from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from navprobe.llm.image_preprocessing import LLM_CAMERA_IMAGE_MAX_SIZE

if TYPE_CHECKING:
    from navprobe.agent.state import NavProbeAgentState, NavProbeStepState
    from navprobe.memory.graph.edge import Edge


PANORAMA_DIRECTION_ORDER = ("front", "back", "left", "right")
_DIRECTION_BY_ANGLE = {
    0: "front",
    180: "back",
    90: "left",
    270: "right",
}
_ANGLE_BY_DIRECTION = {
    direction: angle for angle, direction in _DIRECTION_BY_ANGLE.items()
}
VISUAL_ACTION_ANGLES = [
    _ANGLE_BY_DIRECTION[direction] for direction in PANORAMA_DIRECTION_ORDER
]


def visual_action_angles(views: list["VisualViewContext"]) -> list[int]:
    return [int(view.angle_deg) for view in views]


def direction_for_angle(angle: int) -> str:
    normalized = int(angle) % 360
    try:
        return _DIRECTION_BY_ANGLE[normalized]
    except KeyError as exc:
        raise ValueError(f"panorama angle has no cardinal direction: {angle}") from exc


def angle_for_direction(direction: object) -> int:
    normalized = str(direction).strip().lower()
    try:
        return int(_ANGLE_BY_DIRECTION[normalized])
    except KeyError as exc:
        raise ValueError(f"invalid panorama direction: {direction!r}") from exc


def ordered_panorama_angles(angles: list[int]) -> list[int]:
    angles_by_direction = {
        direction_for_angle(int(angle)): int(angle) % 360 for angle in angles
    }
    return [
        angles_by_direction[direction]
        for direction in PANORAMA_DIRECTION_ORDER
        if direction in angles_by_direction
    ]


def ordered_visual_views(views: list["VisualViewContext"]) -> list["VisualViewContext"]:
    views_by_angle = {int(view.angle_deg): view for view in views}
    return [
        views_by_angle[int(angle)]
        for angle in ordered_panorama_angles(list(views_by_angle))
    ]


@dataclass(frozen=True)
class VisualViewContext:
    angle_deg: int
    obs_id: str
    rgb_id: str
    depth_id: str
    pose: dict[str, object] = field(default_factory=dict)
    node_overlay_image_id: str = ""
    visible_visited_nodes: list[str] = field(default_factory=list)
    visible_arrival_edge_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "angle_deg": int(self.angle_deg),
            "obs_id": str(self.obs_id),
            "rgb_id": str(self.rgb_id),
            "depth_id": str(self.depth_id),
            "pose": deepcopy(self.pose),
            "node_overlay_image_id": str(self.node_overlay_image_id),
            "visible_visited_nodes": [str(node_id) for node_id in self.visible_visited_nodes],
            "visible_arrival_edge_ids": list(self.visible_arrival_edge_ids),
        }


@dataclass(frozen=True)
class VisualActionContext:
    current_node_id: str
    views: list[VisualViewContext]
    graph_context_visible: bool = True
    arrival_edge_id: str = ""
    arrival_src_node_id: str = ""

    @property
    def available_angles(self) -> list[int]:
        return visual_action_angles(self.views)

    def view_for_angle(self, angle_deg: int) -> VisualViewContext:
        for view in self.views:
            if int(view.angle_deg) == int(angle_deg):
                return view
        raise ValueError(f"missing visual view for angle: {angle_deg}")

    def to_dict(self) -> dict[str, object]:
        return {
            "current_node_id": str(self.current_node_id),
            "views": [view.to_dict() for view in self.views],
            "graph_context_visible": bool(self.graph_context_visible),
            "arrival_edge_id": str(self.arrival_edge_id),
            "arrival_src_node_id": str(self.arrival_src_node_id),
        }


def build_visual_action_context(
    *,
    state: "NavProbeAgentState",
    step: "NavProbeStepState",
    include_graph_overlays: bool = True,
    include_arrival_edge: bool = False,
) -> VisualActionContext:
    current_node_id = str(step.current_place_node_id or state.current_place_node_id or "")
    if current_node_id == "":
        raise ValueError("visual action context requires current_place_node_id")
    views = _views_from_step(state=state, step=step)
    arrival_edge = (
        _latest_arrival_edge(state=state, current_node_id=current_node_id)
        if include_graph_overlays and include_arrival_edge
        else None
    )
    if include_graph_overlays:
        views = _attach_place_node_overlays(state=state, views=views, arrival_edge=arrival_edge)
    return VisualActionContext(
        current_node_id=current_node_id,
        views=views,
        graph_context_visible=include_graph_overlays,
        arrival_edge_id="" if arrival_edge is None else str(arrival_edge.id),
        arrival_src_node_id="" if arrival_edge is None else str(arrival_edge.src_id),
    )


def build_visual_action_context_for_node(
    *,
    state: "NavProbeAgentState",
    node_id: str,
) -> VisualActionContext:
    node_id_text = str(node_id).strip()
    if node_id_text == "":
        raise ValueError("visual action context requires node_id")
    node = state.graph.get_node(node_id_text)
    obs_ids = [str(obs_id) for obs_id in list(node.obs_ids)]
    if obs_ids == []:
        raise ValueError(f"node {node_id_text!r} has no stored panorama obs_ids")
    angle_to_obs_id = _angle_to_obs_id_from_ordered_obs_ids(obs_ids)
    views = _views_from_angle_to_obs_id(state=state, angle_to_obs_id=angle_to_obs_id)
    views = _attach_place_node_overlays(state=state, views=views)
    return VisualActionContext(
        current_node_id=node_id_text,
        views=views,
        graph_context_visible=True,
    )


def _views_from_step(
    *, state: "NavProbeAgentState", step: "NavProbeStepState",
) -> list[VisualViewContext]:
    angle_to_obs_id = {
        int(angle): str(obs_id)
        for angle, obs_id in step.angle_to_obs_id.items()
    }
    if angle_to_obs_id == {}:
        raise ValueError("visual action context missing panorama observations")
    # Reused places have observation IDs but no newly captured panorama_views.
    # Resolve pose and RGB-D from the same cached observation.
    return _views_from_angle_to_obs_id(state=state, angle_to_obs_id=angle_to_obs_id)


def _views_from_angle_to_obs_id(
    *,
    state: "NavProbeAgentState",
    angle_to_obs_id: dict[int, str],
) -> list[VisualViewContext]:
    views: list[VisualViewContext] = []
    for angle in sorted(angle_to_obs_id):
        obs_id = str(angle_to_obs_id[int(angle)])
        observation = state.cache.get_observation(obs_id).observation
        views.append(
            VisualViewContext(
                angle_deg=int(angle),
                obs_id=obs_id,
                rgb_id=f"{obs_id}:rgb",
                depth_id=f"{obs_id}:depth",
                pose=observation.pose.to_dict(),
            )
        )
    return views


def _angle_to_obs_id_from_ordered_obs_ids(obs_ids: list[str]) -> dict[int, str]:
    normalized_obs_ids = [str(obs_id) for obs_id in obs_ids]
    count = len(normalized_obs_ids)
    if count <= 0:
        return {}
    return {
        int(round(float(index) * 360.0 / float(count))) % 360: str(obs_id)
        for index, obs_id in enumerate(normalized_obs_ids)
    }


def _attach_place_node_overlays(
    *,
    state: "NavProbeAgentState",
    views: list[VisualViewContext],
    arrival_edge: Edge | None = None,
) -> list[VisualViewContext]:
    floor_id = str(state.system.current_floor_id)
    exploration = state.global_exploration_for_floor(floor_id)
    overlay_views = exploration.render_place_node_annotated_views(
        cache=state.cache,
        graph=state.graph,
        obs_ids=[str(view.obs_id) for view in views],
        floor_id=floor_id,
        image_max_size=LLM_CAMERA_IMAGE_MAX_SIZE,
        **({"arrival_edge": arrival_edge} if arrival_edge is not None else {}),
    )
    overlays_by_obs_id = {
        str(overlay.obs_id): overlay
        for overlay in overlay_views
    }
    annotated_views: list[VisualViewContext] = []
    for view in views:
        overlay = overlays_by_obs_id.get(str(view.obs_id))
        if overlay is None:
            annotated_views.append(view)
            continue
        annotated_views.append(
            replace(
                view,
                node_overlay_image_id=str(overlay.overlay_id),
                visible_visited_nodes=[str(node_id) for node_id in overlay.node_ids],
                visible_arrival_edge_ids=list(overlay.edge_ids),
            )
        )
    return annotated_views


def _latest_arrival_edge(*, state: "NavProbeAgentState", current_node_id: str) -> Edge | None:
    history = state.node_move_history
    if not history or str(state.current_place_node_id) != str(current_node_id):
        return None
    latest = history[-1]
    if str(latest.get("to_node", "")) != str(current_node_id):
        return None
    for edge in state.graph.iter_edges(include_vertical=True):
        if (
            str(edge.src_id) != str(latest.get("from_node", ""))
            or str(edge.dst_id) != str(current_node_id)
            or edge.relation not in {"move", "stairs_up", "stairs_down"}
        ):
            continue
        # Graph edges retain their first traversal; the arrival uses this move's path.
        path = latest.get("path_xy", edge.path_xy if edge.traversal_count == 1 else [])
        if len(path) >= 2:
            return replace(edge, path_xy=[tuple(point) for point in path])
    return None

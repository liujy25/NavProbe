"""Construct requested RGB/BEV evidence in the supplied retrieval workspace.

Request validation, retrieval budgets, round commits, and evidence lifetime
remain in episodic_retrieval. These builders do not request model decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from navprobe.agent.visual_action_context import ordered_visual_views, VisualViewContext
from navprobe.agent.landmark_context import _global_landmark_index
from navprobe.agent.visual_policy_prompt_images import (
    image_content_for_array,
    image_content_for_current_panorama_views,
    image_content_for_movement_history_sheet,
)
from navprobe.llm.image_preprocessing import LLM_CAMERA_IMAGE_MAX_SIZE, resize_rgb_to_fit
from navprobe.mapping.exploration.manager import project_visible_trajectory
from navprobe.mapping.exploration.overlay_drawing import _draw_place_node_circle
from navprobe.visualization.trajectory import edge_display_path_xyz, draw_rgb_trajectory
from navprobe.visualization.action_mode_overlays import render_task_state_bev_overlay_result, _node_labels_by_id

if TYPE_CHECKING:
    from navprobe.agent.state import NavProbeAgentState
    from navprobe.agent.episodic_retrieval import RetrievalWorkspace
    from navprobe.memory.landmarks import NavProbeLandmarkRecord


@dataclass(frozen=True)
class _EvidencePanel:
    key: str
    label: str
    content: tuple[dict[str, object], ...]
    source_obs_ids: tuple[str, ...]


@dataclass(frozen=True)
class _RenderedEdgeRgb:
    obs_id: str
    image: np.ndarray
    endpoint_visible: bool


def _render_shared_bev(
    *,
    state: "NavProbeAgentState",
    workspace: RetrievalWorkspace,
) -> None:
    floor_ids = set(workspace.selected_node_ids_by_floor)
    floor_ids.update(workspace.selected_edge_ids_by_floor)
    floor_ids.update(workspace.selected_landmark_ids_by_floor)
    for floor_id in sorted(floor_ids):
        floor_nodes = list(state.graph.iter_nodes(floor_id=str(floor_id)))
        all_node_labels = _node_labels_by_id(floor_nodes)
        workspace.node_display_labels_by_floor[floor_id] = {
            node_id: int(all_node_labels[node_id])
            for node_id in workspace.selected_node_ids_by_floor.get(floor_id, set())
            if node_id in all_node_labels
        }
        landmark_labels: dict[str, int] = {}
        workspace.landmark_display_labels_by_floor[floor_id] = landmark_labels
        landmark_markers: list[dict[str, object]] = []
        for landmark_id in sorted(
            workspace.selected_landmark_ids_by_floor.get(floor_id, set()),
            key=_reference_sort_key,
        ):
            candidate = state.landmark_controller.landmark_memory.find_record(landmark_id)
            if candidate is None:
                continue
            landmark_labels[landmark_id] = _global_landmark_index(landmark_id)
            position = candidate.position
            landmark_markers.append(
                {
                    "label": landmark_labels[landmark_id],
                    "xy": [float(position["x"]), float(position["y"])],
                }
            )
        render_result = render_task_state_bev_overlay_result(
            graph=state.graph,
            global_exploration=state.global_exploration_for_floor(floor_id),
            floor_id=floor_id,
            landmark_markers=landmark_markers,
            show_graph=True,
            node_ids=set(workspace.selected_node_ids_by_floor.get(floor_id, set())),
            edge_ids=set(workspace.selected_edge_ids_by_floor.get(floor_id, set())),
        )
        workspace.shared_bev_by_floor[floor_id] = render_result.image
        workspace.shared_bev_transform_by_floor[floor_id] = render_result.transform


def _add_node_rgb_panel(
    *,
    state: "NavProbeAgentState",
    workspace: RetrievalWorkspace,
    key: str,
    node_id: str,
    obs_ids: list[str],
) -> None:
    if key in workspace.image_panels or obs_ids == []:
        return
    views = _views_for_obs_ids(obs_ids)
    workspace.image_panels[key] = _EvidencePanel(
        key=key,
        label=f"Retrieved node {node_id} panorama RGB.",
        content=tuple(
            image_content_for_current_panorama_views(
                cache=state.cache,
                views=views,
                include_visited_nodes=False,
                label_prefix=f"Retrieved node {node_id} panorama view",
            )
        ),
        source_obs_ids=tuple(obs_ids),
    )


def _add_node_landmark_panel(
    *,
    state: "NavProbeAgentState",
    workspace: RetrievalWorkspace,
    key: str,
    node_id: str,
    floor_id: str,
) -> list[str]:
    node = state.graph.get_node(node_id)
    detections_by_obs = state.landmark_controller.landmark_memory.detections_for_obs_ids(node.obs_ids)
    detections_by_landmark: dict[str, list[dict[str, object]]] = {}
    for detections in detections_by_obs.values():
        for detection in detections:
            detections_by_landmark.setdefault(str(detection["landmark_id"]), []).append(detection)
    # Observation associations already resolve merged IDs and retain boxes from
    # historical views, even when their representative image has been evicted.
    detections_by_landmark = {
        landmark_id: detections_by_landmark[landmark_id]
        for landmark_id in sorted(detections_by_landmark, key=_reference_sort_key)
    }
    workspace.selected_landmark_ids_by_floor.setdefault(floor_id, set()).update(detections_by_landmark)
    if key in workspace.image_panels or not detections_by_landmark:
        return []
    obs_ids = [str(obs_id) for obs_id in node.obs_ids]
    ordered_views = ordered_visual_views(_views_for_obs_ids(obs_ids))
    ordered_obs_ids = [str(view.obs_id) for view in ordered_views]
    angles = [int(view.angle_deg) for view in ordered_views]
    images = _annotated_landmark_images(
        state=state,
        obs_ids=ordered_obs_ids,
        detections_by_landmark=detections_by_landmark,
    )
    image_overrides_by_angle = dict(zip(angles, images))
    workspace.image_panels[key] = _EvidencePanel(
        key=key,
        label=f"Retrieved landmark detections on node {node_id} panorama RGB.",
        content=tuple(
            image_content_for_current_panorama_views(
                cache=state.cache,
                views=ordered_views,
                include_visited_nodes=False,
                image_overrides_by_angle=image_overrides_by_angle,
                label_prefix=f"Retrieved node {node_id} landmark panorama view",
            )
        ),
        source_obs_ids=tuple(obs_ids),
    )
    return obs_ids


def _add_landmark_rgb_panel(
    *,
    state: "NavProbeAgentState",
    workspace: RetrievalWorkspace,
    key: str,
    candidate: "NavProbeLandmarkRecord",
) -> list[str]:
    obs_ids: list[str] = []
    for detection in candidate.detections:
        obs_id = str(detection.get("obs_id", ""))
        if obs_id != "" and obs_id not in obs_ids:
            obs_ids.append(obs_id)
    if key in workspace.image_panels or obs_ids == []:
        return obs_ids
    images = _annotated_landmark_images(
        state=state,
        obs_ids=obs_ids,
        detections_by_landmark={candidate.landmark_id: candidate.detections},
    )
    workspace.image_panels[key] = _EvidencePanel(
        key=key,
        label=f"Retrieved RGB evidence for {candidate.landmark_id}.",
        content=tuple(
            block
            for index, image in enumerate(images, start=1)
            for block in (
                {
                    "type": "text",
                    "text": f"Retrieved landmark evidence image {index}.",
                },
                image_content_for_array(image),
            )
        ),
        source_obs_ids=tuple(obs_ids),
    )
    return obs_ids


def _add_movement_panel(
    *,
    state: "NavProbeAgentState",
    workspace: RetrievalWorkspace,
    key: str,
    edge_id: str,
    obs_ids: list[str],
) -> None:
    if key in workspace.image_panels:
        return
    content = image_content_for_movement_history_sheet(cache=state.cache, obs_ids=obs_ids)
    if content is None:
        return
    workspace.image_panels[key] = _EvidencePanel(
        key=key,
        label=f"Retrieved chronological movement RGB for edge {edge_id}.",
        content=(content,),
        source_obs_ids=tuple(obs_ids),
    )


def _add_edge_rgb_panel(
    *,
    state: "NavProbeAgentState",
    workspace: RetrievalWorkspace,
    key: str,
    edge,
    src_node,
    dst_node,
) -> list[str]:
    if key in workspace.image_panels:
        return list(workspace.image_panels[key].source_obs_ids)
    rendered = _render_edge_rgb_overlay(
        state=state,
        edge=edge,
        src_node=src_node,
        dst_node=dst_node,
    )
    if rendered is None:
        workspace.unavailable_fields[key] = (
            f"No start-view RGB can display the endpoint or a visible trajectory segment "
            f"of edge {edge.id} from {src_node.id} to {dst_node.id}. "
            "This is an overlay visibility limitation, not evidence that the edge was not traversed."
        )
        return []
    workspace.unavailable_fields.pop(key, None)
    obs_id, image = rendered.obs_id, rendered.image
    endpoint_description = (
        "Arrows show travel from start to endpoint; the numbered node marker is the endpoint. "
        if rendered.endpoint_visible else
        "Arrows show the direction of travel; the endpoint is not visible in this view, "
        "so no endpoint marker is drawn. "
    )
    workspace.image_panels[key] = _EvidencePanel(
        key=key,
        label=(
            f"Retrieved edge {edge.id} start-view RGB from {src_node.id} toward "
            f"{dst_node.id}; the blue curve is the executed path, smoothed for display. "
            + endpoint_description
            + "Only visible path segments are drawn."
        ),
        content=(image_content_for_array(image),),
        source_obs_ids=(str(obs_id),),
    )
    return [str(obs_id)]


def _render_edge_rgb_overlay(
    *,
    state: "NavProbeAgentState",
    edge,
    src_node,
    dst_node,
) -> _RenderedEdgeRgb | None:
    if len(edge.path_xy) < 2:
        return None
    path_xyz = edge_display_path_xyz(
        edge=edge, src_node=src_node, dst_node=dst_node,
    )
    selected: tuple[tuple[int, float, int], str, np.ndarray, np.ndarray] | None = None
    segment_lengths = np.linalg.norm(np.diff(path_xyz, axis=0), axis=1)
    for obs_id in src_node.obs_ids:
        observation = state.cache.get_observation(str(obs_id)).observation
        if (
            observation.rgb is None
            or observation.depth is None
            or observation.intrinsics is None
            or observation.T_cam_odom is None
        ):
            continue
        rgb = resize_rgb_to_fit(
            observation.rgb, max_size=LLM_CAMERA_IMAGE_MAX_SIZE,
        ).image
        image_height, image_width = rgb.shape[:2]
        projected, valid = project_visible_trajectory(
            observation=observation, path_xyz=path_xyz,
            image_size=(image_width, image_height),
        )
        endpoint_visible = bool(valid[-1])
        if endpoint_visible:
            endpoint = projected[-1]
            coverage_score = -float(
                (float(endpoint[0]) - (float(image_width) - 1.0) / 2.0) ** 2
                + (float(endpoint[1]) - (float(image_height) - 1.0) / 2.0) ** 2
            )
        else:
            # Prefer the greatest visible path length, not the most densely sampled view.
            coverage_score = float(segment_lengths[valid[:-1] & valid[1:]].sum())
            if coverage_score <= 0:
                continue
        # Prefer views that show the endpoint, then rank by coverage and visibility.
        score = (int(endpoint_visible), coverage_score, int(np.count_nonzero(valid)))
        if selected is None or score > selected[0]:
            selected = (score, str(obs_id), rgb, np.column_stack([projected, valid]))
    if selected is None:
        return None

    _score, obs_id, rgb, projection_with_valid = selected
    projected = projection_with_valid[:, :2]
    valid = projection_with_valid[:, 2].astype(bool)
    image = Image.fromarray(rgb).convert("RGBA")
    draw_rgb_trajectory(
        draw=ImageDraw.Draw(image, "RGBA"), projected=projected, valid=valid,
    )
    if valid[-1]:
        _draw_place_node_circle(
            center_xy=(float(projected[-1, 0]), float(projected[-1, 1])),
            label=str(dst_node.id),
            image=image,
        )
    return _RenderedEdgeRgb(
        obs_id=obs_id,
        image=np.asarray(image.convert("RGB"), dtype=np.uint8),
        endpoint_visible=bool(valid[-1]),
    )


def _annotated_landmark_images(
    *,
    state: "NavProbeAgentState",
    obs_ids: list[str],
    detections_by_landmark: dict[str, list[dict[str, object]]],
) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for obs_id in obs_ids:
        raw = np.asarray(state.cache.get_observation(str(obs_id)).observation.rgb, dtype=np.uint8)
        canvas = Image.fromarray(raw).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        for landmark_id, detections in detections_by_landmark.items():
            display_label = str(_global_landmark_index(landmark_id))
            for detection in detections:
                if str(detection.get("obs_id", "")) != str(obs_id):
                    continue
                bbox = list(detection.get("bbox", []))
                if len(bbox) != 4:
                    continue
                box = tuple(float(value) for value in bbox)
                draw.rectangle(box, outline=(235, 67, 53), width=4)
                draw.text(
                    (box[0] + 2, max(0.0, box[1] - 12)),
                    display_label,
                    fill=(235, 67, 53),
                    font=font,
                )
        images.append(np.asarray(canvas, dtype=np.uint8))
    return images


def _views_for_obs_ids(obs_ids: list[str]) -> list[VisualViewContext]:
    count = len(obs_ids)
    return [
        VisualViewContext(
            angle_deg=int(round(float(index) * 360.0 / float(count))) % 360,
            obs_id=str(obs_id),
            rgb_id=f"{obs_id}:rgb",
            depth_id=f"{obs_id}:depth",
        )
        for index, obs_id in enumerate(obs_ids)
    ]


def _movement_keyframes(
    *,
    state: "NavProbeAgentState",
    obs_ids: list[str],
) -> list[str]:
    if len(obs_ids) <= 2:
        return list(obs_ids)
    observations = [state.cache.get_observation(obs_id).observation for obs_id in obs_ids]
    selected = [0]
    last_kept = observations[0].pose
    previous = observations[0].pose
    distance = 0.0
    for index in range(1, len(observations) - 1):
        pose = observations[index].pose
        distance += math.sqrt(
            (float(pose.x) - float(previous.x)) ** 2
            + (float(pose.y) - float(previous.y)) ** 2
            + (float(pose.z) - float(previous.z)) ** 2
        )
        yaw_difference = abs((float(pose.yaw) - float(last_kept.yaw) + 180.0) % 360.0 - 180.0)
        if distance >= 0.8 or yaw_difference >= 45.0:
            selected.append(index)
            last_kept = pose
            distance = 0.0
        previous = pose
    selected.append(len(obs_ids) - 1)
    return [str(obs_ids[index]) for index in selected]


def _reference_sort_key(value: str) -> tuple[str, int, str]:
    text = str(value)
    prefix, separator, suffix = text.rpartition("_")
    if separator == "":
        prefix = "".join(character for character in text if not character.isdigit())
        suffix = text[len(prefix) :]
    return prefix, int(suffix), text

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from navprobe.agent.visual_action_context import direction_for_angle
from navprobe.agent.visual_action_context import ordered_panorama_angles
from navprobe.agent.visual_action_context import ordered_visual_views
from navprobe.agent.visual_action_context import VisualActionContext, VisualViewContext
from navprobe.agent.visual_policy_prompt_images import _image_array_for_view
from navprobe.llm.image_preprocessing import LLM_CAMERA_IMAGE_MAX_SIZE
from navprobe.llm.image_preprocessing import resize_rgb_to_fit
from navprobe.llm.image_preprocessing import scale_bbox

if TYPE_CHECKING:
    from navprobe.memory.landmarks import NavProbeLandmarkRecord
    from navprobe.perception.landmark_detection import LandmarkDetectionController
    from navprobe.runtime.cache import RuntimeCache


@dataclass(frozen=True)
class NavProbeLandmarkEvidence:
    landmark_id: str
    class_name: str
    obs_id: str
    angle_deg: int
    bbox: list[float]
    score: float
    world_position: dict[str, float] = field(default_factory=dict)
    display_label: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "landmark_id": str(self.landmark_id),
            "class_name": str(self.class_name),
            "obs_id": str(self.obs_id),
            "angle_deg": int(self.angle_deg),
            "bbox": [float(value) for value in self.bbox],
            "score": float(self.score),
            "world_position": dict(self.world_position),
            "display_label": str(self.display_label),
        }


@dataclass(frozen=True)
class NavProbeLandmarkBevMarker:
    label: int
    class_name: str
    xy: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "label": int(self.label),
            "class_name": str(self.class_name),
            "xy": [float(self.xy[0]), float(self.xy[1])],
        }


@dataclass(frozen=True)
class NavProbeLandmarkContext:
    evidences: list[NavProbeLandmarkEvidence] = field(default_factory=list)
    bev_markers: list[NavProbeLandmarkBevMarker] = field(default_factory=list)
    text: str = ""


def build_landmark_context(
    *,
    landmark_controller: "LandmarkDetectionController",
    visual_context: VisualActionContext,
) -> NavProbeLandmarkContext:
    if not landmark_controller.is_active():
        return NavProbeLandmarkContext()
    candidates = landmark_controller.landmark_memory.iter_records()
    labels_by_landmark_id = _landmark_labels_by_id(candidates)
    evidences = _current_view_evidences(
        landmark_controller=landmark_controller,
        visual_context=visual_context,
        labels_by_landmark_id=labels_by_landmark_id,
    )
    bev_markers = _bev_markers(
        candidates=candidates,
        labels_by_landmark_id=labels_by_landmark_id,
    )
    return NavProbeLandmarkContext(
        evidences=evidences,
        bev_markers=bev_markers,
        text=_detected_landmarks_by_view_text(evidences, visual_context),
    )


def draw_landmark_panorama_views(
    *,
    cache: "RuntimeCache",
    visual_context: VisualActionContext,
    evidences: list[NavProbeLandmarkEvidence],
) -> dict[int, np.ndarray]:
    images_by_angle: dict[int, np.ndarray] = {}
    for view in ordered_visual_views(visual_context.views):
        raw_rgb = np.asarray(_image_array_for_view(cache=cache, view=view), dtype=np.uint8)
        resized = resize_rgb_to_fit(raw_rgb, max_size=LLM_CAMERA_IMAGE_MAX_SIZE)
        image = np.asarray(resized.image, dtype=np.uint8)
        image = draw_landmarks_on_view_image(
            image=image,
            cache=cache,
            view=view,
            evidences=evidences,
        )
        images_by_angle[int(view.angle_deg) % 360] = image
    return images_by_angle


def draw_landmarks_on_view_image(
    *,
    image: np.ndarray,
    cache: "RuntimeCache",
    view: VisualViewContext,
    evidences: list[NavProbeLandmarkEvidence],
) -> np.ndarray:
    canvas = Image.fromarray(np.asarray(image, dtype=np.uint8).copy()).convert("RGB")
    observation = cache.get_observation(str(view.obs_id)).observation
    original_rgb = np.asarray(observation.rgb, dtype=np.uint8)
    original_height, original_width = original_rgb.shape[:2]
    render_height, render_width = np.asarray(image).shape[:2]
    scale_x = float(render_width) / float(original_width)
    scale_y = float(render_height) / float(original_height)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for evidence in evidences:
        if str(evidence.obs_id) != str(view.obs_id):
            continue
        _draw_landmark_bbox(draw=draw, evidence=evidence, scale_x=scale_x, scale_y=scale_y, font=font)
    return np.asarray(canvas, dtype=np.uint8)


def _landmark_labels_by_id(
    candidates: list[NavProbeLandmarkRecord],
) -> dict[str, int]:
    return {
        candidate.landmark_id: _global_landmark_index(candidate.landmark_id)
        for candidate in candidates
    }


def _global_landmark_index(landmark_id: str) -> int:
    prefix, separator, index_text = str(landmark_id).rpartition("_")
    if prefix != "landmark" or separator == "" or not index_text.isdigit():
        raise ValueError(f"invalid global landmark id: {landmark_id!r}")
    return int(index_text)


def _current_view_evidences(
    *,
    landmark_controller: "LandmarkDetectionController",
    visual_context: VisualActionContext,
    labels_by_landmark_id: dict[str, int],
) -> list[NavProbeLandmarkEvidence]:
    angle_by_obs_id = {str(view.obs_id): int(view.angle_deg) for view in visual_context.views}
    obs_ids = list(angle_by_obs_id.keys())
    raw_by_obs_id = landmark_controller.detections_for_obs_ids(obs_ids)
    detections = [
        (str(obs_id), str(detection["landmark_id"]), detection)
        for obs_id in obs_ids
        for detection in raw_by_obs_id.get(str(obs_id), [])
        if str(detection["landmark_id"]) in labels_by_landmark_id
    ]
    evidences: list[NavProbeLandmarkEvidence] = []
    for obs_id, landmark_id, detection in sorted(
        detections,
        key=lambda item: (angle_by_obs_id[item[0]], labels_by_landmark_id[item[1]]),
    ):
        position = dict(detection.get("position", {}))
        evidences.append(
            NavProbeLandmarkEvidence(
                landmark_id=landmark_id,
                class_name=str(detection.get("class_name", "")),
                obs_id=str(obs_id),
                angle_deg=int(angle_by_obs_id[str(obs_id)]),
                bbox=[float(value) for value in list(detection.get("bbox", []))],
                score=float(detection.get("score", 0.0)),
                world_position={str(key): float(value) for key, value in position.items()},
                display_label=str(labels_by_landmark_id[landmark_id]),
            )
        )
    return evidences


def _bev_markers(
    *,
    candidates: list[NavProbeLandmarkRecord],
    labels_by_landmark_id: dict[str, int],
) -> list[NavProbeLandmarkBevMarker]:
    markers: list[NavProbeLandmarkBevMarker] = []
    for candidate in candidates:
        landmark_id = candidate.landmark_id
        if landmark_id not in labels_by_landmark_id:
            continue
        position = candidate.position
        if position.get("x") is None or position.get("y") is None:
            continue
        markers.append(
            NavProbeLandmarkBevMarker(
                label=int(labels_by_landmark_id[landmark_id]),
                class_name=candidate.class_name,
                xy=(float(position["x"]), float(position["y"])),
            )
        )
    markers.sort(key=lambda item: int(item.label))
    return markers


def _detected_landmarks_by_view_text(
    evidences: list[NavProbeLandmarkEvidence],
    visual_context: VisualActionContext,
) -> str:
    by_angle: dict[int, list[NavProbeLandmarkEvidence]] = {}
    for evidence in evidences:
        by_angle.setdefault(int(evidence.angle_deg), []).append(evidence)
    lines = [
        "Detected landmarks by view (canonical refs; boxes show their numeric suffixes):"
    ]
    for angle_value in ordered_panorama_angles(visual_context.available_angles):
        items = sorted(by_angle.get(int(angle_value), []), key=lambda item: int(item.display_label or 10**9))
        item_text = ", ".join(
            f"{item.landmark_id} {item.class_name} confidence={float(item.score):.2f}"
            for item in items
            if str(item.landmark_id).strip() != ""
        )
        if item_text != "":
            lines.append(f"- {direction_for_angle(angle_value)}: {item_text}")
    return "\n".join(lines)


def _draw_landmark_bbox(
    *,
    draw: ImageDraw.ImageDraw,
    evidence: NavProbeLandmarkEvidence,
    scale_x: float,
    scale_y: float,
    font: ImageFont.ImageFont,
) -> None:
    if len(evidence.bbox) != 4:
        return
    x0, y0, x1, y1 = scale_bbox(evidence.bbox, scale_x=scale_x, scale_y=scale_y)
    color = _landmark_color(str(evidence.class_name))
    draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
    label = str(evidence.display_label).strip()
    if label == "":
        return
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = int(text_bbox[2] - text_bbox[0])
    text_h = int(text_bbox[3] - text_bbox[1])
    pad = 3
    box = (x0, max(0, y0 - text_h - 2 * pad), x0 + text_w + 2 * pad, max(text_h + 2 * pad, y0))
    draw.rectangle(box, fill=color)
    draw.text((box[0] + pad, box[1] + pad), label, fill=(255, 255, 255), font=font)


def _landmark_color(class_name: str) -> tuple[int, int, int]:
    palette = [
        (235, 67, 53),
        (52, 168, 83),
        (66, 133, 244),
        (251, 188, 5),
        (171, 71, 188),
        (0, 172, 193),
    ]
    index = sum(ord(ch) for ch in str(class_name)) % len(palette)
    return palette[index]

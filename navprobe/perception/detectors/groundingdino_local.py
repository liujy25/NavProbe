from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert

from navprobe.perception.detectors.interface import Detection2D, DetectorInterface


class GroundingDINOLocalDetector(DetectorInterface):
    """Local GroundingDINO detector adapter for NavProbe's DetectorInterface."""

    def __init__(
        self,
        *,
        model_config_path: str,
        model_checkpoint_path: str,
        device: str,
        box_threshold: float,
        text_threshold: float,
    ) -> None:
        self.model_config_path = str(model_config_path)
        self.model_checkpoint_path = str(model_checkpoint_path)
        self.device = str(device)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self._model = None

    def detect(self, image_rgb: np.ndarray, class_name: str) -> list[Detection2D]:
        return self.detect_classes(
            image_rgb=image_rgb,
            class_names=[class_name],
            agnostic_nms=False,
        )

    def detect_classes(
        self,
        image_rgb: np.ndarray,
        class_names: list[str],
        agnostic_nms: bool = False,
    ) -> list[Detection2D]:
        del agnostic_nms
        query_classes = _normalize_query_classes(class_names)
        if query_classes == []:
            raise ValueError("GroundingDINOLocalDetector.detect_classes requires non-empty class_names")
        image = np.asarray(image_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("GroundingDINOLocalDetector requires RGB image with shape HxWx3")

        model = self._ensure_model()
        device = self._predict_device()
        image_tensor = _preprocess_image(image).to(device)
        detections: list[Detection2D] = []
        for query_class in query_classes:
            boxes, scores, phrases = _predict_groundingdino(
                model=model,
                image=image_tensor,
                caption=query_class,
                box_threshold=float(self.box_threshold),
                text_threshold=float(self.text_threshold),
                device=device,
            )
            detections.extend(
                _detections_from_predictions(
                    image_rgb=image,
                    boxes=boxes,
                    scores=scores,
                    phrases=phrases,
                    query_classes=[query_class],
                )
            )
        detections.sort(key=lambda item: float(item.score), reverse=True)
        return detections

    def _predict_device(self) -> str:
        requested = str(self.device).strip()
        if requested == "":
            raise ValueError("GroundingDINOLocalDetector requires a non-empty device")
        if requested.startswith("cuda") and not bool(torch.cuda.is_available()):
            raise ValueError(
                "GroundingDINOLocalDetector requested "
                f"device={requested!r} but torch.cuda.is_available() is False"
            )
        return requested

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        config_path = Path(self.model_config_path).expanduser().resolve()
        checkpoint_path = Path(self.model_checkpoint_path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"GroundingDINO config does not exist: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"GroundingDINO checkpoint does not exist: {checkpoint_path}")

        from groundingdino.models import build_model
        from groundingdino.util.misc import clean_state_dict
        from groundingdino.util.slconfig import SLConfig

        args = SLConfig.fromfile(str(config_path))
        args.device = self._predict_device()
        args.text_encoder_type = _cached_huggingface_snapshot(
            str(args.text_encoder_type)
        )
        model = build_model(args)
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        model.to(self._predict_device())
        self._model = model
        return self._model


def _cached_huggingface_snapshot(model_id: str) -> str:
    model_id_text = str(model_id).strip()
    if model_id_text == "" or Path(model_id_text).expanduser().exists():
        return model_id_text
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return str(snapshot_download(repo_id=model_id_text, local_files_only=True))
    except LocalEntryNotFoundError:
        return model_id_text


def _preprocess_image(image_rgb: np.ndarray) -> torch.Tensor:
    from groundingdino.datasets import transforms as grounding_transforms

    transform = grounding_transforms.Compose(
        [
            grounding_transforms.RandomResize([800], max_size=1333),
            grounding_transforms.ToTensor(),
            grounding_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_pil = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")
    image_transformed, _ = transform(image_pil, None)
    return image_transformed


def _predict_groundingdino(
    *,
    model: Any,
    image: torch.Tensor,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    from groundingdino.util.utils import get_phrases_from_posmap

    prompt = _preprocess_caption(caption)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[prompt])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]
    prediction_boxes = outputs["pred_boxes"].cpu()[0]
    mask = prediction_logits.max(dim=1)[0] > float(box_threshold)
    logits = prediction_logits[mask]
    boxes = prediction_boxes[mask]

    tokenizer = model.tokenizer
    tokenized = tokenizer(prompt)
    phrases = [
        get_phrases_from_posmap(logit > float(text_threshold), tokenized, tokenizer).replace(".", "")
        for logit in logits
    ]
    return boxes, logits.max(dim=1)[0], phrases


def _detections_from_predictions(
    *,
    image_rgb: np.ndarray,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    phrases: list[str],
    query_classes: list[str],
) -> list[Detection2D]:
    height, width = image_rgb.shape[:2]
    xyxy = box_convert(
        boxes=boxes * torch.tensor([width, height, width, height], dtype=boxes.dtype),
        in_fmt="cxcywh",
        out_fmt="xyxy",
    ).detach().cpu().numpy()
    score_values = scores.detach().cpu().numpy()
    detections: list[Detection2D] = []
    for bbox, score, phrase in zip(xyxy, score_values, phrases):
        class_name = _class_name_from_phrase(phrase=str(phrase), query_classes=query_classes)
        if class_name is None:
            continue
        x0, y0, x1, y1 = _clip_bbox(bbox=bbox, image_width=width, image_height=height)
        if x1 <= x0 or y1 <= y0:
            continue
        detections.append(
            Detection2D(
                class_name=class_name,
                bbox=[float(x0), float(y0), float(x1), float(y1)],
                score=float(score),
            )
        )
    detections.sort(key=lambda item: float(item.score), reverse=True)
    return detections


def _normalize_query_classes(class_names: list[str]) -> list[str]:
    query_classes: list[str] = []
    seen: set[str] = set()
    for class_name in class_names:
        query_class = str(class_name).strip().replace("_", " ")
        if query_class == "" or query_class in seen:
            continue
        query_classes.append(query_class)
        seen.add(query_class)
    return query_classes


def _preprocess_caption(caption: str) -> str:
    prompt = str(caption).lower().strip()
    if prompt.endswith("."):
        return prompt
    return prompt + "."


def _class_name_from_phrase(*, phrase: str, query_classes: list[str]) -> str | None:
    normalized_phrase = str(phrase).strip().lower()
    if normalized_phrase == "" and len(query_classes) == 1:
        return query_classes[0]
    for query_class in query_classes:
        if str(query_class).lower() in normalized_phrase:
            return query_class
    if len(query_classes) == 1:
        return query_classes[0]
    return None


def _clip_bbox(
    *,
    bbox: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return (
        _clip(x0, 0.0, float(image_width)),
        _clip(y0, 0.0, float(image_height)),
        _clip(x1, 0.0, float(image_width)),
        _clip(y1, 0.0, float(image_height)),
    )


def _clip(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))

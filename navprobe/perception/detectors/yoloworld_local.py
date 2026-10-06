from __future__ import annotations

import numpy as np
import torch
from ultralytics import YOLOWorld

from navprobe.perception.detectors.interface import Detection2D, DetectorInterface




class YOLOWorldLocalDetector(DetectorInterface):
    def __init__(
        self,
        model_path: str,
        device: str,
        conf_threshold: float,
    ) -> None:
        self.model_path = str(model_path)
        self.device = str(device)
        self.conf_threshold = float(conf_threshold)
        self._model = None
        self._active_query_classes: tuple[str, ...] | None = None

    def _predict_device(self) -> str:
        requested = str(self.device).strip()
        if requested == "":
            raise ValueError("YOLOWorldLocalDetector requires a non-empty device")
        if requested.startswith("cuda") and not bool(torch.cuda.is_available()):
            raise ValueError(
                f"YOLOWorldLocalDetector requested device={requested!r} but torch.cuda.is_available() is False"
            )
        return requested

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._model = YOLOWorld(self.model_path)

    def _sync_model_device(self) -> str:
        if self._model is None:
            raise ValueError("YOLOWorld model failed to initialize")
        predict_device = self._predict_device()
        target_device = torch.device(predict_device)
        if target_device.type == "cuda" and target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())
        if self._model.device != target_device:
            self._model.to(predict_device)
            if getattr(self._model.model, "clip_model", None) is not None:
                self._model.model.clip_model = None
        return predict_device

    @staticmethod
    def _normalize_query_class(class_name: str) -> str:
        return str(class_name).strip().replace("_", " ")

    @classmethod
    def _normalize_query_classes(cls, class_names: list[str]) -> list[str]:
        query_classes: list[str] = []
        seen: set[str] = set()
        for class_name in class_names:
            query_class = cls._normalize_query_class(str(class_name))
            if query_class == "" or query_class in seen:
                continue
            query_classes.append(query_class)
            seen.add(query_class)
        return query_classes

    def _set_active_classes(self, query_classes: list[str]) -> None:
        if self._model is None:
            raise ValueError("YOLOWorld model failed to initialize")
        active_query_classes = tuple(query_classes)
        if self._active_query_classes == active_query_classes:
            return
        self._model.set_classes(list(query_classes))
        self._active_query_classes = active_query_classes

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
        self._ensure_model()
        if self._model is None:
            raise ValueError("YOLOWorld model failed to initialize")

        query_classes = self._normalize_query_classes(class_names)
        if query_classes == []:
            raise ValueError("YOLOWorldLocalDetector.detect_classes requires at least one non-empty class_name")

        predict_device = self._sync_model_device()
        self._set_active_classes(query_classes)
        image_bgr = np.ascontiguousarray(np.asarray(image_rgb, dtype=np.uint8)[:, :, ::-1])
        results = self._model.predict(
            source=image_bgr,
            conf=float(self.conf_threshold),
            device=predict_device,
            agnostic_nms=bool(agnostic_nms),
            verbose=False,
        )
        if results == []:
            return []
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes.xyxy) == 0:
            return []

        xyxy_values = boxes.xyxy.detach().cpu().numpy()
        confidence_values = boxes.conf.detach().cpu().numpy()
        class_index_values = boxes.cls.detach().cpu().numpy()
        height, width = image_rgb.shape[:2]
        detections: list[Detection2D] = []
        for xyxy, confidence, class_index in zip(xyxy_values, confidence_values, class_index_values):
            class_index_int = int(round(float(class_index)))
            if class_index_int < 0 or class_index_int >= len(query_classes):
                continue
            x0 = max(0, min(width, int(round(float(xyxy[0])))))
            y0 = max(0, min(height, int(round(float(xyxy[1])))))
            x1 = max(0, min(width, int(round(float(xyxy[2])))))
            y1 = max(0, min(height, int(round(float(xyxy[3])))))
            if x1 <= x0 or y1 <= y0:
                continue
            detections.append(
                Detection2D(
                    class_name=str(query_classes[class_index_int]),
                    bbox=[float(x0), float(y0), float(x1), float(y1)],
                    score=float(confidence),
                )
            )
        return detections

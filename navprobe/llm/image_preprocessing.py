from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image


LLM_IMAGE_TILE_SIZE = (512, 512)
LLM_IMAGE_PADDING_RGB = (128, 128, 128)
LLM_IMAGE_SHEET_GAP_PX = 8
LLM_IMAGE_SHEET_MAX_COLUMNS = 3
LLM_CAMERA_IMAGE_MAX_SIZE = (640, 480)


@dataclass(frozen=True)
class LLMImagePreprocessResult:
    image: np.ndarray
    metadata: dict[str, object]


def normalize_rgb_array(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=2)
    if array.ndim != 3:
        raise ValueError(f"rgb image must be HxWxC, got shape={array.shape!r}")
    if int(array.shape[2]) == 4:
        array = array[:, :, :3]
    if int(array.shape[2]) != 3:
        raise ValueError(f"rgb image must have 3 channels, got shape={array.shape!r}")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if np.issubdtype(array.dtype, np.floating):
        return np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.clip(array, 0, 255).astype(np.uint8)


def letterbox_rgb(
    image: Any,
    *,
    target_size: tuple[int, int] = LLM_IMAGE_TILE_SIZE,
    padding_rgb: tuple[int, int, int] = LLM_IMAGE_PADDING_RGB,
) -> LLMImagePreprocessResult:
    """Resize an RGB image into a fixed canvas without changing its aspect ratio."""
    rgb = normalize_rgb_array(image)
    original_height, original_width = int(rgb.shape[0]), int(rgb.shape[1])
    target_width, target_height = int(target_size[0]), int(target_size[1])
    if original_width <= 0 or original_height <= 0:
        raise ValueError(f"cannot letterbox empty image: shape={rgb.shape!r}")
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"target_size must be positive, got {target_size!r}")

    scale = min(float(target_width) / float(original_width), float(target_height) / float(original_height))
    resized_width = max(1, int(round(float(original_width) * scale)))
    resized_height = max(1, int(round(float(original_height) * scale)))
    offset_x = int((target_width - resized_width) // 2)
    offset_y = int((target_height - resized_height) // 2)

    resized = Image.fromarray(rgb).resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    canvas = np.full((target_height, target_width, 3), np.asarray(padding_rgb, dtype=np.uint8), dtype=np.uint8)
    canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = np.asarray(resized, dtype=np.uint8)
    return LLMImagePreprocessResult(
        image=canvas,
        metadata={
            "mode": "letterbox",
            "original_width": original_width,
            "original_height": original_height,
            "target_width": target_width,
            "target_height": target_height,
            "resized_width": resized_width,
            "resized_height": resized_height,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "scale": float(scale),
            "padding_rgb": [int(value) for value in padding_rgb],
        },
    )


def resize_to_fit_metadata(
    image_shape: tuple[int, ...],
    *,
    max_size: tuple[int, int] | None = LLM_CAMERA_IMAGE_MAX_SIZE,
) -> dict[str, object]:
    """Compute resize dimensions and exact coordinate scales without pixels."""
    original_height, original_width = int(image_shape[0]), int(image_shape[1])
    if original_width <= 0 or original_height <= 0:
        raise ValueError(f"cannot resize empty image: shape={image_shape!r}")
    if max_size is None:
        resized_width, resized_height = original_width, original_height
    else:
        max_width, max_height = int(max_size[0]), int(max_size[1])
        if max_width <= 0 or max_height <= 0:
            raise ValueError(f"max_size must be positive, got {max_size!r}")
        scale = min(
            1.0,
            float(max_width) / float(original_width),
            float(max_height) / float(original_height),
        )
        resized_width = max(1, int(round(float(original_width) * scale)))
        resized_height = max(1, int(round(float(original_height) * scale)))
    return {
        "mode": "resize_to_fit",
        "original_width": original_width,
        "original_height": original_height,
        "width": resized_width,
        "height": resized_height,
        "scale_x": float(resized_width) / float(original_width),
        "scale_y": float(resized_height) / float(original_height),
    }


def resize_rgb_to_fit(
    image: Any,
    *,
    max_size: tuple[int, int] | None = LLM_CAMERA_IMAGE_MAX_SIZE,
) -> LLMImagePreprocessResult:
    """Resize an RGB image to fit inside max_size without padding or upscaling."""
    rgb = normalize_rgb_array(image)
    metadata = resize_to_fit_metadata(rgb.shape, max_size=max_size)
    resized_width, resized_height = int(metadata["width"]), int(metadata["height"])
    if (resized_height, resized_width) == rgb.shape[:2]:
        resized = rgb
    else:
        resized = np.asarray(
            Image.fromarray(rgb).resize((resized_width, resized_height), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
    return LLMImagePreprocessResult(
        image=np.ascontiguousarray(resized),
        metadata=metadata,
    )


def scale_xy(
    xy: tuple[float, float],
    *,
    scale_x: float,
    scale_y: float,
) -> tuple[float, float]:
    return (float(xy[0]) * float(scale_x), float(xy[1]) * float(scale_y))


def scale_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    *,
    scale_x: float,
    scale_y: float,
) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError(f"bbox must have length 4, got {bbox!r}")
    return (
        float(bbox[0]) * float(scale_x),
        float(bbox[1]) * float(scale_y),
        float(bbox[2]) * float(scale_x),
        float(bbox[3]) * float(scale_y),
    )


def compose_llm_image_tile_sheet(
    images: list[Any],
    *,
    tile_size: tuple[int, int] = LLM_IMAGE_TILE_SIZE,
    padding_rgb: tuple[int, int, int] = LLM_IMAGE_PADDING_RGB,
    gap_px: int = LLM_IMAGE_SHEET_GAP_PX,
    max_columns: int = LLM_IMAGE_SHEET_MAX_COLUMNS,
) -> LLMImagePreprocessResult:
    """Build a dynamic prompt sheet from independently letterboxed image tiles."""
    if images == []:
        raise ValueError("LLM tile sheet requires at least one image")
    if int(max_columns) <= 0:
        raise ValueError(f"max_columns must be positive, got {max_columns!r}")
    if int(gap_px) < 0:
        raise ValueError(f"gap_px must be non-negative, got {gap_px!r}")

    tile_results = [
        letterbox_rgb(image, target_size=tile_size, padding_rgb=padding_rgb)
        for image in images
    ]
    tile_width, tile_height = int(tile_size[0]), int(tile_size[1])
    columns = min(int(max_columns), len(tile_results))
    rows = int(math.ceil(float(len(tile_results)) / float(columns)))
    sheet_width = columns * tile_width + max(0, columns - 1) * int(gap_px)
    sheet_height = rows * tile_height + max(0, rows - 1) * int(gap_px)
    sheet = np.full((sheet_height, sheet_width, 3), np.asarray(padding_rgb, dtype=np.uint8), dtype=np.uint8)
    tile_metadata: list[dict[str, object]] = []
    for index, result in enumerate(tile_results):
        row = index // columns
        col = index % columns
        x0 = col * (tile_width + int(gap_px))
        y0 = row * (tile_height + int(gap_px))
        sheet[y0:y0 + tile_height, x0:x0 + tile_width] = result.image
        metadata = dict(result.metadata)
        metadata.update(
            {
                "tile_index": int(index),
                "sheet_x": int(x0),
                "sheet_y": int(y0),
                "sheet_width": int(tile_width),
                "sheet_height": int(tile_height),
            }
        )
        tile_metadata.append(metadata)

    return LLMImagePreprocessResult(
        image=sheet,
        metadata={
            "mode": "tile_sheet",
            "tile_count": int(len(tile_results)),
            "columns": int(columns),
            "rows": int(rows),
            "sheet_width": int(sheet_width),
            "sheet_height": int(sheet_height),
            "tile_width": int(tile_width),
            "tile_height": int(tile_height),
            "gap_px": int(gap_px),
            "padding_rgb": [int(value) for value in padding_rgb],
            "tiles": tile_metadata,
        },
    )


def encode_rgb_to_jpeg_data_url(image: Any, *, quality: int = 90) -> str:
    """Encode an RGB image array as a JPEG data URL for multimodal chat messages."""
    rgb = normalize_rgb_array(image)
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=int(quality))
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

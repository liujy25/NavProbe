from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BEV_UNKNOWN_BACKGROUND_COLOR = (92, 92, 92)
BEV_EXPLORED_FREE_COLOR = (190, 248, 200)
BEV_DILATED_OBSTACLE_COLOR = (220, 38, 38)
BEV_OBSTACLE_COLOR = (220, 38, 38)

BEV_PLACE_NODE_FILL = (26, 112, 220, 255)
PLACE_NODE_MARKER_FILL = (26, 112, 220, 230)
PLACE_NODE_MARKER_OUTLINE = (255, 255, 255, 255)
PLACE_NODE_MARKER_TEXT = (255, 255, 255, 255)
BEV_GRAPH_EDGE_COLOR = (20, 20, 20, 210)

BEV_VIEW_SCALE = 2.0
BEV_PLACE_NODE_RADIUS_PX = 10


def load_visual_font(size_px: int) -> ImageFont.ImageFont:
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, size=int(size_px))
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass(frozen=True)
class NumberedCircleMarkerStyle:
    radius_px: int
    font_size_px: int
    fill: tuple[int, int, int, int] = (255, 255, 255, 145)
    outline: tuple[int, int, int, int] = (35, 35, 35, 72)
    text_fill: tuple[int, int, int, int] = (18, 18, 18, 245)
    outline_width_px: int = 1
    supersample: int = 4


def _clamp_marker_int(value: float, min_value: int, max_value: int) -> int:
    return max(int(min_value), min(int(max_value), int(round(float(value)))))


def numbered_circle_marker_style(
    *,
    image_width: int,
    image_height: int,
    mode: str = "rgb",
) -> NumberedCircleMarkerStyle:
    short_side = max(1, min(int(image_width), int(image_height)))
    if str(mode).strip().lower() == "bev":
        radius_px = _clamp_marker_int(float(short_side) * 0.018, 8, 11)
        font_size_px = int(round(float(radius_px) * 1.28))
    else:
        radius_px = _clamp_marker_int(float(short_side) * 0.024, 9, 12)
        font_size_px = int(round(float(radius_px) * 1.18))
    return NumberedCircleMarkerStyle(
        radius_px=int(radius_px),
        font_size_px=int(font_size_px),
    )


def place_node_marker_style(
    *,
    image_width: int,
    image_height: int,
    mode: str = "rgb",
) -> NumberedCircleMarkerStyle:
    base_style = numbered_circle_marker_style(
        image_width=int(image_width),
        image_height=int(image_height),
        mode=str(mode),
    )
    return NumberedCircleMarkerStyle(
        radius_px=int(base_style.radius_px),
        font_size_px=int(base_style.font_size_px),
        fill=PLACE_NODE_MARKER_FILL,
        outline=PLACE_NODE_MARKER_OUTLINE,
        text_fill=PLACE_NODE_MARKER_TEXT,
        outline_width_px=max(2, int(base_style.outline_width_px)),
        supersample=int(base_style.supersample),
    )


def clamp_numbered_circle_marker_center(
    *,
    center_xy: tuple[float, float],
    style: NumberedCircleMarkerStyle,
    image_size: tuple[int, int],
) -> tuple[float, float]:
    radius = float(style.radius_px) + float(style.outline_width_px)
    width, height = int(image_size[0]), int(image_size[1])
    x = min(max(float(center_xy[0]), radius), max(radius, float(width) - radius - 1.0))
    y = min(max(float(center_xy[1]), radius), max(radius, float(height) - radius - 1.0))
    return (float(x), float(y))


def _numbered_circle_text_y_offset(font: ImageFont.ImageFont) -> float:
    probe = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(probe)
    bbox = draw.textbbox((0.0, 0.0), "88", font=font, anchor="mm")
    return -((float(bbox[1]) + float(bbox[3])) / 2.0)


def numbered_circle_marker_draw_center(
    center_xy: tuple[float, float],
    *,
    style: NumberedCircleMarkerStyle,
    image_size: tuple[int, int],
) -> tuple[float, float]:
    patch_radius = math.ceil(style.radius_px + math.ceil(style.outline_width_px + 3.0))
    return tuple(
        float(min(max(value, patch_radius), max(patch_radius, size - patch_radius - 1)))
        for value, size in zip(center_xy, image_size)
    )


def draw_numbered_circle_marker(
    image: Image.Image,
    center_xy: tuple[float, float],
    label: str,
    *,
    style: NumberedCircleMarkerStyle,
) -> None:
    if image.mode != "RGBA":
        raise ValueError("draw_numbered_circle_marker requires an RGBA image")
    scale = max(1, int(style.supersample))
    radius = float(style.radius_px)
    patch_margin = int(math.ceil(float(style.outline_width_px) + 3.0))
    patch_radius = int(math.ceil(radius + float(patch_margin)))
    patch_size = max(1, patch_radius * 2 + 1)
    cx, cy = numbered_circle_marker_draw_center(center_xy, style=style, image_size=image.size)
    high_size = int(patch_size * scale)
    high_radius = float(radius) * float(scale)
    high_center = float(high_size) / 2.0
    marker_patch = Image.new("RGBA", (high_size, high_size), (0, 0, 0, 0))
    marker_draw = ImageDraw.Draw(marker_patch, "RGBA")
    marker_draw.ellipse(
        (
            high_center - high_radius,
            high_center - high_radius,
            high_center + high_radius,
            high_center + high_radius,
        ),
        fill=style.fill,
        outline=style.outline,
        width=max(1, int(style.outline_width_px) * scale),
    )
    marker_patch = marker_patch.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
    left = int(round(cx - float(patch_size) / 2.0))
    top = int(round(cy - float(patch_size) / 2.0))
    image.alpha_composite(marker_patch, (left, top))

    text = str(label).strip()
    if text == "":
        return
    font = load_visual_font(int(style.font_size_px))
    text_draw = ImageDraw.Draw(image, "RGBA")
    text_draw.text(
        (cx, cy + _numbered_circle_text_y_offset(font)),
        text,
        fill=style.text_fill,
        font=font,
        anchor="mm",
    )

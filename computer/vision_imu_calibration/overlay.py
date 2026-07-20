from __future__ import annotations

from typing import Any

from .geometry import KneeAngleResult
from .models import LegLandmarks


VALID_COLOR = (70, 230, 90)
INVALID_COLOR = (70, 70, 245)
BADGE_BACKGROUND = (20, 26, 37)


def draw_leg_overlay(
    frame: Any,
    leg: LegLandmarks | None,
    angle: KneeAngleResult,
    *,
    inference_fps: float,
    source_description: str,
    recording_active: bool = False,
) -> Any:
    """Return the camera frame with only a compact tracking-status badge.

    Detailed pose geometry, angles, confidence, FPS, source, and recorder status
    remain available in the web GUI. Keeping the camera image visually clean makes
    body positioning and framing easier during acquisition.
    """
    del leg, inference_fps, source_description, recording_active
    cv2 = _load_cv2()
    output = frame.copy()

    status_text = "TRACKING VALID" if angle.valid else "TRACKING INVALID"
    status_color = VALID_COLOR if angle.valid else INVALID_COLOR
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.68
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        status_text,
        font,
        scale,
        thickness,
    )

    margin = 14
    horizontal_padding = 12
    vertical_padding = 9
    x1 = margin
    y1 = margin
    x2 = x1 + text_width + 2 * horizontal_padding
    y2 = y1 + text_height + baseline + 2 * vertical_padding

    translucent = output.copy()
    cv2.rectangle(translucent, (x1, y1), (x2, y2), BADGE_BACKGROUND, -1)
    cv2.addWeighted(translucent, 0.78, output, 0.22, 0.0, output)
    cv2.rectangle(output, (x1, y1), (x2, y2), status_color, 2)
    cv2.putText(
        output,
        status_text,
        (x1 + horizontal_padding, y2 - vertical_padding - baseline),
        font,
        scale,
        status_color,
        thickness,
        cv2.LINE_AA,
    )
    return output


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is not installed") from exc
    return cv2

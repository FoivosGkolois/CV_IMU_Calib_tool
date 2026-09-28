from __future__ import annotations

from typing import Any

from .marker_reference import MarkerDetection, MarkerReferenceResult, SHANK_TAG_ID, THIGH_TAG_ID


THIGH_COLOR = (255, 190, 0)
SHANK_COLOR = (255, 0, 210)
VALID_COLOR = (70, 230, 90)
INVALID_COLOR = (70, 70, 245)
BADGE_BACKGROUND = (20, 26, 37)


def draw_marker_overlay(
    frame: Any,
    result: MarkerReferenceResult,
    *,
    processing_fps: float,
    source_description: str,
    recording_active: bool = False,
) -> Any:
    """Draw only measured fiducial geometry and a compact reference status."""
    del processing_fps, source_description, recording_active
    cv2 = _load_cv2()
    output = frame.copy()
    if result.thigh is not None:
        _draw_tag(output, result.thigh, "THIGH / TAG 0", THIGH_COLOR)
    if result.shank is not None:
        _draw_tag(output, result.shank, "SHANK / TAG 1", SHANK_COLOR)
    if result.thigh is not None and result.shank is not None:
        a = tuple(int(round(v)) for v in result.thigh.center)
        b = tuple(int(round(v)) for v in result.shank.center)
        cv2.line(output, a, b, (170, 170, 170), 2, cv2.LINE_AA)
    _draw_reference_badge(output, result)
    return output


def _draw_tag(frame: Any, detection: MarkerDetection, label: str, color: tuple[int, int, int]) -> None:
    cv2 = _load_cv2()
    import numpy as np

    points = np.asarray(
        [[int(round(x)), int(round(y))] for x, y in detection.corners], dtype=np.int32
    ).reshape((-1, 1, 2))
    cv2.polylines(frame, [points], True, color, 4, cv2.LINE_AA)
    cx, cy = (int(round(v)) for v in detection.center)
    cv2.circle(frame, (cx, cy), 7, color, -1, cv2.LINE_AA)
    x0, y0 = points[0, 0]
    cv2.putText(
        frame,
        label,
        (int(x0), max(20, int(y0) - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_reference_badge(frame: Any, result: MarkerReferenceResult) -> None:
    cv2 = _load_cv2()
    if result.valid:
        if result.angle_deg is None:
            status_text = "REFERENCE VALID"
        else:
            status_text = f"REFERENCE VALID  {result.angle_deg:.1f} deg"
        status_color = VALID_COLOR
    elif result.thigh_detected and result.shank_detected:
        status_text = "TAGS FOUND - SETUP REQUIRED"
        status_color = (0, 210, 255)
    else:
        status_text = "REFERENCE INVALID"
        status_color = INVALID_COLOR

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(status_text, font, scale, thickness)
    margin = 12
    horizontal_padding = 10
    vertical_padding = 8
    x1 = margin
    y1 = margin
    x2 = x1 + text_width + 2 * horizontal_padding
    y2 = y1 + text_height + baseline + 2 * vertical_padding
    translucent = frame.copy()
    cv2.rectangle(translucent, (x1, y1), (x2, y2), BADGE_BACKGROUND, -1)
    cv2.addWeighted(translucent, 0.78, frame, 0.22, 0.0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
    cv2.putText(
        frame,
        status_text,
        (x1 + horizontal_padding, y2 - vertical_padding - baseline),
        font,
        scale,
        status_color,
        thickness,
        cv2.LINE_AA,
    )


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is not installed") from exc
    return cv2

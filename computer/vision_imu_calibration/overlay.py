from __future__ import annotations

import math
from typing import Any

from .geometry import KneeAngleResult
from .models import LegLandmarks
from .positioning import SAFE_AREA


THIGH_COLOR = (255, 190, 0)
SHANK_COLOR = (255, 0, 210)
JOINT_COLOR = (0, 240, 255)
ARC_COLOR = (70, 230, 90)
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
    hide_angle: bool = False,
    positioning: dict | None = None,
) -> Any:
    """Return the live camera frame with pose geometry and compact status.

    The overlay intentionally keeps the large information panel removed while
    restoring the acquisition-critical visual cues: thigh/shank bars, hip/knee/
    ankle markers, posterior knee-angle arc, live angle value, and tracking state.
    """
    del inference_fps, source_description, recording_active
    cv2 = _load_cv2()
    output = frame.copy()
    height, width = output.shape[:2]

    if leg is not None and angle.valid and angle.angle_deg is not None:
        hip = _pixel(leg.hip.x, leg.hip.y, width, height)
        knee = _pixel(leg.knee.x, leg.knee.y, width, height)
        ankle = _pixel(leg.ankle.x, leg.ankle.y, width, height)

        line_width = max(5, int(round(min(width, height) * 0.012)))
        joint_radius = max(6, int(round(min(width, height) * 0.014)))
        cv2.line(output, hip, knee, THIGH_COLOR, line_width, cv2.LINE_AA)
        cv2.line(output, knee, ankle, SHANK_COLOR, line_width, cv2.LINE_AA)
        _draw_knee_arc(output, hip, knee, ankle, angle, hide_angle=hide_angle)

        for point in (hip, knee, ankle):
            cv2.circle(output, point, joint_radius + 3, (10, 10, 10), -1, cv2.LINE_AA)
            cv2.circle(output, point, joint_radius, JOINT_COLOR, -1, cv2.LINE_AA)

        cv2.putText(
            output,
            leg.side.value.upper(),
            (knee[0] + joint_radius + 8, knee[1] - joint_radius - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            JOINT_COLOR,
            2,
            cv2.LINE_AA,
        )

    _draw_tracking_badge(output, angle)
    if positioning and positioning.get("enabled"):
        # Inference and guides share the already-rotated frame, independent of
        # the browser's object-fit scaling and landscape/portrait dimensions.
        color = (100, 160, 100) if positioning.get("state") == "PASSED" else (230, 180, 70)
        thickness = 1 if positioning.get("state") == "PASSED" else 2
        cv2.rectangle(output, _pixel(SAFE_AREA[0], SAFE_AREA[1], width, height),
                      _pixel(SAFE_AREA[2], SAFE_AREA[3], width, height), color, thickness)
        if positioning.get("warning"):
            cv2.putText(output, "POSITION WARNING", (12, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, INVALID_COLOR, 2, cv2.LINE_AA)
    return output


def _draw_tracking_badge(frame: Any, angle: KneeAngleResult) -> None:
    cv2 = _load_cv2()
    status_text = "TRACKING VALID" if angle.valid else "TRACKING INVALID"
    status_color = VALID_COLOR if angle.valid else INVALID_COLOR
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
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


def _draw_knee_arc(
    frame: Any,
    hip: tuple[int, int],
    knee: tuple[int, int],
    ankle: tuple[int, int],
    angle: KneeAngleResult,
    *, hide_angle: bool = False,
) -> None:
    cv2 = _load_cv2()
    thigh_length = math.hypot(hip[0] - knee[0], hip[1] - knee[1])
    shank_length = math.hypot(ankle[0] - knee[0], ankle[1] - knee[1])
    radius = int(max(24.0, min(thigh_length, shank_length) * 0.27))
    start_deg = _screen_vector_angle_deg(knee, hip)
    end_deg = _screen_vector_angle_deg(knee, ankle)
    sweep_deg = _shortest_signed_angle_deg(start_deg, end_deg)
    points = _sample_arc_points(knee, radius, start_deg, sweep_deg)
    if len(points) >= 2:
        cv2.polylines(frame, [points], False, ARC_COLOR, 4, cv2.LINE_AA)

    mid_deg = start_deg + 0.5 * sweep_deg
    label_radius = radius + 25
    label_position = (
        int(round(knee[0] + label_radius * math.cos(math.radians(mid_deg)))),
        int(round(knee[1] - label_radius * math.sin(math.radians(mid_deg)))),
    )
    if hide_angle:
        return
    cv2.putText(
        frame,
        f"{angle.angle_deg:.0f} deg",
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        ARC_COLOR,
        2,
        cv2.LINE_AA,
    )


def _sample_arc_points(
    center: tuple[int, int],
    radius: int,
    start_deg: float,
    sweep_deg: float,
) -> Any:
    import numpy as np

    count = max(16, int(abs(sweep_deg) / 4.0) + 2)
    samples = np.linspace(0.0, 1.0, count, dtype=np.float32)
    points: list[list[int]] = []
    for sample in samples:
        angle_deg = float(start_deg + sweep_deg * sample)
        angle_rad = math.radians(angle_deg)
        points.append(
            [
                int(round(center[0] + radius * math.cos(angle_rad))),
                int(round(center[1] - radius * math.sin(angle_rad))),
            ]
        )
    return np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))


def _screen_vector_angle_deg(origin: tuple[int, int], point: tuple[int, int]) -> float:
    return math.degrees(math.atan2(-(point[1] - origin[1]), point[0] - origin[0]))


def _shortest_signed_angle_deg(start_deg: float, end_deg: float) -> float:
    return (end_deg - start_deg + 180.0) % 360.0 - 180.0


def _pixel(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    return (
        int(round(max(0.0, min(1.0, x)) * (width - 1))),
        int(round(max(0.0, min(1.0, y)) * (height - 1))),
    )


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is not installed") from exc
    return cv2

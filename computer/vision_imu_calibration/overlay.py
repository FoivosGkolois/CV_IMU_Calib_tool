from __future__ import annotations

import math
from typing import Any

from .geometry import KneeAngleResult
from .models import LegLandmarks


THIGH_COLOR = (255, 190, 0)
SHANK_COLOR = (255, 0, 210)
JOINT_COLOR = (0, 240, 255)
ARC_COLOR = (70, 230, 90)
VALID_COLOR = (70, 230, 90)
INVALID_COLOR = (70, 70, 245)
PANEL_COLOR = (20, 26, 37)


def draw_leg_overlay(
    frame: Any,
    leg: LegLandmarks | None,
    angle: KneeAngleResult,
    *,
    inference_fps: float,
    source_description: str,
    recording_active: bool = False,
) -> Any:
    """Return an annotated copy of the camera frame."""
    cv2 = _load_cv2()
    output = frame.copy()
    height, width = output.shape[:2]

    if leg is not None:
        hip = _pixel(leg.hip.x, leg.hip.y, width, height)
        knee = _pixel(leg.knee.x, leg.knee.y, width, height)
        ankle = _pixel(leg.ankle.x, leg.ankle.y, width, height)
        cv2.line(output, hip, knee, THIGH_COLOR, 12, cv2.LINE_AA)
        cv2.line(output, knee, ankle, SHANK_COLOR, 12, cv2.LINE_AA)
        _draw_knee_arc(output, hip, knee, ankle, angle)
        for point in (hip, knee, ankle):
            cv2.circle(output, point, 13, (10, 10, 10), -1, cv2.LINE_AA)
            cv2.circle(output, point, 9, JOINT_COLOR, -1, cv2.LINE_AA)
        cv2.putText(
            output,
            leg.side.value.upper(),
            (knee[0] + 18, knee[1] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            JOINT_COLOR,
            2,
            cv2.LINE_AA,
        )

    panel_width = min(430, max(320, int(width * 0.34)))
    overlay = output.copy()
    cv2.rectangle(overlay, (18, 18), (18 + panel_width, 250), PANEL_COLOR, -1)
    cv2.addWeighted(overlay, 0.84, output, 0.16, 0.0, output)

    status_color = VALID_COLOR if angle.valid else INVALID_COLOR
    status_text = "TRACKING VALID" if angle.valid else f"TRACKING INVALID: {angle.reason}"
    angle_text = f"{angle.angle_deg:5.1f} deg" if angle.angle_deg is not None else "--.- deg"
    lines = [
        ("VISION-IMU · MOVENET THUNDER", (240, 245, 255), 0.72),
        (status_text, status_color, 0.58),
        (f"Posterior knee angle: {angle_text}", (240, 245, 255), 0.68),
        (f"Confidence: {angle.confidence:.2f}", (210, 220, 235), 0.54),
        (f"Inference: {inference_fps:.1f} FPS", (210, 220, 235), 0.54),
        (f"Source: {source_description}", (210, 220, 235), 0.46),
        ("REC ●" if recording_active else "Recorder: idle", (60, 80, 255) if recording_active else (210, 220, 235), 0.52),
    ]
    y = 50
    for label, color, scale in lines:
        cv2.putText(output, label, (38, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        y += 31
    return output


def _draw_knee_arc(frame: Any, hip: tuple[int, int], knee: tuple[int, int], ankle: tuple[int, int], angle: KneeAngleResult) -> None:
    if not angle.valid or angle.angle_deg is None:
        return
    cv2 = _load_cv2()
    thigh_len = math.hypot(hip[0] - knee[0], hip[1] - knee[1])
    shank_len = math.hypot(ankle[0] - knee[0], ankle[1] - knee[1])
    radius = int(max(24.0, min(thigh_len, shank_len) * 0.28))
    start_deg = _screen_vector_angle_deg(knee, hip)
    end_deg = _screen_vector_angle_deg(knee, ankle)
    sweep_deg = _shortest_signed_angle_deg(start_deg, end_deg)
    points = _sample_arc_points(knee, radius, start_deg, sweep_deg)
    if len(points) >= 2:
        cv2.polylines(frame, [points], False, ARC_COLOR, 5, cv2.LINE_AA)
    mid_deg = start_deg + 0.5 * sweep_deg
    label_radius = radius + 24
    label_pos = (
        int(round(knee[0] + label_radius * math.cos(math.radians(mid_deg)))),
        int(round(knee[1] - label_radius * math.sin(math.radians(mid_deg)))),
    )
    cv2.putText(
        frame,
        f"{angle.angle_deg:.0f}°",
        label_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        ARC_COLOR,
        2,
        cv2.LINE_AA,
    )


def _sample_arc_points(center: tuple[int, int], radius: int, start_deg: float, sweep_deg: float) -> Any:
    import numpy as np
    count = max(16, int(abs(sweep_deg) / 4.0) + 2)
    samples = np.linspace(0.0, 1.0, count, dtype=np.float32)
    points = []
    for sample in samples:
        angle_deg = float(start_deg + sweep_deg * sample)
        angle_rad = math.radians(angle_deg)
        points.append([
            int(round(center[0] + radius * math.cos(angle_rad))),
            int(round(center[1] - radius * math.sin(angle_rad))),
        ])
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

from __future__ import annotations

from typing import Any

from .geometry import KneeAngleResult
from .models import LegLandmarks


THIGH_COLOR = (255, 190, 0)
SHANK_COLOR = (255, 0, 210)
JOINT_COLOR = (0, 240, 255)
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
        (f"Knee flexion: {angle_text}", (240, 245, 255), 0.68),
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

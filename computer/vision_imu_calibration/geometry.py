from __future__ import annotations

import math
from dataclasses import dataclass

from .models import LegLandmarks


@dataclass(frozen=True)
class KneeAngleResult:
    angle_deg: float | None
    valid: bool
    confidence: float
    reason: str


def calculate_knee_flexion_deg(
    leg: LegLandmarks | None,
    *,
    confidence_threshold: float = 0.55,
    min_segment_length: float = 0.025,
) -> KneeAngleResult:
    """Calculate sagittal-plane knee flexion from normalized 2-D landmarks.

    The convention is 0 degrees for a straight leg and positive flexion.
    The result is an external visual reference, not a clinical 3-D angle.
    """
    if leg is None:
        return KneeAngleResult(None, False, 0.0, "pose_missing")

    confidence = float(leg.confidence)
    if confidence < confidence_threshold:
        return KneeAngleResult(None, False, confidence, "low_confidence")

    thigh = (leg.hip.x - leg.knee.x, leg.hip.y - leg.knee.y)
    shank = (leg.ankle.x - leg.knee.x, leg.ankle.y - leg.knee.y)
    thigh_length = math.hypot(*thigh)
    shank_length = math.hypot(*shank)
    if thigh_length < min_segment_length or shank_length < min_segment_length:
        return KneeAngleResult(None, False, confidence, "degenerate_segment")

    cosine = (thigh[0] * shank[0] + thigh[1] * shank[1]) / (thigh_length * shank_length)
    cosine = max(-1.0, min(1.0, cosine))
    included_angle_deg = math.degrees(math.acos(cosine))
    flexion_deg = max(0.0, min(180.0, 180.0 - included_angle_deg))
    return KneeAngleResult(flexion_deg, True, confidence, "ok")

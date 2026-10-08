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
    image_aspect_ratio: float = 1.0,
) -> KneeAngleResult:
    """Calculate the posterior knee angle from normalized 2-D landmarks.

    Convention:
    - 180 degrees: straight leg
    - smaller angle: more flexion

    The landmarks are normalized by image width and height separately. Scale
    horizontal differences by width / height to recover the image-plane angle.
    The default aspect ratio preserves the previous public API for callers
    that do not provide image dimensions.

    The result is an external visual reference, not a clinical 3-D angle.
    The function name is preserved for compatibility with the existing tool.
    """
    if leg is None:
        return KneeAngleResult(None, False, 0.0, "pose_missing")

    if not math.isfinite(image_aspect_ratio) or image_aspect_ratio <= 0.0:
        raise ValueError("image_aspect_ratio must be finite and positive")

    confidence = float(leg.confidence)
    if confidence < confidence_threshold:
        return KneeAngleResult(None, False, confidence, "low_confidence")

    thigh = (leg.hip.x - leg.knee.x, leg.hip.y - leg.knee.y)
    shank = (leg.ankle.x - leg.knee.x, leg.ankle.y - leg.knee.y)
    thigh_length = math.hypot(*thigh)
    shank_length = math.hypot(*shank)
    if thigh_length < min_segment_length or shank_length < min_segment_length:
        return KneeAngleResult(None, False, confidence, "degenerate_segment")

    # Apply the pixel aspect correction only after the existing segment-length
    # validity gate, preserving its normalized-coordinate threshold semantics.
    thigh = (thigh[0] * image_aspect_ratio, thigh[1])
    shank = (shank[0] * image_aspect_ratio, shank[1])
    thigh_length = math.hypot(*thigh)
    shank_length = math.hypot(*shank)
    cosine = (thigh[0] * shank[0] + thigh[1] * shank[1]) / (thigh_length * shank_length)
    cosine = max(-1.0, min(1.0, cosine))
    posterior_angle_deg = math.degrees(math.acos(cosine))
    posterior_angle_deg = max(0.0, min(180.0, posterior_angle_deg))
    return KneeAngleResult(posterior_angle_deg, True, confidence, "ok")

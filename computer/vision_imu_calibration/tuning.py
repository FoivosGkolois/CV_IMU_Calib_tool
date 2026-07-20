from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping

from .models import LegSide


class FrameRotation(str, Enum):
    NONE = "none"
    CLOCKWISE_90 = "clockwise_90"
    COUNTERCLOCKWISE_90 = "counterclockwise_90"
    ROTATE_180 = "rotate_180"


@dataclass(frozen=True)
class RuntimeTuning:
    """Live application-level tuning around the fixed pretrained pose model.

    These values do not retrain or alter the neural-network weights. They control
    frame orientation, accepted pose quality, landmark/angle filtering, and outlier
    rejection before display and recording.
    """

    leg_side: LegSide = LegSide.AUTO
    frame_rotation: FrameRotation = FrameRotation.COUNTERCLOCKWISE_90
    confidence_threshold: float = 0.30
    landmark_smoothing: float = 0.55
    angle_smoothing: float = 0.35
    min_segment_length: float = 0.025
    max_angle_jump_deg: float = 45.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["leg_side"] = self.leg_side.value
        payload["frame_rotation"] = self.frame_rotation.value
        return payload

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        current: "RuntimeTuning | None" = None,
    ) -> "RuntimeTuning":
        base = current or cls()
        leg_value = payload.get("leg_side", base.leg_side.value)
        try:
            leg_side = LegSide(str(leg_value))
        except ValueError as exc:
            raise ValueError("leg_side must be auto, left, or right") from exc

        rotation_value = payload.get("frame_rotation", base.frame_rotation.value)
        try:
            frame_rotation = FrameRotation(str(rotation_value))
        except ValueError as exc:
            raise ValueError(
                "frame_rotation must be none, clockwise_90, counterclockwise_90, or rotate_180"
            ) from exc

        confidence_threshold = _bounded_float(
            payload.get("confidence_threshold", base.confidence_threshold),
            "confidence_threshold",
            0.05,
            0.99,
        )
        landmark_smoothing = _bounded_float(
            payload.get("landmark_smoothing", base.landmark_smoothing),
            "landmark_smoothing",
            0.0,
            0.95,
        )
        angle_smoothing = _bounded_float(
            payload.get("angle_smoothing", base.angle_smoothing),
            "angle_smoothing",
            0.0,
            0.95,
        )
        min_segment_length = _bounded_float(
            payload.get("min_segment_length", base.min_segment_length),
            "min_segment_length",
            0.005,
            0.20,
        )
        max_angle_jump_deg = _bounded_float(
            payload.get("max_angle_jump_deg", base.max_angle_jump_deg),
            "max_angle_jump_deg",
            5.0,
            180.0,
        )
        return cls(
            leg_side=leg_side,
            frame_rotation=frame_rotation,
            confidence_threshold=confidence_threshold,
            landmark_smoothing=landmark_smoothing,
            angle_smoothing=angle_smoothing,
            min_segment_length=min_segment_length,
            max_angle_jump_deg=max_angle_jump_deg,
        )


def _bounded_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed

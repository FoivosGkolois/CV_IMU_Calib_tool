from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping


class FrameRotation(str, Enum):
    NONE = "none"
    CLOCKWISE_90 = "clockwise_90"
    COUNTERCLOCKWISE_90 = "counterclockwise_90"
    ROTATE_180 = "rotate_180"


@dataclass(frozen=True)
class RuntimeTuning:
    """Non-scientific camera presentation settings.

    The previous pose-estimation quality/smoothing knobs were removed. The tag
    reference has no user-tuned acceptance or smoothing parameters.
    """

    frame_rotation: FrameRotation = FrameRotation.COUNTERCLOCKWISE_90

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
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
        rotation_value = payload.get("frame_rotation", base.frame_rotation.value)
        try:
            frame_rotation = FrameRotation(str(rotation_value))
        except ValueError as exc:
            raise ValueError(
                "frame_rotation must be none, clockwise_90, counterclockwise_90, or rotate_180"
            ) from exc
        return cls(frame_rotation=frame_rotation)

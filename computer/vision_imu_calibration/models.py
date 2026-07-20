from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class LegSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    AUTO = "auto"


@dataclass(frozen=True)
class Landmark:
    """One normalized image landmark returned by the pose backend."""

    x: float
    y: float
    visibility: float
    presence: float

    @property
    def confidence(self) -> float:
        return max(0.0, min(1.0, min(float(self.visibility), float(self.presence))))


@dataclass(frozen=True)
class LegLandmarks:
    side: LegSide
    hip: Landmark
    knee: Landmark
    ankle: Landmark

    @property
    def confidence(self) -> float:
        return min(self.hip.confidence, self.knee.confidence, self.ankle.confidence)


@dataclass(frozen=True)
class PoseFrame:
    """Pose output normalized into an implementation-independent contract."""

    timestamp_ms: int
    landmarks: Mapping[str, Landmark]

    def leg(self, requested_side: LegSide) -> LegLandmarks | None:
        if requested_side is LegSide.AUTO:
            candidates = [
                self._leg_for_side(LegSide.LEFT),
                self._leg_for_side(LegSide.RIGHT),
            ]
            valid = [candidate for candidate in candidates if candidate is not None]
            return max(valid, key=lambda candidate: candidate.confidence, default=None)
        return self._leg_for_side(requested_side)

    def _leg_for_side(self, side: LegSide) -> LegLandmarks | None:
        prefix = side.value
        try:
            return LegLandmarks(
                side=side,
                hip=self.landmarks[f"{prefix}_hip"],
                knee=self.landmarks[f"{prefix}_knee"],
                ankle=self.landmarks[f"{prefix}_ankle"],
            )
        except KeyError:
            return None

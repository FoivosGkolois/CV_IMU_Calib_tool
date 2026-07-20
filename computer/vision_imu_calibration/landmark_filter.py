from __future__ import annotations

from dataclasses import replace

from .models import Landmark, LegLandmarks


class LegLandmarkEmaFilter:
    """Exponential moving-average filter for one tracked leg.

    The filter smooths only normalized image coordinates. Confidence remains the
    current model confidence, so filtering cannot turn an uncertain landmark into
    an accepted one. State resets when the tracked side changes or tracking is lost.
    """

    def __init__(self) -> None:
        self._previous: LegLandmarks | None = None

    def reset(self) -> None:
        self._previous = None

    def apply(self, leg: LegLandmarks | None, *, smoothing: float) -> LegLandmarks | None:
        if leg is None:
            self.reset()
            return None
        strength = max(0.0, min(0.95, float(smoothing)))
        previous = self._previous
        if previous is None or previous.side is not leg.side or strength <= 0.0:
            self._previous = leg
            return leg

        filtered = LegLandmarks(
            side=leg.side,
            hip=_smooth_landmark(previous.hip, leg.hip, strength),
            knee=_smooth_landmark(previous.knee, leg.knee, strength),
            ankle=_smooth_landmark(previous.ankle, leg.ankle, strength),
        )
        self._previous = filtered
        return filtered


def _smooth_landmark(previous: Landmark, current: Landmark, strength: float) -> Landmark:
    current_weight = 1.0 - strength
    return replace(
        current,
        x=strength * previous.x + current_weight * current.x,
        y=strength * previous.y + current_weight * current.y,
    )

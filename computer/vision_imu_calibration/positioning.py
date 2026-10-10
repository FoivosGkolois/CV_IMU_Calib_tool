"""Short tracking check on fresh RAW landmarks in the rotated camera frame.

The check verifies full-leg visibility, confidence, geometry, fresh timestamps and
slow knee movement over multiple frames. It cannot infer anatomical side-on pose.
"""
from __future__ import annotations

import math

from .geometry import KneeAngleResult
from .models import LegLandmarks


SAFE_AREA = (0.05, 0.05, 0.95, 0.95)
CHECK_DURATION_S = 3.0
MAX_CHECK_DURATION_S = 8.0
MAX_FRESH_GAP_S = 0.75
MIN_VALID_FRAMES = 15
MIN_MOTION_RANGE_DEG = 8.0
MAX_SLOW_SPEED_DEG_S = 120.0


def visibility_reason(leg: LegLandmarks | None, angle: KneeAngleResult,
                      confidence_threshold: float) -> str:
    if leg is None or not angle.valid or angle.angle_deg is None:
        return "Tracking invalid: " + angle.reason
    if not math.isfinite(angle.angle_deg) or not 0 <= angle.angle_deg <= 180:
        return "Nonfinite / out-of-range angle"
    for name in ("hip", "knee", "ankle"):
        point = getattr(leg, name)
        if not all(math.isfinite(v) for v in
                   (point.x, point.y, point.visibility, point.presence)):
            return f"{name.title()} is not a finite observation"
        if point.confidence < confidence_threshold:
            return f"{name.title()} confidence too low"
        if not (SAFE_AREA[0] <= point.x <= SAFE_AREA[2]
                and SAFE_AREA[1] <= point.y <= SAFE_AREA[3]):
            return f"{name.title()} outside safe area; move away from frame edge"
    return ""


class PositioningCheck:
    """Deterministic state; caller supplies time, observations and context."""

    def __init__(self) -> None:
        self.context: tuple | None = None
        self.enabled = False
        self.reset("Position Subject to begin")

    def reset(self, reason: str) -> None:
        self.state = "READY"
        self.reason = reason
        self.warning = ""
        self.manual_side_on = False
        self.count = 0
        self.first_time: float | None = None
        self.last_time: float | None = None
        self.minimum = math.inf
        self.maximum = -math.inf
        self.total_motion = 0.0
        self.directions: set[int] = set()
        self.previous_angle: float | None = None
        self.previous_angle_time: float | None = None
        self.failure = ""
        self.latest_issue = ""

    def set_context(self, context: tuple) -> None:
        if self.context != context:
            self.context = context
            self.reset("Setup/tracked leg changed; repeat positioning check")

    def begin(self, manual_side_on: bool = False) -> None:
        self.enabled = True
        self.reset("Checking fresh frames; bend and straighten the knee slowly")
        # Retained only for API compatibility. A 2-D tracker cannot prove side-on pose.
        self.manual_side_on = False
        self.state = "CHECKING"

    def observe(self, seconds: float, leg: LegLandmarks | None,
                angle: KneeAngleResult, confidence_threshold: float) -> None:
        if not self.enabled:
            return
        reason = visibility_reason(leg, angle, confidence_threshold)
        if self.state == "PASSED":
            self.warning = reason
            return
        if self.state != "CHECKING":
            return
        if not math.isfinite(seconds):
            self.latest_issue = "Invalid sample timestamp"
            return
        if self.last_time is not None and seconds <= self.last_time:
            self.latest_issue = "Stale or duplicate camera timestamp"
            return
        if self.first_time is None:
            self.first_time = seconds
        gap = self.last_time is not None and seconds - self.last_time > MAX_FRESH_GAP_S
        self.last_time = seconds
        if gap:
            self.latest_issue = "Brief camera timestamp gap"
            self.previous_angle = None
            self.previous_angle_time = None
            self._evaluate(seconds, allow_pass=False)
            return
        if reason:
            self.latest_issue = reason
            self.previous_angle = None
            self.previous_angle_time = None
            self._evaluate(seconds, allow_pass=False)
            return

        value = float(angle.angle_deg)
        delta: float | None = None
        if self.previous_angle is not None and self.previous_angle_time is not None:
            delta = value - self.previous_angle
            speed = abs(delta) / (seconds - self.previous_angle_time)
            if speed > MAX_SLOW_SPEED_DEG_S:
                self.latest_issue = "Isolated implausible knee-speed sample"
                self._evaluate(seconds, allow_pass=False)
                return

        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        if delta is not None:
            self.total_motion += abs(delta)
            if abs(delta) >= 0.5:
                self.directions.add(1 if delta > 0 else -1)
        self.previous_angle = value
        self.previous_angle_time = seconds
        self.count += 1
        self.latest_issue = ""
        self._evaluate(seconds, allow_pass=True)

    def finish(self, seconds: float) -> dict:
        if self.state in {"PASSED", "RETRY"}:
            return self.snapshot()
        if self.state != "CHECKING":
            raise RuntimeError("Start a positioning check first")
        self._evaluate(seconds, allow_pass=True, force=True)
        return self.snapshot()

    def _retry_reason(self, elapsed_s: float) -> str:
        if elapsed_s < CHECK_DURATION_S:
            return "Observe for at least 3 seconds, then retry"
        if self.count < MIN_VALID_FRAMES:
            return "Not enough valid fresh frames; keep the full leg visible and retry"
        observed_range = self.maximum - self.minimum
        if not math.isfinite(observed_range) or observed_range < MIN_MOTION_RANGE_DEG:
            return "Not enough knee motion; bend and straighten slowly, then retry"
        if len(self.directions) < 2:
            return "Motion must include a slow bend and return"
        return ""

    def _evaluate(self, seconds: float, *, allow_pass: bool, force: bool = False) -> None:
        if not math.isfinite(seconds):
            if force:
                self.state = "RETRY"
                self.reason = "Invalid finish timestamp; retry"
            return
        elapsed_s = seconds - self.first_time if self.first_time is not None else 0.0
        reason = self._retry_reason(elapsed_s)
        if allow_pass and not reason:
            self.state = "PASSED"
            self.reason = (
                "Tracking check passed: fresh full-leg frames and slow knee motion observed; "
                "side-on orientation remains the operator's responsibility"
            )
            return
        if force or elapsed_s >= MAX_CHECK_DURATION_S:
            self.state = "RETRY"
            self.reason = reason or "Tracking was not consistently valid; retry"

    def snapshot(self) -> dict:
        return {"enabled": self.enabled, "state": self.state, "reason": self.reason,
                "warning": self.warning, "manual_side_on": self.manual_side_on,
                "frames": self.count, "safe_area": SAFE_AREA,
                "observed_min_deg": self.minimum if math.isfinite(self.minimum) else None,
                "observed_max_deg": self.maximum if math.isfinite(self.maximum) else None,
                "total_motion_deg": self.total_motion}

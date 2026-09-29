from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .final_protocol import (
    FINAL_PROTOCOL_VERSION,
    REFERENCE_VALIDATION_CAPTURE_WINDOW_S,
    REFERENCE_VALIDATION_MAX_PER_ANGLE_MAE_DEG,
    REFERENCE_VALIDATION_MAX_REPEATABILITY_SD_DEG,
    REFERENCE_VALIDATION_MAX_RMSE_DEG,
    REFERENCE_VALIDATION_MIN_VALID_FRAMES,
    REFERENCE_VALIDATION_REPEATS_PER_ANGLE,
    REFERENCE_VALIDATION_TARGET_ANGLES_DEG,
)


@dataclass(frozen=True)
class ReferenceValidationMetrics:
    mean_error_deg: float
    mae_deg: float
    rmse_deg: float
    max_abs_error_deg: float
    max_per_angle_mae_deg: float
    max_repeatability_sd_deg: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class KnownAngleReferenceValidationSession:
    """One-time validation of the complete camera + two-tag angle reference.

    The user places the same rigid tag plates on a known-angle hinged fixture and
    captures five short measurements at each predefined angle. Each capture uses
    the median of the last one second of valid marker-reference frames.
    """

    def __init__(self) -> None:
        self._state = "IDLE"
        self._message = "Run this once after camera calibration using the rigid known-angle fixture."
        self._captures: dict[float, list[float]] = {
            float(angle): [] for angle in REFERENCE_VALIDATION_TARGET_ANGLES_DEG
        }
        self._result: dict[str, Any] | None = None

    @property
    def active(self) -> bool:
        return self._state == "ACTIVE"

    def start(self) -> dict[str, Any]:
        self._captures = {float(angle): [] for angle in REFERENCE_VALIDATION_TARGET_ANGLES_DEG}
        self._state = "ACTIVE"
        self._result = None
        self._message = self._instruction_message()
        return self.status()

    def cancel(self) -> dict[str, Any]:
        self._state = "IDLE"
        self._result = None
        self._message = "Known-angle reference validation cancelled."
        return self.status()

    def capture(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not self.active:
            raise RuntimeError("Start known-angle reference validation first")
        target = self._current_target()
        if target is None:
            raise RuntimeError("Known-angle reference validation is already complete")
        measured = _capture_window_median(rows)
        self._captures[target].append(measured)

        if self._current_target() is None:
            self._result = self._finalize()
            self._state = "COMPLETE" if self._result["metrics"]["passed"] else "FAILED"
            metrics = self._result["metrics"]
            if metrics["passed"]:
                self._message = (
                    f"Reference validation passed: RMSE {metrics['rmse_deg']:.2f} deg, "
                    f"repeatability SD <= {metrics['max_repeatability_sd_deg']:.2f} deg."
                )
            else:
                self._message = (
                    f"Reference validation did not pass: RMSE {metrics['rmse_deg']:.2f} deg, "
                    f"worst-angle MAE {metrics['max_per_angle_mae_deg']:.2f} deg, "
                    f"max repeatability SD {metrics['max_repeatability_sd_deg']:.2f} deg. "
                    "Check tag rigidity/alignment, camera setup and viewing geometry, then repeat."
                )
        else:
            self._message = self._instruction_message()
        return self.status()

    def status(self) -> dict[str, Any]:
        current = self._current_target()
        total_required = len(REFERENCE_VALIDATION_TARGET_ANGLES_DEG) * REFERENCE_VALIDATION_REPEATS_PER_ANGLE
        completed = sum(len(values) for values in self._captures.values())
        current_done = 0 if current is None else len(self._captures[current])
        return {
            "state": self._state,
            "active": self.active,
            "message": self._message,
            "protocol_version": FINAL_PROTOCOL_VERSION,
            "target_angles_deg": list(REFERENCE_VALIDATION_TARGET_ANGLES_DEG),
            "repeats_per_angle": REFERENCE_VALIDATION_REPEATS_PER_ANGLE,
            "capture_window_s": REFERENCE_VALIDATION_CAPTURE_WINDOW_S,
            "min_valid_frames_per_capture": REFERENCE_VALIDATION_MIN_VALID_FRAMES,
            "current_target_deg": current,
            "current_target_capture_count": current_done,
            "completed_captures": completed,
            "required_captures": total_required,
            "progress": completed / total_required if total_required else 0.0,
            "captures_deg": {f"{angle:.0f}": list(values) for angle, values in self._captures.items()},
            "result": self._result,
            "acceptance": {
                "max_rmse_deg": REFERENCE_VALIDATION_MAX_RMSE_DEG,
                "max_per_angle_mae_deg": REFERENCE_VALIDATION_MAX_PER_ANGLE_MAE_DEG,
                "max_repeatability_sd_deg": REFERENCE_VALIDATION_MAX_REPEATABILITY_SD_DEG,
            },
        }

    def result(self) -> dict[str, Any] | None:
        return None if self._result is None else dict(self._result)

    def _current_target(self) -> float | None:
        for angle in REFERENCE_VALIDATION_TARGET_ANGLES_DEG:
            key = float(angle)
            if len(self._captures[key]) < REFERENCE_VALIDATION_REPEATS_PER_ANGLE:
                return key
        return None

    def _instruction_message(self) -> str:
        target = self._current_target()
        if target is None:
            return "Known-angle reference validation complete."
        done = len(self._captures[target])
        return (
            f"Set the rigid fixture to {target:.0f} deg and press Capture "
            f"({done + 1}/{REFERENCE_VALIDATION_REPEATS_PER_ANGLE}). Keep both tags still and visible."
        )

    def _finalize(self) -> dict[str, Any]:
        errors: list[float] = []
        per_angle: list[dict[str, Any]] = []
        repeatability_sds: list[float] = []
        per_angle_maes: list[float] = []
        for target in REFERENCE_VALIDATION_TARGET_ANGLES_DEG:
            measurements = self._captures[float(target)]
            target_errors = [float(value) - float(target) for value in measurements]
            errors.extend(target_errors)
            sd = statistics.pstdev(measurements) if len(measurements) > 1 else 0.0
            repeatability_sds.append(sd)
            angle_mae = statistics.fmean(abs(value) for value in target_errors)
            per_angle_maes.append(angle_mae)
            per_angle.append(
                {
                    "target_angle_deg": float(target),
                    "measurements_deg": [float(value) for value in measurements],
                    "mean_measured_deg": statistics.fmean(measurements),
                    "mean_error_deg": statistics.fmean(target_errors),
                    "mae_deg": angle_mae,
                    "repeatability_sd_deg": sd,
                }
            )
        rmse = math.sqrt(statistics.fmean(value * value for value in errors))
        max_repeatability = max(repeatability_sds) if repeatability_sds else math.inf
        metrics = ReferenceValidationMetrics(
            mean_error_deg=statistics.fmean(errors),
            mae_deg=statistics.fmean(abs(value) for value in errors),
            rmse_deg=rmse,
            max_abs_error_deg=max(abs(value) for value in errors),
            max_per_angle_mae_deg=max(per_angle_maes) if per_angle_maes else math.inf,
            max_repeatability_sd_deg=max_repeatability,
            passed=(
                rmse <= REFERENCE_VALIDATION_MAX_RMSE_DEG
                and (max(per_angle_maes) if per_angle_maes else math.inf)
                <= REFERENCE_VALIDATION_MAX_PER_ANGLE_MAE_DEG
                and max_repeatability <= REFERENCE_VALIDATION_MAX_REPEATABILITY_SD_DEG
            ),
        )
        return {
            "status": "passed" if metrics.passed else "failed",
            "validated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "protocol_version": FINAL_PROTOCOL_VERSION,
            "targets_deg": list(REFERENCE_VALIDATION_TARGET_ANGLES_DEG),
            "repeats_per_angle": REFERENCE_VALIDATION_REPEATS_PER_ANGLE,
            "capture_window_s": REFERENCE_VALIDATION_CAPTURE_WINDOW_S,
            "per_angle": per_angle,
            "metrics": metrics.to_dict(),
            "acceptance": {
                "max_rmse_deg": REFERENCE_VALIDATION_MAX_RMSE_DEG,
                "max_per_angle_mae_deg": REFERENCE_VALIDATION_MAX_PER_ANGLE_MAE_DEG,
                "max_repeatability_sd_deg": REFERENCE_VALIDATION_MAX_REPEATABILITY_SD_DEG,
            },
        }


def _capture_window_median(rows: Sequence[Mapping[str, Any]]) -> float:
    valid = []
    latest_ns = 0
    for row in rows:
        try:
            host_ns = int(row.get("host_monotonic_ns", 0) or 0)
        except (TypeError, ValueError):
            continue
        latest_ns = max(latest_ns, host_ns)
    cutoff_ns = latest_ns - int(REFERENCE_VALIDATION_CAPTURE_WINDOW_S * 1_000_000_000)
    for row in rows:
        if not bool(row.get("visual_valid")):
            continue
        try:
            host_ns = int(row.get("host_monotonic_ns", 0) or 0)
            angle = float(row.get("visual_knee_angle_deg"))
        except (TypeError, ValueError):
            continue
        if host_ns < cutoff_ns or not math.isfinite(angle):
            continue
        valid.append(angle)
    if len(valid) < REFERENCE_VALIDATION_MIN_VALID_FRAMES:
        raise RuntimeError(
            "Not enough stable valid tag-reference frames in the last second. "
            "Hold the fixture still with both tags visible, then capture again."
        )
    return float(statistics.median(valid))

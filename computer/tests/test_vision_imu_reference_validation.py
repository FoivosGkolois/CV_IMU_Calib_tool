from __future__ import annotations

from computer.vision_imu_calibration.final_protocol import (
    REFERENCE_VALIDATION_REPEATS_PER_ANGLE,
    REFERENCE_VALIDATION_TARGET_ANGLES_DEG,
)
from computer.vision_imu_calibration.reference_validation import (
    KnownAngleReferenceValidationSession,
)


def _rows(angle_deg: float, *, start_ns: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    # 25 valid frames over 1.0 s; enough margin above the frozen minimum of 15.
    for index in range(25):
        rows.append(
            {
                "host_monotonic_ns": start_ns + index * 40_000_000,
                "visual_valid": True,
                "visual_knee_angle_deg": angle_deg + (0.15 if index % 2 else -0.15),
            }
        )
    return rows


def test_known_angle_reference_validation_passes_and_reports_metrics() -> None:
    session = KnownAngleReferenceValidationSession()
    session.start()
    capture_index = 0
    for target in REFERENCE_VALIDATION_TARGET_ANGLES_DEG:
        for _ in range(REFERENCE_VALIDATION_REPEATS_PER_ANGLE):
            capture_index += 1
            session.capture(_rows(float(target), start_ns=capture_index * 2_000_000_000))

    status = session.status()
    assert status["state"] == "COMPLETE"
    assert status["completed_captures"] == 20
    assert status["result"]["metrics"]["passed"] is True
    assert status["result"]["metrics"]["rmse_deg"] < 0.5
    assert status["result"]["metrics"]["max_repeatability_sd_deg"] < 0.5


def test_known_angle_capture_rejects_insufficient_valid_frames() -> None:
    session = KnownAngleReferenceValidationSession()
    session.start()
    rows = _rows(180.0, start_ns=1_000_000_000)[:5]
    try:
        session.capture(rows)
    except RuntimeError as exc:
        assert "Not enough stable valid" in str(exc)
    else:
        raise AssertionError("insufficient reference frames should be rejected")


def test_known_angle_reference_validation_rejects_one_bad_angle_even_if_global_rmse_is_below_limit() -> None:
    session = KnownAngleReferenceValidationSession()
    session.start()
    capture_index = 0
    for target in REFERENCE_VALIDATION_TARGET_ANGLES_DEG:
        measured = float(target) + (6.0 if float(target) == 120.0 else 0.0)
        for _ in range(REFERENCE_VALIDATION_REPEATS_PER_ANGLE):
            capture_index += 1
            session.capture(_rows(measured, start_ns=capture_index * 2_000_000_000))

    status = session.status()
    assert status["state"] == "FAILED"
    assert status["result"]["metrics"]["rmse_deg"] < 5.0
    assert status["result"]["metrics"]["max_per_angle_mae_deg"] > 5.0
    assert status["result"]["metrics"]["passed"] is False

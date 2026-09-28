from __future__ import annotations

import math

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from computer.vision_imu_calibration.camera_calibration import (
    CheckerboardCalibrationSession,
    CheckerboardSpecification,
    calibrate_checkerboard_views,
)


def _synthetic_views():
    spec = CheckerboardSpecification()
    obj = np.zeros((spec.corner_count, 3), np.float32)
    obj[:, :2] = (
        np.mgrid[0 : spec.inner_corners_x, 0 : spec.inner_corners_y]
        .T.reshape(-1, 2)
        .astype(np.float32)
        * spec.square_size_m
    )
    k = np.array([[780.0, 0.0, 320.0], [0.0, 795.0, 240.0], [0.0, 0.0, 1.0]])
    dist = np.zeros(5)
    views = []
    for i in range(15):
        ax = -0.22 + 0.035 * (i % 5)
        ay = -0.16 + 0.045 * (i // 5)
        az = -0.10 + 0.05 * (i % 4)
        rvec = np.array([ax, ay, az], dtype=float)
        tvec = np.array(
            [-0.10 + 0.045 * (i % 5), -0.07 + 0.05 * (i // 5), 0.72 + 0.025 * (i % 3)],
            dtype=float,
        )
        projected, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
        views.append(projected.astype(np.float32))
    return views


def test_checkerboard_solver_recovers_valid_intrinsics_from_clean_views() -> None:
    result = calibrate_checkerboard_views(_synthetic_views(), image_size=(640, 480))
    cal = result.calibration
    assert result.accepted_views == 15
    assert result.rms_reprojection_error_px < 0.05
    assert math.isclose(cal.camera_matrix[0][0], 780.0, rel_tol=0.03)
    assert math.isclose(cal.camera_matrix[1][1], 795.0, rel_tol=0.03)
    assert cal.image_width == 640
    assert cal.image_height == 480


def test_checkerboard_solver_rejects_too_few_views() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        calibrate_checkerboard_views(_synthetic_views()[:7], image_size=(640, 480))


def test_checkerboard_session_exposes_fixed_non_persistent_contract() -> None:
    session = CheckerboardCalibrationSession(required_views=15)
    started = session.start()
    assert started["active"] is True
    assert started["required_views"] == 15
    assert started["specification"]["inner_corners_x"] == 9
    assert started["specification"]["inner_corners_y"] == 6
    assert started["specification"]["square_size_mm"] == 25.0
    assert started["persistent"] is False
    cancelled = session.cancel()
    assert cancelled["active"] is False
    assert cancelled["state"] == "IDLE"

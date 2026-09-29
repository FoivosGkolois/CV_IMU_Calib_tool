from __future__ import annotations

"""Frozen user-facing protocol for the final thesis calibration tool.

These constants are deliberately centralized so the final experimental protocol
cannot drift between the GUI, offline analysis, and reports.
"""

FINAL_PROTOCOL_VERSION = "imu_calib_final_v1"

# Static stage: 12 s at ~100 Hz provides roughly 1200 samples for robust bias/noise
# statistics while remaining short enough for a comfortable stationary hold.
STATIC_TEST_DURATION_S = 12.0

# Dynamic stage: 12 accepted complete cycles, ordered as 4 deliberately slow,
# 4 comfortable/normal, and 4 brisk cycles. The fourth cycle in each block is
# held out, giving 9 training cycles and 3 independent validation cycles.
DYNAMIC_TEST_DURATION_S = 60.0
DYNAMIC_PROTOCOL_CYCLES = 12
DYNAMIC_SPEED_BLOCK_SIZE = 4
DYNAMIC_SPEED_BLOCKS = ("slow", "normal", "brisk")
HELD_OUT_CYCLE_INDICES = (4, 8, 12)

# One-time marker-reference validation. These posterior knee-angle targets cover
# straight extension through 90 degrees of flexion, which spans the intended
# calibration range of the knee-assist prototype.
REFERENCE_VALIDATION_TARGET_ANGLES_DEG = (180.0, 150.0, 120.0, 90.0)
REFERENCE_VALIDATION_REPEATS_PER_ANGLE = 5
REFERENCE_VALIDATION_CAPTURE_WINDOW_S = 1.0
REFERENCE_VALIDATION_MIN_VALID_FRAMES = 15

# Acceptance gates. 5 deg is used as a conservative upper bound for acceptable
# joint-angle agreement in gait/motion-analysis literature. A <=2 deg within-angle
# repeatability SD keeps the reference repeatability materially tighter than that
# accuracy bound.
REFERENCE_VALIDATION_MAX_RMSE_DEG = 5.0
REFERENCE_VALIDATION_MAX_PER_ANGLE_MAE_DEG = 5.0
REFERENCE_VALIDATION_MAX_REPEATABILITY_SD_DEG = 2.0

# Human-readable protocol note stored with final outputs.
PROTOCOL_SUMMARY = (
    "Static: 12 s fully extended and still. Dynamic: 12 complete cycles in order "
    "(4 slow, 4 normal, 4 brisk); cycles 4, 8, and 12 are held out. Reference "
    "validation: 180/150/120/90 deg, 5 captures per angle."
)

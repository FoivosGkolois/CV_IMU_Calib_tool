# Final calibration protocol and rationale

Protocol ID: `imu_calib_final_v1`

This document freezes the final thesis protocol. It distinguishes literature-supported choices from engineering choices made specifically for this prototype. The numbers below must not be changed per trial after final experiments begin.

## 1. One-time camera calibration

**Frozen protocol**
- Printed planar checkerboard: 9 x 6 inner corners, 25 mm squares.
- 15 automatically accepted views.
- Views must cover different positions, distances and tilts.
- Same physical camera, lens, resolution and frame rotation must then be used.

**Rationale**
OpenCV documents that a chessboard calibration theoretically needs only a small number of views, but in practical noisy images recommends at least 10 good snapshots at different positions. Fifteen accepted views gives margin above that practical minimum without making setup burdensome. The existing tool also rejects near-duplicate views, so the count represents geometrically diverse observations rather than repeated copies of the same pose.

## 2. One-time validation of the complete camera + two-tag reference

**Frozen protocol**
- Use the same rigid Tag 0 and Tag 1 plates that will be used on the exoskeleton.
- Place them on a rigid two-link hinged fixture/goniometer with the same proximal orientation convention used on the body.
- Tested posterior knee angles: 180, 150, 120 and 90 deg.
- Five independent captures at each angle: 20 captures total.
- Each capture is the median of the last 1.0 s of valid tag frames and requires at least 15 valid frames.
- Pass only if:
  - overall RMSE <= 5 deg,
  - MAE at every tested angle <= 5 deg,
  - within-angle repeatability SD <= 2 deg.

**Rationale**
The four target angles span the intended 90 deg operating range in uniform 30 deg intervals and include full extension. Five repeats per condition are a conventional small repeatability sample used in biomechanical validation work; the point is to expose repeatability rather than rely on one frame. Published AprilTag-based motion work has reported orientation error around 1.8 deg for a 58 mm marker in a representative camera setup, but marker accuracy depends strongly on the actual viewing geometry. Therefore this thesis measures the error of its own exact setup rather than borrowing a published accuracy value. Motion-analysis literature commonly treats errors below 2 deg as very good and 2–5 deg as contextually acceptable, while errors above about 5 deg should be avoided for direct joint-angle interpretation. The 5 deg accuracy gate is therefore a conservative upper acceptance bound, while the 2 deg repeatability gate requires the reference itself to be stable.

This is a validation, not a correction fit: the tool does not tune the tag result to force agreement with the fixture. If the setup fails, mounting/viewing geometry must be fixed and the validation repeated.

## 3. Static IMU calibration

**Frozen protocol**
- 12 s.
- Leg fully extended, defined as posterior knee angle 180 deg.
- Remain still.

**Rationale**
At the ~100 Hz acquisition rate this supplies about 1200 stationary samples, which is ample for robust estimates of gyro bias and short-term noise while remaining a short hold. The 180 deg condition is a physical constraint, so the straight-leg offset is derived without fitting against the filter that will later be optimized.

## 4. Dynamic calibration movement

**Frozen protocol**
- Up to 60 s recording time; the user may stop after completing the protocol.
- Exactly the first 12 accepted complete flexion-extension cycles are analyzed.
- Order:
  - cycles 1–4: deliberately slow,
  - cycles 5–8: comfortable/normal,
  - cycles 9–12: brisk but controlled.
- The tool checks that median cycle duration decreases from slow -> normal -> brisk.
- Cycles 4, 8 and 12 are held out and never used to choose filter settings.
- Result: 9 tuning cycles + 3 held-out validation cycles, with one held-out cycle from every speed block.

**Rationale**
There is no published paper that can prescribe an exact cycle count for this unique calibration tool. The count is therefore an engineering synthesis, not a claimed biological constant. Recent gait/biomechanics work reports that roughly 7 strides can be sufficient for reliable kinematics and that about 10 cycles can be required for reliable inter-joint coordination across running speeds. Twelve complete cycles places the acquisition above those practical ranges while remaining a short calibration task. It also divides cleanly into three motion-speed regimes and permits a balanced held-out cycle from each regime.

The speed variation is deliberate: IMU fusion error can change under more dynamic motion because linear acceleration makes accelerometer-based orientation less reliable. The filter should therefore not be tuned only on one slow, easy motion regime.

The 60 s limit is operational headroom, not a scientific threshold. It avoids forcing the participant to rush the slow block and allows short transitions between blocks.

## 5. Optimization and validation

**Frozen protocol**
- Optimize only `alpha_stable`, `alpha_dynamic`, and `max_accel_correction_deg`.
- Primary objective: synchronized, unshifted training RMSE in degrees.
- No candidate-specific time shifting.
- Bayesian Optimization budget: 30 candidate evaluations.
- Same evaluation budget for random-search and budgeted-grid engineering comparators.
- Final recommendation is evaluated only after selection on the three held-out cycles.

**Rationale**
The three search variables are genuine filter-design parameters; quantities that can be measured in the static stage are measured instead of optimized. RMSE remains in physical units and avoids arbitrary weighted objective coefficients. Thirty evaluations is a fixed engineering compute budget for this low-dimensional search, not a literature-derived physiological constant. Its scientific value comes from freezing the same budget for all compared search methods before the final experiments.

## 6. Uncertainty wording

The final report separates:
- GP surrogate uncertainty: uncertainty of the optimizer's model of the search space;
- marker-reference validation error: measured error/repeatability of the camera + tag reference against the known-angle fixture;
- held-out performance variation: performance on cycles that were not used for parameter selection.

None of these should be described as absolute clinical ground-truth uncertainty.

## 7. Final freeze rule

After the first successful end-to-end hardware acceptance and before collecting final thesis comparison data, do not modify:
- checkerboard geometry/count,
- tag size/IDs/orientation,
- known-angle validation protocol/gates,
- static duration,
- dynamic cycle count/order/split,
- optimizer parameter bounds or evaluation budget,
- primary objective or candidate feasibility rules.

Any later change invalidates direct comparison with earlier final-protocol runs and must trigger a new protocol version.

## References used to freeze the protocol

1. OpenCV, *Camera calibration With OpenCV*: practical chessboard calibration recommends at least 10 good snapshots in different positions. https://docs.opencv.org/4.13.0/d4/d94/tutorial_camera_calibration.html
2. Sarvestan J, et al. *Determining the optimum number of cycles for calculating joint coordination and its variability during running at different speeds*. Journal of Biomechanics. 2024;176:112375. DOI: 10.1016/j.jbiomech.2024.112375.
3. Carvalho A, et al. *How many strides are needed for reliable markerless gait analysis?* Gait & Posture. 2026;123:110020. DOI: 10.1016/j.gaitpost.2025.110020.
4. Tihanyi B, et al. *Affordable gait analysis using augmented reality markers*. PLOS ONE. 2019;14:e0212319. DOI: 10.1371/journal.pone.0212319.
5. Robert-Lachaine X, et al. *Autonomous Quality Control of Joint Orientation Measured with Inertial Sensors*. Sensors. 2016;16:1237. The paper summarizes joint-angle QC guidance using <2 deg as widely acceptable and <5 deg as a practical upper bound depending on use.
6. McGrath T, et al. *A Framework for Analytical Validation of Inertial-Sensor-Based Knee Kinematics Using a Six-Degrees-of-Freedom Joint Simulator*. Sensors. 2023;23:529. The work cites the recommendation to avoid gait-analysis errors above 5 deg.
7. *Knee Angle Estimation with Dynamic Calibration Using Inertial Measurement Units for Running*. Sensors. 2024;24:598. Reports reduced IMU knee-angle accuracy under faster/more dynamic conditions, supporting inclusion of multiple movement speeds in calibration data.

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
import zipfile
from bisect import bisect_left
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .imu_filter import KneeAngleEstimate, KneeAngleFilterMirror, KneeAngleFilterParameters
from .imu_serial import ImuCalibrationSample

STATIC_TEST_DURATION_S = 12.0
DYNAMIC_TEST_DURATION_S = 25.0
MAX_SYNC_ERROR_MS = 20.0
MIN_VISUAL_VALID_FRACTION = 0.70


@dataclass(frozen=True)
class CalibrationStatus:
    state: str
    active: bool
    current_test: str
    elapsed_s: float
    duration_s: float
    progress: float
    run_directory: str
    export_path: str
    static_archive_path: str
    dynamic_archive_path: str
    recommendation_path: str
    recommended_parameters: dict[str, Any]
    report_path: str
    overview_path: str
    valid: bool
    message: str
    error: str


@dataclass(frozen=True)
class EvaluationMetrics:
    pair_count: int
    mae_deg: float
    rmse_deg: float
    p95_abs_error_deg: float
    max_abs_error_deg: float
    lag_ms: float
    jitter_deg_per_frame: float
    drift_deg: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArchiveData:
    raw_rows: list[dict[str, str]]
    sync_rows: list[dict[str, str]]


class CalibrationWorkflow:
    """State and offline analysis for the guided static/dynamic calibration tests."""

    def __init__(self, output_root: str | Path = "calibration_results") -> None:
        self.output_root = Path(output_root)
        self._lock = threading.RLock()
        self._state = "IDLE"
        self._active = False
        self._current_test = ""
        self._started_ns = 0
        self._duration_s = 0.0
        self._run_directory: Path | None = None
        self._static_archive: Path | None = None
        self._dynamic_archive: Path | None = None
        self._recommendation_path: Path | None = None
        self._recommended_parameters: dict[str, Any] = {}
        self._report_path: Path | None = None
        self._overview_path: Path | None = None
        self._base_parameters: KneeAngleFilterParameters | None = None
        self._valid = False
        self._message = "Run the static test first, then the dynamic test."
        self._error = ""

    def prepare_test(
        self,
        test_type: str,
        *,
        base_parameters: KneeAngleFilterParameters,
    ) -> dict[str, Any]:
        test = _normalize_test_type(test_type)
        with self._lock:
            if self._active:
                raise RuntimeError("A calibration test is already running")
            if test == "static":
                self._create_new_run_locked(base_parameters)
            else:
                if self._run_directory is None or self._static_archive is None:
                    raise RuntimeError("Complete the static test before starting the dynamic test")
                if self._state not in {"STATIC_COMPLETE", "INVALID", "FAILED"}:
                    raise RuntimeError("The current calibration run is not ready for the dynamic test")
                if self._base_parameters != base_parameters:
                    raise RuntimeError(
                        "The IMU filter settings changed after the static test; repeat the static test"
                    )
                self._dynamic_archive = None
                self._recommendation_path = None
                self._recommended_parameters = {}
                self._report_path = None
                self._overview_path = None
                self._valid = False
                self._error = ""

            self._active = True
            self._current_test = test
            self._started_ns = time.perf_counter_ns()
            self._duration_s = (
                STATIC_TEST_DURATION_S if test == "static" else DYNAMIC_TEST_DURATION_S
            )
            self._state = "RUNNING_STATIC" if test == "static" else "RUNNING_DYNAMIC"
            self._message = (
                "Keep the leg completely still until the static test finishes."
                if test == "static"
                else "Perform slow, smooth flexion-extension cycles while keeping the leg visible."
            )
            return {
                "test_type": test,
                "duration_s": self._duration_s,
                "run_directory": str(self._run_directory.resolve()),
            }

    def finish_test(self, test_type: str, archive_path: str | Path) -> CalibrationStatus:
        test = _normalize_test_type(test_type)
        archive = Path(archive_path).expanduser().resolve()
        if not archive.is_file():
            raise FileNotFoundError(f"Calibration recording archive not found: {archive}")

        with self._lock:
            if not self._active or self._current_test != test:
                raise RuntimeError(f"No active {test} calibration test")
            self._active = False
            self._current_test = ""
            self._started_ns = 0
            self._duration_s = 0.0
            if test == "static":
                self._static_archive = archive
                self._state = "STATIC_COMPLETE"
                self._message = "Static test complete. Start the dynamic test next."
                return self._status_locked()
            self._dynamic_archive = archive
            self._state = "ANALYZING"
            self._message = "Analyzing both tests and generating recommended settings…"
            base_parameters = self._base_parameters
            run_directory = self._run_directory
            static_archive = self._static_archive

        if base_parameters is None or run_directory is None or static_archive is None:
            self.fail("Calibration analysis state is incomplete")
            return self.status()

        try:
            result = analyze_calibration_run(
                static_archive,
                archive,
                base_parameters=base_parameters,
                output_directory=run_directory,
            )
        except Exception as exc:
            self.fail(f"Calibration analysis failed: {exc}")
            return self.status()

        with self._lock:
            self._recommendation_path = _optional_path(result.get("recommendation_path"))
            self._recommended_parameters = dict(result.get("recommended_parameters") or {})
            self._report_path = _optional_path(result.get("report_path"))
            self._overview_path = _optional_path(result.get("overview_path"))
            self._valid = bool(result.get("valid"))
            self._state = "COMPLETE" if self._valid else "INVALID"
            self._message = str(result.get("message") or "Calibration analysis complete")
            self._error = "" if self._valid else str(result.get("error") or "")
            return self._status_locked()

    def cancel_test(self, message: str = "Calibration test cancelled") -> CalibrationStatus:
        with self._lock:
            self._active = False
            self._current_test = ""
            self._started_ns = 0
            self._duration_s = 0.0
            if self._static_archive is not None:
                self._state = "STATIC_COMPLETE"
                self._message = message + ". Static result is still available."
            else:
                self._state = "IDLE"
                self._message = message
            return self._status_locked()

    def fail(self, error: str) -> CalibrationStatus:
        with self._lock:
            self._active = False
            self._current_test = ""
            self._started_ns = 0
            self._duration_s = 0.0
            self._state = "FAILED"
            self._valid = False
            self._recommended_parameters = {}
            self._error = str(error)
            self._message = "Calibration failed. Review the error and repeat the affected test."
            return self._status_locked()

    def status(self) -> CalibrationStatus:
        with self._lock:
            return self._status_locked()

    def open_output_location(self) -> str:
        with self._lock:
            target = self._run_directory or self.output_root
        target = target.expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            startfile = getattr(os, "startfile", None)
            if startfile is None:
                raise RuntimeError("Windows folder opening is unavailable in this Python runtime")
            startfile(str(target))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
        return str(target)

    def run_directory(self) -> Path | None:
        with self._lock:
            return self._run_directory

    def _create_new_run_locked(self, base_parameters: KneeAngleFilterParameters) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_directory = self.output_root / f"calibration_run_{timestamp}"
        suffix = 1
        while run_directory.exists():
            run_directory = self.output_root / f"calibration_run_{timestamp}_{suffix:02d}"
            suffix += 1
        run_directory.mkdir(parents=True)
        self._run_directory = run_directory
        self._static_archive = None
        self._dynamic_archive = None
        self._recommendation_path = None
        self._recommended_parameters = {}
        self._report_path = None
        self._overview_path = None
        self._base_parameters = base_parameters
        self._valid = False
        self._error = ""

    def _status_locked(self) -> CalibrationStatus:
        elapsed_s = 0.0
        progress = 0.0
        if self._active and self._started_ns:
            elapsed_s = max(0.0, (time.perf_counter_ns() - self._started_ns) / 1_000_000_000.0)
            progress = min(1.0, elapsed_s / self._duration_s) if self._duration_s > 0 else 0.0
        elif self._state in {"STATIC_COMPLETE", "ANALYZING", "COMPLETE", "INVALID"}:
            progress = 1.0
        run_directory = str(self._run_directory.resolve()) if self._run_directory else ""
        return CalibrationStatus(
            state=self._state,
            active=self._active,
            current_test=self._current_test,
            elapsed_s=elapsed_s,
            duration_s=self._duration_s,
            progress=progress,
            run_directory=run_directory,
            export_path=run_directory,
            static_archive_path=str(self._static_archive or ""),
            dynamic_archive_path=str(self._dynamic_archive or ""),
            recommendation_path=str(self._recommendation_path or ""),
            recommended_parameters=dict(self._recommended_parameters),
            report_path=str(self._report_path or ""),
            overview_path=str(self._overview_path or ""),
            valid=self._valid,
            message=self._message,
            error=self._error,
        )


def analyze_calibration_run(
    static_archive_path: str | Path,
    dynamic_archive_path: str | Path,
    *,
    base_parameters: KneeAngleFilterParameters,
    output_directory: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_directory).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    static_archive = Path(static_archive_path).expanduser().resolve()
    dynamic_archive = Path(dynamic_archive_path).expanduser().resolve()
    static_data = load_archive_data(static_archive)
    dynamic_data = load_archive_data(dynamic_archive)

    gates = _quality_gates(static_data, dynamic_data, base_parameters)
    failed_gates = [name for name, gate in gates.items() if not gate["passed"]]
    report_path = output_dir / "calibration_report.json"
    overview_path = output_dir / "calibration_overview.png"
    recommendation_path = output_dir / "recommended_imu_settings.yaml"

    if failed_gates:
        recommendation_path.unlink(missing_ok=True)
        report = {
            "schema_version": "vision_imu_calibration_report_v1",
            "generated_at": datetime.now().astimezone().isoformat(),
            "status": "INVALID",
            "method": "synchronized_static_dynamic_vision_imu_comparison",
            "quality_gates": gates,
            "failed_gates": failed_gates,
            "source_archives": {
                "static": str(static_archive),
                "dynamic": str(dynamic_archive),
            },
            "baseline_parameters": _parameter_payload(base_parameters),
            "message": "Test data did not pass the quality gates. No settings were recommended.",
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_invalid_overview(overview_path, report)
        return {
            "valid": False,
            "message": "TEST INVALID — repeat the failed acquisition.",
            "error": ", ".join(failed_gates),
            "report_path": str(report_path),
            "overview_path": str(overview_path),
            "recommendation_path": "",
            "recommended_parameters": {},
        }

    static_result = _derive_static_parameters(static_data, base_parameters)
    static_parameters = static_result["parameters"]
    baseline_dynamic = evaluate_parameters(dynamic_data, base_parameters)

    candidates = _candidate_parameters(static_parameters)
    evaluated: list[tuple[KneeAngleFilterParameters, EvaluationMetrics]] = []
    for candidate in candidates:
        metrics = evaluate_parameters(dynamic_data, candidate)
        if metrics.pair_count:
            evaluated.append((candidate, metrics))
    if not evaluated:
        raise RuntimeError("No valid dynamic parameter candidate could be evaluated")
    recommended_parameters, recommended_dynamic = min(evaluated, key=lambda item: item[1].score)
    recommended_static = evaluate_parameters(static_data, recommended_parameters)

    improvement_percent = 0.0
    if baseline_dynamic.score > 1e-9:
        improvement_percent = 100.0 * (
            baseline_dynamic.score - recommended_dynamic.score
        ) / baseline_dynamic.score

    report = {
        "schema_version": "vision_imu_calibration_report_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "VALID",
        "method": "synchronized_static_dynamic_vision_imu_comparison",
        "reference": {
            "type": "vision_assisted_comparative_reference",
            "statement": (
                "Recommendations minimize disagreement with the synchronized visual reference "
                "inside these two tests; they are not absolute ground truth."
            ),
        },
        "quality_gates": gates,
        "source_archives": {
            "static": str(static_archive),
            "dynamic": str(dynamic_archive),
        },
        "baseline_parameters": _parameter_payload(base_parameters),
        "recommended_parameters": _parameter_payload(recommended_parameters),
        "static_derivation": {
            key: value for key, value in static_result.items() if key != "parameters"
        },
        "metrics": {
            "static_recommended": recommended_static.to_dict(),
            "dynamic_baseline": baseline_dynamic.to_dict(),
            "dynamic_recommended": recommended_dynamic.to_dict(),
            "dynamic_score_improvement_percent": improvement_percent,
            "candidate_count": len(evaluated),
        },
        "limitations": [
            "The visual estimator is a comparative reference, not a certified angle standard.",
            "The recommendation is valid only for the tested sensor mounting and movement range.",
            "Orientation conventions are preserved; this tool does not rewrite axis mapping automatically.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_recommendation_yaml(
        recommendation_path,
        recommended_parameters,
        report=report,
    )
    _write_calibration_overview(
        overview_path,
        dynamic_data,
        base_parameters,
        recommended_parameters,
        baseline_dynamic,
        recommended_dynamic,
    )
    return {
        "valid": True,
        "message": (
            "Calibration complete. Recommended settings and report were exported. "
            f"Dynamic score change: {improvement_percent:+.1f}%."
        ),
        "error": "",
        "report_path": str(report_path),
        "overview_path": str(overview_path),
        "recommendation_path": str(recommendation_path),
        "recommended_parameters": _parameter_payload(recommended_parameters),
    }


def load_archive_data(archive_path: str | Path) -> ArchiveData:
    path = Path(archive_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        required = {"raw_imu.csv", "synchronized_timeline.csv"}
        missing = required - names
        if missing:
            raise ValueError(f"Archive {path.name} is missing: {', '.join(sorted(missing))}")
        raw_rows = _read_csv_text(archive.read("raw_imu.csv").decode("utf-8-sig"))
        sync_rows = _read_csv_text(
            archive.read("synchronized_timeline.csv").decode("utf-8-sig")
        )
    return ArchiveData(raw_rows=raw_rows, sync_rows=sync_rows)


def evaluate_parameters(
    data: ArchiveData,
    parameters: KneeAngleFilterParameters,
) -> EvaluationMetrics:
    estimates = replay_filter(data.raw_rows, parameters)
    pairs = _paired_values(data.sync_rows, estimates, lag_ms=0.0)
    if not pairs:
        return _empty_metrics()
    lag_ms = _estimate_lag_ms(data.sync_rows, estimates)
    lagged_pairs = _paired_values(data.sync_rows, estimates, lag_ms=lag_ms)
    if lagged_pairs:
        pairs = lagged_pairs
    errors = [visual - imu for _, visual, imu in pairs]
    abs_errors = [abs(value) for value in errors]
    mae = statistics.fmean(abs_errors)
    rmse = math.sqrt(statistics.fmean(value * value for value in errors))
    p95 = _percentile(abs_errors, 95.0)
    maximum = max(abs_errors)
    jitter = statistics.median(
        abs(errors[index] - errors[index - 1]) for index in range(1, len(errors))
    ) if len(errors) > 1 else 0.0
    segment = max(1, len(errors) // 5)
    drift = statistics.median(errors[-segment:]) - statistics.median(errors[:segment])
    score = mae + 0.20 * p95 + 0.015 * abs(lag_ms) + 0.15 * jitter + 0.10 * abs(drift)
    return EvaluationMetrics(
        pair_count=len(pairs),
        mae_deg=mae,
        rmse_deg=rmse,
        p95_abs_error_deg=p95,
        max_abs_error_deg=maximum,
        lag_ms=lag_ms,
        jitter_deg_per_frame=jitter,
        drift_deg=drift,
        score=score,
    )


def replay_filter(
    raw_rows: Sequence[Mapping[str, str]],
    parameters: KneeAngleFilterParameters,
) -> dict[int, KneeAngleEstimate]:
    estimator = KneeAngleFilterMirror(parameters)
    estimates: dict[int, KneeAngleEstimate] = {}
    for row in raw_rows:
        sample = _sample_from_row(row)
        estimate = estimator.update(sample)
        estimates[sample.sample_index] = estimate
    return estimates


def _derive_static_parameters(
    data: ArchiveData,
    base_parameters: KneeAngleFilterParameters,
) -> dict[str, Any]:
    baseline_estimates = replay_filter(data.raw_rows, base_parameters)
    pairs = _paired_values(data.sync_rows, baseline_estimates, lag_ms=0.0)
    if not pairs:
        raise RuntimeError("Static test has no synchronized valid camera/IMU pairs")
    angle_offset_delta = statistics.median(visual - imu for _, visual, imu in pairs)

    thigh_rates = [_signed_axis_rate(row, "thigh", base_parameters) for row in data.raw_rows]
    shank_rates = [_signed_axis_rate(row, "shank", base_parameters) for row in data.raw_rows]
    thigh_bias = statistics.median(thigh_rates)
    shank_bias = statistics.median(shank_rates)
    residual_rates = [
        *(abs(rate - thigh_bias) for rate in thigh_rates),
        *(abs(rate - shank_bias) for rate in shank_rates),
    ]
    deadband = min(2.5, max(0.15, 1.5 * _percentile(residual_rates, 95.0)))
    parameters = replace(
        base_parameters,
        straight_leg_offset_deg=base_parameters.straight_leg_offset_deg + angle_offset_delta,
        thigh_gyro_bias_deg_s=thigh_bias,
        shank_gyro_bias_deg_s=shank_bias,
        gyro_deadband_deg_s=deadband,
    )
    parameters.validate()
    return {
        "parameters": parameters,
        "angle_offset_delta_deg": angle_offset_delta,
        "thigh_gyro_bias_deg_s": thigh_bias,
        "shank_gyro_bias_deg_s": shank_bias,
        "gyro_deadband_deg_s": deadband,
        "pair_count": len(pairs),
    }


def _candidate_parameters(
    static_parameters: KneeAngleFilterParameters,
) -> list[KneeAngleFilterParameters]:
    stable_values = _unique_sorted(
        [static_parameters.alpha_stable, 0.80, 0.85, 0.90, 0.93, 0.95]
    )
    dynamic_values = _unique_sorted(
        [static_parameters.alpha_dynamic, 0.95, 0.97, 0.975, 0.985, 0.99]
    )
    correction_values = _unique_sorted(
        [static_parameters.max_accel_correction_deg, 5.0, 8.0, 10.0, 15.0]
    )
    candidates: list[KneeAngleFilterParameters] = []
    for alpha_stable in stable_values:
        for alpha_dynamic in dynamic_values:
            if alpha_dynamic < alpha_stable:
                continue
            for correction in correction_values:
                candidate = replace(
                    static_parameters,
                    alpha_stable=alpha_stable,
                    alpha_dynamic=alpha_dynamic,
                    max_accel_correction_deg=correction,
                )
                candidate.validate()
                candidates.append(candidate)
    return candidates


def _quality_gates(
    static_data: ArchiveData,
    dynamic_data: ArchiveData,
    parameters: KneeAngleFilterParameters,
) -> dict[str, dict[str, Any]]:
    static_sync = _valid_sync_rows(static_data.sync_rows)
    dynamic_sync = _valid_sync_rows(dynamic_data.sync_rows)
    static_valid_fraction = _visual_valid_fraction(static_data.sync_rows)
    dynamic_valid_fraction = _visual_valid_fraction(dynamic_data.sync_rows)
    static_sync_p95 = _sync_p95_ms(static_sync)
    dynamic_sync_p95 = _sync_p95_ms(dynamic_sync)
    static_gap_fraction = _sequence_gap_fraction(static_data.raw_rows)
    dynamic_gap_fraction = _sequence_gap_fraction(dynamic_data.raw_rows)
    static_ready_fraction = _imu_ready_fraction(static_data.raw_rows)
    dynamic_ready_fraction = _imu_ready_fraction(dynamic_data.raw_rows)
    static_base_pairs = len(
        _paired_values(static_data.sync_rows, replay_filter(static_data.raw_rows, parameters), lag_ms=0.0)
    )
    dynamic_base_pairs = len(
        _paired_values(dynamic_data.sync_rows, replay_filter(dynamic_data.raw_rows, parameters), lag_ms=0.0)
    )
    dynamic_angles = [float(row["visual_knee_angle_deg"]) for row in dynamic_sync]
    dynamic_range = max(dynamic_angles) - min(dynamic_angles) if dynamic_angles else 0.0
    static_motion = _static_motion_p95(static_data.raw_rows, parameters)

    def gate(passed: bool, value: Any, requirement: str) -> dict[str, Any]:
        return {"passed": bool(passed), "value": value, "requirement": requirement}

    return {
        "static_imu_samples": gate(
            len(static_data.raw_rows) >= 300,
            len(static_data.raw_rows),
            ">= 300 raw IMU samples",
        ),
        "static_imu_ready_fraction": gate(
            static_ready_fraction >= 0.95,
            static_ready_fraction,
            ">= 95% samples with both IMUs ready",
        ),
        "static_valid_visual_rows": gate(
            len(static_sync) >= 30,
            len(static_sync),
            ">= 30 synchronized valid visual rows",
        ),
        "static_valid_comparison_pairs": gate(
            static_base_pairs >= 30,
            static_base_pairs,
            ">= 30 valid baseline camera/IMU pairs",
        ),
        "static_tracking_fraction": gate(
            static_valid_fraction >= MIN_VISUAL_VALID_FRACTION,
            static_valid_fraction,
            f">= {MIN_VISUAL_VALID_FRACTION:.0%}",
        ),
        "static_sync_p95_ms": gate(
            static_sync_p95 <= MAX_SYNC_ERROR_MS,
            static_sync_p95,
            f"<= {MAX_SYNC_ERROR_MS:.0f} ms",
        ),
        "static_motion_p95_deg_s": gate(
            static_motion <= 5.0,
            static_motion,
            "<= 5 deg/s; leg must remain still",
        ),
        "static_sequence_gap_fraction": gate(
            static_gap_fraction <= 0.01,
            static_gap_fraction,
            "<= 1% missing sequence steps",
        ),
        "dynamic_imu_samples": gate(
            len(dynamic_data.raw_rows) >= 800,
            len(dynamic_data.raw_rows),
            ">= 800 raw IMU samples",
        ),
        "dynamic_imu_ready_fraction": gate(
            dynamic_ready_fraction >= 0.95,
            dynamic_ready_fraction,
            ">= 95% samples with both IMUs ready",
        ),
        "dynamic_valid_visual_rows": gate(
            len(dynamic_sync) >= 80,
            len(dynamic_sync),
            ">= 80 synchronized valid visual rows",
        ),
        "dynamic_valid_comparison_pairs": gate(
            dynamic_base_pairs >= 80,
            dynamic_base_pairs,
            ">= 80 valid baseline camera/IMU pairs",
        ),
        "dynamic_tracking_fraction": gate(
            dynamic_valid_fraction >= MIN_VISUAL_VALID_FRACTION,
            dynamic_valid_fraction,
            f">= {MIN_VISUAL_VALID_FRACTION:.0%}",
        ),
        "dynamic_sync_p95_ms": gate(
            dynamic_sync_p95 <= MAX_SYNC_ERROR_MS,
            dynamic_sync_p95,
            f"<= {MAX_SYNC_ERROR_MS:.0f} ms",
        ),
        "dynamic_visual_range_deg": gate(
            dynamic_range >= 30.0,
            dynamic_range,
            ">= 30 deg movement range",
        ),
        "dynamic_sequence_gap_fraction": gate(
            dynamic_gap_fraction <= 0.01,
            dynamic_gap_fraction,
            "<= 1% missing sequence steps",
        ),
    }


def _paired_values(
    sync_rows: Sequence[Mapping[str, str]],
    estimates: Mapping[int, KneeAngleEstimate],
    *,
    lag_ms: float,
) -> list[tuple[int, float, float]]:
    valid_rows = _valid_sync_rows(sync_rows)
    if not valid_rows:
        return []
    estimate_series = sorted(
        (
            estimate.host_monotonic_ns,
            estimate.knee_angle_deg,
        )
        for estimate in estimates.values()
        if estimate.valid and estimate.knee_angle_deg is not None
    )
    estimate_times = [item[0] for item in estimate_series]
    lag_ns = int(lag_ms * 1_000_000.0)
    pairs: list[tuple[int, float, float]] = []
    for row in valid_rows:
        pose_time = int(float(row["host_monotonic_ns"]))
        target = pose_time + lag_ns
        index = _nearest_index(estimate_times, target)
        if index is None:
            continue
        estimate_time, estimate_angle = estimate_series[index]
        if abs(estimate_time - target) > 25_000_000:
            continue
        pairs.append((pose_time, float(row["visual_knee_angle_deg"]), float(estimate_angle)))
    return pairs


def _estimate_lag_ms(
    sync_rows: Sequence[Mapping[str, str]],
    estimates: Mapping[int, KneeAngleEstimate],
) -> float:
    visual_values = [
        float(row["visual_knee_angle_deg"])
        for row in _valid_sync_rows(sync_rows)
        if row.get("visual_knee_angle_deg") not in (None, "")
    ]
    # A nearly static signal has no identifiable time shift. Searching the lag
    # range would otherwise select an arbitrary boundary value and inflate the
    # static score, even though lag is meaningful only during motion.
    if len(visual_values) < 20 or max(visual_values) - min(visual_values) < 10.0:
        return 0.0

    best_lag = 0.0
    best_rmse = math.inf
    for lag_ms in range(-250, 251, 10):
        pairs = _paired_values(sync_rows, estimates, lag_ms=float(lag_ms))
        if len(pairs) < 20:
            continue
        errors = [visual - imu for _, visual, imu in pairs]
        centered = [value - statistics.fmean(errors) for value in errors]
        rmse = math.sqrt(statistics.fmean(value * value for value in centered))
        if rmse < best_rmse:
            best_rmse = rmse
            best_lag = float(lag_ms)
    return best_lag


def _valid_sync_rows(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    valid: list[Mapping[str, str]] = []
    for row in rows:
        if not _to_bool(row.get("tracking_valid")):
            continue
        visual = _to_float(row.get("visual_knee_angle_deg"))
        sync_delta = _to_float(row.get("sync_delta_ms"))
        if visual is None or sync_delta is None or abs(sync_delta) > MAX_SYNC_ERROR_MS:
            continue
        valid.append(row)
    return valid


def _visual_valid_fraction(rows: Sequence[Mapping[str, str]]) -> float:
    if not rows:
        return 0.0
    valid = sum(1 for row in rows if _to_bool(row.get("tracking_valid")))
    return valid / len(rows)


def _sync_p95_ms(rows: Sequence[Mapping[str, str]]) -> float:
    values = [abs(float(row["sync_delta_ms"])) for row in rows if row.get("sync_delta_ms")]
    return _percentile(values, 95.0) if values else math.inf


def _sequence_gap_fraction(rows: Sequence[Mapping[str, str]]) -> float:
    indices = [int(float(row["sample_index"])) for row in rows if row.get("sample_index")]
    if len(indices) < 2:
        return 1.0
    missing = sum(max(0, current - previous - 1) for previous, current in zip(indices, indices[1:]))
    expected = max(1, indices[-1] - indices[0])
    return missing / expected


def _imu_ready_fraction(rows: Sequence[Mapping[str, str]]) -> float:
    if not rows:
        return 0.0
    ready = sum(
        1
        for row in rows
        if _to_bool(row.get("thigh_ok")) and _to_bool(row.get("shank_ok"))
    )
    return ready / len(rows)


def _static_motion_p95(
    rows: Sequence[Mapping[str, str]],
    parameters: KneeAngleFilterParameters,
) -> float:
    relative_rates = [
        abs(
            _signed_axis_rate(row, "shank", parameters)
            - _signed_axis_rate(row, "thigh", parameters)
        )
        for row in rows
    ]
    return _percentile(relative_rates, 95.0) if relative_rates else math.inf


def _signed_axis_rate(
    row: Mapping[str, str],
    prefix: str,
    parameters: KneeAngleFilterParameters,
) -> float:
    axis = ("gx", "gy", "gz")[parameters.rotation_axis]
    return parameters.gyro_rate_sign * float(row[f"{prefix}_{axis}_dps"])


def _sample_from_row(row: Mapping[str, str]) -> ImuCalibrationSample:
    return ImuCalibrationSample(
        host_monotonic_ns=int(float(row["host_monotonic_ns"])),
        schema_version=int(float(row["schema_version"])),
        sample_index=int(float(row["sample_index"])),
        mcu_timestamp_us=int(float(row["mcu_timestamp_us"])),
        thigh_ok=_to_bool(row.get("thigh_ok")),
        shank_ok=_to_bool(row.get("shank_ok")),
        thigh_ax_raw=int(float(row["thigh_ax_raw"])),
        thigh_ay_raw=int(float(row["thigh_ay_raw"])),
        thigh_az_raw=int(float(row["thigh_az_raw"])),
        thigh_gx_raw=int(float(row["thigh_gx_raw"])),
        thigh_gy_raw=int(float(row["thigh_gy_raw"])),
        thigh_gz_raw=int(float(row["thigh_gz_raw"])),
        shank_ax_raw=int(float(row["shank_ax_raw"])),
        shank_ay_raw=int(float(row["shank_ay_raw"])),
        shank_az_raw=int(float(row["shank_az_raw"])),
        shank_gx_raw=int(float(row["shank_gx_raw"])),
        shank_gy_raw=int(float(row["shank_gy_raw"])),
        shank_gz_raw=int(float(row["shank_gz_raw"])),
    )


def _write_recommendation_yaml(
    path: Path,
    parameters: KneeAngleFilterParameters,
    *,
    report: Mapping[str, Any],
) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to export recommended settings") from exc
    payload = {
        "schema_version": "vision_imu_recommended_settings_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "VALID",
        "method": "static_offset_bias_plus_dynamic_grid_search",
        "reference": "synchronized MoveNet knee-angle comparison",
        "dynamic_score_improvement_percent": report["metrics"][
            "dynamic_score_improvement_percent"
        ],
        "stm32_control": {"knee_angle": _parameter_payload(parameters)},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_calibration_overview(
    path: Path,
    data: ArchiveData,
    baseline_parameters: KneeAngleFilterParameters,
    recommended_parameters: KneeAngleFilterParameters,
    baseline_metrics: EvaluationMetrics,
    recommended_metrics: EvaluationMetrics,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to export the calibration overview") from exc
    baseline = replay_filter(data.raw_rows, baseline_parameters)
    recommended = replay_filter(data.raw_rows, recommended_parameters)
    visual_rows = _valid_sync_rows(data.sync_rows)
    if not visual_rows:
        raise RuntimeError("No valid dynamic visual rows for overview plot")
    t0 = int(float(visual_rows[0]["host_monotonic_ns"]))
    times = [(int(float(row["host_monotonic_ns"])) - t0) / 1_000_000_000.0 for row in visual_rows]
    visual = [float(row["visual_knee_angle_deg"]) for row in visual_rows]
    baseline_values = _angles_at_rows(visual_rows, baseline)
    recommended_values = _angles_at_rows(visual_rows, recommended)

    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(times, visual, label="Visual reference", linewidth=1.8)
    axis.plot(times, baseline_values, label="Baseline IMU", linewidth=1.2)
    axis.plot(times, recommended_values, label="Recommended IMU", linewidth=1.2)
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Posterior knee angle [deg]")
    axis.grid(True, alpha=0.3)
    axis.legend()
    axis.set_title(
        "Vision–IMU calibration overview\n"
        f"baseline MAE {baseline_metrics.mae_deg:.2f}° → recommended {recommended_metrics.mae_deg:.2f}°"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_invalid_overview(path: Path, report: Mapping[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    failed = report.get("failed_gates", [])
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.axis("off")
    axis.text(
        0.02,
        0.95,
        "Calibration test invalid",
        fontsize=18,
        fontweight="bold",
        va="top",
    )
    axis.text(
        0.02,
        0.82,
        "Repeat the acquisition after correcting:\n\n" + "\n".join(f"• {item}" for item in failed),
        fontsize=11,
        va="top",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _angles_at_rows(
    rows: Sequence[Mapping[str, str]],
    estimates: Mapping[int, KneeAngleEstimate],
) -> list[float]:
    series = sorted(
        (estimate.host_monotonic_ns, float(estimate.knee_angle_deg))
        for estimate in estimates.values()
        if estimate.valid and estimate.knee_angle_deg is not None
    )
    times = [item[0] for item in series]
    values: list[float] = []
    for row in rows:
        target = int(float(row["host_monotonic_ns"]))
        index = _nearest_index(times, target)
        values.append(series[index][1] if index is not None else math.nan)
    return values


def _parameter_payload(parameters: KneeAngleFilterParameters) -> dict[str, Any]:
    payload = parameters.to_dict()
    payload.pop("rotation_axis_name", None)
    payload["rotation_axis"] = ("x", "y", "z")[parameters.rotation_axis]
    return payload


def _read_csv_text(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(text.splitlines()))


def _nearest_index(values: Sequence[int], target: int) -> int | None:
    if not values:
        return None
    insertion = bisect_left(values, target)
    candidates: list[int] = []
    if insertion > 0:
        candidates.append(insertion - 1)
    if insertion < len(values):
        candidates.append(insertion)
    return min(candidates, key=lambda index: abs(values[index] - target)) if candidates else None


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _to_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _unique_sorted(values: Iterable[float]) -> list[float]:
    return sorted({round(float(value), 6) for value in values})


def _normalize_test_type(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in {"static", "dynamic"}:
        raise ValueError("Calibration test_type must be 'static' or 'dynamic'")
    return normalized


def _optional_path(value: Any) -> Path | None:
    text = str(value or "").strip()
    return Path(text) if text else None


def _empty_metrics() -> EvaluationMetrics:
    return EvaluationMetrics(
        pair_count=0,
        mae_deg=math.inf,
        rmse_deg=math.inf,
        p95_abs_error_deg=math.inf,
        max_abs_error_deg=math.inf,
        lag_ms=0.0,
        jitter_deg_per_frame=math.inf,
        drift_deg=math.inf,
        score=math.inf,
    )

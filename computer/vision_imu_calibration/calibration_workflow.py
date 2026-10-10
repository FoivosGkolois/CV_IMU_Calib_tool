from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import threading
import time
import uuid
import zipfile
from bisect import bisect_left
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .imu_filter import KneeAngleEstimate, KneeAngleFilterMirror, KneeAngleFilterParameters
from .imu_serial import ACCEL_G_PER_LSB, GYRO_DPS_PER_LSB, ImuCalibrationSample
from .bayesian_optimizer import (BO_EVALUATION_BUDGET, BO_RANDOM_SEED,
                                 optimize_filter_parameters, parameter_bounds_payload)

STATIC_TEST_DURATION_S = 12.0
DYNAMIC_TEST_DURATION_S = 25.0
MAX_SYNC_ERROR_MS = 20.0
MIN_VISUAL_VALID_FRACTION = 0.70
SUGGESTIONS_FILENAME = "imu_filter_suggestions.yaml"
STATIC_ARCHIVE_FILENAME = "static_trial.zip"
DYNAMIC_ARCHIVE_FILENAME = "dynamic_trial.zip"
HELDOUT_REPORT_FILENAME = "heldout_validation.json"
SETUP_INVALIDATION_FILENAME = "setup_invalidation.json"
HELDOUT_PAIRING_POLICY = (
    "actual valid optical rows; nearest ready raw IMU at host lag 0; "
    "no interpolation; frozen identities; full pair coverage required"
)


@dataclass(frozen=True)
class CalibrationStatus:
    state: str
    active: bool
    current_test: str
    elapsed_s: float
    duration_s: float
    progress: float
    run_id: str
    run_directory: str
    export_path: str
    static_archive_path: str
    dynamic_archive_path: str
    recommendation_path: str
    static_suggestion: dict[str, Any]
    dynamic_suggestion: dict[str, Any]
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

    def __init__(self, output_root: str | Path = "calibration_suggestions") -> None:
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
        self._static_suggestion: dict[str, Any] = {}
        self._dynamic_suggestion: dict[str, Any] = {}
        self._recommended_parameters: dict[str, Any] = {}
        self._report_path: Path | None = None
        self._overview_path: Path | None = None
        self._base_parameters: KneeAngleFilterParameters | None = None
        self._setup_metadata: dict[str, Any] | None = None
        self._heldout_result: dict[str, Any] = {}
        self._valid = False
        self._message = "Start calibration with the static stage."
        self._error = ""
        self._restore_latest_completed_run()

    def prepare_test(
        self,
        test_type: str,
        *,
        base_parameters: KneeAngleFilterParameters,
        setup_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        test = _normalize_test_type(test_type)
        setup = _normalize_setup_metadata(setup_metadata)
        with self._lock:
            if self._active:
                raise RuntimeError("A calibration test is already running")
            if test == "static":
                self._create_new_run_locked(base_parameters, setup)
            else:
                if self._run_directory is None or self._static_archive is None:
                    raise RuntimeError("Complete the static test before starting the dynamic test")
                if not self._static_suggestion or self._state not in {"AWAITING_DYNAMIC", "INVALID"}:
                    raise RuntimeError("The current calibration run is not ready for the dynamic test")
                if self._base_parameters != base_parameters:
                    raise RuntimeError(
                        "The IMU filter settings changed after the static test; repeat the static test"
                    )
                if self._setup_metadata != setup:
                    raise RuntimeError(
                        "The physical/camera setup changed after the static test; repeat the static test"
                    )
                self._dynamic_archive = None
                expected_dynamic = self._run_directory / DYNAMIC_ARCHIVE_FILENAME
                expected_dynamic.unlink(missing_ok=True)
                self._dynamic_suggestion = {}
                self._recommended_parameters = dict(self._static_suggestion)
                self._report_path = None
                self._overview_path = None
                self._heldout_result = {}
                (self._run_directory / "calibration_report.json").unlink(missing_ok=True)
                (self._run_directory / "calibration_overview.png").unlink(missing_ok=True)
                self._valid = False
                self._error = ""
                _write_suggestions_yaml(
                    self._run_directory / SUGGESTIONS_FILENAME,
                    status="awaiting_dynamic",
                    run_id=self._run_directory.name,
                    static_suggestion=self._static_suggestion,
                    dynamic_suggestion=None,
                    final_parameters=None,
                )

            self._active = True
            self._current_test = test
            self._started_ns = time.perf_counter_ns()
            self._duration_s = (
                STATIC_TEST_DURATION_S if test == "static" else DYNAMIC_TEST_DURATION_S
            )
            self._state = "RUNNING_STATIC" if test == "static" else "RUNNING_DYNAMIC"
            self._message = (
                "Straight-Leg Calibration: hold a comfortable near-straight position completely still. "
                "Alignment uses the measured optical angle, without forcing 180 degrees."
                if test == "static"
                else "Perform slow, smooth flexion-extension cycles while keeping the leg visible."
            )
            return {
                "test_type": test,
                "duration_s": self._duration_s,
                "run_directory": str(self._run_directory.resolve()),
                "archive_name": (
                    STATIC_ARCHIVE_FILENAME if test == "static" else DYNAMIC_ARCHIVE_FILENAME
                ),
            }

    def finish_test(self, test_type: str, archive_path: str | Path) -> CalibrationStatus:
        test = _normalize_test_type(test_type)
        archive = Path(archive_path).expanduser().resolve()
        if not archive.is_file():
            raise FileNotFoundError(f"Calibration recording archive not found: {archive}")

        with self._lock:
            if not self._active or self._current_test != test:
                raise RuntimeError(f"No active {test} calibration test")
            run_directory = self._run_directory
            base_parameters = self._base_parameters
            if run_directory is None or base_parameters is None:
                raise RuntimeError("Calibration run ownership state is incomplete")
            expected_name = (
                STATIC_ARCHIVE_FILENAME if test == "static" else DYNAMIC_ARCHIVE_FILENAME
            )
            expected_archive = (run_directory / expected_name).resolve()
            if archive != expected_archive:
                raise RuntimeError(
                    f"Calibration archive must be owned by the active run: {expected_archive}"
                )
            self._active = False
            self._current_test = ""
            self._started_ns = 0
            self._duration_s = 0.0
            if test == "static":
                self._static_archive = archive
                self._state = "ANALYZING_STATIC"
                self._message = "Analyzing the static stage and writing its suggestion…"
                static_archive = archive
            else:
                static_archive = self._static_archive
                self._dynamic_archive = archive
                self._state = "ANALYZING"
                self._message = "Running 30 deterministic Bayesian training evaluations; exporting suggestions only…"

        if static_archive is None:
            self.fail("Calibration analysis state is incomplete")
            return self.status()

        if test == "static":
            try:
                result = analyze_static_calibration_stage(
                    static_archive,
                    base_parameters=base_parameters,
                    output_directory=run_directory,
                    calibration_setup=self._setup_metadata,
                )
            except Exception as exc:
                self.fail(f"Static calibration analysis failed: {exc}", stage="static")
                return self.status()

            with self._lock:
                self._recommendation_path = _optional_path(result.get("recommendation_path"))
                self._report_path = _optional_path(result.get("report_path"))
                self._static_suggestion = dict(result.get("static_suggestion") or {})
                self._dynamic_suggestion = {}
                self._recommended_parameters = dict(self._static_suggestion)
                self._valid = False
                if bool(result.get("valid")):
                    self._state = "AWAITING_DYNAMIC"
                    self._message = "Static stage complete. Continue to dynamic calibration."
                    self._error = ""
                else:
                    self._state = "INVALID"
                    self._message = str(result.get("message") or "Static stage invalid")
                    self._error = str(result.get("error") or "")
                return self._status_locked()

        try:
            result = analyze_calibration_run(
                static_archive,
                archive,
                base_parameters=base_parameters,
                output_directory=run_directory,
                calibration_setup=self._setup_metadata,
            )
        except Exception as exc:
            self.fail(f"Calibration analysis failed: {exc}", stage="dynamic")
            return self.status()

        with self._lock:
            self._recommendation_path = _optional_path(result.get("recommendation_path"))
            self._static_suggestion = dict(result.get("static_suggestion") or {})
            self._dynamic_suggestion = dict(result.get("dynamic_suggestion") or {})
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
            cancelled_test = self._current_test
            run_directory = self._run_directory
            self._active = False
            self._current_test = ""
            self._started_ns = 0
            self._duration_s = 0.0
            if cancelled_test == "static" and run_directory is not None:
                shutil.rmtree(run_directory, ignore_errors=True)
                self._reset_run_locked()
                self._state = "IDLE"
                self._message = message + ". No calibration run was kept."
            elif self._static_archive is not None:
                if run_directory is not None:
                    (run_directory / DYNAMIC_ARCHIVE_FILENAME).unlink(missing_ok=True)
                    _write_suggestions_yaml(
                        run_directory / SUGGESTIONS_FILENAME,
                        status="awaiting_dynamic",
                        run_id=run_directory.name,
                        static_suggestion=self._static_suggestion,
                        dynamic_suggestion=None,
                        final_parameters=None,
                    )
                self._dynamic_archive = None
                self._dynamic_suggestion = {}
                self._state = "AWAITING_DYNAMIC"
                self._message = message + ". Static stage is still available."
            else:
                self._state = "IDLE"
                self._message = message
            return self._status_locked()

    def fail(self, error: str, *, stage: str | None = None) -> CalibrationStatus:
        with self._lock:
            failed_test = stage or self._current_test
            self._active = False
            self._current_test = ""
            self._started_ns = 0
            self._duration_s = 0.0
            if self._static_archive is not None and failed_test != "static":
                if self._run_directory is not None:
                    (self._run_directory / DYNAMIC_ARCHIVE_FILENAME).unlink(missing_ok=True)
                    _write_suggestions_yaml(
                        self._run_directory / SUGGESTIONS_FILENAME,
                        status="awaiting_dynamic",
                        run_id=self._run_directory.name,
                        static_suggestion=self._static_suggestion,
                        dynamic_suggestion=None,
                        final_parameters=None,
                        error=str(error),
                    )
                self._dynamic_archive = None
                self._dynamic_suggestion = {}
                self._state = "AWAITING_DYNAMIC"
                self._message = "Dynamic stage failed. Fix the issue and retry the dynamic stage."
            else:
                if self._run_directory is not None:
                    _write_suggestions_yaml(
                        self._run_directory / SUGGESTIONS_FILENAME,
                        status="failed_static",
                        run_id=self._run_directory.name,
                        static_suggestion=None,
                        dynamic_suggestion=None,
                        final_parameters=None,
                        error=str(error),
                    )
                self._state = "FAILED"
                self._message = "Calibration failed. Review the error and repeat the static stage."
            self._valid = False
            self._recommended_parameters = dict(self._static_suggestion)
            self._error = str(error)
            return self._status_locked()

    def status(self) -> CalibrationStatus:
        with self._lock:
            return self._status_locked()

    def open_output_location(self) -> str:
        """Return the active calibration folder without invoking a desktop shell."""
        with self._lock:
            target = self._run_directory or self.output_root
        target = target.expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    def run_directory(self) -> Path | None:
        with self._lock:
            return self._run_directory

    def setup_metadata(self) -> dict[str, Any] | None:
        """Return the immutable setup snapshot owned by the active/restored run."""
        with self._lock:
            return dict(self._setup_metadata) if self._setup_metadata is not None else None

    def invalidate_setup(self, reason: str) -> CalibrationStatus:
        """Invalidate reuse when the live pose/camera setup changes."""
        message = str(reason or "Calibration setup changed").strip()
        with self._lock:
            if self._run_directory is not None or self._setup_metadata is not None:
                self._valid = False
                self._heldout_result = {}
                self._state = "INVALID"
                self._message = message
                self._error = message
                if self._run_directory is not None:
                    try:
                        _atomic_write_json(
                            self._run_directory / SETUP_INVALIDATION_FILENAME,
                            {
                                "schema_version": "vision_imu_setup_invalidation_v1",
                                "invalidated_at": datetime.now().astimezone().isoformat(),
                                "reason": message,
                            },
                        )
                    except OSError as exc:
                        raise RuntimeError(
                            f"Could not persist calibration invalidation: {exc}"
                        ) from exc
            return self._status_locked()

    def heldout_status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._heldout_result)

    def heldout_context(self) -> dict[str, Any]:
        with self._lock:
            if not self._valid or self._state != "COMPLETE" or self._run_directory is None:
                raise RuntimeError("Complete a valid Bayesian calibration before held-out validation")
            return _load_heldout_context(self._run_directory)

    def validate_heldout(
        self,
        archive_path: str | Path,
        *,
        current_setup: Mapping[str, Any] | None,
        unchanged_setup: bool,
    ) -> dict[str, Any]:
        with self._lock:
            if not self._valid or self._state != "COMPLETE" or self._run_directory is None:
                raise RuntimeError("Complete a valid Bayesian calibration before held-out validation")
            run_directory = self._run_directory
        result = validate_heldout_recording(
            run_directory,
            archive_path,
            current_setup=current_setup,
            unchanged_setup=unchanged_setup,
        )
        with self._lock:
            self._heldout_result = dict(result)
        return dict(result)

    def _create_new_run_locked(
        self,
        base_parameters: KneeAngleFilterParameters,
        setup_metadata: Mapping[str, Any] | None,
    ) -> None:
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
        self._static_suggestion = {}
        self._dynamic_suggestion = {}
        self._recommended_parameters = {}
        self._report_path = None
        self._overview_path = None
        self._base_parameters = base_parameters
        self._setup_metadata = dict(setup_metadata) if setup_metadata is not None else None
        self._heldout_result = {}
        self._valid = False
        self._error = ""
        _write_suggestions_yaml(
            run_directory / SUGGESTIONS_FILENAME,
            status="awaiting_static",
            run_id=run_directory.name,
            static_suggestion=None,
            dynamic_suggestion=None,
            final_parameters=None,
        )
        self._recommendation_path = run_directory / SUGGESTIONS_FILENAME

    def _reset_run_locked(self) -> None:
        self._run_directory = None
        self._static_archive = None
        self._dynamic_archive = None
        self._recommendation_path = None
        self._static_suggestion = {}
        self._dynamic_suggestion = {}
        self._recommended_parameters = {}
        self._report_path = None
        self._overview_path = None
        self._base_parameters = None
        self._setup_metadata = None
        self._heldout_result = {}
        self._valid = False
        self._error = ""

    def _status_locked(self) -> CalibrationStatus:
        elapsed_s = 0.0
        progress = 0.0
        if self._active and self._started_ns:
            elapsed_s = max(0.0, (time.perf_counter_ns() - self._started_ns) / 1_000_000_000.0)
            progress = min(1.0, elapsed_s / self._duration_s) if self._duration_s > 0 else 0.0
        elif self._state in {
            "ANALYZING_STATIC",
            "AWAITING_DYNAMIC",
            "ANALYZING",
            "COMPLETE",
            "INVALID",
        }:
            progress = 1.0
        run_directory = str(self._run_directory.resolve()) if self._run_directory else ""
        return CalibrationStatus(
            state=self._state,
            active=self._active,
            current_test=self._current_test,
            elapsed_s=elapsed_s,
            duration_s=self._duration_s,
            progress=progress,
            run_id=self._run_directory.name if self._run_directory else "",
            run_directory=run_directory,
            export_path=run_directory,
            static_archive_path=str(self._static_archive or ""),
            dynamic_archive_path=str(self._dynamic_archive or ""),
            recommendation_path=str(self._recommendation_path or ""),
            static_suggestion=dict(self._static_suggestion),
            dynamic_suggestion=dict(self._dynamic_suggestion),
            recommended_parameters=dict(self._recommended_parameters),
            report_path=str(self._report_path or ""),
            overview_path=str(self._overview_path or ""),
            valid=self._valid,
            message=self._message,
            error=self._error,
        )

    def _restore_latest_completed_run(self) -> None:
        if not self.output_root.is_dir():
            return
        for run_directory in sorted(self.output_root.glob("calibration_run_*"), reverse=True):
            try:
                context = _load_heldout_context(run_directory)
                report = context["calibration_report"]
                self._run_directory = run_directory.resolve()
                self._static_archive = (run_directory / STATIC_ARCHIVE_FILENAME).resolve()
                self._dynamic_archive = (run_directory / DYNAMIC_ARCHIVE_FILENAME).resolve()
                self._recommendation_path = (run_directory / SUGGESTIONS_FILENAME).resolve()
                self._report_path = (run_directory / "calibration_report.json").resolve()
                overview = (run_directory / "calibration_overview.png").resolve()
                self._overview_path = overview if overview.is_file() else None
                self._base_parameters = context["parameters"]["baseline"]
                self._setup_metadata = dict(context["calibration_setup"])
                self._static_suggestion = dict(report["static_suggestion"])
                self._dynamic_suggestion = dict(report["dynamic_suggestion"])
                self._recommended_parameters = dict(report["recommended_parameters"])
                invalidation_path = run_directory / SETUP_INVALIDATION_FILENAME
                if invalidation_path.is_file():
                    try:
                        invalidation = json.loads(invalidation_path.read_text(encoding="utf-8"))
                        if (not isinstance(invalidation, dict)
                                or invalidation.get("schema_version") != "vision_imu_setup_invalidation_v1"
                                or not str(invalidation.get("reason", "")).strip()):
                            raise ValueError("Malformed calibration setup invalidation marker")
                        reason = str(invalidation["reason"]).strip()
                    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                        reason = f"Saved calibration invalidation marker is unreadable: {exc}"
                    self._valid = False
                    self._state = "INVALID"
                    self._message = reason
                    self._error = reason
                    self._heldout_result = {}
                    return
                self._valid = True
                self._state = "COMPLETE"
                self._message = "Completed Bayesian calibration restored; record a new held-out trial."
                heldout_path = run_directory / HELDOUT_REPORT_FILENAME
                if heldout_path.is_file():
                    loaded = json.loads(heldout_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        self._heldout_result = loaded
                return
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError,
                    zipfile.BadZipFile):
                continue


def analyze_static_calibration_stage(
    static_archive_path: str | Path,
    *,
    base_parameters: KneeAngleFilterParameters,
    output_directory: str | Path,
    calibration_setup: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the static stage and atomically update the shared suggestions YAML."""
    output_dir = Path(output_directory).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    static_archive = Path(static_archive_path).expanduser().resolve()
    static_data = load_archive_data(static_archive)
    gates = _static_quality_gates(static_data, base_parameters)
    failed_gates = [name for name, gate in gates.items() if not gate["passed"]]
    report_path = output_dir / "calibration_report.json"
    recommendation_path = output_dir / SUGGESTIONS_FILENAME
    run_id = output_dir.name
    setup = _normalize_setup_metadata(calibration_setup)

    if failed_gates:
        report = {
            "schema_version": "vision_imu_calibration_report_v2",
            "generated_at": datetime.now().astimezone().isoformat(),
            "status": "INVALID_STATIC",
            "method": "static_offset_bias_analysis",
            "quality_gates": gates,
            "failed_gates": failed_gates,
            "source_archives": {"static": static_archive.name, "dynamic": None},
            "baseline_parameters": _parameter_payload(base_parameters),
            "message": "Static data did not pass the quality gates. Repeat the static stage.",
        }
        if setup is not None:
            report["calibration_setup"] = setup
        _atomic_write_json(report_path, report)
        _write_suggestions_yaml(
            recommendation_path,
            status="invalid_static",
            run_id=run_id,
            static_suggestion=None,
            dynamic_suggestion=None,
            final_parameters=None,
            failed_gates=failed_gates,
        )
        return {
            "valid": False,
            "message": "Static stage invalid — repeat the static stage.",
            "error": ", ".join(failed_gates),
            "report_path": str(report_path),
            "recommendation_path": str(recommendation_path),
            "static_suggestion": {},
        }

    static_result = _derive_static_parameters(static_data, base_parameters)
    static_parameters = static_result["parameters"]
    static_suggestion = _parameter_payload(static_parameters)
    report = {
        "schema_version": "vision_imu_calibration_report_v2",
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "AWAITING_DYNAMIC",
        "method": "static_offset_bias_analysis",
        "quality_gates": gates,
        "source_archives": {"static": static_archive.name, "dynamic": None},
        "baseline_parameters": _parameter_payload(base_parameters),
        "static_suggestion": static_suggestion,
        "static_derivation": {
            key: value for key, value in static_result.items() if key != "parameters"
        },
    }
    if setup is not None:
        report["calibration_setup"] = setup
    _atomic_write_json(report_path, report)
    _write_suggestions_yaml(
        recommendation_path,
        status="awaiting_dynamic",
        run_id=run_id,
        static_suggestion=static_suggestion,
        dynamic_suggestion=None,
        final_parameters=None,
    )
    return {
        "valid": True,
        "message": "Static stage complete. Continue to dynamic calibration.",
        "error": "",
        "report_path": str(report_path),
        "recommendation_path": str(recommendation_path),
        "static_suggestion": static_suggestion,
    }


def analyze_calibration_run(
    static_archive_path: str | Path,
    dynamic_archive_path: str | Path,
    *,
    base_parameters: KneeAngleFilterParameters,
    output_directory: str | Path,
    calibration_setup: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_directory).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    static_archive = Path(static_archive_path).expanduser().resolve()
    dynamic_archive = Path(dynamic_archive_path).expanduser().resolve()
    static_data = load_archive_data(static_archive)
    dynamic_data = load_archive_data(dynamic_archive)

    static_gates = _static_quality_gates(static_data, base_parameters)
    static_failed = [name for name, gate in static_gates.items() if not gate["passed"]]
    static_result: dict[str, Any] | None = None
    static_suggestion: dict[str, Any] = {}
    if not static_failed:
        static_result = _derive_static_parameters(static_data, base_parameters)
        static_suggestion = _parameter_payload(static_result["parameters"])

    gates = _quality_gates(static_data, dynamic_data, base_parameters)
    failed_gates = [name for name, gate in gates.items() if not gate["passed"]]
    report_path = output_dir / "calibration_report.json"
    overview_path = output_dir / "calibration_overview.png"
    recommendation_path = output_dir / SUGGESTIONS_FILENAME
    run_id = output_dir.name
    setup = _normalize_setup_metadata(calibration_setup)

    optimization_details: dict[str, Any] = {
        "method": "existing_gp_bayesian_optimizer", "objective": "training_rmse_deg",
        "seed": BO_RANDOM_SEED, "requested_budget": BO_EVALUATION_BUDGET,
        "fixed_lag_ms": 0.0,
        "reference_sampling_policy": HELDOUT_PAIRING_POLICY,
        "parameter_bounds": parameter_bounds_payload(),
        "source_run_ids": {"calibration": run_id, "static": static_archive.parent.name,
                           "dynamic": dynamic_archive.parent.name},
        "source_sha256": {"static": _archive_sha256(static_archive),
                          "dynamic": _archive_sha256(dynamic_archive)},
    }
    optimization = None
    if not failed_gates:
        if static_result is None:
            raise RuntimeError("Static calibration result is unexpectedly unavailable")
        static_parameters = static_result["parameters"]
        # Owned copies keep all replay inputs and the immutable pairing plan
        # unchanged throughout the optimizer and both baseline comparisons.
        training_data = ArchiveData(raw_rows=[dict(row) for row in dynamic_data.raw_rows],
                                    sync_rows=[dict(row) for row in dynamic_data.sync_rows])
        started = time.perf_counter()
        cpu_started = time.process_time()
        try:
            fixed_pairs = _fixed_training_pairs(training_data)
            optimization_details["frozen_pair_count"] = len(fixed_pairs)
            baseline_dynamic = _evaluate_training_parameters(training_data, base_parameters, fixed_pairs)
            calibrated_dynamic = _evaluate_training_parameters(training_data, static_parameters, fixed_pairs)
            for label, metric in (("baseline", baseline_dynamic), ("static_calibrated", calibrated_dynamic)):
                feasible, reasons = _training_feasibility(metric, len(fixed_pairs))
                if not feasible:
                    raise ValueError(label + ": " + ", ".join(reasons))
            optimization = optimize_filter_parameters(
                static_parameters,
                lambda candidate: _evaluate_training_parameters(training_data, candidate, fixed_pairs),
                lambda metrics: _training_feasibility(metrics, len(fixed_pairs)),
                evaluation_budget=BO_EVALUATION_BUDGET, seed=BO_RANDOM_SEED,
            )
            optimization_details.update(
                evaluation_budget=optimization.evaluation_budget,
                initial_design_size=optimization.initial_design_size,
                candidate_pool_size=optimization.candidate_pool_size,
                trace=optimization.trace_payload(), gp_uncertainty=optimization.gp_uncertainty,
                frozen_pairs=[{"pose_host_monotonic_ns": pose_ns, "visual_knee_angle_deg": visual,
                               "imu_sample_index": index, "imu_host_monotonic_ns": imu_ns}
                              for pose_ns, visual, index, imu_ns in fixed_pairs])
        except (RuntimeError, ValueError) as exc:
            failed_gates.append("bayesian_training_feasibility")
            gates["bayesian_training_feasibility"] = {"passed": False, "value": str(exc),
                                                      "requirement": "finite training metrics with identical >=80 pairs; feasible BO result"}
            optimization_details["error"] = str(exc)
        finally:
            optimization_details["wall_seconds"] = time.perf_counter() - started
            optimization_details["cpu_seconds"] = time.process_time() - cpu_started

    if failed_gates:
        report = {
            "schema_version": "vision_imu_calibration_report_v2",
            "generated_at": datetime.now().astimezone().isoformat(),
            "status": "INVALID",
            "method": "synchronized_static_dynamic_gp_bayesian_training",
            "quality_gates": gates,
            "failed_gates": failed_gates,
            "source_archives": {
                "static": static_archive.name,
                "dynamic": dynamic_archive.name,
            },
            "baseline_parameters": _parameter_payload(base_parameters),
            "static_suggestion": static_suggestion or None,
            "optimization": optimization_details,
            "message": "Test data did not pass the quality gates. No settings were recommended.",
        }
        if setup is not None:
            report["calibration_setup"] = setup
        _atomic_write_json(report_path, report)
        _write_suggestions_yaml(
            recommendation_path,
            status="invalid",
            run_id=run_id,
            static_suggestion=static_suggestion or None,
            dynamic_suggestion=None,
            final_parameters=None,
            failed_gates=failed_gates,
            optimization=optimization_details,
        )
        _write_invalid_overview(overview_path, report)
        return {
            "valid": False,
            "message": "TEST INVALID — repeat the failed acquisition.",
            "error": ", ".join(failed_gates),
            "report_path": str(report_path),
            "overview_path": str(overview_path),
            "recommendation_path": str(recommendation_path),
            "static_suggestion": static_suggestion,
            "dynamic_suggestion": {},
            "recommended_parameters": {},
        }

    if optimization is None:
        raise RuntimeError("Bayesian result is unexpectedly unavailable")
    recommended_parameters = optimization.parameters
    recommended_dynamic = optimization.metrics
    recommended_static = _evaluate_training_parameters(static_data, recommended_parameters,
                                                        _fixed_training_pairs(static_data))

    improvement_percent = 0.0
    if baseline_dynamic.score > 1e-9:
        improvement_percent = 100.0 * (
            baseline_dynamic.score - recommended_dynamic.score
        ) / baseline_dynamic.score

    report = {
        "schema_version": "vision_imu_calibration_report_v2",
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "VALID",
        "method": "synchronized_static_dynamic_gp_bayesian_training",
        "reference": {
            "type": "vision_assisted_comparative_reference",
            "statement": (
                "Recommendations minimize disagreement with the synchronized visual reference "
                "inside these two tests; they are not absolute ground truth."
            ),
        },
        "quality_gates": gates,
        "source_archives": {
            "static": static_archive.name,
            "dynamic": dynamic_archive.name,
        },
        "baseline_parameters": _parameter_payload(base_parameters),
        "static_suggestion": static_suggestion,
        "dynamic_suggestion": _parameter_payload(recommended_parameters),
        "recommended_parameters": _parameter_payload(recommended_parameters),
        "static_derivation": {
            key: value for key, value in static_result.items() if key != "parameters"
        },
        "optimization": optimization_details,
        "metrics": {
            "static_recommended": recommended_static.to_dict(),
            "dynamic_baseline": baseline_dynamic.to_dict(),
            "dynamic_static_calibrated": calibrated_dynamic.to_dict(),
            "dynamic_recommended": recommended_dynamic.to_dict(),
            "dynamic_score_improvement_percent": improvement_percent,
            "candidate_count": optimization.evaluation_budget,
            "dynamic_training_rmse_improvement_percent": (100 * (calibrated_dynamic.rmse_deg - recommended_dynamic.rmse_deg)
                                                          / calibrated_dynamic.rmse_deg if calibrated_dynamic.rmse_deg > 1e-9 else 0.0),
        },
        "limitations": [
            "The visual estimator is a comparative reference, not a certified angle standard.",
            "The recommendation is valid only for the tested sensor mounting and movement range.",
            "Orientation conventions are preserved; this tool does not rewrite axis mapping automatically.",
            "Optimization and all reported comparisons use this training trial only; no held-out validation is claimed.",
            "Zero host-time lag is fixed; physical camera/inference latency is not identified or optimized here.",
        ],
    }
    if setup is not None:
        report["calibration_setup"] = setup
    _atomic_write_json(report_path, report)
    dynamic_suggestion = _parameter_payload(recommended_parameters)
    _write_suggestions_yaml(
        recommendation_path,
        status="complete",
        run_id=run_id,
        static_suggestion=static_suggestion,
        dynamic_suggestion=dynamic_suggestion,
        final_parameters=dynamic_suggestion,
        dynamic_score_improvement_percent=improvement_percent,
        optimization=optimization_details,
    )
    _write_calibration_overview(
        overview_path,
        dynamic_data,
        base_parameters,
        recommended_parameters,
        baseline_dynamic,
        recommended_dynamic,
        fixed_pairs=fixed_pairs,
        static_parameters=static_parameters,
    )
    return {
        "valid": True,
        "message": (
            "Bayesian training comparison exported; settings were not applied. "
            f"RMSE baseline/static/BO: {baseline_dynamic.rmse_deg:.2f}/{calibrated_dynamic.rmse_deg:.2f}/{recommended_dynamic.rmse_deg:.2f} deg; "
            f"MAE BO: {recommended_dynamic.mae_deg:.2f} deg; {recommended_dynamic.pair_count} identical pairs."
        ),
        "error": "",
        "report_path": str(report_path),
        "overview_path": str(overview_path),
        "recommendation_path": str(recommendation_path),
        "static_suggestion": static_suggestion,
        "dynamic_suggestion": dynamic_suggestion,
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


def validate_heldout_recording(
    calibration_run_directory: str | Path,
    heldout_archive_path: str | Path,
    *,
    current_setup: Mapping[str, Any] | None,
    unchanged_setup: bool,
) -> dict[str, Any]:
    """Evaluate frozen T2 snapshots on one independent recording.

    This path deliberately has no optimizer, bias/offset derivation, visual
    correction, or lag fitting. All candidates replay the same raw samples and
    the same frozen host-time pair identities.
    """
    run_directory = Path(calibration_run_directory).expanduser().resolve()
    report_path = run_directory / HELDOUT_REPORT_FILENAME
    archive = Path(heldout_archive_path).expanduser().resolve()
    archive_hash = _archive_sha256(archive) if archive.is_file() else ""
    validation_id = ""
    try:
        context = _load_heldout_context(run_directory)
        setup = _normalize_setup_metadata(current_setup)
        if not unchanged_setup:
            raise ValueError("Operator confirmation of unchanged setup is required")
        if setup is None or setup != context["calibration_setup"]:
            raise ValueError("Current physical/camera/filter setup does not match calibration")
        if not archive.is_file():
            raise ValueError("Held-out recording archive was not found")
        if archive_hash in set(context["training_sha256"].values()):
            raise ValueError("Held-out recording is a training archive or an identical copy")

        metadata = _load_recording_metadata(archive)
        validation_id = str(metadata.get("validation_id", ""))
        try:
            if uuid.UUID(hex=validation_id).hex != validation_id:
                raise ValueError
        except (ValueError, AttributeError):
            raise ValueError("Held-out recording has no valid immutable validation ID") from None
        if archive.stem != "heldout_" + validation_id:
            raise ValueError("Held-out archive name does not match its validation ID")
        if metadata.get("recording_purpose") != "vision_imu_filter_heldout_validation":
            raise ValueError("Latest recording was not captured for held-out validation")
        if metadata.get("calibration_run_id") != run_directory.name:
            raise ValueError("Held-out recording belongs to a different calibration run")
        if metadata.get("calibration_report_sha256") != context["calibration_report_sha256"]:
            raise ValueError("Held-out recording references a changed calibration report")
        if metadata.get("training_archive_sha256") != context["training_sha256"]:
            raise ValueError("Held-out training archive identities do not match")
        if metadata.get("parameter_snapshot_sha256") != context["parameter_sha256"]:
            raise ValueError("Held-out frozen parameter identities do not match")
        if metadata.get("fixed_lag_ms") != 0.0:
            raise ValueError("Held-out recording must retain fixed host lag 0 ms")
        if metadata.get("pairing_policy") != HELDOUT_PAIRING_POLICY:
            raise ValueError("Held-out pairing policy does not match training")
        archived_setup = _normalize_setup_metadata(metadata.get("calibration_setup"))
        if archived_setup != context["calibration_setup"]:
            raise ValueError("Held-out recording setup differs from the training setup")

        history_path = run_directory / f"heldout_validation_{validation_id}.json"
        if history_path.is_file():
            previous = json.loads(history_path.read_text(encoding="utf-8"))
            if (not isinstance(previous, dict)
                    or previous.get("validation_archive", {}).get("sha256") != archive_hash
                    or previous.get("calibration_report_sha256")
                    != context["calibration_report_sha256"]):
                raise ValueError("Validation ID already exists with different immutable inputs")
            _atomic_write_json(report_path, previous)
            return previous

        data = load_archive_data(archive)
        pairs = _fixed_training_pairs(data)
        gates = _heldout_quality_gates(
            data,
            context["parameters"]["baseline"],
            pairs,
            expected_leg=str(context["calibration_setup"].get("tracked_leg") or ""),
        )
        failed = [name for name, gate in gates.items() if not gate["passed"]]
        if failed:
            raise ValueError("Held-out data failed quality gates: " + ", ".join(failed))

        metrics: dict[str, EvaluationMetrics] = {}
        for name in ("baseline", "static_calibrated", "bo_recommended"):
            metric = _evaluate_training_parameters(data, context["parameters"][name], pairs)
            feasible, reasons = _training_feasibility(metric, len(pairs))
            if not feasible:
                raise ValueError(name + " changed held-out coverage: " + ", ".join(reasons))
            metrics[name] = metric

        static_rmse = metrics["static_calibrated"].rmse_deg
        bo_rmse = metrics["bo_recommended"].rmse_deg
        status = "IMPROVED_AGREEMENT" if bo_rmse < static_rmse else "NO_IMPROVEMENT"
        result = {
            "schema_version": "vision_imu_heldout_validation_v1",
            "generated_at": datetime.now().astimezone().isoformat(),
            "status": status,
            "calibration_run_id": run_directory.name,
            "calibration_report_sha256": context["calibration_report_sha256"],
            "training_archives": context["training_archives"],
            "validation_archive": {
                "id": validation_id,
                "name": archive.name,
                "sha256": archive_hash,
            },
            "calibration_setup": context["calibration_setup"],
            "unchanged_setup_confirmed": True,
            "frozen_parameters": context["parameter_payloads"],
            "parameter_snapshot_sha256": context["parameter_sha256"],
            "fixed_lag_ms": 0.0,
            "pairing_policy": HELDOUT_PAIRING_POLICY,
            "frozen_pair_count": len(pairs),
            "quality_gates": gates,
            "metrics": {name: metric.to_dict() for name, metric in metrics.items()},
            "bo_vs_static_rmse_improvement_percent": (
                100.0 * (static_rmse - bo_rmse) / static_rmse if static_rmse > 1e-9 else 0.0
            ),
            "training_rmse_deg": {
                "static_calibrated": context["training_rmse"]["static_calibrated"],
                "bo_recommended": context["training_rmse"]["bo_recommended"],
            },
            "interpretation": (
                "Independent comparative agreement with the synchronized visual reference; "
                "this is not clinical accuracy or proof of camera accuracy."
            ),
            "limitations": [
                "The visual estimate is a comparative reference, not absolute ground truth.",
                "Fixed host lag 0 ms does not establish zero camera or inference latency.",
                "The result applies only to the recorded person, mounting, leg, camera and profile.",
            ],
        }
        _write_json_exclusive(history_path, result)
        _atomic_write_json(report_path, result)
        return result
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        invalid = {
            "schema_version": "vision_imu_heldout_validation_v1",
            "generated_at": datetime.now().astimezone().isoformat(),
            "status": "INVALID",
            "calibration_run_id": run_directory.name,
            "validation_archive": {
                "id": validation_id,
                "name": archive.name,
                "sha256": archive_hash,
            },
            "error": str(exc),
            "interpretation": "No held-out comparison is valid; correct the acquisition/setup issue and retry.",
        }
        if run_directory.is_dir():
            _atomic_write_json(report_path, invalid)
        return invalid


def _load_heldout_context(run_directory: Path) -> dict[str, Any]:
    report_path = (run_directory / "calibration_report.json").resolve()
    if not report_path.is_file():
        raise ValueError("Completed calibration report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (not isinstance(report, dict)
            or report.get("schema_version") != "vision_imu_calibration_report_v2"
            or report.get("status") != "VALID"
            or report.get("method") != "synchronized_static_dynamic_gp_bayesian_training"):
        raise ValueError("Calibration run is not a supported completed T2 run")
    optimization = report.get("optimization")
    if not isinstance(optimization, dict) or optimization.get("fixed_lag_ms") != 0.0:
        raise ValueError("Calibration report does not contain the frozen zero-lag policy")
    if optimization.get("reference_sampling_policy") != HELDOUT_PAIRING_POLICY:
        raise ValueError("Calibration report pairing policy is unsupported")

    source_names = report.get("source_archives")
    source_hashes = optimization.get("source_sha256")
    if (source_names != {"static": STATIC_ARCHIVE_FILENAME, "dynamic": DYNAMIC_ARCHIVE_FILENAME}
            or not isinstance(source_hashes, dict)):
        raise ValueError("Calibration training archive identities are incomplete")
    training_archives: dict[str, dict[str, str]] = {}
    setup = _normalize_setup_metadata(report.get("calibration_setup"))
    required_setup = {
        "description", "camera_profile", "camera_source", "pose_backend", "pose_tuning",
        "tracked_leg", "imu_source", "imu_transport", "filter_parameters",
    }
    if (setup is None or not required_setup.issubset(setup)
            or not str(setup.get("description", "")).strip()
            or not str(setup.get("camera_profile", "")).strip()
            or not str(setup.get("camera_source", "")).strip()
            or not str(setup.get("pose_backend", "")).strip()
            or setup.get("tracked_leg") not in {"left", "right"}
            or not isinstance(setup.get("pose_tuning"), dict)):
        raise ValueError("Calibration run has no physical setup metadata")
    for stage, name in source_names.items():
        path = (run_directory / name).resolve()
        if path.parent != run_directory.resolve() or not path.is_file():
            raise ValueError(f"Calibration {stage} training archive is missing")
        digest = _archive_sha256(path)
        if source_hashes.get(stage) != digest:
            raise ValueError(f"Calibration {stage} archive hash changed")
        metadata = _load_recording_metadata(path)
        if (metadata.get("recording_purpose") != "vision_imu_filter_calibration"
                or metadata.get("calibration_test_type") != stage
                or _normalize_setup_metadata(metadata.get("calibration_setup")) != setup):
            raise ValueError(f"Calibration {stage} archive setup metadata is unsupported")
        training_archives[stage] = {"id": name, "name": name, "sha256": digest}

    payloads = {
        "baseline": _validated_parameter_payload(report.get("baseline_parameters"), "baseline"),
        "static_calibrated": _validated_parameter_payload(report.get("static_suggestion"), "static"),
        "bo_recommended": _validated_parameter_payload(report.get("recommended_parameters"), "BO"),
    }
    if _validated_parameter_payload(setup.get("filter_parameters"), "setup filter") != payloads["baseline"]:
        raise ValueError("Calibration setup filter metadata differs from the frozen baseline")
    dynamic_payload = _validated_parameter_payload(report.get("dynamic_suggestion"), "dynamic")
    if dynamic_payload != payloads["bo_recommended"]:
        raise ValueError("BO recommendation snapshots disagree inside the calibration report")
    optimized = {"alpha_stable", "alpha_dynamic", "max_accel_correction_deg"}
    if ({key: value for key, value in payloads["static_calibrated"].items() if key not in optimized}
            != {key: value for key, value in payloads["bo_recommended"].items() if key not in optimized}):
        raise ValueError("BO snapshot changed parameters outside the three trained dynamic fields")
    parameters = {name: KneeAngleFilterParameters.from_mapping(payload)
                  for name, payload in payloads.items()}
    parameter_hashes = {name: _mapping_sha256(payload) for name, payload in payloads.items()}
    metrics = report.get("metrics")
    try:
        static_rmse = float(metrics["dynamic_static_calibrated"]["rmse_deg"])
        bo_rmse = float(metrics["dynamic_recommended"]["rmse_deg"])
    except (TypeError, KeyError, ValueError):
        raise ValueError("Calibration report training metrics are incomplete") from None
    if not math.isfinite(static_rmse) or not math.isfinite(bo_rmse):
        raise ValueError("Calibration report training metrics are nonfinite")
    return {
        "calibration_report": report,
        "calibration_report_sha256": _archive_sha256(report_path),
        "calibration_setup": setup,
        "training_archives": training_archives,
        "training_sha256": {name: item["sha256"] for name, item in training_archives.items()},
        "parameter_payloads": payloads,
        "parameter_sha256": parameter_hashes,
        "parameters": parameters,
        "training_rmse": {"static_calibrated": static_rmse, "bo_recommended": bo_rmse},
    }


def _heldout_quality_gates(
    data: ArchiveData,
    parameters: KneeAngleFilterParameters,
    pairs: Sequence[tuple[int, float, int, int]],
    *,
    expected_leg: str,
) -> dict[str, dict[str, Any]]:
    valid_sync = _valid_sync_rows(data.sync_rows)
    raw_valid = _raw_rows_valid(data.raw_rows)
    angles = [float(row["visual_knee_angle_deg"]) for row in valid_sync]
    legs = {str(row.get("tracked_leg", "")) for row in valid_sync if row.get("tracked_leg")}
    base_metrics = (_evaluate_training_parameters(data, parameters, pairs)
                    if raw_valid and pairs else _empty_metrics())

    def gate(passed: bool, value: Any, requirement: str) -> dict[str, Any]:
        return {"passed": bool(passed), "value": value, "requirement": requirement}

    return {
        "heldout_raw_imu_valid": gate(raw_valid, raw_valid,
                                      "finite ordered raw samples with consistent scaled columns"),
        "heldout_imu_samples": gate(len(data.raw_rows) >= 800, len(data.raw_rows),
                                    ">= 800 raw IMU samples"),
        "heldout_imu_ready_fraction": gate(_imu_ready_fraction(data.raw_rows) >= .95,
                                            _imu_ready_fraction(data.raw_rows),
                                            ">= 95% samples with both IMUs ready"),
        "heldout_valid_visual_rows": gate(len(valid_sync) >= 80, len(valid_sync),
                                          ">= 80 synchronized valid visual rows"),
        "heldout_valid_comparison_pairs": gate(len(pairs) >= 80, len(pairs),
                                                ">= 80 frozen camera/IMU pairs"),
        "heldout_candidate_pair_coverage": gate(base_metrics.pair_count == len(pairs),
                                                 base_metrics.pair_count,
                                                 "baseline replay must cover every frozen pair"),
        "heldout_tracking_fraction": gate(_visual_valid_fraction(data.sync_rows) >= MIN_VISUAL_VALID_FRACTION,
                                           _visual_valid_fraction(data.sync_rows),
                                           f">= {MIN_VISUAL_VALID_FRACTION:.0%}"),
        "heldout_sync_p95_ms": gate(_sync_p95_ms(valid_sync) <= MAX_SYNC_ERROR_MS,
                                    _sync_p95_ms(valid_sync), f"<= {MAX_SYNC_ERROR_MS:.0f} ms"),
        "heldout_visual_range_deg": gate(bool(angles) and max(angles) - min(angles) >= 30.0,
                                         max(angles) - min(angles) if angles else 0.0,
                                         ">= 30 deg movement range"),
        "heldout_sequence_gap_fraction": gate(
            raw_valid and _sequence_gap_fraction(data.raw_rows) <= .01,
            _sequence_gap_fraction(data.raw_rows) if raw_valid else 1.0,
            "<= 1% missing sequence steps"),
        "heldout_tracked_leg": gate(not expected_leg or legs == {expected_leg}, sorted(legs),
                                    "all valid frames must use the calibrated tracked leg"),
    }


def _load_recording_metadata(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        try:
            payload = json.loads(archive.read("recording_metadata.json").decode("utf-8-sig"))
        except KeyError:
            raise ValueError(f"Archive {path.name} has no recording metadata") from None
    if not isinstance(payload, dict):
        raise ValueError(f"Archive {path.name} has malformed recording metadata")
    return payload


def _normalize_setup_metadata(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Calibration setup metadata must be an object")
    try:
        return json.loads(json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False))
    except (TypeError, ValueError):
        raise ValueError("Calibration setup metadata must contain finite JSON values") from None


def _validated_parameter_payload(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing {label} parameter snapshot")
    required = set(asdict(KneeAngleFilterParameters()))
    if not required.issubset(value):
        raise ValueError(f"Incomplete {label} parameter snapshot")
    return _parameter_payload(KneeAngleFilterParameters.from_mapping(value))


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


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
    return _metrics_from_pairs(pairs, lag_ms=lag_ms)


def _metrics_from_pairs(pairs: Sequence[tuple[int, float, float]], *, lag_ms: float) -> EvaluationMetrics:
    if not pairs:
        return _empty_metrics()
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


def _fixed_training_pairs(data: ArchiveData) -> tuple[tuple[int, float, int, int], ...]:
    """Freeze actual optical rows and nearest RAW IMU identity at host lag 0.

    Eligibility does not depend on candidate filter outputs. Never interpolate
    optical frames or bridge a not-ready nearest IMU sample with another sample.
    """
    raw_times = [int(row["host_monotonic_ns"]) for row in data.raw_rows]
    pairs = []
    previous_pose_ns = -1
    for row in _valid_sync_rows(data.sync_rows):
        pose_ns = int(row["host_monotonic_ns"])
        if pose_ns <= previous_pose_ns:
            raise ValueError("Optical timestamps must be strictly ordered; duplicated frames are not extra observations")
        previous_pose_ns = pose_ns
        index = _nearest_index(raw_times, pose_ns)
        if index is None or abs(raw_times[index] - pose_ns) > 25_000_000:
            continue
        raw = data.raw_rows[index]
        if not (_to_bool(raw.get("thigh_ok")) and _to_bool(raw.get("shank_ok"))):
            continue
        recorded_index = row.get("imu_sample_index")
        if recorded_index not in (None, "") and int(recorded_index) != int(raw["sample_index"]):
            raise ValueError("Recorded camera/IMU pairing differs from nearest host-time sample")
        actual_delta = (raw_times[index] - pose_ns) / 1_000_000
        if abs(float(row["sync_delta_ms"]) - actual_delta) > 0.001:
            raise ValueError("Recorded sync delta differs from raw sample timestamp")
        pairs.append((pose_ns, float(row["visual_knee_angle_deg"]),
                      int(raw["sample_index"]), raw_times[index]))
    return tuple(pairs)


def _evaluate_training_parameters(
    data: ArchiveData, parameters: KneeAngleFilterParameters,
    pairs: Sequence[tuple[int, float, int, int]],
) -> EvaluationMetrics:
    estimates = replay_filter(data.raw_rows, parameters)
    values = []
    for pose_ns, visual, sample_index, imu_ns in pairs:
        estimate = estimates.get(sample_index)
        if (estimate is None or not estimate.valid or estimate.host_monotonic_ns != imu_ns
                or estimate.knee_angle_deg is None or not math.isfinite(estimate.knee_angle_deg)):
            # Feasibility requires the original full pair count, so a candidate
            # cannot improve its objective by dropping troublesome observations.
            continue
        values.append((pose_ns, visual, estimate.knee_angle_deg))
    return _metrics_from_pairs(values, lag_ms=0.0)


def _training_feasibility(metrics: EvaluationMetrics, pair_count: int) -> tuple[bool, list[str]]:
    reasons = []
    if metrics.pair_count < 80:
        reasons.append("insufficient_training_pairs")
    if metrics.pair_count != pair_count:
        reasons.append("candidate_changed_pair_coverage")
    if not all(math.isfinite(v) for v in asdict(metrics).values()):
        reasons.append("nonfinite_training_metrics")
    return not reasons, reasons


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
    if not _raw_rows_valid(data.raw_rows):
        raise ValueError("Static raw IMU samples must be finite, ordered and internally consistent")
    ready_rows = [row for row in data.raw_rows
                  if _to_bool(row.get("thigh_ok")) and _to_bool(row.get("shank_ok"))]
    if not ready_rows:
        raise ValueError("Static test has no ready IMU samples")
    if _static_motion_p95(ready_rows, base_parameters) > 5.0:
        raise ValueError("Static test is not stationary")
    thigh_rates = [_signed_axis_rate(row, "thigh", base_parameters) for row in ready_rows]
    shank_rates = [_signed_axis_rate(row, "shank", base_parameters) for row in ready_rows]
    thigh_bias = statistics.median(thigh_rates)
    shank_bias = statistics.median(shank_rates)
    residual_rates = [
        *(abs(rate - thigh_bias) for rate in thigh_rates),
        *(abs(rate - shank_bias) for rate in shank_rates),
    ]
    deadband = min(2.5, max(0.15, 1.5 * _percentile(residual_rates, 95.0)))
    gyro_calibrated = replace(
        base_parameters,
        thigh_gyro_bias_deg_s=thigh_bias,
        shank_gyro_bias_deg_s=shank_bias,
        gyro_deadband_deg_s=deadband,
    )
    frozen_pairs = _fixed_training_pairs(data)
    estimates = replay_filter(data.raw_rows, gyro_calibrated)
    pairs = [(pose_ns, visual, estimates[index].knee_angle_deg)
             for pose_ns, visual, index, _ in frozen_pairs
             if estimates[index].valid and estimates[index].knee_angle_deg is not None
             and math.isfinite(estimates[index].knee_angle_deg)]
    if not pairs:
        raise ValueError("Static test has no synchronized valid camera/IMU pairs")
    measured_angles = [visual for _, visual, _ in pairs]
    errors = [visual - imu for _, visual, imu in pairs]
    if (_percentile(measured_angles, 95) - _percentile(measured_angles, 5) > 3.0
            or _percentile(errors, 95) - _percentile(errors, 5) > 3.0):
        raise ValueError("Static optical/alignment signal is moving or has an inconsistent angular branch")
    # Mirror replay already unwraps +/-180 crossings continuously. Keep its
    # actual branch and a linear optical-minus-IMU offset: wrapping this delta
    # into +/-180 would shift the displayed posterior angle by a full turn.
    angle_offset_delta = statistics.median(errors)
    parameters = replace(gyro_calibrated,
                         straight_leg_offset_deg=base_parameters.straight_leg_offset_deg + angle_offset_delta)
    parameters.validate()
    return {
        "parameters": parameters,
        "angle_offset_delta_deg": angle_offset_delta,
        "thigh_gyro_bias_deg_s": thigh_bias,
        "shank_gyro_bias_deg_s": shank_bias,
        "gyro_deadband_deg_s": deadband,
        "pair_count": len(pairs),
        "measured_optical_angle_deg": statistics.median(measured_angles),
        "alignment": "relative optical alignment after gyro bias estimation; no forced 180 deg",
    }



def _derive_physics_static_parameters(
    data: ArchiveData,
    base_parameters: KneeAngleFilterParameters,
) -> dict[str, Any]:
    """Compute a *candidate* static IMU calibration without camera readings.

    Assumes the operator held the leg stationary in full extension and both IMUs
    are mounted consistently. The result is not an independently verified knee
    angle; static quality gates must pass before the active workflow uses it.

    Deliberately not called by the current calibration workflow (T05a).
    """
    if not data.raw_rows:
        raise RuntimeError("Static test contains no raw IMU samples")

    straight_offsets: list[float] = []
    thigh_rates: list[float] = []
    shank_rates: list[float] = []
    for row in data.raw_rows:
        thigh_acc = _static_acceleration_angle_deg(row, "thigh", base_parameters)
        shank_acc = _static_acceleration_angle_deg(row, "shank", base_parameters)
        thigh_segment = thigh_acc - base_parameters.thigh_offset_deg
        shank_segment = shank_acc - base_parameters.shank_offset_deg
        relative = (shank_segment - thigh_segment + 180.0) % 360.0 - 180.0
        straight_offsets.append(base_parameters.relative_angle_sign * relative)
        thigh_rates.append(_signed_axis_rate(row, "thigh", base_parameters))
        shank_rates.append(_signed_axis_rate(row, "shank", base_parameters))

    thigh_bias = statistics.median(thigh_rates)
    shank_bias = statistics.median(shank_rates)
    thigh_residuals = [rate - thigh_bias for rate in thigh_rates]
    shank_residuals = [rate - shank_bias for rate in shank_rates]
    thigh_noise_sigma = _static_robust_sigma(thigh_residuals)
    shank_noise_sigma = _static_robust_sigma(shank_residuals)
    deadband = min(2.5, max(0.05, 3.0 * max(thigh_noise_sigma, shank_noise_sigma)))
    straight_offset = statistics.median(straight_offsets)

    parameters = replace(
        base_parameters,
        straight_leg_offset_deg=straight_offset,
        thigh_gyro_bias_deg_s=thigh_bias,
        shank_gyro_bias_deg_s=shank_bias,
        gyro_deadband_deg_s=deadband,
    )
    parameters.validate()
    return {
        "parameters": parameters,
        "physical_reference_condition": "assumed_full_extension",
        "straight_leg_offset_deg": straight_offset,
        "straight_leg_offset_std_deg": statistics.pstdev(straight_offsets) if len(straight_offsets) > 1 else 0.0,
        "thigh_gyro_bias_deg_s": thigh_bias,
        "shank_gyro_bias_deg_s": shank_bias,
        "thigh_gyro_noise_sigma_deg_s": thigh_noise_sigma,
        "shank_gyro_noise_sigma_deg_s": shank_noise_sigma,
        "gyro_deadband_deg_s": deadband,
        "sample_count": len(data.raw_rows),
        "used_visual_reference": False,
    }


def _static_acceleration_angle_deg(
    row: Mapping[str, str], prefix: str, parameters: KneeAngleFilterParameters,
) -> float:
    ax, ay, az = (float(row[f"{prefix}_{axis}_g"]) for axis in ("ax", "ay", "az"))
    if parameters.rotation_axis == 0:
        axis_1, axis_2 = parameters.acc_axis_1_sign * ay, parameters.acc_axis_2_sign * az
    elif parameters.rotation_axis == 2:
        axis_1, axis_2 = parameters.acc_axis_1_sign * ay, parameters.acc_axis_2_sign * ax
    else:
        axis_1, axis_2 = parameters.acc_axis_1_sign * az, parameters.acc_axis_2_sign * ax
    return math.degrees(math.atan2(axis_1, axis_2))


def _static_robust_sigma(values: Sequence[float]) -> float:
    median = statistics.median(values)
    return 1.4826 * statistics.median(abs(value - median) for value in values)


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


def _static_quality_gates(
    static_data: ArchiveData,
    parameters: KneeAngleFilterParameters,
) -> dict[str, dict[str, Any]]:
    static_sync = _valid_sync_rows(static_data.sync_rows)
    static_valid_fraction = _visual_valid_fraction(static_data.sync_rows)
    static_sync_p95 = _sync_p95_ms(static_sync)
    raw_valid = _raw_rows_valid(static_data.raw_rows)
    static_gap_fraction = _sequence_gap_fraction(static_data.raw_rows) if raw_valid else 1.0
    static_ready_fraction = _imu_ready_fraction(static_data.raw_rows)
    static_base_pairs = len(
        _paired_values(
            static_data.sync_rows,
            replay_filter(static_data.raw_rows, parameters),
            lag_ms=0.0,
        )
    ) if raw_valid else 0
    ready_rows = [row for row in static_data.raw_rows
                  if _to_bool(row.get("thigh_ok")) and _to_bool(row.get("shank_ok"))]
    static_motion = _static_motion_p95(ready_rows, parameters) if raw_valid else math.inf
    alignment_error = ""
    try:
        _derive_static_parameters(static_data, parameters)
    except (ValueError, KeyError, OverflowError) as exc:
        alignment_error = str(exc)

    def gate(passed: bool, value: Any, requirement: str) -> dict[str, Any]:
        return {"passed": bool(passed), "value": value, "requirement": requirement}

    return {
        "static_raw_imu_valid": gate(raw_valid, raw_valid,
                                     "finite ordered raw samples with consistent scaled columns"),
        "static_relative_alignment": gate(not alignment_error, alignment_error or "stationary aligned pairs",
                                          "stationary optical/IMU pairs on a consistent unwrapped branch"),
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
    }


def _quality_gates(
    static_data: ArchiveData,
    dynamic_data: ArchiveData,
    parameters: KneeAngleFilterParameters,
) -> dict[str, dict[str, Any]]:
    static_extra_gates = _static_quality_gates(static_data, parameters)
    static_sync = _valid_sync_rows(static_data.sync_rows)
    dynamic_sync = _valid_sync_rows(dynamic_data.sync_rows)
    static_valid_fraction = _visual_valid_fraction(static_data.sync_rows)
    dynamic_valid_fraction = _visual_valid_fraction(dynamic_data.sync_rows)
    static_sync_p95 = _sync_p95_ms(static_sync)
    dynamic_sync_p95 = _sync_p95_ms(dynamic_sync)
    static_raw_valid = _raw_rows_valid(static_data.raw_rows)
    dynamic_raw_valid = _raw_rows_valid(dynamic_data.raw_rows)
    static_gap_fraction = _sequence_gap_fraction(static_data.raw_rows) if static_raw_valid else 1.0
    dynamic_gap_fraction = _sequence_gap_fraction(dynamic_data.raw_rows) if dynamic_raw_valid else 1.0
    static_ready_fraction = _imu_ready_fraction(static_data.raw_rows)
    dynamic_ready_fraction = _imu_ready_fraction(dynamic_data.raw_rows)
    static_base_pairs = len(
        _paired_values(static_data.sync_rows, replay_filter(static_data.raw_rows, parameters), lag_ms=0.0)
    ) if static_raw_valid else 0
    dynamic_base_pairs = len(
        _paired_values(dynamic_data.sync_rows, replay_filter(dynamic_data.raw_rows, parameters), lag_ms=0.0)
    ) if dynamic_raw_valid else 0
    dynamic_angles = [float(row["visual_knee_angle_deg"]) for row in dynamic_sync]
    dynamic_range = max(dynamic_angles) - min(dynamic_angles) if dynamic_angles else 0.0
    static_ready_rows = [row for row in static_data.raw_rows
                         if _to_bool(row.get("thigh_ok")) and _to_bool(row.get("shank_ok"))]
    static_motion = _static_motion_p95(static_ready_rows, parameters) if static_raw_valid else math.inf

    def gate(passed: bool, value: Any, requirement: str) -> dict[str, Any]:
        return {"passed": bool(passed), "value": value, "requirement": requirement}

    return {
        "static_raw_imu_valid": static_extra_gates["static_raw_imu_valid"],
        "static_relative_alignment": static_extra_gates["static_relative_alignment"],
        "dynamic_raw_imu_valid": gate(dynamic_raw_valid, dynamic_raw_valid,
                                      "finite ordered raw samples with consistent scaled columns"),
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
        timestamp = _to_float(row.get("host_monotonic_ns"))
        if (visual is None or not 0 <= visual <= 180 or sync_delta is None
                or abs(sync_delta) > MAX_SYNC_ERROR_MS or timestamp is None
                or timestamp < 0 or not timestamp.is_integer()):
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
    # Equal thigh/shank rotation is still movement; relative speed alone could
    # incorrectly classify it as a stationary hold and absorb it into biases.
    relative_rates = [max(abs(_signed_axis_rate(row, "shank", parameters)),
                          abs(_signed_axis_rate(row, "thigh", parameters)),
                          abs(_signed_axis_rate(row, "shank", parameters)
                              - _signed_axis_rate(row, "thigh", parameters))) for row in rows]
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
        host_monotonic_ns=_integer_value(row["host_monotonic_ns"]),
        schema_version=_integer_value(row["schema_version"]),
        sample_index=_integer_value(row["sample_index"]),
        mcu_timestamp_us=_integer_value(row["mcu_timestamp_us"]),
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


def _integer_value(value: Any) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("Expected a finite integer")
        return int(numeric)


def _raw_rows_valid(rows: Sequence[Mapping[str, str]]) -> bool:
    previous_ns = previous_index = -1
    for row in rows:
        try:
            sample = _sample_from_row(row)
            if sample.host_monotonic_ns <= previous_ns or sample.sample_index <= previous_index:
                return False
            if not 0 <= sample.mcu_timestamp_us <= 0xFFFFFFFF or sample.schema_version < 1:
                return False
            for name, value in asdict(sample).items():
                if name in {"thigh_ok", "shank_ok"}:
                    continue
                original = _to_float(row.get(name))
                if original is None or not original.is_integer():
                    return False
                if name.endswith("_raw") and not -32768 <= value <= 32767:
                    return False
            for prefix in ("thigh", "shank"):
                if getattr(sample, prefix + "_ok") and not any(
                        getattr(sample, f"{prefix}_a{axis}_raw") for axis in "xyz"):
                    return False
            # Replay consumes raw integers, while bias derivation uses scaled CSV
            # columns. Reject contradictory inputs; allow half-LSB rounding in
            # older synthetic datasets without altering any recorded values.
            for name, value in sample.to_record().items():
                if name.endswith(("_g", "_dps")):
                    numeric = _to_float(row.get(name))
                    lsb = ACCEL_G_PER_LSB if name.endswith("_g") else GYRO_DPS_PER_LSB
                    if numeric is None or abs(numeric - value) > lsb * .500001:
                        return False
            previous_ns, previous_index = sample.host_monotonic_ns, sample.sample_index
        except (ValueError, TypeError, KeyError, OverflowError):
            return False
    return bool(rows)


def _archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as archive:
        for chunk in iter(lambda: archive.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_suggestions_yaml(
    path: Path,
    *,
    status: str,
    run_id: str,
    static_suggestion: Mapping[str, Any] | None,
    dynamic_suggestion: Mapping[str, Any] | None,
    final_parameters: Mapping[str, Any] | None,
    dynamic_score_improvement_percent: float | None = None,
    failed_gates: Sequence[str] | None = None,
    error: str | None = None,
    optimization: Mapping[str, Any] | None = None,
) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to export calibration suggestions") from exc
    static_stage_status = (
        "valid"
        if static_suggestion
        else "invalid"
        if status in {"invalid_static", "failed_static"}
        else "pending"
    )
    dynamic_stage_status = (
        "valid"
        if dynamic_suggestion
        else "invalid"
        if status == "invalid"
        else "pending"
    )
    payload = {
        "schema_version": "vision_imu_filter_suggestions_v2",
        "calibration_id": run_id,
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": str(status),
        "method": "static_optical_alignment_plus_dynamic_gp_bayesian_rmse",
        "reference": "synchronized MoveNet knee-angle comparison",
        "source_trials": {
            "static": STATIC_ARCHIVE_FILENAME,
            "dynamic": DYNAMIC_ARCHIVE_FILENAME,
        },
        "static_stage": {
            "status": static_stage_status,
            "suggestion": dict(static_suggestion) if static_suggestion else None,
        },
        "dynamic_stage": {
            "status": dynamic_stage_status,
            "suggestion": dict(dynamic_suggestion) if dynamic_suggestion else None,
        },
        "final_recommended_settings": (
            dict(final_parameters) if final_parameters is not None else None
        ),
    }
    if dynamic_score_improvement_percent is not None:
        payload["dynamic_score_improvement_percent"] = float(
            dynamic_score_improvement_percent
        )
    if failed_gates:
        payload["failed_quality_gates"] = list(failed_gates)
    if error:
        payload["last_error"] = str(error)
    if optimization is not None:
        payload["optimization"] = dict(optimization)
    if final_parameters is not None:
        # Keep the final YAML directly loadable by the existing filter loader.
        payload["stm32_control"] = {"knee_angle": dict(final_parameters)}
    _atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_calibration_overview(
    path: Path,
    data: ArchiveData,
    baseline_parameters: KneeAngleFilterParameters,
    recommended_parameters: KneeAngleFilterParameters,
    baseline_metrics: EvaluationMetrics,
    recommended_metrics: EvaluationMetrics,
    *,
    fixed_pairs: Sequence[tuple[int, float, int, int]] | None = None,
    static_parameters: KneeAngleFilterParameters | None = None,
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
    if fixed_pairs is not None:
        t0 = fixed_pairs[0][0]
        times = [(pose_ns - t0) / 1e9 for pose_ns, _, _, _ in fixed_pairs]
        visual = [angle for _, angle, _, _ in fixed_pairs]
        baseline_values = [baseline[index].knee_angle_deg for _, _, index, _ in fixed_pairs]
        recommended_values = [recommended[index].knee_angle_deg for _, _, index, _ in fixed_pairs]

    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(times, visual, label="Visual reference", linewidth=1.8)
    axis.plot(times, baseline_values, label="Baseline IMU", linewidth=1.2)
    axis.plot(times, recommended_values, label="Recommended IMU", linewidth=1.2)
    if static_parameters is not None and fixed_pairs is not None:
        calibrated = replay_filter(data.raw_rows, static_parameters)
        axis.plot(times, [calibrated[index].knee_angle_deg for _, _, index, _ in fixed_pairs],
                  label="Static-calibrated IMU", linewidth=1.0)
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Posterior knee angle [deg]")
    axis.grid(True, alpha=0.3)
    axis.legend()
    axis.set_title(
        "Vision–IMU training comparison (fixed host lag 0)\n"
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

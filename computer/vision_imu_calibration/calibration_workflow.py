from __future__ import annotations

import csv
import json
import math
import os
import shutil
import statistics
import threading
import time
import zipfile
from bisect import bisect_left
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .bayesian_optimizer import (
    BO_EVALUATION_BUDGET,
    BO_RANDOM_SEED,
    optimize_filter_parameters,
    parameter_bounds_payload,
    sample_random_parameters,
)
from .imu_filter import KneeAngleEstimate, KneeAngleFilterMirror, KneeAngleFilterParameters
from .imu_serial import ImuCalibrationSample

STATIC_TEST_DURATION_S = 12.0
DYNAMIC_TEST_DURATION_S = 25.0
MAX_SYNC_ERROR_MS = 20.0
MIN_VISUAL_VALID_FRACTION = 0.70
MIN_ACCEPTED_DYNAMIC_CYCLES = 4
HELD_OUT_CYCLE_STRIDE = 4
SUGGESTIONS_FILENAME = "imu_filter_suggestions.yaml"
STATIC_ARCHIVE_FILENAME = "static_trial.zip"
DYNAMIC_ARCHIVE_FILENAME = "dynamic_trial.zip"


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
    result_summary: dict[str, Any]
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
        self._result_summary: dict[str, Any] = {}
        self._report_path: Path | None = None
        self._overview_path: Path | None = None
        self._base_parameters: KneeAngleFilterParameters | None = None
        self._valid = False
        self._message = "Start calibration with the static stage."
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
                if not self._static_suggestion or self._state not in {"AWAITING_DYNAMIC", "INVALID"}:
                    raise RuntimeError("The current calibration run is not ready for the dynamic test")
                if self._base_parameters != base_parameters:
                    raise RuntimeError(
                        "The IMU filter settings changed after the static test; repeat the static test"
                    )
                self._dynamic_archive = None
                expected_dynamic = self._run_directory / DYNAMIC_ARCHIVE_FILENAME
                expected_dynamic.unlink(missing_ok=True)
                self._dynamic_suggestion = {}
                self._recommended_parameters = dict(self._static_suggestion)
                self._result_summary = {}
                self._report_path = None
                self._overview_path = None
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
                "Keep the leg completely still until the static test finishes."
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
                self._message = "Analyzing both stages and completing the shared suggestions file…"

        if static_archive is None:
            self.fail("Calibration analysis state is incomplete")
            return self.status()

        if test == "static":
            try:
                result = analyze_static_calibration_stage(
                    static_archive,
                    base_parameters=base_parameters,
                    output_directory=run_directory,
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
            )
        except Exception as exc:
            self.fail(f"Calibration analysis failed: {exc}", stage="dynamic")
            return self.status()

        with self._lock:
            self._recommendation_path = _optional_path(result.get("recommendation_path"))
            self._static_suggestion = dict(result.get("static_suggestion") or {})
            self._dynamic_suggestion = dict(result.get("dynamic_suggestion") or {})
            self._recommended_parameters = dict(result.get("recommended_parameters") or {})
            self._result_summary = dict(result.get("result_summary") or {})
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
        self._static_suggestion = {}
        self._dynamic_suggestion = {}
        self._recommended_parameters = {}
        self._result_summary = {}
        self._report_path = None
        self._overview_path = None
        self._base_parameters = base_parameters
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
        self._result_summary = {}
        self._report_path = None
        self._overview_path = None
        self._base_parameters = None
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
            result_summary=dict(self._result_summary),
            report_path=str(self._report_path or ""),
            overview_path=str(self._overview_path or ""),
            valid=self._valid,
            message=self._message,
            error=self._error,
        )


def analyze_static_calibration_stage(
    static_archive_path: str | Path,
    *,
    base_parameters: KneeAngleFilterParameters,
    output_directory: str | Path,
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

    if failed_gates:
        report = {
            "schema_version": "vision_imu_calibration_report_v2",
            "generated_at": datetime.now().astimezone().isoformat(),
            "status": "INVALID_STATIC",
            "method": "physics_based_static_calibration",
            "quality_gates": gates,
            "failed_gates": failed_gates,
            "source_archives": {"static": static_archive.name, "dynamic": None},
            "baseline_parameters": _parameter_payload(base_parameters),
            "message": "Static data did not pass the quality gates. Repeat the static stage.",
        }
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
        "method": "physics_based_static_calibration",
        "quality_gates": gates,
        "source_archives": {"static": static_archive.name, "dynamic": None},
        "baseline_parameters": _parameter_payload(base_parameters),
        "static_suggestion": static_suggestion,
        "static_derivation": {
            key: value for key, value in static_result.items() if key != "parameters"
        },
    }
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
) -> dict[str, Any]:
    """Complete calibration with GP Bayesian optimization and held-out validation.

    Static quantities come only from the stationary physical-reference stage. The
    optimizer sees training cycles only. Held-out cycles are evaluated once after
    each search method has selected its candidate from training performance.
    """
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

    cycles = _detect_dynamic_cycles(dynamic_data.sync_rows)
    train_cycles, validation_cycles = _split_dynamic_cycles(cycles)
    train_rows = _rows_for_cycles(dynamic_data.sync_rows, train_cycles)
    validation_rows = _rows_for_cycles(dynamic_data.sync_rows, validation_cycles)

    gates = _quality_gates(
        static_data,
        dynamic_data,
        base_parameters,
        cycles=cycles,
        train_rows=train_rows,
        validation_rows=validation_rows,
    )
    failed_gates = [name for name, gate in gates.items() if not gate["passed"]]
    report_path = output_dir / "calibration_report.json"
    overview_path = output_dir / "calibration_overview.png"
    recommendation_path = output_dir / SUGGESTIONS_FILENAME
    run_id = output_dir.name

    split_payload = {
        "rule": "every_4th_accepted_cycle_held_out",
        "accepted_cycle_count": len(cycles),
        "training_cycle_ids": [cycle["cycle_id"] for cycle in train_cycles],
        "held_out_cycle_ids": [cycle["cycle_id"] for cycle in validation_cycles],
        "training_reference_rows": len(train_rows),
        "held_out_reference_rows": len(validation_rows),
        "cycles": cycles,
    }

    if failed_gates:
        report = {
            "schema_version": "vision_imu_calibration_report_v3",
            "generated_at": datetime.now().astimezone().isoformat(),
            "status": "INVALID",
            "method": "physics_static_plus_gp_bayesian_optimization_with_held_out_validation",
            "quality_gates": gates,
            "failed_gates": failed_gates,
            "cycle_split": split_payload,
            "source_archives": {
                "static": static_archive.name,
                "dynamic": dynamic_archive.name,
            },
            "baseline_parameters": _parameter_payload(base_parameters),
            "static_suggestion": static_suggestion or None,
            "message": "Test data did not pass the quality gates. No settings were recommended.",
        }
        _atomic_write_json(report_path, report)
        _write_suggestions_yaml(
            recommendation_path,
            status="invalid",
            run_id=run_id,
            static_suggestion=static_suggestion or None,
            dynamic_suggestion=None,
            final_parameters=None,
            failed_gates=failed_gates,
            cycle_split=split_payload,
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
            "result_summary": {},
        }

    if static_result is None:
        raise RuntimeError("Static calibration result is unexpectedly unavailable")
    static_parameters = static_result["parameters"]

    baseline_train = evaluate_parameters(dynamic_data, static_parameters, sync_rows=train_rows)
    baseline_validation = evaluate_parameters(
        dynamic_data, static_parameters, sync_rows=validation_rows
    )

    def evaluator(candidate: KneeAngleFilterParameters) -> EvaluationMetrics:
        return evaluate_parameters(dynamic_data, candidate, sync_rows=train_rows)

    def feasibility(metrics: EvaluationMetrics) -> tuple[bool, Sequence[str]]:
        return _candidate_quality_constraints(metrics, baseline_train)

    bo_started = time.perf_counter()
    bo_result = optimize_filter_parameters(
        static_parameters,
        evaluator,
        feasibility,
        evaluation_budget=BO_EVALUATION_BUDGET,
        seed=BO_RANDOM_SEED,
    )
    bo_runtime_s = time.perf_counter() - bo_started
    recommended_parameters = bo_result.parameters
    recommended_train = bo_result.metrics
    recommended_validation = evaluate_parameters(
        dynamic_data, recommended_parameters, sync_rows=validation_rows
    )

    random_started = time.perf_counter()
    random_candidates = sample_random_parameters(
        static_parameters,
        BO_EVALUATION_BUDGET,
        seed=BO_RANDOM_SEED + 101,
    )
    random_selection = _select_best_from_candidates(
        dynamic_data,
        random_candidates,
        train_rows=train_rows,
        baseline_train=baseline_train,
    )
    random_runtime_s = time.perf_counter() - random_started

    grid_started = time.perf_counter()
    grid_candidates = _budgeted_grid_candidates(static_parameters, BO_EVALUATION_BUDGET)
    grid_selection = _select_best_from_candidates(
        dynamic_data,
        grid_candidates,
        train_rows=train_rows,
        baseline_train=baseline_train,
    )
    grid_runtime_s = time.perf_counter() - grid_started

    held_out_improvement_percent = _rmse_improvement_percent(
        baseline_validation.rmse_deg,
        recommended_validation.rmse_deg,
    )
    bootstrap = _bootstrap_validation_uncertainty(
        dynamic_data,
        baseline_parameters=static_parameters,
        recommended_parameters=recommended_parameters,
        validation_cycles=validation_cycles,
    )
    reference_uncertainty = _reference_uncertainty_summary(validation_rows)

    search_comparison: dict[str, Any] = {
        "same_training_evaluation_budget": BO_EVALUATION_BUDGET,
        "selection_data": "training_cycles_only",
        "held_out_used_for_selection": False,
        "bayesian_optimization": {
            "evaluation_count": bo_result.evaluation_budget,
            "training_best_rmse_deg": recommended_train.rmse_deg,
            "held_out_rmse_deg": recommended_validation.rmse_deg,
            "runtime_s": bo_runtime_s,
        },
    }
    if random_selection is not None:
        random_parameters, random_train = random_selection
        random_validation = evaluate_parameters(
            dynamic_data, random_parameters, sync_rows=validation_rows
        )
        search_comparison["random_search"] = {
            "evaluation_count": len(random_candidates),
            "training_best_rmse_deg": random_train.rmse_deg,
            "held_out_rmse_deg": random_validation.rmse_deg,
            "runtime_s": random_runtime_s,
            "selected_parameters": _parameter_payload(random_parameters),
        }
    else:
        search_comparison["random_search"] = {
            "evaluation_count": len(random_candidates),
            "status": "no_feasible_candidate",
            "runtime_s": random_runtime_s,
        }
    if grid_selection is not None:
        grid_parameters, grid_train = grid_selection
        grid_validation = evaluate_parameters(
            dynamic_data, grid_parameters, sync_rows=validation_rows
        )
        search_comparison["budgeted_grid_search"] = {
            "evaluation_count": len(grid_candidates),
            "training_best_rmse_deg": grid_train.rmse_deg,
            "held_out_rmse_deg": grid_validation.rmse_deg,
            "runtime_s": grid_runtime_s,
            "selected_parameters": _parameter_payload(grid_parameters),
        }
    else:
        search_comparison["budgeted_grid_search"] = {
            "evaluation_count": len(grid_candidates),
            "status": "no_feasible_candidate",
            "runtime_s": grid_runtime_s,
        }

    optimization_payload = {
        "backend": "gaussian_process_bayesian_optimization",
        "surrogate": "gaussian_process_rbf_normalized_parameter_space",
        "acquisition": "expected_improvement",
        "evaluation_budget": bo_result.evaluation_budget,
        "random_seed": bo_result.random_seed,
        "initial_design_size": bo_result.initial_design_size,
        "candidate_pool_size": bo_result.candidate_pool_size,
        "optimized_parameters": [
            "alpha_stable",
            "alpha_dynamic",
            "max_accel_correction_deg",
        ],
        "parameter_bounds": parameter_bounds_payload(),
        "primary_objective": "training_cycle_rmse_deg",
        "candidate_specific_time_shift": False,
        "held_out_used_for_selection": False,
        "hard_candidate_constraints": _candidate_constraint_definition(baseline_train),
        "feasible_evaluation_count": sum(
            1 for observation in bo_result.observations if observation.feasible
        ),
        "gp_uncertainty": bo_result.gp_uncertainty,
        "search_trace": bo_result.trace_payload(),
        "search_comparison": search_comparison,
    }

    validation_payload = {
        "training_baseline": baseline_train.to_dict(),
        "training_recommended": recommended_train.to_dict(),
        "held_out_baseline": baseline_validation.to_dict(),
        "held_out_recommended": recommended_validation.to_dict(),
        "held_out_rmse_improvement_percent": held_out_improvement_percent,
        "bootstrap": bootstrap,
        "reference_uncertainty": reference_uncertainty,
    }

    report = {
        "schema_version": "vision_imu_calibration_report_v3",
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "VALID",
        "method": "physics_static_plus_gp_bayesian_optimization_with_held_out_validation",
        "reference": {
            "type": "marker_based_geometric_reference",
            "statement": (
                "Dynamic performance is evaluated against the synchronized calibrated "
                "marker reference. Held-out cycles are excluded from parameter selection."
            ),
        },
        "quality_gates": gates,
        "cycle_split": split_payload,
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
        "optimization": optimization_payload,
        "metrics": validation_payload,
        "uncertainty": {
            "gp_model": bo_result.gp_uncertainty,
            "held_out_validation": bootstrap,
            "marker_reference": reference_uncertainty,
        },
        "limitations": [
            "The marker system is an engineering reference, not absolute ground truth.",
            "Image-space marker quality is reported, but angular reference uncertainty remains unquantified until the separate known-angle validation is performed.",
            "The recommendation is valid only for the tested sensor mounting and movement range.",
            "Estimated lag is reported diagnostically and is never optimized away per candidate.",
            "Random and budgeted-grid searches are same-budget engineering comparators, not a statistical superiority claim from one calibration run.",
        ],
    }
    _atomic_write_json(report_path, report)
    dynamic_suggestion = _parameter_payload(recommended_parameters)
    result_summary = {
        "optimizer": "GP Bayesian Optimization",
        "evaluation_budget": bo_result.evaluation_budget,
        "held_out_rmse_deg": recommended_validation.rmse_deg,
        "held_out_mae_deg": recommended_validation.mae_deg,
        "held_out_lag_ms": recommended_validation.lag_ms,
        "held_out_rmse_improvement_percent": held_out_improvement_percent,
        "held_out_rmse_ci95_deg": bootstrap.get("recommended_rmse_ci95_deg"),
        "gp_recommended_std_rmse_deg": bo_result.gp_uncertainty.get(
            "recommended_posterior_std_rmse_deg"
        ),
    }
    _write_suggestions_yaml(
        recommendation_path,
        status="complete",
        run_id=run_id,
        static_suggestion=static_suggestion,
        dynamic_suggestion=dynamic_suggestion,
        final_parameters=dynamic_suggestion,
        held_out_rmse_improvement_percent=held_out_improvement_percent,
        cycle_split=split_payload,
        optimization_summary={
            "backend": "gaussian_process_bayesian_optimization",
            "evaluation_budget": bo_result.evaluation_budget,
            "primary_objective": "training_cycle_rmse_deg",
            "held_out_used_for_selection": False,
            "gp_uncertainty": bo_result.gp_uncertainty,
        },
        validation_summary={
            "held_out_rmse_deg": recommended_validation.rmse_deg,
            "held_out_mae_deg": recommended_validation.mae_deg,
            "held_out_lag_ms": recommended_validation.lag_ms,
            "held_out_rmse_improvement_percent": held_out_improvement_percent,
            "bootstrap": bootstrap,
        },
    )
    _write_calibration_overview(
        overview_path,
        dynamic_data,
        static_parameters,
        recommended_parameters,
        baseline_validation,
        recommended_validation,
    )
    return {
        "valid": True,
        "message": (
            "Calibration complete. GP Bayesian Optimization recommendation validated on "
            f"held-out cycles; RMSE change {held_out_improvement_percent:+.1f}%."
        ),
        "error": "",
        "report_path": str(report_path),
        "overview_path": str(overview_path),
        "recommendation_path": str(recommendation_path),
        "static_suggestion": static_suggestion,
        "dynamic_suggestion": dynamic_suggestion,
        "recommended_parameters": _parameter_payload(recommended_parameters),
        "result_summary": result_summary,
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
    *,
    sync_rows: Sequence[Mapping[str, str]] | None = None,
) -> EvaluationMetrics:
    """Replay one parameter set and evaluate it without candidate-specific time shifting.

    RMSE is the primary objective. Lag, jitter and drift are descriptive metrics only.
    The same synchronized timestamps are used for every candidate, which prevents a
    candidate from improving its score by shifting itself in time.
    """
    estimates = replay_filter(data.raw_rows, parameters)
    rows = data.sync_rows if sync_rows is None else sync_rows
    pairs = _paired_values(rows, estimates, lag_ms=0.0)
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
    lag_ms = _estimate_lag_ms(rows, estimates)
    return EvaluationMetrics(
        pair_count=len(pairs),
        mae_deg=mae,
        rmse_deg=rmse,
        p95_abs_error_deg=p95,
        max_abs_error_deg=maximum,
        lag_ms=lag_ms,
        jitter_deg_per_frame=jitter,
        drift_deg=drift,
        score=rmse,
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
    """Derive static quantities from physics, not from the filter being tuned.

    The user supplies the physical condition: the leg is fully extended, therefore
    knee flexion is zero (posterior knee angle 180 deg). Gravity gives the segment
    orientations, while stationary gyro samples give bias and short-term noise.
    """
    if not data.raw_rows:
        raise RuntimeError("Static test contains no raw IMU samples")

    straight_offsets: list[float] = []
    thigh_rates: list[float] = []
    shank_rates: list[float] = []
    for row in data.raw_rows:
        thigh_acc = _acceleration_angle_from_row(row, "thigh", base_parameters)
        shank_acc = _acceleration_angle_from_row(row, "shank", base_parameters)
        thigh_segment = thigh_acc - base_parameters.thigh_offset_deg
        shank_segment = shank_acc - base_parameters.shank_offset_deg
        relative = _shortest_angle_error_deg(shank_segment, thigh_segment)
        straight_offsets.append(base_parameters.relative_angle_sign * relative)
        thigh_rates.append(_signed_axis_rate(row, "thigh", base_parameters))
        shank_rates.append(_signed_axis_rate(row, "shank", base_parameters))

    straight_leg_offset = statistics.median(straight_offsets)
    thigh_bias = statistics.median(thigh_rates)
    shank_bias = statistics.median(shank_rates)
    thigh_residuals = [rate - thigh_bias for rate in thigh_rates]
    shank_residuals = [rate - shank_bias for rate in shank_rates]
    thigh_noise_sigma = _robust_sigma(thigh_residuals)
    shank_noise_sigma = _robust_sigma(shank_residuals)
    residual_abs = [abs(value) for value in thigh_residuals + shank_residuals]
    noise_p95 = _percentile(residual_abs, 95.0)
    deadband = min(2.5, max(0.05, 3.0 * max(thigh_noise_sigma, shank_noise_sigma)))

    parameters = replace(
        base_parameters,
        straight_leg_offset_deg=straight_leg_offset,
        thigh_gyro_bias_deg_s=thigh_bias,
        shank_gyro_bias_deg_s=shank_bias,
        gyro_deadband_deg_s=deadband,
    )
    parameters.validate()
    return {
        "parameters": parameters,
        "physical_reference_condition": "full_extension_180_deg",
        "straight_leg_offset_deg": straight_leg_offset,
        "straight_leg_offset_std_deg": statistics.pstdev(straight_offsets) if len(straight_offsets) > 1 else 0.0,
        "thigh_gyro_bias_deg_s": thigh_bias,
        "shank_gyro_bias_deg_s": shank_bias,
        "thigh_gyro_noise_sigma_deg_s": thigh_noise_sigma,
        "shank_gyro_noise_sigma_deg_s": shank_noise_sigma,
        "combined_gyro_noise_p95_deg_s": noise_p95,
        "gyro_deadband_rule": "3x robust stationary gyro sigma, clamped to [0.05, 2.5] deg/s",
        "gyro_deadband_deg_s": deadband,
        "sample_count": len(data.raw_rows),
        "marker_reference_sanity": _static_marker_reference_sanity(data.sync_rows),
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


def _candidate_quality_constraints(
    metrics: EvaluationMetrics,
    baseline_train: EvaluationMetrics,
) -> tuple[bool, list[str]]:
    """Hard candidate gates used by Patch 4 instead of a weighted score."""
    reasons: list[str] = []
    numeric_values = (
        metrics.mae_deg,
        metrics.rmse_deg,
        metrics.p95_abs_error_deg,
        metrics.max_abs_error_deg,
        metrics.lag_ms,
        metrics.jitter_deg_per_frame,
        metrics.drift_deg,
    )
    if not all(math.isfinite(float(value)) for value in numeric_values):
        reasons.append("non_finite_metric")
    minimum_pairs = max(20, int(math.floor(0.95 * baseline_train.pair_count)))
    if metrics.pair_count < minimum_pairs:
        reasons.append("insufficient_comparison_pairs")
    # Lag estimation scans only [-250, +250] ms. Landing on the boundary means
    # the lag estimate is unresolved rather than a trustworthy in-range optimum.
    if abs(metrics.lag_ms) >= 250.0:
        reasons.append("lag_at_diagnostic_search_boundary")
    # Reject clearly noisier candidates relative to the measured baseline rather
    # than adding jitter to the objective with an arbitrary score weight.
    jitter_limit = max(1.0, 2.0 * baseline_train.jitter_deg_per_frame + 0.25)
    if metrics.jitter_deg_per_frame > jitter_limit:
        reasons.append("excessive_error_jitter")
    return not reasons, reasons


def _candidate_constraint_definition(
    baseline_train: EvaluationMetrics,
) -> dict[str, Any]:
    return {
        "comparison_pairs": {
            "rule": ">= 95% of baseline training pairs, minimum 20",
            "threshold": max(20, int(math.floor(0.95 * baseline_train.pair_count))),
        },
        "lag": {
            "rule": "absolute estimated lag must stay inside the diagnostic search range",
            "threshold_ms": 250.0,
            "boundary_is_rejected": True,
        },
        "jitter": {
            "rule": "candidate error jitter <= max(1.0 deg/frame, 2x baseline + 0.25 deg/frame)",
            "threshold_deg_per_frame": max(
                1.0, 2.0 * baseline_train.jitter_deg_per_frame + 0.25
            ),
        },
        "finite_metrics_required": True,
    }


def _select_best_from_candidates(
    data: ArchiveData,
    candidates: Sequence[KneeAngleFilterParameters],
    *,
    train_rows: Sequence[Mapping[str, str]],
    baseline_train: EvaluationMetrics,
) -> tuple[KneeAngleFilterParameters, EvaluationMetrics] | None:
    best: tuple[KneeAngleFilterParameters, EvaluationMetrics] | None = None
    for candidate in candidates:
        metrics = evaluate_parameters(data, candidate, sync_rows=train_rows)
        feasible, _reasons = _candidate_quality_constraints(metrics, baseline_train)
        if not feasible:
            continue
        if best is None or metrics.rmse_deg < best[1].rmse_deg:
            best = (candidate, metrics)
    return best


def _budgeted_grid_candidates(
    static_parameters: KneeAngleFilterParameters,
    budget: int,
) -> list[KneeAngleFilterParameters]:
    all_candidates = _candidate_parameters(static_parameters)
    target = max(1, min(int(budget), len(all_candidates)))
    if target >= len(all_candidates):
        return all_candidates
    indices = [
        int(round(index * (len(all_candidates) - 1) / max(1, target - 1)))
        for index in range(target)
    ]
    selected = [static_parameters]
    selected.extend(all_candidates[index] for index in indices)
    deduped: list[KneeAngleFilterParameters] = []
    seen: set[tuple[float, float, float]] = set()
    for candidate in selected:
        signature = (
            round(candidate.alpha_stable, 9),
            round(candidate.alpha_dynamic, 9),
            round(candidate.max_accel_correction_deg, 9),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(candidate)
        if len(deduped) >= target:
            break
    if len(deduped) < target:
        for candidate in all_candidates:
            signature = (
                round(candidate.alpha_stable, 9),
                round(candidate.alpha_dynamic, 9),
                round(candidate.max_accel_correction_deg, 9),
            )
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(candidate)
            if len(deduped) >= target:
                break
    return deduped


def _rmse_improvement_percent(baseline_rmse: float, candidate_rmse: float) -> float:
    if baseline_rmse <= 1e-9 or not math.isfinite(baseline_rmse):
        return 0.0
    return 100.0 * (baseline_rmse - candidate_rmse) / baseline_rmse


def _bootstrap_validation_uncertainty(
    data: ArchiveData,
    *,
    baseline_parameters: KneeAngleFilterParameters,
    recommended_parameters: KneeAngleFilterParameters,
    validation_cycles: Sequence[Mapping[str, Any]],
    resamples: int = 2000,
) -> dict[str, Any]:
    per_cycle: list[dict[str, Any]] = []
    for cycle in validation_cycles:
        rows = _rows_for_cycles(data.sync_rows, [cycle])
        baseline = evaluate_parameters(data, baseline_parameters, sync_rows=rows)
        recommended = evaluate_parameters(data, recommended_parameters, sync_rows=rows)
        if baseline.pair_count <= 0 or recommended.pair_count <= 0:
            continue
        per_cycle.append(
            {
                "cycle_id": int(cycle["cycle_id"]),
                "baseline_rmse_deg": baseline.rmse_deg,
                "recommended_rmse_deg": recommended.rmse_deg,
                "pair_count": min(baseline.pair_count, recommended.pair_count),
            }
        )
    payload: dict[str, Any] = {
        "method": "cycle_level_nonparametric_bootstrap",
        "resamples": int(resamples),
        "held_out_cycle_count": len(per_cycle),
        "available": len(per_cycle) >= 2,
        "per_cycle": per_cycle,
    }
    if len(per_cycle) < 2:
        payload["reason"] = "At least two held-out cycles are required for a cycle-level confidence interval."
        payload["recommended_rmse_ci95_deg"] = None
        payload["improvement_percent_ci95"] = None
        return payload

    rng = __import__("numpy").random.default_rng(BO_RANDOM_SEED + 404)
    recommended_draws: list[float] = []
    improvement_draws: list[float] = []
    count = len(per_cycle)
    for _ in range(int(resamples)):
        selected = [per_cycle[int(index)] for index in rng.integers(0, count, size=count)]
        total_pairs = sum(max(1, int(item["pair_count"])) for item in selected)
        baseline_sq = sum(
            max(1, int(item["pair_count"])) * float(item["baseline_rmse_deg"]) ** 2
            for item in selected
        ) / total_pairs
        recommended_sq = sum(
            max(1, int(item["pair_count"])) * float(item["recommended_rmse_deg"]) ** 2
            for item in selected
        ) / total_pairs
        baseline_rmse = math.sqrt(max(0.0, baseline_sq))
        recommended_rmse = math.sqrt(max(0.0, recommended_sq))
        recommended_draws.append(recommended_rmse)
        improvement_draws.append(
            _rmse_improvement_percent(baseline_rmse, recommended_rmse)
        )
    payload["recommended_rmse_ci95_deg"] = [
        _percentile(recommended_draws, 2.5),
        _percentile(recommended_draws, 97.5),
    ]
    payload["improvement_percent_ci95"] = [
        _percentile(improvement_draws, 2.5),
        _percentile(improvement_draws, 97.5),
    ]
    return payload


def _reference_uncertainty_summary(
    validation_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    valid_rows = _valid_sync_rows(validation_rows)
    thigh = [
        value
        for row in valid_rows
        if (value := _to_float(row.get("thigh_reprojection_rmse_px"))) is not None
    ]
    shank = [
        value
        for row in valid_rows
        if (value := _to_float(row.get("shank_reprojection_rmse_px"))) is not None
    ]
    reprojection = thigh + shank
    return {
        "angular_uncertainty_quantified": False,
        "status": "known_angle_reference_validation_required",
        "statement": (
            "This run reports image-space pose quality only. Angular reference uncertainty "
            "must come from the separate known-angle mechanical validation of the marker setup."
        ),
        "valid_reference_row_count": len(valid_rows),
        "reprojection_rmse_px_median": (
            statistics.median(reprojection) if reprojection else None
        ),
        "reprojection_rmse_px_p95": (
            _percentile(reprojection, 95.0) if reprojection else None
        ),
        "reprojection_rmse_px_max": max(reprojection) if reprojection else None,
    }

def _static_quality_gates(
    static_data: ArchiveData,
    parameters: KneeAngleFilterParameters,
) -> dict[str, dict[str, Any]]:
    static_gap_fraction = _sequence_gap_fraction(static_data.raw_rows)
    static_ready_fraction = _imu_ready_fraction(static_data.raw_rows)
    static_motion = _static_motion_p95(static_data.raw_rows, parameters)
    gravity_fraction = _static_gravity_valid_fraction(static_data.raw_rows, parameters)

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
        "static_gravity_valid_fraction": gate(
            gravity_fraction >= 0.95,
            gravity_fraction,
            ">= 95% samples with both accelerometer norms inside the configured no-trust bound around 1 g",
        ),
        "static_motion_p95_deg_s": gate(
            static_motion <= 5.0,
            static_motion,
            "<= 5 deg/s; leg must remain still and fully extended",
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
    *,
    cycles: Sequence[Mapping[str, Any]] | None = None,
    train_rows: Sequence[Mapping[str, str]] | None = None,
    validation_rows: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, dict[str, Any]]:
    static_gates = _static_quality_gates(static_data, parameters)
    dynamic_sync = _valid_sync_rows(dynamic_data.sync_rows)
    dynamic_valid_fraction = _visual_valid_fraction(dynamic_data.sync_rows)
    dynamic_sync_p95 = _sync_p95_ms(dynamic_sync)
    dynamic_gap_fraction = _sequence_gap_fraction(dynamic_data.raw_rows)
    dynamic_ready_fraction = _imu_ready_fraction(dynamic_data.raw_rows)
    dynamic_base_pairs = len(
        _paired_values(dynamic_data.sync_rows, replay_filter(dynamic_data.raw_rows, parameters), lag_ms=0.0)
    )
    dynamic_angles = [float(row["visual_knee_angle_deg"]) for row in dynamic_sync]
    dynamic_range = max(dynamic_angles) - min(dynamic_angles) if dynamic_angles else 0.0
    detected_cycles = list(cycles) if cycles is not None else _detect_dynamic_cycles(dynamic_data.sync_rows)
    if train_rows is None or validation_rows is None:
        train_cycles, held_out_cycles = _split_dynamic_cycles(detected_cycles)
        train_rows = _rows_for_cycles(dynamic_data.sync_rows, train_cycles)
        validation_rows = _rows_for_cycles(dynamic_data.sync_rows, held_out_cycles)

    def gate(passed: bool, value: Any, requirement: str) -> dict[str, Any]:
        return {"passed": bool(passed), "value": value, "requirement": requirement}

    return {
        **static_gates,
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
            ">= 80 synchronized valid marker-reference rows",
        ),
        "dynamic_valid_comparison_pairs": gate(
            dynamic_base_pairs >= 80,
            dynamic_base_pairs,
            ">= 80 valid baseline marker/IMU pairs",
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
        "dynamic_accepted_cycles": gate(
            len(detected_cycles) >= MIN_ACCEPTED_DYNAMIC_CYCLES,
            len(detected_cycles),
            f">= {MIN_ACCEPTED_DYNAMIC_CYCLES} complete flexion-extension cycles",
        ),
        "dynamic_training_reference_rows": gate(
            len(train_rows) >= 40,
            len(train_rows),
            ">= 40 valid marker-reference rows in training cycles",
        ),
        "dynamic_held_out_reference_rows": gate(
            len(validation_rows) >= 10,
            len(validation_rows),
            ">= 10 valid marker-reference rows in held-out cycles",
        ),
    }

def _detect_dynamic_cycles(
    sync_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Detect complete flexion-extension cycles from the marker reference only."""
    rows = _valid_sync_rows(sync_rows)
    if len(rows) < 20:
        return []
    angles = [float(row["visual_knee_angle_deg"]) for row in rows]
    low = _percentile(angles, 10.0)
    high = _percentile(angles, 90.0)
    span = high - low
    if not math.isfinite(span) or span < 20.0:
        return []
    low_threshold = low + 0.25 * span
    high_threshold = high - 0.25 * span

    events: list[tuple[str, int]] = []
    last_state = ""
    for row, angle in zip(rows, angles):
        state = "LOW" if angle <= low_threshold else "HIGH" if angle >= high_threshold else "MID"
        if state == "MID" or state == last_state:
            continue
        events.append((state, int(float(row["host_monotonic_ns"]))))
        last_state = state

    cycles: list[dict[str, Any]] = []
    event_index = 0
    while event_index + 2 < len(events):
        first, middle, last = events[event_index:event_index + 3]
        if first[0] == last[0] and first[0] != middle[0]:
            start_ns = first[1]
            end_ns = last[1]
            interval_angles = [
                float(row["visual_knee_angle_deg"])
                for row in rows
                if start_ns <= int(float(row["host_monotonic_ns"])) <= end_ns
            ]
            cycle_range = max(interval_angles) - min(interval_angles) if interval_angles else 0.0
            duration_s = (end_ns - start_ns) / 1_000_000_000.0
            if cycle_range >= max(20.0, 0.45 * span) and 0.5 <= duration_s <= 10.0:
                cycles.append(
                    {
                        "cycle_id": len(cycles) + 1,
                        "start_host_monotonic_ns": start_ns,
                        "end_host_monotonic_ns": end_ns,
                        "duration_s": duration_s,
                        "range_deg": cycle_range,
                        "start_extreme": first[0].lower(),
                    }
                )
                event_index += 2
                continue
        event_index += 1
    return cycles


def _split_dynamic_cycles(
    cycles: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Deterministic ~75/25 split: every fourth accepted cycle is held out."""
    accepted = list(cycles)
    if len(accepted) < MIN_ACCEPTED_DYNAMIC_CYCLES:
        return accepted, []
    validation = [cycle for index, cycle in enumerate(accepted, start=1) if index % HELD_OUT_CYCLE_STRIDE == 0]
    if not validation:
        validation = [accepted[-1]]
    validation_ids = {int(cycle["cycle_id"]) for cycle in validation}
    training = [cycle for cycle in accepted if int(cycle["cycle_id"]) not in validation_ids]
    return training, validation


def _rows_for_cycles(
    sync_rows: Sequence[Mapping[str, str]],
    cycles: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, str]]:
    valid = _valid_sync_rows(sync_rows)
    if not cycles:
        return []
    windows = [
        (int(cycle["start_host_monotonic_ns"]), int(cycle["end_host_monotonic_ns"]))
        for cycle in cycles
    ]
    return [
        row
        for row in valid
        if any(start <= int(float(row["host_monotonic_ns"])) < end for start, end in windows)
    ]


def _acceleration_angle_from_row(
    row: Mapping[str, str],
    prefix: str,
    parameters: KneeAngleFilterParameters,
) -> float:
    ax = float(row[f"{prefix}_ax_g"])
    ay = float(row[f"{prefix}_ay_g"])
    az = float(row[f"{prefix}_az_g"])
    if parameters.rotation_axis == 0:
        axis_1 = parameters.acc_axis_1_sign * ay
        axis_2 = parameters.acc_axis_2_sign * az
    elif parameters.rotation_axis == 2:
        axis_1 = parameters.acc_axis_1_sign * ay
        axis_2 = parameters.acc_axis_2_sign * ax
    else:
        axis_1 = parameters.acc_axis_1_sign * az
        axis_2 = parameters.acc_axis_2_sign * ax
    return math.degrees(math.atan2(axis_1, axis_2))


def _shortest_angle_error_deg(target_deg: float, reference_deg: float) -> float:
    error = target_deg - reference_deg
    while error > 180.0:
        error -= 360.0
    while error < -180.0:
        error += 360.0
    return error


def _robust_sigma(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    center = statistics.median(values)
    mad = statistics.median(abs(float(value) - center) for value in values)
    return 1.4826 * mad


def _static_gravity_valid_fraction(
    rows: Sequence[Mapping[str, str]],
    parameters: KneeAngleFilterParameters,
) -> float:
    if not rows:
        return 0.0
    bound = max(0.15, float(parameters.acc_no_trust_deviation_g))
    valid = 0
    for row in rows:
        try:
            thigh_norm = math.sqrt(sum(float(row[f"thigh_{axis}_g"]) ** 2 for axis in ("ax", "ay", "az")))
            shank_norm = math.sqrt(sum(float(row[f"shank_{axis}_g"]) ** 2 for axis in ("ax", "ay", "az")))
        except (KeyError, TypeError, ValueError):
            continue
        if abs(thigh_norm - 1.0) <= bound and abs(shank_norm - 1.0) <= bound:
            valid += 1
    return valid / len(rows)


def _static_marker_reference_sanity(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    valid = _valid_sync_rows(rows)
    if not valid:
        return {
            "available": False,
            "used_for_static_derivation": False,
            "valid_rows": 0,
            "median_knee_angle_deg": None,
            "median_error_from_180_deg": None,
        }
    angles = [float(row["visual_knee_angle_deg"]) for row in valid]
    median_angle = statistics.median(angles)
    return {
        "available": True,
        "used_for_static_derivation": False,
        "valid_rows": len(valid),
        "median_knee_angle_deg": median_angle,
        "median_error_from_180_deg": median_angle - 180.0,
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


def _write_suggestions_yaml(
    path: Path,
    *,
    status: str,
    run_id: str,
    static_suggestion: Mapping[str, Any] | None,
    dynamic_suggestion: Mapping[str, Any] | None,
    final_parameters: Mapping[str, Any] | None,
    held_out_rmse_improvement_percent: float | None = None,
    cycle_split: Mapping[str, Any] | None = None,
    failed_gates: Sequence[str] | None = None,
    error: str | None = None,
    optimization_summary: Mapping[str, Any] | None = None,
    validation_summary: Mapping[str, Any] | None = None,
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
        "method": "physics_static_plus_cycle_split_gp_bayesian_optimization",
        "reference": "synchronized marker-based knee-angle comparison",
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
    if held_out_rmse_improvement_percent is not None:
        payload["held_out_rmse_improvement_percent"] = float(
            held_out_rmse_improvement_percent
        )
    if cycle_split is not None:
        payload["cycle_split"] = dict(cycle_split)
    if optimization_summary is not None:
        payload["optimization"] = dict(optimization_summary)
    if validation_summary is not None:
        payload["validation"] = dict(validation_summary)
    if failed_gates:
        payload["failed_quality_gates"] = list(failed_gates)
    if error:
        payload["last_error"] = str(error)
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
        f"held-out RMSE {baseline_metrics.rmse_deg:.2f}° → recommended {recommended_metrics.rmse_deg:.2f}°"
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

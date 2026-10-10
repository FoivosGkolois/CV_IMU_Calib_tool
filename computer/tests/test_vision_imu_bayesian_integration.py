"""Ticket 2 replay tests: train-trial comparison, without hardware or lab data."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import asdict, replace
from zipfile import ZipFile

import pytest
import yaml

from computer.vision_imu_calibration import calibration_workflow as cw
from computer.vision_imu_calibration.bayesian_optimizer import BO_EVALUATION_BUDGET, BO_RANDOM_SEED
from computer.vision_imu_calibration.imu_filter import KneeAngleFilterParameters
from computer.vision_imu_calibration.imu_serial import ACCEL_G_PER_LSB, GYRO_DPS_PER_LSB


def raw_row(index, flexion=25, rate=0, thigh_rate=0, *, start_ns=9_000_000_000):
    raw = {"host_monotonic_ns": str(start_ns + index * 10_000_000), "schema_version": "1",
           "sample_index": str(index), "mcu_timestamp_us": str(index * 10_000),
           "thigh_ok": "true", "shank_ok": "true"}
    for prefix in ("thigh", "shank"):
        angle = math.radians(flexion if prefix == "shank" else 0)
        accel = (0, -math.sin(angle), math.cos(angle))
        gyro = (rate if prefix == "shank" else thigh_rate, 0, 0)
        for axis, value in zip("xyz", accel):
            raw[f"{prefix}_a{axis}_raw"] = str(round(value / ACCEL_G_PER_LSB))
        for axis, value in zip("xyz", gyro):
            raw[f"{prefix}_g{axis}_raw"] = str(round(value / GYRO_DPS_PER_LSB))
    raw.update({key: str(value) for key, value in cw._sample_from_row(raw).to_record().items()
                if key.endswith(("_g", "_dps"))})
    return raw


def dataset(*, dynamic=False, bias=False, wrap_start=None, start_ns=9_000_000_000):
    rows, sync = [], []
    for i in range(1000 if dynamic else 400):
        t = i * .01
        angle = 55 + 25 * math.sin(2 * math.pi * t / 2.5) if dynamic else 25
        rate = 25 * 2 * math.pi / 2.5 * math.cos(2 * math.pi * t / 2.5) if dynamic else 0
        if wrap_start is not None:
            angle = wrap_start + .2 * math.sin(i / 20)
        row = raw_row(i, angle, rate + (1.75 if bias else 0), .875 if bias else 0, start_ns=start_ns)
        rows.append(row)
        if i % 10 == 0:
            sync.append({"host_monotonic_ns": row["host_monotonic_ns"],
                         "visual_knee_angle_deg": str(199 - angle if dynamic else 174),
                         "raw_knee_angle_deg": str(178 - angle if dynamic else 172),
                         "tracking_valid": "true", "sync_delta_ms": "0",
                         "imu_sample_index": str(i)})
    return cw.ArchiveData(rows, sync)


def write_archive(path, data):
    with ZipFile(path, "w") as archive:
        for name, rows in (("raw_imu.csv", data.raw_rows), ("synchronized_timeline.csv", data.sync_rows)):
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            archive.writestr(name, buffer.getvalue())
    return path


def poison(*args, **kwargs):
    raise AssertionError("Inactive physics, lag estimation or old grid search was called")


def test_straight_alignment_uses_measured_optical_value_after_biases_and_preserves_base():
    data = dataset(bias=True)
    base = replace(KneeAngleFilterParameters(), straight_leg_offset_deg=4)
    original = asdict(base)
    result = cw._derive_static_parameters(data, base)
    p = result["parameters"]
    assert p.thigh_gyro_bias_deg_s == pytest.approx(.875)
    assert p.shank_gyro_bias_deg_s == pytest.approx(1.75)
    assert p.gyro_deadband_deg_s == pytest.approx(.15)
    assert result["measured_optical_angle_deg"] == 174
    paired = cw._evaluate_training_parameters(data, p, cw._fixed_training_pairs(data))
    assert paired.mae_deg < 1e-9
    import statistics
    old_replay = cw.replay_filter(data.raw_rows, base)
    old_delta = statistics.median(float(row["visual_knee_angle_deg"]) -
                                  old_replay[int(row["imu_sample_index"])].knee_angle_deg for row in data.sync_rows)
    old_order = replace(p, straight_leg_offset_deg=base.straight_leg_offset_deg + old_delta)
    assert cw._evaluate_training_parameters(data, old_order, cw._fixed_training_pairs(data)).mae_deg > .05
    assert p.straight_leg_offset_deg != 0
    assert asdict(base) == original


@pytest.mark.parametrize("start", [179.9, 180.1, -179.9, -180.1])
def test_static_wraparound_retains_mirror_branch_instead_of_forcing_180(start):
    data = dataset(wrap_start=start)
    result = cw._derive_static_parameters(data, KneeAngleFilterParameters())
    metric = cw._evaluate_training_parameters(data, result["parameters"], cw._fixed_training_pairs(data))
    assert metric.mae_deg < .3
    assert result["measured_optical_angle_deg"] == 174
    assert abs(result["angle_offset_delta_deg"]) > 170


def test_static_bias_excludes_not_ready_samples():
    data = dataset(bias=True)
    for i in range(1, 4):
        changed = raw_row(i, rate=100, thigh_rate=100)
        changed.update(thigh_ok="false", shank_ok="false")
        data.raw_rows[i] = changed
    result = cw._derive_static_parameters(data, KneeAngleFilterParameters())
    assert result["parameters"].thigh_gyro_bias_deg_s == pytest.approx(.875)
    assert result["parameters"].shank_gyro_bias_deg_s == pytest.approx(1.75)


def test_active_static_alignment_honours_gyro_and_relative_signs():
    data = dataset(bias=True)
    base = replace(KneeAngleFilterParameters(), gyro_rate_sign=-1, relative_angle_sign=-1,
                   thigh_offset_deg=7, shank_offset_deg=-3)
    result = cw._derive_static_parameters(data, base)
    p = result["parameters"]
    assert p.thigh_gyro_bias_deg_s == pytest.approx(-.875)
    assert p.shank_gyro_bias_deg_s == pytest.approx(-1.75)
    assert p.relative_angle_sign == -1 and p.gyro_rate_sign == -1
    assert p.thigh_offset_deg == 7 and p.shank_offset_deg == -3
    assert cw._evaluate_training_parameters(data, p, cw._fixed_training_pairs(data)).mae_deg < 1e-9


def test_static_branch_reset_dropout_needs_retry_instead_of_mixing_full_turns():
    data = dataset(wrap_start=179.9)
    data.raw_rows[25].update(thigh_ok="false", shank_ok="false")
    gate = cw._static_quality_gates(data, KneeAngleFilterParameters())["static_relative_alignment"]
    assert not gate["passed"]
    assert "branch" in gate["value"]


def test_nonfinite_and_insufficient_metrics_fail_training_feasibility():
    metrics = cw._metrics_from_pairs([(i, 100, 101) for i in range(100)], lag_ms=0)
    assert not cw._training_feasibility(replace(metrics, rmse_deg=math.inf), 100)[0]
    assert not cw._training_feasibility(replace(metrics, pair_count=79), 100)[0]


@pytest.mark.parametrize("field,value", [
    ("shank_gx_dps", "nan"), ("thigh_ax_g", "inf"),
    ("thigh_az_raw", "missing"), ("shank_gx_raw", "3.5"),
    ("thigh_ax_raw", "32768"), ("host_monotonic_ns", "nan"),
    ("mcu_timestamp_us", "-1"), ("shank_gx_dps", "2"),
])
def test_invalid_raw_samples_are_rejected_before_derivation(field, value):
    data = dataset()
    data.raw_rows[10][field] = value
    gates = cw._static_quality_gates(data, KneeAngleFilterParameters())
    assert not gates["static_raw_imu_valid"]["passed"]
    with pytest.raises(ValueError, match="finite"):
        cw._derive_static_parameters(data, KneeAngleFilterParameters())


def test_static_common_motion_and_optical_motion_cannot_be_absorbed_into_calibration():
    moving = dataset()
    moving.raw_rows[:] = [raw_row(i, rate=8, thigh_rate=8) for i in range(400)]
    assert not cw._static_quality_gates(moving, KneeAngleFilterParameters())["static_motion_p95_deg_s"]["passed"]
    data = dataset()
    for i, row in enumerate(data.sync_rows):
        row["visual_knee_angle_deg"] = str(150 + i)
    assert not cw._static_quality_gates(data, KneeAngleFilterParameters())["static_relative_alignment"]["passed"]


def test_frozen_pairing_keeps_exact_large_timestamps_and_does_not_bridge_dropout():
    data = dataset(dynamic=True, start_ns=9_000_000_000_000_000_001)
    data.raw_rows[10].update(thigh_ok="false")
    plan = cw._fixed_training_pairs(data)
    assert plan[0][0] == 9_000_000_000_000_000_001
    assert all(index != 10 for _, _, index, _ in plan)
    assert cw._sample_from_row(data.raw_rows[0]).host_monotonic_ns == plan[0][0]
    assert len(plan) == 99


def test_candidate_missing_frame_cannot_change_pair_mask_or_rematch(monkeypatch):
    data = dataset(dynamic=True)
    base = KneeAngleFilterParameters()
    plan = cw._fixed_training_pairs(data)
    original = cw.replay_filter
    def replay(rows, parameters):
        result = original(rows, parameters)
        if parameters.alpha_stable == .8:
            result[10] = replace(result[10], valid=False)
        return result
    monkeypatch.setattr(cw, "replay_filter", replay)
    full = cw._evaluate_training_parameters(data, base, plan)
    missing = cw._evaluate_training_parameters(data, replace(base, alpha_stable=.8), plan)
    assert full.pair_count == 100 and missing.pair_count == 99
    assert cw._training_feasibility(full, 100) == (True, [])
    assert "candidate_changed_pair_coverage" in cw._training_feasibility(missing, 100)[1]
    assert full.lag_ms == missing.lag_ms == 0


def test_default_public_evaluator_retains_its_lag_estimation(monkeypatch):
    data = dataset(dynamic=True)
    seen = []
    def lag(*args):
        seen.append(True)
        return 10.0
    monkeypatch.setattr(cw, "_estimate_lag_ms", lag)
    assert cw.evaluate_parameters(data, KneeAngleFilterParameters()).lag_ms == 10
    assert seen == [True]


def test_bayesian_integration_is_deterministic_fixed_and_export_only(tmp_path, monkeypatch):
    static = write_archive(tmp_path / "static.zip", dataset(bias=True))
    dynamic = write_archive(tmp_path / "dynamic.zip", dataset(dynamic=True, bias=True))
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (static, dynamic)]
    current = tmp_path / "settings" / "current.yaml"
    current.parent.mkdir()
    current.write_text("untouched current settings", encoding="utf-8")
    for name in ("_candidate_parameters", "_derive_physics_static_parameters", "_estimate_lag_ms"):
        monkeypatch.setattr(cw, name, poison)
    base = replace(KneeAngleFilterParameters(), thigh_offset_deg=3, shank_offset_deg=-2,
                   acc_full_trust_deviation_g=.04, acc_no_trust_deviation_g=.12)
    seen = []
    original = cw._evaluate_training_parameters
    def evaluate(data, params, pairs):
        seen.append((asdict(params), id(pairs), tuple(pairs), json.dumps(data.raw_rows, sort_keys=True)))
        return original(data, params, pairs)
    monkeypatch.setattr(cw, "_evaluate_training_parameters", evaluate)
    results = []
    for name in ("first", "second"):
        result = cw.analyze_calibration_run(static, dynamic, base_parameters=base, output_directory=tmp_path / name)
        assert result["valid"], result
        results.append(json.loads((tmp_path / name / "calibration_report.json").read_text(encoding="utf-8")))
    first, second = results
    bo = first["optimization"]
    assert first["recommended_parameters"] == second["recommended_parameters"]
    assert bo["trace"] == second["optimization"]["trace"]
    assert bo["seed"] == BO_RANDOM_SEED
    assert bo["evaluation_budget"] == BO_EVALUATION_BUDGET == len(bo["trace"]) == 30
    assert bo["fixed_lag_ms"] == 0
    assert bo["frozen_pair_count"] == 100
    assert bo["source_sha256"] == dict(zip(("static", "dynamic"), hashes))
    assert "no interpolation" in bo["reference_sampling_policy"]
    optimized = {"alpha_stable", "alpha_dynamic", "max_accel_correction_deg"}
    # Per run: baseline, static calibrated, thirty BO candidates, static report.
    for block in (seen[:33], seen[33:]):
        dynamic_calls = block[:-1]
        static_params = dynamic_calls[1][0]
        for params, identity, plan, raw_input in dynamic_calls[2:]:
            assert {k: v for k, v in params.items() if k not in optimized} == {
                k: v for k, v in static_params.items() if k not in optimized}
            assert identity == dynamic_calls[0][1]
            assert plan == dynamic_calls[0][2]
            assert raw_input == dynamic_calls[0][3]
    for key in ("dynamic_baseline", "dynamic_static_calibrated", "dynamic_recommended"):
        metric = first["metrics"][key]
        assert metric["pair_count"] == 100 and metric["lag_ms"] == 0
        assert math.isfinite(metric["mae_deg"]) and math.isfinite(metric["rmse_deg"])
    recommended_rmse = first["metrics"]["dynamic_recommended"]["rmse_deg"]
    assert recommended_rmse == min(o["objective_rmse_deg"] for o in bo["trace"] if o["feasible"])
    assert recommended_rmse <= first["metrics"]["dynamic_static_calibrated"]["rmse_deg"] + 1e-9
    exported = yaml.safe_load((tmp_path / "first" / cw.SUGGESTIONS_FILENAME).read_text(encoding="utf-8"))
    assert exported["schema_version"] == "vision_imu_filter_suggestions_v2"
    assert exported["final_recommended_settings"] == first["recommended_parameters"]
    assert exported["static_stage"]["suggestion"] == first["static_suggestion"]
    assert exported["optimization"]["evaluation_budget"] == 30
    KneeAngleFilterParameters.load_yaml(tmp_path / "first" / cw.SUGGESTIONS_FILENAME).validate()
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in (static, dynamic)] == hashes
    assert current.read_text(encoding="utf-8") == "untouched current settings"
    assert "training trial only" in " ".join(first["limitations"])


@pytest.mark.parametrize("failure", ["mismatched_sync", "invalid_raw", "nonfinite_visual", "infeasible_bo"])
def test_failed_data_or_bo_is_invalid_without_grid_fallback(tmp_path, monkeypatch, failure):
    static_data, dynamic_data = dataset(), dataset(dynamic=True)
    if failure == "mismatched_sync":
        dynamic_data.sync_rows[5]["imu_sample_index"] = "51"
    elif failure == "invalid_raw":
        dynamic_data.raw_rows[5]["shank_gx_raw"] = "nan"
    elif failure == "nonfinite_visual":
        for row in dynamic_data.sync_rows:
            row["visual_knee_angle_deg"] = "nan"
    else:
        def fail(*args, **kwargs):
            raise RuntimeError("Bayesian optimization produced no feasible filter candidate")
        monkeypatch.setattr(cw, "optimize_filter_parameters", fail)
    monkeypatch.setattr(cw, "_candidate_parameters", poison)
    static = write_archive(tmp_path / "static.zip", static_data)
    dynamic = write_archive(tmp_path / "dynamic.zip", dynamic_data)
    result = cw.analyze_calibration_run(static, dynamic, base_parameters=KneeAngleFilterParameters(),
                                        output_directory=tmp_path / "invalid")
    assert result["valid"] is False and result["recommended_parameters"] == {}
    report = json.loads((tmp_path / "invalid" / "calibration_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "INVALID" and report["failed_gates"]
    exported = yaml.safe_load((tmp_path / "invalid" / cw.SUGGESTIONS_FILENAME).read_text(encoding="utf-8"))
    assert exported["final_recommended_settings"] is None
    assert "stm32_control" not in exported


def test_workflow_infeasible_bo_returns_invalid_and_keeps_static_retryable(tmp_path, monkeypatch):
    workflow = cw.CalibrationWorkflow(tmp_path / "runs")
    base = KneeAngleFilterParameters()
    prepared = workflow.prepare_test("static", base_parameters=base)
    assert "Straight-Leg Calibration" in workflow.status().message
    directory = workflow.run_directory()
    static = write_archive(directory / prepared["archive_name"], dataset())
    assert workflow.finish_test("static", static).state == "AWAITING_DYNAMIC"
    prepared = workflow.prepare_test("dynamic", base_parameters=base)
    dynamic = write_archive(directory / prepared["archive_name"], dataset(dynamic=True))
    monkeypatch.setattr(cw, "optimize_filter_parameters", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("No feasible BO candidate")))
    state = workflow.finish_test("dynamic", dynamic)
    assert state.state == "INVALID" and not state.valid
    assert state.static_suggestion and not state.dynamic_suggestion
    assert static.is_file() and dynamic.is_file()

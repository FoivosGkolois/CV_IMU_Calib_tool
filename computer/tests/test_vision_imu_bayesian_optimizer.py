from __future__ import annotations

from dataclasses import dataclass

import pytest

from computer.vision_imu_calibration.bayesian_optimizer import optimize_filter_parameters
from computer.vision_imu_calibration.imu_filter import KneeAngleFilterParameters


@dataclass(frozen=True)
class _Metric:
    rmse_deg: float

    def to_dict(self) -> dict[str, float]:
        return {"rmse_deg": self.rmse_deg}


def _evaluate(parameters: KneeAngleFilterParameters) -> _Metric:
    value = (
        120.0 * (parameters.alpha_stable - 0.88) ** 2
        + 220.0 * (parameters.alpha_dynamic - 0.977) ** 2
        + 0.02 * (parameters.max_accel_correction_deg - 9.0) ** 2
    )
    return _Metric(rmse_deg=value)


def _feasible(_metric: _Metric) -> tuple[bool, list[str]]:
    return True, []


def test_gp_bayesian_optimizer_is_deterministic_and_budgeted() -> None:
    base = KneeAngleFilterParameters()
    first = optimize_filter_parameters(
        base,
        _evaluate,
        _feasible,
        evaluation_budget=18,
        seed=1234,
    )
    second = optimize_filter_parameters(
        base,
        _evaluate,
        _feasible,
        evaluation_budget=18,
        seed=1234,
    )

    assert first.evaluation_budget == 18
    assert len(first.observations) == 18
    assert first.parameters.alpha_stable == pytest.approx(second.parameters.alpha_stable)
    assert first.parameters.alpha_dynamic == pytest.approx(second.parameters.alpha_dynamic)
    assert first.parameters.max_accel_correction_deg == pytest.approx(
        second.parameters.max_accel_correction_deg
    )
    assert first.metrics.rmse_deg < _evaluate(base).rmse_deg
    assert first.gp_uncertainty["candidate_pool_max_std_rmse_deg"] >= 0.0


def test_gp_bayesian_optimizer_keeps_static_parameters_fixed() -> None:
    base = KneeAngleFilterParameters(
        straight_leg_offset_deg=3.2,
        thigh_gyro_bias_deg_s=0.4,
        shank_gyro_bias_deg_s=-0.3,
        gyro_deadband_deg_s=0.8,
    )
    result = optimize_filter_parameters(
        base,
        _evaluate,
        _feasible,
        evaluation_budget=10,
        seed=7,
    )

    assert result.parameters.straight_leg_offset_deg == pytest.approx(3.2)
    assert result.parameters.thigh_gyro_bias_deg_s == pytest.approx(0.4)
    assert result.parameters.shank_gyro_bias_deg_s == pytest.approx(-0.3)
    assert result.parameters.gyro_deadband_deg_s == pytest.approx(0.8)

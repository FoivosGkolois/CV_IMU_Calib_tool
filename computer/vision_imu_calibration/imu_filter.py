from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .imu_serial import ImuCalibrationSample

SAMPLE_PERIOD_S = 0.01
MIN_VALID_DT_S = 0.001
MAX_VALID_DT_S = 0.250
DEFAULT_BUNDLED_FILTER_CONFIG = Path(__file__).with_name("default_knee_filter.yaml")


def resolve_filter_config_path(
    explicit_path: str | Path | None = None,
    *,
    working_directory: str | Path | None = None,
) -> Path:
    """Resolve the active filter configuration for full-repo or standalone use.

    An explicitly provided path is strict. Without one, a full thesis checkout
    uses ``settings/current.yaml``; a standalone calibration-tool copy falls
    back to the bundled, versioned snapshot.
    """

    if explicit_path is not None:
        candidate = Path(explicit_path).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Knee-angle filter config not found: {candidate}")
        return candidate

    base = Path(working_directory) if working_directory is not None else Path.cwd()
    thesis_candidate = (base / "settings" / "current.yaml").expanduser().resolve()
    if thesis_candidate.is_file():
        return thesis_candidate

    bundled = DEFAULT_BUNDLED_FILTER_CONFIG.resolve()
    if not bundled.is_file():
        raise FileNotFoundError(
            "No settings/current.yaml and no bundled knee-angle filter snapshot were found"
        )
    return bundled


@dataclass(frozen=True)
class KneeAngleFilterParameters:
    rotation_axis: int = 0
    gyro_rate_sign: float = 1.0
    acc_axis_1_sign: float = -1.0
    acc_axis_2_sign: float = 1.0
    thigh_offset_deg: float = 0.0
    shank_offset_deg: float = 0.0
    relative_angle_sign: float = 1.0
    straight_leg_offset_deg: float = 0.0
    gyro_deadband_deg_s: float = 1.0
    thigh_gyro_bias_deg_s: float = 0.0
    shank_gyro_bias_deg_s: float = 0.0
    alpha_stable: float = 0.90
    alpha_dynamic: float = 0.975
    acc_full_trust_deviation_g: float = 0.06
    acc_no_trust_deviation_g: float = 0.30
    max_accel_correction_deg: float = 10.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "KneeAngleFilterParameters":
        axis_value = mapping.get("rotation_axis", "x")
        if isinstance(axis_value, str):
            axis_lookup = {"x": 0, "y": 1, "z": 2}
            normalized = axis_value.strip().lower()
            if normalized not in axis_lookup:
                raise ValueError(f"Unsupported rotation_axis {axis_value!r}; expected x, y, or z")
            rotation_axis = axis_lookup[normalized]
        else:
            rotation_axis = int(axis_value)
            if rotation_axis not in {0, 1, 2}:
                raise ValueError("rotation_axis must be 0, 1, 2 or x, y, z")

        params = cls(
            rotation_axis=rotation_axis,
            gyro_rate_sign=float(mapping.get("gyro_rate_sign", 1.0)),
            acc_axis_1_sign=float(mapping.get("acc_axis_1_sign", -1.0)),
            acc_axis_2_sign=float(mapping.get("acc_axis_2_sign", 1.0)),
            thigh_offset_deg=float(mapping.get("thigh_offset_deg", 0.0)),
            shank_offset_deg=float(mapping.get("shank_offset_deg", 0.0)),
            relative_angle_sign=float(mapping.get("relative_angle_sign", 1.0)),
            straight_leg_offset_deg=float(mapping.get("straight_leg_offset_deg", 0.0)),
            gyro_deadband_deg_s=float(mapping.get("gyro_deadband_deg_s", 1.0)),
            thigh_gyro_bias_deg_s=float(mapping.get("thigh_gyro_bias_deg_s", 0.0)),
            shank_gyro_bias_deg_s=float(mapping.get("shank_gyro_bias_deg_s", 0.0)),
            alpha_stable=float(mapping.get("alpha_stable", 0.90)),
            alpha_dynamic=float(mapping.get("alpha_dynamic", 0.975)),
            acc_full_trust_deviation_g=float(mapping.get("acc_full_trust_deviation_g", 0.06)),
            acc_no_trust_deviation_g=float(mapping.get("acc_no_trust_deviation_g", 0.30)),
            max_accel_correction_deg=float(mapping.get("max_accel_correction_deg", 10.0)),
        )
        params.validate()
        return params

    @classmethod
    def load_yaml(cls, path: str | Path) -> "KneeAngleFilterParameters":
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is not installed. Re-run 08_SETUP_VISION_IMU_PC.bat") from exc
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Knee-angle filter config not found: {config_path}")
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        mapping = payload.get("stm32_control", {}).get("knee_angle")
        if not isinstance(mapping, Mapping):
            raise ValueError(f"Missing stm32_control.knee_angle mapping in {config_path}")
        return cls.from_mapping(mapping)

    def validate(self) -> None:
        numeric = asdict(self)
        for key, value in numeric.items():
            if key == "rotation_axis":
                continue
            if not math.isfinite(float(value)):
                raise ValueError(f"Non-finite knee-angle parameter: {key}")
        if not 0.0 <= self.alpha_stable <= 1.0:
            raise ValueError("alpha_stable must be within [0, 1]")
        if not 0.0 <= self.alpha_dynamic <= 1.0:
            raise ValueError("alpha_dynamic must be within [0, 1]")
        if self.acc_full_trust_deviation_g < 0.0 or self.acc_no_trust_deviation_g < 0.0:
            raise ValueError("accelerometer trust deviations must be non-negative")
        if self.max_accel_correction_deg < 0.0:
            raise ValueError("max_accel_correction_deg must be non-negative")
        if self.gyro_deadband_deg_s < 0.0:
            raise ValueError("gyro_deadband_deg_s must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rotation_axis_name"] = ("x", "y", "z")[self.rotation_axis]
        return payload


@dataclass(frozen=True)
class KneeAngleEstimate:
    host_monotonic_ns: int = 0
    sample_index: int | None = None
    mcu_timestamp_us: int | None = None
    valid: bool = False
    knee_angle_deg: float | None = None
    knee_speed_deg_s: float | None = None
    thigh_angle_deg: float | None = None
    shank_angle_deg: float | None = None
    reason: str = "no_sample"

    def to_record(self) -> dict[str, Any]:
        return {
            "host_monotonic_ns": self.host_monotonic_ns,
            "sample_index": self.sample_index,
            "mcu_timestamp_us": self.mcu_timestamp_us,
            "filter_valid": self.valid,
            "knee_angle_deg": self.knee_angle_deg,
            "knee_speed_deg_s": self.knee_speed_deg_s,
            "thigh_angle_deg": self.thigh_angle_deg,
            "shank_angle_deg": self.shank_angle_deg,
            "filter_reason": self.reason,
        }


class KneeAngleFilterMirror:
    """Stateful Python mirror of the knee-angle estimator adapted for the standalone calibration tool.

    This module performs acquisition-time estimation only. It does not search,
    score, tune, or recommend parameters. The presentation convention is:
    180 deg = straight leg, smaller angle = more flexion.
    """

    implementation_id = "stm32_knee_angle_cpp_mirror_v1"

    def __init__(self, parameters: KneeAngleFilterParameters) -> None:
        self.parameters = parameters
        self.reset()

    def reset(self) -> None:
        self._filter_initialized = False
        self._thigh_angle_deg = 0.0
        self._shank_angle_deg = 0.0
        self._relative_angle_initialized = False
        self._unwrapped_relative_angle_deg = 0.0
        self._previous_timestamp_us = 0

    def update(self, sample: ImuCalibrationSample) -> KneeAngleEstimate:
        if not (sample.thigh_ok and sample.shank_ok):
            self.reset()
            return KneeAngleEstimate(
                host_monotonic_ns=sample.host_monotonic_ns,
                sample_index=sample.sample_index,
                mcu_timestamp_us=sample.mcu_timestamp_us,
                reason="imu_not_ready",
            )

        record = sample.to_record()
        thigh = _ImuVector.from_record(record, "thigh")
        shank = _ImuVector.from_record(record, "shank")
        p = self.parameters

        thigh_acc_angle_deg = self._acceleration_angle_deg(thigh)
        shank_acc_angle_deg = self._acceleration_angle_deg(shank)

        thigh_gyro_rate = self._gyro_rate_for_axis(thigh) - p.thigh_gyro_bias_deg_s
        shank_gyro_rate = self._gyro_rate_for_axis(shank) - p.shank_gyro_bias_deg_s
        thigh_gyro_rate = self._apply_deadband(thigh_gyro_rate)
        shank_gyro_rate = self._apply_deadband(shank_gyro_rate)

        knee_speed_deg_s = -p.relative_angle_sign * (shank_gyro_rate - thigh_gyro_rate)
        dt_s = self._valid_delta_time_s(sample.mcu_timestamp_us)

        if not self._filter_initialized:
            self._thigh_angle_deg = thigh_acc_angle_deg
            self._shank_angle_deg = shank_acc_angle_deg
            self._filter_initialized = True
        else:
            self._thigh_angle_deg = self._adaptive_filter(
                self._thigh_angle_deg,
                thigh_acc_angle_deg,
                thigh_gyro_rate,
                thigh.norm_g,
                dt_s,
            )
            self._shank_angle_deg = self._adaptive_filter(
                self._shank_angle_deg,
                shank_acc_angle_deg,
                shank_gyro_rate,
                shank.norm_g,
                dt_s,
            )

        self._previous_timestamp_us = sample.mcu_timestamp_us
        thigh_angle_deg = self._thigh_angle_deg - p.thigh_offset_deg
        shank_angle_deg = self._shank_angle_deg - p.shank_offset_deg
        wrapped_relative = self._shortest_angle_error(shank_angle_deg, thigh_angle_deg)
        if not self._relative_angle_initialized:
            self._unwrapped_relative_angle_deg = wrapped_relative
            self._relative_angle_initialized = True
        else:
            self._unwrapped_relative_angle_deg = self._unwrap_near_reference(
                wrapped_relative,
                self._unwrapped_relative_angle_deg,
            )
        flexion_angle_deg = (
            p.relative_angle_sign * self._unwrapped_relative_angle_deg
            - p.straight_leg_offset_deg
        )
        knee_angle_deg = 180.0 - flexion_angle_deg

        values = (knee_angle_deg, knee_speed_deg_s, thigh_angle_deg, shank_angle_deg)
        if not all(math.isfinite(value) for value in values):
            self.reset()
            return KneeAngleEstimate(
                host_monotonic_ns=sample.host_monotonic_ns,
                sample_index=sample.sample_index,
                mcu_timestamp_us=sample.mcu_timestamp_us,
                reason="non_finite_output",
            )

        return KneeAngleEstimate(
            host_monotonic_ns=sample.host_monotonic_ns,
            sample_index=sample.sample_index,
            mcu_timestamp_us=sample.mcu_timestamp_us,
            valid=True,
            knee_angle_deg=knee_angle_deg,
            knee_speed_deg_s=knee_speed_deg_s,
            thigh_angle_deg=thigh_angle_deg,
            shank_angle_deg=shank_angle_deg,
            reason="ok",
        )

    def _acceleration_angle_deg(self, imu: "_ImuVector") -> float:
        p = self.parameters
        if p.rotation_axis == 0:
            axis_1 = p.acc_axis_1_sign * imu.ay_g
            axis_2 = p.acc_axis_2_sign * imu.az_g
        elif p.rotation_axis == 2:
            axis_1 = p.acc_axis_1_sign * imu.ay_g
            axis_2 = p.acc_axis_2_sign * imu.ax_g
        else:
            axis_1 = p.acc_axis_1_sign * imu.az_g
            axis_2 = p.acc_axis_2_sign * imu.ax_g
        return math.degrees(math.atan2(axis_1, axis_2))

    def _gyro_rate_for_axis(self, imu: "_ImuVector") -> float:
        axis_value = (imu.gx_dps, imu.gy_dps, imu.gz_dps)[self.parameters.rotation_axis]
        return self.parameters.gyro_rate_sign * axis_value

    def _apply_deadband(self, value: float) -> float:
        return 0.0 if abs(value) < self.parameters.gyro_deadband_deg_s else value

    def _adaptive_filter(
        self,
        previous_angle_deg: float,
        acceleration_angle_deg: float,
        gyro_rate_deg_s: float,
        acceleration_norm_g: float,
        dt_s: float,
    ) -> float:
        prediction = previous_angle_deg + gyro_rate_deg_s * dt_s
        accel_unwrapped = self._unwrap_near_reference(acceleration_angle_deg, prediction)
        alpha = self._adaptive_alpha(acceleration_norm_g)
        correction = self._shortest_angle_error(accel_unwrapped, prediction)
        limit = self.parameters.max_accel_correction_deg
        correction = max(-limit, min(limit, correction))
        return prediction + (1.0 - alpha) * correction

    def _adaptive_alpha(self, acceleration_norm_g: float) -> float:
        p = self.parameters
        full = p.acc_full_trust_deviation_g
        no_trust = p.acc_no_trust_deviation_g
        if no_trust <= full:
            no_trust = full + 0.001
        deviation = abs(acceleration_norm_g - 1.0)
        if deviation <= full:
            return p.alpha_stable
        if deviation >= no_trust:
            return p.alpha_dynamic
        ratio = (deviation - full) / (no_trust - full)
        return p.alpha_stable + ratio * (p.alpha_dynamic - p.alpha_stable)

    def _valid_delta_time_s(self, timestamp_us: int) -> float:
        if self._previous_timestamp_us == 0:
            return SAMPLE_PERIOD_S
        delta_us = (int(timestamp_us) - int(self._previous_timestamp_us)) & 0xFFFFFFFF
        dt_s = delta_us / 1_000_000.0
        return dt_s if MIN_VALID_DT_S <= dt_s <= MAX_VALID_DT_S else SAMPLE_PERIOD_S

    @staticmethod
    def _shortest_angle_error(target_deg: float, reference_deg: float) -> float:
        error = target_deg - reference_deg
        while error > 180.0:
            error -= 360.0
        while error < -180.0:
            error += 360.0
        return error

    @classmethod
    def _unwrap_near_reference(cls, new_angle_deg: float, reference_angle_deg: float) -> float:
        return reference_angle_deg + cls._shortest_angle_error(new_angle_deg, reference_angle_deg)


@dataclass(frozen=True)
class _ImuVector:
    ax_g: float
    ay_g: float
    az_g: float
    gx_dps: float
    gy_dps: float
    gz_dps: float

    @property
    def norm_g(self) -> float:
        return math.sqrt(self.ax_g * self.ax_g + self.ay_g * self.ay_g + self.az_g * self.az_g)

    @classmethod
    def from_record(cls, record: Mapping[str, Any], prefix: str) -> "_ImuVector":
        return cls(
            ax_g=float(record[f"{prefix}_ax_g"]),
            ay_g=float(record[f"{prefix}_ay_g"]),
            az_g=float(record[f"{prefix}_az_g"]),
            gx_dps=float(record[f"{prefix}_gx_dps"]),
            gy_dps=float(record[f"{prefix}_gy_dps"]),
            gz_dps=float(record[f"{prefix}_gz_dps"]),
        )

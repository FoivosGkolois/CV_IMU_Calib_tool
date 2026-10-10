"""Read-only static validation of recorded 2-D knee angles against measured references.

Usage:
    python -m computer.vision_imu_calibration.reference_validation \
        --manifest reference_trials.csv --output reference_report.json

Does not fit/correct pose angles, access the camera, or modify calibration state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from zipfile import BadZipFile, ZipFile


REQUIRED_COLUMNS = ("trial_id", "reference_angle_deg", "recording_zip")
ANGLE_COLUMNS = ("raw_knee_angle_deg", "visual_knee_angle_deg")
GONIOMETER_TARGETS = (90, 120, 140, 160, 175)


def measured_reference(value: object) -> float:
    """Targets are prompts; only the operator's finite measurement is a reference."""
    if isinstance(value, bool):
        raise ValueError("Reference angle must be numeric in 0..180")
    reference = _finite_number(value, "reference_angle_deg")
    if not 0 <= reference <= 180:
        raise ValueError("Reference angle must be in 0..180")
    return reference


# At 1280x720 the full vision pipeline may only produce ~5 processed FPS.
# Stationarity is evaluated in degrees, not from absolute image landmark motion:
# the latter mixes pose-model jitter, small body translations, and actual rotation.
GUIDED_MIN_VALID_FRAMES = 8
GUIDED_MIN_VALID_FRACTION = 0.70
GUIDED_MAX_GAP_S = 1.0
GUIDED_MIN_VALID_SPAN_S = 2.0
GUIDED_MAX_CORE_ANGLE_SPAN_DEG = 8.0
GUIDED_MAX_END_TO_END_DRIFT_DEG = 5.0


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def evaluate_static_samples(samples: list[dict], *, start_ns: int | None = None,
                            end_ns: int | None = None) -> dict:
    """Accept a guided 3-second hold with time coverage and robust angular stability.

    Tolerates isolated tracking dropouts and pixel-level landmark jitter; still
    rejects low frame coverage, long temporal gaps, leg swaps, invalid values,
    and substantial knee-angle movement. Raw/filtered variation is retained in
    the report and is NOT confused with angle accuracy against the goniometer.
    """
    reason = ""
    valid = [s for s in samples if s.get("valid") is True]
    coverage = len(valid) / len(samples) if samples else 0.0
    if len(valid) < GUIDED_MIN_VALID_FRAMES:
        reason = "Too few valid frames (need 8); retry"
    elif coverage < GUIDED_MIN_VALID_FRACTION:
        reason = "Insufficient valid tracking coverage (need 70%); retry"

    try:
        times = [s["host_monotonic_ns"] / 1e9 for s in samples]
        valid_times = [s["host_monotonic_ns"] / 1e9 for s in valid]
    except (KeyError, TypeError, ValueError):
        times, valid_times = [], []
        reason = reason or "Missing camera timestamps; retry"
    if any(not math.isfinite(t) for t in times):
        reason = reason or "Invalid camera timestamps; retry"
    if not times or times[-1] - times[0] < GUIDED_MIN_VALID_SPAN_S:
        reason = reason or "Hold too short / camera interrupted; retry"
    if len(valid_times) < 2 or valid_times[-1] - valid_times[0] < GUIDED_MIN_VALID_SPAN_S:
        reason = reason or "Insufficient valid camera time coverage; retry"
    if any(not 0 < b - a <= GUIDED_MAX_GAP_S for a, b in zip(times, times[1:])):
        reason = reason or "Camera sample gap; retry"
    if any(not 0 < b - a <= GUIDED_MAX_GAP_S for a, b in zip(valid_times, valid_times[1:])):
        reason = reason or "Valid tracking gap; retry"
    if valid_times and ((start_ns is not None and not 0 <= valid_times[0] - start_ns / 1e9 <= GUIDED_MAX_GAP_S)
                        or (end_ns is not None and not 0 <= end_ns / 1e9 - valid_times[-1] <= GUIDED_MAX_GAP_S)):
        reason = reason or "Valid camera samples missing near hold start/end; retry"
    if len({s.get("tracked_leg") for s in valid}) > 1:
        reason = reason or "Tracked leg changed; retry"

    angles: dict[str, list[float]] = {}
    for key in ANGLE_COLUMNS:
        values = [s.get(key) for s in valid]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 180
               for v in values):
            reason = reason or "Missing / nonfinite angle; retry"
        else:
            angles[key] = [float(v) for v in values]

    for key in ("hip_x", "hip_y", "knee_x", "knee_y", "ankle_x", "ankle_y"):
        values = [s.get(key) for s in valid]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
               for v in values):
            reason = reason or "Missing / invalid landmark; retry"

    stability: dict[str, dict[str, float]] = {}
    for key, values in angles.items():
        if not values:
            continue
        edge_count = max(2, len(values) // 3)
        core_span = _percentile(values, 0.8) - _percentile(values, 0.2)
        end_to_end_drift = abs(statistics.median(values[:edge_count]) - statistics.median(values[-edge_count:]))
        stability[key] = {"full_range_deg": max(values) - min(values),
                          "central_20_80_range_deg": core_span,
                          "early_late_median_shift_deg": end_to_end_drift}
    filtered = stability.get("visual_knee_angle_deg")
    if filtered and (filtered["central_20_80_range_deg"] > GUIDED_MAX_CORE_ANGLE_SPAN_DEG
                     or filtered["early_late_median_shift_deg"] > GUIDED_MAX_END_TO_END_DRIFT_DEG):
        reason = reason or "Knee angle moved substantially during hold; retry"

    return {"accepted": not reason, "reason": reason, "frames_total": len(samples),
            "frames_valid": len(valid), "coverage": coverage,
            "valid_time_span_s": valid_times[-1] - valid_times[0] if len(valid_times) > 1 else 0.0,
            "stability_deg": stability,
            "criteria": {"min_valid_frames": GUIDED_MIN_VALID_FRAMES,
                         "min_valid_fraction": GUIDED_MIN_VALID_FRACTION,
                         "min_valid_span_s": GUIDED_MIN_VALID_SPAN_S,
                         "max_valid_gap_s": GUIDED_MAX_GAP_S,
                         "max_filtered_central_span_deg": GUIDED_MAX_CORE_ANGLE_SPAN_DEG,
                         "max_filtered_early_late_shift_deg": GUIDED_MAX_END_TO_END_DRIFT_DEG}}


def guided_capture_report(path: Path, inputs: dict, samples: list[dict]) -> dict:
    reference = measured_reference(inputs.get("reference_angle_deg"))
    hold = evaluate_static_samples(samples, start_ns=inputs.get("hold_start_host_monotonic_ns"),
                                   end_ns=inputs.get("hold_end_host_monotonic_ns"))
    metrics = evaluate_recording(path, reference_angle_deg=reference, trim_seconds=0,
                                 min_valid_frames=GUIDED_MIN_VALID_FRAMES)
    if not metrics["sufficient_data"]:
        hold.update(accepted=False, reason=hold["reason"] or "Recording has too few valid frames")
    if metrics["frames_total"] != len(samples):
        hold.update(accepted=False, reason="Recording/sample coverage mismatch; retry")
    return {"schema_version": 1, **inputs, "recording_zip": path.name,
            "angle_convention": "180 deg straight; smaller means increased flexion",
            "status": "Measured" if hold["accepted"] else "Needs retry",
            "hold": hold, "metrics": metrics, "samples": samples,
            "interpretation": "Independent static 2-D comparison; no correction or accuracy certification"}


def guided_repeatability(captures: list[dict]) -> list[dict]:
    """Per prompt; reference differences remain visible in error repeatability."""
    result = []
    for target in GONIOMETER_TARGETS:
        rows = [r for r in captures if r.get("target_deg") == target and r.get("status") == "Measured"]
        point = {"target_deg": target, "accepted_captures": len(rows)}
        for key in ("raw", "filtered"):
            means = [r["metrics"][key]["mean_deg"] for r in rows]
            errors = [r["metrics"][key]["bias_deg"] for r in rows]
            point[key] = {"mean_sd_deg": statistics.stdev(means) if len(means) > 1 else None,
                          "signed_error_sd_deg": statistics.stdev(errors) if len(errors) > 1 else None}
        result.append(point)
    return result


def _finite_number(value: str | None, name: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Invalid {name}: {value!r} (not finite)")
    return number


def _optional_number(value: str | None, name: str) -> float | None:
    return None if value is None or not value.strip() else _finite_number(value, name)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _angle_summary(values: list[float], reference: float) -> dict[str, float | int] | None:
    if not values:
        return None
    errors = [angle - reference for angle in values]
    return {
        "n": len(values),
        "mean_deg": statistics.mean(values),
        "sd_deg": statistics.stdev(values) if len(values) > 1 else 0.0,
        "bias_deg": statistics.mean(errors),
        "mae_frames_deg": statistics.mean(abs(error) for error in errors),
        "rmse_frames_deg": math.sqrt(statistics.mean(error * error for error in errors)),
    }


def evaluate_recording(
    path: Path, *, reference_angle_deg: float, trim_seconds: float = 1.0,
    min_valid_frames: int = 5,
) -> dict[str, object]:
    """Analyze one stationary hold; do not infer a reference from the vision signal."""
    reference_angle_deg = measured_reference(reference_angle_deg)
    if not path.is_file():
        raise FileNotFoundError(f"Recording ZIP not found: {path}")
    try:
        with ZipFile(path) as archive:
            with archive.open("pose_timeline.csv") as stream:
                reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""))
                required = {"recording_time_s", "tracking_valid", *ANGLE_COLUMNS}
                missing = required.difference(reader.fieldnames or ())
                if missing:
                    raise ValueError(f"{path.name}: pose_timeline.csv missing {sorted(missing)}")
                rows = list(reader)
            try:
                metadata = json.loads(archive.read("recording_metadata.json"))
            except KeyError:
                metadata = {}
    except (BadZipFile, KeyError) as exc:
        raise ValueError(f"{path.name}: invalid recording archive: {exc}") from exc
    if not rows:
        raise ValueError(f"{path.name}: pose_timeline.csv is empty")

    timed_rows: list[tuple[float, dict[str, str | None]]] = []
    for row in rows:
        timed_rows.append((_finite_number(row.get("recording_time_s"), "recording_time_s"), row))
    start = min(t for t, _ in timed_rows) + trim_seconds
    end = max(t for t, _ in timed_rows) - trim_seconds
    selected = [row for t, row in timed_rows if start <= t <= end] if start <= end else []

    accepted = [row for row in selected if str(row.get("tracking_valid", "")).strip().lower() in {"true", "1"}]
    values: dict[str, list[float]] = {key: [] for key in ANGLE_COLUMNS}
    for row in accepted:
        for key in ANGLE_COLUMNS:
            item = row.get(key)
            if item is None or not item.strip():
                continue
            value = _finite_number(item, key)
            if not 0.0 <= value <= 180.0:
                raise ValueError(f"{path.name}: {key} outside 0..180 degrees: {value}")
            values[key].append(value)

    sample_summaries = {
        key: _angle_summary(values[key], reference_angle_deg) for key in ANGLE_COLUMNS
    }
    return {
        "recording_sha256": _digest(path),
        "source": metadata.get("source"),
        "recorded_pose_backend": (metadata.get("tuning") or {}).get("pose_backend"),
        "recorded_tuning": metadata.get("tuning"),
        "recording_created_at": metadata.get("created_at"),
        "frames_total": len(rows),
        "frames_in_hold_window": len(selected),
        "frames_tracking_valid": len(accepted),
        "valid_tracking_fraction": len(accepted) / len(selected) if selected else 0.0,
        "tracked_legs": dict(Counter(str(row.get("tracked_leg") or "") for row in accepted)),
        "raw": sample_summaries["raw_knee_angle_deg"],
        "filtered": sample_summaries["visual_knee_angle_deg"],
        "sufficient_data": all(len(values[key]) >= min_valid_frames for key in ANGLE_COLUMNS),
    }


def evaluate_manifest(
    manifest: Path, *, trim_seconds: float = 1.0, min_valid_frames: int = 5,
) -> dict[str, object]:
    if trim_seconds < 0 or not math.isfinite(trim_seconds):
        raise ValueError("trim_seconds must be finite and nonnegative")
    if min_valid_frames < 2:
        raise ValueError("min_valid_frames must be at least 2")
    with manifest.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = set(REQUIRED_COLUMNS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        entries = list(reader)
    if not entries:
        raise ValueError("Manifest contains no trials")

    result: list[dict[str, object]] = []
    used_ids: set[str] = set()
    used_files: set[Path] = set()
    for row in entries:
        trial_id = (row.get("trial_id") or "").strip()
        if not trial_id or trial_id in used_ids:
            raise ValueError(f"Empty or duplicate trial_id: {trial_id!r}")
        used_ids.add(trial_id)
        filename = (row.get("recording_zip") or "").strip()
        if not filename:
            raise ValueError(f"{trial_id}: recording_zip is missing")
        archive = (manifest.parent / filename).resolve()
        if archive in used_files:
            raise ValueError(f"{trial_id}: recording ZIP is reused by another trial")
        used_files.add(archive)
        reference = _finite_number(row.get("reference_angle_deg"), "reference_angle_deg")
        if not 0 <= reference <= 180:
            raise ValueError(f"{trial_id}: reference_angle_deg must be in 0..180")
        yaw = _optional_number(row.get("orientation_deg"), "orientation_deg")
        uncertainty = _optional_number(row.get("reference_uncertainty_deg"), "reference_uncertainty_deg")
        if uncertainty is not None and uncertainty < 0:
            raise ValueError(f"{trial_id}: reference_uncertainty_deg must be nonnegative")
        metrics = evaluate_recording(
            archive, reference_angle_deg=reference,
            trim_seconds=trim_seconds, min_valid_frames=min_valid_frames,
        )
        result.append({
            "trial_id": trial_id,
            "recording_zip": filename,
            "reference_angle_deg": reference,
            "reference_method": (row.get("reference_method") or "unspecified").strip(),
            "reference_uncertainty_deg": uncertainty,
            "orientation_deg": yaw,
            **metrics,
        })

    complete = all(bool(trial["sufficient_data"]) for trial in result)
    def aggregate(key: str) -> dict[str, float] | None:
        if not complete:
            return None
        errors = [float(trial[key]["bias_deg"]) for trial in result]  # type: ignore[index]
        return {
            "trial_count": len(errors),
            "mean_signed_error_deg": statistics.mean(errors),
            "mae_of_trial_means_deg": statistics.mean(abs(error) for error in errors),
            "rmse_of_trial_means_deg": math.sqrt(statistics.mean(e * e for e in errors)),
        }

    return {
        "schema_version": 1,
        "status": "COMPLETE" if complete else "INSUFFICIENT_DATA",
        "interpretation": "Descriptive 2-D image-plane static comparison; no camera calibration, correction, anatomical 3-D accuracy claim, or filter tuning is performed.",
        "angle_convention": "180 deg straight; smaller means increased flexion",
        "reference_requirement": "reference_angle_deg must come from an external measurement, not the pose estimator",
        "manifest_sha256": _digest(manifest),
        "trim_seconds_each_end": trim_seconds,
        "minimum_valid_frames": min_valid_frames,
        "raw_summary": aggregate("raw"),
        "filtered_summary": aggregate("filtered"),
        "trials": result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with trial_id,reference_angle_deg,recording_zip")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--trim-seconds", type=float, default=1.0, help="Seconds excluded at each end of the recording")
    args = parser.parse_args(argv)
    try:
        report = evaluate_manifest(args.manifest, trim_seconds=args.trim_seconds)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Reference validation failed: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Static reference report: {args.output} ({report['status']})")
    for trial in report["trials"]:
        filtered = trial["filtered"]
        print(f"  {trial['trial_id']}: ref={trial['reference_angle_deg']:.1f}°, "
              f"filtered={filtered['mean_deg']:.1f}° bias={filtered['bias_deg']:+.1f}° "
              f"valid={trial['frames_tracking_valid']}/{trial['frames_in_hold_window']}"
              if filtered else f"  {trial['trial_id']}: no valid filtered angle")
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

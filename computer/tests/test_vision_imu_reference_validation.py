from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from computer.vision_imu_calibration.reference_validation import (
    evaluate_manifest,
    evaluate_recording,
    main,
)


def _recording(path: Path, angles: list[float | None], *, raw_bias: float = 1.0,
               validity: list[bool] | None = None) -> Path:
    validity = validity if validity is not None else [True] * len(angles)
    fields = ["recording_time_s", "tracking_valid", "visual_knee_angle_deg",
              "raw_knee_angle_deg", "tracked_leg"]
    from io import StringIO
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for index, (angle, valid) in enumerate(zip(angles, validity)):
        writer.writerow({
            "recording_time_s": index * 0.5,
            "tracking_valid": valid,
            "visual_knee_angle_deg": "" if angle is None else angle,
            "raw_knee_angle_deg": "" if angle is None else angle + raw_bias,
            "tracked_leg": "right",
        })
    with ZipFile(path, "w") as z:
        z.writestr("pose_timeline.csv", buffer.getvalue())
        z.writestr("recording_metadata.json", json.dumps({
            "source": "camera:0", "tuning": {"pose_backend": "movenet"},
            "created_at": "2026-10-08T20:00:00Z",
        }))
    return path


def _manifest(path: Path, records: list[tuple[str, int, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "reference_angle_deg", "recording_zip",
                         "orientation_deg", "reference_uncertainty_deg",
                         "reference_method"])
        for trial_id, reference, recording in records:
            writer.writerow([trial_id, reference, recording, 0, 3, "manual_goniometer"])
    return path


def test_evaluates_four_independent_static_recordings_without_tuning(tmp_path: Path):
    rows = []
    for idx, reference in enumerate((175, 150, 120, 90)):
        filename = f"trial_{idx}.zip"
        _recording(tmp_path / filename, [reference + 2] * 21)
        rows.append((f"trial_{idx}", reference, filename))
    manifest = _manifest(tmp_path / "trials.csv", rows)
    report = evaluate_manifest(manifest)
    assert report["status"] == "COMPLETE"
    assert report["raw_summary"]["mae_of_trial_means_deg"] == 3.0
    assert report["filtered_summary"]["mae_of_trial_means_deg"] == 2.0
    assert report["filtered_summary"]["trial_count"] == 4
    assert len(report["manifest_sha256"]) == 64
    for trial in report["trials"]:
        assert trial["frames_in_hold_window"] == 17
        assert trial["frames_tracking_valid"] == 17
        assert trial["filtered"]["sd_deg"] == 0.0
        assert trial["raw"]["bias_deg"] == 3.0
        assert trial["reference_uncertainty_deg"] == 3
        assert trial["recorded_pose_backend"] == "movenet"
        assert len(trial["recording_sha256"]) == 64
    assert not (tmp_path / "calibration.yaml").exists()


def test_invalid_and_missing_frames_are_not_interpolated(tmp_path: Path):
    recording = _recording(tmp_path / "trial.zip", [90] * 21,
                           validity=[True] * 5 + [False] * 7 + [True] * 9)
    report = evaluate_recording(recording, reference_angle_deg=90)
    assert report["frames_in_hold_window"] == 17
    assert report["frames_tracking_valid"] == 10
    assert report["filtered"]["n"] == 10
    assert report["valid_tracking_fraction"] == pytest.approx(10 / 17)


def test_too_few_valid_frames_produces_incomplete_not_fake_accuracy(tmp_path: Path):
    _recording(tmp_path / "short.zip", [120] * 5)
    manifest = _manifest(tmp_path / "trials.csv", [("short", 120, "short.zip")])
    report = evaluate_manifest(manifest)
    assert report["status"] == "INSUFFICIENT_DATA"
    assert report["filtered_summary"] is None
    assert report["trials"][0]["sufficient_data"] is False
    assert main(["--manifest", str(manifest), "--output", str(tmp_path / "report.json")]) == 2


def test_duplicate_archive_is_rejected(tmp_path: Path):
    _recording(tmp_path / "one.zip", [150] * 21)
    manifest = _manifest(tmp_path / "trials.csv", [
        ("first", 150, "one.zip"), ("second", 180, "one.zip")])
    with pytest.raises(ValueError, match="reused"):
        evaluate_manifest(manifest)


def test_invalid_reference_and_missing_zip_are_rejected(tmp_path: Path):
    _recording(tmp_path / "one.zip", [150] * 21)
    manifest = _manifest(tmp_path / "trials.csv", [("first", 200, "one.zip")])
    with pytest.raises(ValueError, match="0..180"):
        evaluate_manifest(manifest)
    manifest = _manifest(tmp_path / "trials.csv", [("first", 150, "not_there.zip")])
    with pytest.raises(FileNotFoundError, match="Recording ZIP not found"):
        evaluate_manifest(manifest)


def test_cli_writes_valid_json_report(tmp_path: Path):
    _recording(tmp_path / "one.zip", [90] * 21)
    manifest = _manifest(tmp_path / "trials.csv", [("one", 90, "one.zip")])
    output = tmp_path / "report.json"
    assert main(["--manifest", str(manifest), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "COMPLETE"
    assert report["trials"][0]["filtered"]["bias_deg"] == 0.0


def test_rejects_invalid_pose_angle(tmp_path: Path):
    _recording(tmp_path / "bad.zip", [190] * 21)
    with pytest.raises(ValueError, match="outside 0..180"):
        evaluate_recording(tmp_path / "bad.zip", reference_angle_deg=90)

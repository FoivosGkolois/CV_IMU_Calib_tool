from __future__ import annotations

import csv
import io
import json
from dataclasses import replace
from zipfile import ZipFile

import pytest

from computer.vision_imu_calibration.reference_validation import (
    evaluate_static_samples, guided_capture_report, guided_repeatability, measured_reference)
from computer.vision_imu_calibration.runtime import VisionRuntime
from computer.vision_imu_calibration.web_app import create_app


def samples():
    return [dict(host_monotonic_ns=int((10 + i * .15) * 1e9), valid=True,
                 tracked_leg="right", raw_knee_angle_deg=122 + (i % 2) * .2,
                 visual_knee_angle_deg=121.5, hip_x=.5, hip_y=.22,
                 knee_x=.5, knee_y=.5, ankle_x=.65, ankle_y=.8) for i in range(21)]


def archive(path, rows):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["recording_time_s", "tracking_valid", "tracked_leg",
                                           "raw_knee_angle_deg", "visual_knee_angle_deg"])
    writer.writeheader()
    for i, row in enumerate(rows):
        writer.writerow(dict(recording_time_s=i * .15, tracking_valid=row["valid"],
                             tracked_leg=row["tracked_leg"], raw_knee_angle_deg=row["raw_knee_angle_deg"],
                             visual_knee_angle_deg=row["visual_knee_angle_deg"]))
    with ZipFile(path, "w") as zip_file:
        zip_file.writestr("pose_timeline.csv", buf.getvalue())
        zip_file.writestr("recording_metadata.json", json.dumps({"source": "synthetic", "tuning": {"pose_backend": "synthetic"}}))


@pytest.mark.parametrize("reference", [None, "", "nan", "inf", -1, 181, True])
def test_manual_reference_rejects_invalid_values(reference):
    with pytest.raises(ValueError):
        measured_reference(reference)


def test_signed_errors_use_measured_reference_instead_of_prompt_and_repeat_cleanly(tmp_path):
    path = tmp_path / "one.zip"
    rows = samples()
    archive(path, rows)
    result = guided_capture_report(path, {"capture_id": "one", "target_deg": 120,
                                          "reference_angle_deg": 119}, rows)
    assert result["status"] == "Measured"
    assert result["metrics"]["raw"]["bias_deg"] == pytest.approx(3 + 10 / 21 * .2)
    assert result["metrics"]["filtered"]["bias_deg"] == pytest.approx(2.5)
    assert result["metrics"]["raw"]["mean_deg"] != result["metrics"]["filtered"]["mean_deg"]
    assert result["hold"]["coverage"] == 1
    failed_rows = samples()
    failed_rows[5]["valid"] = False
    second = tmp_path / "two.zip"
    archive(second, failed_rows)
    failed = guided_capture_report(second, {"target_deg": 120, "reference_angle_deg": 119}, failed_rows)
    assert failed["status"] == "Needs retry"
    assert result["status"] == "Measured"
    repeated = guided_capture_report(path, {"target_deg": 120, "reference_angle_deg": 118}, rows)
    point = guided_repeatability([result, failed, repeated])[1]
    assert point["accepted_captures"] == 2
    assert point["raw"]["mean_sd_deg"] == 0
    assert point["filtered"]["signed_error_sd_deg"] == pytest.approx(2 ** -.5)


@pytest.mark.parametrize("change,reason", [
    (lambda r: r[4].update(valid=False), "dropout"),
    (lambda r: r[4].update(raw_knee_angle_deg=130), "Moving hold"),
    (lambda r: r[4].update(visual_knee_angle_deg=float("nan")), "nonfinite"),
    (lambda r: r[4].update(hip_x=.55), "landmark moved"),
    (lambda r: r[4].update(ankle_x=None), "landmark"),
    (lambda r: r[4].update(tracked_leg="left"), "leg changed"),
    (lambda r: r[4].update(host_monotonic_ns=r[3]["host_monotonic_ns"]), "gap"),
    (lambda r: r.__delitem__(slice(0, 12)), "Too few"),
])
def test_stationary_hold_rejects_dropout_movement_missing_and_insufficient(change, reason):
    rows = samples()
    change(rows)
    result = evaluate_static_samples(rows)
    assert not result["accepted"]
    assert reason in result["reason"]


def test_hold_checks_unobserved_head_tail_gaps_and_recorder_coverage(tmp_path):
    rows = samples()
    assert not evaluate_static_samples(rows, start_ns=int(8e9))["accepted"]
    assert not evaluate_static_samples(rows, end_ns=int(15e9))["accepted"]
    path = tmp_path / "short.zip"
    archive(path, rows[:-1])
    result = guided_capture_report(path, {"target_deg": 120, "reference_angle_deg": 120}, rows)
    assert result["status"] == "Needs retry"
    assert "mismatch" in result["hold"]["reason"]


class Camera:
    description = "test-camera"
    width, height, fps = 640, 480, 30
    camera_index = 0
    active_camera_index = 0


class Estimator:
    description = "test-pose"


def runtime(tmp_path):
    r = VisionRuntime(Camera(), Estimator(), recording_root=tmp_path / "records",
                      calibration_root=tmp_path / "calibration")
    r._status = replace(r._status, state="RUNNING", tracking_valid=True,
                        visual_knee_angle_deg=121.5, raw_knee_angle_deg=122.0)
    return r


def test_gui_controls_endpoints_and_entry_blindness_leave_underlying_samples_unchanged(tmp_path):
    r = runtime(tmp_path)
    client = create_app(r).test_client()
    html = client.get("/").get_data(as_text=True)
    assert "Check Tracking" in html and "Capture 3-second hold" in html
    assert "2-D model cannot prove side-on orientation" in html
    assert client.post("/api/positioning", json={"action": "show"}).status_code == 200
    assert client.post("/api/positioning", json={"action": "finish"}).status_code == 409
    assert client.post("/api/positioning", json=[1]).status_code == 400
    assert client.post("/api/goniometer", json={"action": "begin", "target_deg": 120}).status_code == 400
    r._latest_jpeg = b"old numerical image"
    r._acquisition_history.append({"host_monotonic_ns": __import__('time').perf_counter_ns(),
                                    "visual_knee_angle_deg": 121.5, "visual_valid": True,
                                    "angle_difference_deg": 1})
    response = client.post("/api/goniometer", json={"action": "begin", "target_deg": 120,
                                                    "setup": "S1 / tripod 2m / floor A / manual goniometer"})
    assert response.status_code == 200 and response.json["hide_angle"]
    status = client.get("/api/status").json
    assert status["raw_knee_angle_deg"] is None and status["visual_knee_angle_deg"] is None
    assert r._status.raw_knee_angle_deg == 122
    assert r._latest_jpeg is None
    history = client.get("/api/acquisition/history").json["rows"]
    assert history[0]["visual_knee_angle_deg"] is None
    assert history[0]["angle_difference_deg"] is None
    assert r._acquisition_history[0]["visual_knee_angle_deg"] == 121.5
    assert client.post("/api/goniometer", json={"action": "capture"}).status_code == 409
    assert client.post("/api/goniometer", json={"action": "confirm", "reference_angle_deg": "NaN"}).status_code == 400
    assert r.reference_status()["hide_angle"]
    assert client.post("/api/goniometer", json={"action": "confirm", "reference_angle_deg": 119}).status_code == 200
    assert r._reference_inputs["reference_angle_deg"] == 119
    assert client.get("/api/status").json["raw_knee_angle_deg"] == 122
    assert client.post("/api/goniometer", json={"action": "cancel"}).status_code == 200


def test_confirmed_capture_rejects_changed_profile_and_busy_calibration(tmp_path):
    r = runtime(tmp_path)
    r.reference_action("begin", {"target_deg": 90, "setup": "setup"})
    r.reference_action("confirm", {"reference_angle_deg": 91})
    r.update_tuning({"leg_side": "left"})
    with pytest.raises(RuntimeError, match="profile changed"):
        r.reference_action("capture", {})
    r._reference["state"] = "CAPTURING"
    with pytest.raises(RuntimeError):
        r.start_calibration_test("static")
    with pytest.raises(RuntimeError):
        r.update_tuning({"frame_rotation": "none"})


def test_saved_capture_restart_reuse_requires_explicit_equivalent_setup_and_digest(tmp_path):
    import uuid
    r = runtime(tmp_path)
    root = r.recording_root / "goniometer"
    root.mkdir()
    capture_id = uuid.uuid4().hex
    path = root / f"goniometer_{capture_id}.zip"
    rows = samples()
    archive(path, rows)
    inputs = {"capture_id": capture_id, "target_deg": 120, "reference_angle_deg": 119,
              "setup": "unchanged setup", "profile": r._vision_setup()}
    report = guided_capture_report(path, inputs, rows)
    path.with_suffix(".json").write_text(json.dumps(report), encoding="utf-8")
    r = runtime(tmp_path)
    assert len(r.reference_status()["captures"]) == 1
    payload = {"capture_id": capture_id, "setup": "unchanged setup"}
    with pytest.raises(ValueError, match="confirm"):
        r.reference_action("reuse", payload)
    payload["unchanged_setup"] = True
    assert r.reference_action("reuse", payload)["state"] == "REUSED"
    r.update_tuning({"leg_side": "left"})
    assert not r.reference_status()["reused_capture_id"]
    with pytest.raises(ValueError, match="differs"):
        r.reference_action("reuse", payload)
    r.update_tuning({"leg_side": "right"})
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="missing or changed"):
        r.reference_action("reuse", payload)


def test_malformed_saved_report_is_reported_without_breaking_runtime(tmp_path):
    root = tmp_path / "records" / "goniometer"
    root.mkdir(parents=True)
    (root / "goniometer_bad.json").write_text('{"status":"Measured"}', encoding="utf-8")
    r = runtime(tmp_path)
    assert not r.reference_status()["captures"]
    assert r.reference_status()["load_errors"]


def test_real_runtime_and_existing_recorder_capture_four_synthetic_holds_and_repeat(tmp_path):
    """Exercise actual camera loop, recorder, timer, export, dropout and restart."""
    import math
    import time
    import numpy as np
    pytest.importorskip("cv2")
    from computer.vision_imu_calibration.models import Landmark, PoseFrame

    class SyntheticCamera(Camera):
        width, height = 320, 240
        def open(self):
            pass
        def read(self):
            time.sleep(.04)
            return np.zeros((240, 320, 3), dtype=np.uint8)
        def close(self):
            pass

    class SyntheticPose(Estimator):
        target = 120
        dropout = False
        def open(self):
            pass
        def close(self):
            pass
        def estimate(self, frame, timestamp_ms):
            if self.dropout:
                return None
            aspect = frame.shape[1] / frame.shape[0]
            theta = math.radians(self.target)
            def p(x, y):
                return Landmark(x, y, 1, 1)
            return PoseFrame(timestamp_ms, {"left_hip": p(.5, .22), "left_knee": p(.5, .5),
                                          "left_ankle": p(.5 + math.sin(theta) * .2 / aspect,
                                                          .5 - math.cos(theta) * .2)})
    estimator = SyntheticPose()
    r = VisionRuntime(SyntheticCamera(), estimator, recording_root=tmp_path / "records",
                      calibration_root=tmp_path / "calibration")
    r.update_tuning({"landmark_smoothing": 0, "angle_smoothing": 0, "max_angle_jump_deg": 180})

    def wait_for(predicate, timeout=8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.02)
        pytest.fail(str(r.reference_status()))

    r.start()
    try:
        wait_for(lambda: r.status()["tracking_valid"])
        for target in (90, 120, 150, 175):
            estimator.target = target
            wait_for(lambda: abs(r._status.raw_knee_angle_deg - target) < 1e-6)
            r.reference_action("begin", {"target_deg": target, "setup": "S1 / tripod / floor A / goniometer"})
            wait_for(lambda: r._latest_jpeg is not None)
            assert r.status()["raw_knee_angle_deg"] is None
            r.reference_action("confirm", {"reference_angle_deg": target - 1})
            r.reference_action("capture", {})
            wait_for(lambda: r.reference_status()["state"] not in {"CAPTURING", "FINALIZING"})
            state = r.reference_status()
            assert state["state"] == "MEASURED", state
            assert state["result"]["metrics"]["raw"]["bias_deg"] == pytest.approx(1, abs=1e-6)
        accepted_ids = [c["capture_id"] for c in r.reference_status()["captures"]]
        assert len(accepted_ids) == 4
        # A failed repeat leaves the four accepted points intact.
        r.reference_action("begin", {"target_deg": 175, "setup": "S1 / tripod / floor A / goniometer"})
        r.reference_action("confirm", {"reference_angle_deg": 174})
        r.reference_action("capture", {})
        time.sleep(.5)
        estimator.dropout = True
        time.sleep(.25)
        estimator.dropout = False
        wait_for(lambda: r.reference_status()["state"] not in {"CAPTURING", "FINALIZING"})
        assert r.reference_status()["state"] == "RETRY"
        assert [c["capture_id"] for c in r.reference_status()["captures"]] == accepted_ids
        # Export paths use the existing browser directory route.
        client = create_app(r).test_client()
        first = r.reference_status()["captures"][0]
        assert client.get(first["report_url"]).status_code == 200
        archive_response = client.get(first["download_url"])
        assert archive_response.status_code == 200
        archive_response.close()
        with ZipFile(r._reference_root / first["recording_zip"]) as zipped:
            metadata = json.loads(zipped.read("recording_metadata.json"))
            assert metadata["goniometer_reference"]["reference_angle_deg"] == 89
            assert metadata["goniometer_reference"]["target_deg"] == 90
            assert "pose_timeline.csv" in zipped.namelist()
            assert "raw_imu.csv" in zipped.namelist()
    finally:
        r.stop()
    restarted = VisionRuntime(SyntheticCamera(), estimator, recording_root=tmp_path / "records",
                              calibration_root=tmp_path / "calibration")
    assert len(restarted.reference_status()["captures"]) == 4
    assert not restarted.reference_status()["load_errors"]

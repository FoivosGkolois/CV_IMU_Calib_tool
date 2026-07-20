from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from computer.vision_imu_calibration.download_model import _looks_like_tflite
from computer.vision_imu_calibration.models import LegSide
from computer.vision_imu_calibration.pose_estimator import (
    CropRegion,
    MoveNetThunderPoseEstimator,
    _determine_crop_region,
    _initial_crop_region,
    _map_keypoints_to_source,
)


class FakeInterpreter:
    def __init__(self, *, model_path: str, num_threads: int = 1, output=None, input_shape=None):
        self.model_path = model_path
        self.num_threads = num_threads
        self.output = output if output is not None else _pose_output()
        self.input_shape = input_shape or [1, 256, 256, 3]
        self.last_tensor = None
        self.invoked = False

    def allocate_tensors(self) -> None:
        return None

    def get_input_details(self):
        return [{"index": 3, "shape": np.array(self.input_shape), "dtype": np.uint8}]

    def get_output_details(self):
        return [{"index": 7, "shape": np.array([1, 1, 17, 3]), "dtype": np.float32}]

    def set_tensor(self, index, value) -> None:
        assert index == 3
        self.last_tensor = value

    def invoke(self) -> None:
        self.invoked = True

    def get_tensor(self, index):
        assert index == 7
        return self.output


def _pose_output() -> np.ndarray:
    output = np.zeros((1, 1, 17, 3), dtype=np.float32)
    # y, x, score. Build a simple full-body pose centered in the input crop.
    output[0, 0, 5] = [0.25, 0.42, 0.9]
    output[0, 0, 6] = [0.25, 0.58, 0.9]
    output[0, 0, 11] = [0.48, 0.45, 0.9]
    output[0, 0, 12] = [0.48, 0.55, 0.9]
    output[0, 0, 13] = [0.68, 0.45, 0.9]
    output[0, 0, 14] = [0.68, 0.55, 0.9]
    output[0, 0, 15] = [0.88, 0.45, 0.9]
    output[0, 0, 16] = [0.88, 0.55, 0.9]
    return output


def _dummy_model(tmp_path: Path) -> Path:
    model = tmp_path / "movenet.tflite"
    model.write_bytes(b"dummy")
    return model


def test_movenet_estimator_converts_output_to_pose_contract(tmp_path: Path) -> None:
    created: list[FakeInterpreter] = []

    def factory(**kwargs):
        interpreter = FakeInterpreter(**kwargs)
        created.append(interpreter)
        return interpreter

    estimator = MoveNetThunderPoseEstimator(
        _dummy_model(tmp_path),
        num_threads=4,
        smart_crop=False,
        interpreter_factory=factory,
    )
    estimator.open()
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    pose = estimator.estimate(frame, timestamp_ms=1234)

    assert pose is not None
    assert pose.timestamp_ms == 1234
    left = pose.leg(LegSide.LEFT)
    assert left is not None
    assert left.hip.x == pytest.approx(0.45)
    assert left.hip.y == pytest.approx(0.48)
    assert left.knee.x == pytest.approx(0.45)
    assert left.ankle.y == pytest.approx(0.88)
    assert left.confidence == pytest.approx(0.9)
    assert created[0].num_threads == 4
    assert created[0].last_tensor is not None
    assert created[0].last_tensor.shape == (1, 256, 256, 3)
    assert created[0].last_tensor.dtype == np.uint8
    assert created[0].invoked


def test_movenet_estimator_rejects_wrong_model_input_shape(tmp_path: Path) -> None:
    def factory(**kwargs):
        return FakeInterpreter(**kwargs, input_shape=[1, 192, 192, 3])

    estimator = MoveNetThunderPoseEstimator(
        _dummy_model(tmp_path),
        interpreter_factory=factory,
    )
    with pytest.raises(RuntimeError, match="input shape"):
        estimator.open()


def test_initial_crop_pads_landscape_frame_to_square() -> None:
    crop = _initial_crop_region(100, 200)
    assert crop.x_min == pytest.approx(0.0)
    assert crop.x_max == pytest.approx(1.0)
    assert crop.y_min == pytest.approx(-0.5)
    assert crop.y_max == pytest.approx(1.5)


def test_keypoints_map_back_from_crop_to_source_coordinates() -> None:
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[:, :2] = [0.25, 0.75]
    crop = CropRegion(y_min=0.1, x_min=0.2, y_max=0.9, x_max=0.8)
    mapped = _map_keypoints_to_source(keypoints, crop)
    assert mapped[0, 0] == pytest.approx(0.3)
    assert mapped[0, 1] == pytest.approx(0.65)


def test_smart_crop_returns_compact_region_for_visible_full_body() -> None:
    keypoints = np.squeeze(_pose_output())
    crop = _determine_crop_region(keypoints, image_height=720, image_width=1280)
    initial = _initial_crop_region(720, 1280)
    assert crop.height < initial.height
    assert crop.width < initial.width
    assert crop.y_min < 0.48 < crop.y_max
    assert crop.x_min < 0.50 < crop.x_max


def test_smart_crop_falls_back_when_torso_confidence_is_low() -> None:
    keypoints = np.squeeze(_pose_output())
    keypoints[[5, 6, 11, 12], 2] = 0.01
    crop = _determine_crop_region(keypoints, image_height=720, image_width=1280)
    assert crop == _initial_crop_region(720, 1280)


def test_tflite_file_check_rejects_html_and_accepts_tflite_header(tmp_path: Path) -> None:
    html = tmp_path / "bad.tflite"
    html.write_bytes(b"<html>" + b"x" * 10_000_000)
    assert not _looks_like_tflite(html)

    model = tmp_path / "good.tflite"
    model.write_bytes(b"\x1c\x00\x00\x00TFL3" + b"x" * 10_000_000)
    assert _looks_like_tflite(model)

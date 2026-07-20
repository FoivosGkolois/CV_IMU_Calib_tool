from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from .models import Landmark, PoseFrame


# MoveNet output order, defined by the official model card.
_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
_MOVENET_INDEX = {name: index for index, name in enumerate(_KEYPOINT_NAMES)}
_LEG_LANDMARK_NAMES = (
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

_MIN_CROP_KEYPOINT_SCORE = 0.20


class PoseEstimator(Protocol):
    """Implementation-independent pose backend used by the runtime."""

    @property
    def description(self) -> str: ...

    def open(self) -> None: ...

    def estimate(self, frame_bgr: Any, timestamp_ms: int) -> PoseFrame | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class CropRegion:
    """Square crop represented in normalized coordinates of the source frame."""

    y_min: float
    x_min: float
    y_max: float
    x_max: float

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def width(self) -> float:
        return self.x_max - self.x_min


class MoveNetThunderPoseEstimator:
    """MoveNet SinglePose Thunder FP16 backend using the LiteRT interpreter.

    The backend follows Google's video reference pipeline: the first frame uses a
    full-image square crop and subsequent frames use a crop inferred from the
    previous pose. The public output remains the existing normalized ``PoseFrame``
    contract, so geometry, overlay, recording, and future IMU integration do not
    depend on the model implementation.
    """

    INPUT_SIZE = 256

    def __init__(
        self,
        model_path: str | Path,
        *,
        num_threads: int = 4,
        smart_crop: bool = True,
        interpreter_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise RuntimeError(
                f"MoveNet model not found: {self.model_path}. "
                "Run 08_SETUP_VISION_IMU_PC.bat or download_model.py."
            )
        if int(num_threads) < 1:
            raise ValueError("num_threads must be at least 1")
        self.num_threads = int(num_threads)
        self.smart_crop = bool(smart_crop)
        self._interpreter_factory = interpreter_factory
        self._interpreter: Any | None = None
        self._input_detail: dict[str, Any] | None = None
        self._output_detail: dict[str, Any] | None = None
        self._crop_region: CropRegion | None = None
        self._frame_shape: tuple[int, int] | None = None

    @property
    def description(self) -> str:
        crop = "smart-crop" if self.smart_crop else "full-frame"
        return f"MoveNet SinglePose Thunder FP16 ({crop}, {self.num_threads} threads)"

    def open(self) -> None:
        factory = self._interpreter_factory or _load_litert_interpreter()
        try:
            interpreter = factory(model_path=str(self.model_path), num_threads=self.num_threads)
        except TypeError:
            # Some compatible interpreter wrappers do not expose num_threads.
            interpreter = factory(model_path=str(self.model_path))
        interpreter.allocate_tensors()

        inputs = interpreter.get_input_details()
        outputs = interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"Unexpected MoveNet tensor count: inputs={len(inputs)}, outputs={len(outputs)}"
            )

        input_detail = dict(inputs[0])
        output_detail = dict(outputs[0])
        input_shape = tuple(int(value) for value in input_detail.get("shape", ()))
        output_shape = tuple(int(value) for value in output_detail.get("shape", ()))
        if input_shape != (1, self.INPUT_SIZE, self.INPUT_SIZE, 3):
            raise RuntimeError(f"Unexpected MoveNet input shape: {input_shape}")
        if output_shape[-2:] != (17, 3):
            raise RuntimeError(f"Unexpected MoveNet output shape: {output_shape}")

        self._interpreter = interpreter
        self._input_detail = input_detail
        self._output_detail = output_detail
        self._crop_region = None
        self._frame_shape = None

    def estimate(self, frame_bgr: Any, timestamp_ms: int) -> PoseFrame | None:
        if self._interpreter is None or self._input_detail is None or self._output_detail is None:
            raise RuntimeError("Pose estimator is not open")
        if frame_bgr is None or getattr(frame_bgr, "ndim", 0) != 3:
            raise ValueError("frame_bgr must be an HxWx3 image")

        frame_height, frame_width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        if frame_height <= 0 or frame_width <= 0:
            raise ValueError("frame_bgr has invalid dimensions")
        shape = (frame_height, frame_width)
        if self._crop_region is None or self._frame_shape != shape:
            self._crop_region = _initial_crop_region(frame_height, frame_width)
            self._frame_shape = shape

        cv2 = _load_cv2()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        model_input = _crop_and_resize(
            frame_rgb,
            self._crop_region,
            output_size=self.INPUT_SIZE,
            cv2_module=cv2,
        )
        model_input = _convert_input_dtype(model_input, self._input_detail)

        self._interpreter.set_tensor(self._input_detail["index"], model_input[np.newaxis, ...])
        self._interpreter.invoke()
        raw_output = np.asarray(
            self._interpreter.get_tensor(self._output_detail["index"]),
            dtype=np.float32,
        )
        keypoints = _normalize_output_shape(raw_output)
        keypoints = _map_keypoints_to_source(keypoints, self._crop_region)

        if self.smart_crop:
            self._crop_region = _determine_crop_region(keypoints, frame_height, frame_width)
        else:
            self._crop_region = _initial_crop_region(frame_height, frame_width)

        landmarks = {
            name: Landmark(
                x=float(np.clip(keypoints[index, 1], 0.0, 1.0)),
                y=float(np.clip(keypoints[index, 0], 0.0, 1.0)),
                visibility=float(np.clip(keypoints[index, 2], 0.0, 1.0)),
                presence=float(np.clip(keypoints[index, 2], 0.0, 1.0)),
            )
            for name in _LEG_LANDMARK_NAMES
            for index in (_MOVENET_INDEX[name],)
        }
        return PoseFrame(timestamp_ms=int(timestamp_ms), landmarks=landmarks)

    def reset_tracking(self) -> None:
        """Reset smart-crop state after a frame-orientation change."""
        self._crop_region = None
        self._frame_shape = None

    def close(self) -> None:
        self._interpreter = None
        self._input_detail = None
        self._output_detail = None
        self._crop_region = None
        self._frame_shape = None


def _load_litert_interpreter() -> Callable[..., Any]:
    """Load a small cross-platform LiteRT interpreter implementation.

    ``ai-edge-litert`` is the preferred package for both Windows x86-64 and Linux
    ARM64. The fallbacks keep the estimator portable to existing TFLite images.
    """

    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter
    except ImportError:
        pass
    try:
        from tensorflow.lite import Interpreter

        return Interpreter
    except ImportError as exc:
        raise RuntimeError(
            "No LiteRT interpreter is installed. Re-run 08_SETUP_VISION_IMU_PC.bat."
        ) from exc


def _convert_input_dtype(image_rgb: np.ndarray, input_detail: dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(input_detail.get("dtype", np.uint8))
    if dtype == np.uint8:
        return image_rgb.astype(np.uint8, copy=False)
    if dtype == np.int32:
        return image_rgb.astype(np.int32)
    if dtype == np.float32:
        return image_rgb.astype(np.float32)
    raise RuntimeError(f"Unsupported MoveNet input dtype: {dtype}")


def _normalize_output_shape(output: np.ndarray) -> np.ndarray:
    squeezed = np.squeeze(output)
    if squeezed.shape != (17, 3):
        raise RuntimeError(f"Unexpected MoveNet output data shape: {tuple(output.shape)}")
    return squeezed.astype(np.float32, copy=True)


def _initial_crop_region(image_height: int, image_width: int) -> CropRegion:
    """Pad the full image to a square, matching Google's reference pipeline."""

    if image_width > image_height:
        box_height = image_width / image_height
        box_width = 1.0
        y_min = (image_height / 2.0 - image_width / 2.0) / image_height
        x_min = 0.0
    else:
        box_height = 1.0
        box_width = image_height / image_width
        y_min = 0.0
        x_min = (image_width / 2.0 - image_height / 2.0) / image_width
    return CropRegion(y_min, x_min, y_min + box_height, x_min + box_width)


def _crop_and_resize(
    image_rgb: np.ndarray,
    crop: CropRegion,
    *,
    output_size: int,
    cv2_module: Any,
) -> np.ndarray:
    """Crop with zero padding outside the frame and resize to a square tensor."""

    image_height, image_width = image_rgb.shape[:2]
    y0 = int(np.floor(crop.y_min * image_height))
    x0 = int(np.floor(crop.x_min * image_width))
    y1 = int(np.ceil(crop.y_max * image_height))
    x1 = int(np.ceil(crop.x_max * image_width))
    crop_height = max(1, y1 - y0)
    crop_width = max(1, x1 - x0)
    canvas = np.zeros((crop_height, crop_width, 3), dtype=image_rgb.dtype)

    source_y0 = max(0, y0)
    source_x0 = max(0, x0)
    source_y1 = min(image_height, y1)
    source_x1 = min(image_width, x1)
    if source_y1 > source_y0 and source_x1 > source_x0:
        dest_y0 = source_y0 - y0
        dest_x0 = source_x0 - x0
        dest_y1 = dest_y0 + (source_y1 - source_y0)
        dest_x1 = dest_x0 + (source_x1 - source_x0)
        canvas[dest_y0:dest_y1, dest_x0:dest_x1] = image_rgb[
            source_y0:source_y1,
            source_x0:source_x1,
        ]

    return cv2_module.resize(
        canvas,
        (int(output_size), int(output_size)),
        interpolation=cv2_module.INTER_LINEAR,
    )


def _map_keypoints_to_source(keypoints: np.ndarray, crop: CropRegion) -> np.ndarray:
    mapped = keypoints.copy()
    mapped[:, 0] = crop.y_min + crop.height * mapped[:, 0]
    mapped[:, 1] = crop.x_min + crop.width * mapped[:, 1]
    return mapped


def _torso_visible(keypoints: np.ndarray) -> bool:
    left_hip = keypoints[_MOVENET_INDEX["left_hip"], 2]
    right_hip = keypoints[_MOVENET_INDEX["right_hip"], 2]
    left_shoulder = keypoints[_MOVENET_INDEX["left_shoulder"], 2]
    right_shoulder = keypoints[_MOVENET_INDEX["right_shoulder"], 2]
    return (
        (left_hip > _MIN_CROP_KEYPOINT_SCORE or right_hip > _MIN_CROP_KEYPOINT_SCORE)
        and (
            left_shoulder > _MIN_CROP_KEYPOINT_SCORE
            or right_shoulder > _MIN_CROP_KEYPOINT_SCORE
        )
    )


def _determine_crop_region(
    keypoints: np.ndarray,
    image_height: int,
    image_width: int,
) -> CropRegion:
    """Derive the next-frame crop from the current full-body keypoints."""

    if not _torso_visible(keypoints):
        return _initial_crop_region(image_height, image_width)

    coordinates = {
        name: (
            float(keypoints[index, 0] * image_height),
            float(keypoints[index, 1] * image_width),
        )
        for name, index in _MOVENET_INDEX.items()
    }
    center_y = (coordinates["left_hip"][0] + coordinates["right_hip"][0]) / 2.0
    center_x = (coordinates["left_hip"][1] + coordinates["right_hip"][1]) / 2.0

    torso_names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    max_torso_y = max(abs(center_y - coordinates[name][0]) for name in torso_names)
    max_torso_x = max(abs(center_x - coordinates[name][1]) for name in torso_names)

    visible_coordinates = [
        coordinates[name]
        for name, index in _MOVENET_INDEX.items()
        if keypoints[index, 2] >= _MIN_CROP_KEYPOINT_SCORE
    ]
    if not visible_coordinates:
        return _initial_crop_region(image_height, image_width)
    max_body_y = max(abs(center_y - point[0]) for point in visible_coordinates)
    max_body_x = max(abs(center_x - point[1]) for point in visible_coordinates)

    crop_half = max(
        max_torso_x * 1.9,
        max_torso_y * 1.9,
        max_body_y * 1.2,
        max_body_x * 1.2,
    )
    crop_half = min(
        crop_half,
        max(center_x, image_width - center_x, center_y, image_height - center_y),
    )
    if crop_half <= 1.0 or crop_half > max(image_width, image_height) / 2.0:
        return _initial_crop_region(image_height, image_width)

    crop_length = crop_half * 2.0
    return CropRegion(
        y_min=(center_y - crop_half) / image_height,
        x_min=(center_x - crop_half) / image_width,
        y_max=(center_y + crop_half) / image_height,
        x_max=(center_x + crop_half) / image_width,
    )


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is not installed") from exc
    return cv2

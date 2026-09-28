from __future__ import annotations

from dataclasses import dataclass
from typing import Any


APRILTAG_DICTIONARY_NAME = "DICT_APRILTAG_36h11"
THIGH_TAG_ID = 0
SHANK_TAG_ID = 1


@dataclass(frozen=True)
class MarkerDetection:
    tag_id: int
    corners: tuple[tuple[float, float], ...]

    @property
    def center(self) -> tuple[float, float]:
        x = sum(point[0] for point in self.corners) / len(self.corners)
        y = sum(point[1] for point in self.corners) / len(self.corners)
        return float(x), float(y)


@dataclass(frozen=True)
class MarkerReferenceResult:
    """One frame of the tag-based geometric reference.

    Patch V2.1 intentionally stops at deterministic tag detection. The 3-D angle
    remains invalid until the persistent camera/reference setup is added in the
    next patch. This prevents an uncalibrated 2-D shortcut from silently becoming
    the scientific reference.
    """

    angle_deg: float | None
    valid: bool
    reason: str
    quality: float
    thigh: MarkerDetection | None
    shank: MarkerDetection | None

    @property
    def thigh_detected(self) -> bool:
        return self.thigh is not None

    @property
    def shank_detected(self) -> bool:
        return self.shank is not None


class AprilTagReference:
    """Detect the fixed thigh/shank AprilTags from one camera frame.

    Tag IDs are part of the measurement contract:
      0 -> thigh
      1 -> shank

    No user-tuned smoothing, confidence threshold, or learned pose model is used.
    3-D pose and the relative knee angle are deliberately gated until a calibrated
    camera/reference setup is available.
    """

    thigh_tag_id = THIGH_TAG_ID
    shank_tag_id = SHANK_TAG_ID

    def __init__(self) -> None:
        self._cv2: Any | None = None
        self._detector: Any | None = None
        self._dictionary: Any | None = None

    @property
    def description(self) -> str:
        return (
            f"AprilTag 36h11 geometric reference "
            f"(thigh={self.thigh_tag_id}, shank={self.shank_tag_id})"
        )

    def open(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is not installed") from exc
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "OpenCV AprilTag support is unavailable. Install opencv-contrib-python "
                "(or opencv-contrib-python-headless on UNO Q)."
            )
        dictionary_id = getattr(cv2.aruco, APRILTAG_DICTIONARY_NAME, None)
        if dictionary_id is None:
            raise RuntimeError(f"OpenCV does not provide {APRILTAG_DICTIONARY_NAME}")
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        parameters = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self._dictionary = dictionary
        self._cv2 = cv2

    def close(self) -> None:
        self._detector = None
        self._dictionary = None
        self._cv2 = None

    def process(self, frame_bgr: Any) -> MarkerReferenceResult:
        if self._detector is None or self._cv2 is None:
            raise RuntimeError("AprilTag reference is not open")
        gray = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = self._detector.detectMarkers(gray)
        detections: dict[int, MarkerDetection] = {}
        if ids is not None:
            for tag_id_raw, corner_array in zip(ids.flatten().tolist(), corners):
                tag_id = int(tag_id_raw)
                if tag_id not in {self.thigh_tag_id, self.shank_tag_id}:
                    continue
                points = corner_array.reshape(-1, 2)
                if points.shape != (4, 2):
                    continue
                detections[tag_id] = MarkerDetection(
                    tag_id=tag_id,
                    corners=tuple((float(x), float(y)) for x, y in points),
                )

        thigh = detections.get(self.thigh_tag_id)
        shank = detections.get(self.shank_tag_id)
        if thigh is None and shank is None:
            reason = "missing_both_tags"
        elif thigh is None:
            reason = "missing_thigh_tag"
        elif shank is None:
            reason = "missing_shank_tag"
        else:
            reason = "reference_setup_required"

        # V2.1 does not invent an angle from uncalibrated 2-D geometry.
        quality = 1.0 if thigh is not None and shank is not None else (0.5 if thigh or shank else 0.0)
        return MarkerReferenceResult(
            angle_deg=None,
            valid=False,
            reason=reason,
            quality=quality,
            thigh=thigh,
            shank=shank,
        )

    def generate_marker_image(self, tag_id: int, side_pixels: int = 1000) -> Any:
        """Generate one exact marker image for printing/tests."""
        if self._dictionary is None or self._cv2 is None:
            self.open()
        if tag_id < 0:
            raise ValueError("tag_id must be non-negative")
        side_pixels = int(side_pixels)
        if side_pixels < 64:
            raise ValueError("side_pixels must be at least 64")
        return self._cv2.aruco.generateImageMarker(self._dictionary, int(tag_id), side_pixels)

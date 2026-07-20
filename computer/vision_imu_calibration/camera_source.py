from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any


class OpenCvCameraSource:
    """Small OpenCV camera/video adapter used by the PC feasibility prototype."""

    def __init__(
        self,
        *,
        camera_index: int | None = 0,
        video_path: str | Path | None = None,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        mirror: bool = True,
        backend: str = "auto",
        cv2_module: Any | None = None,
        platform_name: str | None = None,
    ) -> None:
        if (camera_index is None) == (video_path is None):
            raise ValueError("Provide exactly one of camera_index or video_path")
        self.camera_index = camera_index
        self.video_path = Path(video_path).expanduser().resolve() if video_path is not None else None
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.mirror = bool(mirror)
        self.backend = self._normalize_backend(backend)
        self._cv2 = cv2_module
        self._platform_name = platform_name or platform.system()
        self._capture: Any | None = None
        self._prefetched_frame: Any | None = None
        self._active_backend_name: str | None = None

    @property
    def description(self) -> str:
        if self.video_path is not None:
            return str(self.video_path)
        backend_suffix = f" [{self._active_backend_name}]" if self._active_backend_name else ""
        return f"camera:{self.camera_index}{backend_suffix}"

    def open(self) -> None:
        cv2 = self._load_cv2()
        self.close()

        if self.video_path is not None:
            if not self.video_path.is_file():
                raise RuntimeError(f"Video file does not exist: {self.video_path}")
            capture = cv2.VideoCapture(str(self.video_path))
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Could not open {self.description}")
            self._capture = capture
            self._active_backend_name = "FILE"
            return

        failures: list[str] = []
        for backend_name, backend_id in self._camera_backend_candidates(cv2):
            capture = self._create_camera_capture(cv2, int(self.camera_index), backend_id)
            if not capture.isOpened():
                capture.release()
                failures.append(f"{backend_name}: open failed")
                continue

            self._configure_camera(cv2, capture)
            frame = self._read_initial_frame(capture)
            if frame is None:
                capture.release()
                failures.append(f"{backend_name}: opened but returned no frame")
                continue

            self._capture = capture
            self._prefetched_frame = frame
            self._active_backend_name = backend_name
            return

        attempted = "; ".join(failures) if failures else "no OpenCV backend candidates"
        raise RuntimeError(
            f"Could not open camera:{self.camera_index}. Tried {attempted}. "
            "Close other camera applications, verify Windows camera privacy permissions, "
            "or try another index with --camera 1."
        )

    def read(self) -> Any | None:
        if self._capture is None:
            raise RuntimeError("Camera source is not open")

        frame, self._prefetched_frame = self._prefetched_frame, None
        if frame is None:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                if self.video_path is not None:
                    self._capture.set(self._load_cv2().CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self._capture.read()
                if not ok or frame is None:
                    return None

        if self.mirror:
            frame = self._load_cv2().flip(frame, 1)
        return frame

    def close(self) -> None:
        capture, self._capture = self._capture, None
        self._prefetched_frame = None
        self._active_backend_name = None
        if capture is not None:
            capture.release()

    def _camera_backend_candidates(self, cv2: Any) -> list[tuple[str, int | None]]:
        if self._platform_name != "Windows":
            return [("DEFAULT", None)]

        available: dict[str, int | None] = {"DEFAULT": None}
        for name, attribute in (("DSHOW", "CAP_DSHOW"), ("MSMF", "CAP_MSMF")):
            backend_id = getattr(cv2, attribute, None)
            if backend_id is not None:
                available[name] = int(backend_id)

        if self.backend != "AUTO":
            if self.backend not in available:
                raise RuntimeError(
                    f"Requested camera backend {self.backend} is unavailable in this OpenCV build"
                )
            return [(self.backend, available[self.backend])]

        # DirectShow is tried first on Windows because it usually opens USB webcams
        # faster and more deterministically than Media Foundation. MSMF and the
        # OpenCV default remain as compatibility fallbacks.
        order = ("DSHOW", "MSMF", "DEFAULT")
        return [(name, available[name]) for name in order if name in available]

    @staticmethod
    def _normalize_backend(backend: str) -> str:
        normalized = str(backend).strip().upper()
        aliases = {
            "AUTO": "AUTO",
            "DSHOW": "DSHOW",
            "DIRECTSHOW": "DSHOW",
            "MSMF": "MSMF",
            "MEDIAFOUNDATION": "MSMF",
            "DEFAULT": "DEFAULT",
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            allowed = "auto, dshow, msmf, default"
            raise ValueError(f"Unsupported camera backend {backend!r}; choose {allowed}") from exc

    @staticmethod
    def _create_camera_capture(cv2: Any, camera_index: int, backend_id: int | None) -> Any:
        if backend_id is None:
            return cv2.VideoCapture(camera_index)
        return cv2.VideoCapture(camera_index, backend_id)

    def _configure_camera(self, cv2: Any, capture: Any) -> None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    @staticmethod
    def _read_initial_frame(capture: Any, attempts: int = 6, delay_s: float = 0.03) -> Any | None:
        for attempt in range(attempts):
            ok, frame = capture.read()
            if ok and frame is not None:
                return frame
            if attempt + 1 < attempts:
                time.sleep(delay_s)
        return None

    def _load_cv2(self) -> Any:
        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("OpenCV is not installed. Run 08_SETUP_VISION_IMU_PC.bat") from exc
            self._cv2 = cv2
        return self._cv2

from __future__ import annotations

import math
import queue
import threading
import time
from typing import Any

from .imu_serial import IMUCAL_SCHEMA_VERSION, ImuCalibrationSample

BRIDGE_SAMPLE_METHOD = "vision_imu_sample"
BRIDGE_START_CAPTURE_METHOD = "vision_imu_start_capture"
BRIDGE_STOP_CAPTURE_METHOD = "vision_imu_stop_capture"
BRIDGE_STATUS_METHOD = "vision_imu_status_flags"

STATUS_THIGH_READY = 1 << 0
STATUS_SHANK_READY = 1 << 1
STATUS_CAPTURE_ACTIVE = 1 << 2
STATUS_BRIDGE_READY = 1 << 3


class BridgeImuSource:
    """UNO Q Linux source using Arduino Bridge instead of USB serial.

    The public methods intentionally mirror ``SerialImuSource`` so the camera,
    estimator, recorder, calibration workflow, and web GUI can remain transport-neutral.
    """

    transport = "uno-q-bridge"
    replaceable_by_serial = False

    def __init__(
        self,
        *,
        queue_size: int = 2048,
        bridge_api: Any | None = None,
        clock_ns: Any = time.perf_counter_ns,
    ) -> None:
        self.port = "UNO_Q_BRIDGE"
        self.baud_rate = 0
        self.queue_size = max(32, int(queue_size))
        self._bridge_api = bridge_api
        self._clock_ns = clock_ns
        self._queue: queue.Queue[ImuCalibrationSample] = queue.Queue(maxsize=self.queue_size)
        self._lock = threading.RLock()
        self._connected_event = threading.Event()
        self._started = False
        self._capture_active = False
        self._state = "IDLE"
        self._sample_rate_hz = 0.0
        self._sample_count = 0
        self._parse_errors = 0
        self._sequence_gaps = 0
        self._queue_drops = 0
        self._thigh_ready = False
        self._shank_ready = False
        self._last_sample_index: int | None = None
        self._last_sample_host_ns: int | None = None
        self._last_error = ""

    @property
    def configured(self) -> bool:
        return True

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._state = "CONNECTING"
            self._last_error = ""
        try:
            bridge = self._bridge()
            bridge.provide(BRIDGE_SAMPLE_METHOD, self._handle_sample_notification)
            with self._lock:
                self._started = True
            self.request_status(timeout_s=3.0)
        except Exception as exc:
            with self._lock:
                self._state = "FAILED"
                self._last_error = str(exc)
                self._started = False
            self._connected_event.clear()
            try:
                self._bridge().unprovide(BRIDGE_SAMPLE_METHOD)
            except Exception:
                pass
            raise RuntimeError(f"UNO Q Bridge startup failed: {exc}") from exc

    def stop(self, timeout_s: float = 2.0) -> None:
        if self._capture_active:
            try:
                self.stop_capture(timeout_s=min(1.0, max(0.1, float(timeout_s))))
            except Exception:
                pass
        if self._started:
            try:
                self._bridge().unprovide(BRIDGE_SAMPLE_METHOD)
            except Exception:
                pass
        with self._lock:
            self._started = False
            self._capture_active = False
            self._state = "STOPPED"
        self._connected_event.clear()

    def wait_until_connected(self, timeout_s: float = 3.0) -> bool:
        return self._connected_event.wait(max(0.0, float(timeout_s)))

    def start_capture(self, timeout_s: float = 1.5) -> None:
        flags = self._call(BRIDGE_START_CAPTURE_METHOD, timeout_s=timeout_s)
        self._apply_status_flags(flags)
        if not self._capture_active:
            raise RuntimeError("UNO Q MCU rejected capture because one or both IMUs are not ready")
        with self._lock:
            self._state = "CAPTURING"
            self._last_sample_index = None
            self._last_sample_host_ns = None
            self._sample_rate_hz = 0.0
        self._clear_queue()

    def stop_capture(self, timeout_s: float = 1.5) -> None:
        flags = self._call(BRIDGE_STOP_CAPTURE_METHOD, timeout_s=timeout_s)
        self._apply_status_flags(flags)
        with self._lock:
            self._capture_active = False
            self._state = "CONNECTED" if self._connected_event.is_set() else "IDLE"

    def request_status(self, timeout_s: float = 1.0) -> None:
        flags = self._call(BRIDGE_STATUS_METHOD, timeout_s=timeout_s)
        self._apply_status_flags(flags)

    def get_sample(self, timeout_s: float = 0.1) -> ImuCalibrationSample | None:
        try:
            return self._queue.get(timeout=max(0.0, float(timeout_s)))
        except queue.Empty:
            return None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "configured": True,
                "connected": self._connected_event.is_set(),
                "capture_active": self._capture_active,
                "port": self.port,
                "baud_rate": self.baud_rate,
                "transport": self.transport,
                "state": self._state,
                "sample_rate_hz": self._sample_rate_hz,
                "sample_count": self._sample_count,
                "parse_errors": self._parse_errors,
                "sequence_gaps": self._sequence_gaps,
                "queue_drops": self._queue_drops,
                "thigh_ready": self._thigh_ready,
                "shank_ready": self._shank_ready,
                "last_sample_index": self._last_sample_index,
                "last_error": self._last_error,
            }

    def _bridge(self) -> Any:
        if self._bridge_api is not None:
            return self._bridge_api
        try:
            from arduino.app_utils import Bridge
        except ImportError as exc:
            raise RuntimeError(
                "arduino.app_utils is unavailable. Run Phase 2 from Arduino App Lab on UNO Q."
            ) from exc
        self._bridge_api = Bridge
        return Bridge

    def _call(self, method: str, *, timeout_s: float) -> int:
        try:
            result = self._bridge().call(
                method,
                timeout=max(1, int(math.ceil(max(0.1, float(timeout_s))))),
            )
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
                self._state = "FAILED"
            self._connected_event.clear()
            raise RuntimeError(f"UNO Q Bridge call failed ({method}): {exc}") from exc
        try:
            return int(result)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"UNO Q Bridge returned invalid status flags for {method}: {result!r}") from exc

    def _apply_status_flags(self, flags: int) -> None:
        value = int(flags)
        with self._lock:
            self._thigh_ready = bool(value & STATUS_THIGH_READY)
            self._shank_ready = bool(value & STATUS_SHANK_READY)
            self._capture_active = bool(value & STATUS_CAPTURE_ACTIVE)
            bridge_ready = bool(value & STATUS_BRIDGE_READY)
            if bridge_ready:
                self._last_error = ""
                self._state = "CAPTURING" if self._capture_active else "CONNECTED"
                self._connected_event.set()
            else:
                self._state = "WAITING_FOR_MCU"
                self._connected_event.clear()

    def _handle_sample_notification(self, *values: Any) -> None:
        if len(values) == 1 and isinstance(values[0], (list, tuple)):
            values = tuple(values[0])
        if len(values) != 17:
            with self._lock:
                self._parse_errors += 1
            return
        try:
            fields = [int(value) for value in values]
        except (TypeError, ValueError):
            with self._lock:
                self._parse_errors += 1
            return
        if fields[0] != IMUCAL_SCHEMA_VERSION:
            with self._lock:
                self._parse_errors += 1
            return

        sample = ImuCalibrationSample(
            host_monotonic_ns=int(self._clock_ns()),
            schema_version=fields[0],
            sample_index=fields[1],
            mcu_timestamp_us=fields[2],
            thigh_ok=bool(fields[3]),
            shank_ok=bool(fields[4]),
            thigh_ax_raw=fields[5],
            thigh_ay_raw=fields[6],
            thigh_az_raw=fields[7],
            thigh_gx_raw=fields[8],
            thigh_gy_raw=fields[9],
            thigh_gz_raw=fields[10],
            shank_ax_raw=fields[11],
            shank_ay_raw=fields[12],
            shank_az_raw=fields[13],
            shank_gx_raw=fields[14],
            shank_gy_raw=fields[15],
            shank_gz_raw=fields[16],
        )
        self._accept_sample(sample)

    def _accept_sample(self, sample: ImuCalibrationSample) -> None:
        with self._lock:
            self._connected_event.set()
            self._capture_active = True
            self._state = "CAPTURING"
            self._thigh_ready = sample.thigh_ok
            self._shank_ready = sample.shank_ok
            if self._last_sample_index is not None:
                delta = (sample.sample_index - self._last_sample_index) & 0xFFFFFFFF
                if delta > 1:
                    self._sequence_gaps += delta - 1
            if self._last_sample_host_ns is not None:
                dt_s = (sample.host_monotonic_ns - self._last_sample_host_ns) / 1_000_000_000.0
                if 0.001 <= dt_s <= 1.0:
                    instant = 1.0 / dt_s
                    self._sample_rate_hz = (
                        instant
                        if self._sample_rate_hz <= 0.0
                        else 0.9 * self._sample_rate_hz + 0.1 * instant
                    )
            self._last_sample_index = sample.sample_index
            self._last_sample_host_ns = sample.host_monotonic_ns
            self._sample_count += 1

        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            with self._lock:
                self._queue_drops += 1
            try:
                self._queue.put_nowait(sample)
            except queue.Full:
                with self._lock:
                    self._queue_drops += 1

    def _clear_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

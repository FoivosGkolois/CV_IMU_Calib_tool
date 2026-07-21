from __future__ import annotations

import queue
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

ACCEL_G_PER_LSB = 0.000061
GYRO_DPS_PER_LSB = 0.00875
IMUCAL_SCHEMA_VERSION = 1
IMUCAL_BAUD_RATE = 921600


@dataclass(frozen=True)
class ImuCalibrationSample:
    host_monotonic_ns: int
    schema_version: int
    sample_index: int
    mcu_timestamp_us: int
    thigh_ok: bool
    shank_ok: bool
    thigh_ax_raw: int
    thigh_ay_raw: int
    thigh_az_raw: int
    thigh_gx_raw: int
    thigh_gy_raw: int
    thigh_gz_raw: int
    shank_ax_raw: int
    shank_ay_raw: int
    shank_az_raw: int
    shank_gx_raw: int
    shank_gy_raw: int
    shank_gz_raw: int

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.update(
            {
                "thigh_ax_g": self.thigh_ax_raw * ACCEL_G_PER_LSB,
                "thigh_ay_g": self.thigh_ay_raw * ACCEL_G_PER_LSB,
                "thigh_az_g": self.thigh_az_raw * ACCEL_G_PER_LSB,
                "thigh_gx_dps": self.thigh_gx_raw * GYRO_DPS_PER_LSB,
                "thigh_gy_dps": self.thigh_gy_raw * GYRO_DPS_PER_LSB,
                "thigh_gz_dps": self.thigh_gz_raw * GYRO_DPS_PER_LSB,
                "shank_ax_g": self.shank_ax_raw * ACCEL_G_PER_LSB,
                "shank_ay_g": self.shank_ay_raw * ACCEL_G_PER_LSB,
                "shank_az_g": self.shank_az_raw * ACCEL_G_PER_LSB,
                "shank_gx_dps": self.shank_gx_raw * GYRO_DPS_PER_LSB,
                "shank_gy_dps": self.shank_gy_raw * GYRO_DPS_PER_LSB,
                "shank_gz_dps": self.shank_gz_raw * GYRO_DPS_PER_LSB,
            }
        )
        return record


@dataclass(frozen=True)
class ImuSerialStatus:
    configured: bool
    connected: bool
    capture_active: bool
    port: str
    baud_rate: int
    state: str
    sample_rate_hz: float
    sample_count: int
    parse_errors: int
    sequence_gaps: int
    queue_drops: int
    thigh_ready: bool
    shank_ready: bool
    last_sample_index: int | None
    last_error: str


class ImuSerialProtocol:
    """Fixed CSV protocol used only by the standalone calibration sketch."""

    FRAME_PREFIX = "IMUCAL"
    EXPECTED_FIELD_COUNT = 18

    @classmethod
    def parse_sample(cls, line: str, *, host_monotonic_ns: int) -> ImuCalibrationSample:
        tokens = [token.strip() for token in line.strip().split(",")]
        if len(tokens) != cls.EXPECTED_FIELD_COUNT:
            raise ValueError(
                f"IMUCAL field count {len(tokens)} != {cls.EXPECTED_FIELD_COUNT}"
            )
        if tokens[0] != cls.FRAME_PREFIX:
            raise ValueError(f"Unexpected frame prefix: {tokens[0]!r}")

        try:
            values = [int(token, 10) for token in tokens[1:]]
        except ValueError as exc:
            raise ValueError("IMUCAL contains a non-integer field") from exc

        schema_version = values[0]
        if schema_version != IMUCAL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported IMUCAL schema {schema_version}; expected {IMUCAL_SCHEMA_VERSION}"
            )

        return ImuCalibrationSample(
            host_monotonic_ns=int(host_monotonic_ns),
            schema_version=schema_version,
            sample_index=values[1],
            mcu_timestamp_us=values[2],
            thigh_ok=bool(values[3]),
            shank_ok=bool(values[4]),
            thigh_ax_raw=values[5],
            thigh_ay_raw=values[6],
            thigh_az_raw=values[7],
            thigh_gx_raw=values[8],
            thigh_gy_raw=values[9],
            thigh_gz_raw=values[10],
            shank_ax_raw=values[11],
            shank_ay_raw=values[12],
            shank_az_raw=values[13],
            shank_gx_raw=values[14],
            shank_gy_raw=values[15],
            shank_gz_raw=values[16],
        )


def list_serial_ports(list_ports_module: Any | None = None) -> list[dict[str, str]]:
    """Return user-facing serial ports without opening them."""
    module = list_ports_module or _load_list_ports_module()
    ports: list[dict[str, str]] = []
    for item in module.comports():
        device = str(getattr(item, "device", "") or "").strip()
        if not device:
            continue
        ports.append(
            {
                "device": device,
                "description": str(getattr(item, "description", "") or ""),
                "hwid": str(getattr(item, "hwid", "") or ""),
            }
        )
    return sorted(ports, key=lambda entry: entry["device"].lower())


class SerialImuSource:
    """Background pyserial source with bounded buffering and command ACKs."""

    transport = "serial"
    replaceable_by_serial = True

    def __init__(
        self,
        port: str | None,
        *,
        baud_rate: int = IMUCAL_BAUD_RATE,
        queue_size: int = 2048,
        serial_factory: Any | None = None,
    ) -> None:
        self.port = str(port or "").strip()
        self.baud_rate = int(baud_rate)
        self.queue_size = max(32, int(queue_size))
        self._serial_factory = serial_factory
        self._queue: queue.Queue[ImuCalibrationSample] = queue.Queue(maxsize=self.queue_size)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._serial: Any | None = None
        self._capture_active = False
        self._state = "DISABLED" if not self.port else "IDLE"
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
        self._next_command_id = 1
        self._pending: dict[int, tuple[threading.Event, dict[str, str]]] = {}

    @property
    def configured(self) -> bool:
        return bool(self.port)

    def start(self) -> None:
        if not self.configured:
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._state = "CONNECTING"
            self._thread = threading.Thread(target=self._run, name="vision-imu-serial", daemon=True)
            self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        if self._capture_active:
            try:
                self.stop_capture(timeout_s=0.5)
            except Exception:
                pass
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout_s)))
        with self._lock:
            serial_obj, self._serial = self._serial, None
            self._capture_active = False
            self._connected_event.clear()
            if self.configured and self._state != "FAILED":
                self._state = "STOPPED"
        if serial_obj is not None:
            try:
                serial_obj.close()
            except Exception:
                pass

    def wait_until_connected(self, timeout_s: float = 3.0) -> bool:
        return self._connected_event.wait(max(0.0, float(timeout_s)))

    def start_capture(self, timeout_s: float = 1.5) -> None:
        self._send_command("START_CAPTURE", timeout_s=timeout_s)
        with self._lock:
            self._capture_active = True
            self._state = "CAPTURING"
            self._last_sample_index = None
            self._last_sample_host_ns = None
            self._sample_rate_hz = 0.0

    def stop_capture(self, timeout_s: float = 1.5) -> None:
        self._send_command("STOP_CAPTURE", timeout_s=timeout_s)
        with self._lock:
            self._capture_active = False
            if self._connected_event.is_set():
                self._state = "CONNECTED"

    def request_status(self, timeout_s: float = 1.0) -> None:
        self._send_command("STATUS", timeout_s=timeout_s)

    def get_sample(self, timeout_s: float = 0.1) -> ImuCalibrationSample | None:
        try:
            return self._queue.get(timeout=max(0.0, float(timeout_s)))
        except queue.Empty:
            return None

    def status(self) -> dict[str, Any]:
        with self._lock:
            payload = asdict(
                ImuSerialStatus(
                    configured=self.configured,
                    connected=self._connected_event.is_set(),
                    capture_active=self._capture_active,
                    port=self.port,
                    baud_rate=self.baud_rate,
                    state=self._state,
                    sample_rate_hz=self._sample_rate_hz,
                    sample_count=self._sample_count,
                    parse_errors=self._parse_errors,
                    sequence_gaps=self._sequence_gaps,
                    queue_drops=self._queue_drops,
                    thigh_ready=self._thigh_ready,
                    shank_ready=self._shank_ready,
                    last_sample_index=self._last_sample_index,
                    last_error=self._last_error,
                )
            )
        payload["transport"] = self.transport
        return payload

    def _run(self) -> None:
        try:
            factory = self._serial_factory or _load_serial_factory()
            serial_obj = factory(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=0.1,
                write_timeout=0.5,
            )
            with self._lock:
                self._serial = serial_obj
                self._state = "WAITING_FOR_DEVICE"
                self._last_error = ""
            try:
                serial_obj.reset_input_buffer()
            except Exception:
                pass

            next_probe = 0.0
            while not self._stop_event.is_set():
                now = time.monotonic()
                if not self._connected_event.is_set() and now >= next_probe:
                    try:
                        serial_obj.write(b"CMD,0,STATUS\n")
                        serial_obj.flush()
                    except Exception:
                        pass
                    next_probe = now + 1.0
                raw = serial_obj.readline()
                if not raw:
                    continue
                host_ns = time.perf_counter_ns()
                try:
                    line = raw.decode("ascii", errors="strict").strip()
                except UnicodeDecodeError:
                    with self._lock:
                        self._parse_errors += 1
                    continue
                if not line:
                    continue
                self._handle_line(line, host_ns)
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
                self._state = "FAILED"
                self._capture_active = False
            self._connected_event.clear()
        finally:
            with self._lock:
                serial_obj, self._serial = self._serial, None
                self._capture_active = False
            if serial_obj is not None:
                try:
                    serial_obj.close()
                except Exception:
                    pass
            self._connected_event.clear()

    def _handle_line(self, line: str, host_ns: int) -> None:
        if line.startswith(("IMUCAL,", "INFO,", "ACK,", "NACK,")):
            self._mark_device_connected()
        if line.startswith("IMUCAL,"):
            try:
                sample = ImuSerialProtocol.parse_sample(line, host_monotonic_ns=host_ns)
            except ValueError:
                with self._lock:
                    self._parse_errors += 1
                return
            self._accept_sample(sample)
            return

        tokens = [token.strip() for token in line.split(",")]
        if len(tokens) >= 3 and tokens[0] in {"ACK", "NACK"}:
            try:
                command_id = int(tokens[1], 10)
            except ValueError:
                return
            with self._lock:
                pending = self._pending.get(command_id)
                if pending is not None:
                    event, result = pending
                    result["kind"] = tokens[0]
                    result["command"] = tokens[2]
                    result["detail"] = ",".join(tokens[3:])
                    event.set()
            return

        if line.startswith("INFO,STATUS"):
            fields = {}
            for token in tokens[2:]:
                if "=" in token:
                    key, value = token.split("=", 1)
                    fields[key] = value
            with self._lock:
                if "thigh_ready" in fields:
                    self._thigh_ready = fields["thigh_ready"] == "1"
                if "shank_ready" in fields:
                    self._shank_ready = fields["shank_ready"] == "1"

    def _mark_device_connected(self) -> None:
        with self._lock:
            if not self._connected_event.is_set():
                self._state = "CAPTURING" if self._capture_active else "CONNECTED"
                self._last_error = ""
                self._connected_event.set()

    def _accept_sample(self, sample: ImuCalibrationSample) -> None:
        with self._lock:
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
            self._thigh_ready = sample.thigh_ok
            self._shank_ready = sample.shank_ok

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

    def _send_command(self, command: str, *, timeout_s: float) -> None:
        if not self.configured:
            raise RuntimeError("No IMU serial port configured")
        if not self.wait_until_connected(timeout_s=min(3.0, max(0.1, timeout_s))):
            raise RuntimeError(f"IMU serial port is not connected: {self.port}")

        with self._lock:
            serial_obj = self._serial
            if serial_obj is None:
                raise RuntimeError("IMU serial connection is unavailable")
            command_id = self._next_command_id
            self._next_command_id += 1
            event = threading.Event()
            result: dict[str, str] = {}
            self._pending[command_id] = (event, result)
            payload = f"CMD,{command_id},{command}\n".encode("ascii")
            serial_obj.write(payload)
            serial_obj.flush()

        if not event.wait(max(0.1, float(timeout_s))):
            with self._lock:
                self._pending.pop(command_id, None)
            raise RuntimeError(f"Timeout waiting for {command} ACK")

        with self._lock:
            self._pending.pop(command_id, None)
        if result.get("kind") != "ACK":
            detail = result.get("detail", "")
            raise RuntimeError(f"{command} rejected: {detail or 'NACK'}")


def _load_list_ports_module() -> Any:
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError("pyserial is not installed. Re-run 08_SETUP_VISION_IMU_PC.bat") from exc
    return list_ports


def _load_serial_factory() -> Any:
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial is not installed. Re-run 08_SETUP_VISION_IMU_PC.bat") from exc
    return serial.Serial

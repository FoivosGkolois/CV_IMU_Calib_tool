from __future__ import annotations

import pytest

import computer.vision_imu_calibration.imu_bridge as imu_bridge_module
from computer.vision_imu_calibration.imu_bridge import (
    BRIDGE_SAMPLE_METHOD,
    BRIDGE_START_CAPTURE_METHOD,
    BRIDGE_STATUS_METHOD,
    BRIDGE_STOP_CAPTURE_METHOD,
    STATUS_BRIDGE_READY,
    STATUS_CAPTURE_ACTIVE,
    STATUS_SHANK_READY,
    STATUS_THIGH_READY,
    BridgeImuSource,
)


class FakeBridge:
    def __init__(self) -> None:
        self.handlers = {}
        self.calls: list[tuple[str, int]] = []
        self.flags = STATUS_BRIDGE_READY | STATUS_THIGH_READY | STATUS_SHANK_READY

    def provide(self, name, handler) -> None:
        self.handlers[name] = handler

    def unprovide(self, name) -> None:
        self.handlers.pop(name, None)

    def call(self, name, *, timeout: int):
        self.calls.append((name, timeout))
        if name == BRIDGE_START_CAPTURE_METHOD:
            self.flags |= STATUS_CAPTURE_ACTIVE
        elif name == BRIDGE_STOP_CAPTURE_METHOD:
            self.flags &= ~STATUS_CAPTURE_ACTIVE
        return self.flags


def sample_values(index: int = 42, timestamp_us: int = 123456) -> tuple[int, ...]:
    return (
        1,
        index,
        timestamp_us,
        1,
        1,
        100,
        -200,
        300,
        10,
        -20,
        30,
        400,
        -500,
        600,
        40,
        -50,
        60,
    )


def test_bridge_source_registers_handler_and_reports_ready() -> None:
    bridge = FakeBridge()
    source = BridgeImuSource(bridge_api=bridge)

    source.start()

    assert BRIDGE_SAMPLE_METHOD in bridge.handlers
    assert bridge.calls == [(BRIDGE_STATUS_METHOD, 3)]
    status = source.status()
    assert status["connected"]
    assert status["transport"] == "uno-q-bridge"
    assert status["thigh_ready"] and status["shank_ready"]


def test_bridge_source_maps_notification_to_existing_sample_contract() -> None:
    bridge = FakeBridge()
    source = BridgeImuSource(bridge_api=bridge, clock_ns=lambda: 999)
    source.start()
    bridge.handlers[BRIDGE_SAMPLE_METHOD](*sample_values())

    sample = source.get_sample(timeout_s=0.0)

    assert sample is not None
    assert sample.host_monotonic_ns == 999
    assert sample.sample_index == 42
    assert sample.thigh_ay_raw == -200
    assert sample.shank_gz_raw == 60
    assert source.status()["sample_count"] == 1


def test_bridge_source_capture_calls_mcu_and_preserves_interface() -> None:
    bridge = FakeBridge()
    source = BridgeImuSource(bridge_api=bridge)
    source.start()

    source.start_capture(timeout_s=1.2)
    assert source.status()["capture_active"]
    source.stop_capture(timeout_s=1.2)
    assert not source.status()["capture_active"]
    assert [name for name, _ in bridge.calls] == [
        BRIDGE_STATUS_METHOD,
        BRIDGE_START_CAPTURE_METHOD,
        BRIDGE_STOP_CAPTURE_METHOD,
    ]


def test_bridge_source_counts_gaps_and_invalid_payloads() -> None:
    bridge = FakeBridge()
    times = iter((1_000_000_000, 1_010_000_000))
    source = BridgeImuSource(bridge_api=bridge, clock_ns=lambda: next(times))
    source.start()
    handler = bridge.handlers[BRIDGE_SAMPLE_METHOD]

    handler(*sample_values(index=10))
    handler(*sample_values(index=13, timestamp_us=153456))
    handler(1, 2, 3)
    handler(*((2,) + sample_values()[1:]))

    status = source.status()
    assert status["sequence_gaps"] == 2
    assert status["parse_errors"] == 2
    assert status["sample_rate_hz"] == pytest.approx(100.0)


def test_bridge_import_failure_is_clear(monkeypatch) -> None:
    source = BridgeImuSource()

    import builtins

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "arduino.app_utils":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    with pytest.raises(RuntimeError, match="Arduino App Lab"):
        source.start()


class DelayedStatusBridge(FakeBridge):
    def __init__(self, unavailable_calls: int) -> None:
        super().__init__()
        self.unavailable_calls = unavailable_calls

    def call(self, name, *, timeout: int):
        self.calls.append((name, timeout))
        if name == BRIDGE_STATUS_METHOD and self.unavailable_calls > 0:
            self.unavailable_calls -= 1
            raise ValueError(
                f"Request '{name}' failed: method {name} not available (2)"
            )
        if name == BRIDGE_START_CAPTURE_METHOD:
            self.flags |= STATUS_CAPTURE_ACTIVE
        elif name == BRIDGE_STOP_CAPTURE_METHOD:
            self.flags &= ~STATUS_CAPTURE_ACTIVE
        return self.flags


def test_bridge_start_retries_temporary_method_unavailable(monkeypatch) -> None:
    bridge = DelayedStatusBridge(unavailable_calls=2)
    source = BridgeImuSource(bridge_api=bridge)
    monkeypatch.setattr(imu_bridge_module.time, "sleep", lambda _seconds: None)

    source.start()

    assert [name for name, _ in bridge.calls] == [
        BRIDGE_STATUS_METHOD,
        BRIDGE_STATUS_METHOD,
        BRIDGE_STATUS_METHOD,
    ]
    assert source.status()["connected"]
    assert source.status()["state"] == "CONNECTED"


def test_bridge_start_does_not_retry_unrelated_failure(monkeypatch) -> None:
    class BrokenBridge(FakeBridge):
        def call(self, name, *, timeout: int):
            raise ValueError("transport unavailable")

    source = BridgeImuSource(bridge_api=BrokenBridge())
    monkeypatch.setattr(imu_bridge_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="UNO Q Bridge startup failed"):
        source.start()

    assert source.status()["state"] == "FAILED"


def test_bridge_start_timeout_is_bounded(monkeypatch) -> None:
    bridge = DelayedStatusBridge(unavailable_calls=100)
    source = BridgeImuSource(bridge_api=bridge)
    monkeypatch.setattr(imu_bridge_module, "BRIDGE_STARTUP_WAIT_S", 0.0)
    monkeypatch.setattr(imu_bridge_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="no MCU route named"):
        source.start()

    assert source.status()["state"] == "FAILED"

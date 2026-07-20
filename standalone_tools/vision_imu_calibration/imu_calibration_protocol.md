# IMUCAL protocol v1

Transport: USB serial, `921600 baud`, ASCII CSV, one message per line.

## Commands

```text
CMD,<command_id>,HELLO
CMD,<command_id>,STATUS
CMD,<command_id>,START_CAPTURE
CMD,<command_id>,STOP_CAPTURE
```

Responses:

```text
ACK,<command_id>,<command>
NACK,<command_id>,<command>,<reason>
INFO,STATUS,thigh_ready=<0|1>,shank_ready=<0|1>,capture_active=<0|1>,sample_rate_hz=100,schema=1
```

## Sample frame

```text
IMUCAL,
1,
sample_index,
timestamp_us,
thigh_ok,
shank_ok,
thigh_ax_raw,
thigh_ay_raw,
thigh_az_raw,
thigh_gx_raw,
thigh_gy_raw,
thigh_gz_raw,
shank_ax_raw,
shank_ay_raw,
shank_az_raw,
shank_gx_raw,
shank_gy_raw,
shank_gz_raw
```

The actual wire representation is one comma-separated line without line breaks.

Conversion constants copied from the current production sensor driver:

```text
acceleration_g = raw * 0.000061
gyroscope_dps = raw * 0.00875
```

The fixed schema is calibration-only and does not alter the production `TELEM` packet.

#include <Arduino.h>
#include <Arduino_RouterBridge.h>
#include <Wire.h>

#if !defined(ARDUINO_UNO_Q)
#error "Select Arduino UNO Q before compiling this sketch."
#endif

#define IMUCAL_I2C_BUS Wire1

namespace {
constexpr uint32_t SAMPLE_PERIOD_US = 10000;  // 100 Hz
constexpr uint8_t THIGH_IMU_ADDRESS = 0x6A;
constexpr uint8_t SHANK_IMU_ADDRESS = 0x6B;
constexpr uint8_t LSM6DSOX_WHO_AM_I_REG = 0x0F;
constexpr uint8_t LSM6DSOX_WHO_AM_I_VALUE = 0x6C;
constexpr uint8_t LSM6DSOX_CTRL1_XL = 0x10;
constexpr uint8_t LSM6DSOX_CTRL2_G = 0x11;
constexpr uint8_t LSM6DSOX_CTRL3_C = 0x12;
constexpr uint8_t LSM6DSOX_OUTX_L_G = 0x22;
constexpr uint8_t SCHEMA_VERSION = 1;

constexpr uint32_t STATUS_THIGH_READY = 1UL << 0;
constexpr uint32_t STATUS_SHANK_READY = 1UL << 1;
constexpr uint32_t STATUS_CAPTURE_ACTIVE = 1UL << 2;
constexpr uint32_t STATUS_BRIDGE_READY = 1UL << 3;

struct RawImuSample {
  bool ok = false;
  int16_t ax = 0;
  int16_t ay = 0;
  int16_t az = 0;
  int16_t gx = 0;
  int16_t gy = 0;
  int16_t gz = 0;
};

bool g_i2c_started = false;
bool g_bridge_ready = false;
bool g_thigh_ready = false;
bool g_shank_ready = false;
bool g_capture_active = false;
uint32_t g_sample_index = 0;
uint32_t g_next_sample_us = 0;

bool writeRegister(uint8_t address, uint8_t reg, uint8_t value) {
  IMUCAL_I2C_BUS.beginTransmission(address);
  IMUCAL_I2C_BUS.write(reg);
  IMUCAL_I2C_BUS.write(value);
  return IMUCAL_I2C_BUS.endTransmission() == 0;
}

bool readRegisters(uint8_t address, uint8_t start_reg, uint8_t* buffer, uint8_t length) {
  IMUCAL_I2C_BUS.beginTransmission(address);
  IMUCAL_I2C_BUS.write(start_reg);
  if (IMUCAL_I2C_BUS.endTransmission(false) != 0) {
    return false;
  }

  const size_t received = IMUCAL_I2C_BUS.requestFrom(address, static_cast<size_t>(length));
  if (received != length) {
    while (IMUCAL_I2C_BUS.available() > 0) {
      IMUCAL_I2C_BUS.read();
    }
    return false;
  }

  for (uint8_t i = 0; i < length; ++i) {
    buffer[i] = static_cast<uint8_t>(IMUCAL_I2C_BUS.read());
  }
  return true;
}

int16_t readInt16LE(uint8_t low, uint8_t high) {
  return static_cast<int16_t>((static_cast<uint16_t>(high) << 8) | low);
}

bool configureImu(uint8_t address) {
  uint8_t who = 0;
  if (!readRegisters(address, LSM6DSOX_WHO_AM_I_REG, &who, 1) ||
      who != LSM6DSOX_WHO_AM_I_VALUE) {
    return false;
  }

  bool ok = true;
  ok = writeRegister(address, LSM6DSOX_CTRL3_C, 0x44) && ok;  // BDU + auto increment
  ok = writeRegister(address, LSM6DSOX_CTRL1_XL, 0x40) && ok; // 104 Hz, +/-2 g
  ok = writeRegister(address, LSM6DSOX_CTRL2_G, 0x40) && ok;  // 104 Hz, 250 dps
  return ok;
}

RawImuSample readImu(uint8_t address, bool configured) {
  RawImuSample sample;
  if (!configured) {
    return sample;
  }

  uint8_t data[12] = {};
  if (!readRegisters(address, LSM6DSOX_OUTX_L_G, data, sizeof(data))) {
    return sample;
  }

  sample.ok = true;
  sample.gx = readInt16LE(data[0], data[1]);
  sample.gy = readInt16LE(data[2], data[3]);
  sample.gz = readInt16LE(data[4], data[5]);
  sample.ax = readInt16LE(data[6], data[7]);
  sample.ay = readInt16LE(data[8], data[9]);
  sample.az = readInt16LE(data[10], data[11]);
  return sample;
}

void initializeImus() {
  if (!g_i2c_started) {
    IMUCAL_I2C_BUS.begin();
    g_i2c_started = true;
  }
  g_thigh_ready = configureImu(THIGH_IMU_ADDRESS);
  g_shank_ready = configureImu(SHANK_IMU_ADDRESS);
}

uint32_t statusFlags() {
  uint32_t flags = 0;
  if (g_thigh_ready) flags |= STATUS_THIGH_READY;
  if (g_shank_ready) flags |= STATUS_SHANK_READY;
  if (g_capture_active) flags |= STATUS_CAPTURE_ACTIVE;
  if (g_bridge_ready) flags |= STATUS_BRIDGE_READY;
  return flags;
}

uint32_t startCapture() {
  initializeImus();
  g_capture_active = g_thigh_ready && g_shank_ready;
  if (g_capture_active) {
    g_sample_index = 0;
    g_next_sample_us = micros();
  }
  return statusFlags();
}

uint32_t stopCapture() {
  g_capture_active = false;
  return statusFlags();
}

void notifySample(uint32_t timestamp_us, const RawImuSample& thigh, const RawImuSample& shank) {
  Bridge.notify(
      "vision_imu_sample",
      SCHEMA_VERSION,
      g_sample_index,
      timestamp_us,
      thigh.ok ? 1 : 0,
      shank.ok ? 1 : 0,
      thigh.ax,
      thigh.ay,
      thigh.az,
      thigh.gx,
      thigh.gy,
      thigh.gz,
      shank.ax,
      shank.ay,
      shank.az,
      shank.gx,
      shank.gy,
      shank.gz);
}

void captureIfDue() {
  if (!g_capture_active) {
    return;
  }

  const uint32_t now_us = micros();
  if (static_cast<int32_t>(now_us - g_next_sample_us) < 0) {
    return;
  }

  g_next_sample_us += SAMPLE_PERIOD_US;
  if (static_cast<int32_t>(now_us - g_next_sample_us) >= 0) {
    g_next_sample_us = now_us + SAMPLE_PERIOD_US;
  }

  const RawImuSample thigh = readImu(THIGH_IMU_ADDRESS, g_thigh_ready);
  const RawImuSample shank = readImu(SHANK_IMU_ADDRESS, g_shank_ready);
  notifySample(now_us, thigh, shank);
  ++g_sample_index;
}
}  // namespace

void setup() {
  Bridge.begin();
  g_bridge_ready = true;
  initializeImus();
  Bridge.provide_safe("vision_imu_start_capture", startCapture);
  Bridge.provide_safe("vision_imu_stop_capture", stopCapture);
  Bridge.provide_safe("vision_imu_status_flags", statusFlags);
}

void loop() {
  captureIfDue();
  delay(0);
}

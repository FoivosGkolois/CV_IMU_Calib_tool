#include <Arduino.h>
#include <Wire.h>

#if !defined(ARDUINO_UNO_Q)
#error "Select Arduino UNO Q before compiling this sketch."
#endif

// UNO Q Qwiic connector: Wire1.
#define IMUCAL_I2C_BUS Wire1

namespace {
constexpr unsigned long SERIAL_BAUD_RATE = 921600;
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
constexpr size_t COMMAND_BUFFER_SIZE = 96;

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
bool g_thigh_ready = false;
bool g_shank_ready = false;
bool g_capture_active = false;
uint32_t g_sample_index = 0;
uint32_t g_next_sample_us = 0;
char g_command_buffer[COMMAND_BUFFER_SIZE] = {};
size_t g_command_length = 0;

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

void printStatus() {
  Serial.print("INFO,STATUS,thigh_ready=");
  Serial.print(g_thigh_ready ? 1 : 0);
  Serial.print(",shank_ready=");
  Serial.print(g_shank_ready ? 1 : 0);
  Serial.print(",capture_active=");
  Serial.print(g_capture_active ? 1 : 0);
  Serial.print(",sample_rate_hz=100,schema=");
  Serial.println(SCHEMA_VERSION);
}

void printAck(long command_id, const char* command) {
  Serial.print("ACK,");
  Serial.print(command_id);
  Serial.print(',');
  Serial.println(command);
}

void printNack(long command_id, const char* command, const char* reason) {
  Serial.print("NACK,");
  Serial.print(command_id);
  Serial.print(',');
  Serial.print(command);
  Serial.print(',');
  Serial.println(reason);
}

void handleCommand(char* line) {
  char* save = nullptr;
  char* prefix = strtok_r(line, ",", &save);
  char* id_token = strtok_r(nullptr, ",", &save);
  char* command = strtok_r(nullptr, ",", &save);
  if (prefix == nullptr || id_token == nullptr || command == nullptr || strcmp(prefix, "CMD") != 0) {
    return;
  }

  const long command_id = strtol(id_token, nullptr, 10);

  if (strcmp(command, "START_CAPTURE") == 0) {
    initializeImus();
    if (!g_thigh_ready || !g_shank_ready) {
      printNack(command_id, command, "imu_not_ready");
      printStatus();
      return;
    }
    g_sample_index = 0;
    g_next_sample_us = micros();
    g_capture_active = true;
    printAck(command_id, command);
    printStatus();
    return;
  }

  if (strcmp(command, "STOP_CAPTURE") == 0) {
    g_capture_active = false;
    printAck(command_id, command);
    printStatus();
    return;
  }

  if (strcmp(command, "STATUS") == 0 || strcmp(command, "HELLO") == 0) {
    printAck(command_id, command);
    printStatus();
    return;
  }

  printNack(command_id, command, "unknown_command");
}

void pollCommands() {
  while (Serial.available() > 0) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\r') {
      continue;
    }
    if (c == '\n') {
      g_command_buffer[g_command_length] = '\0';
      if (g_command_length > 0) {
        handleCommand(g_command_buffer);
      }
      g_command_length = 0;
      continue;
    }
    if (g_command_length + 1 < COMMAND_BUFFER_SIZE) {
      g_command_buffer[g_command_length++] = c;
    } else {
      g_command_length = 0;
    }
  }
}

void printSample(uint32_t timestamp_us, const RawImuSample& thigh, const RawImuSample& shank) {
  Serial.print("IMUCAL,");
  Serial.print(SCHEMA_VERSION);
  Serial.print(',');
  Serial.print(g_sample_index);
  Serial.print(',');
  Serial.print(timestamp_us);
  Serial.print(',');
  Serial.print(thigh.ok ? 1 : 0);
  Serial.print(',');
  Serial.print(shank.ok ? 1 : 0);
  Serial.print(',');
  Serial.print(thigh.ax); Serial.print(',');
  Serial.print(thigh.ay); Serial.print(',');
  Serial.print(thigh.az); Serial.print(',');
  Serial.print(thigh.gx); Serial.print(',');
  Serial.print(thigh.gy); Serial.print(',');
  Serial.print(thigh.gz); Serial.print(',');
  Serial.print(shank.ax); Serial.print(',');
  Serial.print(shank.ay); Serial.print(',');
  Serial.print(shank.az); Serial.print(',');
  Serial.print(shank.gx); Serial.print(',');
  Serial.print(shank.gy); Serial.print(',');
  Serial.print(shank.gz);
  Serial.println();
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
  printSample(now_us, thigh, shank);
  ++g_sample_index;
}
}  // namespace

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);
  initializeImus();
  Serial.println("INFO,IMUCAL_READY,firmware=uno_q_dual_imu_stream_v1");
  printStatus();
}

void loop() {
  pollCommands();
  captureIfDue();
}

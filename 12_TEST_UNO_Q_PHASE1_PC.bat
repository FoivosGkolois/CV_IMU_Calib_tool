@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv-vision\Scripts\python.exe" (
  echo ERROR: .venv-vision is missing.
  echo Run 08_SETUP_VISION_IMU_PC.bat first.
  exit /b 1
)

echo Upload this sketch to the Arduino UNO Q before continuing:
echo standalone_tools\vision_imu_calibration\arduino\uno_q_dual_imu_stream\uno_q_dual_imu_stream.ino
echo.
echo Close Arduino Serial Monitor and press any key to start the 15-second validation.
pause >nul

".venv-vision\Scripts\python.exe" -m computer.vision_imu_calibration.uno_q_phase1_validation %*
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% EQU 0 (
  echo UNO Q phase 1 validation PASSED.
) else (
  echo UNO Q phase 1 validation FAILED with exit code %EXIT_CODE%.
)
exit /b %EXIT_CODE%

@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=.venv-vision\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo ERROR: Vision-IMU environment not found.
  echo Run 08_SETUP_VISION_IMU_PC.bat first.
  exit /b 1
)

"%PYTHON_EXE%" -m computer.vision_imu_calibration.build_uno_q_phase2_app --project-root "%CD%" --output-dir "%CD%"
if errorlevel 1 exit /b 1

echo.
echo Import or open uno_q_phase2_app_ready.zip in Arduino App Lab.
echo The original PC and Phase 1 workflows remain unchanged.
endlocal

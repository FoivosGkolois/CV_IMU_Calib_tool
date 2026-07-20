@echo off
setlocal
cd /d "%~dp0"

set "VISION_PYTHON=.venv-vision\Scripts\python.exe"

if not exist "%VISION_PYTHON%" (
    echo ERROR: Vision environment not found.
    echo Run 08_SETUP_VISION_IMU_PC.bat first.
    goto :failed
)

"%VISION_PYTHON%" -c "import pytest" >nul 2>&1
if errorlevel 1 (
    echo ERROR: pytest is not installed in .venv-vision.
    echo Re-run 08_SETUP_VISION_IMU_PC.bat.
    goto :failed
)

echo Running Vision-IMU targeted tests...
echo.

"%VISION_PYTHON%" -m pytest ^
    "computer\tests\test_vision_imu_geometry.py" ^
    "computer\tests\test_vision_imu_camera_source.py" ^
    "computer\tests\test_vision_imu_tuning.py" ^
    "computer\tests\test_vision_imu_landmark_filter.py" ^
    "computer\tests\test_vision_imu_runtime.py" ^
    "computer\tests\test_vision_imu_recorder.py" ^
    "computer\tests\test_vision_imu_movenet.py" ^
    "computer\tests\test_vision_imu_serial_protocol.py" ^
    "computer\tests\test_vision_imu_synchronization.py" ^
    -q
set "TEST_EXIT_CODE=%ERRORLEVEL%"

echo.
if "%TEST_EXIT_CODE%"=="0" (
    echo Vision-IMU targeted tests PASSED.
) else (
    echo Vision-IMU targeted tests FAILED with exit code %TEST_EXIT_CODE%.
    echo Read the complete pytest output above.
)
echo.
pause
exit /b %TEST_EXIT_CODE%

:failed
echo.
pause
exit /b 1

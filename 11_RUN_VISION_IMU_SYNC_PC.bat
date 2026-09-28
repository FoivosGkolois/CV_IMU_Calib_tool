@echo off
setlocal
cd /d "%~dp0"

set "VISION_PYTHON=.venv-vision\Scripts\python.exe"
if not exist "%VISION_PYTHON%" (
    echo ERROR: Vision environment not found.
    echo Run 08_SETUP_VISION_IMU_PC.bat first.
    goto :failed
)

"%VISION_PYTHON%" -c "import serial" >nul 2>&1
if errorlevel 1 (
    echo ERROR: pyserial is missing from .venv-vision.
    echo Run 08_SETUP_VISION_IMU_PC.bat once, then retry.
    goto :failed
)

echo Starting synchronized tag-reference / IMU recorder...
echo Camera selector: auto
echo Required tags: Tag 0 THIGH, Tag 1 SHANK
echo Camera profile: select Fast or Quality inside the web GUI
echo UNO R4 IMU: select COM port and press Connect inside the web GUI
echo URL: http://127.0.0.1:8765
echo.

"%VISION_PYTHON%" -m computer.vision_imu_calibration --camera auto --camera-backend auto --width 640 --height 480
set "APP_EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%APP_EXIT_CODE%"=="0" (
    echo The synchronized recorder stopped with exit code %APP_EXIT_CODE%.
    echo Check the camera and error above.
) else (
    echo Vision-IMU application stopped normally.
)
echo.
pause
exit /b %APP_EXIT_CODE%

:failed
echo.
pause
exit /b 1

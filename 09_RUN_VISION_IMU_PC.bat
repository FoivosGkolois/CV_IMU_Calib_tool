@echo off
setlocal
cd /d "%~dp0"

set "VISION_PYTHON=.venv-vision\Scripts\python.exe"

if not exist "%VISION_PYTHON%" (
    echo ERROR: Vision environment not found.
    echo Run 08_SETUP_VISION_IMU_PC.bat first.
    goto :failed
)

echo Starting Vision-IMU AprilTag reference prototype...
echo Camera index: auto
echo Camera mode: 640x480 at 30 FPS
echo Required tags: Tag 0 THIGH, Tag 1 SHANK
echo URL: http://127.0.0.1:8765
echo Stop the application with Ctrl+C.
echo.

"%VISION_PYTHON%" -m computer.vision_imu_calibration --camera auto --camera-backend auto --width 640 --height 480
set "APP_EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%APP_EXIT_CODE%"=="0" (
    echo The application stopped with exit code %APP_EXIT_CODE%.
    echo Read the error above. Common causes are:
    echo - 08_SETUP was not rerun after the tag-reference patch,
    echo - the camera is unavailable,
    echo - another program is using the webcam,
    echo - port 8765 is already in use.
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

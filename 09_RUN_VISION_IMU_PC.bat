@echo off
setlocal
cd /d "%~dp0"

set "VISION_PYTHON=.venv-vision\Scripts\python.exe"

if not exist "%VISION_PYTHON%" (
    echo ERROR: Vision environment not found.
    echo Run 08_SETUP_VISION_IMU_PC.bat first.
    goto :failed
)

echo Starting Vision-IMU MoveNet Thunder prototype...
echo Camera index: 0
echo Camera mode: 640x480 at 30 FPS, fast Windows backend selection
echo URL: http://127.0.0.1:8765
echo After the page opens, select the camera-near leg explicitly.
echo Stop the application with Ctrl+C.
echo.

"%VISION_PYTHON%" -m computer.vision_imu_calibration --camera 0 --camera-backend auto --width 640 --height 480 --leg auto --inference-threads 4
set "APP_EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%APP_EXIT_CODE%"=="0" (
    echo The application stopped with exit code %APP_EXIT_CODE%.
    echo Read the error above. Common causes are:
    echo - 08_SETUP was not rerun after the MoveNet patch,
    echo - the model download failed,
    echo - camera 0 is unavailable,
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

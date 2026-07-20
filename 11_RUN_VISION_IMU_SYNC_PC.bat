@echo off
setlocal
cd /d "%~dp0"

set "VISION_PYTHON=.venv-vision\Scripts\python.exe"
if not exist "%VISION_PYTHON%" (
    echo ERROR: Vision environment not found.
    echo Run 08_SETUP_VISION_IMU_PC.bat first.
    goto :failed
)

set "IMU_PORT=%~1"
if "%IMU_PORT%"=="" (
    set /p IMU_PORT=Enter UNO R4 COM port ^(example COM3^): 
)
if "%IMU_PORT%"=="" (
    echo ERROR: No COM port provided.
    goto :failed
)

echo Starting synchronized Vision-IMU recorder...
echo Camera index: 0
echo Camera mode: 640x480 at 30 FPS, fast Windows backend selection
echo UNO R4 IMU port: %IMU_PORT%
echo URL: http://127.0.0.1:8765
echo.

"%VISION_PYTHON%" -m computer.vision_imu_calibration --camera 0 --camera-backend auto --width 640 --height 480 --leg auto --inference-threads 4 --imu-port "%IMU_PORT%"
set "APP_EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%APP_EXIT_CODE%"=="0" (
    echo The synchronized recorder stopped with exit code %APP_EXIT_CODE%.
    echo Check the camera, COM port, uploaded UNO R4 sketch, and error above.
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

@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python Launcher ^(py.exe^) was not found.
    echo Install 64-bit Python 3.12, then run this script again.
    goto :failed
)

py -3.12 -c "import sys; print(sys.version)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.12 was not found.
    echo Install 64-bit Python 3.12 and keep the existing thesis Python installation unchanged.
    goto :failed
)

if not exist ".venv-vision\Scripts\python.exe" (
    echo Creating isolated vision environment...
    py -3.12 -m venv .venv-vision
    if errorlevel 1 goto :failed
)

call ".venv-vision\Scripts\activate.bat"
if errorlevel 1 goto :failed

echo Updating isolated MoveNet vision environment...
python -m pip install --upgrade pip
if errorlevel 1 goto :failed

python -m pip install --upgrade -r "computer\vision_imu_calibration\requirements.txt"
if errorlevel 1 goto :failed

python -m computer.vision_imu_calibration.download_model
if errorlevel 1 goto :failed

python -c "from ai_edge_litert.interpreter import Interpreter; print('LiteRT interpreter: OK')"
if errorlevel 1 goto :failed

python -c "import serial; from serial.tools import list_ports; print('pyserial: OK')"
if errorlevel 1 goto :failed

echo.
echo MoveNet Thunder vision environment is ready.
echo Next command: 10_TEST_VISION_IMU_PC.bat
echo.
pause
exit /b 0

:failed
echo.
echo SETUP FAILED. The error above is the useful diagnostic.
echo Copy the complete error text if support is needed.
echo.
pause
exit /b 1

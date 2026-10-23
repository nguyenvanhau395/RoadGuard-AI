@echo off
setlocal
title RoadGuard AI Launcher

set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=%PROJECT_DIR%.conda\python.exe"
set "SERVER_SCRIPT=%PROJECT_DIR%scripts\test_interface.py"
set "LOCAL_URL=http://127.0.0.1:8765/"

if not exist "%PYTHON_EXE%" (
    echo [LOI] Khong tim thay moi truong Python:
    echo %PYTHON_EXE%
    echo.
    pause
    exit /b 1
)

if not exist "%SERVER_SCRIPT%" (
    echo [LOI] Khong tim thay server:
    echo %SERVER_SCRIPT%
    echo.
    pause
    exit /b 1
)

pushd "%PROJECT_DIR%"

echo ============================================
echo   RoadGuard AI - Khoi dong nhanh
echo   Local:  %LOCAL_URL%
echo ============================================
echo.

REM Khoi dong RoadGuard server neu chua chay
powershell.exe -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo Dang khoi dong RoadGuard AI Server...
    start "RoadGuard AI Server" /min "%PYTHON_EXE%" "%SERVER_SCRIPT%" --host 127.0.0.1 --port 8765
) else (
    echo RoadGuard AI Server dang chay.
)

REM --- Doi server san sang ---
for /L %%I in (1,1,20) do (
    powershell.exe -NoProfile -Command "try { $r = Invoke-WebRequest -Uri '%LOCAL_URL%' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 }; exit 1 } catch { exit 1 }" >nul 2>&1
    if not errorlevel 1 goto :server_ready
    timeout /t 1 /nobreak >nul
)

echo [LOI] Server khong san sang sau 20 giay.
echo Hay kiem tra cua so RoadGuard AI Server.
popd
pause
exit /b 1

:server_ready
echo       Server san sang!
echo.
echo ============================================
echo   RoadGuard AI da san sang!
echo.
echo   Local:  %LOCAL_URL%
echo ============================================
echo.

if /I "%ROADGUARD_NO_BROWSER%"=="1" goto :launcher_done

REM Mo giao dien local
start "" "%LOCAL_URL%"

:launcher_done
popd
exit /b 0

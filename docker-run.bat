@echo off
setlocal enabledelayedexpansion

echo ==========================================================================
echo   Obscuro Erebus 32B Foundation Model - Docker Launcher (Windows)
echo ==========================================================================
echo.
echo   [1] Full 32B Pipeline Run (install datasets + GPU/CPU train + save)
echo   [2] Start Autonomous Worker Daemon (continuous: learn + train + repeat)
echo   [3] Run Chromium Self-Browser Learning Session Only
echo   [4] GPU + Hardware Diagnostics Check (syscheck)
echo   [5] Stop All Running Containers
echo   [6] View Live Worker Logs
echo   [7] Rebuild Docker Image (after code changes)
echo.
echo ==========================================================================
set /p choice="Select option (1-7): "

if "%choice%"=="1" (
    echo.
    echo [START] Building and running full 32B pipeline...
    docker compose run --rm app fullrun
    goto end
)

if "%choice%"=="2" (
    echo.
    echo [START] Launching autonomous self-training worker daemon...
    docker compose up -d worker
    echo.
    echo Worker is running in background!
    echo Monitor live training logs with:  docker compose logs -f worker
    goto end
)

if "%choice%"=="3" (
    echo.
    echo [START] Running autonomous Chromium browser learning session...
    docker compose run --rm learner
    goto end
)

if "%choice%"=="4" (
    echo.
    echo [START] Running system + GPU diagnostics...
    docker compose run --rm app syscheck
    goto end
)

if "%choice%"=="5" (
    echo.
    echo [STOP] Stopping all containers...
    docker compose down
    goto end
)

if "%choice%"=="6" (
    echo.
    echo [LOGS] Streaming live worker logs (Ctrl+C to stop)...
    docker compose logs -f worker
    goto end
)

if "%choice%"=="7" (
    echo.
    echo [BUILD] Rebuilding Docker image with latest code changes...
    docker compose build --no-cache
    echo [OK] Rebuild complete.
    goto end
)

echo Invalid choice. Please run the script again and select 1-7.

:end
endlocal
pause

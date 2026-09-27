@echo off
cd /d "%~dp0"
if exist ".\dist\combined_foundation_model.exe" (
    echo Obscuro Erebus launcher
    echo Choose a command:
    echo 1) verify
    echo 2) industrial
    echo 3) export_erebus
    echo 4) fullrun
    echo 5) help
    set /p choice="Enter choice [1-5]: "

    if "%choice%"=="1" (
        .\dist\combined_foundation_model.exe verify
    ) else if "%choice%"=="2" (
        .\dist\combined_foundation_model.exe industrial
    ) else if "%choice%"=="3" (
        .\dist\combined_foundation_model.exe export_erebus
    ) else if "%choice%"=="4" (
        .\dist\combined_foundation_model.exe fullrun
    ) else (
        .\dist\combined_foundation_model.exe --help
    )
) else (
    echo ERROR: EXE not found at .\dist\combined_foundation_model.exe
    echo Build it first with:
    echo py -m PyInstaller --onefile --console main.py --name combined_foundation_model
)
pause

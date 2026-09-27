@echo off
cd /d "%~dp0"
if exist ".\dist\combined_foundation_model.exe" (
    echo Starting Erebus model...
    .\dist\combined_foundation_model.exe demo
) else (
    echo ERROR: EXE not found at .\dist\combined_foundation_model.exe
    echo Build it first with:
    echo py -m PyInstaller --onefile --console main.py --name combined_foundation_model
)
pause

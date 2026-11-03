@echo off
cd /d "%~dp0"

if not exist venv\Scripts\activate.bat (
    echo [x] Folder venv tidak ditemukan di sini.
    echo     Pastikan file .bat ini ada di folder yang sama dengan "venv".
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python gui.py

if errorlevel 1 (
    echo.
    echo [x] Program berhenti dengan error, lihat pesan di atas.
    pause
)

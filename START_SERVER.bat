:start
@echo off
echo ===================================================
echo MelodyCanvas Pro v3 - Lead Sheet Master [REPAIRED]
echo ===================================================

:: Setting paths
set PYTHON_EXE=C:\Users\PC\AppData\Local\Programs\Python\Python310\python.exe
set PIP_EXE=C:\Users\PC\AppData\Local\Programs\Python\Python310\Scripts\pip.exe
set PROJECT_DIR=%~dp0backend

echo [1] Checking Flask environment...
"%PIP_EXE%" install flask flask-cors librosa numpy scipy soundfile --quiet

echo [2] Starting app.py on port 5050...
cd /d "%PROJECT_DIR%"
"%PYTHON_EXE%" app.py

pause

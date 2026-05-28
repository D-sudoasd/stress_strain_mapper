@echo off
setlocal
chcp 65001 >nul

set "APP_DIR=%~dp0"
set "APP_FILE=sxrd_stress_strain_mapper_gui_v3.py"

pushd "%APP_DIR%" >nul 2>nul
if errorlevel 1 (
    echo Failed to enter app directory:
    echo %APP_DIR%
    pause
    exit /b 1
)

if not exist "%APP_FILE%" (
    echo Cannot find %APP_FILE% in:
    echo %CD%
    pause
    exit /b 1
)

call :find_python
if errorlevel 1 (
    echo No usable Python 3 interpreter was found.
    echo Install Python 3, or make sure py.exe/python.exe is available in PATH.
    pause
    exit /b 1
)

echo Using Python: %PYTHON_CMD%

%PYTHON_CMD% -c "import tkinter, numpy, pandas, matplotlib, scipy, openpyxl" >nul 2>nul
if errorlevel 1 (
    echo Installing or updating required Python packages...
    %PYTHON_CMD% -m pip install --upgrade pip pandas numpy matplotlib scipy openpyxl
    if errorlevel 1 (
        echo Failed to install required Python packages.
        echo If the message mentions tkinter, reinstall Python with Tcl/Tk support.
        pause
        exit /b 1
    )
)

%PYTHON_CMD% -c "import tkinter, numpy, pandas, matplotlib, scipy, openpyxl" >nul 2>nul
if errorlevel 1 (
    echo Python dependencies are still incomplete.
    echo Try running this file from a Command Prompt to see the full error output.
    pause
    exit /b 1
)

%PYTHON_CMD% "%APP_FILE%"
if errorlevel 1 (
    echo Application exited with an error.
    pause
    exit /b 1
)

popd >nul
exit /b 0

:find_python
set "PYTHON_CMD="

where py >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info[0] == 3 else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3"
        exit /b 0
    )
)

for /f "delims=" %%P in ('where python 2^>nul') do (
    "%%P" -c "import sys; raise SystemExit(0 if sys.version_info[0] == 3 else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD="%%P""
        exit /b 0
    )
)

exit /b 1

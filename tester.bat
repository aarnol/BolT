@echo off
setlocal enabledelayedexpansion

:: Example values
set smoothings=0 6 10
set atlas=brodmann


for %%A in (%atlas%) do (
    for %%S in (%smoothings%) do (
        set datasets=!datasets! hcpMotor_%%S_%%A
    )
)

for %%D in (%datasets%) do (
    echo.
    echo Running: python tester.py %%D
    echo ------------------------------------------------------------
    python tester.py -a True -d hcpMotor -a True --name %%D -m bolT
    echo ------------------------------------------------------------
    echo Done: python tester.py %%D
    echo ------------------------------------------------------------
    echo Press any key to continue to the next run...
    pause >nul
)

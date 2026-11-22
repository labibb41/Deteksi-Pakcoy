@echo off
title Deteksi Samping Pakcoy - SemaiKu
cls
echo ========================================================================
echo   🌿 DETEKSI SISI SAMPING PAKCOY (YOLO26n-Seg)
echo ========================================================================
echo.
if "%~1"=="" (
    "F:\Bismillah TA\.venv\Scripts\python.exe" deteksi_samping_cli.py
) else (
    "F:\Bismillah TA\.venv\Scripts\python.exe" deteksi_samping_cli.py "%~1"
)
pause


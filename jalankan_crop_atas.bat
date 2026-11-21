@echo off
cd /d "f:\Bismillah TA"
echo.
echo =============================================
echo   CROP TOOL - TAMPAK ATAS (TOP VIEW)
echo =============================================
echo   Letakkan foto kamera di folder:
echo   f:\Bismillah TA\foto_kamera_atas\
echo.
echo   Kontrol:
echo     Klik + Drag        = Gambar area crop
echo     1 / N              = NORMAL
echo     2 / R              = ETIOLASI RINGAN
echo     3 / B              = ETIOLASI BERAT
echo     4 / E              = EMPTY
echo     Z                  = Undo
echo     SPACE              = Gambar berikutnya
echo     Q / ESC            = Keluar
echo =============================================
echo.
set /p FOLDER="Masukkan path folder foto (Enter=default): "
if "%FOLDER%"=="" set FOLDER=f:\Bismillah TA\foto_kamera_atas
C:\penv\Scripts\python crop_tool_atas.py "%FOLDER%"
pause

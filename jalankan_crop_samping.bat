@echo off
cd /d "f:\Bismillah TA"
echo.
echo =============================================
echo   CROP TOOL - TAMPAK SAMPING (SIDE VIEW)
echo =============================================
echo   Letakkan foto kamera di folder:
echo   f:\Bismillah TA\foto_kamera_samping\
echo.
echo   Kontrol:
echo     C                  = Kalibrasi skala (cm/pixel)
echo     Klik + Drag        = Crop area batang
echo     ENTER              = Simpan (kelas otomatis)
echo     N                  = Simpan NORMAL (paksa)
echo     E                  = Simpan ETIOLASI (paksa)
echo     Z                  = Undo
echo     SPACE              = Gambar berikutnya
echo     Q / ESC            = Keluar
echo.
echo   Threshold: Normal = tinggi batang kurang dari 7 cm
echo =============================================
echo.
set /p FOLDER="Masukkan path folder foto (Enter=default): "
if "%FOLDER%"=="" set FOLDER=f:\Bismillah TA\foto_kamera_samping
C:\penv\Scripts\python crop_tool_samping.py "%FOLDER%"
pause

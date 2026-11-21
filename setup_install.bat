@echo off
echo ============================================
echo   SETUP: Aktifkan Long Path + Install Paket
echo   (Jalankan sebagai Administrator!)
echo ============================================
echo.

:: Aktifkan Windows Long Path Support
echo [1/3] Mengaktifkan Windows Long Path Support...
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
if %errorlevel% == 0 (
    echo      OK: Long Path berhasil diaktifkan!
) else (
    echo      GAGAL: Jalankan file ini sebagai Administrator
    pause
    exit /b 1
)

echo.
echo [2/3] Update pip...
python -m pip install --upgrade pip --quiet

echo.
echo [3/3] Install semua dependensi...
pip install tensorflow ultralytics gradio opencv-python matplotlib Pillow
if %errorlevel% == 0 (
    echo.
    echo ============================================
    echo   SELESAI! Semua paket berhasil diinstall.
    echo   Sekarang jalankan: jalankan_app.bat
    echo ============================================
) else (
    echo.
    echo   Ada error saat install. Cek pesan di atas.
)

pause

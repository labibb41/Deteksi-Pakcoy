@echo off
echo ==================================================
echo   SETUP: Buat Virtual Environment + Install Paket
echo   (Solusi: path pendek C:\penv agar tidak error)
echo ==================================================
echo.

echo [1/4] Membuat virtual environment di C:\penv ...
python -m venv C:\penv
if errorlevel 1 (
    echo ERROR: Gagal buat venv. Pastikan Python terinstall.
    pause & exit /b 1
)
echo       OK!

echo.
echo [2/4] Update pip...
C:\penv\Scripts\python -m pip install --upgrade pip --quiet
echo       OK!

echo.
echo [3/4] Install paket utama (tensorflow, ultralytics, gradio)...
echo       Proses ini 5-15 menit tergantung internet...
C:\penv\Scripts\pip install tensorflow ultralytics gradio
if errorlevel 1 ( echo ERROR saat install paket utama & pause & exit /b 1 )

echo.
echo [4/4] Install paket pendukung...
C:\penv\Scripts\pip install opencv-python matplotlib Pillow
if errorlevel 1 ( echo ERROR saat install paket pendukung & pause & exit /b 1 )

echo.
echo ==================================================
echo   SELESAI! Semua paket berhasil diinstall.
echo.
echo   Untuk menjalankan aplikasi deteksi:
echo     jalankan_app.bat
echo.
echo   Untuk tool crop dataset atas:
echo     jalankan_crop_atas.bat
echo.
echo   Untuk tool crop dataset samping:
echo     jalankan_crop_samping.bat
echo ==================================================
pause

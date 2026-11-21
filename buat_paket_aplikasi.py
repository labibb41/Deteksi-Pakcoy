import os
import zipfile
import shutil

app_dir = "aplikasi_deteksi_pakcoy"
zip_filename = "aplikasi_deteksi_pakcoy.zip"

if os.path.exists(app_dir):
    shutil.rmtree(app_dir)
os.makedirs(app_dir, exist_ok=True)
os.makedirs(os.path.join(app_dir, "exp_pakcoy_yolo26_seg_v4", "weights"), exist_ok=True)

print("📦 1. Menyalin model YOLO...")
shutil.copy2("exp_pakcoy_yolo26_seg_v4/weights/best.pt", os.path.join(app_dir, "exp_pakcoy_yolo26_seg_v4", "weights", "best.pt"))

print("📦 2. Menyalin 3 Model MobileNetV2 Terbaru...")
shutil.copy2("mobilenetv2_pakcoy_opt.h5", os.path.join(app_dir, "mobilenetv2_pakcoy_opt.h5"))
shutil.copy2("mobilenetv2_pakcoy_murni.h5", os.path.join(app_dir, "mobilenetv2_pakcoy_murni.h5"))
shutil.copy2("mobilenetv2_pakcoy_augmented.h5", os.path.join(app_dir, "mobilenetv2_pakcoy_augmented.h5"))

print("📦 3. Menyalin Skrip Deteksi, Foto Contoh, dan Requirements...")
shutil.copy2("deteksi_cli.py", os.path.join(app_dir, "deteksi_cli.py"))
shutil.copy2("deteksi_pakcoy.py", os.path.join(app_dir, "deteksi_pakcoy.py"))
shutil.copy2("test_tray.jpg", os.path.join(app_dir, "test_tray.jpg"))
shutil.copy2("requirements.txt", os.path.join(app_dir, "requirements.txt"))

print("📦 4. Membuat Launcher Batch...")
with open(os.path.join(app_dir, "jalankan_gui.bat"), "w", encoding="utf-8") as f:
    f.write("@echo off\n")
    f.write("cd /d \"%~dp0\"\n")
    f.write("echo ===================================================\n")
    f.write("echo   Menjalankan Aplikasi Web Deteksi Pakcoy (Gradio)...\n")
    f.write("echo ===================================================\n")
    f.write("python deteksi_pakcoy.py\n")
    f.write("pause\n")

with open(os.path.join(app_dir, "jalankan_cli.bat"), "w", encoding="utf-8") as f:
    f.write("@echo off\n")
    f.write("cd /d \"%~dp0\"\n")
    f.write("echo ===================================================\n")
    f.write("echo   Menjalankan Deteksi Pakcoy Interaktif (OpenCV)...\n")
    f.write("echo ===================================================\n")
    f.write("python deteksi_cli.py\n")
    f.write("pause\n")

print("📦 5. Membuat Buku Panduan...")
panduan = """========================================================================
  🌿 SISTEM DETEKSI & KLASIFIKASI PERTUMBUHAN PAKCOY (TA)
========================================================================

📦 ISI PAKET:
  1. exp_pakcoy_yolo26_seg_v4/weights/best.pt -> Model YOLO (Deteksi Tray & Bibit)
  2. mobilenetv2_pakcoy_opt.h5                 -> Model MobileNetV2 Optimasi (Terbaru #1)
  3. mobilenetv2_pakcoy_murni.h5               -> Model MobileNetV2 Murni (#2)
  4. mobilenetv2_pakcoy_augmented.h5           -> Model MobileNetV2 Augmented (#3)
  5. deteksi_cli.py                            -> Program CLI Interaktif OpenCV (Zoom/Pan/Shortcut 1-3)
  6. deteksi_pakcoy.py                         -> Program Web GUI Gradio
  7. test_tray.jpg                             -> Foto contoh bibit pakcoy untuk tes
  8. requirements.txt                          -> Daftar library Python yang dibutuhkan
  9. jalankan_gui.bat                          -> Shortcut 1-klik untuk membuka Web UI
 10. jalankan_cli.bat                          -> Shortcut 1-klik untuk membuka Layar Interaktif

🚀 CARA INSTALL & JALANKAN:
  1. Pastikan Python sudah terinstall (disarankan Python 3.10 atau 3.11).
  2. Buka terminal (CMD / PowerShell) di folder ini, lalu jalankan:
       pip install -r requirements.txt
  3. Cara menjalankan:
     - Klik 2x file "jalankan_gui.bat" untuk membuka antarmuka Web Gradio di browser.
     - Klik 2x file "jalankan_cli.bat" untuk membuka antarmuka Interaktif OpenCV di layar.

⌨️ TOMBOL PADA TAMPILAN INTERAKTIF (CLI):
  - Scroll Roda Mouse : Zoom In / Zoom Out
  - Klik & Drag Kiri  : Geser / Pan melihat bibit dari dekat
  - Tombol '1'        : Tampilkan Model Optimasi Terbaru (Default)
  - Tombol '2'        : Tampilkan Model Murni
  - Tombol '3'        : Tampilkan Model Augmented
  - Tombol 'R'        : Reset Zoom ke tampilan penuh
  - Tombol 'ESC' / 'Q': Keluar dari aplikasi
========================================================================
"""
with open(os.path.join(app_dir, "BACA_PANDUAN.txt"), "w", encoding="utf-8") as f:
    f.write(panduan)

print(f"📦 6. Mengompresi folder ke {zip_filename}...")
with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files_list in os.walk(app_dir):
        for file in files_list:
            full_p = os.path.join(root, file)
            rel_p = os.path.relpath(full_p, app_dir)
            z.write(full_p, rel_p)

zip_size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
print(f"✅ SUKSES! File ZIP siap dipakai: {zip_filename} ({zip_size_mb:.2f} MB)")

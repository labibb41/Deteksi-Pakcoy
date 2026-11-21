"""
==============================================================
  CEK DATASET - Quick Summary
  Menampilkan jumlah gambar per kelas di semua versi dataset
==============================================================
"""
import os
import sys

# Auto-relaunch jika dipanggil tanpa venv
PENV_PYTHON = r"C:\penv\Scripts\python.exe"
if os.path.exists(PENV_PYTHON) and os.path.normpath(sys.executable).lower() != os.path.normpath(PENV_PYTHON).lower():
    import subprocess
    sys.exit(subprocess.call([PENV_PYTHON] + sys.argv))

import update_dataset

if __name__ == "__main__":
    update_dataset.show_stats()

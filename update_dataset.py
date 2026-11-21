"""
==============================================================
  UPDATE DATASET - Pakcoy Project
  Kelola, merge, split, dan export dataset untuk Colab
==============================================================
"""

import os
import sys
import shutil
import glob
import random
import json
import zipfile
from datetime import datetime
from pathlib import Path

# ───────────────────────────────────────────────────────────────
#  KONFIGURASI PATH
# ───────────────────────────────────────────────────────────────
BASE = r"f:\Bismillah TA"

DATASETS = {
    "Dataset Colab (awal)": os.path.join(BASE, "dataset_top_colab (1)"),
    "Dataset V1":           os.path.join(BASE, "dataset_top_v1"),
    "Dataset V2":           os.path.join(BASE, "dataset_top_v2"),
    "Dataset V3":           os.path.join(BASE, "dataset_top_v3"),
    "Dataset Baru Atas":    os.path.join(BASE, "dataset_baru_atas"),
    "Dataset Baru Samping": os.path.join(BASE, "dataset_baru_samping"),
}

# Dataset utama untuk training (target merge)
MAIN_DATASET  = os.path.join(BASE, "dataset_top_v3")
SPLIT_OUTPUT  = os.path.join(BASE, "dataset_split")
ZIP_OUTPUT    = os.path.join(BASE, "dataset_colab_export.zip")

CLASSES_ATAS    = ["normal", "etiolasi_ringan", "etiolasi_berat", "empty"]
CLASSES_SAMPING = ["normal", "etiolasi"]

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

# ───────────────────────────────────────────────────────────────
#  HELPER: HITUNG FILE
# ───────────────────────────────────────────────────────────────
def count_classes(folder):
    """Return dict {class_name: count} untuk semua subfolder."""
    result = {}
    if not os.path.isdir(folder):
        return result
    for cls_dir in sorted(os.listdir(folder)):
        full_path = os.path.join(folder, cls_dir)
        if os.path.isdir(full_path):
            files = [f for f in os.listdir(full_path)
                     if f.lower().endswith(IMG_EXTS)]
            result[cls_dir] = len(files)
    return result


def print_stats_table(label, folder):
    counts = count_classes(folder)
    if not counts:
        print(f"   {label}: (folder tidak ada atau kosong)")
        return 0
    total = sum(counts.values())
    print(f"\n   📂 {label}  ({folder})")
    for cls, cnt in counts.items():
        bar  = "█" * (cnt // 10) if cnt >= 10 else "░"
        pct  = cnt / total * 100 if total > 0 else 0
        print(f"      {cls:<20}: {cnt:5d}  ({pct:5.1f}%)  {bar}")
    print(f"      {'─'*22}")
    print(f"      {'TOTAL':<20}: {total:5d}")
    return total


# ───────────────────────────────────────────────────────────────
#  MENU 1: STATISTIK
# ───────────────────────────────────────────────────────────────
def show_stats():
    print("\n" + "="*65)
    print("  📊 STATISTIK SEMUA DATASET")
    print("="*65)
    grand_total = 0
    for label, folder in DATASETS.items():
        grand_total += print_stats_table(label, folder)
    print(f"\n  {'─'*63}")
    print(f"  GRAND TOTAL SEMUA DATASET: {grand_total} gambar")
    print("="*65)


# ───────────────────────────────────────────────────────────────
#  MENU 2: MERGE DATASET BARU → UTAMA
# ───────────────────────────────────────────────────────────────
def merge_new_to_main():
    print("\n" + "="*65)
    print("  🔀 MERGE DATASET BARU → DATASET UTAMA")
    print(f"  Target: {MAIN_DATASET}")
    print("="*65)

    sources = {
        "Dataset Baru Atas":    os.path.join(BASE, "dataset_baru_atas"),
        "Dataset Baru Samping": os.path.join(BASE, "dataset_baru_samping"),
    }

    total_copied = 0
    for src_label, src_folder in sources.items():
        if not os.path.isdir(src_folder):
            print(f"\n  ⚠️  {src_label}: folder tidak ada ({src_folder})")
            continue

        counts = count_classes(src_folder)
        if not counts:
            print(f"\n  ⚠️  {src_label}: tidak ada gambar untuk di-merge")
            continue

        print(f"\n  Memproses {src_label}...")
        for cls_name, cnt in counts.items():
            src_cls  = os.path.join(src_folder, cls_name)
            dst_cls  = os.path.join(MAIN_DATASET, cls_name)
            os.makedirs(dst_cls, exist_ok=True)

            files   = [f for f in os.listdir(src_cls)
                       if f.lower().endswith(IMG_EXTS)]
            copied  = 0
            skipped = 0
            for fname in files:
                src_f = os.path.join(src_cls, fname)
                dst_f = os.path.join(dst_cls, fname)
                if os.path.exists(dst_f):
                    skipped += 1
                    continue
                shutil.copy2(src_f, dst_f)
                copied += 1

            print(f"    {cls_name:<20}: {copied} disalin, {skipped} sudah ada")
            total_copied += copied

    print(f"\n  ✅ Total {total_copied} gambar berhasil di-merge ke dataset utama")

    if total_copied > 0:
        print("\n  Statistik dataset utama setelah merge:")
        print_stats_table("Dataset Utama (V3)", MAIN_DATASET)


# ───────────────────────────────────────────────────────────────
#  MENU 3: BUAT TRAIN/VAL SPLIT
# ───────────────────────────────────────────────────────────────
def create_train_val_split():
    print("\n" + "="*65)
    print("  ✂️  BUAT TRAIN / VAL SPLIT")
    print("="*65)

    while True:
        try:
            ratio_str = input("\n  Masukkan rasio train (default=0.8 = 80% train, 20% val): ").strip()
            train_ratio = float(ratio_str) if ratio_str else 0.8
            if 0.5 <= train_ratio <= 0.95:
                break
            print("  ⚠️  Rasio harus antara 0.5 dan 0.95")
        except ValueError:
            train_ratio = 0.8
            break

    print(f"\n  Rasio: Train={train_ratio:.0%}  Val={1-train_ratio:.0%}")
    print(f"  Output: {SPLIT_OUTPUT}")

    # Pilih sumber
    print(f"\n  Sumber: {MAIN_DATASET}")

    counts = count_classes(MAIN_DATASET)
    if not counts:
        print("  ⚠️  Dataset utama kosong! Jalankan Merge terlebih dahulu.")
        return

    seed = 42
    random.seed(seed)
    total_train = 0
    total_val   = 0

    for cls_name in counts:
        src_cls = os.path.join(MAIN_DATASET, cls_name)
        files   = [f for f in os.listdir(src_cls)
                   if f.lower().endswith(IMG_EXTS)]
        random.shuffle(files)

        n_train  = int(len(files) * train_ratio)
        train_fs = files[:n_train]
        val_fs   = files[n_train:]

        for split, file_list in [("train", train_fs), ("val", val_fs)]:
            dst_dir = os.path.join(SPLIT_OUTPUT, split, cls_name)
            os.makedirs(dst_dir, exist_ok=True)
            for f in file_list:
                shutil.copy2(os.path.join(src_cls, f),
                             os.path.join(dst_dir, f))

        print(f"  {cls_name:<20}: train={len(train_fs):4d}  val={len(val_fs):4d}")
        total_train += len(train_fs)
        total_val   += len(val_fs)

    print(f"\n  ✅ Split selesai!")
    print(f"     Total Train: {total_train}")
    print(f"     Total Val:   {total_val}")
    print(f"     Output: {SPLIT_OUTPUT}")

    # Generate class_names.txt
    cls_file = os.path.join(SPLIT_OUTPUT, "class_names.txt")
    with open(cls_file, "w") as f:
        for cls in sorted(counts.keys()):
            f.write(cls + "\n")
    print(f"     class_names.txt: {cls_file}")


# ───────────────────────────────────────────────────────────────
#  MENU 4: EXPORT ZIP UNTUK COLAB
# ───────────────────────────────────────────────────────────────
def export_for_colab():
    print("\n" + "="*65)
    print("  📦 EXPORT DATASET → ZIP UNTUK GOOGLE COLAB")
    print("="*65)

    # Pilih dataset yang akan di-zip
    print("\n  Yang akan di-zip:")
    print(f"    1. Dataset split (train/val): {SPLIT_OUTPUT}")
    print(f"    2. Dataset utama (V3): {MAIN_DATASET}")
    choice = input("\n  Pilih (1/2, default=1): ").strip() or "1"

    if choice == "2":
        src = MAIN_DATASET
        zip_name = os.path.join(BASE, f"dataset_main_{datetime.now().strftime('%Y%m%d_%H%M')}.zip")
    else:
        src = SPLIT_OUTPUT
        zip_name = os.path.join(BASE, f"dataset_split_{datetime.now().strftime('%Y%m%d_%H%M')}.zip")

    if not os.path.isdir(src):
        print(f"  ⚠️  Folder tidak ditemukan: {src}")
        print("  Jalankan Split terlebih dahulu (menu 3)")
        return

    print(f"\n  Membuat zip dari: {src}")
    print(f"  Output: {zip_name}")
    print("  Memproses...")

    total_files = 0
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(src):
            dirs.sort()
            for file in sorted(files):
                if file.lower().endswith(IMG_EXTS) or file.endswith('.txt') or file.endswith('.yaml'):
                    full_path = os.path.join(root, file)
                    arc_name  = os.path.relpath(full_path, os.path.dirname(src))
                    zf.write(full_path, arc_name)
                    total_files += 1

    size_mb = os.path.getsize(zip_name) / 1024 / 1024
    print(f"\n  ✅ Zip selesai!")
    print(f"     File: {zip_name}")
    print(f"     Total gambar: {total_files}")
    print(f"     Ukuran: {size_mb:.1f} MB")
    print(f"\n  Upload file ini ke Google Drive, lalu di Colab:")
    print(f"     from google.colab import drive")
    print(f"     drive.mount('/content/drive')")
    print(f"     !unzip '/content/drive/MyDrive/...' -d '/content/dataset'")


# ───────────────────────────────────────────────────────────────
#  MENU 5: GENERATE class_names.txt + INFO JSON
# ───────────────────────────────────────────────────────────────
def generate_info():
    print("\n" + "="*65)
    print("  📄 GENERATE INFO FILE")
    print("="*65)

    counts   = count_classes(MAIN_DATASET)
    info     = {
        "project":    "Pakcoy Growth Detection",
        "generated":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        "classes":    list(sorted(counts.keys())),
        "total":      sum(counts.values()),
        "per_class":  {k: v for k, v in sorted(counts.items())},
        "threshold_batang_cm": 7.0,
        "model_top_view":  "MobileNetV2",
        "model_detection": "YOLO26n-seg",
    }

    info_path = os.path.join(BASE, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n  ✅ Disimpan: {info_path}")
    print(json.dumps(info, indent=2))


# ───────────────────────────────────────────────────────────────
#  MENU 6: VALIDASI GABUNGAN (ATAS + SAMPING)
# ───────────────────────────────────────────────────────────────
def show_combined_logic():
    print("\n" + "="*65)
    print("  🔍 LOGIKA VALIDASI GABUNGAN")
    print("="*65)
    print("""
  Pipeline validasi untuk setiap titik tanam:

  ┌──────────────────────────────────────────────────┐
  │  YOLO26n-Seg  →  Deteksi tanaman (top view)      │
  │       ↓                                          │
  │  MobileNetV2  →  Klasifikasi kondisi daun        │
  │       ↓                                          │
  │  Pengukuran   →  Tinggi batang (side view)       │
  └──────────────────────────────────────────────────┘

  TABEL VALIDASI (1 titik tanam):
  ┌─────────────────────┬──────────────┬─────────────┐
  │  Kondisi Daun       │  Tinggi      │  Hasil      │
  ├─────────────────────┼──────────────┼─────────────┤
  │  normal             │  < 7 cm      │  ✅ NORMAL  │
  │  normal             │  >= 7 cm     │  ⚠️  ETIOL. │
  │  etiolasi_ringan    │  < 7 cm      │  ⚠️  ETIOL. │
  │  etiolasi_ringan    │  >= 7 cm     │  ❌ ETIOL.  │
  │  etiolasi_berat     │  (apapun)    │  ❌ BERAT   │
  │  empty              │  (apapun)    │  ⬜ KOSONG  │
  └─────────────────────┴──────────────┴─────────────┘

  TRAY: 6 × 12 = 72 titik
  Analisis samping: 1 baris (6 titik) agar tidak tumpang tindih
""")


# ───────────────────────────────────────────────────────────────
#  MAIN MENU
# ───────────────────────────────────────────────────────────────
def main():
    menu_items = {
        "1": ("📊 Lihat statistik semua dataset",         show_stats),
        "2": ("🔀 Merge dataset baru → dataset utama",    merge_new_to_main),
        "3": ("✂️  Buat train/val split (80/20)",          create_train_val_split),
        "4": ("📦 Export ZIP untuk Google Colab",         export_for_colab),
        "5": ("📄 Generate info file (JSON)",             generate_info),
        "6": ("🔍 Tampilkan logika validasi gabungan",    show_combined_logic),
        "0": ("🚪 Keluar",                                None),
    }

    while True:
        print("\n" + "═"*65)
        print("  🌿 UPDATE DATASET - Pakcoy Project")
        print("═"*65)
        for key, (label, _) in menu_items.items():
            print(f"  [{key}] {label}")
        print("─"*65)

        choice = input("\n  Pilih menu: ").strip()

        if choice not in menu_items:
            print("  ⚠️  Pilihan tidak valid.")
            continue

        label, fn = menu_items[choice]
        if fn is None:
            print("\n  Sampai jumpa! 👋")
            break

        try:
            fn()
        except KeyboardInterrupt:
            print("\n  (Dibatalkan)")
        except Exception as ex:
            print(f"\n  ❌ Error: {ex}")

        input("\n  Tekan Enter untuk kembali ke menu...")


if __name__ == "__main__":
    main()

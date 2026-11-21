"""
==============================================================
  CROP TOOL - TAMPAK ATAS (TOP VIEW)
  Dataset Labeling Tool untuk Tanaman Pakcoy
==============================================================
  Kontrol:
    Klik + Drag  : Gambar area crop
    1 / N        : Simpan → NORMAL
    2 / R        : Simpan → ETIOLASI RINGAN
    3 / B        : Simpan → ETIOLASI BERAT
    4 / E        : Simpan → EMPTY / KOSONG
    Z            : Undo (hapus crop terakhir)
    ESC          : Batal gambar aktif
    SPACE        : Gambar berikutnya
    Q            : Keluar program
==============================================================
"""

import cv2
import numpy as np
import os
import sys

# ── AUTO-RELAUNCH DENGAN C:\penv ──
PENV_PYTHON = r"C:\penv\Scripts\python.exe"
if os.path.exists(PENV_PYTHON) and os.path.normpath(sys.executable).lower() != os.path.normpath(PENV_PYTHON).lower():
    import subprocess
    sys.exit(subprocess.call([PENV_PYTHON] + sys.argv))

import glob
from datetime import datetime

# ───────────────────────────────────────────────────────────────
#  KONFIGURASI
# ───────────────────────────────────────────────────────────────
DEFAULT_SOURCE  = r"f:\Bismillah TA\foto_kamera_atas"
DEFAULT_DATASET = r"f:\Bismillah TA\dataset_baru_atas"

CLASS_KEYS = {
    ord('1'): 'normal',         ord('n'): 'normal',
    ord('2'): 'etiolasi_ringan', ord('r'): 'etiolasi_ringan',
    ord('3'): 'etiolasi_berat',  ord('b'): 'etiolasi_berat',
    ord('4'): 'empty',           ord('e'): 'empty',
}

CLASS_COLOR = {           # BGR
    'normal':           (50,  200, 50),
    'etiolasi_ringan':  (0,   165, 255),
    'etiolasi_berat':   (0,   0,   220),
    'empty':            (140, 140, 140),
}

CLASS_LABEL = {
    'normal':           '1/N  NORMAL',
    'etiolasi_ringan':  '2/R  ETIOLASI RINGAN',
    'etiolasi_berat':   '3/B  ETIOLASI BERAT',
    'empty':            '4/E  EMPTY',
}

IMG_EXTS   = ('*.jpg','*.jpeg','*.png','*.bmp','*.JPG','*.JPEG','*.PNG')
MIN_PX     = 20          # ukuran crop minimum (pixels)
MAX_DISPLAY= 1100        # panjang sisi maksimal untuk tampilan

# ───────────────────────────────────────────────────────────────
#  STATE GLOBAL MOUSE
# ───────────────────────────────────────────────────────────────
mx_state  = "IDLE"      # IDLE | DRAWING | PENDING
mx_start  = (0, 0)
mx_curr   = (0, 0)
mx_draw   = False

def mouse_cb(event, x, y, flags, param):
    global mx_state, mx_start, mx_curr, mx_draw
    if mx_state == "PENDING":
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        mx_draw  = True
        mx_start = (x, y)
        mx_curr  = (x, y)
        mx_state = "DRAWING"
    elif event == cv2.EVENT_MOUSEMOVE and mx_draw:
        mx_curr = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        mx_curr = (x, y)
        mx_draw = False
        w = abs(mx_curr[0] - mx_start[0])
        h = abs(mx_curr[1] - mx_start[1])
        mx_state = "PENDING" if (w > MIN_PX and h > MIN_PX) else "IDLE"


def get_rect():
    x1 = min(mx_start[0], mx_curr[0])
    y1 = min(mx_start[1], mx_curr[1])
    x2 = max(mx_start[0], mx_curr[0])
    y2 = max(mx_start[1], mx_curr[1])
    return x1, y1, x2, y2


# ───────────────────────────────────────────────────────────────
#  HELPER: DRAW OVERLAY
# ───────────────────────────────────────────────────────────────
def draw_overlay(disp, img_name, idx, total, stats, state):
    h, w = disp.shape[:2]
    bar_h = 95

    # Semi-transparent top bar
    overlay = disp.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, disp, 0.28, 0, disp)

    # Garis batas bawah bar
    cv2.line(disp, (0, bar_h), (w, bar_h), (80, 80, 80), 1)

    def txt(text, pos, scale=0.44, color=(220,220,220), thick=1):
        cv2.putText(disp, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    # Baris 1 - info gambar
    txt(f"[{idx+1}/{total}]  {img_name}", (10, 18), 0.5, (255,255,255))

    # Baris 2 - kontrol kelas
    txt("KELAS:", (10, 38), 0.4, (180,180,180))
    colors_text = [
        ("1/N=Normal",         CLASS_COLOR['normal']),
        ("2/R=E.Ringan",       CLASS_COLOR['etiolasi_ringan']),
        ("3/B=E.Berat",        CLASS_COLOR['etiolasi_berat']),
        ("4/E=Empty",          CLASS_COLOR['empty']),
    ]
    x_pos = 70
    for label, color in colors_text:
        txt(label, (x_pos, 38), 0.4, color)
        x_pos += len(label)*6 + 20

    # Baris 3 - aksi
    txt("AKSI:  Z=Undo  |  SPACE=Gambar Berikutnya  |  Q/ESC=Keluar", (10, 56), 0.4, (180,180,180))

    # Baris 4 - statistik
    s = stats
    stat = (f"Tersimpan:  "
            f"Normal={s['normal']}  "
            f"E.Ringan={s['etiolasi_ringan']}  "
            f"E.Berat={s['etiolasi_berat']}  "
            f"Empty={s['empty']}  "
            f"| Total={sum(s.values())}")
    txt(stat, (10, 74), 0.42, (100, 220, 100))

    # Baris 5 - status state
    if state == "PENDING":
        status = ">>>  Tekan tombol kelas untuk SIMPAN  |  ESC untuk BATAL  <<<"
        txt(status, (10, 90), 0.42, (0, 255, 255))
    elif state == "DRAWING":
        txt("Sedang menggambar area crop...", (10, 90), 0.42, (255, 255, 0))
    else:
        txt("Klik + Drag pada gambar untuk mulai crop", (10, 90), 0.42, (160, 160, 160))


# ───────────────────────────────────────────────────────────────
#  HELPER: SIMPAN CROP
# ───────────────────────────────────────────────────────────────
def save_crop(img, x1, y1, x2, y2, class_name, dataset_folder, stats, undo_stack):
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    out_dir = os.path.join(dataset_folder, class_name)
    os.makedirs(out_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"{class_name}_crop_{ts}.jpg"
    fpath = os.path.join(out_dir, fname)
    cv2.imwrite(fpath, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    stats[class_name] += 1
    undo_stack.append((fpath, class_name))
    return fpath


def count_existing(dataset_folder, stats):
    for cls in stats:
        d = os.path.join(dataset_folder, cls)
        if os.path.isdir(d):
            stats[cls] = len([f for f in os.listdir(d)
                               if f.lower().endswith(('.jpg','.jpeg','.png'))])


# ───────────────────────────────────────────────────────────────
#  MAIN
# ───────────────────────────────────────────────────────────────
def main():
    global mx_state, mx_start, mx_curr, mx_draw

    # --- Ambil argumen / input ---
    source_folder = sys.argv[1] if len(sys.argv) > 1 else ""
    if not source_folder:
        print(f"\n📂 Folder foto kamera atas")
        print(f"   (Enter = default: {DEFAULT_SOURCE}): ", end="")
        source_folder = input().strip() or DEFAULT_SOURCE

    dataset_folder = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DATASET

    # Buat folder jika belum ada
    os.makedirs(source_folder,  exist_ok=True)
    os.makedirs(dataset_folder, exist_ok=True)

    # Kumpulkan gambar
    images = []
    for ext in IMG_EXTS:
        images += glob.glob(os.path.join(source_folder, ext))
    images = sorted(set(images))

    if not images:
        print(f"\n⚠️  Tidak ada gambar di: {source_folder}")
        print("   Letakkan foto kamera ke folder tersebut, lalu jalankan ulang.\n")
        input("Tekan Enter untuk keluar...")
        return

    print(f"\n✅ {len(images)} gambar ditemukan")
    print(f"📁 Crop akan disimpan ke: {dataset_folder}\n")
    print("Tekan Enter untuk mulai...")
    input()

    stats = {'normal': 0, 'etiolasi_ringan': 0, 'etiolasi_berat': 0, 'empty': 0}
    undo_stack = []
    count_existing(dataset_folder, stats)

    # Setup OpenCV window
    WIN_MAIN    = "CROP TOOL - TOP VIEW  |  Q=Keluar  SPACE=Next"
    WIN_PREVIEW = "Preview Crop"
    cv2.namedWindow(WIN_MAIN,    cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_PREVIEW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_PREVIEW, 300, 300)
    cv2.setMouseCallback(WIN_MAIN, mouse_cb)

    img_idx    = 0
    exit_all   = False

    while img_idx < len(images) and not exit_all:
        img_path = images[img_idx]
        orig     = cv2.imread(img_path)
        if orig is None:
            print(f"⚠️  Gagal load: {img_path}")
            img_idx += 1
            continue

        img_name = os.path.basename(img_path)

        # Resize untuk display jika terlalu besar
        h, w  = orig.shape[:2]
        scale = 1.0
        if max(h, w) > MAX_DISPLAY:
            scale = MAX_DISPLAY / max(h, w)
            work  = cv2.resize(orig, (int(w*scale), int(h*scale)))
        else:
            work = orig.copy()

        # Reset state
        mx_state = "IDLE"
        mx_start = (0, 0)
        mx_curr  = (0, 0)
        mx_draw  = False
        saved_msg = ""
        saved_timer = 0

        while True:
            disp = work.copy()

            # Gambar rectangle aktif
            if mx_state in ("DRAWING", "PENDING"):
                x1, y1, x2, y2 = get_rect()
                color = (0, 255, 255) if mx_state == "PENDING" else (255, 255, 0)
                cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
                # Ukuran area
                pw, ph = x2 - x1, y2 - y1
                cv2.putText(disp, f"{pw}x{ph}px",
                            (x1+2, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, color, 1, cv2.LINE_AA)

                if mx_state == "PENDING":
                    crop_prev = work[y1:y2, x1:x2]
                    if crop_prev.size > 0:
                        prev = cv2.resize(crop_prev, (280, 280))
                        cv2.imshow(WIN_PREVIEW, prev)

            # Pesan konfirmasi simpan
            if saved_msg and saved_timer > 0:
                dh, dw = disp.shape[:2]
                cv2.putText(disp, saved_msg, (10, dh - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2, cv2.LINE_AA)
                saved_timer -= 1

            draw_overlay(disp, img_name, img_idx, len(images), stats, mx_state)
            cv2.imshow(WIN_MAIN, disp)

            key = cv2.waitKey(30) & 0xFF

            # ── KELUAR ──
            if key == ord('q'):
                exit_all = True
                break

            # ── ESC: batal crop aktif ──
            elif key == 27:
                if mx_state in ("DRAWING", "PENDING"):
                    mx_state = "IDLE"
                    cv2.destroyWindow(WIN_PREVIEW)
                    cv2.namedWindow(WIN_PREVIEW, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(WIN_PREVIEW, 300, 300)
                else:
                    exit_all = True
                    break

            # ── SPACE: gambar berikutnya ──
            elif key == ord(' '):
                break

            # ── Z: UNDO ──
            elif key == ord('z'):
                if undo_stack:
                    fpath, cls = undo_stack.pop()
                    if os.path.exists(fpath):
                        os.remove(fpath)
                        stats[cls] = max(0, stats[cls] - 1)
                        print(f"↩️  Undo: {os.path.basename(fpath)}")
                        saved_msg   = f"UNDO: {os.path.basename(fpath)}"
                        saved_timer = 40
                mx_state = "IDLE"

            # ── KUNCI KELAS (hanya saat PENDING) ──
            elif mx_state == "PENDING" and key in CLASS_KEYS:
                class_name = CLASS_KEYS[key]
                x1, y1, x2, y2 = get_rect()
                fpath = save_crop(work, x1, y1, x2, y2,
                                  class_name, dataset_folder, stats, undo_stack)
                if fpath:
                    msg = f"✅ Simpan [{class_name}]: {os.path.basename(fpath)}"
                    print(msg)
                    saved_msg   = f"SIMPAN  {class_name.upper()}  ({os.path.basename(fpath)})"
                    saved_timer = 50
                mx_state = "IDLE"

        img_idx += 1

    cv2.destroyAllWindows()

    # Statistik akhir
    print("\n" + "="*55)
    print("📊 STATISTIK AKHIR DATASET ATAS:")
    for cls, cnt in stats.items():
        bar = "█" * (cnt // 2) if cnt else ""
        print(f"   {cls:<18}: {cnt:4d}  {bar}")
    print(f"   {'TOTAL':<18}: {sum(stats.values()):4d}")
    print(f"\n📁 Tersimpan di: {dataset_folder}")
    print("="*55)
    input("\nTekan Enter untuk keluar...")


if __name__ == "__main__":
    main()

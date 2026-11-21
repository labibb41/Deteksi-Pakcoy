"""
==============================================================
  CROP TOOL - TAMPAK ATAS (TOP VIEW) - VERSI INTERAKTIF
  Dataset Labeling Tool untuk Tanaman Pakcoy
==============================================================
  Fitur:
    - Otomatis langsung membuka foto di folder 'foto_kamera_atas'
    - Kotak bibit yang sudah dicrop tetap terlihat di layar (Hijau, Oranye, Merah)
    - Next / Back antar foto dengan mudah
    - Hapus / Pindahkan foto yang sudah selesai ke folder 'selesai'
  Kontrol:
    Klik + Drag  : Gambar area crop
    1 / N        : Simpan → NORMAL (Kotak Hijau)
    2 / R        : Simpan → ETIOLASI RINGAN (Kotak Oranye)
    3 / B        : Simpan → ETIOLASI BERAT (Kotak Merah)
    4 / E        : Simpan → EMPTY (Kotak Abu-abu)
    Z            : Undo crop terakhir
    SPACE / D / →: Foto Berikutnya (Next)
    A / ←        : Foto Sebelumnya (Back)
    X / DEL      : Hapus / Pindahkan foto saat ini ke folder 'selesai'
    ESC          : Batal kotak aktif
    Q            : Keluar program
==============================================================
"""

import os
import sys

# ── AUTO-RELAUNCH DENGAN C:\penv ──
PENV_PYTHON = r"C:\penv\Scripts\python.exe"
if os.path.exists(PENV_PYTHON) and os.path.normpath(sys.executable).lower() != os.path.normpath(PENV_PYTHON).lower():
    import subprocess
    sys.exit(subprocess.call([PENV_PYTHON] + sys.argv))

import cv2
import numpy as np
import glob
import shutil
from datetime import datetime

# ───────────────────────────────────────────────────────────────
#  KONFIGURASI
# ───────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCE  = os.path.join(BASE_DIR, "foto_kamera_atas")
DEFAULT_DATASET = os.path.join(BASE_DIR, "dataset_baru_atas")
SELESAI_DIR     = os.path.join(DEFAULT_SOURCE, "selesai")

CLASS_KEYS = {
    ord('1'): 'normal',         ord('n'): 'normal', ord('N'): 'normal',
    ord('2'): 'etiolasi_ringan', ord('r'): 'etiolasi_ringan', ord('R'): 'etiolasi_ringan',
    ord('3'): 'etiolasi_berat',  ord('b'): 'etiolasi_berat',  ord('B'): 'etiolasi_berat',
    ord('4'): 'empty',           ord('e'): 'empty',           ord('E'): 'empty',
}

CLASS_COLOR = {           # BGR
    'normal':           (50,  220, 50),   # Hijau
    'etiolasi_ringan':  (0,   165, 255),  # Oranye
    'etiolasi_berat':   (30,  30,  230),  # Merah
    'empty':            (150, 150, 150),  # Abu-abu
}

CLASS_LABEL = {
    'normal':           '1/N  NORMAL',
    'etiolasi_ringan':  '2/R  E.RINGAN',
    'etiolasi_berat':   '3/B  E.BERAT',
    'empty':            '4/E  EMPTY',
}

IMG_EXTS   = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')
MIN_PX     = 15           # ukuran crop minimum (pixels)
MAX_DISPLAY= 1050         # panjang sisi maksimal tampilan

# ───────────────────────────────────────────────────────────────
#  STATE MOUSE
# ───────────────────────────────────────────────────────────────
mx_state = "IDLE"  # IDLE | DRAWING | PENDING
mx_start = (0, 0)
mx_curr  = (0, 0)
mx_draw  = False

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
#  OVERLAY HEADER
# ───────────────────────────────────────────────────────────────
def draw_overlay(disp, img_name, idx, total, stats, state, n_crops_current, confirm_delete=False):
    h, w = disp.shape[:2]
    bar_h = 100

    overlay = disp.copy()
    bg_color = (20, 20, 60) if confirm_delete else (22, 22, 22)
    cv2.rectangle(overlay, (0, 0), (w, bar_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.85, disp, 0.15, 0, disp)
    cv2.line(disp, (0, bar_h), (w, bar_h), (0, 180, 255) if not confirm_delete else (0, 0, 255), 2)

    def txt(text, pos, scale=0.42, color=(220, 220, 220), thick=1):
        cv2.putText(disp, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    if confirm_delete:
        txt(f"⚠️ KONFIRMASI: HAPUS / PINDAHKAN FILE INI?", (12, 26), 0.55, (0, 220, 255), 2)
        txt(f"File: {img_name}", (12, 54), 0.44, (255, 255, 255), 1)
        txt("Tekan [Y] = Pindahkan ke 'selesai'  |  [N] atau [ESC] = Batal", (12, 84), 0.48, (50, 255, 50), 2)
        return

    # Baris 1: Info Gambar & Riwayat Crop
    txt(f"[{idx+1}/{total}]  {img_name}", (12, 22), 0.50, (255, 255, 255), 1)
    txt(f"Foto ini: {n_crops_current} bibit dicrop", (w - 230, 22), 0.45, (0, 240, 255), 1)

    # Baris 2: Shortcut Kelas
    txt("LABEL KELAS:", (12, 44), 0.40, (180, 180, 180))
    colors_text = [
        ("1/N: Normal",     CLASS_COLOR['normal']),
        ("2/R: E.Ringan",   CLASS_COLOR['etiolasi_ringan']),
        ("3/B: E.Berat",    CLASS_COLOR['etiolasi_berat']),
        ("4/E: Empty",      CLASS_COLOR['empty']),
    ]
    x_pos = 115
    for label, color in colors_text:
        txt(label, (x_pos, 44), 0.42, color, 1)
        x_pos += len(label) * 8 + 18

    # Baris 3: Navigasi & Aksi
    txt("NAVIGASI:  [SPACE/D] Next  |  [A] Back  |  [X/DEL] Hapus/Pindah  |  [Z] Undo  |  [Q] Keluar", 
        (12, 68), 0.40, (200, 200, 200))

    # Baris 4: Status Saat Ini
    if state == "PENDING":
        txt(">>>  PILIH KELAS: Tekan [1/N], [2/R], [3/B], atau [4/E] untuk simpan  (ESC=Batal)  <<<", 
            (12, 90), 0.44, (0, 255, 255), 2)
    elif state == "DRAWING":
        txt("Sedang menarik kotak area tanaman...", (12, 90), 0.42, (255, 255, 0))
    else:
        s = stats
        txt(f"Total Dataset Baru: Normal={s['normal']} | E.Ringan={s['etiolasi_ringan']} | E.Berat={s['etiolasi_berat']} | Empty={s['empty']}", 
            (12, 90), 0.40, (130, 230, 130))


# ───────────────────────────────────────────────────────────────
#  SIMPAN HASIL CROP
# ───────────────────────────────────────────────────────────────
def save_crop_hires(orig_img, scale, x1, y1, x2, y2, class_name, dataset_folder, stats, undo_stack):
    # Mapping koordinat display ke koordinat asli beresolusi tinggi
    ox1 = int(round(x1 / scale))
    oy1 = int(round(y1 / scale))
    ox2 = int(round(x2 / scale))
    oy2 = int(round(y2 / scale))

    ox1 = max(0, min(orig_img.shape[1], ox1))
    ox2 = max(0, min(orig_img.shape[1], ox2))
    oy1 = max(0, min(orig_img.shape[0], oy1))
    oy2 = max(0, min(orig_img.shape[0], oy2))

    crop = orig_img[oy1:oy2, ox1:ox2]
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return None

    out_dir = os.path.join(dataset_folder, class_name)
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"{class_name}_crop_{ts}.jpg"
    fpath = os.path.join(out_dir, fname)

    cv2.imwrite(fpath, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    stats[class_name] += 1
    item = (fpath, class_name, x1, y1, x2, y2)
    undo_stack.append(item)
    return item


def count_existing(dataset_folder, stats):
    for cls in stats:
        d = os.path.join(dataset_folder, cls)
        if os.path.isdir(d):
            stats[cls] = len([f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])


# ───────────────────────────────────────────────────────────────
#  PROGRAM UTAMA
# ───────────────────────────────────────────────────────────────
def main():
    global mx_state, mx_start, mx_curr, mx_draw

    source_folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
    dataset_folder = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DATASET

    os.makedirs(source_folder, exist_ok=True)
    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(SELESAI_DIR, exist_ok=True)

    print("\n" + "="*65)
    print("  🌿 CROP TOOL - TAMPAK ATAS PAKCOY (TOP-VIEW)")
    print("="*65)
    print(f"📂 Folder Sumber Foto : {source_folder}")
    print(f"📁 Target Simpan Crop : {dataset_folder}")

    # Kumpulkan gambar
    images = []
    for ext in IMG_EXTS:
        images += glob.glob(os.path.join(source_folder, ext))
    images = sorted(set(images))

    if not images:
        print(f"\n⚠️  Tidak ada gambar di: {source_folder}")
        print("   Letakkan foto kamera atas Anda ke folder tersebut, lalu jalankan program ini lagi.")
        input("\nTekan Enter untuk keluar...")
        return

    print(f"✅ Ditemukan {len(images)} gambar siap dilabeli!")

    stats = {'normal': 0, 'etiolasi_ringan': 0, 'etiolasi_berat': 0, 'empty': 0}
    count_existing(dataset_folder, stats)

    # Riwayat crop per gambar: {img_path: [(fpath, class_name, x1, y1, x2, y2)]}
    labeled_history = {}

    WIN_MAIN = "CROP TOOL PAKCOY - TOP VIEW"
    WIN_PREV = "Preview Crop"
    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_PREV, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_PREV, 260, 260)
    cv2.setMouseCallback(WIN_MAIN, mouse_cb)

    img_idx = 0
    saved_msg = ""
    saved_timer = 0
    confirm_delete = False

    while 0 <= img_idx < len(images):
        img_path = images[img_idx]
        orig = cv2.imread(img_path)
        if orig is None:
            print(f"⚠️  Gagal membuka: {img_path}")
            img_idx += 1
            continue

        img_name = os.path.basename(img_path)
        if img_path not in labeled_history:
            labeled_history[img_path] = []

        # Resize untuk layar tampilan
        h, w = orig.shape[:2]
        scale = 1.0
        if max(h, w) > MAX_DISPLAY:
            scale = MAX_DISPLAY / max(h, w)
            work = cv2.resize(orig, (int(w * scale), int(h * scale)))
        else:
            work = orig.copy()

        # Ukuran awal jendela
        disp_h, disp_w = work.shape[:2]
        cv2.resizeWindow(WIN_MAIN, disp_w, disp_h)

        # Reset state mouse
        mx_state = "IDLE"
        mx_start = (0, 0)
        mx_curr = (0, 0)
        mx_draw = False
        confirm_delete = False

        while True:
            disp = work.copy()

            # 1. Gambar semua kotak yang SUDAH dicrop pada gambar ini
            for crop_item in labeled_history[img_path]:
                _, c_cls, cx1, cy1, cx2, cy2 = crop_item
                col = CLASS_COLOR.get(c_cls, (255, 255, 255))
                cv2.rectangle(disp, (cx1, cy1), (cx2, cy2), col, 2)
                # Label kecil
                lbl_txt = c_cls[:6].upper()
                cv2.rectangle(disp, (cx1, max(0, cy1 - 15)), (cx1 + len(lbl_txt)*8 + 4, cy1), (15, 15, 15), -1)
                cv2.putText(disp, lbl_txt, (cx1 + 2, cy1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)

            # 2. Gambar kotak yang SEDANG ditarik mouse
            if mx_state in ("DRAWING", "PENDING"):
                x1, y1, x2, y2 = get_rect()
                color = (0, 255, 255) if mx_state == "PENDING" else (255, 255, 0)
                cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
                pw, ph = abs(x2 - x1), abs(y2 - y1)
                cv2.putText(disp, f"{pw}x{ph}px", (x1 + 2, max(12, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

                # Preview crop di jendela kecil
                if mx_state == "PENDING":
                    crop_prev = work[y1:y2, x1:x2]
                    if crop_prev.size > 0:
                        prev_resized = cv2.resize(crop_prev, (240, 240))
                        cv2.imshow(WIN_PREV, prev_resized)

            # 3. Pesan notifikasi sementara di pojok bawah
            if saved_msg and saved_timer > 0:
                dh = disp.shape[0]
                cv2.putText(disp, saved_msg, (12, dh - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2, cv2.LINE_AA)
                saved_timer -= 1

            # 4. Header Bar
            n_crops = len(labeled_history[img_path])
            draw_overlay(disp, img_name, img_idx, len(images), stats, mx_state, n_crops, confirm_delete)

            cv2.imshow(WIN_MAIN, disp)
            key = cv2.waitKey(25) & 0xFF

            # ── A. MODE KONFIRMASI HAPUS FOTO ──
            if confirm_delete:
                if key in (ord('y'), ord('Y')):
                    # Pindahkan foto ke folder 'selesai' (atau hapus)
                    target_selesai = os.path.join(SELESAI_DIR, img_name)
                    try:
                        shutil.move(img_path, target_selesai)
                        print(f"📦 Foto dipindahkan ke: {target_selesai}")
                        saved_msg = f"FOTO DIPINDAHKAN KE 'selesai': {img_name}"
                        saved_timer = 50
                    except Exception as e:
                        try:
                            os.remove(img_path)
                            print(f"🗑️  Foto dihapus: {img_path}")
                        except:
                            pass
                    
                    images.pop(img_idx)
                    if img_idx >= len(images):
                        img_idx = max(0, len(images) - 1)
                    break
                elif key in (ord('n'), ord('N'), 27):
                    confirm_delete = False
                    saved_msg = "Batal menghapus foto."
                    saved_timer = 30
                continue

            # ── B. KONTROL REGULER ──
            if key in (27, ord('q'), ord('Q')):
                if mx_state in ("DRAWING", "PENDING"):
                    mx_state = "IDLE"
                else:
                    img_idx = len(images)  # Keluar
                    break

            # NEXT IMAGE: SPACE, D, atau Panah Kanan (83 / 2555904)
            elif key in (ord(' '), ord('d'), ord('D'), 83, 2555904):
                img_idx += 1
                break

            # PREV IMAGE: A atau Panah Kiri (81 / 2424832)
            elif key in (ord('a'), ord('A'), 81, 2424832):
                if img_idx > 0:
                    img_idx -= 1
                    break
                else:
                    saved_msg = "Ini adalah foto pertama."
                    saved_timer = 30

            # HAPUS / PINDAHKAN FOTO: X atau DELETE
            elif key in (ord('x'), ord('X'), 127, 255, 3014656):
                confirm_delete = True

            # UNDO CROP TERAKHIR: Z
            elif key in (ord('z'), ord('Z')):
                if labeled_history[img_path]:
                    fpath, cls, _, _, _, _ = labeled_history[img_path].pop()
                    if os.path.exists(fpath):
                        os.remove(fpath)
                    stats[cls] = max(0, stats[cls] - 1)
                    saved_msg = f"UNDO CROP: {os.path.basename(fpath)}"
                    saved_timer = 40
                else:
                    saved_msg = "Tidak ada crop pada foto ini yang bisa di-undo."
                    saved_timer = 30
                mx_state = "IDLE"

            # SIMPAN KELAS SAAT PENDING (1, 2, 3, 4 atau N, R, B, E)
            elif mx_state == "PENDING" and key in CLASS_KEYS:
                class_name = CLASS_KEYS[key]
                x1, y1, x2, y2 = get_rect()
                item = save_crop_hires(orig, scale, x1, y1, x2, y2,
                                       class_name, dataset_folder, stats, labeled_history[img_path])
                if item:
                    fpath = item[0]
                    saved_msg = f"TERSIMPAN: [{class_name.upper()}] {os.path.basename(fpath)}"
                    saved_timer = 50
                    print(f"✅ [{class_name}] {os.path.basename(fpath)}")
                mx_state = "IDLE"

    cv2.destroyAllWindows()

    print("\n" + "="*55)
    print("📊 REKAPITULASI DATASET BARU:")
    for cls, cnt in stats.items():
        print(f"   👉 {cls:<18}: {cnt:4d} gambar")
    print(f"   👉 {'TOTAL':<18}: {sum(stats.values()):4d} gambar")
    print(f"\n📁 Lokasi Dataset : {dataset_folder}")
    print("="*55)
    print("🎉 Selesai! Anda bisa menjalankan 'update_dataset.py' untuk menggabungkannya ke dataset utama.")


if __name__ == "__main__":
    main()

"""
==============================================================
  CROP TOOL - TAMPAK ATAS (TOP VIEW) - DENGAN ZOOM & ANTI-MELEBAR
  Dataset Labeling Tool untuk Tanaman Pakcoy
==============================================================
  Fitur Utama:
    - ANTI MELEBAR / GEPENG: Aspek rasio gambar terkunci 100% proporsional.
    - ZOOM & PAN INTERAKTIF:
        * Scroll Roda Mouse      : Zoom In / Zoom Out ke posisi kursor
        * Klik KANAN Tahan & Geser: Geser (Pan) tampilan saat di-zoom
        * Double Click / [R]     : Reset Zoom kembali ke tampilan penuh
        * Tombol Panah           : Geser tampilan
    - CROP & LABEL:
        * Klik KIRI Tahan & Geser : Tarik kotak crop pada bibit
        * 1 / N                  : Simpan -> NORMAL (Kotak Hijau)
        * 2 / R                  : Simpan -> ETIOLASI RINGAN (Kotak Oranye)
        * 3 / B                  : Simpan -> ETIOLASI BERAT (Kotak Merah)
        * 4 / E                  : Simpan -> EMPTY (Kotak Abu-abu)
        * Z                      : Undo crop terakhir
    - NAVIGASI & ARSIP:
        * SPACE / D / ->         : Foto Berikutnya (Next)
        * A / <-                 : Foto Sebelumnya (Back)
        * X / DEL                : Hapus / Pindahkan foto ini ke 'selesai'
        * Q                      : Keluar program
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
    'normal':           (50,  225, 50),   # Hijau
    'etiolasi_ringan':  (0,   165, 255),  # Oranye
    'etiolasi_berat':   (30,  30,  235),  # Merah
    'empty':            (150, 150, 150),  # Abu-abu
}

IMG_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')
MIN_PX   = 12             # Ukuran crop minimum (pixels)

# ───────────────────────────────────────────────────────────────
#  STATE GLOBAL
# ───────────────────────────────────────────────────────────────
state = {
    # Tampilan Canvas & Viewport
    "zoom": 1.0,
    "center_x": 0.5,
    "center_y": 0.5,
    "canvas_scale": 1.0,
    "canvas_ox": 0,
    "canvas_oy": 0,
    "canvas_nw": 800,
    "canvas_nh": 800,
    "roi_x1": 0,
    "roi_y1": 0,
    "roi_x2": 800,
    "roi_y2": 800,
    "orig_w": 800,
    "orig_h": 800,
    # Drag Pan (Klik Kanan)
    "is_panning": False,
    "pan_start_x": 0,
    "pan_start_y": 0,
    "pan_start_cx": 0.5,
    "pan_start_cy": 0.5,
    # Crop Drawing (Klik Kiri)
    "crop_state": "IDLE",  # IDLE | DRAWING | PENDING
    "crop_start_img": (0, 0),
    "crop_curr_img": (0, 0),
    "is_drawing_crop": False,
}


def canvas_to_img_coords(mx, my):
    """Konversi koordinat kursor mouse pada canvas ke koordinat asli gambar."""
    ox = state["canvas_ox"]
    oy = state["canvas_oy"]
    scale = state["canvas_scale"]
    rx1 = state["roi_x1"]
    ry1 = state["roi_y1"]
    ow = state["orig_w"]
    oh = state["orig_h"]

    if scale <= 0:
        return 0, 0

    ix = int(rx1 + (mx - ox) / scale)
    iy = int(ry1 + (my - oy) / scale)
    ix = int(np.clip(ix, 0, ow))
    iy = int(np.clip(iy, 0, oh))
    return ix, iy


def mouse_cb(event, mx, my, flags, param):
    global state

    # 1. Roda Mouse -> Zoom In / Zoom Out
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            state["zoom"] = min(8.0, state["zoom"] * 1.25)
        else:
            state["zoom"] = max(1.0, state["zoom"] / 1.25)

        # Geser titik fokus ke arah kursor jika sedang di-zoom
        if state["zoom"] > 1.0:
            ix, iy = canvas_to_img_coords(mx, my)
            state["center_x"] = np.clip(ix / max(1, state["orig_w"]), 0.0, 1.0)
            state["center_y"] = np.clip(iy / max(1, state["orig_h"]), 0.0, 1.0)
        else:
            state["center_x"] = 0.5
            state["center_y"] = 0.5

    # 2. Double Click -> Reset Zoom
    elif event == cv2.EVENT_LBUTTONDBLCLK or event == cv2.EVENT_RBUTTONDBLCLK:
        state["zoom"] = 1.0
        state["center_x"] = 0.5
        state["center_y"] = 0.5

    # 3. Klik Kanan -> Batal Crop / Drag Pan (Menggeser Tampilan saat Zoom)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if state["crop_state"] in ("PENDING", "DRAWING"):
            state["crop_state"] = "IDLE"
            state["is_drawing_crop"] = False
            return
        state["is_panning"] = True
        state["pan_start_x"] = mx
        state["pan_start_y"] = my
        state["pan_start_cx"] = state["center_x"]
        state["pan_start_cy"] = state["center_y"]

    elif event == cv2.EVENT_MOUSEMOVE and state["is_panning"]:
        scale = max(1e-4, state["canvas_scale"])
        dx = mx - state["pan_start_x"]
        dy = my - state["pan_start_y"]
        delta_cx = (dx / scale) / max(1, state["orig_w"])
        delta_cy = (dy / scale) / max(1, state["orig_h"])
        state["center_x"] = np.clip(state["pan_start_cx"] - delta_cx, 0.0, 1.0)
        state["center_y"] = np.clip(state["pan_start_cy"] - delta_cy, 0.0, 1.0)

    elif event == cv2.EVENT_RBUTTONUP:
        state["is_panning"] = False

    # 4. Klik Kiri -> Gambar Kotak Crop (Klik 1x di luar untuk batalkan)
    elif event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = canvas_to_img_coords(mx, my)
        state["is_drawing_crop"] = True
        state["crop_start_img"] = (ix, iy)
        state["crop_curr_img"] = (ix, iy)
        state["crop_state"] = "DRAWING"

    elif event == cv2.EVENT_MOUSEMOVE and state["is_drawing_crop"]:
        ix, iy = canvas_to_img_coords(mx, my)
        state["crop_curr_img"] = (ix, iy)

    elif event == cv2.EVENT_LBUTTONUP and state["is_drawing_crop"]:
        ix, iy = canvas_to_img_coords(mx, my)
        state["crop_curr_img"] = (ix, iy)
        state["is_drawing_crop"] = False
        
        sx, sy = state["crop_start_img"]
        w = abs(ix - sx)
        h = abs(iy - sy)
        if w >= MIN_PX and h >= MIN_PX:
            state["crop_state"] = "PENDING"
        else:
            # Jika hanya klik 1x tanpa drag -> batalkan kotak yang aktif!
            state["crop_state"] = "IDLE"


def get_current_crop_rect():
    """Dapatkan koordinat bounding box crop dalam skala gambar asli."""
    sx, sy = state["crop_start_img"]
    cx, cy = state["crop_curr_img"]
    x1 = min(sx, cx)
    y1 = min(sy, cy)
    x2 = max(sx, cx)
    y2 = max(sy, cy)
    return x1, y1, x2, y2


# ───────────────────────────────────────────────────────────────
#  HEADER BAR TAMPILAN
# ───────────────────────────────────────────────────────────────
def draw_header_bar(canvas, img_name, idx, total, stats, crop_state, n_crops_current, confirm_delete):
    win_w = canvas.shape[1]
    bar_h = 88

    overlay = canvas.copy()
    bg_color = (25, 25, 75) if confirm_delete else (20, 20, 20)
    cv2.rectangle(overlay, (0, 0), (win_w, bar_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.90, canvas, 0.10, 0, canvas)
    border_col = (0, 70, 255) if confirm_delete else (0, 185, 255)
    cv2.line(canvas, (0, bar_h), (win_w, bar_h), border_col, 2)

    def txt(text, pos, scale=0.42, color=(220, 220, 220), thick=1):
        cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    if confirm_delete:
        txt("[PERINGATAN] HAPUS / PINDAHKAN FILE INI?", (14, 26), 0.54, (0, 225, 255), 2)
        txt(f"File: {img_name}", (14, 52), 0.44, (255, 255, 255), 1)
        txt("Tekan [Y] = Pindahkan ke 'selesai'   |   [N] atau [ESC] = Batal", (14, 78), 0.46, (50, 255, 50), 2)
        return

    # Baris 1: Info Gambar & Zoom
    zoom_txt = f"ZOOM: {state['zoom']:.1f}x"
    txt(f"[{idx+1}/{total}]  {img_name}   |   {zoom_txt}", (14, 22), 0.48, (255, 255, 255), 1)
    txt(f"Foto ini: {n_crops_current} bibit dicrop", (win_w - 240, 22), 0.44, (0, 240, 255), 1)

    # Baris 2: Tombol Kelas
    txt("SIMPAN KELAS:", (14, 44), 0.40, (180, 180, 180))
    colors_text = [
        ("1/N: Normal",     CLASS_COLOR['normal']),
        ("2/R: E.Ringan",   CLASS_COLOR['etiolasi_ringan']),
        ("3/B: E.Berat",    CLASS_COLOR['etiolasi_berat']),
        ("4/E: Empty",      CLASS_COLOR['empty']),
    ]
    x_pos = 125
    for label, color in colors_text:
        txt(label, (x_pos, 44), 0.42, color, 1)
        x_pos += len(label) * 8 + 20

    # Baris 3: Status / Panduan
    if crop_state == "PENDING":
        txt(">>> SIMPAN: [1/N], [2/R], [3/B], [4/E]   |   BATAL: [ESC], [C], [Klik Kanan], atau [Klik 1x di luar] <<<", 
            (14, 72), 0.43, (0, 255, 255), 2)
    elif crop_state == "DRAWING":
        txt("Sedang menarik kotak crop bibit...", (14, 72), 0.42, (255, 255, 0))
    else:
        txt("KONTROL: [Klik Kiri] Tarik Crop  |  [Scroll] Zoom  |  [Klik Kanan/Panah] Geser  |  [SPACE/D] Next  |  [X] Selesai", 
            (14, 72), 0.38, (190, 190, 190))


# ───────────────────────────────────────────────────────────────
#  SIMPAN HASIL CROP
# ───────────────────────────────────────────────────────────────
def save_crop_to_disk(orig_img, x1, y1, x2, y2, class_name, dataset_folder, stats, current_crops):
    crop = orig_img[y1:y2, x1:x2]
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
    current_crops.append(item)
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
    global state

    source_folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
    dataset_folder = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DATASET

    os.makedirs(source_folder, exist_ok=True)
    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(SELESAI_DIR, exist_ok=True)

    print("\n" + "="*65)
    print("  🌿 CROP TOOL TAMPAK ATAS (DENGAN ZOOM & ANTI-MELEBAR)")
    print("="*65)
    print(f"📂 Folder Sumber Foto : {source_folder}")
    print(f"📁 Target Simpan Crop : {dataset_folder}")

    # Pindai gambar
    images = []
    for ext in IMG_EXTS:
        images += glob.glob(os.path.join(source_folder, ext))
    images = sorted(set(images))

    if not images:
        print(f"\n⚠️  Tidak ada gambar di: {source_folder}")
        print("   Letakkan foto kamera atas Anda ke folder tersebut, lalu jalankan program ini lagi.")
        input("\nTekan Enter untuk keluar...")
        return

    print(f"✅ Ditemukan {len(images)} foto kamera atas siap dilabeli!")

    stats = {'normal': 0, 'etiolasi_ringan': 0, 'etiolasi_berat': 0, 'empty': 0}
    count_existing(dataset_folder, stats)

    # Riwayat crop per gambar
    labeled_history = {}

    WIN_MAIN = "CROP TOOL PAKCOY - TOP VIEW"
    WIN_PREV = "Preview Crop"
    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_PREV, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_PREV, 260, 260)
    cv2.setMouseCallback(WIN_MAIN, mouse_cb)

    # Ukuran jendela default yang nyaman
    disp_w, disp_h = 1000, 750
    cv2.resizeWindow(WIN_MAIN, disp_w, disp_h)

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

        orig_h, orig_w = orig.shape[:2]
        state["orig_w"] = orig_w
        state["orig_h"] = orig_h

        # Reset zoom saat berganti foto
        state["zoom"] = 1.0
        state["center_x"] = 0.5
        state["center_y"] = 0.5
        state["crop_state"] = "IDLE"
        confirm_delete = False

        while True:
            # Dapatkan ukuran jendela aktif secara dinamis
            rect = cv2.getWindowImageRect(WIN_MAIN)
            win_w = rect[2] if (rect is not None and rect[2] > 100) else disp_w
            win_h = rect[3] if (rect is not None and rect[3] > 100) else disp_h
            bar_h = 88

            # 1. Hitung Region of Interest (ROI) sesuai Zoom & Pan
            zoom = state["zoom"]
            crop_w = int(orig_w / zoom)
            crop_h = int(orig_h / zoom)

            cx = state["center_x"]
            cy = state["center_y"]

            rx1 = int(np.clip(cx * orig_w - crop_w / 2, 0, max(0, orig_w - crop_w)))
            ry1 = int(np.clip(cy * orig_h - crop_h / 2, 0, max(0, orig_h - crop_h)))
            rx2 = min(orig_w, rx1 + crop_w)
            ry2 = min(orig_h, ry1 + crop_h)

            state["roi_x1"] = rx1
            state["roi_y1"] = ry1
            state["roi_x2"] = rx2
            state["roi_y2"] = ry2

            # Potong area gambar sesuai zoom
            roi_img = orig[ry1:ry2, rx1:rx2].copy()

            # 2. Letterboxing Presisi: Kunci Aspek Rasio (TIDAK MELEBAR / GEPENG!)
            rw = rx2 - rx1
            rh = ry2 - ry1
            avail_h = max(10, win_h - bar_h)
            scale = min(win_w / max(1, rw), avail_h / max(1, rh))

            nw = max(1, int(rw * scale))
            nh = max(1, int(rh * scale))
            resized_roi = cv2.resize(roi_img, (nw, nh), interpolation=cv2.INTER_LINEAR)

            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            ox = (win_w - nw) // 2
            oy = bar_h + (avail_h - nh) // 2
            canvas[oy:oy+nh, ox:ox+nw] = resized_roi

            # Simpan koordinat canvas
            state["canvas_scale"] = scale
            state["canvas_ox"] = ox
            state["canvas_oy"] = oy
            state["canvas_nw"] = nw
            state["canvas_nh"] = nh

            # 3. Gambar kotak bibit yang SUDAH dicrop sebelumnya
            for crop_item in labeled_history[img_path]:
                _, c_cls, bx1, by1, bx2, by2 = crop_item
                # Konversi koordinat asli gambar ke koordinat canvas
                cx1 = int(ox + (bx1 - rx1) * scale)
                cy1 = int(oy + (by1 - ry1) * scale)
                cx2 = int(ox + (bx2 - rx1) * scale)
                cy2 = int(oy + (by2 - ry1) * scale)

                # Gambar hanya jika berada dalam area pandang canvas
                col = CLASS_COLOR.get(c_cls, (255, 255, 255))
                cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), col, 2)
                lbl_txt = c_cls[:6].upper()
                cv2.rectangle(canvas, (cx1, max(oy, cy1 - 14)), (cx1 + len(lbl_txt)*7 + 4, cy1), (15, 15, 15), -1)
                cv2.putText(canvas, lbl_txt, (cx1 + 2, cy1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)

            # 4. Gambar kotak crop yang SEDANG ditarik / PENDING
            if state["crop_state"] in ("DRAWING", "PENDING"):
                bx1, by1, bx2, by2 = get_current_crop_rect()
                cx1 = int(ox + (bx1 - rx1) * scale)
                cy1 = int(oy + (by1 - ry1) * scale)
                cx2 = int(ox + (bx2 - rx1) * scale)
                cy2 = int(oy + (by2 - ry1) * scale)

                color = (0, 255, 255) if state["crop_state"] == "PENDING" else (255, 255, 0)
                cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), color, 2)
                pw, ph = abs(bx2 - bx1), abs(by2 - by1)
                cv2.putText(canvas, f"{pw}x{ph}px", (cx1 + 2, max(oy + 12, cy1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

                # Update preview crop di jendela kecil
                if state["crop_state"] == "PENDING":
                    crop_prev = orig[by1:by2, bx1:bx2]
                    if crop_prev.size > 0:
                        prev_resized = cv2.resize(crop_prev, (240, 240))
                        cv2.imshow(WIN_PREV, prev_resized)

            # 5. Pesan notifikasi sementara di bawah
            if saved_msg and saved_timer > 0:
                cv2.putText(canvas, saved_msg, (16, win_h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 120), 2, cv2.LINE_AA)
                saved_timer -= 1

            # 6. Header Bar di Bagian Atas
            n_crops = len(labeled_history[img_path])
            draw_header_bar(canvas, img_name, img_idx, len(images), stats, state["crop_state"], n_crops, confirm_delete)

            cv2.imshow(WIN_MAIN, canvas)
            key = cv2.waitKey(25) & 0xFF

            # ── A. MODE KONFIRMASI HAPUS FOTO ──
            if confirm_delete:
                if key in (ord('y'), ord('Y')):
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

            # ── B. KONTROL UTAMA ──
            # BATALKAN KOTAK CROP AKTIF: ESC, C, atau Backspace (8)
            if key in (27, ord('c'), ord('C'), 8):
                if state["crop_state"] in ("DRAWING", "PENDING"):
                    state["crop_state"] = "IDLE"
                    saved_msg = "Crop dibatalkan (tidak disimpan)"
                    saved_timer = 30
                elif key == 27:
                    # Jika tidak ada kotak aktif dan tekan ESC -> Keluar
                    img_idx = len(images)
                    break

            # KELUAR PROGRAM: Q
            elif key in (ord('q'), ord('Q')):
                img_idx = len(images)
                break

            # NEXT FOTO: SPACE, D, atau Panah Kanan
            elif key in (ord(' '), ord('d'), ord('D'), 83, 2555904):
                img_idx += 1
                break

            # BACK FOTO: A atau Panah Kiri
            elif key in (ord('a'), ord('A'), 81, 2424832):
                if img_idx > 0:
                    img_idx -= 1
                    break
                else:
                    saved_msg = "Ini adalah foto pertama."
                    saved_timer = 30

            # HAPUS / PINDAHKAN: X atau DEL
            elif key in (ord('x'), ord('X'), 127, 255, 3014656):
                confirm_delete = True

            # RESET ZOOM: R atau 0
            elif key in (ord('r'), ord('R'), ord('0')):
                state["zoom"] = 1.0
                state["center_x"] = 0.5
                state["center_y"] = 0.5
                saved_msg = "Zoom di-reset (1.0x)"
                saved_timer = 30

            # GESER PAN DENGAN TOMBOL PANAH
            elif key == 82 or key == 2490368: # Panah Atas
                state["center_y"] = max(0.0, state["center_y"] - 0.05 / state["zoom"])
            elif key == 84 or key == 2621440: # Panah Bawah
                state["center_y"] = min(1.0, state["center_y"] + 0.05 / state["zoom"])
            elif key == 81 or key == 2424832: # Panah Kiri
                state["center_x"] = max(0.0, state["center_x"] - 0.05 / state["zoom"])
            elif key == 83 or key == 2555904: # Panah Kanan
                state["center_x"] = min(1.0, state["center_x"] + 0.05 / state["zoom"])

            # UNDO CROP TERAKHIR: Z
            elif key in (ord('z'), ord('Z')):
                if labeled_history[img_path]:
                    fpath, cls, _, _, _, _ = labeled_history[img_path].pop()
                    if os.path.exists(fpath):
                        os.remove(fpath)
                    stats[cls] = max(0, stats[cls] - 1)
                    saved_msg = f"UNDO: {os.path.basename(fpath)}"
                    saved_timer = 40
                else:
                    saved_msg = "Tidak ada crop pada foto ini yang bisa di-undo."
                    saved_timer = 30
                state["crop_state"] = "IDLE"

            # SIMPAN CROP SAAT PENDING (1, 2, 3, 4 atau N, R, B, E)
            elif state["crop_state"] == "PENDING" and key in CLASS_KEYS:
                class_name = CLASS_KEYS[key]
                bx1, by1, bx2, by2 = get_current_crop_rect()
                item = save_crop_to_disk(orig, bx1, by1, bx2, by2,
                                        class_name, dataset_folder, stats, labeled_history[img_path])
                if item:
                    fpath = item[0]
                    saved_msg = f"TERSIMPAN: [{class_name.upper()}] {os.path.basename(fpath)}"
                    saved_timer = 50
                    print(f"✅ [{class_name}] {os.path.basename(fpath)}")
                state["crop_state"] = "IDLE"

    cv2.destroyAllWindows()

    print("\n" + "="*55)
    print("📊 REKAPITULASI DATASET BARU:")
    for cls, cnt in stats.items():
        print(f"   👉 {cls:<18}: {cnt:4d} gambar")
    print(f"   👉 {'TOTAL':<18}: {sum(stats.values()):4d} gambar")
    print(f"\n📁 Lokasi Dataset : {dataset_folder}")
    print("="*55)
    print("🎉 Selesai! Jalankan 'update_dataset.py' jika ingin menggabungkannya ke dataset utama.")


if __name__ == "__main__":
    main()

"""
==============================================================
  CROP TOOL - TAMPAK ATAS (TOP VIEW) - DENGAN ZOOM & ANTI-MELEBAR
  Dataset Labeling Tool untuk Tanaman Pakcoy
==============================================================
  Fitur Utama:
    - ANTI MELEBAR / GEPENG: Aspek rasio gambar terkunci 100% proporsional.
    - ZOOM & PAN INTERAKTIF:
        * Scroll Roda Mouse       : Zoom In / Zoom Out ke posisi kursor
        * Klik KANAN Tahan & Geser : Geser (Pan) tampilan saat di-zoom
        * Double Click / [0]      : Reset Zoom kembali ke tampilan penuh
        * Tombol Panah            : Geser tampilan
    - CROP & LABEL (TOMBOL / KLIK LAYAR):
        * Klik KIRI Tahan & Geser  : Tarik kotak crop pada bibit
        * 1 / N / Numpad 1 / Klik  : Simpan -> NORMAL (Kotak Hijau)
        * 2 / R / Numpad 2 / Klik  : Simpan -> ETIOLASI RINGAN (Kotak Oranye)
        * 3 / B / Numpad 3 / Klik  : Simpan -> ETIOLASI BERAT (Kotak Merah)
        * 4 / E / Numpad 4 / Klik  : Simpan -> EMPTY / KOSONG (Kotak Abu-abu)
        * ESC / C / Klik Kanan    : Batalkan kotak crop aktif
        * Z                        : Undo crop terakhir
    - JEJAK KOTAK PERSISTEN:
        * Setiap bibit yang sudah dicrop tetap meninggalkan kotak berwarna dan nomor (#1, #2, dst).
        * Tersimpan ke file riwayat (crops_history.json) sehingga tidak hilang saat ganti foto atau buka ulang.
    - NAVIGASI & ARSIP:
        * SPACE / D / ->          : Foto Berikutnya (Next)
        * A / <-                  : Foto Sebelumnya (Back)
        * X / DEL                 : Hapus / Pindahkan foto ini ke 'selesai'
        * Q                       : Keluar program
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
import json
import shutil
from datetime import datetime

# ───────────────────────────────────────────────────────────────
#  KONFIGURASI
# ───────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCE  = os.path.join(BASE_DIR, "foto_kamera_atas")
DEFAULT_DATASET = os.path.join(BASE_DIR, "dataset_baru_atas")
SELESAI_DIR     = os.path.join(DEFAULT_SOURCE, "selesai")

CLASS_COLOR = {           # BGR
    'normal':           (50,  225, 50),   # Hijau Terang
    'etiolasi_ringan':  (0,   165, 255),  # Oranye Terang
    'etiolasi_berat':   (30,  30,  245),  # Merah Terang
    'empty':            (170, 170, 170),  # Abu-abu
}

IMG_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')
MIN_PX   = 10             # Ukuran crop minimum (pixels)
BAR_H    = 92             # Tinggi header bar

# ───────────────────────────────────────────────────────────────
#  TOMBOL HEADER (KLIK DENGAN MOUSE)
# ───────────────────────────────────────────────────────────────
HEADER_BUTTONS = [
    # (action_id, label, x, y, w, h, bg_color, border_color)
    ('normal',          ' 1: NORMAL ',    14,  46, 125, 34, (20, 80, 20),   CLASS_COLOR['normal']),
    ('etiolasi_ringan', ' 2: E.RINGAN ',  149, 46, 140, 34, (15, 70, 130), CLASS_COLOR['etiolasi_ringan']),
    ('etiolasi_berat',  ' 3: E.BERAT ',   299, 46, 130, 34, (20, 20, 110), CLASS_COLOR['etiolasi_berat']),
    ('empty',           ' 4: KOSONG ',    439, 46, 120, 34, (55, 55, 55),   CLASS_COLOR['empty']),
    ('cancel',          ' X: BATAL ',     569, 46, 115, 34, (40, 20, 50),   (180, 180, 220)),
]

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
    # Tombol klik yang dipilih dari header
    "header_action_clicked": None,
    # Flash animasi saat crop berhasil disimpan
    "flash_box": None,        # (bx1, by1, bx2, by2, color, label)
    "flash_timer": 0,         # durasi kilat konfirmasi
    "show_past_boxes": False, # DEFAULT FALSE: Tampilan bersih tanpa kotak numpuk!
}


def get_class_from_key(key):
    """
    Menerjemahkan keycode keyboard ke nama kelas pakcoy.
    Mendukung tombol angka atas, Numpad, dan tombol huruf.
    """
    # 1: NORMAL (Kotak Hijau)
    if key in (
        ord('1'), ord('n'), ord('N'),
        177, 225, 0xFFB1, 65457, 65436
    ):
        return 'normal'

    # 2: ETIOLASI RINGAN (Kotak Oranye)
    if key in (
        ord('2'), ord('r'), ord('R'),
        178, 226, 0xFFB2, 65458, 65433
    ):
        return 'etiolasi_ringan'

    # 3: ETIOLASI BERAT (Kotak Merah)
    if key in (
        ord('3'), ord('b'), ord('B'),
        179, 227, 0xFFB3, 65459, 65435
    ):
        return 'etiolasi_berat'

    # 4: EMPTY / KOSONG (Kotak Abu-abu)
    if key in (
        ord('4'), ord('e'), ord('E'),
        180, 228, 0xFFB4, 65460, 65430
    ):
        return 'empty'

    return None


def is_cancel_key(key):
    """Cek apakah tombol adalah pembatalan crop."""
    return key in (27, 8, ord('c'), ord('C'), 65288)  # ESC, Backspace, C


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

    # 3. Klik Kanan -> Batalkan Crop / Drag Pan
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

    # 4. Klik Kiri
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Cek apakah klik tombol di Header Bar
        if my <= BAR_H and state["crop_state"] == "PENDING":
            for btn_id, _, bx, by, bw, bh, _, _ in HEADER_BUTTONS:
                if bx <= mx <= bx + bw and by <= my <= by + bh:
                    state["header_action_clicked"] = btn_id
                    return

        # Klik di area gambar -> mulai tarik kotak crop baru
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
            # Jika hanya klik 1x tanpa drag -> batalkan
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

    overlay = canvas.copy()
    bg_color = (25, 25, 75) if confirm_delete else (18, 18, 22)
    cv2.rectangle(overlay, (0, 0), (win_w, BAR_H), bg_color, -1)
    cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0, canvas)
    border_col = (0, 70, 255) if confirm_delete else (0, 185, 255)
    cv2.line(canvas, (0, BAR_H), (win_w, BAR_H), border_col, 2)

    def txt(text, pos, scale=0.42, color=(220, 220, 220), thick=1):
        cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    if confirm_delete:
        txt("⚠️  KONFIRMASI: HAPUS / PINDAHKAN FILE INI?", (14, 26), 0.54, (0, 225, 255), 2)
        txt(f"File: {img_name}", (14, 52), 0.44, (255, 255, 255), 1)
        txt("Tekan [Y] = Pindahkan ke 'selesai'   |   [N] atau [ESC] = Batal", (14, 78), 0.46, (50, 255, 50), 2)
        return

    # Baris 1: Info Gambar & Zoom & Statistik
    zoom_txt = f"ZOOM: {state['zoom']:.1f}x"
    txt(f"[{idx+1}/{total}] {img_name}   |   {zoom_txt}", (14, 22), 0.46, (255, 255, 255), 1)
    
    # Ringkasan hasil crop foto ini
    summary_txt = f"Foto Ini: {n_crops_current} bibit terlabeli | Total Baru: N:{stats['normal']} R:{stats['etiolasi_ringan']} B:{stats['etiolasi_berat']}"
    txt(summary_txt, (win_w - 480, 22), 0.40, (0, 240, 255), 1)

    # Baris 2: Tombol Kelas Interaktif (Bisa Diklik & Ditekan Keyboard)
    is_pending = (crop_state == "PENDING")
    for btn_id, label, bx, by, bw, bh, b_bg, b_col in HEADER_BUTTONS:
        # Highlight tombol saat kotak sedang PENDING
        active_bg = b_bg if not is_pending else tuple(min(255, c + 35) for c in b_bg)
        border_thick = 2 if is_pending else 1
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), active_bg, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), b_col, border_thick)
        txt(label, (bx + 8, by + 22), 0.42, (255, 255, 255) if is_pending else (210, 210, 210), 1)

    # Indikator petunjuk di sebelah tombol
    if is_pending:
        txt("<<< PILIH KELAS DENGAN KLIK TOMBOL DI KIRI ATAU TEKAN ANGKA [1], [2], [3] PADA KEYBOARD", 
            (700, 68), 0.38, (0, 255, 255), 1)
    else:
        txt("[SPACE/D] Next  |  [A] Back  |  [Scroll] Zoom  |  [Klik Kanan] Geser  |  [Z] Undo  |  [X] Hapus/Selesai",
            (700, 68), 0.36, (170, 170, 170), 1)


# ───────────────────────────────────────────────────────────────
#  PENGELOLAAN RIWAYAT & PENYIMPANAN
# ───────────────────────────────────────────────────────────────
def get_history_file(dataset_folder):
    return os.path.join(dataset_folder, "crops_history.json")


def load_persistent_history(dataset_folder):
    """Muat riwayat bounding box yang tersimpan agar kotak tidak hilang saat dibuka ulang."""
    hfile = get_history_file(dataset_folder)
    if os.path.exists(hfile):
        try:
            with open(hfile, "r") as f:
                data = json.load(f)
                return data
        except Exception as e:
            print(f"⚠️  Gagal membaca riwayat history: {e}")
    return {}


def save_persistent_history(dataset_folder, history_dict):
    """Simpan riwayat kotak ke JSON."""
    hfile = get_history_file(dataset_folder)
    try:
        clean_dict = {}
        for k, v in history_dict.items():
            valid_list = [item for item in v if os.path.exists(item[0])]
            if valid_list:
                clean_dict[k] = valid_list
        with open(hfile, "w") as f:
            json.dump(clean_dict, f, indent=2)
    except Exception as e:
        print(f"⚠️  Gagal menyimpan crops_history.json: {e}")


def save_crop_to_disk(orig_img, img_path, x1, y1, x2, y2, class_name, dataset_folder, stats, labeled_history):
    """Potong gambar dan simpan ke folder kelas target."""
    crop = orig_img[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return None

    out_dir = os.path.join(dataset_folder, class_name)
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"{class_name}_crop_{ts}.jpg"
    fpath = os.path.join(out_dir, fname)

    cv2.imwrite(fpath, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    stats[class_name] = stats.get(class_name, 0) + 1

    item = (fpath, class_name, int(x1), int(y1), int(x2), int(y2))
    
    img_key = os.path.abspath(img_path)
    if img_key not in labeled_history:
        labeled_history[img_key] = []
    labeled_history[img_key].append(item)

    save_persistent_history(dataset_folder, labeled_history)
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

    # Riwayat crop sesi ini (Mulai dengan tampilan bersih tanpa kotak hantu masa lalu)
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
        img_key = os.path.abspath(img_path)
        orig = cv2.imread(img_path)
        if orig is None:
            print(f"⚠️  Gagal membuka: {img_path}")
            img_idx += 1
            continue

        img_name = os.path.basename(img_path)
        if img_key not in labeled_history:
            labeled_history[img_key] = []

        orig_h, orig_w = orig.shape[:2]
        state["orig_w"] = orig_w
        state["orig_h"] = orig_h

        # Reset zoom saat berganti foto
        state["zoom"] = 1.0
        state["center_x"] = 0.5
        state["center_y"] = 0.5
        state["crop_state"] = "IDLE"
        state["is_drawing_crop"] = False
        state["header_action_clicked"] = None

        while True:
            # 1. Update ukuran jendela saat ini
            try:
                rect = cv2.getWindowImageRect(WIN_MAIN)
                win_w = max(400, rect[2])
                win_h = max(300, rect[3])
            except:
                win_w, win_h = disp_w, disp_h

            # Hitung viewport zoom
            zoom = max(1.0, state["zoom"])
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
            avail_h = max(10, win_h - BAR_H)
            scale = min(win_w / max(1, rw), avail_h / max(1, rh))

            nw = max(1, int(rw * scale))
            nh = max(1, int(rh * scale))
            resized_roi = cv2.resize(roi_img, (nw, nh), interpolation=cv2.INTER_LINEAR)

            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            ox = (win_w - nw) // 2
            oy = BAR_H + (avail_h - nh) // 2
            canvas[oy:oy+nh, ox:ox+nw] = resized_roi

            # Simpan koordinat canvas
            state["canvas_scale"] = scale
            state["canvas_ox"] = ox
            state["canvas_oy"] = oy
            state["canvas_nw"] = nw
            state["canvas_nh"] = nh

            # 3. TAMPILKAN EFEK KILAT (FLASH) SAAT ANGKA 1, 2, 3 DITEKAN
            if state["flash_timer"] > 0 and state["flash_box"]:
                fbx1, fby1, fbx2, fby2, fcol, flbl = state["flash_box"]
                fcx1 = int(ox + (fbx1 - rx1) * scale)
                fcy1 = int(oy + (fby1 - ry1) * scale)
                fcx2 = int(ox + (fbx2 - rx1) * scale)
                fcy2 = int(oy + (fby2 - ry1) * scale)

                # Kotak kilat tebal + tulisan konfirmasi
                cv2.rectangle(canvas, (fcx1, fcy1), (fcx2, fcy2), (255, 255, 255), 4)
                cv2.rectangle(canvas, (fcx1, fcy1), (fcx2, fcy2), fcol, 3)
                badge_w = len(flbl) * 8 + 12
                top_badge = max(oy, fcy1 - 22)
                cv2.rectangle(canvas, (fcx1, top_badge), (fcx1 + badge_w, top_badge + 20), (0, 0, 0), -1)
                cv2.rectangle(canvas, (fcx1, top_badge), (fcx1 + badge_w, top_badge + 20), fcol, 1)
                cv2.putText(canvas, flbl, (fcx1 + 4, top_badge + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
                state["flash_timer"] -= 1

            # 3. GAMBAR JEJAK KOTAK DARI BIBIT YANG SUDAH DILABELI
            current_crops = labeled_history.get(img_key, [])
            for c_idx, crop_item in enumerate(current_crops, start=1):
                _, c_cls, bx1, by1, bx2, by2 = crop_item
                cx1 = int(ox + (bx1 - rx1) * scale)
                cy1 = int(oy + (by1 - ry1) * scale)
                cx2 = int(ox + (bx2 - rx1) * scale)
                cy2 = int(oy + (by2 - ry1) * scale)

                col = CLASS_COLOR.get(c_cls, (255, 255, 255))
                # Border hitam 3px + border warna 2px agar sangat jelas dan rapi
                cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (0, 0, 0), 3)
                cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), col, 2)

                # Label ringkas: "#1 N", "#2 R", "#3 B", "#4 K"
                cls_initial = "N" if c_cls == "normal" else ("R" if c_cls == "etiolasi_ringan" else ("B" if c_cls == "etiolasi_berat" else "K"))
                short_lbl = f"#{c_idx} {cls_initial}"
                bw_badge = len(short_lbl) * 7 + 8
                top_badge = max(oy, cy1 - 16)
                cv2.rectangle(canvas, (cx1, top_badge), (cx1 + bw_badge, top_badge + 16), (0, 0, 0), -1)
                cv2.rectangle(canvas, (cx1, top_badge), (cx1 + bw_badge, top_badge + 16), col, 1)
                cv2.putText(canvas, short_lbl, (cx1 + 4, top_badge + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

            # 4. Gambar kotak crop yang SEDANG ditarik / PENDING
            if state["crop_state"] in ("DRAWING", "PENDING"):
                bx1, by1, bx2, by2 = get_current_crop_rect()
                cx1 = int(ox + (bx1 - rx1) * scale)
                cy1 = int(oy + (by1 - ry1) * scale)
                cx2 = int(ox + (bx2 - rx1) * scale)
                cy2 = int(oy + (by2 - ry1) * scale)

                color = (0, 255, 255) if state["crop_state"] == "PENDING" else (255, 255, 0)
                # Outer shadow hitam + inner border warna
                cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (0, 0, 0), 4)
                cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), color, 2)
                
                pw, ph = abs(bx2 - bx1), abs(by2 - by1)
                size_txt = f"{pw}x{ph}px [TEKAN 1/2/3 ATAU KLIK TOMBOL ATAS]" if state["crop_state"] == "PENDING" else f"{pw}x{ph}px"
                cv2.rectangle(canvas, (cx1, max(oy, cy1 - 20)), (cx1 + len(size_txt)*7 + 8, cy1), (0, 0, 0), -1)
                cv2.putText(canvas, size_txt, (cx1 + 4, max(oy + 14, cy1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

                # Update preview crop di jendela kecil
                if state["crop_state"] == "PENDING":
                    crop_prev = orig[by1:by2, bx1:bx2]
                    if crop_prev.size > 0:
                        prev_resized = cv2.resize(crop_prev, (240, 240))
                        cv2.imshow(WIN_PREV, prev_resized)

            # 5. Pesan notifikasi sementara di bawah
            if saved_msg and saved_timer > 0:
                cv2.rectangle(canvas, (10, win_h - 35), (win_w - 10, win_h - 5), (10, 40, 10), -1)
                cv2.rectangle(canvas, (10, win_h - 35), (win_w - 10, win_h - 5), (0, 255, 120), 1)
                cv2.putText(canvas, saved_msg, (20, win_h - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 150), 1, cv2.LINE_AA)
                saved_timer -= 1

            # 6. Header Bar di Bagian Atas
            n_crops = len(current_crops)
            draw_header_bar(canvas, img_name, img_idx, len(images), stats, state["crop_state"], n_crops, confirm_delete)

            cv2.imshow(WIN_MAIN, canvas)
            key = cv2.waitKey(25) & 0xFF

            # ───────────────────────────────────────────────────────────
            #  LOGIKA EVENT: PRIORITAS TINGGI SAAT CROP SEDANG PENDING
            # ───────────────────────────────────────────────────────────
            # 1. Cek apakah ada aksi klik dari tombol Header
            clicked_action = state["header_action_clicked"]
            state["header_action_clicked"] = None

            if state["crop_state"] == "PENDING":
                # A. Simpan jika tombol Header diklik
                if clicked_action in CLASS_COLOR:
                    bx1, by1, bx2, by2 = get_current_crop_rect()
                    item = save_crop_to_disk(orig, img_path, bx1, by1, bx2, by2,
                                            clicked_action, dataset_folder, stats, labeled_history)
                    if item:
                        tot = len(labeled_history[img_key])
                        saved_msg = f"✅ TERSIMPAN: [{clicked_action.upper()}] (Foto ini: {tot} bibit)"
                        saved_timer = 40
                        # Efek kilat penanda sesaat di bibit
                        state["flash_box"] = (bx1, by1, bx2, by2, CLASS_COLOR[clicked_action], f"✅ {clicked_action.upper()}")
                        state["flash_timer"] = 12
                        print(f"✅ [{clicked_action.upper()}] #{tot} Tersimpan: {os.path.basename(item[0])}")
                    state["crop_state"] = "IDLE"
                    continue

                elif clicked_action == "cancel":
                    state["crop_state"] = "IDLE"
                    saved_msg = "Crop dibatalkan."
                    saved_timer = 30
                    continue

                # B. Simpan jika tombol Keyboard ditekan (1, 2, 3, 4, N, R, B, E, Numpad)
                class_choice = get_class_from_key(key)
                if class_choice:
                    bx1, by1, bx2, by2 = get_current_crop_rect()
                    item = save_crop_to_disk(orig, img_path, bx1, by1, bx2, by2,
                                            class_choice, dataset_folder, stats, labeled_history)
                    if item:
                        tot = len(labeled_history[img_key])
                        saved_msg = f"✅ TERSIMPAN: [{class_choice.upper()}] (Foto ini: {tot} bibit)"
                        saved_timer = 40
                        # Efek kilat penanda sesaat di bibit
                        state["flash_box"] = (bx1, by1, bx2, by2, CLASS_COLOR[class_choice], f"✅ {class_choice.upper()}")
                        state["flash_timer"] = 12
                        print(f"✅ [{class_choice.upper()}] #{tot} Tersimpan: {os.path.basename(item[0])}")
                    state["crop_state"] = "IDLE"
                    continue

                # C. Batalkan jika tekan ESC, C, Backspace
                if is_cancel_key(key):
                    state["crop_state"] = "IDLE"
                    saved_msg = "Crop dibatalkan (tidak disimpan)"
                    saved_timer = 30
                    continue

            # ───────────────────────────────────────────────────────────
            #  MODE KONFIRMASI HAPUS / PINDAHKAN FOTO
            # ───────────────────────────────────────────────────────────
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
                    confirm_delete = False
                    break
                elif key in (ord('n'), ord('N'), 27):
                    confirm_delete = False
                    saved_msg = "Batal menghapus foto."
                    saved_timer = 30
                continue

            # ───────────────────────────────────────────────────────────
            #  KONTROL NAVIGASI & ALAT UMUM
            # ───────────────────────────────────────────────────────────
            # KELUAR PROGRAM: Q atau ESC (saat tidak ada crop pending)
            if key in (ord('q'), ord('Q')):
                img_idx = len(images)
                break
            elif key == 27 and state["crop_state"] == "IDLE":
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

            # RESET ZOOM: 0 (Nol)
            elif key in (ord('0'), ord('r'), ord('R')) and state["crop_state"] == "IDLE":
                state["zoom"] = 1.0
                state["center_x"] = 0.5
                state["center_y"] = 0.5
                saved_msg = "Zoom di-reset (1.0x)"
                saved_timer = 30

            # GESER PAN DENGAN TOMBOL PANAH
            elif key in (82, 2490368): # Panah Atas
                state["center_y"] = max(0.0, state["center_y"] - 0.05 / state["zoom"])
            elif key in (84, 2621440): # Panah Bawah
                state["center_y"] = min(1.0, state["center_y"] + 0.05 / state["zoom"])
            elif key in (81, 2424832): # Panah Kiri
                state["center_x"] = max(0.0, state["center_x"] - 0.05 / state["zoom"])
            elif key in (83, 2555904): # Panah Kanan
                state["center_x"] = min(1.0, state["center_x"] + 0.05 / state["zoom"])

            # BERSIHKAN SEMUA KOTAK DI FOTO INI: C (saat IDLE)
            elif key in (ord('c'), ord('C')) and state["crop_state"] == "IDLE":
                crops_list = labeled_history.get(img_key, [])
                if crops_list:
                    for item in crops_list:
                        fpath, cls = item[0], item[1]
                        if os.path.exists(fpath):
                            try:
                                os.remove(fpath)
                            except:
                                pass
                        stats[cls] = max(0, stats.get(cls, 0) - 1)
                    labeled_history[img_key] = []
                    saved_msg = "🧹 Semua kotak pada foto ini telah dibersihkan."
                    saved_timer = 40
                    print("🧹 Semua kotak pada foto ini telah dibersihkan.")

            # UNDO CROP TERAKHIR: Z
            elif key in (ord('z'), ord('Z')):
                crops_list = labeled_history.get(img_key, [])
                if crops_list:
                    fpath, cls, _, _, _, _ = crops_list.pop()
                    if os.path.exists(fpath):
                        try:
                            os.remove(fpath)
                        except:
                            pass
                    stats[cls] = max(0, stats.get(cls, 0) - 1)
                    saved_msg = f"↩️  UNDO: Crop {os.path.basename(fpath)} dihapus."
                    saved_timer = 40
                    print(f"↩️  [UNDO] {os.path.basename(fpath)} dihapus.")
                else:
                    saved_msg = "Tidak ada crop pada foto ini yang bisa di-undo."
                    saved_timer = 30
                state["crop_state"] = "IDLE"

    cv2.destroyAllWindows()

    print("\n" + "="*55)
    print("📊 REKAPITULASI DATASET BARU:")
    for cls, cnt in stats.items():
        print(f"   👉 {cls:<18}: {cnt:4d} gambar")
    print(f"   👉 {'TOTAL':<18}: {sum(stats.values()):4d} gambar")
    print(f"\n📁 Lokasi Dataset : {dataset_folder}")


if __name__ == "__main__":
    main()

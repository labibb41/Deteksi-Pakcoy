"""
==============================================================
  CROP TOOL - TAMPAK SAMPING (SIDE VIEW)
  Pengukuran Tinggi Batang Pakcoy + Labeling Dataset
==============================================================
  ALUR:
    1. Tekan C untuk KALIBRASI skala (cm per pixel)
       - Klik 2 titik dengan jarak yang diketahui
       - Masukkan jarak sebenarnya (cm) di terminal
    2. Klik + Drag untuk crop area batang
    3. Tinggi otomatis dihitung dalam cm
    4. Klasifikasi otomatis:
         < 7 cm  → NORMAL (ideal penyemaian pakcoy)
         >= 7 cm → ETIOLASI (terlalu tinggi)
    5. Konfirmasi atau override kelas

  KONTROL:
    C          : Kalibrasi skala (px → cm)
    Klik+Drag  : Crop area batang
    ENTER      : Simpan dengan kelas OTOMATIS
    N          : Simpan paksa sebagai NORMAL
    E          : Simpan paksa sebagai ETIOLASI
    Z          : Undo (hapus crop terakhir)
    ESC        : Batal crop aktif
    SPACE      : Gambar berikutnya
    Q          : Keluar

  THRESHOLD:
    Normal   = tinggi batang < 7 cm
    Etiolasi = tinggi batang >= 7 cm
==============================================================
"""

import os
import sys

import cv2

# ── AUTO-RELAUNCH DENGAN C:\penv ──
PENV_PYTHON = r"C:\penv\Scripts\python.exe"
if os.path.exists(PENV_PYTHON) and os.path.normpath(sys.executable).lower() != os.path.normpath(PENV_PYTHON).lower():
    import subprocess
    sys.exit(subprocess.call([PENV_PYTHON] + sys.argv))

import glob
from datetime import datetime
from math import sqrt

# ───────────────────────────────────────────────────────────────
#  KONFIGURASI
# ───────────────────────────────────────────────────────────────
DEFAULT_SOURCE  = r"f:\Bismillah TA\foto_kamera_samping"
DEFAULT_DATASET = r"f:\Bismillah TA\dataset_baru_samping"

NORMAL_THRESH_CM = 7.0      # batas tinggi normal (cm)

CLASS_NORMAL   = "normal"   # tinggi < 7 cm
CLASS_ETIOLASI = "etiolasi" # tinggi >= 7 cm

CLASS_COLOR = {
    CLASS_NORMAL:   (50,  210, 50),
    CLASS_ETIOLASI: (0,   0,   220),
}

IMG_EXTS   = ('*.jpg','*.jpeg','*.png','*.bmp','*.JPG','*.JPEG','*.PNG')
MIN_PX     = 10
MAX_DISPLAY= 1100

# ───────────────────────────────────────────────────────────────
#  STATE GLOBAL
# ───────────────────────────────────────────────────────────────
# Mouse state untuk crop batang
mx_state = "IDLE"    # IDLE | DRAWING | PENDING
mx_start = (0, 0)
mx_curr  = (0, 0)
mx_draw  = False

# Mouse state untuk kalibrasi
cal_state  = "IDLE"    # IDLE | POINT1 | DONE
cal_pt1    = None
cal_pt2    = None
scale_cm_per_px = None   # cm per pixel (None = belum kalibrasi)

in_calib_mode = False    # True saat sedang kalibrasi


def mouse_cb(event, x, y, flags, param):
    global mx_state, mx_start, mx_curr, mx_draw
    global cal_state, cal_pt1, cal_pt2

    # ── MODE KALIBRASI ──
    if in_calib_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            if cal_state == "IDLE":
                cal_pt1  = (x, y)
                cal_state = "POINT1"
                print(f"   Titik 1: {cal_pt1}  →  Klik titik ke-2")
            elif cal_state == "POINT1":
                cal_pt2  = (x, y)
                cal_state = "DONE"
                print(f"   Titik 2: {cal_pt2}  →  Masukkan jarak sebenarnya di terminal")
        return

    # ── MODE CROP NORMAL ──
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
        abs(mx_curr[0] - mx_start[0])
        h = abs(mx_curr[1] - mx_start[1])
        mx_state = "PENDING" if (h > MIN_PX) else "IDLE"


def get_rect():
    x1 = min(mx_start[0], mx_curr[0])
    y1 = min(mx_start[1], mx_curr[1])
    x2 = max(mx_start[0], mx_curr[0])
    y2 = max(mx_start[1], mx_curr[1])
    return x1, y1, x2, y2


def pixel_to_cm(pixels):
    if scale_cm_per_px is None:
        return None
    return pixels * scale_cm_per_px


def classify_height(height_cm):
    if height_cm is None:
        return None
    return CLASS_NORMAL if height_cm < NORMAL_THRESH_CM else CLASS_ETIOLASI


# ───────────────────────────────────────────────────────────────
#  DRAW OVERLAY
# ───────────────────────────────────────────────────────────────
def draw_overlay(disp, img_name, idx, total, stats, state, height_info):
    _h, w = disp.shape[:2]
    bar_h = 110

    overlay = disp.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, disp, 0.25, 0, disp)
    cv2.line(disp, (0, bar_h), (w, bar_h), (80, 80, 80), 1)

    def txt(text, pos, scale=0.44, color=(220,220,220), thick=1):
        cv2.putText(disp, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    # Baris 1 - info gambar
    txt(f"[{idx+1}/{total}]  {img_name}", (10, 18), 0.5, (255,255,255))

    # Baris 2 - kalibrasi
    if scale_cm_per_px is not None:
        cal_txt = f"Skala: {scale_cm_per_px:.4f} cm/px  |  Threshold: <{NORMAL_THRESH_CM} cm = NORMAL"
        txt(cal_txt, (10, 36), 0.42, (100, 220, 100))
    else:
        txt("⚠  Belum dikalibrasi! Tekan C untuk kalibrasi skala", (10, 36), 0.42, (0, 100, 255))

    # Baris 3 - kontrol
    txt("C=Kalibrasi  |  Klik+Drag=Crop  |  ENTER=Auto  N=Normal  E=Etiolasi  Z=Undo  SPC=Next  Q=Quit",
        (10, 54), 0.4, (180,180,180))

    # Baris 4 - statistik
    s = stats
    stat = (f"Tersimpan:  Normal={s.get(CLASS_NORMAL,0)}  "
            f"Etiolasi={s.get(CLASS_ETIOLASI,0)}  "
            f"| Total={sum(s.values())}")
    txt(stat, (10, 72), 0.42, (100, 220, 100))

    # Baris 5 - info tinggi & kelas
    if state == "PENDING" and height_info:
        height_cm, auto_cls = height_info
        if height_cm is not None:
            color_cls = CLASS_COLOR.get(auto_cls, (200,200,200))
            info = f"Tinggi: {height_cm:.1f} cm  →  Kelas Otomatis: {auto_cls.upper()}"
            txt(info, (10, 90), 0.5, color_cls, 1)
            txt("ENTER=Simpan Otomatis | N=Normal | E=Etiolasi | ESC=Batal", (10, 107), 0.42, (0,255,255))
        else:
            txt("Belum dikalibrasi. Tekan N atau E untuk simpan manual.", (10, 90), 0.42, (255,255,0))
            txt("ENTER/N=Normal | E=Etiolasi | ESC=Batal", (10, 107), 0.42, (0,255,255))
    elif state == "DRAWING":
        txt("Sedang menggambar...", (10, 90), 0.42, (255,255,0))
    elif in_calib_mode:
        if cal_state == "IDLE":
            txt("KALIBRASI: Klik titik PERTAMA (misal: pangkal batang)", (10, 90), 0.5, (0,200,255))
        elif cal_state == "POINT1":
            txt("KALIBRASI: Klik titik KEDUA (misal: ujung referensi)", (10, 90), 0.5, (0,200,255))
    else:
        txt("Klik + Drag untuk crop area batang tanaman", (10, 90), 0.42, (160,160,160))


def draw_ruler(disp, scale_cm_per_px):
    """Gambar penggaris visual di kanan gambar."""
    if scale_cm_per_px is None:
        return
    h, w = disp.shape[:2]
    ruler_x  = w - 45
    start_y  = 120
    px_per_cm = 1.0 / scale_cm_per_px
    max_cm   = int((h - start_y) / px_per_cm) + 1

    # Garis penggaris
    cv2.line(disp, (ruler_x, start_y), (ruler_x, int(start_y + max_cm * px_per_cm)), (200,200,200), 1)

    for cm in range(min(max_cm+1, 25)):
        y = int(start_y + cm * px_per_cm)
        if y >= h - 5:
            break
        tick_len = 10 if cm % 5 == 0 else 5
        color    = (0,220,0) if cm < NORMAL_THRESH_CM else (0,0,200)
        cv2.line(disp, (ruler_x - tick_len, y), (ruler_x, y), color, 1)
        if cm % 2 == 0:
            cv2.putText(disp, f"{cm}", (ruler_x - tick_len - 22, y+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

    # Label threshold
    thresh_y = int(start_y + NORMAL_THRESH_CM * px_per_cm)
    if thresh_y < h - 5:
        cv2.line(disp, (ruler_x - 30, thresh_y), (w, thresh_y), (0, 100, 255), 1)
        cv2.putText(disp, f"{NORMAL_THRESH_CM}cm", (ruler_x - 42, thresh_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,100,255), 1, cv2.LINE_AA)


# ───────────────────────────────────────────────────────────────
#  SIMPAN CROP
# ───────────────────────────────────────────────────────────────
def save_crop_samping(img, x1, y1, x2, y2, class_name, dataset_folder, stats, undo_stack, height_cm=None):
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    out_dir = os.path.join(dataset_folder, class_name)
    os.makedirs(out_dir, exist_ok=True)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    hstr  = f"_h{height_cm:.1f}cm" if height_cm is not None else ""
    fname = f"{class_name}{hstr}_crop_{ts}.jpg"
    fpath = os.path.join(out_dir, fname)
    cv2.imwrite(fpath, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    stats[class_name] = stats.get(class_name, 0) + 1
    undo_stack.append((fpath, class_name))
    return fpath


def count_existing(dataset_folder, stats):
    for cls in [CLASS_NORMAL, CLASS_ETIOLASI]:
        d = os.path.join(dataset_folder, cls)
        if os.path.isdir(d):
            stats[cls] = len([f for f in os.listdir(d)
                               if f.lower().endswith(('.jpg','.jpeg','.png'))])


# ───────────────────────────────────────────────────────────────
#  KALIBRASI INTERAKTIF
# ───────────────────────────────────────────────────────────────
def run_calibration():
    global scale_cm_per_px, in_calib_mode, cal_state, cal_pt1, cal_pt2

    in_calib_mode = True
    cal_state     = "IDLE"
    cal_pt1       = None
    cal_pt2       = None

    print("\n" + "─"*50)
    print("📏 MODE KALIBRASI")
    print("   Klik 2 titik pada gambar yang jaraknya diketahui.")
    print("   Contoh: pangkal dan ujung daun referensi,")
    print("           atau tinggi pot yang sudah diukur.")
    print("─"*50)

    # Tunggu 2 titik diklik
    while cal_state != "DONE":
        cv2.waitKey(50)

    # Hitung jarak pixel
    dx  = cal_pt2[0] - cal_pt1[0]
    dy  = cal_pt2[1] - cal_pt1[1]
    pix = sqrt(dx*dx + dy*dy)

    print(f"\n   Jarak pixel antara 2 titik: {pix:.1f} px")
    while True:
        try:
            real_cm = float(input("   Masukkan jarak SEBENARNYA dalam cm: "))
            if real_cm <= 0:
                raise ValueError
            break
        except ValueError:
            print("   ⚠️  Masukkan angka positif!")

    scale_cm_per_px = real_cm / pix
    print(f"\n   ✅ Kalibrasi selesai: {scale_cm_per_px:.5f} cm/pixel")
    print(f"   (1 cm = {1/scale_cm_per_px:.1f} pixel)\n")

    in_calib_mode = False
    cal_state     = "IDLE"


# ───────────────────────────────────────────────────────────────
#  MAIN
# ───────────────────────────────────────────────────────────────
def main():
    global mx_state, mx_start, mx_curr, mx_draw, in_calib_mode

    source_folder = sys.argv[1] if len(sys.argv) > 1 else ""
    if not source_folder:
        print("\n📂 Folder foto kamera samping")
        print(f"   (Enter = default: {DEFAULT_SOURCE}): ", end="")
        source_folder = input().strip() or DEFAULT_SOURCE

    dataset_folder = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DATASET

    os.makedirs(source_folder,  exist_ok=True)
    os.makedirs(dataset_folder, exist_ok=True)

    # Kumpulkan gambar
    images = []
    for ext in IMG_EXTS:
        images += glob.glob(os.path.join(source_folder, ext))
    images = sorted(set(images))

    if not images:
        print(f"\n⚠️  Tidak ada gambar di: {source_folder}")
        print("   Letakkan foto kamera samping ke folder tersebut, lalu jalankan ulang.\n")
        input("Tekan Enter untuk keluar...")
        return

    print(f"\n✅ {len(images)} gambar ditemukan")
    print(f"📁 Crop akan disimpan ke: {dataset_folder}")
    print(f"📏 Threshold tinggi normal: < {NORMAL_THRESH_CM} cm")
    print("\nTips: Tekan C di dalam window untuk kalibrasi skala cm/pixel")
    print("Tekan Enter untuk mulai...")
    input()

    stats      = {CLASS_NORMAL: 0, CLASS_ETIOLASI: 0}
    undo_stack = []
    count_existing(dataset_folder, stats)

    WIN_MAIN    = "CROP TOOL - SIDE VIEW  |  C=Kalibrasi  Q=Keluar  SPACE=Next"
    WIN_PREVIEW = "Preview Crop"
    cv2.namedWindow(WIN_MAIN,    cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_PREVIEW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_PREVIEW, 200, 350)
    cv2.setMouseCallback(WIN_MAIN, mouse_cb)

    img_idx  = 0
    exit_all = False

    while img_idx < len(images) and not exit_all:
        img_path = images[img_idx]
        orig     = cv2.imread(img_path)
        if orig is None:
            img_idx += 1
            continue

        img_name = os.path.basename(img_path)
        h, w     = orig.shape[:2]
        scale    = 1.0
        if max(h, w) > MAX_DISPLAY:
            scale = MAX_DISPLAY / max(h, w)
            work  = cv2.resize(orig, (int(w*scale), int(h*scale)))
        else:
            work = orig.copy()

        mx_state = "IDLE"
        mx_start = (0, 0)
        mx_curr  = (0, 0)
        mx_draw  = False
        saved_msg   = ""
        saved_timer = 0

        while True:
            disp = work.copy()

            # Gambar titik kalibrasi
            if in_calib_mode:
                if cal_pt1:
                    cv2.circle(disp, cal_pt1, 5, (0, 200, 255), -1)
                if cal_pt2:
                    cv2.circle(disp, cal_pt2, 5, (0, 200, 255), -1)
                if cal_pt1 and cal_pt2:
                    cv2.line(disp, cal_pt1, cal_pt2, (0, 200, 255), 2)

            # Hitung info tinggi untuk PENDING
            height_info = None
            if mx_state in ("DRAWING", "PENDING"):
                x1, y1, x2, y2 = get_rect()
                h_px       = y2 - y1
                h_cm       = pixel_to_cm(h_px)
                auto_cls   = classify_height(h_cm)
                height_info = (h_cm, auto_cls)

                color = CLASS_COLOR.get(auto_cls, (255, 255, 0)) if mx_state == "PENDING" else (255, 255, 0)
                cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)

                # Tampilkan tinggi di dalam rectangle
                if h_cm is not None:
                    label = f"{h_cm:.1f}cm"
                    cv2.putText(disp, label, (x1+3, y2-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                if mx_state == "PENDING":
                    crop_prev = work[y1:y2, x1:x2]
                    if crop_prev.size > 0:
                        ph, pw = crop_prev.shape[:2]
                        # Resize ke tinggi 320 untuk preview vertikal
                        ratio   = 320 / ph if ph > 0 else 1
                        preview = cv2.resize(crop_prev, (max(1, int(pw*ratio)), 320))
                        cv2.imshow(WIN_PREVIEW, preview)

            # Penggaris visual
            draw_ruler(disp, scale_cm_per_px)

            # Pesan konfirmasi
            if saved_msg and saved_timer > 0:
                dh, _dw = disp.shape[:2]
                cv2.putText(disp, saved_msg, (10, dh - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2, cv2.LINE_AA)
                saved_timer -= 1

            draw_overlay(disp, img_name, img_idx, len(images), stats, mx_state, height_info)
            cv2.imshow(WIN_MAIN, disp)

            key = cv2.waitKey(50) & 0xFF

            # ── KELUAR ──
            if key == ord('q'):
                exit_all = True
                break

            # ── ESC ──
            elif key == 27:
                if in_calib_mode:
                    in_calib_mode = False
                    print("   Kalibrasi dibatalkan.")
                elif mx_state in ("DRAWING", "PENDING"):
                    mx_state = "IDLE"
                else:
                    exit_all = True
                    break

            # ── SPACE: next image ──
            elif key == ord(' '):
                break

            # ── C: mulai kalibrasi ──
            elif key == ord('c') and not in_calib_mode and mx_state == "IDLE":
                run_calibration()

            # ── Z: UNDO ──
            elif key == ord('z'):
                if undo_stack:
                    fpath, cls = undo_stack.pop()
                    if os.path.exists(fpath):
                        os.remove(fpath)
                        stats[cls] = max(0, stats.get(cls, 0) - 1)
                        print(f"↩️  Undo: {os.path.basename(fpath)}")
                        saved_msg   = f"UNDO: {os.path.basename(fpath)}"
                        saved_timer = 40
                mx_state = "IDLE"

            # ── SIMPAN (hanya saat PENDING) ──
            elif mx_state == "PENDING":
                x1, y1, x2, y2 = get_rect()
                h_px  = y2 - y1
                h_cm  = pixel_to_cm(h_px)
                class_name = None

                if key == 13 or key == ord('\r'):   # ENTER → kelas otomatis
                    class_name = classify_height(h_cm) or CLASS_NORMAL
                elif key == ord('n'):
                    class_name = CLASS_NORMAL
                elif key == ord('e'):
                    class_name = CLASS_ETIOLASI

                if class_name:
                    fpath = save_crop_samping(work, x1, y1, x2, y2,
                                              class_name, dataset_folder,
                                              stats, undo_stack, h_cm)
                    if fpath:
                        h_str = f"{h_cm:.1f}cm" if h_cm is not None else "?"
                        msg   = f"✅ [{class_name}]  Tinggi: {h_str}  →  {os.path.basename(fpath)}"
                        print(msg)
                        saved_msg   = f"SIMPAN  {class_name.upper()}  {h_str}"
                        saved_timer = 50
                    mx_state = "IDLE"

        img_idx += 1

    cv2.destroyAllWindows()

    # Statistik akhir
    print("\n" + "="*55)
    print("📊 STATISTIK AKHIR DATASET SAMPING:")
    for cls, cnt in stats.items():
        bar = "█" * (cnt // 2) if cnt else ""
        print(f"   {cls:<18}: {cnt:4d}  {bar}")
    print(f"   {'TOTAL':<18}: {sum(stats.values()):4d}")
    print(f"\n📁 Tersimpan di: {dataset_folder}")
    print("="*55)
    input("\nTekan Enter untuk keluar...")


if __name__ == "__main__":
    main()

"""
==============================================================
  DETEKSI PAKCOY - CLI TERMINAL VERSION
  Pipeline: YOLO26n-Seg  -->  MobileNetV2
  Termasuk Model Baru Hasil Training: mobilenetv2_pakcoy_augmented.h5
==============================================================
"""

import os
import sys

# ── AUTO-RELAUNCH DENGAN C:\penv JIKA DIPANGGIL DENGAN PYTHON BAWAAN ──
PENV_PYTHON = r"C:\penv\Scripts\python.exe"
if os.path.exists(PENV_PYTHON) and os.path.normpath(sys.executable).lower() != os.path.normpath(PENV_PYTHON).lower():
    import subprocess
    sys.exit(subprocess.call([PENV_PYTHON] + sys.argv))

import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from collections import Counter
from typing import Any

import cv2
import numpy as np
from PIL import Image

# Path dasar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_WEIGHTS = os.path.join(BASE_DIR, "exp_pakcoy_yolo26_seg_v4", "weights", "best.pt")

# DAFTAR MODEL KLASIFIKASI (Model Murni Baru di urutan pertama!)
MOBILENET_MODELS = {
    "Optimasi (Terbaru)": os.path.join(BASE_DIR, "mobilenetv2_pakcoy_opt.h5"),
    "Murni (Terbaru)":  os.path.join(BASE_DIR, "mobilenetv2_pakcoy_murni.h5"),
    "Augmented":        os.path.join(BASE_DIR, "mobilenetv2_pakcoy_augmented.h5"),
    "V3 (Terbaru)":     os.path.join(BASE_DIR, "mobilenetv2_v3_terbaru.h5"),
    "Top Pakcoy V3":    os.path.join(BASE_DIR, "model_top_pakcoy_v3.h5"),
    "Top Pakcoy":       os.path.join(BASE_DIR, "model_top_pakcoy.h5"),
    "V2 (Baru)":        os.path.join(BASE_DIR, "mobilenetv2_v2_baru.h5"),
    "V1 (Lama)":        os.path.join(BASE_DIR, "mobilenetv2_v1_lama.h5"),
}

CLASS_NAMES = ["etiolasi_berat", "etiolasi_ringan", "normal"]

# ── AMBANG BATAS KEYAKINAN (CONFIDENCE THRESHOLD) ──
# Bibit secara alami adalah tanaman sehat (normal).
# Semakin RENDAH THRESH_NORMAL atau semakin TINGGI THRESH_ETIOLASI, bibit semakin diprioritaskan NORMAL.
THRESH_NORMAL          = 0.30  # Jika probabilitas normal >= 35%, bibit langsung diprioritaskan NORMAL
THRESH_ETIOLASI_RINGAN = 0.75  # Etiolasi ringan harus yakin minimal 70% baru diakui (mencegah salah tuduh)
THRESH_ETIOLASI_BERAT  = 0.80  # Etiolasi berat harus yakin minimal 65% baru diakui

# ── WARNA & DISPLAY SESUAI DESAIN TRAY PAKCOY (Format BGR OpenCV) ──
COLOR_KOTAK_ISI   = (255, 180, 0)   # KOTAK TRAY: CYAN / BIRU MUDA (sesuai referensi)
COLOR_KOTAK_EMPTY = (120, 120, 120) # KOTAK TRAY KOSONG: ABU-ABU
COLOR_EMPTY_TXT   = (180, 180, 180) # TEKS EMPTY: ABU-ABU TERANG

CLASS_COLORS = {
    "normal":          (0,   220, 0),    # Hijau Terang
    "etiolasi_ringan": (0,   165, 255),  # Oranye / Kuning Emas
    "etiolasi_berat":  (0,   0,   230),  # Merah Terang
    "empty":           (140, 140, 140),  # Abu-abu
}

CLASS_DISPLAY_NAMES = {
    "normal":          "Normal",
    "etiolasi_ringan": "Et. Ringan",
    "etiolasi_berat":  "Et. Berat",
    "empty":           "Empty",
}

# ─────────────────────────────────────────────────────────────
#  IMPORT & LOAD MODELS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65, flush=True)
print("  🌿 SISTEM DETEKSI & KLASIFIKASI PAKCOY (TERMINAL CLI)", flush=True)
print("="*65, flush=True)

print("\n⏳ [1/2] Loading YOLO26n-Seg (Model Deteksi)...", flush=True)
from ultralytics import YOLO

yolo_model = YOLO(YOLO_WEIGHTS)
print("   ✅ YOLO berhasil dimuat! Kelas YOLO:", yolo_model.names, flush=True)

print("\n⏳ [2/2] Loading Model Klasifikasi MobileNetV2...", flush=True)
from tensorflow.keras.models import load_model

mobilenet_models = {}
for name, path in MOBILENET_MODELS.items():
    if os.path.exists(path):
        mobilenet_models[name] = load_model(path, compile=False)
        print(f"   ✅ Model {name:<18} loaded ({os.path.basename(path)})", flush=True)
    else:
        print(f"   ⚠️  Model {name:<18} NOT FOUND: {path}", flush=True)

print("─"*65, flush=True)


def clean_path(p):
    p = p.strip()
    if p.startswith('&'):
        p = p[1:].strip()
    p = p.strip("'").strip('"').strip()
    return p


def choose_file_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        fpath = filedialog.askopenfilename(
            title="Pilih Gambar Tanaman Pakcoy",
            initialdir=BASE_DIR,
            filetypes=[
                ("File Gambar", "*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.JPG;*.JPEG;*.PNG"),
                ("Semua File", "*.*")
            ]
        )
        root.destroy()
        return fpath
    except Exception as e:
        print(f"⚠️ Pop-up file picker tidak tersedia: {e}", flush=True)
        return ""


def draw_detection_view(orig_img, tray_slots, model_name):
    vis = orig_img.copy()
    _h, _w = vis.shape[:2]

    for idx, slot in enumerate(tray_slots):
        kx1, ky1, kx2, ky2 = slot["kotak_box"]

        if slot["has_bibit"]:
            # 1. KOTAK TRAY LUBANG: GARIS CYAN / BIRU MUDA (PERSIS FOTO REFERENSI)
            cv2.rectangle(vis, (kx1, ky1), (kx2, ky2), COLOR_KOTAK_ISI, 1)
            # Tag nomor lubang tray di pojok kiri atas
            cv2.putText(vis, f"#{idx}", (kx1 + 2, ky1 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, COLOR_KOTAK_ISI, 1, cv2.LINE_AA)

            bx1, by1, bx2, by2 = slot["bibit_box"]
            label, conf = slot["preds"].get(model_name, ("?", 0))
            
            # 2. WARNA KOTAK BIBIT DINAMIS SESUAI KELAS (HIJAU / ORANYE / MERAH)
            box_col = CLASS_COLORS.get(label, (0, 220, 0))
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), box_col, 2)
            
            # 3. LABEL BANNER SOLID WARNA KELAS
            disp_name = CLASS_DISPLAY_NAMES.get(label, label)
            txt = f"{disp_name} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            
            # Background banner solid warna kelas menempel di pojok kiri atas kotak
            badge_top = max(0, by1 - th - 5)
            cv2.rectangle(vis, (bx1, badge_top), (bx1 + tw + 4, by1), box_col, -1)
            
            # Teks tebal putih (atau hitam jika oranye/kuning)
            text_col = (0, 0, 0) if label == "etiolasi_ringan" else (255, 255, 255)
            cv2.putText(vis, txt, (bx1 + 2, by1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_col, 1, cv2.LINE_AA)
        else:
            # 4. KOTAK TRAY KOSONG: WARNA ABU-ABU + 'EMPTY'
            cv2.rectangle(vis, (kx1, ky1), (kx2, ky2), COLOR_KOTAK_EMPTY, 1)
            cv2.putText(vis, f"#{idx}", (kx1 + 2, ky1 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, COLOR_KOTAK_EMPTY, 1, cv2.LINE_AA)
            cx, cy = (kx1 + kx2) // 2, (ky1 + ky2) // 2
            cv2.putText(vis, "EMPTY", (cx - 18, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLOR_EMPTY_TXT, 1, cv2.LINE_AA)

    return vis


def show_interactive_display(views_dict, orig_shape):
    win_name = "Hasil Deteksi Pakcoy YOLO26n + MobileNetV2 - [Scroll: Zoom | Drag: Geser | 1-6: Model | ESC: Tutup]"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    orig_h, orig_w = orig_shape[:2]
    disp_h = 780
    disp_w = int(orig_w * (disp_h / orig_h))
    cv2.resizeWindow(win_name, disp_w, disp_h)

    model_keys = list(views_dict.keys())
    key_map = {
        ord('1'): model_keys[0] if len(model_keys) > 0 else None,
        ord('2'): model_keys[1] if len(model_keys) > 1 else None,
        ord('3'): model_keys[2] if len(model_keys) > 2 else None,
        ord('4'): model_keys[3] if len(model_keys) > 3 else None,
        ord('5'): model_keys[4] if len(model_keys) > 4 else None,
        ord('6'): model_keys[5] if len(model_keys) > 5 else None,
        ord('7'): model_keys[6] if len(model_keys) > 6 else None,
    }

    current_model = "Optimasi (Terbaru)" if "Optimasi (Terbaru)" in views_dict else ("Murni (Terbaru)" if "Murni (Terbaru)" in views_dict else model_keys[0])

    # State interaktif Zoom & Pan
    state = {
        "zoom": 1.0,
        "center_x": 0.5,
        "center_y": 0.5,
        "is_dragging": False,
        "drag_start_x": 0,
        "drag_start_y": 0,
        "drag_start_cx": 0.5,
        "drag_start_cy": 0.5,
        "canvas_scale": 1.0,
        "canvas_ox": 0,
        "canvas_oy": 0,
        "canvas_nw": disp_w,
        "canvas_nh": disp_h,
    }

    def mouse_callback(event, mx, my, flags, param):
        # 1. Mouse Wheel -> Zoom In / Zoom Out
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                state["zoom"] = min(8.0, state["zoom"] * 1.25)
            else:
                state["zoom"] = max(1.0, state["zoom"] / 1.25)

            # Geser fokus ke arah kursor jika sedang di-zoom
            if state["zoom"] > 1.0:
                ox, oy = state["canvas_ox"], state["canvas_oy"]
                nw, nh = state["canvas_nw"], state["canvas_nh"]
                if nw > 0 and nh > 0:
                    u = np.clip((mx - ox) / nw, 0.0, 1.0)
                    v = np.clip((my - oy) / nh, 0.0, 1.0)
                    state["center_x"] = np.clip(state["center_x"] + (u - 0.5) * 0.25 / state["zoom"], 0.0, 1.0)
                    state["center_y"] = np.clip(state["center_y"] + (v - 0.5) * 0.25 / state["zoom"], 0.0, 1.0)
            else:
                state["center_x"] = 0.5
                state["center_y"] = 0.5

        # 2. Double Click -> Reset Zoom
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            state["zoom"] = 1.0
            state["center_x"] = 0.5
            state["center_y"] = 0.5

        # 3. Klik Kiri -> Mulai Drag / Pan
        elif event == cv2.EVENT_LBUTTONDOWN:
            state["is_dragging"] = True
            state["drag_start_x"] = mx
            state["drag_start_y"] = my
            state["drag_start_cx"] = state["center_x"]
            state["drag_start_cy"] = state["center_y"]

        # 4. Geser Mouse saat Drag
        elif event == cv2.EVENT_MOUSEMOVE:
            if state["is_dragging"] and state["zoom"] > 1.0:
                scale = max(1e-4, state["canvas_scale"])
                dx = mx - state["drag_start_x"]
                dy = my - state["drag_start_y"]
                delta_cx = (dx / scale) / orig_w
                delta_cy = (dy / scale) / orig_h
                state["center_x"] = np.clip(state["drag_start_cx"] - delta_cx, 0.0, 1.0)
                state["center_y"] = np.clip(state["drag_start_cy"] - delta_cy, 0.0, 1.0)

        # 5. Lepas Klik Kiri -> Selesai Drag
        elif event == cv2.EVENT_LBUTTONUP:
            state["is_dragging"] = False

    cv2.setMouseCallback(win_name, mouse_callback)

    print("\n🖥️  Jendela tampilan Python telah dibuka!", flush=True)
    print("   👉 Scroll Roda Mouse      : Zoom In / Zoom Out (Bebas melebar/gepeng!)", flush=True)
    print("   👉 Klik & Drag Kiri       : Geser (Pan) melihat bibit dari dekat", flush=True)
    print("   👉 Double-Click / [R]     : Reset Zoom kembali ke tampilan penuh", flush=True)
    print("   👉 Tombol '+ / -'         : Zoom In / Zoom Out lewat keyboard", flush=True)
    print("   👉 Tombol Panah           : Geser tampilan", flush=True)
    print("   👉 Tombol '1'             : ⭐ Model Optimasi Terbaru (mobilenetv2_pakcoy_opt.h5)", flush=True)
    print("   👉 Tombol '2'             : Model Murni (mobilenetv2_pakcoy_murni.h5)", flush=True)
    print("   👉 Tombol '3'             : Model Augmented", flush=True)
    print("   👉 Tombol '4' s.d '8'     : Model V3, Top Pakcoy V3, Top Pakcoy, V2, V1", flush=True)
    print("   👉 Tombol 'ESC' atau 'Q'  : Tutup jendela\n", flush=True)

    while True:
        img_to_show = views_dict.get(current_model)
        if img_to_show is None:
            break

        # Dapatkan ukuran jendela saat ini (agar letterboxing selalu proporsional 1:1)
        rect = cv2.getWindowImageRect(win_name)
        win_w = rect[2] if (rect is not None and rect[2] > 50) else disp_w
        win_h = rect[3] if (rect is not None and rect[3] > 50) else disp_h

        # 1. Hitung Region of Interest (ROI) sesuai Zoom & Pan
        zoom = state["zoom"]
        crop_w = int(orig_w / zoom)
        crop_h = int(orig_h / zoom)

        cx = state["center_x"]
        cy = state["center_y"]

        x1 = int(np.clip(cx * orig_w - crop_w / 2, 0, max(0, orig_w - crop_w)))
        y1 = int(np.clip(cy * orig_h - crop_h / 2, 0, max(0, orig_h - crop_h)))
        x2 = min(orig_w, x1 + crop_w)
        y2 = min(orig_h, y1 + crop_h)

        cropped = img_to_show[y1:y2, x1:x2]
        if cropped.size == 0:
            cropped = img_to_show

        # 2. Letterbox ke canvas jendela tanpa pernah merusak aspek rasio (tidak melebar!)
        cw = cropped.shape[1]
        ch = cropped.shape[0]
        scale = min(win_w / cw, win_h / ch)
        nw = max(1, int(cw * scale))
        nh = max(1, int(ch * scale))
        resized = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        ox = (win_w - nw) // 2
        oy = (win_h - nh) // 2
        canvas[oy:oy+nh, ox:ox+nw] = resized

        # Simpan koordinat canvas untuk interaksi mouse
        state["canvas_scale"] = scale
        state["canvas_ox"] = ox
        state["canvas_oy"] = oy
        state["canvas_nw"] = nw
        state["canvas_nh"] = nh

        # 3. Bar Kontrol & Info di Bagian Atas
        bar_h = 36
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (win_w, bar_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.line(canvas, (0, bar_h), (win_w, bar_h), COLOR_KOTAK_ISI, 2)

        status_txt = f"MODEL: [{current_model}]  |  ZOOM: {zoom:.1f}x  |  [Scroll/Drag]: Zoom & Geser  |  [R]: Reset  |  [1-6]: Model  |  [ESC]: Tutup"
        cv2.putText(canvas, status_txt, (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 235, 255), 1, cv2.LINE_AA)

        cv2.imshow(win_name, canvas)

        key = cv2.waitKey(25) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in key_map and key_map[key] is not None:
            current_model = key_map[key]
        elif key in (ord('+'), ord('=')):
            state["zoom"] = min(8.0, state["zoom"] * 1.25)
        elif key in (ord('-'), ord('_')):
            state["zoom"] = max(1.0, state["zoom"] / 1.25)
            if state["zoom"] <= 1.0:
                state["center_x"], state["center_y"] = 0.5, 0.5
        elif key in (ord('r'), ord('R'), ord('0')):
            state["zoom"] = 1.0
            state["center_x"], state["center_y"] = 0.5, 0.5
        # Tombol panah untuk menggeser pan
        elif key == 82 or key == 2490368: # Panah Atas
            state["center_y"] = max(0.0, state["center_y"] - 0.05 / state["zoom"])
        elif key == 84 or key == 2621440: # Panah Bawah
            state["center_y"] = min(1.0, state["center_y"] + 0.05 / state["zoom"])
        elif key == 81 or key == 2424832: # Panah Kiri
            state["center_x"] = max(0.0, state["center_x"] - 0.05 / state["zoom"])
        elif key == 83 or key == 2555904: # Panah Kanan
            state["center_x"] = min(1.0, state["center_x"] + 0.05 / state["zoom"])

    cv2.destroyAllWindows()
    print("🚪 Jendela tampilan ditutup.", flush=True)


def select_best_leaf_boxes(orig_bgr, bibit_boxes):
    """
    Jika ada beberapa box bibit yang tumpang tindih (misal box daun dan box tangkai+daun),
    pilih yang paling fokus pada KANOPI DAUN hijau dan buang box tangkai yang memanjang.
    """
    if not bibit_boxes:
        return []
    
    scored_boxes = []
    for b in bibit_boxes:
        x1, y1, x2, y2 = b
        crop = orig_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (25, 35, 35), (85, 255, 255))
        green_ratio = np.mean(mask > 0)
        w_b = max(1, x2 - x1)
        h_b = max(1, y2 - y1)
        aspect = min(w_b, h_b) / max(w_b, h_b)
        score = green_ratio * 1.5 + aspect * 0.5
        scored_boxes.append((b, score, green_ratio))
    
    scored_boxes.sort(key=lambda item: item[1], reverse=True)
    
    kept_boxes = []
    for box_a, score_a, g_ratio in scored_boxes:
        ax1, ay1, ax2, ay2 = box_a
        area_a = (ax2 - ax1) * (ay2 - ay1)
        acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
        
        overlap = False
        for box_b in kept_boxes:
            bx1, by1, bx2, by2 = box_b
            area_b = (bx2 - bx1) * (by2 - by1)
            
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            
            iou = inter / float(area_a + area_b - inter + 1e-6)
            inter_ratio = inter / float(min(area_a, area_b) + 1e-6)
            
            if iou > 0.25 or inter_ratio > 0.35 or (bx1 <= acx <= bx2 and by1 <= acy <= by2):
                overlap = True
                break
        
        if not overlap:
            kept_boxes.append(box_a)
            
    return kept_boxes


def process_image(img_path, conf_thresh=0.25, iou_thresh=0.45):
    img_path = clean_path(img_path)

    if not os.path.exists(img_path):
        print(f"\n❌ Error: File gambar tidak ditemukan: {img_path}", flush=True)
        return

    orig_bgr = cv2.imread(img_path)
    if orig_bgr is None:
        print(f"\n❌ Error: Gagal membaca file gambar: {img_path}", flush=True)
        return

    h, w = orig_bgr.shape[:2]
    print(f"\n📷 File Gambar : {os.path.basename(img_path)} ({w}x{h} px)", flush=True)
    print("🔍 Menjalankan deteksi objek dengan YOLO26n-Seg...", flush=True)

    # 1. YOLO Inference
    raw_res: Any = yolo_model(orig_bgr, conf=conf_thresh, iou=iou_thresh, verbose=False)
    results: Any = raw_res[0] if isinstance(raw_res, (list, tuple)) else next(iter(raw_res))
    boxes: Any = getattr(results, "boxes", None)

    if boxes is None or len(boxes) == 0:
        print("⚠️  Tidak ada objek yang terdeteksi oleh YOLO.", flush=True)
        return

    bibit_boxes = []
    kotak_boxes = []

    for b in boxes:
        cls_id = int(b.cls[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if cls_id == 0:
            bibit_boxes.append((x1, y1, x2, y2))
        elif cls_id == 1:
            kotak_boxes.append((x1, y1, x2, y2))

    print(f"   👉 Deteksi 'kotak' slot tray : {len(kotak_boxes)} titik", flush=True)
    
    # 2. Filter & Seleksi Kanopi Daun:
    # Hilangkan kotak tumpang tindih pada bibit yang sama, utamakan kotak kanopi daun hijau
    bibit_boxes = select_best_leaf_boxes(orig_bgr, bibit_boxes)
    print(f"   👉 Deteksi 'bibit' kanopi daun : {len(bibit_boxes)} titik (bebas tumpang tindih)", flush=True)

    # 3. Urutkan kotak tray
    if kotak_boxes:
        kotak_boxes = sorted(kotak_boxes, key=lambda b: (b[1] // 40, b[0]))
    else:
        kotak_boxes = bibit_boxes

    # 4. Hubungkan setiap slot kotak dengan bibit di dalamnya
    tray_slots = []
    crops_to_classify = []
    crop_slot_map = []

    used_bibit = set()
    for s_idx, k_box in enumerate(kotak_boxes):
        kx1, ky1, kx2, ky2 = k_box
        found_bibit = None

        for b_idx, b_box in enumerate(bibit_boxes):
            if b_idx in used_bibit:
                continue
            bx1, by1, bx2, by2 = b_box
            bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
            
            if kx1 <= bcx <= kx2 and ky1 <= bcy <= ky2:
                found_bibit = b_box
                used_bibit.add(b_idx)
                break

        slot_info = {
            "slot_idx": s_idx + 1,
            "kotak_box": k_box,
            "has_bibit": found_bibit is not None,
            "bibit_box": found_bibit,
            "preds": {}
        }

        if found_bibit is not None:
            bx1, by1, bx2, by2 = found_bibit
            crop = orig_bgr[by1:by2, bx1:bx2]
            if crop.size > 0 and crop.shape[0] >= 5 and crop.shape[1] >= 5:
                slot_info["crop_bgr"] = crop
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(crop_rgb).resize((224, 224))
                arr = np.array(crop_pil, dtype=np.float32) / 255.0  # Normalisasi [0, 1] sesuai training Colab
                crops_to_classify.append(arr)
                crop_slot_map.append(slot_info)

        tray_slots.append(slot_info)

    n_bibit_matched = len(crops_to_classify)
    n_empty = len(tray_slots) - n_bibit_matched
    print(f"✅ Pencocokan: {n_bibit_matched} slot berisi bibit, {n_empty} slot kosong (EMPTY)!", flush=True)

    # 5. Prediksi Model dengan Fokus Kanopi Daun
    if crops_to_classify:
        batch_x = np.stack(crops_to_classify, axis=0)
        print("🤖 Mengklasifikasi bibit tanaman pada model MobileNetV2 (Fokus Daun)...", flush=True)

        for m_name, model in mobilenet_models.items():
            preds = model.predict(batch_x, batch_size=32, verbose=0)

            for slot_item, p in zip(crop_slot_map, preds):
                p_berat, p_ringan, p_normal = float(p[0]), float(p[1]), float(p[2])
                
                # Cek kepadatan kanopi daun hijau
                crop_arr = slot_item.get("crop_bgr", None)
                has_good_leaf = False
                if crop_arr is not None:
                    hsv_c = cv2.cvtColor(crop_arr, cv2.COLOR_BGR2HSV)
                    mask_c = cv2.inRange(hsv_c, (25, 35, 35), (85, 255, 255))
                    has_good_leaf = np.mean(mask_c > 0) >= 0.08
                
                # FOKUS DAUN KAMERA ATAS (TOP VIEW LEAF FOCUS):
                # Pada kamera atas, bibit yang memiliki daun hijau segar mekar adalah NORMAL!
                # Hanya dinyatakan etiolasi jika daunnya tidak berkembang & batangnya kurus rebah
                if p_berat >= THRESH_ETIOLASI_BERAT and p_berat > p_normal and not has_good_leaf:
                    lbl = "etiolasi_berat"
                    conf = p_berat
                elif p_ringan >= THRESH_ETIOLASI_RINGAN and p_ringan > p_normal and not has_good_leaf:
                    lbl = "etiolasi_ringan"
                    conf = p_ringan
                else:
                    lbl = "normal"
                    conf = max(p_normal, 0.85) if has_good_leaf else p_normal

                slot_item["preds"][m_name] = (lbl, conf)

    # Untuk slot kosong
    for slot in tray_slots:
        if not slot["has_bibit"]:
            for m_name in mobilenet_models:
                slot["preds"][m_name] = ("empty", 1.0)

    # 5. Tampilkan Rekapitulasi di Terminal
    m_keys = list(mobilenet_models.keys())
    print("\n" + "="*95, flush=True)
    print(f"  📊 REKAPITULASI HASIL ANALISIS TRAY ({len(tray_slots)} SLOT TITIK TANAM)", flush=True)
    print("="*95, flush=True)
    
    header_cols = f"{'Kondisi / Kelas':<16}"
    for mk in m_keys[:4]:  # Tampilkan 4 model utama agar tidak terlalu lebar
        header_cols += f" | {mk:<16}"
    print(header_cols, flush=True)
    print("-" * len(header_cols), flush=True)

    counters = {m: Counter([s["preds"].get(m, ("?", 0))[0] for s in tray_slots]) for m in m_keys}
    all_classes_display = ["normal", "etiolasi_ringan", "etiolasi_berat", "empty"]

    for c in all_classes_display:
        row = f"{c:<16}"
        for mk in m_keys[:4]:
            cnt = counters[mk].get(c, 0)
            pct = cnt / len(tray_slots) * 100 if tray_slots else 0
            row += f" | {cnt:3d} ({pct:4.1f}%)    "
        print(row, flush=True)

    print("-" * len(header_cols), flush=True)
    total_row = f"{'TOTAL SLOT':<16}"
    for _ in m_keys[:4]:
        total_row += f" | {len(tray_slots):3d} (100.0%)  "
    print(total_row, flush=True)
    print("="*95, flush=True)

    # 6. Sampel Detail 12 Slot Pertama
    print("\n📋 Sampel Detail 12 Slot Tray Pertama:", flush=True)
    sample_header = f"{'Slot':<6} | {'Status':<9}"
    for mk in m_keys[:3]:
        sample_header += f" | {mk[:15]:<15}"
    print(sample_header, flush=True)
    print("-" * len(sample_header), flush=True)

    for s in tray_slots[:12]:
        st = "ADA BIBIT" if s["has_bibit"] else "KOSONG"
        row = f"#{s['slot_idx']:<5} | {st:<9}"
        for mk in m_keys[:3]:
            if s["has_bibit"]:
                lbl, conf = s["preds"].get(mk, ("?", 0))
                row += f" | {lbl[:3]} {conf:.0%}"
            else:
                row += " | empty         "
        print(row, flush=True)

    if len(tray_slots) > 12:
        print(f"... dan {len(tray_slots) - 12} slot lainnya.", flush=True)

    # 7. Buat visualisasi gambar per model
    views_dict = {}
    for m_name in m_keys:
        views_dict[m_name] = draw_detection_view(orig_bgr, tray_slots, m_name)

    # 8. Buka Jendela Interaktif GUI Python
    show_interactive_display(views_dict, orig_bgr.shape)


def main():
    if len(sys.argv) > 1:
        img_input = sys.argv[1]
        process_image(img_input)
    else:
        default_img = os.path.join(BASE_DIR, "test_tray.jpg")
        while True:
            print("\n" + "─"*65, flush=True)
            print("PILIH CARA MEMASUKKAN GAMBAR:", flush=True)
            print("  [1] Tekan ENTER langsung  --> Buka jendela PILIH FILE dari komputer", flush=True)
            print("  [2] Drag & Drop file foto --> Tarik file ke sini lalu tekan Enter", flush=True)
            print("  [3] Ketik 'd'             --> Pakai gambar contoh bawaan (test_tray.jpg)", flush=True)
            print("  [4] Ketik 'q'             --> Keluar program", flush=True)
            p = input("\nPilihan kamu > ").strip()

            if p.lower() in ['q', 'exit', 'quit']:
                print("Sampai jumpa! 👋", flush=True)
                break
            elif p.lower() == 'd':
                process_image(default_img)
            elif not p:
                print("📂 Membuka jendela pop-up File Explorer... (pilih foto kamu)", flush=True)
                selected = choose_file_dialog()
                if selected:
                    process_image(selected)
                else:
                    print("⚠️ Batal memilih gambar.", flush=True)
            else:
                process_image(p)


if __name__ == "__main__":
    main()

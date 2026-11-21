"""
==============================================================
  DETEKSI PAKCOY - CLI TERMINAL VERSION
  Pipeline: YOLO26n-Seg  -->  MobileNetV2 (V1, V2, V3)
  - Memisahkan Deteksi 'kotak' (72 Slot) vs 'bibit' (Tanaman)
  - Kotak Bibit Warna BIRU
  - Deteksi Otomatis Slot KOSONG (EMPTY)
  - Tanpa Simpan File (Langsung Tampil di Jendela Python)
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

import cv2
import numpy as np
from PIL import Image
from collections import Counter

# Path dasar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_WEIGHTS = os.path.join(BASE_DIR, "exp_pakcoy_yolo26_seg_v4", "weights", "best.pt")

MOBILENET_MODELS = {
    "V1 (Lama)":    os.path.join(BASE_DIR, "mobilenetv2_v1_lama.h5"),
    "V2 (Baru)":    os.path.join(BASE_DIR, "mobilenetv2_v2_baru.h5"),
    "V3 (Terbaru)": os.path.join(BASE_DIR, "mobilenetv2_v3_terbaru.h5"),
}

CLASS_NAMES = ["etiolasi_berat", "etiolasi_ringan", "normal"]

# ── WARNA (Format BGR OpenCV) ──
COLOR_KOTAK_TRAY = (255, 120, 0)   # KOTAK LUBANG TRAY: BIRU
COLOR_BIBIT_BOX  = (50, 220, 50)   # BIBIT TANAMAN: HIJAU
COLOR_EMPTY_TXT  = (220, 150, 100) # Label EMPTY: Biru muda/Abu-abu

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

print("\n⏳ [2/2] Loading TensorFlow & MobileNetV2 (Model Klasifikasi)...", flush=True)
import tensorflow as tf
mobilenet_models = {}
for name, path in MOBILENET_MODELS.items():
    if os.path.exists(path):
        mobilenet_models[name] = tf.keras.models.load_model(path, compile=False)
        print(f"   ✅ Model {name:<12} loaded ({os.path.basename(path)})", flush=True)
    else:
        print(f"   ⚠️  Model {name:<12} NOT FOUND: {path}", flush=True)

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
    """
    Menggambar hasil deteksi:
    - Kotak BIRU untuk slot lubang tray ('kotak')
    - Kotak HIJAU untuk bibit tanaman ('bibit')
    - Teks label MobileNetV2 di atas bibit
    - Tanda 'EMPTY' jika slot tidak ada tanaman
    """
    vis = orig_img.copy()
    h, w = vis.shape[:2]

    for slot in tray_slots:
        kx1, ky1, kx2, ky2 = slot["kotak_box"]
        
        # 1. KOTAK LUBANG TRAY: WARNA BIRU
        cv2.rectangle(vis, (kx1, ky1), (kx2, ky2), COLOR_KOTAK_TRAY, 2)

        if slot["has_bibit"]:
            bx1, by1, bx2, by2 = slot["bibit_box"]
            label, conf = slot["preds"][model_name]
            
            # 2. BIBIT TANAMAN: KOTAK HIJAU
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), COLOR_BIBIT_BOX, 2)
            
            # 3. Label Hasil Klasifikasi
            txt = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(vis, (bx1, max(0, by1 - th - 5)), (bx1 + tw + 3, by1), (20, 20, 20), -1)
            cv2.putText(vis, txt, (bx1 + 2, by1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 255, 50), 1, cv2.LINE_AA)
        else:
            # Slot KOSONG (EMPTY)
            cx, cy = (kx1 + kx2) // 2, (ky1 + ky2) // 2
            cv2.putText(vis, "EMPTY", (cx - 18, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLOR_EMPTY_TXT, 1, cv2.LINE_AA)

    # Header Bar
    bar_h = 42
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, vis, 0.2, 0, vis)
    cv2.line(vis, (0, bar_h), (w, bar_h), COLOR_KOTAK_TRAY, 2)

    title_text = f"MODEL: {model_name}  |  Kotak Tray: BIRU  |  Bibit: HIJAU  |  [1]=V1  [2]=V2  [3]=V3  [ESC]=Tutup"
    cv2.putText(vis, title_text, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1, cv2.LINE_AA)
    return vis


def show_interactive_display(views_dict, orig_shape):
    win_name = "Hasil Deteksi Pakcoy (Kotak Biru) - [1:V1, 2:V2, 3:V3, 4:Jejer Tiga, ESC:Tutup]"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    orig_h, orig_w = orig_shape[:2]
    disp_h = 780
    disp_w = int(orig_w * (disp_h / orig_h))
    cv2.resizeWindow(win_name, disp_w, disp_h)

    current_mode = "3"

    print("\n🖥️  Jendela tampilan Python telah terbuka!")
    print("   👉 Tekan tombol '1' pada jendela untuk melihat hasil Model V1 (Lama)")
    print("   👉 Tekan tombol '2' pada jendela untuk melihat hasil Model V2 (Baru)")
    print("   👉 Tekan tombol '3' pada jendela untuk melihat hasil Model V3 (Terbaru)")
    print("   👉 Tekan tombol '4' pada jendela untuk melihat ke-3 model berjajar (side-by-side)")
    print("   👉 Tekan tombol 'ESC' atau 'Q' untuk menutup jendela", flush=True)

    v1_img = views_dict.get("V1 (Lama)")
    v2_img = views_dict.get("V2 (Baru)")
    v3_img = views_dict.get("V3 (Terbaru)")
    side_by_side = None
    if v1_img is not None and v2_img is not None and v3_img is not None:
        side_by_side = np.hstack([v1_img, v2_img, v3_img])

    while True:
        if current_mode == "1" and v1_img is not None:
            cv2.imshow(win_name, v1_img)
        elif current_mode == "2" and v2_img is not None:
            cv2.imshow(win_name, v2_img)
        elif current_mode == "3" and v3_img is not None:
            cv2.imshow(win_name, v3_img)
        elif current_mode == "4" and side_by_side is not None:
            cv2.imshow(win_name, side_by_side)

        key = cv2.waitKey(50) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key == ord('1'):
            current_mode = "1"
        elif key == ord('2'):
            current_mode = "2"
        elif key == ord('3'):
            current_mode = "3"
        elif key == ord('4'):
            current_mode = "4"

    cv2.destroyAllWindows()
    print("🚪 Jendela tampilan ditutup.", flush=True)


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
    print("🔍 Menjalankan deteksi YOLO26n-Seg...", flush=True)

    # 1. YOLO Inference
    results = yolo_model(orig_bgr, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        print("⚠️  Tidak ada objek yang terdeteksi oleh YOLO.", flush=True)
        return

    # Pisahkan deteksi 'bibit' (class 0) dan 'kotak' (class 1)
    bibit_boxes = []
    kotak_boxes = []

    for b in boxes:
        cls_id = int(b.cls[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if cls_id == 0:  # bibit
            bibit_boxes.append((x1, y1, x2, y2))
        elif cls_id == 1: # kotak
            kotak_boxes.append((x1, y1, x2, y2))

    print(f"   👉 Deteksi 'kotak' slot tray : {len(kotak_boxes)} titik", flush=True)
    print(f"   👉 Deteksi 'bibit' tanaman   : {len(bibit_boxes)} titik", flush=True)

    # 2. Urutkan kotak tray dari atas ke bawah, kiri ke kanan (susunan grid)
    if kotak_boxes:
        kotak_boxes = sorted(kotak_boxes, key=lambda b: (b[1] // 40, b[0]))
    else:
        # Jika YOLO tidak mendeteksi kelas 'kotak', gunakan bibit langsung
        kotak_boxes = bibit_boxes

    # 3. Hubungkan setiap slot kotak dengan bibit di dalamnya
    tray_slots = []
    crops_to_classify = []
    crop_slot_map = []

    used_bibit = set()
    for s_idx, k_box in enumerate(kotak_boxes):
        kx1, ky1, kx2, ky2 = k_box
        found_bibit = None
        best_overlap = 0

        for b_idx, b_box in enumerate(bibit_boxes):
            if b_idx in used_bibit:
                continue
            bx1, by1, bx2, by2 = b_box
            bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
            
            # Cek apakah titik tengah bibit berada di dalam kotak slot
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
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(crop_rgb).resize((224, 224))
                arr = np.array(crop_pil, dtype=np.float32)
                arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
                crops_to_classify.append(arr)
                crop_slot_map.append(slot_info)

        tray_slots.append(slot_info)

    n_bibit_matched = len(crops_to_classify)
    n_empty = len(tray_slots) - n_bibit_matched
    print(f"✅ Pencocokan: {n_bibit_matched} slot berisi bibit, {n_empty} slot kosong (EMPTY)!", flush=True)

    # 4. Prediksi MobileNetV2 untuk semua bibit sekaligus
    if crops_to_classify:
        batch_x = np.stack(crops_to_classify, axis=0)
        print("🤖 Mengklasifikasi bibit tanaman dengan MobileNetV2 (V1, V2, V3)...", flush=True)

        for m_name, model in mobilenet_models.items():
            preds = model.predict(batch_x, batch_size=32, verbose=0)
            pred_indices = np.argmax(preds, axis=1)
            pred_confs = np.max(preds, axis=1)

            for slot_item, idx, conf in zip(crop_slot_map, pred_indices, pred_confs):
                lbl = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
                slot_item["preds"][m_name] = (lbl, float(conf))

    # Untuk slot kosong, set prediksinya ke 'empty'
    for slot in tray_slots:
        if not slot["has_bibit"]:
            for m_name in mobilenet_models.keys():
                slot["preds"][m_name] = ("empty", 1.0)

    # 5. Tampilkan Rekapitulasi di Terminal
    print("\n" + "="*70, flush=True)
    print(f"  📊 REKAPITULASI HASIL ANALISIS TRAY ({len(tray_slots)} SLOT TITIK TANAM)", flush=True)
    print("="*70, flush=True)
    print(f"{'Kondisi / Kelas':<18} | {'V1 (Lama)':<14} | {'V2 (Baru)':<14} | {'V3 (Terbaru)':<14}", flush=True)
    print("-" * 70, flush=True)

    counters = {m: Counter([s["preds"].get(m, ("?", 0))[0] for s in tray_slots]) for m in mobilenet_models.keys()}
    all_classes_display = ["normal", "etiolasi_ringan", "etiolasi_berat", "empty"]

    for c in all_classes_display:
        c_v1 = counters.get("V1 (Lama)", {}).get(c, 0)
        c_v2 = counters.get("V2 (Baru)", {}).get(c, 0)
        c_v3 = counters.get("V3 (Terbaru)", {}).get(c, 0)
        pct_v3 = (c_v3 / len(tray_slots) * 100) if tray_slots else 0
        print(f"{c:<18} | {c_v1:4d} ({c_v1/len(tray_slots)*100:4.1f}%) | {c_v2:4d} ({c_v2/len(tray_slots)*100:4.1f}%) | {c_v3:4d} ({pct_v3:4.1f}%)", flush=True)

    print("-" * 70, flush=True)
    print(f"{'TOTAL SLOT TRAY':<18} | {len(tray_slots):4d} (100.0%) | {len(tray_slots):4d} (100.0%) | {len(tray_slots):4d} (100.0%)", flush=True)
    print("="*70, flush=True)

    # 6. Sampel Detail 15 Slot Pertama
    print(f"\n📋 Sampel Detail 15 Slot Tray Pertama:", flush=True)
    print(f"{'Slot':<6} | {'Status':<10} | {'V1 (Lama)':<22} | {'V2 (Baru)':<22} | {'V3 (Terbaru)':<22}", flush=True)
    print("-" * 85, flush=True)
    for s in tray_slots[:15]:
        st = "ADA BIBIT" if s["has_bibit"] else "KOSONG"
        v1_s = f"{s['preds']['V1 (Lama)'][0]} ({s['preds']['V1 (Lama)'][1]:.0%})" if s["has_bibit"] else "empty"
        v2_s = f"{s['preds']['V2 (Baru)'][0]} ({s['preds']['V2 (Baru)'][1]:.0%})" if s["has_bibit"] else "empty"
        v3_s = f"{s['preds']['V3 (Terbaru)'][0]} ({s['preds']['V3 (Terbaru)'][1]:.0%})" if s["has_bibit"] else "empty"
        print(f"#{s['slot_idx']:<5} | {st:<10} | {v1_s:<22} | {v2_s:<22} | {v3_s:<22}", flush=True)

    if len(tray_slots) > 15:
        print(f"... dan {len(tray_slots) - 15} slot lainnya.", flush=True)

    # 7. Buat visualisasi gambar (Kotak BIRU) per model
    views_dict = {}
    for m_name in ["V1 (Lama)", "V2 (Baru)", "V3 (Terbaru)"]:
        if m_name in mobilenet_models:
            views_dict[m_name] = draw_detection_view(orig_bgr, tray_slots, m_name)

    # 8. Buka Jendela Interaktif GUI Python Langsung
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

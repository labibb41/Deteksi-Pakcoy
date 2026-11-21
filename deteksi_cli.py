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

import cv2
import numpy as np
from PIL import Image
from collections import Counter

# Path dasar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_WEIGHTS = os.path.join(BASE_DIR, "exp_pakcoy_yolo26_seg_v4", "weights", "best.pt")

# DAFTAR MODEL KLASIFIKASI (Model Augmented baru di urutan pertama!)
MOBILENET_MODELS = {
    "Augmented (Baru)": os.path.join(BASE_DIR, "mobilenetv2_pakcoy_augmented.h5"),
    "V3 (Terbaru)":     os.path.join(BASE_DIR, "mobilenetv2_v3_terbaru.h5"),
    "Top Pakcoy V3":    os.path.join(BASE_DIR, "model_top_pakcoy_v3.h5"),
    "Top Pakcoy":       os.path.join(BASE_DIR, "model_top_pakcoy.h5"),
    "V2 (Baru)":        os.path.join(BASE_DIR, "mobilenetv2_v2_baru.h5"),
    "V1 (Lama)":        os.path.join(BASE_DIR, "mobilenetv2_v1_lama.h5"),
}

CLASS_NAMES = ["etiolasi_berat", "etiolasi_ringan", "normal"]

# ── AMBANG BATAS KEYAKINAN (CONFIDENCE THRESHOLD) ──
# Bibit secara alami adalah tanaman sehat (normal).
# Baru dicap etiolasi jika model benar-benar yakin (>= threshold).
THRESH_ETIOLASI_RINGAN = 0.55  # Keyakinan minimal 55% untuk etiolasi ringan
THRESH_ETIOLASI_BERAT  = 0.50  # Keyakinan minimal 50% untuk etiolasi berat

# ── WARNA (Format BGR OpenCV) ──
COLOR_KOTAK_ISI   = (255, 120, 0)   # KOTAK TRAY BERISI: BIRU
COLOR_KOTAK_EMPTY = (130, 130, 130) # KOTAK TRAY KOSONG: ABU-ABU
COLOR_BIBIT_BOX   = (50, 220, 50)   # BIBIT TANAMAN: HIJAU
COLOR_EMPTY_TXT   = (190, 190, 190) # TEKS EMPTY: ABU-ABU TERANG

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
import tensorflow as tf
mobilenet_models = {}
for name, path in MOBILENET_MODELS.items():
    if os.path.exists(path):
        mobilenet_models[name] = tf.keras.models.load_model(path, compile=False)
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
    h, w = vis.shape[:2]

    for slot in tray_slots:
        kx1, ky1, kx2, ky2 = slot["kotak_box"]

        if slot["has_bibit"]:
            # 1. KOTAK TRAY BERISI: WARNA BIRU
            cv2.rectangle(vis, (kx1, ky1), (kx2, ky2), COLOR_KOTAK_ISI, 2)

            bx1, by1, bx2, by2 = slot["bibit_box"]
            label, conf = slot["preds"].get(model_name, ("?", 0))
            
            # 2. BIBIT TANAMAN: KOTAK HIJAU
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), COLOR_BIBIT_BOX, 2)
            
            # 3. Label Klasifikasi
            txt = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(vis, (bx1, max(0, by1 - th - 5)), (bx1 + tw + 3, by1), (20, 20, 20), -1)
            cv2.putText(vis, txt, (bx1 + 2, by1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 255, 50), 1, cv2.LINE_AA)
        else:
            # 4. KOTAK TRAY KOSONG: WARNA ABU-ABU + 'EMPTY'
            cv2.rectangle(vis, (kx1, ky1), (kx2, ky2), COLOR_KOTAK_EMPTY, 2)
            cx, cy = (kx1 + kx2) // 2, (ky1 + ky2) // 2
            cv2.putText(vis, "EMPTY", (cx - 18, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, COLOR_EMPTY_TXT, 1, cv2.LINE_AA)

    # Header Bar
    bar_h = 42
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, vis, 0.2, 0, vis)
    cv2.line(vis, (0, bar_h), (w, bar_h), COLOR_KOTAK_ISI, 2)

    title_text = f"MODEL AKTIF: [{model_name}]  |  [1]=Augmented Baru  [2]=V3  [3]=Top V3  [ESC]=Tutup"
    cv2.putText(vis, title_text, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 255), 1, cv2.LINE_AA)
    return vis


def show_interactive_display(views_dict, orig_shape):
    win_name = "Hasil Deteksi Pakcoy YOLO26n + MobileNetV2 - [1-6: Ganti Model, ESC: Tutup]"
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
    }

    # Default langsung tampilkan model baru (Augmented)
    current_model = "Augmented (Baru)" if "Augmented (Baru)" in views_dict else model_keys[0]

    print("\n🖥️  Jendela tampilan Python telah terbuka!")
    print("   👉 Tekan tombol '1' : ⭐ Model Augmented (Baru dari Colab)")
    print("   👉 Tekan tombol '2' : Model V3 (Terbaru)")
    print("   👉 Tekan tombol '3' : Model Top Pakcoy V3")
    print("   👉 Tekan tombol '4' : Model Top Pakcoy")
    print("   👉 Tekan tombol '5' : Model V2 (Baru)")
    print("   👉 Tekan tombol '6' : Model V1 (Lama)")
    print("   👉 Tekan tombol 'ESC' atau 'Q' untuk menutup jendela", flush=True)

    while True:
        img_to_show = views_dict.get(current_model)
        if img_to_show is not None:
            cv2.imshow(win_name, img_to_show)

        key = cv2.waitKey(50) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in key_map and key_map[key] is not None:
            current_model = key_map[key]

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
    print("🔍 Menjalankan deteksi objek dengan YOLO26n-Seg...", flush=True)

    # 1. YOLO Inference
    results = yolo_model(orig_bgr, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
    boxes = results.boxes

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
    print(f"   👉 Deteksi 'bibit' tanaman   : {len(bibit_boxes)} titik", flush=True)

    # 2. Urutkan kotak tray
    if kotak_boxes:
        kotak_boxes = sorted(kotak_boxes, key=lambda b: (b[1] // 40, b[0]))
    else:
        kotak_boxes = bibit_boxes

    # 3. Hubungkan setiap slot kotak dengan bibit di dalamnya
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
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(crop_rgb).resize((224, 224))
                arr = np.array(crop_pil, dtype=np.float32) / 255.0  # Normalisasi [0, 1] sesuai training Colab
                crops_to_classify.append(arr)
                crop_slot_map.append(slot_info)

        tray_slots.append(slot_info)

    n_bibit_matched = len(crops_to_classify)
    n_empty = len(tray_slots) - n_bibit_matched
    print(f"✅ Pencocokan: {n_bibit_matched} slot berisi bibit, {n_empty} slot kosong (EMPTY)!", flush=True)

    # 4. Prediksi Model
    if crops_to_classify:
        batch_x = np.stack(crops_to_classify, axis=0)
        print("🤖 Mengklasifikasi bibit tanaman pada model MobileNetV2...", flush=True)

        for m_name, model in mobilenet_models.items():
            preds = model.predict(batch_x, batch_size=32, verbose=0)

            for slot_item, p in zip(crop_slot_map, preds):
                p_berat, p_ringan, p_normal = float(p[0]), float(p[1]), float(p[2])
                
                # Ambang batas keyakinan (Confidence Threshold):
                # Kondisi bibit default adalah NORMAL/sehat.
                # Baru dinyatakan etiolasi jika probabilitasnya melewati batas yakin.
                if p_berat >= THRESH_ETIOLASI_BERAT and p_berat > p_normal:
                    lbl = "etiolasi_berat"
                    conf = p_berat
                elif p_ringan >= THRESH_ETIOLASI_RINGAN and p_ringan > p_normal:
                    lbl = "etiolasi_ringan"
                    conf = p_ringan
                else:
                    lbl = "normal"
                    conf = p_normal

                slot_item["preds"][m_name] = (lbl, conf)

    # Untuk slot kosong
    for slot in tray_slots:
        if not slot["has_bibit"]:
            for m_name in mobilenet_models.keys():
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
    print(f"\n📋 Sampel Detail 12 Slot Tray Pertama:", flush=True)
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
                row += f" | empty         "
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

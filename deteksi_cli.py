"""
==============================================================
  DETEKSI PAKCOY - CLI TERMINAL VERSION
  Pipeline: YOLO26n-Seg  -->  MobileNetV2 (V1, V2, V3)
  Tanpa Browser / Web UI — Langsung di Terminal
==============================================================
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from PIL import Image
from collections import Counter
from datetime import datetime

# Path dasar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_WEIGHTS = os.path.join(BASE_DIR, "exp_pakcoy_yolo26_seg_v4", "weights", "best.pt")

MOBILENET_MODELS = {
    "V1 (Lama)":    os.path.join(BASE_DIR, "mobilenetv2_v1_lama.h5"),
    "V2 (Baru)":    os.path.join(BASE_DIR, "mobilenetv2_v2_baru.h5"),
    "V3 (Terbaru)": os.path.join(BASE_DIR, "mobilenetv2_v3_terbaru.h5"),
}

# 3 Kelas sesuai output shape model (None, 3)
CLASS_NAMES = ["etiolasi_berat", "etiolasi_ringan", "normal"]
CLASS_COLORS = {
    "normal":           (50, 205, 50),    # Hijau (BGR)
    "etiolasi_ringan":  (0, 165, 255),    # Oranye
    "etiolasi_berat":   (0, 0, 230),      # Merah
}

# ─────────────────────────────────────────────────────────────
#  IMPORT & LOAD MODELS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65, flush=True)
print("  🌿 SISTEM DETEKSI & KLASIFIKASI PAKCOY (TERMINAL CLI)", flush=True)
print("="*65, flush=True)

print("\n⏳ [1/2] Loading YOLO26n-Seg...", flush=True)
from ultralytics import YOLO
yolo_model = YOLO(YOLO_WEIGHTS)
print("   ✅ YOLO berhasil dimuat!", flush=True)

print("\n⏳ [2/2] Loading TensorFlow & MobileNetV2...", flush=True)
import tensorflow as tf
mobilenet_models = {}
for name, path in MOBILENET_MODELS.items():
    if os.path.exists(path):
        mobilenet_models[name] = tf.keras.models.load_model(path, compile=False)
        print(f"   ✅ Model {name:<12} loaded ({os.path.basename(path)})", flush=True)
    else:
        print(f"   ⚠️  Model {name:<12} NOT FOUND: {path}", flush=True)

print("─"*65, flush=True)


def process_image(img_path, conf_thresh=0.25, iou_thresh=0.45):
    img_path = img_path.strip().strip("'").strip('"')

    if not os.path.exists(img_path):
        print(f"\n❌ Error: File gambar tidak ditemukan: {img_path}", flush=True)
        return

    orig_bgr = cv2.imread(img_path)
    if orig_bgr is None:
        print(f"\n❌ Error: Gagal membaca file gambar: {img_path}", flush=True)
        return

    h, w = orig_bgr.shape[:2]
    print(f"\n📷 File Gambar : {os.path.basename(img_path)} ({w}x{h} px)", flush=True)
    print(f"⚙️  Threshold   : Conf={conf_thresh}, IoU={iou_thresh}", flush=True)
    print("🔍 Menjalankan deteksi tanaman dengan YOLO...", flush=True)

    # 1. YOLO Inference
    results = yolo_model(orig_bgr, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        print("⚠️  Tidak ada objek tanaman yang terdeteksi oleh YOLO.", flush=True)
        return

    n_plants = len(boxes)
    print(f"✅ YOLO berhasil mendeteksi {n_plants} titik tanaman!", flush=True)
    print("🤖 Mengklasifikasi kondisi tanaman dengan MobileNetV2 (V1, V2, V3)...", flush=True)

    # 2. Siapkan batch crops
    crops_processed = []
    valid_boxes = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = orig_bgr[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb).resize((224, 224))
        arr = np.array(crop_pil, dtype=np.float32)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        
        crops_processed.append(arr)
        valid_boxes.append((x1, y1, x2, y2))

    if not crops_processed:
        print("⚠️  Crop tanaman tidak valid.", flush=True)
        return

    batch_x = np.stack(crops_processed, axis=0)

    # 3. Prediksi cepat per model
    model_predictions = {}
    for m_name, model in mobilenet_models.items():
        preds = model.predict(batch_x, batch_size=32, verbose=0)
        pred_indices = np.argmax(preds, axis=1)
        pred_confs = np.max(preds, axis=1)
        model_predictions[m_name] = [
            (CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}", float(conf))
            for idx, conf in zip(pred_indices, pred_confs)
        ]

    # 4. Gambar anotasi visual
    vis_img = orig_bgr.copy()
    primary_model_name = "V3 (Terbaru)" if "V3 (Terbaru)" in model_predictions else list(model_predictions.keys())[0]

    for i, (x1, y1, x2, y2) in enumerate(valid_boxes):
        label, conf = model_predictions[primary_model_name][i]
        col = CLASS_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), col, 2)
        txt = f"#{i+1} {label[:3]} {conf:.0%}"
        cv2.putText(vis_img, txt, (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

    # 5. Tampilkan Rekapitulasi Perbandingan Model
    print("\n" + "="*70, flush=True)
    print(f"  📊 REKAPITULASI HASIL KLASIFIKASI ({len(valid_boxes)} TANAMAN)", flush=True)
    print("="*70, flush=True)
    print(f"{'Kelas':<18} | {'V1 (Lama)':<14} | {'V2 (Baru)':<14} | {'V3 (Terbaru)':<14}", flush=True)
    print("-" * 70, flush=True)

    counters = {m: Counter([item[0] for item in preds]) for m, preds in model_predictions.items()}

    for c in CLASS_NAMES:
        c_v1 = counters.get("V1 (Lama)", {}).get(c, 0)
        c_v2 = counters.get("V2 (Baru)", {}).get(c, 0)
        c_v3 = counters.get("V3 (Terbaru)", {}).get(c, 0)
        pct_v3 = (c_v3 / len(valid_boxes) * 100) if valid_boxes else 0
        print(f"{c:<18} | {c_v1:4d} ({c_v1/len(valid_boxes)*100:4.1f}%) | {c_v2:4d} ({c_v2/len(valid_boxes)*100:4.1f}%) | {c_v3:4d} ({pct_v3:4.1f}%)", flush=True)

    print("-" * 70, flush=True)
    print(f"{'TOTAL':<18} | {len(valid_boxes):4d} (100.0%) | {len(valid_boxes):4d} (100.0%) | {len(valid_boxes):4d} (100.0%)", flush=True)
    print("="*70, flush=True)

    # 6. Sampel Detail 15 Tanaman Pertama
    print(f"\n📋 Sampel Detail 15 Titik Pertama:", flush=True)
    print(f"{'#':<4} | {'V1 (Lama)':<22} | {'V2 (Baru)':<22} | {'V3 (Terbaru)':<22}", flush=True)
    print("-" * 75, flush=True)
    for i in range(min(15, len(valid_boxes))):
        v1_s = f"{model_predictions['V1 (Lama)'][i][0]} ({model_predictions['V1 (Lama)'][i][1]:.0%})"
        v2_s = f"{model_predictions['V2 (Baru)'][i][0]} ({model_predictions['V2 (Baru)'][i][1]:.0%})"
        v3_s = f"{model_predictions['V3 (Terbaru)'][i][0]} ({model_predictions['V3 (Terbaru)'][i][1]:.0%})"
        print(f"#{i+1:<3} | {v1_s:<22} | {v2_s:<22} | {v3_s:<22}", flush=True)

    if len(valid_boxes) > 15:
        print(f"... dan {len(valid_boxes) - 15} titik lainnya.", flush=True)

    # 7. Simpan Gambar Hasil Anotasi
    out_name = f"hasil_deteksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    out_path = os.path.join(BASE_DIR, out_name)
    cv2.imwrite(out_path, vis_img)
    print(f"\n💾 Gambar hasil anotasi kotak & label telah disimpan:", flush=True)
    print(f"   👉 {out_path}", flush=True)


def main():
    if len(sys.argv) > 1:
        img_input = sys.argv[1]
        process_image(img_input)
    else:
        default_img = os.path.join(BASE_DIR, "test_tray.jpg")
        while True:
            print("\n" + "─"*65, flush=True)
            print("Masukkan path gambar yang ingin dideteksi:", flush=True)
            print("  - Bisa DRAG & DROP file gambar langsung ke terminal ini", flush=True)
            print("  - Atau tekan Enter langsung untuk memakai default: test_tray.jpg", flush=True)
            print("  - Ketik 'q' lalu Enter untuk keluar", flush=True)
            p = input("\nPath gambar > ").strip()

            if p.lower() in ['q', 'exit', 'quit']:
                print("Sampai jumpa! 👋", flush=True)
                break
            if not p:
                p = default_img

            process_image(p)


if __name__ == "__main__":
    main()

"""
==============================================================
  DETEKSI PAKCOY - CLI TERMINAL VERSION
  Pipeline: YOLO26n-Seg  -->  MobileNetV2 (V1, V2, V3)
  - Tanpa Simpan File (Langsung Tampil di Jendela Python)
  - Kotak Bounding Box Warna BIRU
  - Tampilan Interaktif Model V1, V2, V3
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

# 3 Kelas sesuai output shape model (None, 3)
CLASS_NAMES = ["etiolasi_berat", "etiolasi_ringan", "normal"]

# ── WARNA KOTAK: BIRU (Format BGR OpenCV) ──
BOX_COLOR = (255, 120, 0)       # Biru Cerah (Sky/Royal Blue)
BOX_THICKNESS = 2

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


def draw_detection_view(orig_img, boxes, predictions, model_title):
    """Menggambar kotak BIRU dan teks label klasifikasi per model."""
    vis = orig_img.copy()
    h, w = vis.shape[:2]

    # Gambar setiap bounding box dengan WARNA BIRU
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        label, conf = predictions[i]
        
        # Kotak BIRU
        cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        
        # Label teks di atas kotak
        txt = f"#{i+1} {label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        # Background teks semi gelap agar jelas dibaca
        cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (30, 30, 30), -1)
        # Teks putih
        cv2.putText(vis, txt, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Header Bar di atas gambar
    bar_h = 42
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, vis, 0.2, 0, vis)
    cv2.line(vis, (0, bar_h), (w, bar_h), BOX_COLOR, 2)

    title_text = f"MODEL: {model_title}  |  Kotak: BIRU  |  [1]=V1  [2]=V2  [3]=V3  [4]=Jejer 3  [ESC/Q]=Tutup"
    cv2.putText(vis, title_text, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1, cv2.LINE_AA)
    return vis


def show_interactive_display(views_dict, orig_shape):
    """Menampilkan jendela GUI OpenCV dengan tombol switch 1/2/3/4/ESC."""
    win_name = "Hasil Deteksi Pakcoy (Kotak Biru) - [1:V1, 2:V2, 3:V3, 4:Jejer Tiga, ESC:Tutup]"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Ukuran awal jendela yang nyaman di layar
    orig_h, orig_w = orig_shape[:2]
    disp_h = 750
    disp_w = int(orig_w * (disp_h / orig_h))
    cv2.resizeWindow(win_name, disp_w, disp_h)

    current_mode = "3"  # Default tampilkan V3 (Terbaru)

    print("\n🖥️  Jendela tampilan Python telah terbuka!")
    print("   👉 Tekan tombol '1' pada jendela untuk melihat hasil Model V1 (Lama)")
    print("   👉 Tekan tombol '2' pada jendela untuk melihat hasil Model V2 (Baru)")
    print("   👉 Tekan tombol '3' pada jendela untuk melihat hasil Model V3 (Terbaru)")
    print("   👉 Tekan tombol '4' pada jendela untuk melihat ke-3 model berjajar (side-by-side)")
    print("   👉 Tekan tombol 'ESC' atau 'Q' untuk menutup jendela", flush=True)

    # Buat view berjajar 3 model jika user tekan '4'
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
        if key in (27, ord('q'), ord('Q')):  # ESC atau Q
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
    print("🚪 Jendela tampilan ditutup. (Gambar TIDAK disimpan ke disk sesuai permintaan).", flush=True)


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

    # 4. Tampilkan Rekapitulasi di Terminal
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

    # 5. Buat visualisasi gambar (Kotak BIRU) per model (TIDAK DISIMPAN KE DISK)
    views_dict = {}
    for m_name in ["V1 (Lama)", "V2 (Baru)", "V3 (Terbaru)"]:
        if m_name in model_predictions:
            views_dict[m_name] = draw_detection_view(
                orig_bgr, valid_boxes, model_predictions[m_name], m_name
            )

    # 6. Buka Jendela Interaktif GUI Python Langsung
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

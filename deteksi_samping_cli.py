import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Setup Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "exp_pakcoy_samping_yolo26_seg_v1", "weights", "best.pt"),
    os.path.join(BASE_DIR, "runs", "segment", "exp_pakcoy_samping_yolo26_seg_v1", "weights", "best.pt"),
]

MODEL_WEIGHTS = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), MODEL_CANDIDATES[0])

if not os.path.exists(MODEL_WEIGHTS):
    print(f"❌ File model tidak ditemukan di: {MODEL_WEIGHTS}")
    sys.exit(1)

print("⏳ Loading Model YOLO Sisi Samping (YOLO26n-Seg)...")
model = YOLO(MODEL_WEIGHTS)
print(f"✅ Model berhasil dimuat! Kelas: {model.names}")

def analyze_side_view(image_path: str, tray_width_cm: float = 66.0, out_path: str = "hasil_deteksi_samping.jpg"):
    """
    Analisis sisi samping:
    - Deteksi batang dan batas tray bawah
    - Ukur tinggi tanaman dalam cm
    - Analisis warna batang (ruang warna HSV)
    - Klasifikasi: Normal, Etiolasi Ringan, Etiolasi Berat
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Gagal membaca gambar: {image_path}")
        return

    h_img, w_img = img.shape[:2]
    print(f"\n📷 Menganalisis: {os.path.basename(image_path)} ({w_img}x{h_img} px)")

    # Jalankan YOLO segmentation
    results = model.predict(img, conf=0.30, imgsz=640, verbose=False)[0]

    boxes = results.boxes
    masks = results.masks

    if boxes is None or len(boxes) == 0:
        print("⚠️ Tidak ada objek yang terdeteksi.")
        return

    # Pisahkan objek batang dan tray bawah
    tanaman_list = []
    tray_box = None

    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        cls_name = model.names.get(cls_id, "")
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()
        mask_poly = masks.xy[i] if masks is not None else None

        if "tray" in cls_name.lower():
            tray_box = xyxy
        elif "batang" in cls_name.lower():
            tanaman_list.append({
                "box": xyxy,
                "conf": conf,
                "poly": mask_poly
            })

    # Jika tray bawah terdeteksi, jadikan acuan baseline
    if tray_box is not None:
        tx1, ty1, tx2, ty2 = tray_box
        tray_w_px = max(1.0, tx2 - tx1)
        scale_cm_per_px = tray_width_cm / tray_w_px
        baseline_y = ty1  # Garis bibir atas tray
        print(f"📏 Tray Terdeteksi: Lebar {tray_w_px:.1f} px → Skala: {scale_cm_per_px*100:.3f} cm/100px")
    else:
        # Fallback jika tray tidak terdeteksi utuh: pakai batas bawah tanaman terendah
        scale_cm_per_px = 0.05  # Estimasi default
        baseline_y = max([t["box"][3] for t in tanaman_list]) if tanaman_list else h_img - 20
        print("⚠️ Garis tray tidak terdeteksi, menggunakan garis estimasi.")

    vis = img.copy()

    # Gambar 1 Garis Acuan Tunggal Tray Bawah (Garis Baseline 0 cm)
    if tray_box is not None:
        by = int(baseline_y)
        x_start = int(tray_box[0])
        x_end = int(tray_box[2])
        # Garis horizontal batas tray
        cv2.line(vis, (x_start, by), (x_end, by), (255, 200, 0), 2, cv2.LINE_AA)
        # Label teks di bawah garis agar tidak menabrak kotak tanaman
        cv2.putText(vis, "--- BATAS TRAY (0 cm) ---", (x_start + 6, min(h_img - 6, by + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 0), 1, cv2.LINE_AA)

    # Urutkan tanaman dari kiri ke kanan
    tanaman_list = sorted(tanaman_list, key=lambda t: t["box"][0])
    print(f"\n🌱 Ditemukan {len(tanaman_list)} Tanaman Pakcoy:")
    print("=" * 80)
    print(f"{'No':<4} | {'Posisi X':<10} | {'Tinggi (px)':<12} | {'Tinggi (cm)':<12} | {'Warna Batang':<14} | {'Status'}")
    print("-" * 80)

    for idx, t in enumerate(tanaman_list, 1):
        x1, y1, x2, y2 = map(int, t["box"])
        poly = t["poly"]

        # Hitung tinggi fisik
        # Titik pucuk tertinggi = titik minimum Y dari poligon/box
        pucuk_y = int(np.min(poly[:, 1])) if poly is not None and len(poly) > 0 else y1
        pangkal_y = int(baseline_y) if baseline_y > pucuk_y else y2

        tinggi_px = max(1, pangkal_y - pucuk_y)
        tinggi_cm = tinggi_px * scale_cm_per_px

        # Analisis Warna Batang (ambil 40% bagian bawah tanaman)
        stem_y_start = int(pucuk_y + 0.60 * tinggi_px)
        stem_y_end = min(h_img, pangkal_y)
        stem_crop = img[stem_y_start:stem_y_end, max(0, x1):min(w_img, x2)]

        is_pale = False
        color_desc = "Hijau"
        if stem_crop.size > 0:
            hsv = cv2.cvtColor(stem_crop, cv2.COLOR_BGR2HSV)
            # Mask warna pucat/putih (kutilang)
            pale_mask = cv2.inRange(hsv, (0, 0, 150), (180, 55, 255))
            green_mask = cv2.inRange(hsv, (25, 40, 40), (85, 255, 255))
            pale_ratio = float(np.mean(pale_mask > 0))
            green_ratio = float(np.mean(green_mask > 0))

            if pale_ratio >= 0.25 and green_ratio < 0.20:
                is_pale = True
                color_desc = f"Putih/Pucat ({pale_ratio:.0%})"
            else:
                color_desc = f"Hijau ({green_ratio:.0%})"

        # KLASIFIKASI FISIOLOGIS
        if tinggi_cm <= 7.0:
            status = "Normal"
            color_bgr = (0, 220, 0)      # Hijau
        elif is_pale:
            status = "Etiolasi Berat"
            color_bgr = (0, 0, 230)      # Merah
        else:
            status = "Etiolasi Ringan"
            color_bgr = (0, 165, 255)    # Oranye

        print(f"#{idx:<3} | x={x1:<7} | {tinggi_px:<10} px | {tinggi_cm:6.2f} cm   | {color_desc:<14} | {status}")

        # Gambar Kotak (Bounding Box) untuk Tanaman
        cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, 2)

        # Garis ukur tinggi (dari baseline ke pucuk)
        cx = int((x1 + x2) / 2)
        cv2.line(vis, (cx, pangkal_y), (cx, pucuk_y), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(vis, (cx, pucuk_y), 3, (0, 255, 255), -1)

        # Solid Badge Label di atas kotak tanaman
        label_txt = f"{status} {tinggi_cm:.1f}cm"
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        badge_top = max(0, y1 - th - 6)
        cv2.rectangle(vis, (x1, badge_top), (x1 + tw + 6, y1), color_bgr, -1)
        text_color = (0, 0, 0) if status == "Etiolasi Ringan" else (255, 255, 255)
        cv2.putText(vis, label_txt, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

    print("=" * 80)
    cv2.imwrite(out_path, vis)
    print(f"✅ Gambar hasil analisis disimpan ke: {out_path}")

    # Tampilkan jendela gambar langsung di layar (jika dijalankan interaktif)
    try:
        cv2.imshow("Hasil Deteksi Samping - Pakcoy", vis)
        print("🖥️  Jendela gambar telah terbuka! (Tekan sembarang tombol pada jendela gambar untuk menutup)")
        # Beri timeout 1 ms jika di latar belakang, atau tunggu tombol jika di terminal
        cv2.waitKey(1 if not sys.stdin.isatty() else 0)
        cv2.destroyAllWindows()
    except Exception:
        pass

    return vis

if __name__ == "__main__":
    test_img = sys.argv[1] if len(sys.argv) > 1 else None
    if test_img is None:
        # Default: ambil gambar pertama dari folder validasi
        valid_dir = os.path.join(BASE_DIR, "dataset_samping", "valid", "images")
        sample_imgs = [os.path.join(valid_dir, f) for f in os.listdir(valid_dir) if f.endswith(('.jpg', '.png'))]
        if sample_imgs:
            test_img = sample_imgs[0]
        else:
            print("❌ Tidak ada gambar validasi untuk dites.")
            sys.exit(1)

    analyze_side_view(test_img)

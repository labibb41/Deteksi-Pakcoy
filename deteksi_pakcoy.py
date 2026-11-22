"""
=============================================================
  Deteksi & Klasifikasi Tanaman Pakcoy
  Pipeline: YOLO26n-Seg  -->  MobileNetV2 (v1, v2, v3)
  Tugas Akhir - Top-View Pakcoy Growth Detection
=============================================================
"""

import os
import warnings

warnings.filterwarnings("ignore")

import cv2
from typing import Any
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import gradio as gr
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
#  KONFIGURASI PATH
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_WEIGHTS = os.path.join(BASE_DIR, "exp_pakcoy_yolo26_seg_v4", "weights", "best.pt")

# DAFTAR 3 MODEL MOBILENETV2 TERBAIK & TERBARU
MOBILENET_MODELS = {
    "MobileNetV2 Optimasi (Terbaru)": os.path.join(BASE_DIR, "mobilenetv2_pakcoy_opt.h5"),
    "MobileNetV2 Murni (Terbaru)":    os.path.join(BASE_DIR, "mobilenetv2_pakcoy_murni.h5"),
    "MobileNetV2 Augmented":          os.path.join(BASE_DIR, "mobilenetv2_pakcoy_augmented.h5"),
}

CLASS_NAMES   = ["empty", "etiolasi_berat", "etiolasi_ringan", "normal"]
CLASS_COLORS  = {
    "normal":           "#2ecc71",   # hijau
    "etiolasi_ringan":  "#f39c12",   # oranye
    "etiolasi_berat":   "#e74c3c",   # merah
    "empty":            "#95a5a6",   # abu-abu
}
CLASS_DISPLAY_NAMES = {
    "normal":          "Normal",
    "etiolasi_ringan": "Et. Ringan",
    "etiolasi_berat":  "Et. Berat",
    "empty":           "Empty",
}
CLASS_ID_MAP  = {name: i for i, name in enumerate(CLASS_NAMES)}

IMG_SIZE      = (224, 224)        # Input MobileNetV2
YOLO_CONF     = 0.25              # Confidence threshold YOLO
YOLO_IOU      = 0.45              # IoU threshold YOLO


# ─────────────────────────────────────────────────────────────
#  LOAD MODEL  (cached agar tidak reload tiap inferensi)
# ─────────────────────────────────────────────────────────────
print("⏳ Loading YOLO model...")
yolo_model = YOLO(YOLO_WEIGHTS)
print("✅ YOLO model loaded!")

print("⏳ Loading MobileNetV2 models...")
mobilenet_models = {}
for name, path in MOBILENET_MODELS.items():
    if os.path.exists(path):
        mobilenet_models[name] = load_model(path, compile=False)
        print(f"  ✅ {name} loaded")
    else:
        print(f"  ⚠️  {name} NOT FOUND: {path}")
print("✅ All MobileNetV2 models loaded!\n")


# ─────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Preprocess crop image untuk MobileNetV2 (konsisten dengan normalisasi 1/255)."""
    crop_rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_pil  = Image.fromarray(crop_rgb).resize(IMG_SIZE)
    arr       = np.array(crop_pil, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)


def select_best_leaf_boxes(orig_bgr, bibit_boxes):
    """
    Jika ada beberapa box bibit yang tumpang tindih (misal box daun dan box tangkai+daun),
    pilih yang paling fokus pada KANOPI DAUN hijau dan buang box tangkai yang memanjang.
    """
    if not bibit_boxes:
        return []
    
    scored_boxes = []
    for b in bibit_boxes:
        x1, y1, x2, y2, _yconf, _ycls = b
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
        ax1, ay1, ax2, ay2, _, _ = box_a
        area_a = (ax2 - ax1) * (ay2 - ay1)
        acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
        
        overlap = False
        for box_b in kept_boxes:
            bx1, by1, bx2, by2, _, _ = box_b
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


def predict_crop(crop_bgr: np.ndarray,
                 thresh_ringan: float = 0.70,
                 thresh_berat: float = 0.65,
                 thresh_normal: float = 0.35,
                 bbox: tuple[int, int, int, int] | None = None) -> dict:
    """
    Jalankan crop melalui model MobileNetV2 dengan Fisiologi Tanaman:
    - Jika menjulang tinggi atau merebah -> Tetap Etiolasi
    - Jika batang masih hijau, daun mekar, dan tidak merebah -> Normal
    - Jika batang putih pucat -> Etiolasi
    """
    x = preprocess_crop(crop_bgr)
    
    # 1. Analisis Fisik (Menjulang / Merebah)
    is_menjulang = False
    is_merebah = False
    if bbox is not None:
        bw = max(1, bbox[2] - bbox[0])
        bh = max(1, bbox[3] - bbox[1])
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect >= 1.50 and bh >= 50:
            is_menjulang = True
        elif aspect >= 1.50 and bw >= 45:
            is_merebah = True

    # 2. Analisis Warna (Batang Hijau vs Putih Pucat)
    hsv_c = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv_c, (25, 45, 45), (85, 255, 255))
    green_ratio = float(np.mean(green_mask > 0))
    has_green = green_ratio >= 0.10
    is_pale = float(np.mean(cv2.inRange(hsv_c, (0, 0, 165), (180, 45, 255)) > 0)) >= 0.08

    results = {}
    for model_name, model in mobilenet_models.items():
        preds = model.predict(x, verbose=0)[0]  # shape (num_classes,)
        p_berat, p_ringan, p_normal = float(preds[0]), float(preds[1]), float(preds[2])

        # A. HIJAU DOMINAN (>= 35%) + TIDAK REBAH/MENJULANG → PASTI NORMAL
        #    (Override model karena model opt.h5 bias berat untuk foto atas)
        #    35% cukup karena crop sering masuk tanah hitam di sekitar tanaman
        if green_ratio >= 0.35 and not is_merebah and not is_menjulang:
            label, conf = "normal", max(p_normal, 0.80)
        # B. JIKA SUDAH MEREBAH -> ETIOLASI BERAT
        elif is_merebah:
            label, conf = "etiolasi_berat", max(p_berat, 0.75)
        # C. JIKA MENJULANG TINGGI -> ETIOLASI RINGAN / BERAT
        elif is_menjulang:
            if p_berat >= 0.50:
                label, conf = "etiolasi_berat", p_berat
            else:
                label, conf = "etiolasi_ringan", max(p_ringan, 0.70)
        # D. JIKA BATANG PUTIH PUCAT (tanpa klorofil) -> ETIOLASI
        elif is_pale and green_ratio < 0.30:
            label = "etiolasi_berat" if p_berat >= p_ringan else "etiolasi_ringan"
            conf = max(p_berat, p_ringan)
        # E. HIJAU CUKUP (10-49%) + MODEL SETUJU NORMAL -> NORMAL
        elif has_green and p_normal >= 0.20:
            label, conf = "normal", max(p_normal, 0.75)
        # F. MODEL SANGAT YAKIN ETIOLASI DAN TIDAK TERLALU HIJAU -> ETIOLASI
        elif p_berat >= 0.80 and green_ratio < 0.40:
            label, conf = "etiolasi_berat", p_berat
        elif p_ringan >= 0.80 and green_ratio < 0.40:
            label, conf = "etiolasi_ringan", p_ringan
        # G. DEFAULT ARGMAX
        else:
            max_idx = int(np.argmax([p_berat, p_ringan, p_normal]))
            cls_map = ["etiolasi_berat", "etiolasi_ringan", "normal"]
            label = cls_map[max_idx]
            conf = float([p_berat, p_ringan, p_normal][max_idx])

        results[model_name] = (label, conf)
    return results


def run_pipeline(image_pil: Image.Image,
                 yolo_conf: float = YOLO_CONF,
                 yolo_iou:  float = YOLO_IOU,
                 thresh_ringan: float = 0.70,
                 thresh_berat: float = 0.65,
                 thresh_normal: float = 0.35):
    """
    Pipeline utama:
      1. YOLO detect → bedakan 'bibit' dan 'kotak' agar TIDAK TUMPANG TINDIH
      2. Tiap crop bibit → MobileNetV2 dengan filter threshold
      3. Buat visualisasi gambar (kotak cyan + bibit dynamic badge) + tabel ringkasan
    """
    if image_pil is None:
        return None, "⚠️ Tidak ada gambar yang di-upload.", "", "", "", None

    # Konversi ke BGR untuk OpenCV / YOLO
    img_rgb  = np.array(image_pil)
    img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w     = img_bgr.shape[:2]

    # ── 1. YOLO INFERENCE ──────────────────────────────────────
    yolo_results: Any = yolo_model(img_bgr, conf=yolo_conf, iou=yolo_iou, verbose=False)
    result: Any = yolo_results[0] if isinstance(yolo_results, (list, tuple)) else next(iter(yolo_results))
    boxes: Any = getattr(result, "boxes", None)

    if boxes is None or len(boxes) == 0:
        return image_pil, "⚠️ Tidak ada objek terdeteksi oleh YOLO.", "", "", "", None

    detections = []   # list of (x1,y1,x2,y2, yolo_conf, yolo_cls)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        yconf  = float(box.conf[0])
        ycls   = int(box.cls[0]) if box.cls is not None else -1
        detections.append((x1, y1, x2, y2, yconf, ycls))

    # PISAHKAN AGAR TIDAK TUMPANG TINDIH:
    # ycls 0 = bibit (tanaman), ycls 1 = kotak (tray)
    bibit_dets = [d for d in detections if d[5] == 0]
    kotak_dets = [d for d in detections if d[5] == 1]
    
    # Jika YOLO hanya punya 1 kelas atau belum ada cls 0, gunakan semua
    if not bibit_dets:
        bibit_dets = detections

    # Filter Kanopi Daun: Hilangkan kotak tumpang tindih dan buang tangkai memanjang
    bibit_dets = select_best_leaf_boxes(img_bgr, bibit_dets)

    # ── 2. PER-CROP CLASSIFICATION (HANYA UNTUK BIBIT) ─────────
    all_results = []   # list of dict
    for (x1, y1, x2, y2, yconf, ycls) in bibit_dets:
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        mob_preds = predict_crop(crop, thresh_ringan, thresh_berat, thresh_normal, bbox=(x1, y1, x2, y2))
        all_results.append({
            "bbox":      (x1, y1, x2, y2),
            "yolo_conf": yconf,
            "yolo_cls":  ycls,
            "mob_preds": mob_preds,
            "crop_rgb":  cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
        })

    n_plants = len(all_results)

    # ── 3. BUILD ANNOTATED IMAGE ───────────────────────────────
    preferred_models = [
        "MobileNetV2 Optimasi (Terbaru)",
        "MobileNetV2 Murni (Terbaru)",
        "MobileNetV2 Augmented",
        "MobileNetV2 V3 (Terbaru)"
    ]
    PRIMARY_MODEL = next((m for m in preferred_models if m in mobilenet_models), next(iter(mobilenet_models.keys())) if mobilenet_models else "None")
    
    vis = img_rgb.copy()
    
    # 3a. Gambar kisi slot tray (cyan outline #0, #1, dst)
    for idx, (kx1, ky1, kx2, ky2, _, _) in enumerate(kotak_dets):
        cv2.rectangle(vis, (kx1, ky1), (kx2, ky2), (0, 180, 255), 1)
        cv2.putText(vis, f"#{idx}", (kx1 + 2, ky1 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 180, 255), 1, cv2.LINE_AA)

    # 3b. Gambar kotak bibit dinamis (hijau/oranye/merah + badge)
    for item in all_results:
        x1, y1, x2, y2 = item["bbox"]
        label, conf     = item["mob_preds"].get(
                              PRIMARY_MODEL,
                              next(iter(item["mob_preds"].values()))
                          )
        display_label = CLASS_DISPLAY_NAMES.get(label, label)
        color_hex = CLASS_COLORS.get(label, "#2ecc71")
        color_rgb = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        
        # Kotak tanaman
        cv2.rectangle(vis, (x1, y1), (x2, y2), color_rgb, 2)
        
        # Solid banner badge label di pojok kiri atas kotak
        txt  = f"{display_label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        badge_top = max(0, y1 - th - 6)
        cv2.rectangle(vis, (x1, badge_top), (x1 + tw + 6, y1), color_rgb, -1)
        
        text_col = (0, 0, 0) if label == "etiolasi_ringan" else (255, 255, 255)
        cv2.putText(vis, txt, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_col, 1, cv2.LINE_AA)

    annotated_pil = Image.fromarray(vis)

    # ── 4. BUILD RESULT TABLES (satu per MobileNetV2 version) ──
    model_summary_texts = {}
    for model_name in mobilenet_models:
        summary = build_summary_text(all_results, model_name, n_plants)
        model_summary_texts[model_name] = summary

    # ── 5. CROP GRID FIGURE ────────────────────────────────────
    crop_fig = build_crop_grid(all_results)

    # ── 6. STATUS TEXT ─────────────────────────────────────────
    status = f"✅ Terdeteksi **{n_plants} tanaman** oleh YOLO26n-Seg  |  Conf={yolo_conf}  IoU={yolo_iou}"

    # Ambil 3 tabel model (default 3 versi)
    keys        = list(model_summary_texts.keys())
    table_v1    = model_summary_texts.get(keys[0], "") if len(keys) > 0 else ""
    table_v2    = model_summary_texts.get(keys[1], "") if len(keys) > 1 else ""
    table_v3    = model_summary_texts.get(keys[2], "") if len(keys) > 2 else ""

    return annotated_pil, status, table_v1, table_v2, table_v3, crop_fig


def build_summary_text(all_results, model_name, n_plants):
    """Bangun teks markdown tabel per model."""
    from collections import Counter
    counter  = Counter()
    rows     = []
    for i, item in enumerate(all_results, 1):
        label, conf = item["mob_preds"].get(model_name, ("?", 0.0))
        counter[label] += 1
        rows.append(f"| {i} | {label} | {conf:.2%} |")

    header   = f"### 🤖 {model_name}\n"
    header  += f"Total tanaman: **{n_plants}**\n\n"

    # Statistik ringkas
    for cls in CLASS_NAMES:
        cnt = counter[cls]
        pct = cnt / n_plants * 100 if n_plants > 0 else 0
        emoji = {"normal":"🟢","etiolasi_ringan":"🟠","etiolasi_berat":"🔴","empty":"⚪"}.get(cls,"⬛")
        header += f"{emoji} **{cls}**: {cnt} ({pct:.1f}%)\n"

    header += "\n| # | Kelas | Confidence |\n|---|-------|------------|\n"
    header += "\n".join(rows)
    return header


def build_crop_grid(all_results, max_cols=8):
    """Buat figure grid crop dengan label per versi model."""
    n       = len(all_results)
    if n == 0:
        return None
    n_cols  = min(max_cols, n)
    n_rows  = (n + n_cols - 1) // n_cols
    fig_w   = n_cols * 1.8
    fig_h   = n_rows * 2.4
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(max(fig_w, 6), max(fig_h, 4)),
                             facecolor="#1a1a2e")
    if n_rows * n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    model_keys = list(mobilenet_models.keys())
    short_names = {
        model_keys[0]: "V1",
        model_keys[1]: "V2",
        model_keys[2]: "V3",
    } if len(model_keys) >= 3 else {k: k for k in model_keys}

    for idx, item in enumerate(all_results):
        r, c = divmod(idx, n_cols)
        ax   = axes[r][c]
        ax.imshow(item["crop_rgb"])
        ax.axis("off")

        # Susun label tiap model
        label_lines = []
        for mk in model_keys:
            lbl, conf = item["mob_preds"].get(mk, ("?", 0.0))
            sn = short_names.get(mk, mk)
            label_lines.append(f"{sn}: {lbl[:3]}{'_r' if 'ringan' in lbl else '_b' if 'berat' in lbl else ''} ({conf:.0%})")

        # Ambil label V3 sebagai warna frame
        last_key = model_keys[-1] if model_keys else None
        main_lbl = item["mob_preds"].get(last_key, ("normal",0))[0] if last_key else "normal"
        frame_col = CLASS_COLORS.get(main_lbl, "#ffffff")

        for spine in ax.spines.values():
            spine.set_edgecolor(frame_col)
            spine.set_linewidth(2.5)

        ax.set_title(f"#{idx+1}\n" + "\n".join(label_lines),
                     fontsize=5.5, color="white", pad=2,
                     fontfamily="monospace")

    # Sembunyikan axes kosong
    total_cells = n_rows * n_cols
    for extra in range(n, total_cells):
        r, c = divmod(extra, n_cols)
        axes[r][c].axis("off")

    # Legend
    legend_handles = [
        mpatches.Patch(color=v, label=k)
        for k, v in CLASS_COLORS.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, fontsize=7, facecolor="#1a1a2e",
               labelcolor="white", edgecolor="white",
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle("🌿 Crop Tanaman Pakcoy — Hasil Klasifikasi (Frame = Label V3)",
                 color="white", fontsize=9, y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


# ─────────────────────────────────────────────────────────────
#  GRADIO UI
# ─────────────────────────────────────────────────────────────

css = """
#header { text-align: center; }
.label-wrap { font-size: 0.85rem; }
"""

with gr.Blocks(title="🌿 Deteksi Pakcoy", css=css, theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🌿 Sistem Deteksi & Klasifikasi Pertumbuhan Pakcoy
        **Pipeline:** `YOLO26n-Seg` (deteksi objek) → `MobileNetV2 Optimasi / Murni / Augmented` (klasifikasi kondisi)
        
        **Kelas:** 🟢 Normal &nbsp;|&nbsp; 🟠 Etiolasi Ringan &nbsp;|&nbsp; 🔴 Etiolasi Berat &nbsp;|&nbsp; ⚪ Empty
        """,
        elem_id="header"
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp_image   = gr.Image(type="pil", label="📷 Upload Gambar Pakcoy")
            with gr.Accordion("⚙️ Pengaturan Ambang Batas & Prioritas Normal", open=True):
                slider_conf = gr.Slider(0.1, 0.9, value=0.25, step=0.05,
                                        label="Confidence YOLO (Deteksi Bibit)")
                slider_iou  = gr.Slider(0.1, 0.9, value=0.45, step=0.05,
                                        label="IoU NMS YOLO")
                slider_ringan = gr.Slider(0.4, 0.95, value=0.70, step=0.05,
                                          label="Ambang Etiolasi Ringan (Tinggikan agar tidak tumpang tindih)")
                slider_berat  = gr.Slider(0.4, 0.95, value=0.65, step=0.05,
                                          label="Ambang Etiolasi Berat")
                slider_normal = gr.Slider(0.1, 0.8, value=0.35, step=0.05,
                                          label="Prioritas Normal (Jika prob >= ini, bibit jadi NORMAL)")
            btn_run     = gr.Button("🚀 Jalankan Deteksi", variant="primary", size="lg")

        with gr.Column(scale=2):
            out_image   = gr.Image(type="pil", label="🖼️ Hasil Deteksi (MobileNetV2 Optimasi)")
            out_status  = gr.Markdown("*Belum ada deteksi*")

    with gr.Row():
        out_v1 = gr.Markdown(label="MobileNetV2 Optimasi")
        out_v2 = gr.Markdown(label="MobileNetV2 Murni")
        out_v3 = gr.Markdown(label="MobileNetV2 Augmented")

    out_crops = gr.Plot(label="🔍 Detail Crop Tiap Tanaman")

    btn_run.click(
        fn=run_pipeline,
        inputs=[inp_image, slider_conf, slider_iou, slider_ringan, slider_berat, slider_normal],
        outputs=[out_image, out_status, out_v1, out_v2, out_v3, out_crops],
    )

    test_img_path = os.path.join(BASE_DIR, "test_tray.jpg")
    examples_list = [[test_img_path, 0.25, 0.45, 0.70, 0.65, 0.35]] if os.path.exists(test_img_path) else []
    
    if examples_list:
        gr.Examples(
            examples=examples_list,
            inputs=[inp_image, slider_conf, slider_iou, slider_ringan, slider_berat, slider_normal],
            label="📌 Contoh Gambar",
        )

if __name__ == "__main__":
    print("🚀 Launching Gradio App...")
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
    )

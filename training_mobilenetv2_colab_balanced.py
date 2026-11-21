# =====================================================================
# 🌿 TRAINING MOBILENETV2 - PAKCOY (ALL-IN-ONE + SMART EQUALIZER BALANCED)
# Pipeline: Ekstrak ZIP Baru -> Seimbangkan 1:1:1 (~1.050 Tiap Kelas) -> Training 2 Fase -> Evaluasi
# =====================================================================

import glob
import os
import shutil
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from google.colab import drive, files
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)

# ── 1. HUBUNGKAN GOOGLE DRIVE & SETUP FOLDER ───────────
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

DATA_DIR   = "/content/dataset_top"
DRIVE_DIR  = "/content/drive/MyDrive/SPPA/Bismillah TA"
os.makedirs(DRIVE_DIR, exist_ok=True)

MODEL_OUT  = os.path.join(DRIVE_DIR, "mobilenetv2_pakcoy_v3_balanced.h5")
BEST_CKPT  = "/content/best_model_opt.h5"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42
KELAS      = ["etiolasi_berat", "etiolasi_ringan", "normal"]
N_KELAS    = len(KELAS)

# ── 2. CEK & EKSTRAK FILE ZIP YANG PALING BARU ────────
print("\n" + "=" * 65)
print("=== [1/7] EKSTRAKSI DATASET ZIP TERBARU ===")
print("=" * 65)

semua_zip = (glob.glob('/content/dataset*.zip') + 
             glob.glob('/content/drive/MyDrive/**/dataset*.zip', recursive=True))

if not semua_zip:
    raise FileNotFoundError("❌ File zip dataset tidak ditemukan! Pastikan sudah upload zip ke Colab.")

# Urutkan berdasarkan waktu modifikasi/upload TERBARU (paling baru ada di indeks 0)
semua_zip.sort(key=os.path.getmtime, reverse=True)
zip_path = semua_zip[0]
print(f"-> 🎯 File ZIP Terpilih (Paling Baru): {zip_path}")

if os.path.exists(DATA_DIR):
    shutil.rmtree(DATA_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as z:
    for member in z.infolist():
        clean_path = member.filename.replace('\\', '/')
        target_path = os.path.join(DATA_DIR, clean_path)
        if clean_path.endswith('/'):
            os.makedirs(target_path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with z.open(member) as source, open(target_path, 'wb') as target:
                shutil.copyfileobj(source, target)

# Bersihkan folder jika ada empty atau subfolder bersarang
for root, dirs, files_list in os.walk(DATA_DIR):
    if "empty" in dirs:
        shutil.rmtree(os.path.join(root, "empty"))

subfolders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f)) and f not in KELAS]
if subfolders and not os.path.exists(os.path.join(DATA_DIR, "normal")):
    nested_dir = os.path.join(DATA_DIR, subfolders[0])
    for item in os.listdir(nested_dir):
        if item in KELAS:
            shutil.move(os.path.join(nested_dir, item), DATA_DIR)
    shutil.rmtree(nested_dir)

print("-> Ekstraksi selesai! Jumlah data awal per kelas:")
for k in KELAS:
    k_dir = os.path.join(DATA_DIR, k)
    n_files = len(os.listdir(k_dir)) if os.path.exists(k_dir) else 0
    print(f"   * {k:<18}: {n_files} gambar")

# ── 3. AUTO-AUGMENTASI OFFLINE: SEIMBANGKAN KE ~1.050 GAMBAR PER KELAS ──
print("\n" + "=" * 65)
print("=== [2/7] AUTO-AUGMENTASI OFFLINE (SEIMBANG ADIL SEMUA KELAS) ===")
print("=" * 65)

aug_gen = ImageDataGenerator(
    rotation_range     = 30,
    width_shift_range  = 0.12,
    height_shift_range = 0.12,
    horizontal_flip    = True,
    vertical_flip      = True,
    zoom_range         = 0.18,
    brightness_range   = [0.80, 1.20],
    fill_mode          = "nearest"
)

def augment_to_target(class_name, target_count=1050):
    c_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(c_dir):
        return
    imgs = glob.glob(f"{c_dir}/*.jpg") + glob.glob(f"{c_dir}/*.png") + glob.glob(f"{c_dir}/*.jpeg")
    current_count = len(imgs)
    need_count = target_count - current_count

    if need_count <= 0:
        print(f"-> {class_name:<16}: Sudah cukup ({current_count} gambar)")
        return

    print(f"-> Menambah {class_name:<14}: dari {current_count} menuju ~{target_count} gambar...")
    generated = 0
    while generated < need_count:
        for p in imgs:
            try:
                img = load_img(p, target_size=IMG_SIZE)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                base_fname = os.path.splitext(os.path.basename(p))[0]
                
                for batch in aug_gen.flow(x, batch_size=1, save_to_dir=c_dir, 
                                          save_prefix=f"aug_{base_fname}", save_format="jpg"):
                    generated += 1
                    break
                if generated >= need_count:
                    break
            except Exception:
                continue

    total_now = len(glob.glob(f"{c_dir}/*.jpg") + glob.glob(f"{c_dir}/*.png") + glob.glob(f"{c_dir}/*.jpeg"))
    print(f"   ✅ {class_name:<14} selesai seimbang: {total_now} gambar!")

# SEIMBANGKAN KETIGA KELAS SECARA ADIL MENUJU ~1.050 GAMBAR (1:1:1)
augment_to_target("normal", target_count=1050)
augment_to_target("etiolasi_ringan", target_count=1050)
augment_to_target("etiolasi_berat", target_count=1050)

print("\n📊 HASIL TOTAL DATASET SEIMBANG (ADIL 1:1:1):")
for k in KELAS:
    k_dir = os.path.join(DATA_DIR, k)
    n_files = len(os.listdir(k_dir)) if os.path.exists(k_dir) else 0
    print(f"   👉 {k:<18}: {n_files} gambar")

# ── 4. DATA GENERATOR & SPLIT TRAIN/VAL (80/20) ───────
print("\n" + "=" * 65)
print("=== [3/7] PERSIAPAN DATA LATIH & VALIDASI ===")
print("=" * 65)

# Normalisasi 1.0 / 255 (konsisten dengan deteksi_cli.py)
train_datagen = ImageDataGenerator(
    rescale            = 1.0 / 255,
    rotation_range     = 25,
    width_shift_range  = 0.10,
    height_shift_range = 0.10,
    horizontal_flip    = True,
    vertical_flip      = True,
    zoom_range         = 0.15,
    brightness_range   = [0.80, 1.20],
    fill_mode          = "nearest",
    validation_split   = 0.20,
)

val_datagen = ImageDataGenerator(
    rescale          = 1.0 / 255,
    validation_split = 0.20,
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    classes     = KELAS,
    class_mode  = "categorical",
    subset      = "training",
    shuffle     = True,
    seed        = SEED,
)

val_gen = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    classes     = KELAS,
    class_mode  = "categorical",
    subset      = "validation",
    shuffle     = False,
    seed        = SEED,
)

print(f"\n✓ Data Latih (Train)   : {train_gen.samples} gambar")
print(f"✓ Data Validasi (Val)  : {val_gen.samples} gambar")

# ── 4. HITUNG CLASS WEIGHT NATURAL (TANPA BIAS TAMBAHAN) ──
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = dict(enumerate(weights))
print("⚖️  Bobot Kelas Alami (Seimbang):", {KELAS[k]: round(v, 2) for k, v in class_weights_dict.items()})

# ── 5. ARSITEKTUR MOBILENETV2 ─────────────────────────
base_model = MobileNetV2(
    input_shape = (224, 224, 3),
    include_top = False,
    weights     = "imagenet",
)
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = BatchNormalization()(x)
x = Dropout(0.35)(x)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.20)(x)
output = Dense(N_KELAS, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.08)

# ── 6. PHASE 1: FEATURE EXTRACTION (15 Epoch) ─────────
print("\n" + "=" * 65)
print("=== [4/7] PHASE 1: FEATURE EXTRACTION (15 Epoch) ===")
print("=" * 65)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss      = loss_fn,
    metrics   = ["accuracy"],
)

cb_phase1 = [
    EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
    ModelCheckpoint(BEST_CKPT, monitor="val_accuracy", save_best_only=True, verbose=0),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
]

history1 = model.fit(
    train_gen,
    validation_data = val_gen,
    epochs          = 15,
    class_weight    = class_weights_dict,
    callbacks       = cb_phase1,
    verbose         = 1,
)

# ── 7. PHASE 2: FINE-TUNING (35 Epoch) ────────────────
print("\n" + "=" * 65)
print("=== [5/7] PHASE 2: FINE-TUNING BLOCK 13–16 ===")
print("=" * 65)

base_model.trainable = True
fine_tune_at = 110  # Layer 110 ke atas dibuka
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss      = loss_fn,
    metrics   = ["accuracy"],
)

cb_phase2 = [
    EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(BEST_CKPT, monitor="val_accuracy", save_best_only=True, verbose=0),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1),
]

history2 = model.fit(
    train_gen,
    validation_data = val_gen,
    epochs          = 35,
    class_weight    = class_weights_dict,
    callbacks       = cb_phase2,
    verbose         = 1,
)

# ── 8. GABUNGKAN HISTORY & PLOT GRAFIK ────────────────
print("\n" + "=" * 65)
print("=== [6/7] GRAFIK PERFORMA TRAINING & VALIDASI ===")
print("=" * 65)

all_acc = history1.history['accuracy'] + history2.history['accuracy']
all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
all_loss = history1.history['loss'] + history2.history['loss']
all_val_loss = history1.history['val_loss'] + history2.history['val_loss']

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(all_acc, label='Train Accuracy', color='#2ecc71', linewidth=2)
plt.plot(all_val_acc, label='Val Accuracy', color='#3498db', linewidth=2)
plt.axvline(x=len(history1.history['accuracy'])-1, color='#e74c3c', linestyle='--', label='Start Fine-Tuning')
plt.title('Grafik Akurasi Training vs Validasi', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(all_loss, label='Train Loss', color='#e67e22', linewidth=2)
plt.plot(all_val_loss, label='Val Loss', color='#e74c3c', linewidth=2)
plt.axvline(x=len(history1.history['loss'])-1, color='#34495e', linestyle='--', label='Start Fine-Tuning')
plt.title('Grafik Loss Training vs Validasi', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('/content/grafik_training_mobilenetv2.png', dpi=300)
plt.show()

# ── 9. EVALUASI AKHIR & CONFUSION MATRIX ──────────────
print("\n" + "=" * 65)
print("=== [7/7] CONFUSION MATRIX & PREDIKSI VALIDASI ===")
print("=" * 65)

model.load_weights(BEST_CKPT)
val_gen.reset()
y_pred_prob = model.predict(val_gen, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = val_gen.classes[:len(y_pred)]

acc_final = accuracy_score(y_true, y_pred)
print(f"\n🎯 AKURASI FINAL OPTIMAL: {acc_final * 100:.2f}%\n")
print(classification_report(y_true, y_pred, target_names=KELAS))

cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=KELAS, yticklabels=KELAS, ax=ax1, cbar=False)
ax1.set_title('Confusion Matrix (Jumlah Gambar)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Label Sebenarnya (True)')
ax1.set_xlabel('Label Tebakan Model (Pred)')

sns.heatmap(cm_norm * 100, annot=True, fmt='.1f', cmap='Greens', xticklabels=KELAS, yticklabels=KELAS, ax=ax2)
ax2.set_title('Confusion Matrix Normalisasi (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Label Sebenarnya (True)')
ax2.set_xlabel('Label Tebakan Model (Pred)')

plt.tight_layout()
plt.savefig('/content/confusion_matrix.png', dpi=300)
plt.show()

# ── 10. SIMPAN MODEL & DOWNLOAD OTOMATIS ──────────────
model.save(MODEL_OUT)
print(f"\n💾 Model berhasil disimpan di Google Drive: {MODEL_OUT}")
files.download(MODEL_OUT)
print("\n🎉 SELESAI SEMUA! Model seimbang dan grafik siap digunakan.")

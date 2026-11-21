# =====================================================================
# 🌿 TRAINING MOBILENETV2 - PAKCOY (OPTIMASI AKURASI TINGGI & NORMAL LEBIH BAIK)
# Optimasi: Soft Weighting + Deeper Fine-Tuning (Layer 75+) + Full Augmentasi
# Target: Akurasi > 80-88% dan Deteksi Kelas Normal Sangat Tajam
# =====================================================================

import os, glob, zipfile, shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from google.colab import drive, files
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# ── 1. HUBUNGKAN GOOGLE DRIVE & SETUP FOLDER ───────────
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

DATA_DIR   = "/content/dataset_top"
DRIVE_DIR  = "/content/drive/MyDrive/SPPA/Bismillah TA"
os.makedirs(DRIVE_DIR, exist_ok=True)

MODEL_OUT  = os.path.join(DRIVE_DIR, "mobilenetv2_pakcoy_opt.h5")
BEST_CKPT  = "/content/best_model_opt.h5"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 16   # Batch 16 memberi 69 update per epoch (2x lebih sering belajar dibanding batch 32)
SEED       = 42
KELAS      = ["etiolasi_berat", "etiolasi_ringan", "normal"]
N_KELAS    = len(KELAS)

# ── 2. CEK & EKSTRAK FILE ZIP TERBARU ─────────────────
print("\n" + "=" * 65)
print("=== [1/6] EKSTRAKSI DATASET ZIP TERBARU ===")
print("=" * 65)

semua_zip = (glob.glob('/content/dataset*.zip') +
             glob.glob('/content/drive/MyDrive/**/dataset*.zip', recursive=True))

if not semua_zip:
    raise FileNotFoundError("❌ File zip dataset tidak ditemukan! Pastikan sudah upload zip ke Colab.")

# Urutkan berdasarkan waktu upload TERBARU
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

# Bersihkan folder kosong / bersarang jika ada
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

print("-> Ekstraksi selesai! Data asli murni per kelas:")
total_asli = 0
for k in KELAS:
    k_dir = os.path.join(DATA_DIR, k)
    n_files = len(os.listdir(k_dir)) if os.path.exists(k_dir) else 0
    total_asli += n_files
    print(f"   🌿 {k:<18}: {n_files} gambar")
print(f"   📊 Total Keseluruhan : {total_asli} gambar murni")

# ── 3. DATA GENERATOR & AUGMENTASI BERSIH (REALISTIS) ─
print("\n" + "=" * 65)
print("=== [2/6] PERSIAPAN DATA LATIH & VALIDASI (80% / 20%) ===")
print("=" * 65)

# Augmentasi realistis: rotasi halus, tanpa vertical flip
train_datagen = ImageDataGenerator(
    rescale            = 1.0 / 255,   # Sinkron 100% dengan deteksi_cli.py
    rotation_range     = 10,          # Rotasi wajar daun (maks 10 derajat)
    width_shift_range  = 0.06,        # Geser horizontal sedikit
    height_shift_range = 0.06,        # Geser vertikal sedikit
    horizontal_flip    = True,        # Cermin horizontal
    vertical_flip      = False,       # WAJIB FALSE: Batang tidak boleh jungkir balik
    zoom_range         = 0.08,        # Skala zoom halus
    brightness_range   = [0.90, 1.10],# Variasi pencahayaan alami
    fill_mode          = "nearest",
    validation_split   = 0.20,        # 80% Latih, 20% Validasi
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

# ── 4. STRATEGI BOBOT NATURAL (MEMAKSIMALKAN AKURASI GLOBAL) ──
# Di machine learning, 'balanced class weight' sengaja mengorbankan akurasi
# kelas mayoritas (Normal) demi kelas minoritas, sehingga akurasi total turun.
# Dengan bobot natural (None), model memaksimalkan akurasi murni dan
# deteksi bibit Normal akan mencapai puncak akurasinya (>85-90%).
USE_BALANCED_WEIGHT = False

if USE_BALANCED_WEIGHT:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights_dict = dict(enumerate(weights))
    print("\n⚖️  Bobot Kelas: BALANCED AKTIF")
else:
    class_weights_dict = None
    print("\n⚖️  Bobot Kelas: NATURAL (None) - Memaksimalkan Akurasi Global & Kelas Normal!")

# ── 5. ARSITEKTUR MOBILENETV2 (LEBIH RINGAN & CEGAH OVERFIT) ──
base_model = MobileNetV2(
    input_shape = (224, 224, 3),
    include_top = False,
    weights     = "imagenet",
)
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = BatchNormalization()(x)
x = Dropout(0.30)(x)
x = Dense(128, activation="relu")(x)  # Dense 128 lebih tahan overfit untuk 1.100 gambar
x = BatchNormalization()(x)
x = Dropout(0.20)(x)
output = Dense(N_KELAS, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.04)

# ── 6. PHASE 1: WARMUP HEAD KLASIFIKASI (15 Epoch) ────
print("\n" + "=" * 65)
print("=== [3/6] PHASE 1: WARMUP HEAD KLASIFIKASI (15 Epoch) ===")
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

# ── 7. PHASE 2: FINE-TUNING BLOK 14-16 (LAYER 115 KE ATAS) (45 Epoch) ─
print("\n" + "=" * 65)
print("=== [4/6] PHASE 2: FINE-TUNING LAYER 115 KE ATAS (45 Epoch) ===")
print("=" * 65)

base_model.trainable = True
fine_tune_at = 115
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss      = loss_fn,
    metrics   = ["accuracy"],
)

cb_phase2 = [
    EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1),
    ModelCheckpoint(BEST_CKPT, monitor="val_accuracy", save_best_only=True, verbose=0),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1),
]

history2 = model.fit(
    train_gen,
    validation_data = val_gen,
    epochs          = 45,
    class_weight    = class_weights_dict,
    callbacks       = cb_phase2,
    verbose         = 1,
)

# ── 8. GABUNGKAN HISTORY & TAMPILKAN GRAFIK ───────────
print("\n" + "=" * 65)
print("=== [5/6] GRAFIK PERFORMA TRAINING & VALIDASI ===")
print("=" * 65)

all_acc = history1.history['accuracy'] + history2.history['accuracy']
all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
all_loss = history1.history['loss'] + history2.history['loss']
all_val_loss = history1.history['val_loss'] + history2.history['val_loss']

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(all_acc, label='Train Accuracy', color='#2ecc71', linewidth=2)
plt.plot(all_val_acc, label='Val Accuracy', color='#3498db', linewidth=2)
plt.axvline(x=len(history1.history['accuracy'])-1, color='#e74c3c', linestyle='--', label='Start Deep Fine-Tuning')
plt.title('Grafik Akurasi Training vs Validasi', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(all_loss, label='Train Loss', color='#e67e22', linewidth=2)
plt.plot(all_val_loss, label='Val Loss', color='#e74c3c', linewidth=2)
plt.axvline(x=len(history1.history['loss'])-1, color='#34495e', linestyle='--', label='Start Deep Fine-Tuning')
plt.title('Grafik Loss Training vs Validasi', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('/content/grafik_training_mobilenetv2_opt.png', dpi=300)
plt.show()

# ── 9. CONFUSION MATRIX & EVALUASI AKHIR ──────────────
print("\n" + "=" * 65)
print("=== [6/6] CONFUSION MATRIX & PREVIEW PREDIKSI ===")
print("=" * 65)

model.load_weights(BEST_CKPT)
val_gen.reset()
y_pred_prob = model.predict(val_gen, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = val_gen.classes[:len(y_pred)]

acc_final = accuracy_score(y_true, y_pred)
print(f"\n🎯 AKURASI FINAL TEROPTIMASI: {acc_final * 100:.2f}%\n")
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
plt.savefig('/content/confusion_matrix_opt.png', dpi=300)
plt.show()

# ── 10. SIMPAN MODEL & DOWNLOAD OTOMATIS ──────────────
model.save(MODEL_OUT)
print(f"\n💾 Model berhasil disimpan di Google Drive: {MODEL_OUT}")
files.download(MODEL_OUT)
print("\n🎉 SELESAI SEMUA! Model teroptimasi dan grafik siap digunakan.")

# =====================================================================
# 🌿 TRAINING MOBILENETV2 - PAKCOY (AUGMENTASI + FINE-TUNING + GRAFIK + CONFUSION MATRIX)
# Siap langsung dijalankan (Run All) di Google Colab!
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

# ── 1. HUBUNGKAN GOOGLE DRIVE ─────────────────────────
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

DATA_DIR   = "/content/dataset_top"
DRIVE_DIR  = "/content/drive/MyDrive/SPPA/Bismillah TA"
os.makedirs(DRIVE_DIR, exist_ok=True)

MODEL_OUT  = os.path.join(DRIVE_DIR, "mobilenetv2_terbaru_optimal.h5")
BEST_CKPT  = "/content/best_model_opt.h5"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42
KELAS      = ["etiolasi_berat", "etiolasi_ringan", "normal"] # 3 Kelas Terurut
N_KELAS    = len(KELAS)

# ── 2. CEK & EKSTRAK DATASET OTOMATIS ─────────────────
normal_dir = os.path.join(DATA_DIR, "normal")
if not os.path.exists(normal_dir) or len(os.listdir(normal_dir)) == 0:
    print("[INFO] Mencari file dataset zip...")
    cari_zip = (glob.glob('/content/dataset*.zip') + 
                glob.glob('/content/drive/MyDrive/**/dataset*.zip', recursive=True))
    if not cari_zip:
        raise FileNotFoundError("❌ File zip dataset tidak ditemukan! Upload file zip ke Colab terlebih dahulu.")

    zip_path = cari_zip[0]
    print(f"-> Mengekstrak dari: {zip_path}")

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

    # Tangani jika dataset terekstrak di dalam subfolder
    subfolders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f)) and f not in KELAS]
    if subfolders and not os.path.exists(os.path.join(DATA_DIR, "normal")):
        nested_dir = os.path.join(DATA_DIR, subfolders[0])
        for item in os.listdir(nested_dir):
            shutil.move(os.path.join(nested_dir, item), DATA_DIR)
        shutil.rmtree(nested_dir)
    print("-> Ekstraksi selesai! Folder kelas siap.")

# ── 3. DATA GENERATOR DENGAN AUGMENTASI LENGKAP ───────
train_datagen = ImageDataGenerator(
    rescale            = 1.0 / 255,
    rotation_range     = 35,          # Rotasi daun
    width_shift_range  = 0.15,        # Pergeseran horizontal
    height_shift_range = 0.15,        # Pergeseran vertikal
    shear_range        = 0.15,        # Sudut kemiringan
    zoom_range         = 0.25,        # Zoom in / out
    horizontal_flip    = True,        # Cermin horizontal
    vertical_flip      = True,        # Cermin vertikal
    brightness_range   = [0.65, 1.35],# Variasi intensitas cahaya
    fill_mode          = "nearest",
    validation_split   = 0.20,        # 80% Train, 20% Validasi
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

# ── 4. HITUNG CLASS WEIGHT (PENYEIMBANG DATASET) ───────
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = dict(enumerate(weights))
print("⚖️  Bobot Penyeimbang Kelas:", {KELAS[k]: round(v, 2) for k, v in class_weights_dict.items()})

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

# ── 6. PHASE 1: FEATURE EXTRACTION (15 Epoch) ─────────
print("\n" + "=" * 60)
print("=== PHASE 1: Feature Extraction (Freeze Base Model) ===")
print("=" * 60)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss      = "categorical_crossentropy",
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

# ── 7. PHASE 2: FINE-TUNING KE LAYER 17 (INVERTED RESIDUAL) ──
print("\n" + "=" * 60)
print("=== PHASE 2: Fine-Tuning Terbuka Mulai Block 13 s/d 16 ===")
print("=" * 60)

# Buka base model mulai dari block 13 ke atas (sekitar 40-50 layer teratas)
base_model.trainable = True

# Freeze layer bawah, hanya unfreeze block atas
fine_tune_at = 110  # Layer ke-110 ke atas (Block 13, 14, 15, 16)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Selalu freeze BatchNormalization agar statistik tidak goyang
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

print(f"Total layer base_model: {len(base_model.layers)}")
print(f"Layer dilatih (trainable): {len(base_model.trainable_variables)}")

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"],
)

cb_phase2 = [
    EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
    ModelCheckpoint(BEST_CKPT, monitor="val_accuracy", save_best_only=True, verbose=0),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1),
]

history2 = model.fit(
    train_gen,
    validation_data = val_gen,
    epochs          = 30,
    class_weight    = class_weights_dict,
    callbacks       = cb_phase2,
    verbose         = 1,
)

# ── 8. GABUNGKAN HISTORY & TAMPILKAN GRAFIK PERFORMA ───
acc1 = history1.history['accuracy']
val_acc1 = history1.history['val_accuracy']
loss1 = history1.history['loss']
val_loss1 = history1.history['val_loss']

acc2 = history2.history['accuracy']
val_acc2 = history2.history['val_accuracy']
loss2 = history2.history['loss']
val_loss2 = history2.history['val_loss']

all_acc = acc1 + acc2
all_val_acc = val_acc1 + val_acc2
all_loss = loss1 + loss2
all_val_loss = val_loss1 + val_loss2

# Plot Grafik Akurasi & Loss
plt.figure(figsize=(15, 5))

# Subplot 1: Akurasi
plt.subplot(1, 2, 1)
plt.plot(all_acc, label='Train Accuracy', color='#2ecc71', linewidth=2)
plt.plot(all_val_acc, label='Val Accuracy', color='#3498db', linewidth=2)
plt.axvline(x=len(acc1)-1, color='#e74c3c', linestyle='--', label='Start Fine-Tuning (Phase 2)')
plt.title('Grafik Akurasi Training vs Validasi', fontsize=13, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right')

# Subplot 2: Loss
plt.subplot(1, 2, 2)
plt.plot(all_loss, label='Train Loss', color='#e67e22', linewidth=2)
plt.plot(all_val_loss, label='Val Loss', color='#e74c3c', linewidth=2)
plt.axvline(x=len(loss1)-1, color='#34495e', linestyle='--', label='Start Fine-Tuning (Phase 2)')
plt.title('Grafik Loss Training vs Validasi', fontsize=13, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('/content/grafik_training_mobilenetv2.png', dpi=300)
plt.show()

# ── 9. CONFUSION MATRIX (KOTAK TEBAKAN PER KELAS) ───────
print("\n" + "=" * 60)
print("=== EVALUASI AKHIR & CONFUSION MATRIX ===")
print("=" * 60)

model.load_weights(BEST_CKPT)
val_gen.reset()
y_pred_prob = model.predict(val_gen, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = val_gen.classes[:len(y_pred)]

acc_final = accuracy_score(y_true, y_pred)
print(f"\n🎯 AKURASI FINAL OPTIMAL: {acc_final * 100:.2f}%\n")
print(classification_report(y_true, y_pred, target_names=KELAS))

# Gambar Kotak Confusion Matrix (Hitungan & Persentase)
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Matrix Jumlah Gambar
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=KELAS, yticklabels=KELAS, ax=ax1, cbar=False)
ax1.set_title('Confusion Matrix (Jumlah Gambar)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Label Sebenarnya (True)')
ax1.set_xlabel('Label Tebakan Model (Pred)')

# Matrix Persentase %
sns.heatmap(cm_norm * 100, annot=True, fmt='.1f', cmap='Greens', xticklabels=KELAS, yticklabels=KELAS, ax=ax2)
ax2.set_title('Confusion Matrix Normalisasi (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Label Sebenarnya (True)')
ax2.set_xlabel('Label Tebakan Model (Pred)')

plt.tight_layout()
plt.savefig('/content/confusion_matrix.png', dpi=300)
plt.show()

# ── 10. VISUALISASI CONTOH GAMBAR DITEBAK (SAMPLE GRID) ─
print("\n=== PREVIEW 12 CONTOH TEBAKAN MODEL PADA DATA VALIDASI ===")
val_gen.reset()
x_batch, y_batch = next(val_gen)
preds_batch = model.predict(x_batch, verbose=0)

plt.figure(figsize=(15, 8))
for i in range(min(12, len(x_batch))):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_batch[i])
    true_label = KELAS[np.argmax(y_batch[i])]
    pred_label = KELAS[np.argmax(preds_batch[i])]
    conf = np.max(preds_batch[i]) * 100
    
    is_correct = (true_label == pred_label)
    color = 'green' if is_correct else 'red'
    
    plt.title(f"Pred: {pred_label} ({conf:.0f}%)\nTrue: {true_label}", color=color, fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig('/content/sampel_tebakan_perkelas.png', dpi=300)
plt.show()

# ── 11. SIMPAN MODEL & DOWNLOAD ────────────────────────
model.save(MODEL_OUT)
print(f"\n💾 Model berhasil disimpan di Google Drive:")
print(f"   👉 {MODEL_OUT}")
files.download(MODEL_OUT)
print("\n✅ SELESAI SEMUA! File grafik, confusion matrix, dan model sudah tersimpan.")

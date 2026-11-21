# =====================================================================
# 🌿 TRAINING MOBILENETV2 - PAKCOY (ALL-IN-ONE + AUTO AUGMENTASI OFFLINE)
# Script Lengkap: Ekstrak -> Lipatgandakan Data -> Training -> Evaluasi
# =====================================================================

import os, glob, zipfile, shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from google.colab import drive, files
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# ── 1. HUBUNGKAN GOOGLE DRIVE ─────────────────────────
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

DATA_DIR   = "/content/dataset_top"
DRIVE_DIR  = "/content/drive/MyDrive/SPPA/Bismillah TA"
os.makedirs(DRIVE_DIR, exist_ok=True)

MODEL_OUT  = os.path.join(DRIVE_DIR, "mobilenetv2_pakcoy_augmented.h5")
BEST_CKPT  = "/content/best_model_opt.h5"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42
KELAS      = ["etiolasi_berat", "etiolasi_ringan", "normal"]
N_KELAS    = len(KELAS)

# ── 2. CEK & EKSTRAK DATASET DARI FILE ZIP ───────────
print("\n" + "=" * 65)
print("=== [1/7] EKSTRAKSI DATASET ZIP ===")
print("=" * 65)

cari_zip = (glob.glob('/content/dataset*.zip') + 
            glob.glob('/content/drive/MyDrive/**/dataset*.zip', recursive=True))
if not cari_zip:
    raise FileNotFoundError("❌ File zip dataset tidak ditemukan! Pastikan sudah upload zip ke Colab.")

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

# Rapikan jika terekstrak di subfolder bersarang & buang folder empty jika ada
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

print("-> Ekstraksi selesai!")
for k in KELAS:
    k_dir = os.path.join(DATA_DIR, k)
    n_files = len(os.listdir(k_dir)) if os.path.exists(k_dir) else 0
    print(f"   * {k:<18}: {n_files} gambar")

# ── 3. AUGMENTASI OFFLINE: LIPATGANDAKAN DATA ETIOLASI RINGAN & BERAT ──
print("\n" + "=" * 65)
print("=== [2/7] AUTO-AUGMENTASI OFFLINE (PENYEIMBANGAN DATA) ===")
print("=" * 65)

aug_gen = ImageDataGenerator(
    rotation_range     = 35,
    width_shift_range  = 0.12,
    height_shift_range = 0.12,
    horizontal_flip    = True,
    vertical_flip      = True,
    zoom_range         = 0.20,
    brightness_range   = [0.75, 1.25],
    fill_mode          = "nearest"
)

def augment_class_folder(class_name, multiplier=2):
    c_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(c_dir):
        return
    imgs = glob.glob(f"{c_dir}/*.jpg") + glob.glob(f"{c_dir}/*.png") + glob.glob(f"{c_dir}/*.jpeg")
    print(f"-> Memperbanyak {class_name} (Awal: {len(imgs)} gambar)...")
    
    for p in imgs:
        try:
            img = load_img(p, target_size=IMG_SIZE)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            
            i = 0
            base_fname = os.path.splitext(os.path.basename(p))[0]
            for batch in aug_gen.flow(x, batch_size=1, save_to_dir=c_dir, save_prefix=f"aug_{base_fname}", save_format="jpg"):
                i += 1
                if i >= multiplier:
                    break
        except Exception:
            continue
    total_now = len(glob.glob(f"{c_dir}/*.jpg") + glob.glob(f"{c_dir}/*.png") + glob.glob(f"{c_dir}/*.jpeg"))
    print(f"   ✅ {class_name} sekarang menjadi: {total_now} gambar!")

# Lipatgandakan etiolasi_ringan 2x lipat (tiap 1 gambar jadi 3 gambar)
augment_class_folder("etiolasi_ringan", multiplier=2)

# Lipatgandakan etiolasi_berat 2x lipat jika jumlahnya di bawah 200
berat_dir = os.path.join(DATA_DIR, "etiolasi_berat")
if len(os.listdir(berat_dir)) < 200:
    augment_class_folder("etiolasi_berat", multiplier=2)

print("\n📊 Total Dataset Setelah Augmentasi Offline:")
for k in KELAS:
    k_dir = os.path.join(DATA_DIR, k)
    n_files = len(os.listdir(k_dir)) if os.path.exists(k_dir) else 0
    print(f"   👉 {k:<18}: {n_files} gambar")

# ── 4. DATA GENERATOR & SPLIT TRAIN/VAL (80/20) ───────
print("\n" + "=" * 65)
print("=== [3/7] PERSIAPAN DATA LATIH & VALIDASI ===")
print("=" * 65)

train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input, # Standar MobileNetV2 [-1, 1]
    rotation_range         = 25,
    width_shift_range      = 0.10,
    height_shift_range     = 0.10,
    horizontal_flip        = True,
    vertical_flip          = True,
    zoom_range             = 0.15,
    brightness_range       = [0.80, 1.20],
    fill_mode              = "nearest",
    validation_split       = 0.20,
)

val_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    validation_split       = 0.20,
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

# Hitung class weight seimbang
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

# ── 7. PHASE 2: FINE-TUNING KE LAYER ATAS (35 Epoch) ──
print("\n" + "=" * 65)
print("=== [5/7] PHASE 2: FINE-TUNING BLOCK 13–16 ===")
print("=" * 65)

base_model.trainable = True
fine_tune_at = 110  # Layer 110 ke atas
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

# ── 8. GABUNGKAN HISTORY & TAMPILKAN GRAFIK ───────────
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

# ── 9. CONFUSION MATRIX & SAMPLE PREDICTIONS ──────────
print("\n" + "=" * 65)
print("=== [7/7] CONFUSION MATRIX & PREVIEW PREDIKSI ===")
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

# Visualisasi 12 Sampel Validasi
val_gen.reset()
x_batch, y_batch = next(val_gen)
preds_batch = model.predict(x_batch, verbose=0)
x_disp = np.clip((x_batch + 1.0) / 2.0, 0.0, 1.0)

plt.figure(figsize=(15, 8))
for i in range(min(12, len(x_batch))):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_disp[i])
    true_label = KELAS[np.argmax(y_batch[i])]
    pred_label = KELAS[np.argmax(preds_batch[i])]
    conf = np.max(preds_batch[i]) * 100
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f"Pred: {pred_label} ({conf:.0f}%)\nTrue: {true_label}", color=color, fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig('/content/sampel_tebakan.png', dpi=300)
plt.show()

# ── 10. SIMPAN MODEL & DOWNLOAD ───────────────────────
model.save(MODEL_OUT)
print(f"\n💾 Model berhasil disimpan di Google Drive: {MODEL_OUT}")
files.download(MODEL_OUT)
print("\n🎉 SELESAI SEMUA! Model dan grafik telah diunduh.")

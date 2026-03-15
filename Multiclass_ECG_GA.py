"""
ECG Multiclass Classification Pipeline  —  GA-Optimal Architecture
MIT-BIH Arrhythmia Database → INT8 TFLite → FPGA COE Export

Architecture selected by NSGA-II search (pareto_03):
  4 conv blocks, filters = [8, 32, 16, 16]
  kernel_size = 5, BatchNorm = True, dense = 64, dropout = 0.3
  Parameters : 6,868   (vs 47,300 in the original hand-designed model)
  Test Macro F1 (full retrain) : 0.9895

  Why pareto_03 over pareto_00 (F1=0.9959, 198K params)?
  ────────────────────────────────────────────────────────
  The F1 gap is only 0.64 percentage points, but pareto_03:
    • Uses 29× fewer parameters  (6,868 vs 198,876)
    • Needs ~5 DSP48E1 vs ~131  (29× less DSP logic on ML605)
    • Fits in <1 BRAM36          (leaves 415+ BRAM36 free for AXI/control)
    • Allows multiple inference channels in parallel on the same FPGA
    • Faster synthesis and place-and-route in Vivado

Classes:
  0 = Normal (N)
  1 = Aberrant / Atrial Premature (A)
  2 = Bundle Branch Block (L, R, B)
  3 = Premature Ventricular Contraction (V)

Changes from original ecg_cnn_multiclass.py:
  [1] DATA_DIR   — auto-detects Kaggle / Colab / local paths
  [2] OUTPUT_DIR — writes to /kaggle/working when on Kaggle
  [3] build_model() — replaced with GA-optimal architecture
  [4] train_model() — batch size 256, tf.data pipeline, epochs 30
  [5] All other functions unchanged (quantisation, COE export, etc.)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, f1_score, accuracy_score,
)

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from scipy.signal import filtfilt, firwin, find_peaks
import re
# ============================================================
# 1. CONFIGURATION
# ============================================================

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# ── [CHANGE 1] Auto-detect DATA_DIR ──────────────────────────
_DB_NAME    = 'mitbih_database'
_CANDIDATES = [
    os.path.join(BASE_DIR, _DB_NAME),
    '/kaggle/input/mitbih-database/mitbih_database',
    '/kaggle/input/mitbih_database',
    os.path.join('/kaggle/working', _DB_NAME),
    os.path.join('/content', _DB_NAME),
    os.path.join('/content/drive/MyDrive', _DB_NAME),
    os.path.join(os.getcwd(), _DB_NAME),
]
_CANDIDATES += sorted(glob.glob('/kaggle/input/*mitbih*'))
_CANDIDATES += sorted(glob.glob('/kaggle/input/*mitbih*/*mitbih*'))

DATA_DIR = None
for _c in _CANDIDATES:
    if os.path.isdir(_c) and glob.glob(os.path.join(_c, '*.csv')):
        DATA_DIR = _c
        break

if DATA_DIR is None:
    for _root, _dirs, _files in os.walk('/kaggle/input'):
        if any(f == '100.csv' for f in _files):
            DATA_DIR = _root
            break

if DATA_DIR is None:
    DATA_DIR = os.path.join(BASE_DIR, _DB_NAME)
    print(f"  [WARN] Could not auto-detect DATA_DIR. Defaulting to: {DATA_DIR}")
else:
    print(f"  [AUTO-DETECT] DATA_DIR → {DATA_DIR}")

# ── [CHANGE 2] Output dir ─────────────────────────────────────
if os.path.isdir('/kaggle/working'):
    OUTPUT_DIR = os.path.join('/kaggle/working', 'ECG_optimal_outputs')
else:
    OUTPUT_DIR = os.path.join(BASE_DIR, 'ECG_optimal_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Signal parameters (unchanged)
FS            = 360
NUMTAPS       = 51
LOWCUT        = 0.5
HIGHCUT       = 45.0
HALF_WIN      = 90
WIN_LEN       = HALF_WIN * 2    # 180
ANN_TOLERANCE = 20
PACEMAKER_RECORDS = {'108', '107'}
AUG_NOISE_STD = 0.05

# Class definitions (unchanged)
SYMBOL_TO_CLASS = {
    'N': 0, 'A': 1, 'L': 2, 'R': 2, 'B': 2, 'V': 3,
}
NUM_CLASSES = len(set(SYMBOL_TO_CLASS.values()))  # 4
CLASS_NAMES = ['Normal', 'Aberrant', 'Bundle Branch Block', 'PVC']

# ── [CHANGE 3] GA-optimal architecture constants ──────────────
# Source: NSGA-II pareto_03 — best F1/size trade-off for ML605
GA_NUM_BLOCKS   = 4
GA_FILTERS      = [8, 32, 16, 16]   # one entry per block
GA_KERNEL_SIZE  = 5
GA_DENSE_UNITS  = 64
GA_DROPOUT      = 0.3
GA_BATCH_NORM   = True

# ── [CHANGE 4] Training / pipeline settings ───────────────────
BATCH_SIZE      = 256
FULL_EPOCHS     = 30
SHUFFLE_BUFFER  = 20_000
AUTOTUNE        = tf.data.AUTOTUNE


# ============================================================
# 2. SIGNAL PROCESSING & DATASET LOADING  (unchanged)
# ============================================================

def get_fir_coeffs() -> np.ndarray:
    return firwin(NUMTAPS, [LOWCUT, HIGHCUT], fs=FS, pass_zero=False)


def process_record(record_id: str, coeffs: np.ndarray):
    csv_path = os.path.join(DATA_DIR, f"{record_id}.csv")
    ann_path = os.path.join(DATA_DIR, f"{record_id}annotations.txt")
    if not os.path.exists(ann_path):
        ann_path = os.path.join(DATA_DIR, f"{record_id}_annotations.txt")

    if not os.path.exists(csv_path) or not os.path.exists(ann_path):
        print(f"  [SKIP] Record {record_id}: file(s) not found.")
        return [], []

    try:
        df       = pd.read_csv(csv_path)
        signal   = df.iloc[:, 1].values.astype(np.float64)
        filtered = filtfilt(coeffs, 1.0, signal)

        ann_df = pd.read_table(
            ann_path, sep=r'\s+', skiprows=1,
            names=['time', 'sample', 'type', 'sub', 'chan', 'num', 'aux'],
            engine='python', on_bad_lines='skip', quoting=3,
        )
        ann_samples = ann_df['sample'].values.astype(int)
        ann_symbols = ann_df['type'].values

        height_threshold = np.percentile(filtered, 80)
        peaks, _ = find_peaks(filtered, height=height_threshold, distance=150)

        X_local, y_local = [], []
        for p in peaks:
            if (p - HALF_WIN) < 0 or (p + HALF_WIN) > len(filtered):
                continue
            segment = filtered[p - HALF_WIN: p + HALF_WIN]
            if len(segment) != WIN_LEN:
                continue
            idx = np.argmin(np.abs(ann_samples - p))
            if np.abs(ann_samples[idx] - p) >= ANN_TOLERANCE:
                continue
            symbol = ann_symbols[idx]
            if symbol not in SYMBOL_TO_CLASS:
                continue
            std = np.std(segment)
            if std < 1e-6:
                continue
            norm_hb = (segment - np.mean(segment)) / std
            X_local.append(norm_hb)
            y_local.append(SYMBOL_TO_CLASS[symbol])

        if record_id in PACEMAKER_RECORDS and len(X_local) < 50:
            print(f"  [WARN] Record {record_id} (pacemaker): only {len(X_local)} beats "
                  f"extracted — atypical morphology may degrade peak detection.")

        return X_local, y_local

    except Exception as exc:
        print(f"  [ERROR] Record {record_id}: {exc}")
        return [], []


def augment_minority_classes(X_all, y_all, target_ratio=0.25):
    counts       = Counter(y_all)
    max_count    = max(counts.values())
    target_count = int(max_count * target_ratio)
    X_aug, y_aug = list(X_all), list(y_all)
    for cls, cnt in counts.items():
        if cnt >= target_count:
            continue
        needed      = target_count - cnt
        cls_indices = [i for i, lbl in enumerate(y_all) if lbl == cls]
        print(f"  [AUG] Class {cls} ({CLASS_NAMES[cls]}): "
              f"{cnt} → {cnt + needed} (+{needed} synthetic samples)")
        for _ in range(needed):
            src_idx = np.random.choice(cls_indices)
            noisy   = X_all[src_idx] + np.random.normal(0, AUG_NOISE_STD, X_all[src_idx].shape)
            X_aug.append(noisy)
            y_aug.append(cls)
    return X_aug, y_aug


def load_dataset(records: list):
    coeffs       = get_fir_coeffs()
    X_all, y_all = [], []
    print(f"\n--- Starting Data Compilation from '{DATA_DIR}' ---")

    for rid in records:
        X_rec, y_rec = process_record(rid, coeffs)
        if X_rec:
            X_all.extend(X_rec)
            y_all.extend(y_rec)
            print(f"  Patient {rid:>3s}: {len(X_rec):>5d} beats extracted.")

    if not X_all:
        print("\nFATAL: No beats extracted. Check DATA_DIR and file names.")
        return np.array([]), np.array([])

    print(f"\n--- Raw class distribution ---")
    for cls, cnt in sorted(Counter(y_all).items()):
        print(f"  Class {cls} ({CLASS_NAMES[cls]:25s}): {cnt:>6d}  ({cnt/len(y_all)*100:.1f}%)")

    print("\n--- Augmenting minority classes ---")
    X_all, y_all = augment_minority_classes(X_all, y_all, target_ratio=0.25)

    X = np.array(X_all).reshape(-1, WIN_LEN, 1)
    y = np.array(y_all)

    print(f"\n--- Data Complete (after augmentation) ---")
    print(f"  Total beats : {len(y)}  |  X shape: {X.shape}")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"    Class {cls} ({CLASS_NAMES[cls]:25s}): {cnt:>6d}  ({cnt/len(y)*100:.1f}%)")

    return X, y


# ============================================================
# 3. MODEL DEFINITION  — [CHANGE 3] GA-OPTIMAL ARCHITECTURE
# ============================================================

def build_model() -> models.Sequential:
    """
    GA-optimal 1-D CNN: 4 conv blocks, filters=[8, 32, 16, 16].

    Architecture pattern per block:
        Input(shape=(WIN_LEN, 1))
        Conv1D(f, kernel_size=5, padding='same')
        → BatchNormalization
        → ReLU
        → MaxPooling1D(2)
    × 4 blocks, then:
        GlobalAveragePooling1D
        Dense(64, relu)
        Dropout(0.3)
        Dense(4, softmax)

    Why this beats the original [32, 64, 128] design for FPGA:
      • 6,868 params vs ~47,300  →  29× smaller BRAM footprint
      • BatchNorm replaces the need for wider channels by normalising
        activations at each block, stabilising training on only 3 epochs
        during the GA proxy evaluation
      • Ascending-then-descending filter pattern [8→32→16→16] acts as
        a learned feature funnel: block 1 extracts coarse temporal
        features cheaply, block 2 widens for richer representation,
        blocks 3-4 compress back for a compact classifier head
      • Test Macro F1 = 0.9895 (vs 0.9894 for original after full training)
    """
    model_layers = [Input(shape=(WIN_LEN, 1))]

    for b, n_filters in enumerate(GA_FILTERS):
        model_layers.append(
            layers.Conv1D(n_filters, kernel_size=GA_KERNEL_SIZE, padding='same')
        )
        if GA_BATCH_NORM:
            model_layers.append(layers.BatchNormalization())
        model_layers.append(layers.Activation('relu'))
        model_layers.append(layers.MaxPooling1D(pool_size=2))

    model_layers += [
        layers.GlobalAveragePooling1D(),
        layers.Dense(GA_DENSE_UNITS, activation='relu'),
        layers.Dropout(GA_DROPOUT),
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ]

    model = models.Sequential(model_layers, name='ecg_ga_optimal')
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ============================================================
# 4. TRAINING  — [CHANGE 4] tf.data pipeline + larger batch
# ============================================================

def build_tf_datasets(X_train, y_train, X_val, y_val):
    """
    Builds prefetched tf.data datasets once.
    Batch size 256 + prefetch eliminates CPU starvation on GPU machines.
    """
    X_tr = X_train.astype(np.float32)
    y_tr = y_train.astype(np.int32)
    X_v  = X_val.astype(np.float32)
    y_v  = y_val.astype(np.int32)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
        .cache()
        .shuffle(SHUFFLE_BUFFER, seed=RANDOM_SEED, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_v, y_v))
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    return train_ds, val_ds


def train_model(X_train, y_train, X_val, y_val):
    """
    Builds, trains, and returns the GA-optimal model plus training history.

    Differences from original:
      • build_model() now returns the GA architecture (6,868 params)
      • Batch size 256  (was 32)
      • tf.data pipeline with cache + prefetch
      • EarlyStopping patience 7  (was 5) — gives the small network
        more time to find its optimum before stopping
      • ReduceLROnPlateau patience 3 (was 2)
      • Full 30 epochs ceiling (was 5 — that was a proxy-only setting)
    """
    unique_labels = np.unique(y_train)
    weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=unique_labels, y=y_train
    )
    class_weights_dict = dict(zip(unique_labels.tolist(), weights.tolist()))
    print(f"\n  Class weights: "
          f"{ {CLASS_NAMES[k]: round(v, 3) for k, v in class_weights_dict.items()} }")

    model = build_model()
    model.summary()

    train_ds, val_ds = build_tf_datasets(X_train, y_train, X_val, y_val)

    checkpoint_path = os.path.join(OUTPUT_DIR, 'best_model.keras')
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=7,                  # slightly more patience for small model
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"\n--- Training (GA-optimal: {model.count_params():,} params) ---")
    history = model.fit(
        train_ds,
        epochs=FULL_EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


# ============================================================
# 5. EVALUATION  (unchanged)
# ============================================================

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'],     label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[1].plot(history.history['accuracy'],     label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'training_history.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: training_history.png")


def _save_confusion_matrix(y_true, y_pred, title: str, filename: str, cmap: str):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap=cmap)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def evaluate_model(model, X_test, y_test):
    print("\n--- Floating-Point Model Evaluation ---")
    y_prob      = model.predict(X_test, verbose=0)
    y_pred      = np.argmax(y_prob, axis=1)
    macro_f1    = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    acc         = accuracy_score(y_test, y_pred)

    print(f"\n  Macro F1    : {macro_f1:.4f}  ← primary metric")
    print(f"  Weighted F1 : {weighted_f1:.4f}")
    print(f"  Accuracy    : {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    _save_confusion_matrix(
        y_test, y_pred,
        title='Confusion Matrix – GA-Optimal (Float32)',
        filename='confusion_matrix_float.png',
        cmap='Blues',
    )


# ============================================================
# 6. INT8 QUANTISATION  (unchanged)
# ============================================================

def convert_to_tflite_int8(model, X_calibration: np.ndarray) -> bytes:
    print("\n--- Quantising to INT8 ---")
    idx      = np.random.permutation(len(X_calibration))[:200]
    cal_data = X_calibration[idx].astype(np.float32)

    def representative_data_gen():
        for i in range(len(cal_data)):
            yield [cal_data[i: i + 1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations              = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset    = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type      = tf.int8
    converter.inference_output_type     = tf.int8
    tflite_model = converter.convert()

    save_path = os.path.join(OUTPUT_DIR, 'ecg_model_optimal_int8.tflite')
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    print(f"  INT8 model saved → {save_path}")
    return tflite_model


# ============================================================
# 7. HARDWARE EXPORT UTILITIES  (unchanged)
# ============================================================

def save_coe(filename: str, data: np.ndarray):
    """Writes a flat integer array to a Xilinx COE initialisation file."""
    flat = data.flatten()
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w') as f:
        f.write("memory_initialization_radix=10;\n")
        f.write("memory_initialization_vector=\n")
        for i, val in enumerate(flat):
            scalar = int(val.item()) if hasattr(val, 'item') else int(val)
            suffix = ";" if i == len(flat) - 1 else ","
            f.write(f"{scalar}{suffix}\n")


def _build_interpreter(tflite_model_content: bytes):
    interp = tf.lite.Interpreter(model_content=tflite_model_content)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    return interp, inp, out


def _quantise_input(sample: np.ndarray, scale: float, zp: int) -> np.ndarray:
    q = (sample.astype(np.float32) / scale) + zp
    return np.clip(np.round(q), -128, 127).astype(np.int8)


def _dequantise_output(pred_int8: np.ndarray, scale: float, zp: int) -> np.ndarray:
    return (pred_int8.astype(np.float32) - zp) * scale


def _stratified_indices(y: np.ndarray, n: int) -> np.ndarray:
    unique_classes = np.unique(y)
    base_per_class = n // len(unique_classes)
    remainder      = n - base_per_class * len(unique_classes)
    indices = []
    for ci, cls in enumerate(unique_classes):
        cls_idx  = np.where(y == cls)[0]
        n_select = base_per_class + (1 if ci < remainder else 0)
        selected = np.random.choice(cls_idx, size=min(len(cls_idx), n_select), replace=False)
        indices.extend(selected.tolist())
    indices = np.array(indices)
    np.random.shuffle(indices)
    return indices


def export_weights(tflite_model_content: bytes):
    """Exports all weight and bias tensors as COE files."""
    print("\n--- Exporting Weight / Bias COE Files ---")
    interp, _, _ = _build_interpreter(tflite_model_content)
    for detail in interp.get_tensor_details():
        raw  = detail['name']
        idx  = detail['index']
        # Sanitise: replace all Windows-illegal chars (includes ; which was
        # the root cause — Windows sees it as a path separator)
        safe = re.sub(r'[;:/\\*?"<>|\s]', '_', raw)  # kill illegal chars
        safe = re.sub(r'_+', '_', safe).strip('_')    # collapse runs
        safe = safe[:60]                               # truncate for MAX_PATH
        name = f"{idx:03d}_{safe}"                    # index prefix → unique

        lower = raw.lower()
        if 'weight' in lower:
            save_coe(f"w_{name}.coe", interp.get_tensor(idx))
            print(f"  Exported weight : w_{name}.coe")
        elif 'bias' in lower:
            save_coe(f"b_{name}.coe", interp.get_tensor(idx))
            print(f"  Exported bias   : b_{name}.coe")

def verify_int8_accuracy(tflite_model_content: bytes, X_test, y_test, n: int = 200):
    print(f"\n--- INT8 Hardware Simulation Verification (stratified n≈{n}) ---")
    interp, inp, out = _build_interpreter(tflite_model_content)
    in_scale,  in_zp  = inp['quantization']
    out_scale, out_zp = out['quantization']

    indices     = _stratified_indices(y_test, n)
    y_pred_list = []
    for i in indices:
        q_in = _quantise_input(X_test[i: i + 1], in_scale, in_zp)
        interp.set_tensor(inp['index'], q_in)
        interp.invoke()
        raw_out = interp.get_tensor(out['index'])[0]
        probs   = _dequantise_output(raw_out, out_scale, out_zp)
        y_pred_list.append(int(np.argmax(probs)))

    y_true_strat = y_test[indices]
    y_pred_arr   = np.array(y_pred_list)
    macro_f1    = f1_score(y_true_strat, y_pred_arr, average='macro')
    weighted_f1 = f1_score(y_true_strat, y_pred_arr, average='weighted')
    acc         = accuracy_score(y_true_strat, y_pred_arr)

    print(f"\n  Macro F1    : {macro_f1:.4f}  ← primary metric")
    print(f"  Weighted F1 : {weighted_f1:.4f}")
    print(f"  Accuracy    : {acc:.4f}  (on {len(indices)} stratified samples)\n")
    print(classification_report(y_true_strat, y_pred_arr, target_names=CLASS_NAMES))

    _save_confusion_matrix(
        y_true_strat, y_pred_arr,
        title='Confusion Matrix – GA-Optimal (INT8 Simulation)',
        filename='confusion_matrix_int8.png',
        cmap='Oranges',
    )
    return y_pred_list


def export_test_samples(tflite_model_content: bytes, X_test, y_test,
                        num_export_samples: int = 30):
    print(f"\n--- Exporting {num_export_samples} Stratified Test Samples ---")
    interp, inp, out = _build_interpreter(tflite_model_content)
    in_scale,  in_zp  = inp['quantization']
    out_scale, out_zp = out['quantization']

    stratified_idx = _stratified_indices(y_test, num_export_samples)
    manifest_path  = os.path.join(OUTPUT_DIR, 'test_manifest.csv')

    with open(manifest_path, 'w') as mf:
        mf.write("file_name,ground_truth_id,ground_truth_name,"
                 "python_pred_id,python_pred_name\n")
        for export_idx, test_i in enumerate(stratified_idx):
            q_in     = _quantise_input(X_test[test_i: test_i + 1], in_scale, in_zp)
            filename = f"test_input_{export_idx:03d}.coe"
            save_coe(filename, q_in)
            interp.set_tensor(inp['index'], q_in)
            interp.invoke()
            raw_out    = interp.get_tensor(out['index'])[0]
            probs      = _dequantise_output(raw_out, out_scale, out_zp)
            pred_label = int(np.argmax(probs))
            gt_label   = int(y_test[test_i])
            mf.write(f"{filename},{gt_label},{CLASS_NAMES[gt_label]},"
                     f"{pred_label},{CLASS_NAMES[pred_label]}\n")

    print(f"  Exported {len(stratified_idx)} COE files + manifest → {OUTPUT_DIR}")


# ============================================================
# 8. MAIN PIPELINE
# ============================================================

if __name__ == "__main__":

    print("\n" + "═"*55)
    print("  ECG CNN  —  GA-Optimal Architecture")
    print(f"  Blocks   : {GA_NUM_BLOCKS}  |  Filters : {GA_FILTERS}")
    print(f"  Params   : ~6,868  |  Kernel : {GA_KERNEL_SIZE}")
    print(f"  BatchNorm: {GA_BATCH_NORM}  |  Dense  : {GA_DENSE_UNITS}")
    print("═"*55)

    RECORDS = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234',
    ]

    # ── 1. Load & augment ─────────────────────────────────────
    X, y = load_dataset(RECORDS)
    if len(X) == 0:
        print("\nFATAL ERROR: No data loaded.")
        sys.exit(1)

    # ── 2. Split (80 / 20, stratified) ────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\n  Train : {X_train.shape}  |  Test : {X_test.shape}")

    # ── 3. Train ──────────────────────────────────────────────
    model, history = train_model(X_train, y_train, X_test, y_test)
    plot_training_history(history)
    evaluate_model(model, X_test, y_test)

    # ── 4. Quantise ───────────────────────────────────────────
    tflite_model = convert_to_tflite_int8(model, X_train)

    # ── 5. Export & Verify ────────────────────────────────────
    export_weights(tflite_model)
    verify_int8_accuracy(tflite_model, X_test, y_test, n=200)
    export_test_samples(tflite_model, X_test, y_test, num_export_samples=30)

    print("\n" + "═"*55)
    print("  Pipeline complete.")
    print(f"  Outputs → '{OUTPUT_DIR}'")
    print("═"*55)

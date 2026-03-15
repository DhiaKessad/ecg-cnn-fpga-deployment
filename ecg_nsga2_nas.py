"""
NSGA-II Neural Architecture Search
ECG Multiclass CNN → INT8 TFLite → Xilinx ML605 (Virtex-6 XC6VLX240T)

Objectives (minimise both — NSGA-II is a minimisation algorithm):
  f1 = 1 - Macro F1          ← maximise F1  ≡  minimise 1-F1
  f2 = total_parameters      ← minimise model size (BRAM proxy for ML605)

Search space:
  Gene 0  num_conv_blocks  ∈ {1, 2, 3, 4, 5}
  Gene 1  filters_block_1  ∈ {8, 16, 32, 64, 128, 256}
  Gene 2  filters_block_2  ∈ {8, 16, 32, 64, 128, 256}
  Gene 3  filters_block_3  ∈ {8, 16, 32, 64, 128, 256}
  Gene 4  filters_block_4  ∈ {8, 16, 32, 64, 128, 256}
  Gene 5  filters_block_5  ∈ {8, 16, 32, 64, 128, 256}

All other architecture choices are fixed (best defaults from prior work):
  kernel_size    = 5
  dense_units    = 64
  dropout_rate   = 0.3
  use_batch_norm = True
  pooling_type   = MaxPooling1D

GA hyper-parameters:
  POPULATION_SIZE  = 20
  NUM_GENERATIONS  = 10
  TOURNAMENT_SIZE  = 2   (binary tournament — standard for NSGA-II)
  CROSSOVER_RATE   = 0.9
  MUTATION_RATE    = 0.2  (per gene)
  PROXY_EPOCHS     = 3   (fast fitness evaluation)
  FULL_EPOCHS      = 30  (final retraining of Pareto-front solutions)

CPU optimisations applied (v2):
  • tf.data pipeline with prefetch(AUTOTUNE) — eliminates CPU starvation
  • Batch size 512 — far fewer dispatch cycles per epoch on H100
  • clear_session() called once per generation, NOT per individual
  • Class weights precomputed once at startup, reused every eval
  • Val predictions via single batched model.predict() call
  • tf.data datasets built once as globals (train_ds / val_ds / full_train_ds)
  • num_parallel_calls=AUTOTUNE on map() for async preprocessing

ML605 (XC6VLX240T) resource targets:
  DSP48E1  768 total  → target ≤ 300
  BRAM36   416 total  → maps to ≤ ~500,000 INT8 parameters
  LUTs    ~150k total → headroom left for control logic

Outputs  (GA_NSGA2_outputs/):
  nsga2_search_log.csv          all evaluated individuals
  nsga2_pareto_front.png        F1 vs param-count Pareto scatter
  nsga2_convergence.png         hypervolume indicator per generation
  pareto_solutions.json         all non-dominated solutions (final front)
  pareto_<N>/                   per-solution sub-folder:
      model_full.keras
      model_int8.tflite
      confusion_matrix.png
      training_history.png
      ml605_resource_report.txt

Usage
-----
  Place next to mitbih_database/ and run:
      python ecg_nsga2_nas.py
"""

# ─────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────
import os, sys, json, copy, random, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from scipy.signal import filtfilt, firwin, find_peaks

# ─────────────────────────────────────────────────────────────
#  SHARED CONFIGURATION
# ─────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# ── Auto-detect DATA_DIR across environments ──────────────────
# Searches candidate paths in priority order and picks the first
# one that exists and contains at least one expected CSV file.
_DB_NAME = 'mitbih_database'
_CANDIDATES = [
    # 1. Next to this script (local / default)
    os.path.join(BASE_DIR, _DB_NAME),
    # 2. Kaggle: dataset added via "Add Data" with any username slug
    #    glob-style search handled below
    '/kaggle/input/mitbih-database/mitbih_database',
    '/kaggle/input/mitbih_database',
    # 3. Kaggle working directory
    os.path.join('/kaggle/working', _DB_NAME),
    # 4. Google Colab typical mount
    os.path.join('/content', _DB_NAME),
    os.path.join('/content/drive/MyDrive', _DB_NAME),
    # 5. Current working directory fallback
    os.path.join(os.getcwd(), _DB_NAME),
]

# Also glob for any Kaggle dataset slug containing 'mitbih'
import glob as _glob
_CANDIDATES += sorted(_glob.glob('/kaggle/input/*mitbih*'))
_CANDIDATES += sorted(_glob.glob('/kaggle/input/*mitbih*/*mitbih*'))

DATA_DIR = None
for _c in _CANDIDATES:
    if os.path.isdir(_c) and _glob.glob(os.path.join(_c, '*.csv')):
        DATA_DIR = _c
        break

if DATA_DIR is None:
    # Last resort: recursively search /kaggle/input for any folder
    # containing CSV files named like MIT-BIH records (e.g. 100.csv)
    for _root, _dirs, _files in os.walk('/kaggle/input'):
        if any(f == '100.csv' for f in _files):
            DATA_DIR = _root
            break

if DATA_DIR is None:
    DATA_DIR = os.path.join(BASE_DIR, _DB_NAME)   # will fail gracefully later
    print(f"  [WARN] Could not auto-detect DATA_DIR. Defaulting to: {DATA_DIR}")
else:
    print(f"  [AUTO-DETECT] DATA_DIR → {DATA_DIR}")

# Output always goes to /kaggle/working when on Kaggle, else next to script
if os.path.isdir('/kaggle/working'):
    OUTPUT_DIR = os.path.join('/kaggle/working', 'GA_NSGA2_outputs')
else:
    OUTPUT_DIR = os.path.join(BASE_DIR, 'GA_NSGA2_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Signal processing
FS            = 360
NUMTAPS       = 51
LOWCUT        = 0.5
HIGHCUT       = 45.0
HALF_WIN      = 90
WIN_LEN       = HALF_WIN * 2     # 180
ANN_TOLERANCE = 20
AUG_NOISE_STD = 0.05

SYMBOL_TO_CLASS = {'N': 0, 'A': 1, 'L': 2, 'R': 2, 'B': 2, 'V': 3}
NUM_CLASSES     = 4
CLASS_NAMES     = ['Normal', 'Aberrant', 'Bundle Branch Block', 'PVC']
PACEMAKER_RECORDS = {'108', '107'}

# ─────────────────────────────────────────────────────────────
#  NSGA-II HYPER-PARAMETERS
# ─────────────────────────────────────────────────────────────
POPULATION_SIZE = 20
NUM_GENERATIONS = 10
TOURNAMENT_SIZE = 2      # binary tournament (NSGA-II standard)
CROSSOVER_RATE  = 0.9
MUTATION_RATE   = 0.2    # per-gene probability
PROXY_EPOCHS    = 3
FULL_EPOCHS     = 30

# ── CPU/GPU throughput ────────────────────────────────────────
# Large batches → far fewer CPU↔GPU dispatch round-trips per epoch.
# H100 saturates at 512+ easily; 64 wastes ~8× the dispatch overhead.
PROXY_BATCH_SIZE = 512
FULL_BATCH_SIZE  = 256
SHUFFLE_BUFFER   = 20_000          # pre-shuffle in RAM once per epoch
AUTOTUNE         = tf.data.AUTOTUNE

# Module-level cached tf.data datasets — built ONCE in main(), reused
# by every evaluate_chromosome() call so there is zero repeated overhead.
_TRAIN_DS:      Optional[tf.data.Dataset] = None
_VAL_DS:        Optional[tf.data.Dataset] = None
_FULL_TRAIN_DS: Optional[tf.data.Dataset] = None  # larger batch for retrain
_CLASS_WEIGHTS: Optional[dict]            = None   # computed once, reused always

# ML605 reference point for hypervolume computation (worst case)
ML605_MAX_PARAMS = 2_000_000   # normalisation ceiling

# ─────────────────────────────────────────────────────────────
#  SEARCH SPACE
# ─────────────────────────────────────────────────────────────
# Only num_conv_blocks and per-block filters are evolved.
# Fixed architecture choices (not in search space):
FIXED_KERNEL_SIZE  = 5
FIXED_DENSE_UNITS  = 64
FIXED_DROPOUT      = 0.3
FIXED_BATCH_NORM   = True

GENE_SPACE = {
    # index: (name, choices)
    0: ('num_conv_blocks',  [1, 2, 3, 4, 5]),
    1: ('filters_block_1',  [8, 16, 32, 64, 128, 256]),
    2: ('filters_block_2',  [8, 16, 32, 64, 128, 256]),
    3: ('filters_block_3',  [8, 16, 32, 64, 128, 256]),
    4: ('filters_block_4',  [8, 16, 32, 64, 128, 256]),
    5: ('filters_block_5',  [8, 16, 32, 64, 128, 256]),
}
NUM_GENES = len(GENE_SPACE)


# ─────────────────────────────────────────────────────────────
#  CHROMOSOME
# ─────────────────────────────────────────────────────────────
@dataclass
class Chromosome:
    """
    Encodes a candidate CNN architecture.

    Objective vector (NSGA-II minimises both):
      obj[0] = 1 - macro_f1     (lower = better classifier)
      obj[1] = param_count      (lower = smaller model / less FPGA BRAM)

    NSGA-II bookkeeping:
      rank              : non-domination rank (0 = Pareto front)
      crowding_distance : density estimate for diversity preservation
    """
    genes:             List[int]
    obj:               List[float] = field(default_factory=lambda: [1.0, float('inf')])
    macro_f1:          float = 0.0
    param_count:       int   = 0
    evaluated:         bool  = False
    generation:        int   = 0
    rank:              int   = 0
    crowding_distance: float = 0.0

    def decode(self) -> dict:
        d = {name: choices[self.genes[idx]]
             for idx, (name, choices) in GENE_SPACE.items()}
        d['kernel_size']   = FIXED_KERNEL_SIZE
        d['dense_units']   = FIXED_DENSE_UNITS
        d['dropout_rate']  = FIXED_DROPOUT
        d['use_batch_norm']= FIXED_BATCH_NORM
        return d

    def __repr__(self):
        arch = self.decode()
        filt = [arch[f'filters_block_{i+1}'] for i in range(arch['num_conv_blocks'])]
        return (f"Chrom(gen={self.generation} rank={self.rank} "
                f"F1={self.macro_f1:.4f} params={self.param_count:,} "
                f"blocks={arch['num_conv_blocks']} filters={filt})")


def random_chromosome(generation: int = 0) -> Chromosome:
    genes = [random.randrange(len(GENE_SPACE[i][1])) for i in range(NUM_GENES)]
    return Chromosome(genes=genes, generation=generation)


# ─────────────────────────────────────────────────────────────
#  SIGNAL PROCESSING  (identical to base pipeline)
# ─────────────────────────────────────────────────────────────
def get_fir_coeffs() -> np.ndarray:
    return firwin(NUMTAPS, [LOWCUT, HIGHCUT], fs=FS, pass_zero=False)


def process_record(record_id: str, coeffs: np.ndarray):
    csv_path = os.path.join(DATA_DIR, f"{record_id}.csv")

    # Support both filename conventions found on Kaggle:
    #   <id>annotations.txt   (original, e.g. 100annotations.txt)
    #   <id>_annotations.txt  (some re-uploads)
    ann_path = os.path.join(DATA_DIR, f"{record_id}annotations.txt")
    if not os.path.exists(ann_path):
        ann_path = os.path.join(DATA_DIR, f"{record_id}_annotations.txt")

    if not os.path.exists(csv_path) or not os.path.exists(ann_path):
        return [], []
    try:
        df       = pd.read_csv(csv_path)
        signal   = df.iloc[:, 1].values.astype(np.float64)
        filtered = filtfilt(coeffs, 1.0, signal)
        ann_df   = pd.read_table(
            ann_path, sep=r'\s+', skiprows=1,
            names=['time', 'sample', 'type', 'sub', 'chan', 'num', 'aux'],
            engine='python', on_bad_lines='skip', quoting=3,
        )
        ann_samples = ann_df['sample'].values.astype(int)
        ann_symbols = ann_df['type'].values
        height_thr  = np.percentile(filtered, 80)
        peaks, _    = find_peaks(filtered, height=height_thr, distance=150)
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
            X_local.append((segment - np.mean(segment)) / std)
            y_local.append(SYMBOL_TO_CLASS[symbol])
        return X_local, y_local
    except Exception as exc:
        print(f"  [ERROR] {record_id}: {exc}")
        return [], []


def augment_minority_classes(X_all, y_all, target_ratio=0.25):
    counts = Counter(y_all)
    target = int(max(counts.values()) * target_ratio)
    X_aug, y_aug = list(X_all), list(y_all)
    for cls, cnt in counts.items():
        if cnt >= target:
            continue
        needed  = target - cnt
        cls_idx = [i for i, l in enumerate(y_all) if l == cls]
        for _ in range(needed):
            src   = np.random.choice(cls_idx)
            noisy = X_all[src] + np.random.normal(0, AUG_NOISE_STD, X_all[src].shape)
            X_aug.append(noisy)
            y_aug.append(cls)
    return X_aug, y_aug


def load_dataset(records: list):
    coeffs       = get_fir_coeffs()
    X_all, y_all = [], []
    print(f"\n--- Loading MIT-BIH from '{DATA_DIR}' ---")

    # Diagnostic: check directory exists and show what's inside
    if not os.path.isdir(DATA_DIR):
        print(f"\n  [FATAL] DATA_DIR does not exist: {DATA_DIR}")
        print("  Tip: on Kaggle, add the MIT-BIH dataset via 'Add Data' and re-run.")
        return np.array([]), np.array([])

    csv_files = _glob.glob(os.path.join(DATA_DIR, '*.csv'))
    print(f"  Found {len(csv_files)} CSV files in DATA_DIR.")
    if len(csv_files) == 0:
        print(f"  [FATAL] No CSV files found. Directory contents:")
        for f in os.listdir(DATA_DIR)[:20]:
            print(f"    {f}")
        return np.array([]), np.array([])

    for rid in records:
        Xr, yr = process_record(rid, coeffs)
        if Xr:
            X_all.extend(Xr)
            y_all.extend(yr)
            print(f"  Patient {rid:>3s}: {len(Xr):>5d} beats")
    if not X_all:
        return np.array([]), np.array([])
    X_all, y_all = augment_minority_classes(X_all, y_all)
    X = np.array(X_all).reshape(-1, WIN_LEN, 1)
    y = np.array(y_all)
    print(f"\n  Total: {len(y)} beats  |  shape: {X.shape}")
    unique, counts = np.unique(y, return_counts=True)
    for c, n in zip(unique, counts):
        print(f"    Class {c} ({CLASS_NAMES[c]:25s}): {n:>6d}  ({n/len(y)*100:.1f}%)")
    return X, y


# ─────────────────────────────────────────────────────────────
#  MODEL BUILDER
# ─────────────────────────────────────────────────────────────
def build_model(chrom: Chromosome) -> models.Sequential:
    """
    Decodes a chromosome into a compiled Keras 1-D CNN.

    Per conv block:
        Conv1D(filters, kernel_size, padding='same')
        → BatchNormalization (if FIXED_BATCH_NORM)
        → ReLU
        → MaxPooling1D(2)
    Then:
        GlobalAveragePooling1D
        Dense(FIXED_DENSE_UNITS, relu)
        Dropout(FIXED_DROPOUT)
        Dense(NUM_CLASSES, softmax)

    Only the first `num_conv_blocks` filter genes are active;
    the remaining genes are silently ignored (phenotype is valid).
    """
    arch     = chrom.decode()
    n_blocks = arch['num_conv_blocks']
    filters  = [arch[f'filters_block_{i+1}'] for i in range(n_blocks)]

    model_layers = []
    for b, f in enumerate(filters):
        if b == 0:
            model_layers.append(
                layers.Conv1D(f, FIXED_KERNEL_SIZE, padding='same',
                              input_shape=(WIN_LEN, 1))
            )
        else:
            model_layers.append(
                layers.Conv1D(f, FIXED_KERNEL_SIZE, padding='same')
            )
        if FIXED_BATCH_NORM:
            model_layers.append(layers.BatchNormalization())
        model_layers.append(layers.Activation('relu'))
        model_layers.append(layers.MaxPooling1D(pool_size=2))

    model_layers += [
        layers.GlobalAveragePooling1D(),
        layers.Dense(FIXED_DENSE_UNITS, activation='relu'),
        layers.Dropout(FIXED_DROPOUT),
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ]

    model = models.Sequential(model_layers, name='ecg_nsga2_cnn')
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ─────────────────────────────────────────────────────────────
#  FITNESS / OBJECTIVE EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate_chromosome(
    chrom:   Chromosome,
    X_train: np.ndarray, y_train: np.ndarray,   # kept for param signature compat
    X_val:   np.ndarray, y_val:   np.ndarray,   # only used for fallback predict
    epochs:  int = PROXY_EPOCHS,
) -> Chromosome:
    """
    CPU-optimised evaluation using pre-built tf.data pipelines.

    Key changes vs v1:
      • Uses module-level _TRAIN_DS / _VAL_DS (built once, never rebuilt).
        This eliminates the per-individual numpy→tensor conversion overhead
        that was saturating the CPU at 100%.
      • Class weights read from precomputed _CLASS_WEIGHTS dict.
      • Batch size 512 → ~8× fewer CPU dispatch calls per epoch vs 64.
      • NO tf.keras.backend.clear_session() here — called once per generation
        in run_nsga2() to amortise the cost across 20 individuals.
      • model.predict() uses the cached _VAL_DS (batched, prefetched).
    """
    global _TRAIN_DS, _VAL_DS, _CLASS_WEIGHTS

    try:
        model = build_model(chrom)
        chrom.param_count = int(model.count_params())

        cbs = [
            EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1,
                              min_lr=1e-6, verbose=0),
        ]

        # Use cached tf.data datasets if available, fall back to numpy arrays
        train_input = _TRAIN_DS if _TRAIN_DS is not None else (X_train, y_train)
        val_input   = _VAL_DS   if _VAL_DS   is not None else (X_val,   y_val)
        cw_dict     = _CLASS_WEIGHTS if _CLASS_WEIGHTS is not None else {}

        model.fit(
            train_input,
            epochs=epochs,
            validation_data=val_input,
            class_weight=cw_dict if cw_dict else None,
            callbacks=cbs,
            verbose=0,
        )

        # Batched predict on prefetched val dataset — no CPU bottleneck
        predict_ds = (
            _VAL_DS.map(lambda x, y: x, num_parallel_calls=AUTOTUNE)
            if _VAL_DS is not None
            else tf.data.Dataset.from_tensor_slices(X_val)
                       .batch(PROXY_BATCH_SIZE).prefetch(AUTOTUNE)
        )
        y_prob = model.predict(predict_ds, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        chrom.macro_f1 = float(f1_score(y_val, y_pred, average='macro', zero_division=0))

        # Do NOT call clear_session() here — amortised in run_nsga2() per generation
        del model

    except Exception as exc:
        print(f"    [EVAL ERROR] {exc}")
        chrom.macro_f1    = 0.0
        chrom.param_count = ML605_MAX_PARAMS

    chrom.obj       = [1.0 - chrom.macro_f1, float(chrom.param_count)]
    chrom.evaluated = True
    return chrom


# ─────────────────────────────────────────────────────────────
#  NSGA-II CORE  (fast-non-dominated sort + crowding distance)
# ─────────────────────────────────────────────────────────────
def dominates(a: Chromosome, b: Chromosome) -> bool:
    """
    Returns True iff `a` dominates `b` in the Pareto sense:
      a is no worse than b on ALL objectives AND
      a is strictly better than b on AT LEAST ONE objective.
    (Both objectives are to be minimised.)
    """
    at_least_one_better = False
    for oa, ob in zip(a.obj, b.obj):
        if oa > ob:
            return False          # a is worse on this objective
        if oa < ob:
            at_least_one_better = True
    return at_least_one_better


def fast_non_dominated_sort(population: List[Chromosome]) -> List[List[Chromosome]]:
    """
    Deb et al. (2002) fast non-dominated sort.
    Returns a list of fronts: fronts[0] = Pareto front (rank 0),
    fronts[1] = next best, etc.
    Also sets chrom.rank on every individual.
    """
    n   = len(population)
    S   = [[] for _ in range(n)]   # S[i] = list of solutions dominated by i
    dom_count = [0] * n            # how many solutions dominate i
    fronts    = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(population[i], population[j]):
                S[i].append(j)
            elif dominates(population[j], population[i]):
                dom_count[i] += 1
        if dom_count[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)

    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in S[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    population[j].rank = current_front + 1
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)

    fronts = [f for f in fronts if f]   # remove empty trailing front
    return [[population[i] for i in f] for f in fronts]


def crowding_distance_assignment(front: List[Chromosome]) -> None:
    """
    Assigns crowding distances to all individuals in a single front.
    Boundary individuals receive infinite distance (always preserved).
    """
    n = len(front)
    if n == 0:
        return
    for ind in front:
        ind.crowding_distance = 0.0

    for m in range(len(front[0].obj)):
        front.sort(key=lambda c: c.obj[m])
        front[0].crowding_distance  = float('inf')
        front[-1].crowding_distance = float('inf')
        obj_range = front[-1].obj[m] - front[0].obj[m]
        if obj_range == 0:
            continue
        for k in range(1, n - 1):
            front[k].crowding_distance += (
                (front[k + 1].obj[m] - front[k - 1].obj[m]) / obj_range
            )


def nsga2_sort(population: List[Chromosome]) -> List[Chromosome]:
    """
    Sorts population by (rank ASC, crowding_distance DESC).
    Must be called AFTER fast_non_dominated_sort and crowding_distance_assignment.
    """
    return sorted(population,
                  key=lambda c: (c.rank, -c.crowding_distance))


def crowded_tournament(a: Chromosome, b: Chromosome) -> Chromosome:
    """
    NSGA-II binary tournament:
      - prefer lower rank;
      - break ties by higher crowding distance (more isolated = more diverse).
    """
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    return a if a.crowding_distance >= b.crowding_distance else b


def tournament_select(population: List[Chromosome]) -> Chromosome:
    """Randomly picks TOURNAMENT_SIZE candidates and returns the winner."""
    contestants = random.sample(population, min(TOURNAMENT_SIZE, len(population)))
    winner      = contestants[0]
    for c in contestants[1:]:
        winner = crowded_tournament(winner, c)
    return copy.deepcopy(winner)


# ─────────────────────────────────────────────────────────────
#  GENETIC OPERATORS
# ─────────────────────────────────────────────────────────────
def crossover(p1: Chromosome, p2: Chromosome, generation: int) -> Tuple[Chromosome, Chromosome]:
    """
    Single-point crossover applied with probability CROSSOVER_RATE.
    Children inherit unevaluated status so they are re-evaluated.
    """
    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
    c1.generation = c2.generation = generation
    c1.evaluated  = c2.evaluated  = False
    c1.obj        = [1.0, float('inf')]
    c2.obj        = [1.0, float('inf')]

    if random.random() < CROSSOVER_RATE:
        pt = random.randint(1, NUM_GENES - 1)
        c1.genes = p1.genes[:pt] + p2.genes[pt:]
        c2.genes = p2.genes[:pt] + p1.genes[pt:]
    return c1, c2


def mutate(chrom: Chromosome, rate: float = MUTATION_RATE) -> Chromosome:
    """
    Per-gene uniform mutation.
    The replacement value is guaranteed to differ from the current allele,
    preventing a wasted mutation that changes nothing.
    """
    for i in range(NUM_GENES):
        if random.random() < rate:
            choices = GENE_SPACE[i][1]
            if len(choices) > 1:
                options       = [k for k in range(len(choices)) if k != chrom.genes[i]]
                chrom.genes[i] = random.choice(options)
    chrom.evaluated = False
    return chrom


# ─────────────────────────────────────────────────────────────
#  HYPERVOLUME INDICATOR  (2-D, reference point based)
# ─────────────────────────────────────────────────────────────
def hypervolume_2d(front: List[Chromosome], ref: Tuple[float, float]) -> float:
    """
    Exact 2-D hypervolume dominated by `front` w.r.t. reference point `ref`.
    Both objectives are assumed minimised; `ref` should be a nadir / worst point.

    Algorithm: sort by first objective, sweep contributions along second.
    """
    pts = [(c.obj[0], c.obj[1]) for c in front
           if c.obj[0] <= ref[0] and c.obj[1] <= ref[1]]
    if not pts:
        return 0.0
    pts.sort(key=lambda p: p[0])

    hv         = 0.0
    prev_y     = ref[1]
    prev_x     = ref[0]
    for x, y in reversed(pts):           # sweep right-to-left
        hv     += (prev_x - x) * (prev_y - y)
        prev_x  = x
        prev_y  = y
    return hv


# ─────────────────────────────────────────────────────────────
#  tf.data PIPELINE BUILDER  (called once, cached globally)
# ─────────────────────────────────────────────────────────────
def build_tf_datasets(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
) -> None:
    """
    Builds and caches module-level tf.data.Dataset objects.

    Why this matters:
      Without tf.data, each model.fit() call re-converts the full numpy
      array to tensors and re-stages it to the GPU every epoch — this is
      the primary reason the CPU hits 100%.  With a cached, prefetched
      dataset, the conversion happens exactly once and subsequent epochs
      stream directly from the GPU memory buffer.

    Pipeline per dataset:
      from_tensor_slices → cache() → shuffle (train only) →
      batch(N) → prefetch(AUTOTUNE)

    cache()    : pins the dataset in RAM after first epoch — eliminates
                 all repeat numpy reads.
    prefetch() : overlaps CPU data prep with GPU compute so the GPU
                 is never idle waiting for the next batch.
    """
    global _TRAIN_DS, _VAL_DS, _FULL_TRAIN_DS, _CLASS_WEIGHTS

    print("\n  [tf.data] Building cached datasets ...")

    # Cast to float32/int32 once here — never repeated
    X_tr = X_train.astype(np.float32)
    y_tr = y_train.astype(np.int32)
    X_v  = X_val.astype(np.float32)
    y_v  = y_val.astype(np.int32)

    _TRAIN_DS = (
        tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
        .cache()
        .shuffle(SHUFFLE_BUFFER, seed=RANDOM_SEED, reshuffle_each_iteration=True)
        .batch(PROXY_BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    _VAL_DS = (
        tf.data.Dataset.from_tensor_slices((X_v, y_v))
        .cache()
        .batch(PROXY_BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    # Larger batch for full retraining (less regularisation noise needed)
    _FULL_TRAIN_DS = (
        tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
        .cache()
        .shuffle(SHUFFLE_BUFFER, seed=RANDOM_SEED, reshuffle_each_iteration=True)
        .batch(FULL_BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    # Precompute class weights once
    unique = np.unique(y_tr)
    cw = class_weight.compute_class_weight('balanced', classes=unique, y=y_tr)
    _CLASS_WEIGHTS = dict(zip(unique.tolist(), cw.tolist()))

    print(f"  [tf.data] Train batches : {len(_TRAIN_DS)}  "
          f"Val batches : {len(_VAL_DS)}  "
          f"Batch size  : {PROXY_BATCH_SIZE}")
    print(f"  [tf.data] Class weights : "
          f"{ {CLASS_NAMES[k]: round(v, 3) for k, v in _CLASS_WEIGHTS.items()} }")


# ─────────────────────────────────────────────────────────────
#  MAIN NSGA-II LOOP
# ─────────────────────────────────────────────────────────────
def run_nsga2(X_train, y_train, X_val, y_val):
    """
    Executes the full NSGA-II search.

    Returns
    -------
    pareto_front : List[Chromosome]  — final non-dominated set
    all_log      : List[dict]        — full evaluation log for CSV export
    hv_history   : List[float]       — hypervolume per generation
    """
    log_rows:   list = []
    hv_history: list = []

    # Normalised reference point for hypervolume:
    #   obj[0] = 1 - F1  → worst = 1.0 (F1 = 0)
    #   obj[1] = params  → worst = ML605_MAX_PARAMS (normalised to 1.0)
    ref_point = (1.05, 1.05)   # slightly outside worst-case normalised objectives

    def _norm_obj(c: Chromosome):
        """Returns normalised [0,1] objectives for hypervolume computation."""
        return (c.obj[0], c.obj[1] / ML605_MAX_PARAMS)

    class _NormWrapper:
        def __init__(self, c): self.obj = _norm_obj(c)

    print("\n" + "═"*62)
    print("  NSGA-II – NEURAL ARCHITECTURE SEARCH  (2 objectives)")
    print(f"  Population : {POPULATION_SIZE}   Generations : {NUM_GENERATIONS}")
    print(f"  Objectives : min(1-MacroF1)  ×  min(param_count)")
    print(f"  Search     : num_conv_blocks ∈ [1..5]")
    print(f"               filters/block   ∈ [8,16,32,64,128,256]")
    print("═"*62)

    # ── Initialise ────────────────────────────────────────────
    population: List[Chromosome] = [
        random_chromosome(generation=0) for _ in range(POPULATION_SIZE)
    ]

    for gen in range(NUM_GENERATIONS):
        t0 = time.time()
        print(f"\n┌── Generation {gen+1}/{NUM_GENERATIONS} {'─'*43}")

        # ── Evaluate unevaluated individuals ──────────────────
        unevaluated = [c for c in population if not c.evaluated]
        for idx, chrom in enumerate(unevaluated):
            arch  = chrom.decode()
            filt  = [arch[f'filters_block_{i+1}'] for i in range(arch['num_conv_blocks'])]
            print(f"│  [{idx+1:02d}/{len(unevaluated):02d}]  "
                  f"blocks={arch['num_conv_blocks']}  filters={filt} ...",
                  end='', flush=True)
            chrom = evaluate_chromosome(chrom, X_train, y_train, X_val, y_val)
            print(f"  F1={chrom.macro_f1:.4f}  params={chrom.param_count:,}")
            log_rows.append({
                'generation':  gen + 1,
                'macro_f1':    chrom.macro_f1,
                'param_count': chrom.param_count,
                'obj_f1':      chrom.obj[0],
                'obj_params':  chrom.obj[1],
                **{k: v for k, v in arch.items()},
            })

        # Clear accumulated Keras graph state once per generation,
        # NOT per individual — amortises the cost across all evals.
        tf.keras.backend.clear_session()

        # ── Non-dominated sort + crowding distance ─────────────
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            crowding_distance_assignment(front)

        sorted_pop = nsga2_sort(population)

        # ── Hypervolume of current Pareto front ────────────────
        rank0 = [c for c in population if c.rank == 0]
        norm_front = [_NormWrapper(c) for c in rank0]
        hv = hypervolume_2d(norm_front, ref_point)   # type: ignore[arg-type]
        hv_history.append(hv)

        # ── Generation summary ─────────────────────────────────
        best_f1     = max(c.macro_f1    for c in rank0)
        min_params  = min(c.param_count for c in rank0)
        print(f"│")
        print(f"│  Gen {gen+1:02d} │ Pareto front size : {len(rank0)}")
        print(f"│        │ Best F1 on front  : {best_f1:.4f}")
        print(f"│        │ Smallest on front : {min_params:,} params")
        print(f"│        │ Hypervolume       : {hv:.6f}")
        print(f"│        │ Elapsed           : {time.time()-t0:.1f}s")
        print(f"└{'─'*57}")

        if gen == NUM_GENERATIONS - 1:
            break

        # ── Breed offspring ────────────────────────────────────
        offspring: List[Chromosome] = []
        while len(offspring) < POPULATION_SIZE:
            p1 = tournament_select(sorted_pop)
            p2 = tournament_select(sorted_pop)
            c1, c2 = crossover(p1, p2, generation=gen + 1)
            offspring.extend([mutate(c1), mutate(c2)])
        offspring = offspring[:POPULATION_SIZE]

        # ── Combine parent + offspring, keep best N ────────────
        combined = population + offspring
        # evaluate unevaluated offspring
        for c in combined:
            if not c.evaluated:
                arch  = c.decode()
                filt  = [arch[f'filters_block_{i+1}'] for i in range(arch['num_conv_blocks'])]
                print(f"│  [off]  blocks={arch['num_conv_blocks']}  filters={filt} ...",
                      end='', flush=True)
                c = evaluate_chromosome(c, X_train, y_train, X_val, y_val)
                print(f"  F1={c.macro_f1:.4f}  params={c.param_count:,}")
                log_rows.append({
                    'generation':  gen + 1,
                    'macro_f1':    c.macro_f1,
                    'param_count': c.param_count,
                    'obj_f1':      c.obj[0],
                    'obj_params':  c.obj[1],
                    **{k: v for k, v in c.decode().items()},
                })

        # Clear graph state once — after all offspring, not per model
        tf.keras.backend.clear_session()

        all_fronts = fast_non_dominated_sort(combined)
        for front in all_fronts:
            crowding_distance_assignment(front)

        # Select next generation by filling front-by-front
        next_gen: List[Chromosome] = []
        for front in all_fronts:
            if len(next_gen) + len(front) <= POPULATION_SIZE:
                next_gen.extend(front)
            else:
                needed = POPULATION_SIZE - len(next_gen)
                front.sort(key=lambda c: -c.crowding_distance)
                next_gen.extend(front[:needed])
                break

        population = next_gen

    # ── Final Pareto front ─────────────────────────────────────
    final_fronts = fast_non_dominated_sort(population)
    for front in final_fronts:
        crowding_distance_assignment(front)
    pareto_front = final_fronts[0]
    pareto_front.sort(key=lambda c: c.obj[0])   # sort by F1 (ascending 1-F1)

    return pareto_front, log_rows, hv_history


# ─────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────
def plot_pareto_front(pareto_front: List[Chromosome], all_log: List[dict]):
    """
    2-D scatter of all evaluated individuals (grey) with the Pareto front
    highlighted in gradient colour (best F1 → largest model).
    ML605 BRAM budget shown as vertical dashed line.
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    # All evaluated individuals (background)
    all_params = [r['param_count'] for r in all_log]
    all_f1     = [r['macro_f1']    for r in all_log]
    ax.scatter(all_params, all_f1, c='#cccccc', s=30, alpha=0.5,
               edgecolors='none', label='All evaluated')

    # Pareto front
    pf_params = [c.param_count for c in pareto_front]
    pf_f1     = [c.macro_f1    for c in pareto_front]
    pf_colors = cm.plasma(np.linspace(0.15, 0.85, len(pareto_front)))
    ax.scatter(pf_params, pf_f1, c=pf_colors, s=120,
               edgecolors='black', linewidths=0.8, zorder=5,
               label=f'Pareto front (n={len(pareto_front)})')

    # Connect Pareto front with a step line
    pf_sorted = sorted(zip(pf_params, pf_f1), key=lambda x: x[0])
    xs, ys    = zip(*pf_sorted)
    ax.step(xs, ys, where='post', color='black', linewidth=1.2,
            alpha=0.6, linestyle='--')

    # Label each Pareto solution
    for c in pareto_front:
        arch  = c.decode()
        filt  = [arch[f'filters_block_{i+1}'] for i in range(arch['num_conv_blocks'])]
        label = f"B{arch['num_conv_blocks']}\n{filt}"
        ax.annotate(label,
                    xy=(c.param_count, c.macro_f1),
                    xytext=(8, 4), textcoords='offset points',
                    fontsize=6.5, alpha=0.85)

    ax.axvline(500_000, color='crimson', linestyle='--', linewidth=1.5,
               label='ML605 BRAM target (500K params)')

    ax.set_xlabel('Parameter Count  →  proxy for ML605 BRAM usage', fontsize=11)
    ax.set_ylabel('Macro F1 Score (proxy eval)', fontsize=11)
    ax.set_title('NSGA-II Pareto Front: F1 vs Model Size\n'
                 'Xilinx ML605 (Virtex-6 XC6VLX240T)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    path = os.path.join(OUTPUT_DIR, 'nsga2_pareto_front.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: nsga2_pareto_front.png")


def plot_hypervolume(hv_history: List[float]):
    """Plots hypervolume indicator progression across generations."""
    fig, ax = plt.subplots(figsize=(8, 4))
    gens = list(range(1, len(hv_history) + 1))
    ax.plot(gens, hv_history, 'o-', color='steelblue', linewidth=2, markersize=7)
    ax.fill_between(gens, hv_history, alpha=0.15, color='steelblue')
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Hypervolume Indicator (↑ better)', fontsize=11)
    ax.set_title('NSGA-II Convergence: Hypervolume per Generation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    path = os.path.join(OUTPUT_DIR, 'nsga2_convergence.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: nsga2_convergence.png")


# ─────────────────────────────────────────────────────────────
#  FULL RETRAINING + EXPORT  (per Pareto solution)
# ─────────────────────────────────────────────────────────────
def retrain_and_export_solution(
    chrom:   Chromosome,
    sol_idx: int,
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
):
    """
    Fully retrains one Pareto solution, evaluates on test set,
    quantises to INT8, and writes all artefacts to a sub-folder.
    """
    sol_dir = os.path.join(OUTPUT_DIR, f'pareto_{sol_idx:02d}')
    os.makedirs(sol_dir, exist_ok=True)

    arch  = chrom.decode()
    filt  = [arch[f'filters_block_{i+1}'] for i in range(arch['num_conv_blocks'])]
    print(f"\n  ── Solution {sol_idx:02d}: blocks={arch['num_conv_blocks']}  "
          f"filters={filt}  proxy_F1={chrom.macro_f1:.4f} ──")

    # ── Build & train ─────────────────────────────────────────
    model = build_model(chrom)
    cbs = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0),
    ]
    # Use cached tf.data pipeline if available (much faster than raw numpy)
    train_input = _FULL_TRAIN_DS if _FULL_TRAIN_DS is not None else (X_train, y_train)
    val_input   = _VAL_DS        if _VAL_DS        is not None else (X_val,   y_val)
    cw_dict     = _CLASS_WEIGHTS if _CLASS_WEIGHTS  is not None else {}
    history = model.fit(
        train_input,
        epochs=FULL_EPOCHS,
        validation_data=val_input,
        class_weight=cw_dict if cw_dict else None,
        callbacks=cbs,
        verbose=0,
    )

    # ── Test evaluation ───────────────────────────────────────
    y_prob  = model.predict(X_test, verbose=0)
    y_pred  = np.argmax(y_prob, axis=1)
    f1_test = f1_score(y_test, y_pred, average='macro')
    acc     = accuracy_score(y_test, y_pred)
    print(f"    Test Macro F1 : {f1_test:.4f}   Accuracy : {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

    # ── Confusion matrix ──────────────────────────────────────
    cm_mat = confusion_matrix(y_test, y_pred)
    disp   = ConfusionMatrixDisplay(cm_mat, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Pareto {sol_idx:02d} – Test Confusion Matrix\n'
                 f'blocks={arch["num_conv_blocks"]}  filters={filt}')
    plt.tight_layout()
    plt.savefig(os.path.join(sol_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # ── Training history ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'],     label='Train')
    axes[0].plot(history.history['val_loss'], label='Val')
    axes[0].set_title('Loss'); axes[0].legend()
    axes[1].plot(history.history['accuracy'],     label='Train')
    axes[1].plot(history.history['val_accuracy'], label='Val')
    axes[1].set_title('Accuracy'); axes[1].legend()
    plt.suptitle(f'Pareto {sol_idx:02d}: blocks={arch["num_conv_blocks"]}  filters={filt}')
    plt.tight_layout()
    plt.savefig(os.path.join(sol_dir, 'training_history.png'), dpi=150)
    plt.close()

    # ── Save Keras model ──────────────────────────────────────
    model.save(os.path.join(sol_dir, 'model_full.keras'))

    # ── INT8 quantisation ─────────────────────────────────────
    idx_cal  = np.random.permutation(len(X_train))[:200]
    cal_data = X_train[idx_cal].astype(np.float32)
    def rep_data():
        for i in range(len(cal_data)):
            yield [cal_data[i: i + 1]]
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations              = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset    = rep_data
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type      = tf.int8
    conv.inference_output_type     = tf.int8
    tflite_bytes = conv.convert()
    with open(os.path.join(sol_dir, 'model_int8.tflite'), 'wb') as f:
        f.write(tflite_bytes)

    # ── ML605 resource report ─────────────────────────────────
    params  = int(model.count_params())
    dsp_est = max(1, params // 1500)
    bram_est = max(1, params // (36 * 1024))
    report_lines = [
        f"Pareto Solution {sol_idx:02d}  –  ML605 Resource Estimate",
        "=" * 55,
        f"  num_conv_blocks : {arch['num_conv_blocks']}",
        f"  filters         : {filt}",
        f"  kernel_size     : {FIXED_KERNEL_SIZE}",
        f"  dense_units     : {FIXED_DENSE_UNITS}",
        f"  dropout_rate    : {FIXED_DROPOUT}",
        f"  batch_norm      : {FIXED_BATCH_NORM}",
        "",
        f"  Total parameters: {params:,}",
        f"  Test Macro F1   : {f1_test:.4f}",
        f"  Accuracy        : {acc:.4f}",
        "",
        "  Estimated ML605 Resources (INT8 inference)",
        "  " + "-"*40,
        f"  DSP48E1 (~300 budget) : {dsp_est:>5d}  "
            f"{'✓' if dsp_est <= 300 else '✗ EXCEEDS'}",
        f"  BRAM36  (~150 budget) : {bram_est:>5d}  "
            f"{'✓' if bram_est <= 150 else '✗ EXCEEDS'}",
        f"  Param budget (500K)   : "
            f"{'✓' if params <= 500_000 else '✗ EXCEEDS'}",
        "",
        "  Next steps:",
        "    1. Load model_int8.tflite via hls4ml.convert()",
        "    2. Set backend='VivadoAccelerator', board='ml605'",
        "    3. hls4ml.build() → check Vivado utilisation report",
    ]
    with open(os.path.join(sol_dir, 'ml605_resource_report.txt'), 'w') as f:
        f.write('\n'.join(report_lines) + '\n')

    print(f"    Artefacts → {sol_dir}/")

    tf.keras.backend.clear_session()
    del model

    return f1_test, params


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':

    RECORDS = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234',
    ]

    # ── 1. Load data ──────────────────────────────────────────
    X, y = load_dataset(RECORDS)
    if len(X) == 0:
        print("\nFATAL: No data loaded. Check DATA_DIR.")
        sys.exit(1)

    # 70 / 15 / 15 split
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.176, random_state=RANDOM_SEED, stratify=y_tv)
    print(f"\n  Train:{X_train.shape}  Val:{X_val.shape}  Test:{X_test.shape}")

    # ── 2. Build tf.data pipelines (once — reused by every eval) ─
    # This is the primary fix for CPU saturation: avoids rebuilding
    # the numpy→tensor pipeline 20+ times per generation.
    build_tf_datasets(X_train, y_train, X_val, y_val)

    # ── 3. Run NSGA-II ────────────────────────────────────────
    pareto_front, all_log, hv_history = run_nsga2(X_train, y_train, X_val, y_val)

    # ── 4. Save search log ────────────────────────────────────
    pd.DataFrame(all_log).to_csv(
        os.path.join(OUTPUT_DIR, 'nsga2_search_log.csv'), index=False)
    print(f"\n  Search log saved → nsga2_search_log.csv")

    # ── 5. Save Pareto solutions JSON ─────────────────────────
    pareto_data = []
    for i, c in enumerate(pareto_front):
        pareto_data.append({
            'solution_id':  i,
            'macro_f1_proxy': c.macro_f1,
            'param_count':  c.param_count,
            'rank':         c.rank,
            'genes':        c.genes,
            'architecture': c.decode(),
        })
    with open(os.path.join(OUTPUT_DIR, 'pareto_solutions.json'), 'w') as f:
        json.dump(pareto_data, f, indent=2)
    print(f"  Pareto solutions → pareto_solutions.json")

    # ── 6. Plot ───────────────────────────────────────────────
    plot_pareto_front(pareto_front, all_log)
    plot_hypervolume(hv_history)

    # ── 7. Print Pareto summary ───────────────────────────────
    print("\n" + "═"*62)
    print("  FINAL PARETO FRONT  (sorted by F1, proxy evaluation)")
    print("═"*62)
    print(f"  {'Sol':>3}  {'F1 (proxy)':>10}  {'Params':>10}  "
          f"{'Blocks':>6}  Filters")
    print(f"  {'───':>3}  {'──────────':>10}  {'──────':>10}  "
          f"{'──────':>6}  ───────")
    for i, c in enumerate(pareto_front):
        arch  = c.decode()
        filt  = [arch[f'filters_block_{b+1}'] for b in range(arch['num_conv_blocks'])]
        print(f"  {i:>3d}  {c.macro_f1:>10.4f}  {c.param_count:>10,}  "
              f"{arch['num_conv_blocks']:>6d}  {filt}")

    # ── 8. Full retraining of all Pareto solutions ────────────
    print("\n" + "═"*62)
    print("  FULL RETRAINING OF ALL PARETO SOLUTIONS")
    print("═"*62)
    final_results = []
    for i, chrom in enumerate(pareto_front):
        f1_test, params = retrain_and_export_solution(
            chrom, i, X_train, y_train, X_val, y_val, X_test, y_test)
        final_results.append({'solution': i, 'test_f1': f1_test, 'params': params})

    # ── 9. Final summary ──────────────────────────────────────
    print("\n" + "═"*62)
    print("  FINAL TEST RESULTS  (after full retraining)")
    print("═"*62)
    print(f"  {'Sol':>3}  {'Test F1':>8}  {'Params':>10}  ML605 Budget")
    print(f"  {'───':>3}  {'───────':>8}  {'──────':>10}  ────────────")
    for r in final_results:
        budget_ok = '✓ OK' if r['params'] <= 500_000 else '✗ Large'
        print(f"  {r['solution']:>3d}  {r['test_f1']:>8.4f}  {r['params']:>10,}  {budget_ok}")

    best_sol = max(final_results, key=lambda r: r['test_f1'])
    print(f"\n  Recommended solution : pareto_{best_sol['solution']:02d}/")
    print(f"  Best test Macro F1  : {best_sol['test_f1']:.4f}")
    print(f"  Parameters          : {best_sol['params']:,}")

    print("\n" + "═"*62)
    print("  NSGA-II NAS PIPELINE COMPLETE")
    print(f"  All outputs → '{OUTPUT_DIR}'")
    print("═"*62)

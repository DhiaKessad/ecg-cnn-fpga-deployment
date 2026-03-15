# ECG Arrhythmia Detection & FPGA Deployment Pipeline

A full-stack hardware-software co-design pipeline for biomedical AI at the edge. The project takes raw ECG recordings from the MIT-BIH Arrhythmia Database, processes the signals, searches for an FPGA-optimal neural architecture using a multi-objective genetic algorithm, trains the best model, quantises it to INT8, and exports hardware-ready initialisation files for Xilinx FPGA Block RAM all in pure Python.

---

## Pipeline Overview

```
MIT-BIH Database (.csv + annotations)
        │
        ▼
┌─────────────────────────┐
│  Signal Processing      │  Zero-phase FIR bandpass (0.5–45 Hz)
│                         │  R-peak detection · Z-score normalisation
│                         │  Gaussian noise augmentation (class balance)
└───────────┬─────────────┘
            │  127,000+ labelled beats (4 classes)
            ▼
┌─────────────────────────┐
│  NSGA-II Architecture   │  Multi-objective search over 1D-CNN space
│  Search (NAS)           │  Objectives: 1 − Macro F1  ×  param count
│                         │  10 generations · pop 20 · proxy 3 epochs
└───────────┬─────────────┘
            │  Pareto front of architectures
            ▼
┌─────────────────────────┐
│  Full Training          │  GA-optimal model retrained (30 epochs)
│                         │  tf.data pipeline · class weights · BatchNorm
└───────────┬─────────────┘
            │  Trained Keras model (F1 = 0.9895)
            ▼
┌─────────────────────────┐
│  INT8 Quantisation      │  TFLite post-training full-integer quant
│                         │  200-sample calibration · stratified verify
└───────────┬─────────────┘
            │  ecg_model_optimal_int8.tflite
            ▼
┌─────────────────────────┐
│  FPGA Export            │  Weight/bias → Xilinx .coe (BRAM init)
│                         │  30 stratified INT8 test inputs + manifest
│                         │  Golden model verification (Python ↔ RTL)
└─────────────────────────┘
```

---

## Repository Structure

```
ecg-cnn-fpga-deployment/
│
├── ECG.py                   # Binary classifier (Normal vs Abnormal) — baseline
├── Multiclass_ECG.py        # 4-class pipeline with hand-designed CNN
├── Multiclass_ECG_GA.py     # 4-class pipeline retrained with GA-optimal architecture
├── ecg_nsga2_nas.py         # NSGA-II Neural Architecture Search (main NAS script)
│
│
├── Sim/                     # RTL simulation files (Verilog testbench & waveforms)
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Scripts

### `ECG.py` — Binary Baseline
The starting point. Classifies ECG beats as **Normal vs. Abnormal** (2 classes). Establishes the full pipeline: FIR filtering → peak detection → segmentation → CNN training → INT8 quantisation → COE export. Outputs go to `fpga_outputs/`.

### `Multiclass_ECG.py` — 4-Class Hand-Designed Model
Extends the binary pipeline to four arrhythmia classes:

| Class | Label | ECG Symbols |
|-------|-------|-------------|
| 0 | Normal | N |
| 1 | Aberrant / Atrial Premature | A |
| 2 | Bundle Branch Block | L, R, B |
| 3 | Premature Ventricular Contraction | V |

Uses a hand-designed `[Conv32 → Conv64 → Conv128]` 1D-CNN. Outputs go to `multiclass_fpga_outputs/`.

### `ecg_nsga2_nas.py` — Multi-Objective Neural Architecture Search
The core research contribution. Runs **NSGA-II** to automatically discover CNN architectures that balance classification accuracy against model size (FPGA resource proxy).

**Search space:**
- Number of conv blocks: 1–5
- Filters per block: {8, 16, 32, 64, 128, 256}
- Fixed: kernel size = 5, BatchNorm = True, dense = 64, dropout = 0.3

**Objectives (both minimised):**
- `f1 = 1 − Macro F1` (classification performance)
- `f2 = parameter count` (BRAM proxy for Xilinx ML605)

**GA configuration:**

| Parameter | Value |
|-----------|-------|
| Population size | 20 |
| Generations | 10 |
| Proxy epochs | 3 |
| Full retrain epochs | 30 |
| Crossover probability | 0.9 |
| Mutation probability | 0.2 per gene |
| Selection | Crowded binary tournament |

**Final Pareto Front (Kaggle H100, 10 generations):**

| Solution | Architecture | Params | Test Macro F1 | Accuracy |
|----------|-------------|--------|---------------|----------|
| pareto_00 | B5 [8,32,128,256,8] | 198,876 | 0.9959 | 0.9964 |
| pareto_01 | B5 [32,128,16,256,8] | 64,636 | 0.9954 | 0.9960 |
| pareto_02 | B5 [8,32,16,16,32] | 10,612 | 0.9901 | 0.9920 |
| **pareto_03** | **B4 [8,32,16,16]** | **6,868** | **0.9895** | **0.9913** |
| pareto_04 | B4 [8,32,8,16] | 4,908 | 0.9828 | 0.9865 |
| pareto_05 | B1 [8] | 916 | 0.7816 | 0.8251 |

Hypervolume improved from **0.731 → 0.825** (+12.8%) over 10 generations.

Each Pareto solution is saved with its own `model_full.keras`, `model_int8.tflite`, `confusion_matrix.png`, `training_history.png`, and `ml605_resource_report.txt`.

### `Multiclass_ECG_GA.py` — Full Pipeline with GA-Optimal Architecture
Retrains **pareto_03** (the recommended deployment candidate) using the full 30-epoch training loop on the complete dataset, then runs the entire quantisation and COE export pipeline. This is the file to use for generating production FPGA artefacts.

**Why pareto_03?**

| Metric | pareto_00 | pareto_03 |
|--------|-----------|-----------|
| Parameters | 198,876 | **6,868** |
| Test Macro F1 | 0.9959 | **0.9895** |
| Est. DSP48E1 | ~131 | **~5** |
| Est. BRAM36 | ~3 | **< 1** |
| F1 gap vs best | — | −0.64 pp |

A 29× reduction in parameters and ~26× fewer DSP slices for less than 1 percentage point of F1 — leaving the vast majority of the ML605's resources free for AXI streaming, control logic, and parallel inference channels.

---

## Target Hardware

**Xilinx Virtex-6 ML605 (XC6VLX240T)**

| Resource | Available | pareto_03 (estimated) |
|----------|-----------|-----------------------|
| DSP48E1 | 768 | ~5 |
| BRAM36 | 416 | < 1 |
| Parameters | — | 6,868 |

---

## FPGA Export Outputs

Each pipeline run produces the following in its output directory:

| File | Description |
|------|-------------|
| `best_model.keras` | Best float32 Keras model (checkpoint) |
| `ecg_model_*_int8.tflite` | Full-integer INT8 quantised model |
| `w_*.coe` | Weight tensors → Xilinx BRAM `.coe` init files |
| `b_*.coe` | Bias tensors → Xilinx BRAM `.coe` init files |
| `test_input_NNN.coe` | 30 stratified INT8 test inputs for RTL testbench |
| `test_manifest.csv` | Maps each test COE to ground-truth and predicted label |
| `confusion_matrix_float.png` | Float32 confusion matrix |
| `confusion_matrix_int8.png` | INT8 simulation confusion matrix |
| `training_history.png` | Loss and accuracy curves |

The `test_manifest.csv` serves as a **golden reference** — the RTL testbench can compare its output classifications against the Python-predicted labels to verify bit-exact agreement before synthesis.

---

## Installation

```bash
git clone https://github.com/Ishaq-ML/ecg-cnn-fpga-deployment.git
cd ecg-cnn-fpga-deployment
pip install -r requirements.txt
```

## Dataset Setup

The MIT-BIH Arrhythmia Database is not included due to size. Download it first:

- **Kaggle:** [MIT-BIH Database (CSV)](https://www.kaggle.com/datasets/ishakkolla/mitbih-database)
- **PhysioNet:** [physionet.org/content/mitdb](https://physionet.org/content/mitdb/)

Then place the `.csv` signal files and `*annotations.txt` files into a folder named `mitbih_database/` in the repository root. The scripts auto-detect the database path on Kaggle (`/kaggle/input/`) and fall back to the local directory.

## Usage

```bash
# 1. Binary baseline
python ECG.py

# 2. 4-class hand-designed model
python Multiclass_ECG.py

# 3. Run NSGA-II architecture search (GPU recommended)
python ecg_nsga2_nas.py

# 4. Full training + export with GA-optimal architecture
python Multiclass_ECG_GA.py
```

---

## Results Summary

| Pipeline | Classes | Model | Params | Macro F1 | INT8 F1 |
|----------|---------|-------|--------|----------|---------|
| `ECG.py` | 2 (Binary) | Hand-designed | ~47K | ~0.97 | — |
| `Multiclass_ECG.py` | 4 | Hand-designed [32,64,128] | ~47K | ~0.97 | ~0.97 |
| `Multiclass_ECG_GA.py` | 4 | GA-optimal [8,32,16,16] | **6,868** | **0.9895** | **~0.989** |

---

## License

MIT — see [LICENSE](LICENSE).

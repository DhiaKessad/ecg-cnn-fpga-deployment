#!/usr/bin/env python3
"""
extract_weights.py — Pareto 03 FPGA Weight Extractor
=====================================================
Architecture : 4x Conv1D blocks, filters=[8,32,16,16], kernel=5
               BatchNorm after each conv, GlobalAvgPool, Dense(64), Dense(4)
Quantization : INT8 (weights) / INT32 (biases) from model_int8.tflite
Output format: Q8.8 fixed-point → 4-digit hex per value, one per line

Usage
-----
    python3 scripts/extract_weights.py

Outputs written to:
    RTL/hex/conv<n>_weights.hex
    RTL/hex/conv<n>_bias.hex
    RTL/hex/dense0_weights.hex
    RTL/hex/dense0_bias.hex
    RTL/hex/dense1_weights.hex   (output layer, 4 classes)
    RTL/hex/dense1_bias.hex
    RTL/hex/bn<n>_gamma.hex
    RTL/hex/bn<n>_beta.hex
    RTL/hex/bn<n>_mean.hex
    RTL/hex/bn<n>_var.hex
    RTL/hex/extraction_report.txt
"""

import os
import sys
import struct
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SCRIPT_DIR)
TFLITE_PATH = os.path.join(ROOT_DIR, "model_int8.tflite")
HEX_DIR     = os.path.join(ROOT_DIR, "RTL", "hex")

# ── Q8.8 helper ──────────────────────────────────────────────────────────────
Q88_SCALE = 256.0   # 2^8

def to_q88_hex(arr):
    """Convert float array → Q8.8 signed int → 4-digit hex string list."""
    flat    = arr.flatten().astype(np.float64)
    q88     = np.round(flat * Q88_SCALE).astype(np.int32)
    lines   = []
    for v in q88:
        # 16-bit two's complement
        u16 = int(v) & 0xFFFF
        lines.append(f"{u16:04x}")
    return lines

def write_hex(lines, path):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

# ── TFLite loader (pure-Python, no flatbuffers package required) ──────────────
def load_tflite_tensors(tflite_path: str):
    """
    Use the 'tflite' pip package to walk all tensors in the model.
    Returns a list of dicts with keys:
        name, shape, dtype, float_data (de-quantised as float32)
    for every tensor that has constant buffer data.
    """
    try:
        import tflite
        import tflite.TensorType as TT
    except ImportError:
        sys.exit("[ERROR] Install the 'tflite' package:  pip install tflite")

    with open(tflite_path, "rb") as f:
        buf = f.read()

    model    = tflite.Model.GetRootAsModel(buf, 0)
    subgraph = model.Subgraphs(0)
    n_tensors = subgraph.TensorsLength()
    print(f"  TFLite model loaded — {n_tensors} tensors found")

    tensors = []
    for i in range(n_tensors):
        t      = subgraph.Tensors(i)
        name   = t.Name().decode("utf-8") if t.Name() else f"tensor_{i}"
        shape  = t.ShapeAsNumpy() if t.ShapeLength() else np.array([])
        dtype  = t.Type()
        buf_i  = t.Buffer()
        buf_obj= model.Buffers(buf_i)
        data   = buf_obj.DataAsNumpy()

        if data is None or not isinstance(data, np.ndarray) or data.size == 0:
            continue  # activation tensor, not a constant

        # Decode raw bytes based on dtype
        if dtype == TT.INT8:
            raw = np.frombuffer(data, dtype=np.int8)
        elif dtype == TT.INT32:
            raw = np.frombuffer(data, dtype=np.int32)
        elif dtype == TT.FLOAT32:
            raw = np.frombuffer(data, dtype=np.float32)
        else:
            continue  # skip UINT8, bool, etc.

        # De-quantise integer tensors
        q = t.Quantization()
        if q is not None and q.ScaleLength() > 0 and dtype != TT.FLOAT32:
            scales = q.ScaleAsNumpy()
            zps    = q.ZeroPointAsNumpy()
            raw_f  = raw.astype(np.float64)
            shape_r = shape if len(shape) > 0 else raw.shape

            if len(scales) == 1:
                float_data = (raw_f - float(zps[0])) * float(scales[0])
            else:
                # Per-channel quantisation (first axis = output channels)
                raw_r    = raw_f.reshape(shape_r)
                float_r  = np.zeros_like(raw_r, dtype=np.float64)
                for c in range(shape_r[0]):
                    zp = float(zps[c]) if c < len(zps) else 0.0
                    sc = float(scales[c]) if c < len(scales) else 1.0
                    float_r[c] = (raw_r[c] - zp) * sc
                float_data = float_r.flatten()
        else:
            float_data = raw.astype(np.float64)

        tensors.append({
            "idx"       : i,
            "name"      : name,
            "shape"     : shape.tolist() if len(shape) else [],
            "dtype"     : dtype,
            "n_elements": int(np.prod(shape)) if len(shape) else len(raw),
            "float_data": float_data.reshape(shape) if len(shape) else float_data,
        })

    return tensors


# ── Classification heuristics ─────────────────────────────────────────────────
def classify_tensor(t: dict) -> tuple[str | None, str | None]:
    """
    Returns (layer_label, param_type) for recognised weight/bias/BN tensors,
    or (None, None) if the tensor should be ignored.
    
    Pareto-03 unique tensor-size fingerprints (all layers use kernel=5,
    filters=[8,32,16,16], dense=[64,4]):

    Conv weights shape: [out_ch, in_ch, 1, kernel]  (TFLite stores as 4-D)
       conv1: [8,   1, 1, 5]     → 40  elements
       conv2: [32,  8, 1, 5]     → 1280 elements
       conv3: [16, 32, 1, 5]     → 2560 elements
       conv4: [16, 16, 1, 5]     → 1280 elements  (same count as conv2!)
    Conv bias: [out_ch] INT32
    BN params: gamma/beta as FLOAT32, mean/var as INT32 or FLOAT32
    Dense1 weights: [64, 16]  → 1024 elements
    Dense1 bias   : [64]
    Dense2 weights: [4, 64]   → 256 elements
    Dense2 bias   : [4]
    """
    name  = t["name"].lower()
    shape = t["shape"]
    n     = t["n_elements"]
    ndim  = len(shape)

    # ── Conv weights ── 4-D tensors [out_ch, in_ch, 1, ksize]
    CONV_WEIGHT_SIZES = {40: "conv1", 1280: "convX", 2560: "conv3", }
    #    conv2 and conv4 both have 1280 elements — disambiguate by name/order
    if ndim == 4 and shape[2] == 1 and shape[3] == 5:
        out_ch, in_ch = shape[0], shape[1]
        if   (out_ch, in_ch) == (8,  1):  return "conv1", "weights"
        elif (out_ch, in_ch) == (32, 8):  return "conv2", "weights"
        elif (out_ch, in_ch) == (16, 32): return "conv3", "weights"
        elif (out_ch, in_ch) == (16, 16): return "conv4", "weights"

    # ── Dense weights ── 2-D tensors
    if ndim == 2:
        if (shape[0], shape[1]) == (64, 16): return "dense0", "weights"
        if (shape[0], shape[1]) == (4,  64): return "dense1", "weights"

    # Use name-based heuristics for BN and biases
    for tag in ["conv1", "conv2", "conv3", "conv4", "dense0", "dense1"]:
        if tag in name:
            if "bias" in name or "biasadd" in name:
                return tag, "bias"
            if "beta"  in name: return tag, "bn_beta"
            if "gamma" in name: return tag, "bn_gamma"
            if "mean"  in name: return tag, "bn_mean"
            if "var"   in name or "variance" in name: return tag, "bn_var"

    return None, None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Pareto-03 Weight Extractor — Q8.8 Hex Output")
    print("=" * 60)

    os.makedirs(HEX_DIR, exist_ok=True)

    print(f"\n[1] Loading: {TFLITE_PATH}")
    if not os.path.isfile(TFLITE_PATH):
        sys.exit(f"[ERROR] TFLite model not found at {TFLITE_PATH}")

    tensors = load_tflite_tensors(TFLITE_PATH)

    # ── Classification ────────────────────────────────────────────────────────
    print("\n[2] Classifying tensors …")
    extracted = {}

    # TFLite actual Conv1D weight shapes verified by introspection:
    # [out_ch, 1, kernel, in_ch]
    CONV_WEIGHT_SHAPES = {
        (8,  1, 5, 1):  "conv1",
        (32, 1, 5, 8):  "conv2",
        (16, 1, 5, 32): "conv3",
        (16, 1, 5, 16): "conv4",
    }
    DENSE_WEIGHT_SHAPES = {
        (64, 16): "dense0",
        (4,  64): "dense1",
    }

    # Biases: 1-D tensors matched sequentially by size as they appear in model
    bias_queue = [
        ("dense1", 4),
        ("dense0", 64),
        ("conv4",  16),
        ("conv3",  16),
        ("conv2",  32),
        ("conv1",   8),
    ]
    bias_idx = 0

    for t in tensors:
        shape = tuple(int(x) for x in t["shape"]) if t["shape"] else ()
        data  = t["float_data"]
        ndim  = len(shape)

        label = None
        ptype = None

        if ndim == 4 and shape in CONV_WEIGHT_SHAPES:
            label = CONV_WEIGHT_SHAPES[shape]
            ptype = "weights"
        elif ndim == 2 and shape in DENSE_WEIGHT_SHAPES:
            label = DENSE_WEIGHT_SHAPES[shape]
            ptype = "weights"
        elif ndim == 1 and bias_idx < len(bias_queue):
            expected_label, expected_ch = bias_queue[bias_idx]
            if shape[0] == expected_ch:
                label = expected_label
                ptype = "bias"
                bias_idx += 1

        if label is None:
            continue

        key = f"{label}/{ptype}"
        if key in extracted:
            continue

        hex_lines = to_q88_hex(data)
        extracted[key] = {
            "label"    : label,
            "ptype"    : ptype,
            "shape"    : list(shape),
            "n_hex"    : len(hex_lines),
            "hex_lines": hex_lines,
            "tfl_name" : t["name"],
        }
        print(f"  ✓  {key:30s}  shape={list(shape)}  → {len(hex_lines)} hex values")

    # ── Write hex files ───────────────────────────────────────────────────────
    print(f"\n[3] Writing hex files to:  {HEX_DIR}/")
    report_lines = [
        "Pareto-03 Weight Extraction Report",
        "=" * 60,
        f"Source : {TFLITE_PATH}",
        f"Output : {HEX_DIR}",
        f"Format : Q8.8 fixed-point (16-bit two's complement hex)",
        "",
        f"{'File':<35} {'TFLite tensor':<45} {'Shape':<25} {'N values'}",
        "-" * 120,
    ]

    ORDER = ["conv1", "conv2", "conv3", "conv4", "dense0", "dense1"]
    PTYPES = ["weights", "bias"]
    for lname in ORDER:
        for ptype in PTYPES:
            key = f"{lname}/{ptype}"
            if key not in extracted:
                continue
            info  = extracted[key]
            fname = f"{lname}_{ptype}.hex"
            fpath = os.path.join(HEX_DIR, fname)
            write_hex(info["hex_lines"], fpath)
            report_lines.append(
                f"{fname:<35} {info['tfl_name']:<45} {str(info['shape']):<25} {info['n_hex']}"
            )
            print(f"  → {fname}  ({info['n_hex']} values)")

    report_lines += ["", f"Total tensors extracted : {len(extracted)}"]
    report_path = os.path.join(HEX_DIR, "extraction_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\n  Report saved → {report_path}")
    print(f"\n{'=' * 60}")
    print(f"  Done. {len(extracted)} parameter tensors extracted.")
    if len(extracted) < 8:
        print(f"  WARNING: Expected 8 tensors (4w+4b for conv, 2w+2b for dense).")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

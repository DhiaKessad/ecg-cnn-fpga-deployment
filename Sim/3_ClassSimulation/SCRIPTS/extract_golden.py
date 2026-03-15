"""
extract_golden.py
-----------------
Parse all .coe files in multiclass_fpga_outputs and consolidate them into
a single structured golden reference file (golden.txt).

Output sections (in order):
  [INPUT]           - input_sample_0.coe
  [LAYER 1 – Conv1D (sequential_conv1d)]
      BIAS          - b_sequential_conv1d_BiasAdd_ReadVariableOp_resource.coe
      POST-ReLU     - b_sequential_conv1d_Relu;...BiasAdd_ReadVariableOp_resource.coe
      POST-POOL     - b_sequential_conv1d_Relu;...BiasAdd_ReadVariableOp_resource;...conv1d.coe
  ... (repeated for every conv/dense layer)
  [LAYER N – Dense (output)]
      BIAS
      OUTPUT LOGITS
"""

import os
import re

SRC_DIR = (
    "/Users/sid/Library/CloudStorage/GoogleDrive-m.kessad@nsnn.edu.dz/"
    "My Drive/NHSNN Personal Drive/FPGA/projects/ECG/Multi/RTL/fpga_outputs"
)
OUT_FILE = (
    "/Users/sid/Library/CloudStorage/GoogleDrive-m.kessad@nsnn.edu.dz/"
    "My Drive/NHSNN Personal Drive/FPGA/projects/ECG/Multi/RTL/fpga_outputs/golden.txt"
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def parse_coe(path):
    """Return the integer values from a .coe file as a Python list of strings."""
    values = []
    in_vector = False
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.strip().replace('\r', '')
            if line.startswith('memory_initialization_vector'):
                in_vector = True
                continue
            if in_vector:
                # Strip trailing comma or semicolon, skip empty lines
                v = line.rstrip(',;').strip()
                if v:
                    values.append(v)
    return values


def fmt_values(values, label, indent=4):
    """Format a list of values with a label and aligned columns (10 per row)."""
    pad = ' ' * indent
    lines = [f"{pad}{label} ({len(values)} values):"]
    ROW = 10
    for i in range(0, len(values), ROW):
        chunk = values[i:i+ROW]
        lines.append(pad + '  ' + ', '.join(f"{v:>8}" for v in chunk))
    return '\n'.join(lines)


def find_file(name_fragment):
    """Find a file whose name contains the given fragment (exact basename match)."""
    for fn in os.listdir(SRC_DIR):
        if fn == name_fragment:
            return os.path.join(SRC_DIR, fn)
    return None


def find_files_containing(substr):
    """Return sorted list of filenames that contain substr."""
    return sorted(
        fn for fn in os.listdir(SRC_DIR)
        if substr in fn and fn.endswith('.coe')
    )

# --------------------------------------------------------------------------
# Layer definitions
# Each entry: (human_label, layer_key)
# layer_key is the unique part of the filename that identifies the layer.
# --------------------------------------------------------------------------

LAYERS = [
    # (display name,                   bias key,                          relu/act key,                       pool key)
    # Conv layers – model A (sequential_conv1d*)
    ("Layer 1 – Conv1D  [sequential_conv1d]",
     "b_sequential_conv1d_BiasAdd_ReadVariableOp_resource.coe",
     "b_sequential_conv1d_Relu;sequential_conv1d_BiasAdd;sequential_conv1d_conv1d_Squeeze;sequential_conv1d_BiasAdd_ReadVariableOp_resource.coe",
     "b_sequential_conv1d_Relu;sequential_conv1d_BiasAdd;sequential_conv1d_conv1d_Squeeze;sequential_conv1d_BiasAdd_ReadVariableOp_resource;sequential_conv1d_conv1d.coe"),

    ("Layer 2 – Conv1D  [sequential_conv1d_1]",
     "b_sequential_conv1d_1_BiasAdd_ReadVariableOp_resource.coe",
     "b_sequential_conv1d_1_Relu;sequential_conv1d_1_BiasAdd;sequential_conv1d_1_conv1d_Squeeze;sequential_conv1d_1_BiasAdd_ReadVariableOp_resource.coe",
     "b_sequential_conv1d_1_Relu;sequential_conv1d_1_BiasAdd;sequential_conv1d_1_conv1d_Squeeze;sequential_conv1d_1_BiasAdd_ReadVariableOp_resource;sequential_conv1d_1_conv1d.coe"),

    ("Layer 3 – Conv1D  [sequential_conv1d_2]",
     "b_sequential_conv1d_2_BiasAdd_ReadVariableOp_resource.coe",
     "b_sequential_conv1d_2_Relu;sequential_conv1d_2_BiasAdd;sequential_conv1d_2_conv1d_Squeeze;sequential_conv1d_2_BiasAdd_ReadVariableOp_resource.coe",
     "b_sequential_conv1d_2_Relu;sequential_conv1d_2_BiasAdd;sequential_conv1d_2_conv1d_Squeeze;sequential_conv1d_2_BiasAdd_ReadVariableOp_resource;sequential_conv1d_2_conv1d.coe"),

    # Conv layers – model B (sequential_1_conv1d*)
    ("Layer 4 – Conv1D  [sequential_1_conv1d_3]",
     "b_sequential_1_conv1d_3_BiasAdd_ReadVariableOp_resource.coe",
     "b_sequential_1_conv1d_3_Relu;sequential_1_conv1d_3_BiasAdd;sequential_1_conv1d_3_conv1d_Squeeze;sequential_1_conv1d_3_BiasAdd_ReadVariableOp_resource.coe",
     "b_sequential_1_conv1d_3_Relu;sequential_1_conv1d_3_BiasAdd;sequential_1_conv1d_3_conv1d_Squeeze;sequential_1_conv1d_3_BiasAdd_ReadVariableOp_resource;sequential_1_conv1d_3_conv1d.coe"),

    ("Layer 5 – Conv1D  [sequential_1_conv1d_4]",
     "b_sequential_1_conv1d_4_BiasAdd_ReadVariableOp_resource.coe",
     "b_sequential_1_conv1d_4_Relu;sequential_1_conv1d_4_BiasAdd;sequential_1_conv1d_4_conv1d_Squeeze;sequential_1_conv1d_4_BiasAdd_ReadVariableOp_resource.coe",
     "b_sequential_1_conv1d_4_Relu;sequential_1_conv1d_4_BiasAdd;sequential_1_conv1d_4_conv1d_Squeeze;sequential_1_conv1d_4_BiasAdd_ReadVariableOp_resource;sequential_1_conv1d_4_conv1d.coe"),

    ("Layer 6 – Conv1D  [sequential_1_conv1d_5]",
     "b_sequential_1_conv1d_5_BiasAdd_ReadVariableOp_resource.coe",
     "b_sequential_1_conv1d_5_Relu;sequential_1_conv1d_5_BiasAdd;sequential_1_conv1d_5_conv1d_Squeeze;sequential_1_conv1d_5_BiasAdd_ReadVariableOp_resource.coe",
     "b_sequential_1_conv1d_5_Relu;sequential_1_conv1d_5_BiasAdd;sequential_1_conv1d_5_conv1d_Squeeze;sequential_1_conv1d_5_BiasAdd_ReadVariableOp_resource;sequential_1_conv1d_5_conv1d.coe"),
]

DENSE_LAYERS = [
    ("Layer 7 – Dense   [sequential_dense]",
     "b_sequential_dense_BiasAdd_ReadVariableOp_resource.coe",
     "b_sequential_dense_MatMul;sequential_dense_BiasAdd.coe"),

    ("Layer 8 – Dense   [sequential_1_dense_1]  ← OUTPUT",
     "b_sequential_1_dense_1_BiasAdd_ReadVariableOp_resource.coe",
     "b_sequential_1_dense_1_MatMul;sequential_1_dense_1_BiasAdd.coe"),
]

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    lines = []

    lines.append("=" * 80)
    lines.append("  GOLDEN REFERENCE – ECG Multiclass FPGA Layer-by-Layer Outputs")
    lines.append("  Source: multiclass_fpga_outputs/")
    lines.append("  Values are post-quantisation integers (int8 / int32 accumulator)")
    lines.append("=" * 80)
    lines.append("")

    # ── INPUT ────────────────────────────────────────────────────────────────
    # We will get input_sample_0 from src_folder directly since it's not in SRC_DIR
    import sys
    base_dir = os.path.dirname(os.path.dirname(SRC_DIR))
    inp_path = os.path.join(base_dir, "SCRIPTS", "src_folder", "input_sample_0.coe")
    lines.append("─" * 80)
    lines.append("[INPUT]  input_sample_0  (180 time-steps, quantised int8)")
    lines.append("─" * 80)
    vals = parse_coe(inp_path)
    lines.append(fmt_values(vals, "Data"))
    lines.append("")

    # ── CONV LAYERS ──────────────────────────────────────────────────────────
    for label, bias_fn, relu_fn, pool_fn in LAYERS:
        lines.append("─" * 80)
        lines.append(f"[{label}]")
        lines.append("─" * 80)

        bias_path = os.path.join(SRC_DIR, bias_fn)
        relu_path = os.path.join(SRC_DIR, relu_fn)
        pool_path = os.path.join(SRC_DIR, pool_fn)

        for section, path in [
            ("BIAS  (pre-activation)",         bias_path),
            ("OUTPUT after ReLU  (post-conv)",  relu_path),
            ("OUTPUT after MaxPool / Global",   pool_path),
        ]:
            if os.path.exists(path):
                vals = parse_coe(path)
                lines.append(fmt_values(vals, section))
            else:
                lines.append(f"    {section}: *** FILE NOT FOUND: {os.path.basename(path)} ***")
        lines.append("")

    # ── DENSE LAYERS ─────────────────────────────────────────────────────────
    for label, bias_fn, logit_fn in DENSE_LAYERS:
        lines.append("─" * 80)
        lines.append(f"[{label}]")
        lines.append("─" * 80)

        bias_path  = os.path.join(SRC_DIR, bias_fn)
        logit_path = os.path.join(SRC_DIR, logit_fn)

        for section, path in [
            ("BIAS",             bias_path),
            ("OUTPUT LOGITS",    logit_path),
        ]:
            if os.path.exists(path):
                vals = parse_coe(path)
                lines.append(fmt_values(vals, section))
            else:
                lines.append(f"    {section}: *** FILE NOT FOUND: {os.path.basename(path)} ***")
        lines.append("")

    lines.append("=" * 80)
    lines.append("  END OF GOLDEN REFERENCE")
    lines.append("=" * 80)

    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Golden file written to:\n  {OUT_FILE}")
    print(f"  Total lines: {len(lines)}")

if __name__ == "__main__":
    main()

"""
build_golden_folders.py
-----------------------
Creates two folders from the contents of multiclass_fpga_outputs/:

  golden/
  ├── weights_biases/       ← one .txt per layer with bias + (future: kernel)
  └── layer_outputs/        ← one .txt per layer with golden activation outputs
                               for input_sample_0

Layer mapping (in pipeline order):
  Layer 1  :  sequential_conv1d          (16 filters)
  Layer 2  :  sequential_conv1d_1        (32 filters)
  Layer 3  :  sequential_conv1d_2        (64 filters)
  Layer 4  :  sequential_1_conv1d_3      (16 filters)
  Layer 5  :  sequential_1_conv1d_4      (32 filters)
  Layer 6  :  sequential_1_conv1d_5      (64 filters)
  Layer 7  :  sequential_dense           (3 neurons, binary model output)
  Layer 8  :  sequential_1_dense_1       (3 neurons, multiclass output ← final)
"""

import os
import shutil

SRC = (
    "/Users/sid/Library/CloudStorage/GoogleDrive-m.kessad@nsnn.edu.dz/"
    "My Drive/NHSNN Personal Drive/FPGA/projects/ECG/Multi/RTL/multiclass_fpga_outputs"
)
GOLDEN_ROOT = (
    "/Users/sid/Library/CloudStorage/GoogleDrive-m.kessad@nsnn.edu.dz/"
    "My Drive/NHSNN Personal Drive/FPGA/projects/ECG/Multi/RTL/golden"
)
WEIGHTS_DIR = os.path.join(GOLDEN_ROOT, "weights_biases")
OUTPUTS_DIR = os.path.join(GOLDEN_ROOT, "layer_outputs")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def parse_coe(path):
    """Parse a .coe file and return list of value strings."""
    values = []
    if not os.path.exists(path):
        return values
    in_vec = False
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.strip().replace('\r', '')
            if line.startswith('memory_initialization_vector'):
                in_vec = True
                continue
            if in_vec:
                v = line.rstrip(',;').strip()
                if v:
                    values.append(v)
    return values


def write_txt(path, header, sections):
    """
    Write a formatted .txt file.
    sections = list of (label, values_list)
    """
    with open(path, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        f.write('=' * 70 + '\n\n')
        for label, values in sections:
            f.write(f'[{label}]  ({len(values)} values)\n')
            f.write('-' * 70 + '\n')
            ROW = 10
            for i in range(0, len(values), ROW):
                chunk = values[i:i+ROW]
                f.write('  ' + ',  '.join(f'{v:>8}' for v in chunk) + '\n')
            f.write('\n')


# ---------------------------------------------------------------------------
# Layer definitions
# Each conv entry:
#   (short_name, bias_coe, relu_coe, pool_coe)
# Each dense entry:
#   (short_name, bias_coe, logit_coe)
# ---------------------------------------------------------------------------

CONV_LAYERS = [
    (
        "layer1_conv1d",
        "b_sequential_conv1d_BiasAdd_ReadVariableOp_resource.coe",
        "b_sequential_conv1d_Relu;sequential_conv1d_BiasAdd;sequential_conv1d_conv1d_Squeeze;sequential_conv1d_BiasAdd_ReadVariableOp_resource.coe",
        "b_sequential_conv1d_Relu;sequential_conv1d_BiasAdd;sequential_conv1d_conv1d_Squeeze;sequential_conv1d_BiasAdd_ReadVariableOp_resource;sequential_conv1d_conv1d.coe",
    ),
    (
        "layer2_conv1d_1",
        "b_sequential_conv1d_1_BiasAdd_ReadVariableOp_resource.coe",
        "b_sequential_conv1d_1_Relu;sequential_conv1d_1_BiasAdd;sequential_conv1d_1_conv1d_Squeeze;sequential_conv1d_1_BiasAdd_ReadVariableOp_resource.coe",
        "b_sequential_conv1d_1_Relu;sequential_conv1d_1_BiasAdd;sequential_conv1d_1_conv1d_Squeeze;sequential_conv1d_1_BiasAdd_ReadVariableOp_resource;sequential_conv1d_1_conv1d.coe",
    ),
    (
        "layer3_conv1d_2",
        "b_sequential_conv1d_2_BiasAdd_ReadVariableOp_resource.coe",
        "b_sequential_conv1d_2_Relu;sequential_conv1d_2_BiasAdd;sequential_conv1d_2_conv1d_Squeeze;sequential_conv1d_2_BiasAdd_ReadVariableOp_resource.coe",
        "b_sequential_conv1d_2_Relu;sequential_conv1d_2_BiasAdd;sequential_conv1d_2_conv1d_Squeeze;sequential_conv1d_2_BiasAdd_ReadVariableOp_resource;sequential_conv1d_2_conv1d.coe",
    ),
    (
        "layer4_conv1d_3",
        "b_sequential_1_conv1d_3_BiasAdd_ReadVariableOp_resource.coe",
        "b_sequential_1_conv1d_3_Relu;sequential_1_conv1d_3_BiasAdd;sequential_1_conv1d_3_conv1d_Squeeze;sequential_1_conv1d_3_BiasAdd_ReadVariableOp_resource.coe",
        "b_sequential_1_conv1d_3_Relu;sequential_1_conv1d_3_BiasAdd;sequential_1_conv1d_3_conv1d_Squeeze;sequential_1_conv1d_3_BiasAdd_ReadVariableOp_resource;sequential_1_conv1d_3_conv1d.coe",
    ),
    (
        "layer5_conv1d_4",
        "b_sequential_1_conv1d_4_BiasAdd_ReadVariableOp_resource.coe",
        "b_sequential_1_conv1d_4_Relu;sequential_1_conv1d_4_BiasAdd;sequential_1_conv1d_4_conv1d_Squeeze;sequential_1_conv1d_4_BiasAdd_ReadVariableOp_resource.coe",
        "b_sequential_1_conv1d_4_Relu;sequential_1_conv1d_4_BiasAdd;sequential_1_conv1d_4_conv1d_Squeeze;sequential_1_conv1d_4_BiasAdd_ReadVariableOp_resource;sequential_1_conv1d_4_conv1d.coe",
    ),
    (
        "layer6_conv1d_5",
        "b_sequential_1_conv1d_5_BiasAdd_ReadVariableOp_resource.coe",
        "b_sequential_1_conv1d_5_Relu;sequential_1_conv1d_5_BiasAdd;sequential_1_conv1d_5_conv1d_Squeeze;sequential_1_conv1d_5_BiasAdd_ReadVariableOp_resource.coe",
        "b_sequential_1_conv1d_5_Relu;sequential_1_conv1d_5_BiasAdd;sequential_1_conv1d_5_conv1d_Squeeze;sequential_1_conv1d_5_BiasAdd_ReadVariableOp_resource;sequential_1_conv1d_5_conv1d.coe",
    ),
]

DENSE_LAYERS = [
    (
        "layer7_dense",
        "b_sequential_dense_BiasAdd_ReadVariableOp_resource.coe",
        "b_sequential_dense_MatMul;sequential_dense_BiasAdd.coe",
    ),
    (
        "layer8_dense_1_OUTPUT",
        "b_sequential_1_dense_1_BiasAdd_ReadVariableOp_resource.coe",
        "b_sequential_1_dense_1_MatMul;sequential_1_dense_1_BiasAdd.coe",
    ),
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Clean and recreate output dirs
    if os.path.exists(GOLDEN_ROOT):
        shutil.rmtree(GOLDEN_ROOT)
    os.makedirs(WEIGHTS_DIR)
    os.makedirs(OUTPUTS_DIR)

    # ── INPUT sample (goes into layer_outputs only) ─────────────────────────
    inp = parse_coe(os.path.join(SRC, "input_sample_0.coe"))
    write_txt(
        os.path.join(OUTPUTS_DIR, "input_sample_0.txt"),
        "INPUT SAMPLE 0  –  180 time-steps (quantised int8)",
        [("Input data", inp)],
    )
    print(f"  [OUTPUT] input_sample_0.txt  ({len(inp)} values)")

    # ── Conv layers ──────────────────────────────────────────────────────────
    for name, bias_fn, relu_fn, pool_fn in CONV_LAYERS:
        bias   = parse_coe(os.path.join(SRC, bias_fn))
        relu   = parse_coe(os.path.join(SRC, relu_fn))
        pool   = parse_coe(os.path.join(SRC, pool_fn))

        # weights_biases/  ←  bias only (kernels are in the .tflite, not exported here)
        write_txt(
            os.path.join(WEIGHTS_DIR, f"{name}_bias.txt"),
            f"{name.upper()}  –  Bias vector",
            [("Bias (int32)", bias)],
        )
        print(f"  [WEIGHT] {name}_bias.txt  ({len(bias)} values)")

        # layer_outputs/  ←  post-ReLU and post-pool
        write_txt(
            os.path.join(OUTPUTS_DIR, f"{name}_output.txt"),
            f"{name.upper()}  –  Golden outputs for input_sample_0",
            [
                ("After ReLU  (post-conv)", relu),
                ("After MaxPool / GlobalPool", pool),
            ],
        )
        print(f"  [OUTPUT] {name}_output.txt  (relu:{len(relu)}, pool:{len(pool)})")

    # ── Dense layers ─────────────────────────────────────────────────────────
    for name, bias_fn, logit_fn in DENSE_LAYERS:
        bias   = parse_coe(os.path.join(SRC, bias_fn))
        logits = parse_coe(os.path.join(SRC, logit_fn))

        write_txt(
            os.path.join(WEIGHTS_DIR, f"{name}_bias.txt"),
            f"{name.upper()}  –  Bias vector",
            [("Bias (int32)", bias)],
        )
        print(f"  [WEIGHT] {name}_bias.txt  ({len(bias)} values)")

        write_txt(
            os.path.join(OUTPUTS_DIR, f"{name}_output.txt"),
            f"{name.upper()}  –  Golden outputs for input_sample_0",
            [("Output logits", logits)],
        )
        print(f"  [OUTPUT] {name}_output.txt  ({len(logits)} values)")

    print(f"\nDone.")
    print(f"  weights_biases/ → {WEIGHTS_DIR}")
    print(f"  layer_outputs/  → {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()

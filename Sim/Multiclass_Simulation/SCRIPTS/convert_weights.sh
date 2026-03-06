#!/bin/bash
# Script to convert .coe files to .hex format using coetohex.py

SRC_DIR="../RTL/multiclass_fpga_outputs"
DEST_DIR="../RTL/hex"

mkdir -p "$DEST_DIR"

echo "Running Python conversion script..."
# Using the local python installation to run the script
python3 ./coetohex.py "$SRC_DIR" "$DEST_DIR"

echo "Conversion complete."

# Re-name specifically known weights to match the RTL format
cd "$DEST_DIR" || exit 1

echo "Renaming weight files to match RTL module expectations..."

# layer 1 conv (16x1x5x1)
# b_sequential_1_conv1d_3_BiasAdd_ReadVariableOp_resource.hex -> b_conv1.hex
# b_sequential_1_conv1d_3_Relu;sequential_1_conv1d_3_BiasAdd;sequential_1_conv1d_3_conv1d_Squeeze;sequential_1_conv1d_3_BiasAdd_ReadVariableOp_resource;sequential_1_conv1d_3_conv1d.hex -> w_conv1.hex
cp b_sequential_1_conv1d_3_BiasAdd_ReadVariableOp_resource.hex b_conv1.hex
cp b_sequential_1_conv1d_3_Relu\;sequential_1_conv1d_3_BiasAdd\;sequential_1_conv1d_3_conv1d_Squeeze\;sequential_1_conv1d_3_BiasAdd_ReadVariableOp_resource\;sequential_1_conv1d_3_conv1d.hex w_conv1.hex

# layer 3 conv (32x1x3x16)
# b_sequential_1_conv1d_4_BiasAdd_ReadVariableOp_resource.hex -> b_conv2.hex
# b_sequential_1_conv1d_4_Relu...;sequential_1_conv1d_4_conv1d.hex -> w_conv2.hex
cp b_sequential_1_conv1d_4_BiasAdd_ReadVariableOp_resource.hex b_conv2.hex
cp b_sequential_1_conv1d_4_Relu\;sequential_1_conv1d_4_BiasAdd\;sequential_1_conv1d_4_conv1d_Squeeze\;sequential_1_conv1d_4_BiasAdd_ReadVariableOp_resource\;sequential_1_conv1d_4_conv1d.hex w_conv2.hex

# layer 5 conv (64x1x3x32)
# b_sequential_1_conv1d_5_BiasAdd_ReadVariableOp_resource.hex -> b_conv3.hex
# b_sequential_1_conv1d_5_Relu...;sequential_1_conv1d_5_conv1d.hex -> w_conv3.hex
cp b_sequential_1_conv1d_5_BiasAdd_ReadVariableOp_resource.hex b_conv3.hex
cp b_sequential_1_conv1d_5_Relu\;sequential_1_conv1d_5_BiasAdd\;sequential_1_conv1d_5_conv1d_Squeeze\;sequential_1_conv1d_5_BiasAdd_ReadVariableOp_resource\;sequential_1_conv1d_5_conv1d.hex w_conv3.hex

# layer 7 dense (3x64)
# b_sequential_1_dense_1_BiasAdd_ReadVariableOp_resource.hex -> b_dense.hex
# b_sequential_1_dense_1_MatMul;sequential_1_dense_1_BiasAdd.hex -> w_dense.hex
cp b_sequential_1_dense_1_BiasAdd_ReadVariableOp_resource.hex b_dense.hex
cp b_sequential_1_dense_1_MatMul\;sequential_1_dense_1_BiasAdd.hex w_dense.hex

# input sample
cp input_sample_0.hex input_sample.hex 

echo "All weight files deployed successfully to $DEST_DIR."

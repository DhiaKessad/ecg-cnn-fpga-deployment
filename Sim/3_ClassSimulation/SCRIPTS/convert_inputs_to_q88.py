import os
import glob
import numpy as np

# From TFLite inspection:
# [0] conv1d_input | Type: 9 | Scale: [0.03751991] | ZP: [-30]
INPUT_SCALE = 0.03751991
INPUT_ZP = -30

def to_q88(val_int8):
    # Dequantize to float
    val_float = (val_int8 - INPUT_ZP) * INPUT_SCALE
    # Convert to Q8.8
    return int(np.round(val_float * 256.0))

def process_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    converted = []
    for line in lines:
        line = line.strip()
        if not line: continue
        val_hex = int(line, 16)
        if val_hex >= 0x8000:
            val_int8 = val_hex - 0x10000
        else:
            val_int8 = val_hex
            
        q88 = to_q88(val_int8)
        
        # Format as 16-bit hex, 2's complement
        if q88 < 0:
            q88 = (1 << 16) + q88
        converted.append(f"{q88:04x}")
        
    with open(file_path, 'w') as f:
        f.write('\n'.join(converted))

test_dirs = ['RTL/hex/test_inputs_1']
count = 0
for d in test_dirs:
    for f in glob.glob(os.path.join(d, 'test_input_*.hex')):
        process_file(f)
        count += 1

print(f"Successfully converted {count} test input files to Q8.8 format.")

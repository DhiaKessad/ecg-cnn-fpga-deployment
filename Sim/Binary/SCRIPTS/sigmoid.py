#Values of the sigmoid function need to be pre-calculated, and stored in a memo to avoid heavy math

import numpy as np

def generate_sigmoid_lut(filename="sigmoid_lut.hex", size=256):
    # Map address 0-255 to x-range -8.0 to +8.0
    x = np.linspace(-8, 8, size)
    y = 1 / (1 + np.exp(-x))
    
    # Scale to 16-bit fixed point (Q8.8 format: 256 = 1.0)
    # Output is 0 to 1, so max value is 256
    y_fixed = (y * 256).astype(int)
    
    with open(filename, "w") as f:
        for val in y_fixed:
            # writing it in 4 hex (16 bit)
            f.write(f"{val:04x}\n")

generate_sigmoid_lut()
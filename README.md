# ECG Arrhythmia Detection & FPGA Deployment Pipeline

This repository contains a full-stack workflow for developing a hardware-accelerated biomedical AI system. It takes raw ECG data, processes it, trains a neural network, and prepares the weights for implementation on an FPGA.

The project addresses the challenge of deploying deep learning models onto resource-constrained edge devices by utilizing **INT8 Quantization** and generating hardware-ready initialization files.

## Key Features

- **Signal Processing:** Implementation of FIR Bandpass filters (0.5–45Hz) and R-Peak detection algorithms using `SciPy`.
- **Deep Learning:** A custom 1D-CNN architecture trained on the MIT-BIH Arrhythmia Database using `TensorFlow`/`Keras`.
- **Quantization:** Conversion of the model from Float32 to INT8 using `TFLite` to reduce memory footprint for hardware.
- **Hardware Export:** Custom Python scripts that extract weights and biases from the quantized model and format them into Xilinx `.coe` files for Block RAM initialization.
- **Verification:** Bit-exact verification scripts to ensure the Python simulation matches the expected integer hardware output.

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

# --- 1. IMPORTING LIBRARIES ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import firwin, lfilter, find_peaks
import os
import sys

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, 'mitbih_database')
OUTPUT_DIR = os.path.join(BASE_DIR, 'fpga_outputs')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Signal Processing Constants
FS = 360
NUMTAPS = 51
LOWCUT = 0.5
HIGHCUT = 45.0
DELAY = (NUMTAPS - 1) // 2 

# --- 2. SIGNAL PROCESSING FUNCTIONS ---

def get_fir_coeffs():
    """Generates Bandpass FIR filter coefficients."""
    return firwin(NUMTAPS, [LOWCUT, HIGHCUT], fs=FS, pass_zero=False)

def process_record(record_id, coeffs):
    """
    Loads a single patient record, filters it, extracts heartbeats,
    and matches them with annotations.
    """
    csv_path = os.path.join(DATA_DIR, f"{record_id}.csv")
    ann_path = os.path.join(DATA_DIR, f"{record_id}annotations.txt")
    
    if not os.path.exists(csv_path) or not os.path.exists(ann_path):
        return [], []

    try:
        # Load Signal
        df = pd.read_csv(csv_path)
        signal = df.iloc[:, 1].values
        
        # Apply Filter
        filtered_signal = lfilter(coeffs, 1.0, signal)
        
        # Load Annotations
        ann_df = pd.read_table(
            ann_path, 
            sep=r'\s+', 
            skiprows=1, 
            names=['time', 'sample', 'type', 'sub', 'chan', 'num', 'aux'],
            engine='python',
            on_bad_lines='skip',
            quoting=3
        )
        
        ann_samples = ann_df['sample'].values
        ann_symbols = ann_df['type'].values
        
        # Peak Detection
        peaks, _ = find_peaks(filtered_signal, height=np.max(filtered_signal)*0.4, distance=200)
        
        X_local = []
        y_local = []
        
        for p in peaks:
            adjusted_p = p - DELAY # Correcting group delay
            
            # Boundary checks
            if 90 < p < (len(filtered_signal) - 90):
                # Find closest annotation
                idx = np.argmin(np.abs(ann_samples - adjusted_p))
                
                # Check synchronization (within ~55ms)
                if np.abs(ann_samples[idx] - adjusted_p) < 20:
                    symbol = ann_symbols[idx]
                    
                    # Segment Window
                    segment = filtered_signal[p-90 : p+90]
                    
                    # Z-score Normalization
                    if np.std(segment) == 0: continue # Avoid division by zero
                    norm_hb = (segment - np.mean(segment)) / np.std(segment)
                    
                    # Labeling: 'N' is Normal (0), everything else is Arrhythmia (1)
                    label = 0 if symbol == 'N' else 1
                    
                    X_local.append(norm_hb)
                    y_local.append(label)
                    
        return X_local, y_local

    except Exception as e:
        print(f"Error processing {record_id}: {e}")
        return [], []

def load_dataset(records):
    """Iterates through all records and compiles the dataset."""
    coeffs = get_fir_coeffs()
    X_all = []
    y_all = []
    
    print(f"--- Starting Data Compilation from {DATA_DIR} ---")
    
    for rid in records:
        X_rec, y_rec = process_record(rid, coeffs)
        if len(X_rec) > 0:
            X_all.extend(X_rec)
            y_all.extend(y_rec)
            print(f"Processed Patient {rid}: Found {len(X_rec)} beats")
            
    X_final = np.array(X_all).reshape(-1, 180, 1)
    y_final = np.array(y_all)
    
    return X_final, y_final

# --- 3. MODEL TRAINING & QUANTIZATION ---

def train_model(X_train, y_train, X_test, y_test):
    """Builds and trains the 1D-CNN."""
    
    # Calculate class weights for imbalance
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = {0: weights[0], 1: weights[1]}
    
    model = models.Sequential([
        layers.Conv1D(16, kernel_size=5, activation='relu', input_shape=(180, 1)),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("\n--- Training Model ---")
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weights_dict,
        verbose=1
    )
    return model

def convert_to_tflite_int8(model, X_train):
    """Quantizes the model to INT8."""
    print("\n--- Quantizing Model to INT8 ---")
    
    def representative_data_gen():
        for i in range(100):
            yield [X_train[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    save_path = os.path.join(OUTPUT_DIR, 'ecg_model_quant.tflite')
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"Quantized model saved to: {save_path}")
    return tflite_model

# --- 4. HARDWARE EXPORT FUNCTIONS ---

def save_coe(filename, data):
    """Writes data to a Xilinx COE file."""
    flat_data = data.flatten()
    full_path = os.path.join(OUTPUT_DIR, filename)
    
    with open(full_path, 'w') as f:
        f.write("memory_initialization_radix=10;\n")
        f.write("memory_initialization_vector=\n")
        
        for i, val in enumerate(flat_data):
            scalar_val = int(val.item()) if hasattr(val, 'item') else int(val)
            suffix = ";" if i == len(flat_data) - 1 else ","
            f.write(f"{scalar_val}{suffix}\n")
            
    print(f"Generated: {filename}")

def export_weights_and_verify(tflite_model_content, X_test, y_test):
    """
    Extracts weights for FPGA and runs bit-exact verification.
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_model_content)
    interpreter.allocate_tensors()
    
    # 1. Export Weights & Biases
    print("\n--- Exporting COE Files ---")
    details = interpreter.get_tensor_details()
    for detail in details:
        name = detail['name'].replace('/', '_').replace(':', '_')
        
        if 'weight' in detail['name'].lower():
            data = interpreter.get_tensor(detail['index'])
            save_coe(f"w_{name}.coe", data)
        elif 'bias' in detail['name'].lower():
            data = interpreter.get_tensor(detail['index'])
            save_coe(f"b_{name}.coe", data)

    # 2. Hardware Simulation Verification
    print("\n--- Verifying Hardware Accuracy ---")
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    input_scale, input_zp = input_details['quantization']
    output_scale, output_zp = output_details['quantization']
    
    # Calculate integer threshold for 0.5 probability
    # Formula: (0.5 / S) + ZP
    threshold_int8 = int(0.5 / output_scale) + output_zp
    
    correct = 0
    num_samples = 200  # Test on 200 samples
    
    for i in range(num_samples):
        # Simulate ADC/FPGA Input: Float -> Int8
        val_float = (X_test[i:i+1] / input_scale) + input_zp
        test_sample = np.clip(np.round(val_float), -128, 127).astype(np.int8)
        
        interpreter.set_tensor(input_details['index'], test_sample)
        interpreter.invoke()
        
        pred_int8 = interpreter.get_tensor(output_details['index'])[0][0]
        
        # Check against integer threshold
        is_arrhythmia = pred_int8 > threshold_int8
        
        if is_arrhythmia == y_test[i]:
            correct += 1
            
    acc = (correct / num_samples) * 100
    print(f"Hardware Simulation Accuracy: {acc:.2f}% (on {num_samples} samples)")

# --- 5. MAIN EXECUTION ---

if __name__ == "__main__":
    # List of MIT-BIH records to process
    records = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234'
    ]

    # 1. Load Data
    X, y = load_dataset(records)
    
    if len(X) == 0:
        print("Error: No data loaded. Check 'mitbih_database' folder path.")
        sys.exit(1)

    print(f"\nFinal Dataset Shape: {X.shape}")
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Train
    model = train_model(X_train, y_train, X_test, y_test)
    
    # 4. Quantize
    tflite_model = convert_to_tflite_int8(model, X_train)
    
    # 5. Export & Verify
    export_weights_and_verify(tflite_model, X_test, y_test)
    
    print("\n--- Pipeline Complete ---")
    print(f"Check '{OUTPUT_DIR}' for generated .coe files.")

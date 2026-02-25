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

# --- Multiclass Classification Constants ---
# Example: N (Normal), A (Aberrant), L/R/B (Bundle Branch Block)
SYMBOL_TO_CLASS = {
    'N': 0,  # Normal
    'A': 1,  # Aberrant
    'L': 2,  # Left Bundle Branch Block
    'R': 2,  # Right Bundle Branch Block
    'B': 2,  # Bundle Branch Block (general)
}
NUM_CLASSES = len(set(SYMBOL_TO_CLASS.values()))
CLASS_NAMES = ['Normal', 'Aberrant', 'Bundle Branch Block'] 
# --- 2. SIGNAL PROCESSING FUNCTIONS ---

def get_fir_coeffs():
    """Generates Bandpass FIR filter coefficients."""
    return firwin(NUMTAPS, [LOWCUT, HIGHCUT], fs=FS, pass_zero=False)

def process_record(record_id, coeffs):
    """
    Loads a single patient record, filters it, extracts heartbeats,
    and matches them with annotations for multiclass classification.
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
        peaks, _ = find_peaks(filtered_signal, height=np.max(filtered_signal)*0.3, distance=150)
        
        X_local = []
        y_local = []
        
        for p in peaks:
            # We need exactly 90 samples before and 90 samples after.
            if (p - 90 < 0) or (p + 90 > len(filtered_signal)):
                continue

            # 2. Extract Segment
            segment = filtered_signal[p-90 : p+90]
            
            # 3. Double Check Length
            if len(segment) != 180:
                continue-

            adjusted_p = p - DELAY # Correcting group delay for annotation matching
            
            # Find closest annotation
            idx = np.argmin(np.abs(ann_samples - adjusted_p))
            
            # Check synchronization (within ~20 samples)
            if np.abs(ann_samples[idx] - adjusted_p) < 20:
                symbol = ann_symbols[idx]
                
                if symbol in SYMBOL_TO_CLASS:
                    label = SYMBOL_TO_CLASS[symbol]
                    
                    # Z-score Normalization
                    if np.std(segment) == 0: continue
                    norm_hb = (segment - np.mean(segment)) / np.std(segment)
                    
                    X_local.append(norm_hb)
                    y_local.append(label)
                    
        return X_local, y_local

    except Exception as e:
        print(f"Error processing {record_id}: {e}")
        return [], []
def load_dataset(records):
    """Iterates through all records and compiles the dataset for multiclass classification."""
    coeffs = get_fir_coeffs()
    X_all = []
    y_all = []

    print(f"--- Starting Data Compilation from {DATA_DIR} ---")
    processed_beats_count = 0

    for rid in records:
        X_rec, y_rec = process_record(rid, coeffs)
        if len(X_rec) > 0:
            X_all.extend(X_rec)
            y_all.extend(y_rec)
            print(f"Processed Patient {rid}: Found {len(X_rec)} beats classified into {NUM_CLASSES} classes.")
            processed_beats_count += len(X_rec)

    if not X_all:
        print("Error: No beats were extracted. Please check your data and parameters.")
        return np.array([]), np.array([])

    X_final = np.array(X_all).reshape(-1, 180, 1)
    y_final = np.array(y_all)

    print(f"\n--- Data Compilation Complete ---")
    print(f"Total beats extracted: {processed_beats_count}")
    print(f"Dataset Shape: {X_final.shape}")
    print(f"Labels Shape: {y_final.shape}")

    # Display class distribution
    unique_classes, counts = np.unique(y_final, return_counts=True)
    print("Class Distribution:")
    for cls, count in zip(unique_classes, counts):
        print(f"  Class {cls} ({CLASS_NAMES[cls]}): {count}")

    return X_final, y_final

# --- 3. MODEL TRAINING & QUANTIZATION ---

def train_model(X_train, y_train, X_test, y_test):
    """Builds and trains the 1D-CNN for multiclass classification."""

    # Calculate class weights for imbalance
    # Ensure 'classes' are the unique integer labels, and 'y' is the array of labels.
    unique_labels = np.unique(y_train)
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=y_train
    )
    # Create a dictionary mapping each class label to its weight
    class_weights_dict = dict(zip(unique_labels, weights))

    print("\n--- Model Architecture (Multiclass) ---")
    print(f"Number of Classes: {NUM_CLASSES}")

    model = models.Sequential([
        layers.Conv1D(16, kernel_size=5, activation='relu', input_shape=(180, 1)),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        # --- Multiclass Output Layer ---
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # --- Multiclass Loss Function ---
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', # For integer labels
                  metrics=['accuracy'])

    model.summary()

    print("\n--- Training Model ---")
    # Pass the class_weights_dict to the fit method
    history = model.fit(
        X_train, y_train,
        epochs=10, #
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weights_dict, # Use class weights
        verbose=1
    )
    return model

def convert_to_tflite_int8(model, X_train_for_calibration):
    """Quantizes the model to INT8 for hardware deployment."""
    print("\n--- Quantizing Model to INT8 ---")

    # Ensure calibration data is representative and sufficient
    # Using the first 200 samples for calibration.
    calibration_samples = X_train_for_calibration[:200]
    if len(calibration_samples) == 0:
        print("Warning: No calibration samples available. Quantization might be suboptimal.")
        calibration_samples = np.random.rand(1, 180, 1).astype(np.float32)


    def representative_data_gen():
        # Yield batches of float32 data
        for i in range(len(calibration_samples)):
            # Yield a single sample at a time, shaped for the model input
            yield [calibration_samples[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    # --- INT8 Quantization Target Specification ---
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Specify the input and output types to be INT8 for inference
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    save_path = os.path.join(OUTPUT_DIR, 'ecg_model_multiclass_quant.tflite')
    with open(save_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Quantized INT8 model saved to: {save_path}")
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
            # Ensure conversion to standard Python int for COE format
            if hasattr(val, 'item'):
                scalar_val = int(val.item())
            else:
                scalar_val = int(val)

            suffix = ";" if i == len(flat_data) - 1 else ","
            f.write(f"{scalar_val}{suffix}\n")

    print(f"Generated: {filename}")

def export_weights_and_verify(tflite_model_content, X_test, y_test):
    """
    Extracts weights for FPGA and runs bit-exact verification with INT8 quantized model.
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_model_content)
    interpreter.allocate_tensors()
    
    # 1. Export Weights & Biases
    print("\n--- Exporting COE Files (Weights and Biases) ---")
    details = interpreter.get_tensor_details()
    for detail in details:
        name = detail['name'].replace('/', '_').replace(':', '_')
        
        if 'weight' in detail['name'].lower():
            data = interpreter.get_tensor(detail['index'])
            save_coe(f"w_{name}.coe", data)
        elif 'bias' in detail['name'].lower():
            data = interpreter.get_tensor(detail['index'])
            save_coe(f"b_{name}.coe", data)

    # 2. Hardware Simulation Verification (Bit-exact INT8 inference)
    print("\n--- Verifying Hardware Accuracy (INT8) ---")
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    input_scale, input_zp = input_details['quantization']
    output_scale, output_zp = output_details['quantization']
    
    correct_predictions = 0
    num_verification_samples = min(200, len(X_test))
    predicted_labels = []  # List to store predictions
    
    if num_verification_samples == 0:
        print("Warning: No test samples available for verification.")
        return

    for i in range(num_verification_samples):
        # Simulate FPGA Input: Float -> INT8 quantization
        original_sample_float = X_test[i:i+1].astype(np.float32)
        val_quantized_float = (original_sample_float / input_scale) + input_zp
        test_sample_int8 = np.clip(np.round(val_quantized_float), -128, 127).astype(np.int8)
        
        # Run inference
        interpreter.set_tensor(input_details['index'], test_sample_int8)
        interpreter.invoke()
        
        # Get INT8 output
        pred_int8 = interpreter.get_tensor(output_details['index'])[0]
        
        # Dequantize to find winner
        pred_float_probs = (pred_int8.astype(np.float32) - output_zp) * output_scale
        predicted_class_index = np.argmax(pred_float_probs)
        
        # Store prediction and check accuracy
        predicted_labels.append(predicted_class_index)
        if predicted_class_index == y_test[i]:
            correct_predictions += 1
            
    acc = (correct_predictions / num_verification_samples) * 100
    print(f"Hardware Simulation Accuracy (INT8): {acc:.2f}% (on {num_verification_samples} samples)")
    print(f"Ground Truth Labels: {list(y_test[:num_verification_samples])}")
    print(f"Predicted Labels:    {predicted_labels}")

    # --- Export Sample Input for FPGA Testing ---
    if num_verification_samples > 0:
        print("\n--- Exporting Sample Input COE File ---")
        sample_input_index = 0
        original_sample_float = X_test[sample_input_index:sample_input_index+1].astype(np.float32)
        val_quantized_float = (original_sample_float / input_scale) + input_zp
        test_sample_int8 = np.clip(np.round(val_quantized_float), -128, 127).astype(np.int8)
        save_coe("input_sample_0.coe", test_sample_int8)
        print(f"Exported INT8 quantized input for sample 0.")

# --- 5. MAIN EXECUTION ---

if __name__ == "__main__":
    # List of MIT-BIH records to process
    # Make sure these records exist in your 'mitbih_database' folder
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
        print("\nFATAL ERROR: No data loaded or processed. Please ensure:")
        print("1. The 'mitbih_database' folder exists in the same directory as the script.")
        print("2. The CSV and annotation files for the specified records are present in 'mitbih_database'.")
        print("3. The file paths and record names are correct.")
        sys.exit(1)

    # 2. Split Data
    # Use stratify to maintain class distribution in train/test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining Data Shape: {X_train.shape}, Labels Shape: {y_train.shape}")
    print(f"Testing Data Shape: {X_test.shape}, Labels Shape: {y_test.shape}")

    # 3. Train the Multiclass Model
    model = train_model(X_train, y_train, X_test, y_test)

    # 4. Quantize the Trained Model to INT8
    # Pass a portion of X_train for the representative_dataset to use
    tflite_model = convert_to_tflite_int8(model, X_train)

    # 5. Export Weights (COE files) and Perform Hardware Verification
    export_weights_and_verify(tflite_model, X_test, y_test)

    print("\n--- Pipeline Complete ---")
    print(f"Generated FPGA configuration files (.coe) are in: '{OUTPUT_DIR}'")

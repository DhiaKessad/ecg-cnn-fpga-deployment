import os
import subprocess
import shutil
import re

def run_test_suite(test_dir_name, rtl_dir, results_file):
    test_dir = os.path.join(rtl_dir, "hex", test_dir_name)
    manifest_path = os.path.join(test_dir, "test_manifest.txt")
    target_hex = os.path.join(rtl_dir, "hex", "input_sample_0.hex")
    sim_cmd = ["vvp", "simv"]
    
    if not os.path.exists(manifest_path):
        results_file.write(f"Manifest not found in {test_dir}\n")
        return

    results_file.write(f"\n--- Running Tests for {test_dir_name} ---\n")
    results_file.write(f"{'Sample':<25} | {'Expected':<10} | {'Actual':<10} | {'Status':<10}\n")
    results_file.write("-" * 65 + "\n")

    correct = 0
    total = 0

    print(f"Compiling Verilog files in {rtl_dir}...")
    subprocess.run(["iverilog", "-o", "simv", "tb_ecg.v", "top_ecg_classifier.v"] + [f"layer{i}_conv.v" for i in [1,3,5]] + ["layer6_avg.v", "layer7_dense_multi.v", "layer8_argmax.v", "layer2_maxpool.v", "layer4_maxpool.v"], cwd=rtl_dir, capture_output=True, text=True, timeout=120.0, stdin=subprocess.DEVNULL)
    print("Compilation complete.")

    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2:
                continue
                
            coe_file = parts[0]
            expected_id = parts[1]
            hex_file = coe_file.replace('.coe', '.hex')
            hex_path = os.path.join(test_dir, hex_file)
            
            if not os.path.exists(hex_path):
                results_file.write(f"{hex_file:<25} | {expected_id:<10} | {'MISSING':<10} | FAIL\n")
                continue

            # Copy sample to input
            shutil.copy(hex_path, target_hex)
            
            # Run simulation
            try:
                process = subprocess.run(sim_cmd, cwd=rtl_dir, capture_output=True, text=True, timeout=10.0, stdin=subprocess.DEVNULL)
                stdout = process.stdout
                
                # Parse output: "Classification Output (Argmax Class ID):     1"
                match = re.search(r"Classification Output \(Argmax Class ID\):\s+(\d+)", stdout)
                if match:
                    actual_id = match.group(1)
                else:
                    actual_id = "TIMEOUT_V" if "TIMEOUT" in stdout else "ERROR"
            except subprocess.TimeoutExpired:
                actual_id = "HANG"
            except Exception as e:
                actual_id = f"EXCEPTION"
            
            status = "PASS" if actual_id == expected_id else "FAIL"
            if status == "PASS":
                correct += 1
            total += 1
            
            print(f"{hex_file:<25} | {expected_id:<10} | {actual_id:<10} | {status:<10}")
            results_file.write(f"{hex_file:<25} | {expected_id:<10} | {actual_id:<10} | {status:<10}\n")
            results_file.flush()

    accuracy = (correct / total) * 100 if total > 0 else 0
    error_rate = 100 - accuracy
    results_file.write("-" * 65 + "\n")
    results_file.write(f"Summary for {test_dir_name} (3-Class):\n")
    results_file.write(f"Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%, Error Rate: {error_rate:.2f}%\n")

def run_pooled_analysis(test_dir_name, rtl_dir, results_file):
    test_dir = os.path.join(rtl_dir, "hex", test_dir_name)
    manifest_path = os.path.join(test_dir, "test_manifest.txt")
    target_hex = os.path.join(rtl_dir, "hex", "input_sample_0.hex")
    sim_cmd = ["vvp", "simv"]
    
    results_file.write(f"\n--- Pooled Analysis (Class 1 & 2 combined) for {test_dir_name} ---\n")
    results_file.write(f"{'Sample':<25} | {'Exp (Pooled)':<12} | {'Act (Pooled)':<12} | {'Status':<10}\n")
    results_file.write("-" * 70 + "\n")

    correct = 0
    total = 0

    print(f"Compiling Verilog files in {rtl_dir} for pooled analysis...")
    subprocess.run(["iverilog", "-o", "simv", "tb_ecg.v", "top_ecg_classifier.v"] + [f"layer{i}_conv.v" for i in [1,3,5]] + ["layer6_avg.v", "layer7_dense_multi.v", "layer8_argmax.v", "layer2_maxpool.v", "layer4_maxpool.v"], cwd=rtl_dir, capture_output=True, text=True, timeout=120.0, stdin=subprocess.DEVNULL)
    print("Compilation complete.")

    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2: continue
                
            expected_id = parts[1]
            hex_file = parts[0].replace('.coe', '.hex')
            hex_path = os.path.join(test_dir, hex_file)
            if not os.path.exists(hex_path): continue

            shutil.copy(hex_path, target_hex)
            try:
                process = subprocess.run(sim_cmd, cwd=rtl_dir, capture_output=True, text=True, timeout=10.0, stdin=subprocess.DEVNULL)
                stdout = process.stdout
                match = re.search(r"Classification Output \(Argmax Class ID\):\s+(\d+)", stdout)
                actual_id = match.group(1) if match else "ERROR"
            except subprocess.TimeoutExpired:
                actual_id = "HANG"
            except:
                actual_id = "EXCEPTION"
            
            # Pooling logic: 1 and 2 are both "Abnormal"
            exp_pooled = "Abnormal" if expected_id in ["1", "2"] else "Normal"
            act_pooled = "Abnormal" if actual_id in ["1", "2"] else "Normal"
            
            status = "PASS" if exp_pooled == act_pooled else "FAIL"
            if status == "PASS": correct += 1
            total += 1
            
            results_file.write(f"{hex_file:<25} | {exp_pooled:<12} | {act_pooled:<12} | {status:<10}\n")

    accuracy = (correct / total) * 100 if total > 0 else 0
    results_file.write("-" * 70 + "\n")
    results_file.write(f"Pooled Summary for {test_dir_name}:\n")
    results_file.write(f"Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%\n")

def main():
    rtl_dir = "/Users/sid/Library/CloudStorage/GoogleDrive-m.kessad@nsnn.edu.dz/My Drive/NHSNN Personal Drive/FPGA/projects/ECG/Multi/RTL"
    
    output_path = os.path.join(rtl_dir, "test_results.txt")
    with open(output_path, 'w') as results_file:
        results_file.write("ECG Classifier (Multi-Class) Results\n")
        results_file.write("=" * 70 + "\n")
        run_test_suite("test_inputs_1", rtl_dir, results_file)
    print(f"Test suite completed. Results written to {output_path}")

    output_path_pooled = os.path.join(rtl_dir, "test_results_pooled.txt")
    with open(output_path_pooled, 'w') as results_file:
        results_file.write("ECG Classifier Pooled (Analysis) Results\n")
        results_file.write("=" * 70 + "\n")
        run_pooled_analysis("test_inputs_1", rtl_dir, results_file)
    print(f"Pooled analysis completed. Results written to {output_path_pooled}")

if __name__ == "__main__":
    main()

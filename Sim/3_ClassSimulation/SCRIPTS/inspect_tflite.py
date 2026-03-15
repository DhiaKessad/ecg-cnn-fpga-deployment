import tensorflow as tf
import numpy as np

def inspect_tflite(model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        details = interpreter.get_tensor_details()
        for detail in details:
            name = detail['name']
            shape = detail['shape']
            dtype = detail['dtype']
            index = detail['index']
            print(f"Tensor {index}: {name} | Shape: {shape} | Dtype: {dtype}")
        
        ops = interpreter.get_nodes_details()
        for op in ops:
            print(f"Op: {op['op_name']} | Inputs: {op['inputs']} | Outputs: {op['outputs']}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_tflite("/Users/sid/Library/CloudStorage/GoogleDrive-m.kessad@nsnn.edu.dz/My Drive/NHSNN Personal Drive/FPGA/projects/ECG/Multi/RTL/multiclass_fpga_outputs/ecg_model_multiclass_quant.tflite")

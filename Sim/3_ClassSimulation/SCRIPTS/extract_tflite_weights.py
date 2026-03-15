import os
import sys
import tflite
import numpy as np

def convert_to_hex_lines(numpy_array, bit_width=16):
    """Convets a flat numpy array of ints into a list of hex strings."""
    hex_lines = []
    if numpy_array is None or len(numpy_array) == 0:
        return hex_lines

    for val in numpy_array:
        int_val = int(val)
        if int_val < 0:
            int_val = (1 << bit_width) + int_val
        hex_str = f"{int_val:04x}"
        hex_lines.append(hex_str)
    return hex_lines

def run_extraction(tflite_path, out_dir):
    with open(tflite_path, 'rb') as f:
        buf = f.read()

    model = tflite.Model.GetRootAsModel(buf, 0)
    subgraph = model.Subgraphs(0)

    print(f"Loaded {tflite_path} ({subgraph.TensorsLength()} tensors)")

    # The exact weight tensors can be identified by their constant data element counts
    # layer1: 16 * 5 = 80       | bias: 16
    # layer3: 32 * 3 * 16 = 1536| bias: 32
    # layer5: 64 * 3 * 32 = 6144| bias: 64
    # layer7: 3 * 64 = 192      | bias: 3
    expected_weight_sizes = {
        80: 'w_weight1',
        1536: 'w_weight2',
        6144: 'w_weight3',
        192: 'w_weight_dense'
    }
    expected_bias_sizes = {
        16: 'b_bias1',
        32: 'b_bias2',
        64: 'b_bias3'
    }

    found_tensors = {}

    for i in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(i)
        name = tensor.Name().decode('utf-8')
        shape = tensor.ShapeAsNumpy()

        buffer_idx = tensor.Buffer()
        buffer = model.Buffers(buffer_idx)
        data = buffer.DataAsNumpy()

        if data is not None and shape is not None:
            num_elements = np.prod(shape)
            layer_name = None
            is_bias = False
            
            if num_elements in expected_weight_sizes and tensor.Type() == tflite.TensorType.INT8:
                layer_name = expected_weight_sizes[num_elements]
            elif num_elements in expected_bias_sizes and tensor.Type() == tflite.TensorType.INT32:
                layer_name = expected_bias_sizes[num_elements]
                is_bias = True
            elif num_elements == 3 and tensor.Type() == tflite.TensorType.INT32 and "BiasAdd" in name:
                # Explicit override since there's multiple tensors of size 3
                layer_name = 'b_bias_dense'
                is_bias = True
                
            if layer_name is not None:
                if is_bias:
                    raw_data = np.frombuffer(data, dtype=np.int32)
                else:
                    raw_data = np.frombuffer(data, dtype=np.int8)
                
                # De-quantize to float using the Tensor scales
                q = tensor.Quantization()
                float_data = np.zeros_like(raw_data, dtype=np.float32)
                if q is not None and q.ScaleLength() > 0:
                    scales = q.ScaleAsNumpy()
                    zps = q.ZeroPointAsNumpy()
                    
                    if len(scales) == 1:
                        float_data = (raw_data - zps[0]) * scales[0]
                    else:
                        # Per-channel quantized
                        raw_data_reshaped = raw_data.reshape(shape)
                        float_data = float_data.reshape(shape)
                        if is_bias:
                            # Biases are 1D arrays [OutChannels]
                            for c in range(shape[0]):
                                float_data[c] = (raw_data_reshaped[c] - zps[c]) * scales[c]
                        elif layer_name == 'w_weight_dense':
                            for c in range(shape[0]):
                                float_data[c, :] = (raw_data_reshaped[c, :] - zps[c]) * scales[c]
                        else:
                            for c in range(shape[0]):
                                float_data[c, :, :, :] = (raw_data_reshaped[c, :, :, :] - zps[c]) * scales[c]
                else:
                    float_data = raw_data.astype(np.float32)

                # Convert float to Q8.8 fixed-point (multiply by 256 and round)
                q88_data = np.round(float_data * 256.0).astype(np.int32)
                q88_data = q88_data.reshape(shape)
                
                # Reshape/Transpose to match Verilog loops
                if not is_bias:
                    if len(shape) == 4 and shape[1] == 1:
                        q88_data = q88_data.reshape(shape[0], shape[2], shape[3])
                        
                    if layer_name == 'w_weight_dense':
                        q88_data = q88_data.transpose()
                    
                found_tensors[layer_name] = (name, q88_data.shape, q88_data)

    os.makedirs(out_dir, exist_ok=True)

    
    for layer_name, (tensor_name, shape, weights) in found_tensors.items():
        hex_lines = convert_to_hex_lines(weights.flatten(), bit_width=16)
        
        out_path = os.path.join(out_dir, f"{layer_name}.hex")
        with open(out_path, 'w') as f:
            f.write('\n'.join(hex_lines) + '\n')
            
        print(f"[{layer_name}] Shape {shape} -> '{tensor_name}' -> wrote {len(hex_lines)} hex lines to {layer_name}.hex")

    total_expected = len(expected_weight_sizes) + len(expected_bias_sizes) + 1 # +1 for b_bias_dense
    if len(found_tensors) == total_expected:
        print(f"\nSUCCESS: Extracted all {len(found_tensors)} weight/bias tensors.")
    else:
        print(f"\nWARNING: Only found {len(found_tensors)}/{total_expected} expected tensors.")

if __name__ == '__main__':
    tflite_file = "RTL/fpga_outputs/ecg_model_multiclass_quant.tflite"
    out_dir = "RTL/hex"
    run_extraction(tflite_file, out_dir)

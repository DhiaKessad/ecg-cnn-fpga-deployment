import tflite

with open("RTL/fpga_outputs/ecg_model_multiclass_quant.tflite", "rb") as f:
    buf = f.read()
model = tflite.Model.GetRootAsModel(buf, 0)
subgraph = model.Subgraphs(0)

for i in range(subgraph.TensorsLength()):
    tensor = subgraph.Tensors(i)
    name = tensor.Name().decode('utf-8')
    q = tensor.Quantization()
    if q is not None:
        scales = q.ScaleAsNumpy() if q.ScaleLength() > 0 else "None"
        zps = q.ZeroPointAsNumpy() if q.ZeroPointLength() > 0 else "None"
        print(f"[{i}] {name} | Type: {tensor.Type()} | Scale: {scales} | ZP: {zps}")
    else:
        print(f"[{i}] {name} | Type: {tensor.Type()} | NO QUANT")

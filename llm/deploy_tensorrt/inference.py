import tensorrt as trt
import numpy as np
import torch
import pycuda.driver as cuda
import pycuda.autoinit

engine_file = "bert-base-uncased.plan"

# Load engine
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
with open(engine_file, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# Example input
batch_size, seq_len = 1, 16
input_ids = np.random.randint(0, 1000, size=(batch_size, seq_len)).astype(np.int32)
attention_mask = np.ones_like(input_ids, dtype=np.int32)

# Allocate buffers
inputs, outputs, bindings = [], [], []
stream = cuda.Stream()

for binding in engine:
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    shape = context.get_binding_shape(binding)
    size = np.prod(shape)
    device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
    bindings.append(int(device_mem))
    if engine.binding_is_input(binding):
        inputs.append(device_mem)
    else:
        outputs.append((binding, device_mem, shape, dtype))

# Copy input to device
cuda.memcpy_htod(inputs[0], input_ids)
cuda.memcpy_htod(inputs[1], attention_mask)

# Run inference
context.execute_v2(bindings)

# Retrieve output
for name, device_mem, shape, dtype in outputs:
    host_out = np.empty(shape, dtype=dtype)
    cuda.memcpy_dtoh(host_out, device_mem)
    print(f"{name}:", host_out.shape)

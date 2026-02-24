import os, torch

def fmt(v):
    if v is None: return "None"
    if isinstance(v, (tuple, list)): return ".".join(map(str, v))
    return str(v)

print("=== Torch / CUDA stack ===")
print(f"torch          : {torch.__version__}")
print(f"cuda available : {torch.cuda.is_available()}")
print(f"torch cuda     : {fmt(getattr(torch.version, 'cuda', None))}")

# cuDNN
print(f"cudnn enabled  : {torch.backends.cudnn.enabled}")
print(f"cudnn version  : {torch.backends.cudnn.version()}")

# NCCL (only meaningful if CUDA is available)
nccl_ver = None
if torch.cuda.is_available():
    try:
        nccl_ver = torch.cuda.nccl.version()
    except Exception as e:
        nccl_ver = f"unavailable ({type(e).__name__}: {e})"
print(f"nccl version   : {fmt(nccl_ver)}")

# Useful extras
if torch.cuda.is_available():
    i = torch.cuda.current_device()
    print(f"gpu            : {torch.cuda.get_device_name(i)}")
    print(f"gpu cc         : {fmt(torch.cuda.get_device_capability(i))}")
    print(f"driver/runtime : n/a (use nvidia-smi / nvcc for exact)")
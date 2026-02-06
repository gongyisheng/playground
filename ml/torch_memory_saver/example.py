"""Demo: torch_memory_saver CPU backup behavior."""
import hashlib
import time
import torch
from torch_memory_saver import torch_memory_saver

torch_memory_saver.hook_mode = "torch"

# functionality: keep address & pointer, offload to cpu / reload to cuda safely
# pause: offload to cpu
# resume: onload to cuda
# with torch_memory_saver.disable(): create tensors in a separated mem pool, not tracked
# known issue: torch_memory_saver uses cuMemCreate/cuMemMap, incompatible to cuda IPC (cudaIpcGetMemHandle)
# need to expose a handler through cuMemExportToShareableHandle 


def tensor_hash(tensor: torch.Tensor) -> str:
    """Calculate SHA256 hash of tensor by viewing as uint8 bytes."""
    tensor_bytes = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()


def test_pause_resume(enable_cpu_backup: bool):
    with torch_memory_saver.region(enable_cpu_backup=enable_cpu_backup):
        tensor = torch.full((1024,1024,1024,), 42.0, device='cuda', dtype=torch.float32) # ~4GB
        original_hash = tensor_hash(tensor)
        print(f"Allocated tensor on CUDA: {tensor.size()}")

    time.sleep(5)

    print("Pausing memory saver...")
    torch_memory_saver.pause("demo")

    time.sleep(5)  # Simulate time passing; tensor may be offloaded here.

    print("Resuming memory saver...")
    torch_memory_saver.resume("demo")
    new_hash = tensor_hash(tensor)

    print(f"cpu_backup={enable_cpu_backup}: preserved={original_hash == new_hash}")
    print(f"  original: {original_hash[:16]}...")
    print(f"  after:    {new_hash[:16]}...")


if __name__ == "__main__":
    test_pause_resume(enable_cpu_backup=True)
    # test_pause_resume(enable_cpu_backup=False)

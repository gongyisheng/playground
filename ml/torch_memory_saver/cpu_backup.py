"""Demo: torch_memory_saver CPU backup behavior."""
import torch
import torch_memory_saver


def test_pause_resume(enable_cpu_backup: bool):
    with torch_memory_saver.region(tag="demo", enable_cpu_backup=enable_cpu_backup):
        tensor = torch.full((1000,), 42.0, device='cuda', dtype=torch.float32)

    original_sum = tensor.sum().item()
    torch_memory_saver.pause("demo")
    torch_memory_saver.resume("demo")
    new_sum = tensor.sum().item()

    print(f"cpu_backup={enable_cpu_backup}: {original_sum} -> {new_sum}, preserved={original_sum == new_sum}")


if __name__ == "__main__":
    test_pause_resume(enable_cpu_backup=True)
    test_pause_resume(enable_cpu_backup=False)

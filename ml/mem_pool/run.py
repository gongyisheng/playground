"""Swap optimizer state for three models through one device slot."""

import torch

from .pool import LRUMemoryPool


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [torch.nn.Linear(4, 2, bias=False).to(device) for _ in range(3)]
    optimizers = [
        torch.optim.AdamW(model.parameters(), lr=0.01, capturable=device == "cuda")
        for model in models
    ]
    pool = LRUMemoryPool(capacity=1, device=device)

    # Initialize CPU state once; the pool handles all movement afterward.
    for key, optimizer in enumerate(optimizers):
        for p in models[key].parameters():
            optimizer.state[p] = {
                "step": torch.tensor(0.0),
                "exp_avg": torch.zeros_like(p, device="cpu"),
                "exp_avg_sq": torch.zeros_like(p, device="cpu"),
            }
        pool.register(key, optimizer.state)

    for key in (0, 1, 2, 0):
        optimizer = optimizers[key]
        for p in models[key].parameters():
            p.grad = torch.ones_like(p)  # Fixed gradients for this demonstration.

        # Offload the previous optimizer's state and onload this optimizer's state.
        with pool.acquire(key):
            optimizer.step()
        print(f"Stepped optimizer {key}; resident={pool.resident_keys}")

    for key in range(len(optimizers)):
        pool.offload(key)


if __name__ == "__main__":
    main()

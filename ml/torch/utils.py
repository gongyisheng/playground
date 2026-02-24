import hashlib
import torch


def compute_hash(t: torch.Tensor) -> str:
    """Compute SHA256 hash for a tensor."""
    # view: no conversion, no information lost
    # t.to(): will cause information lost
    return hashlib.sha256(
        t.detach().cpu().contiguous().view(torch.uint8).numpy()
    ).hexdigest()


def test_compute_hash():
    t1 = torch.tensor([1.0, 2.0, 3.0])
    t2 = torch.tensor([1.0, 2.0, 3.0])
    t3 = torch.tensor([1.0, 2.0, 4.0])

    hash1 = compute_hash(t1)
    hash2 = compute_hash(t2)
    hash3 = compute_hash(t3)

    print(f"t1 hash: {hash1}")
    print(f"t2 hash: {hash2}")
    print(f"t3 hash: {hash3}")
    print(f"t1 == t2: {hash1 == hash2}")
    print(f"t1 == t3: {hash1 == hash3}")

if __name__ == "__main__":
    test_compute_hash()

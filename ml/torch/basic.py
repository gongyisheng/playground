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


def test_transpose():
    q = torch.rand(1,4,3)
    print("Before transpose:")
    print(q.shape) # [1,4,3]
    print(q)

    q = q.transpose(-1,-2)
    print("After transpose:")
    print(q.shape) # [1,3,4]
    print(q)

def test_mean():
    q = torch.rand(2,2,5)
    mean_q = q.mean(dim=-1, keepdim=False)
    print("keepdim=False:")
    print(mean_q.shape) # [2,2]
    print(mean_q)

    mean_q = q.mean(dim=-1, keepdim=True)
    print("keepdim=True:")
    print(mean_q.shape) # [2,2,1]
    print(mean_q)

def test_variance():
    q = torch.rand(2,5)
    var_q = q.var(dim=-1, keepdim=True, unbiased=True) # divide by n-1, works with sample (estimation)
    print("unbiased=True:")
    print(var_q.shape)
    print(var_q)

    var_q = q.var(dim=-1, keepdim=True, unbiased=False) # divide by n, works with population variance
    print("unbiased=False:")
    print(var_q.shape)
    print(var_q)

def test_contiguous():
    # operations can break contiguity
    # transpose, permute, narrow, expand, view, tensor[::2]
    
    # need to be contiguous before
    # view/reshape, .numpy(), or other cuda operations
    
    q = torch.rand(2, 3, 4)
    print("Original tensor, is contiguous:", q.is_contiguous())

    q_transposed = q.transpose(1, 2)
    print("After transpose, is contiguous:", q_transposed.is_contiguous())

    q_contiguous = q_transposed.contiguous()
    print("After contiguous, is contiguous:", q_contiguous.is_contiguous())

def test_advanced_indexing():
    batch_size = 4
    log_probs_example = torch.tensor([
        [-2.1, -1.5, -0.8, -3.2, -2.7, -1.2],  # Sample 0
        [-1.3, -2.4, -0.5, -1.8, -2.1, -3.0],  # Sample 1
        [-0.9, -1.7, -2.3, -0.6, -1.1, -2.8],  # Sample 2
        [-2.5, -0.7, -1.9, -2.2, -1.4, -3.1],  # Sample 3
    ])
    targets_example = torch.tensor([2, 2, 3, 1])

    row_indices = torch.arange(batch_size)
    print(f"Row indices: {row_indices}")
    print(f"Column indices: {targets_example}")

    # advanced indexing
    target_log_probs_example = log_probs_example[row_indices, targets_example]
    print(f"Result: {target_log_probs_example}")

def test_init():
    x = torch.tensor([[1,2,3],[4,5,6]])
    exp_1 = torch.zeros_like(x) # init with zeros
    exp_2 = torch.ones_like(x)  # init with ones
    exp_3 = torch.full_like(x, 7) # init with full value
    exp_4 = torch.empty_like(x) # init with uninitialized values in memory (fastest)
    print("Original tensor:")
    print(x)
    print("Zeros like:")
    print(exp_1)
    print("Ones like:")
    print(exp_2)
    print("Full like (7):")
    print(exp_3)
    print("Empty like (uninitialized):")
    print(exp_4)

def test_property():
    x = torch.randn(3,4)
    print("Tensor x:")
    print(x)
    print("Shape:", x.shape)
    print("Size:", x.size())
    print("Dtype:", x.dtype)
    print("Device:", x.device)
    print("Numel:", x.numel())
    print("Is contiguous:", x.is_contiguous())


if __name__ == "__main__":
    test_compute_hash()
    # test_transpose()
    # test_mean()
    # test_variance()
    # test_contiguous()
    # test_advanced_indexing()
    # test_init()
    # test_property()
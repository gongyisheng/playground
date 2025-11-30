import torch

def test_transpose():
    q = torch.rand(1,4,3)
    print("Before transpose:")
    print(q.shape)
    print(q)

    q = q.transpose(-1,-2)
    print("After transpose:")
    print(q.shape)
    print(q)

def test_mean():
    q = torch.rand(2,5)
    mean_q = q.mean(dim=-1, keepdim=False)
    print("keepdim=False:")
    print(mean_q.shape)
    print(mean_q)

    mean_q = q.mean(dim=-1, keepdim=True)
    print("keepdim=True:")
    print(mean_q.shape)
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


if __name__ == "__main__":
    # test_transpose()
    # test_mean()
    # test_variance()
    # test_contiguous()
    test_advanced_indexing()
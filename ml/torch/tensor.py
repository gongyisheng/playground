import torch


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

def test_init_shape():
    # create tensors with explicit shape (vs *_like which copies shape from another tensor)
    a = torch.ones(3)          # [1, 1, 1]
    b = torch.ones(2, 4)      # 2x4 matrix of ones
    c = torch.zeros(5)        # [0, 0, 0, 0, 0]
    d = torch.full((2, 3), 9) # 2x3 matrix filled with 9
    print("ones(3):", a)
    print("ones(2,4):", b)
    print("zeros(5):", c)
    print("full((2,3), 9):", d)

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
    test_init()
    # test_property()
    # test_advanced_indexing()

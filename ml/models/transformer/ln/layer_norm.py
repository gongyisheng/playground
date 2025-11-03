import torch
import torch.nn as nn

# Explanations
# 1. scale (gamma):
#     after (x-mean)/std, all features are forced to have mean=0, std=1
#     it restricted model's ability to decide feature importance (which should be more important)
#     scale gives model flexibility to learn optimal magnitudes for each feature
# 2. shift (beta):
#     allows model to learn optimal bias/offset for each feature
#     without it, normalized features would always be centered at 0
#     shift enables model to learn if certain features should have non-zero centers

class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))  # gamma
        self.shift = nn.Parameter(torch.zeros(dim))  # beta

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        norm = (x - mean) / std
        return self.scale * norm + self.shift

def test():
    x = torch.randn(3, 5)  # e.g., (batch_size=3, features=5)
    custom_layernorm = LayerNorm(dim=5, eps=1e-8)
    custom_output = custom_layernorm(x)
    print("Custom output:")
    print(custom_output)
    assert custom_output.shape == (3, 5), "Output shape mismatch"

    layernorm = torch.nn.LayerNorm(5, eps=1e-8)
    output = layernorm(x)
    print("PyTorch output:")
    print(output)

    assert torch.allclose(output, custom_output, atol=1e-08, rtol=1e-05)

if __name__ == "__main__":
    test()

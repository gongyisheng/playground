import torch
import torch.nn as nn

# Explanations
# 1. scale:
#     after x/norm, all features are forced to have RMS = 1
#     it restricted model's ability to decide feature importance (which should be more important)
#     scale gives model flexibility to klearn optimal magnitudes
#     do not use torch.rand() to avoid introducing random suppression of features (all less than 1)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))  # gamma
    
    def forward(self, x):
        norm = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # RMS
        return (x / norm) * self.scale
    
def test():
    x = torch.randn(3, 5)  # e.g., (batch_size=3, features=5)
    custom_rmsnorm = RMSNorm(dim=5, eps=1e-8)
    custom_output = custom_rmsnorm(x)
    print("Custom output:")
    print(custom_output)
    assert custom_output.shape == (3, 5), "Output shape mismatch"
    
    rmsnorm = torch.nn.RMSNorm(5, eps=1e-8)
    output = rmsnorm(x)
    print("PyTorch output:")
    print(output)

    assert torch.allclose(output, custom_output, atol=1e-08, rtol=1e-05)

if __name__ == "__main__":
    test()
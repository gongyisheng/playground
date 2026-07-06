import torch
import torch.nn as nn
import torch.nn.functional as F


# dense mlp
# x -> up_proj -> activation -> down_proj

class DenseMLP(nn.Module):

    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()
        self.d_model = d_model
        self.intermediate_size = intermediate_size
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        self.act_func = F.silu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        x = self.act_func(x)
        x = self.down_proj(x)
        return x


# dense gated mlp
# x -> gate_proj -> act
# x -> up_proj       |   -> down_proj

class DenseGatedMLP(nn.Module):

    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()
        self.d_model = d_model
        self.intermediate_size = intermediate_size
        self.gate_up_proj = nn.Linear(d_model, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        self.act_func = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        gate = self.act_func(gate)
        x = self.down_proj(gate * up)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# x -> gate -> top-k -> softmax
class MoERouter(nn.Module):

    def __init__(self, d_model: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [BS, D]
        logits = self.gate(x).float() # [BS, num_experts]
        topk_logits, topk_idx = torch.topk(logits, self.top_k) # [BS, top_k]
        weights = torch.softmax(topk_logits, dim=-1).to(x.dtype) # [BS, top_k]
        return topk_idx, weights, logits


# x -> reshape -> router, get topk and weight -> for loop traverse + acc -> reshape back
# for loop: get token_idx+slot_idx -> stack hidden state -> expert forward -> get weights -> accum
# auxloss: 
class SparseMoEBlockV1(nn.Module):

    def __init__(self, d_model: int, intermediate_size: int, n_expert: int, top_k: int):
        super().__init__()
        self.d_model = d_model
        self.intermediate_size = intermediate_size
        self.n_expert = n_expert
        self.top_k = top_k
        self.router = MoERouter(d_model, n_expert, top_k)
        self.experts = nn.ModuleList([DenseGatedMLP(d_model, intermediate_size) for _ in range(n_expert)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x = x.reshape(B*S, D)
        topk_idx, expert_weights, logits = self.router(x)  # [BS, k], [BS, k]

        out = torch.zeros_like(x)  # [BS, D] — accumulate weighted expert outputs here
        for i in range(self.n_expert):
            token_idx, slot_idx = torch.where(topk_idx == i)
            if token_idx.numel() == 0:
                continue
            bucket = x[token_idx] # [n_e, D]
            y = self.experts[i](bucket) # [n_e, D]
            w = expert_weights[token_idx, slot_idx].unsqueeze(-1) # [n_e, 1]
            out.index_add_(0, token_idx, y * w)
            # out[token_idx] += y * w

        return out.reshape(B, S, D)
    

def grouped_mm(a, b, offs):
    # for gate_up_proj:
    # a: [M, K], M = B*S*topk, K = d_model, each token duplicated by topk times
    # b: [E, K, N], K = n_model, N = intermediate_size * 2
    # off: [E], offset of each expert in a, eg, [3,3,8,10,...],
    #      e1: [0,3], e2: [3,3], e3: [3,8], etc
    if hasattr(torch, "_grouped_mm"):
        return torch._grouped_mm(a, b, offs=offs)
    else:
        out = torch.empty(a.shape[0], b.shape[-1], dtype=a.dtype, device=a.device)
        start = 0
        for e in range(b.shape[0]):
            end = int(offs[e])
            if end > start:
                out[start:end] = a[start:end] @ b[e]
            start = end
        return out


class SparseMoEBlockV2(nn.Module):

    def __init__(self, d_model: int, intermediate_size: int, n_expert: int, top_k: int):
        super().__init__()
        self.d_model = d_model
        self.intermediate_size = intermediate_size
        self.n_expert = n_expert
        self.top_k = top_k
        self.router = MoERouter(d_model, n_expert, top_k)
        self.expert_gate_up_proj = nn.Parameter(torch.empty(n_expert, d_model, intermediate_size * 2))
        self.expert_down_proj = nn.Parameter(torch.empty(n_expert, intermediate_size, d_model))
        self.act_func = F.silu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x = x.reshape(B*S, D)
        expert_idx, expert_weight, logits = self.router(x)  # [BS, k], [BS, k]
        BS, k = expert_idx.shape

        # expand
        expert_ids = expert_idx.reshape(-1) # [BS*k]
        token_ids = torch.arange(BS, device=x.device).repeat_interleave(k) # [BS*k]
        w = expert_weight.reshape(-1, 1).to(x.dtype) # [BS*k]

        # sort
        order = torch.argsort(expert_ids) # [BS*k]
        expert_ids = expert_ids[order] # [BS*k]
        token_ids = token_ids[order] # [BS*k]
        w = w[order] # [BS*k]
        x_sorted = x[token_ids] # [BS*k, D]

        # grouped mm
        offs = torch.bincount(expert_ids, minlength=self.n_expert).cumsum(0).to(torch.int32)
        gate_up = grouped_mm(x_sorted, self.expert_gate_up_proj, offs)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = self.act_func(gate) * up
        y = grouped_mm(hidden, self.expert_down_proj, offs)
        y = y * w

        # unsort, sum
        out = torch.zeros_like(x)
        out.index_add_(0, token_ids, y) # when token_ids has dup, sum up
        return out.reshape(B, S, D)


def compute_aux_loss(logits: torch.Tensor, topk_idx: torch.Tensor):
    # logits: [B*S, E]
    # topk_idx: [B*S, topk]
    BS, n_experts = logits.shape
    _, topk = topk_idx.shape
    probs = torch.softmax(logits, dim=-1) # [B*S, E]
    P = probs.mean(dim=0) # [E,]
    f = F.one_hot(topk_idx, n_experts).float().sum(dim=(0, 1)) / (BS * topk) # [B*S, topk, E]
    aux = n_experts * (f * P).sum()
    return aux
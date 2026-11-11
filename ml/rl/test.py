import torch
import torch.nn.functional as F

reward = torch.Tensor(
    [
        [1,1,0,0],
        [0,1,1,1],
        [1,1,1,1],
        [0,0,0,0]
    ]
)
all_zero_group = (reward == 0).all(dim=-1)
all_one_group = (reward == 1).all(dim=-1)
keep_group = ~(all_zero_group | all_one_group)
# keep_group = torch.ones_like(all_zero_group, dtype=torch.bool)
#print(keep_group)
keep_reward = reward[keep_group]

keep_group = reward.amax(dim=-1) != reward.amin(dim=-1)
#print(keep_group)
#print(keep_reward)
#print(reward.mean(dim=-1, keepdim=True))
#print(reward.std(dim=-1, correction=0, keepdim=True))

logprob1 = torch.rand([1, 1024])

# Example layout, using positions in logprob1 (slice ends are exclusive):
# [0:256] prompt | [256:512] assistant/tool call | [512:640] tool response
# [640:1024] assistant's final answer
# True = include in the loss; False = ignore. Use actual token spans in practice.
response_mask1 = torch.ones_like(logprob1, dtype=torch.bool)
response_mask1[:, :256] = False     # Ignore the prompt.
response_mask1[:, 512:640] = False  # Ignore the tool's response.

# This is a LOSS mask, not an attention mask: the model still reads the prompt
# and tool response when predicting assistant tokens. Keep assistant tool calls.
# Match the mask to the tokens scored by logprob1. If logprobs score input_ids
# [:, 1:] after the causal shift, shift a full-input token mask with [:, 1:] too.
masked_logprob1 = logprob1.masked_fill(~response_mask1, 0.0)
assistant_token_count = response_mask1.sum(dim=-1)  # [1], containing 640
# Divide by included tokens, not all 1024 positions (so don't just use .mean()).
mean_logprob1 = masked_logprob1.sum(dim=-1) / assistant_token_count
#print("Assistant token count:", assistant_token_count)
#print("Mean logprob over assistant tokens:", mean_logprob1)
# For PPO/GRPO, also mask the final per-token loss before reducing: zeroing
# logprobs alone is insufficient because exp(0 - 0) still gives a ratio of 1.

logprob2 = torch.rand([1, 512])
# print(logprob1.shape, logprob2.shape)

logprob2_padded = F.pad(logprob2, (0, logprob1.shape[-1]-logprob2.shape[-1]), value=0)
combined = torch.cat([logprob1, logprob2_padded], dim=0)
# print(combined)

resp_mask1 = torch.ones_like(logprob1, dtype=torch.bool)
resp_mask1[:, :256] = False # prompt
resp_mask1[:, 512:640] = False # tool response

masked_logprob1 = logprob1.masked_fill(~response_mask1, 0.0)
mean_logprob1 = masked_logprob1.sum(dim=-1) / response_mask1.sum(dim=-1)

import math
@torch.no_grad()
def grpo_advantage(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert math.isfinite(eps) and eps > 0, "invalid eps"
    assert torch.isfinite(rewards).all(), "nonfinite rewards"

    rewards = rewards.float()
    mean = rewards.mean(dim=-1, keepdim=True)
    # A singleton has no relative reward signal; avoid sample std's 0/0.
    std = (
        rewards.std(dim=-1, correction=1, keepdim=True)
        if rewards.shape[-1] != 1
        else torch.zeros_like(mean)
    )
    advantages = (rewards - mean) / (std + eps)
    assert torch.isfinite(std).all(), "nonfinite reward std"
    assert torch.isfinite(advantages).all(), "nonfinite advantages"
    return advantages

reward = torch.Tensor([
    [1.0, 1.0],
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1e-7],
])
print(grpo_advantage(reward))
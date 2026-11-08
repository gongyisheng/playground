import math

import torch


# advantage: A_i = (R_i - group_mean) / (group_std + eps)
# rewards: [B, G] (batch, groups)
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


def grpo_loss(
    curr_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    clip_eps: float = 0.2,
    eps: float = 1e-6,
    ref_log_probs: torch.Tensor | None = None,
    kl_beta: float = 0.0,
) -> torch.Tensor:
    assert curr_log_probs.ndim == 3
    assert curr_log_probs.shape == old_log_probs.shape
    assert curr_log_probs.shape == response_mask.shape
    assert curr_log_probs.shape[:-1] == rewards.shape
    assert response_mask.dtype == torch.bool
    assert 0 <= clip_eps < 1, "invalid clip_eps"
    assert math.isfinite(kl_beta) and kl_beta >= 0, "invalid kl_beta"

    advantages = grpo_advantage(rewards, eps).unsqueeze(-1)  # [B, G, 1]
    masked_curr_log_probs = curr_log_probs.float().masked_fill(~response_mask, 0.0)
    masked_old_log_probs = (old_log_probs.detach().float().masked_fill(~response_mask, 0.0))
    assert torch.isfinite(masked_curr_log_probs).all(), "nonfinite curr_log_probs"
    assert torch.isfinite(masked_old_log_probs).all(), "nonfinite old_log_probs"

    # clamp log ratio to avoid nan
    log_ratio = (masked_curr_log_probs - masked_old_log_probs).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    assert torch.isfinite(ratio).all(), "nonfinite policy ratio"
    clipped_ratio = ratio.clamp(1 - clip_eps, 1 + clip_eps)
    token_loss = torch.maximum(-ratio * advantages, -clipped_ratio * advantages)  # [B, G, T]
    if kl_beta > 0:
        # D_{KL}(P||Q) = \sum_a P(a)\log \frac{P(a)}{Q(a)}, P is the \pi_ref, Q is \pi_cur
        # D = \sum_a P(a)\log P(a) - \sum_a P(a)\log Q(a)
        # D = 1 - \sum_a P(a)\log Q(a)

        assert ref_log_probs is not None
        assert curr_log_probs.shape == ref_log_probs.shape
        masked_ref_log_probs = (ref_log_probs.detach().float().masked_fill(~response_mask, 0.0))
        assert torch.isfinite(masked_ref_log_probs).all(), "nonfinite ref_log_probs"
        # k3 = ref / curr - 1 - log(ref / curr)
        log_ref_over_curr = masked_ref_log_probs - masked_curr_log_probs
        ref_over_curr = torch.exp(log_ref_over_curr)
        kl_penalty = ref_over_curr - 1.0 - log_ref_over_curr
        assert torch.isfinite(kl_penalty).all(), "nonfinite reference KL"
        token_loss = token_loss + kl_beta * kl_penalty
    masked_token_loss = token_loss.masked_fill(~response_mask, 0.0)  # [B, G, T]
    token_counts = response_mask.sum(dim=-1)  # [B, G]
    assert (token_counts > 0).all()
    response_loss = masked_token_loss.sum(dim=-1) / token_counts
    loss = response_loss.mean()
    assert torch.isfinite(loss), "nonfinite GRPO loss"
    return loss

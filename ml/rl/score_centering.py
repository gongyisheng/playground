"""Miles score centering (arxiv.org/abs/2609.20807, Appendix A).

Aligned with miles/backends/training_utils/loss/hub/score_centering at
Miles commit 86304536fedc14a4f24522f2a67e24d4293ff62b.
"""

import math
from typing import Literal

import torch


@torch.no_grad()
def top_k_top_p_distribution(
    sampler_logits: torch.Tensor,
    top_p: float = 0.95,
    top_k: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return token IDs and detached probabilities, both shaped [..., K].

    Apply top-k first, then top-p to the renormalized top-k distribution.
    Keep the first token crossing the nucleus threshold; always keep one.
    Removed entries have probability zero. top_k=0 and top_p=1 disable the
    respective filters. Apply any sampling temperature to the logits first.
    """
    assert sampler_logits.ndim >= 1 and sampler_logits.shape[-1] > 0
    assert sampler_logits.is_floating_point()
    assert 0 < top_p <= 1, "invalid top_p"
    assert isinstance(top_k, int) and top_k >= 0, "invalid top_k"

    dtype = torch.float64 if sampler_logits.dtype == torch.float64 else torch.float32
    logits = sampler_logits.to(dtype)
    # -inf is allowed for tokens already excluded by the sampler.
    assert (torch.isfinite(logits) | torch.isneginf(logits)).all(), (
        "invalid sampler_logits"
    )
    assert torch.isfinite(logits).any(dim=-1).all(), "empty sampler support"
    vocab_size = logits.shape[-1]
    k = min(top_k, vocab_size) if top_k else vocab_size
    head_logits, head_ids = logits.topk(k, dim=-1, sorted=True)
    if top_p < 1:
        cumulative_probs = head_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        head_logits = head_logits.masked_fill(remove, -torch.inf)
    return head_ids, head_logits.softmax(dim=-1)


def _scored_log_probs(
    log_probs: torch.Tensor, coefficient: torch.Tensor
) -> torch.Tensor:
    # Miles masks before multiplying, so zero weights never produce 0 * -inf.
    return log_probs.masked_fill((coefficient == 0) | ~torch.isfinite(log_probs), 0.0)


def score_centering_token_loss(
    train_log_probs: torch.Tensor,
    train_head_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    rollout_head_log_probs: torch.Tensor,
    head_mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    importance_sampling: Literal["none", "tis", "mis"] = "none",
    tis_cap: float = 2.0,
    mis_min: float = 0.5,
    mis_max: float = 5.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Miles's unreduced estimator using recorded rollout log probabilities.

    Sampled-token log probabilities and advantages: [..., T]. Head log
    probabilities and boolean head_mask: [..., T, K]. The result is [..., T].
    K counts stored candidates, not a sampling cutoff. No filtering or
    renormalization is performed here. Rollout values must be recorded during
    generation. The sampled token may be outside the stored head.

    For filtered sampling, the head must cover the recorded support and BOTH
    trainer and rollout log probabilities must be normalized on that support.
    For unfiltered sampling, use full-vocabulary-normalized log probabilities;
    the unrecorded tail is modeled as q_hat = rho * p, as in Miles.

        rho = max(1 - sum(q[H]), eps) / max(1 - sum(p[H]), eps)
        alpha = rho * f(1 / rho)
        residual = q[H] * f(p[H] / q[H]) - alpha * p[H]
        L = -A * (sg(f(p_y / q_y)) * log(p_y) - sum(sg(residual) * log(p[H])))

    f(r) is 1, min(r, tis_cap), or r within [mis_min, mis_max] (zero outside).
    All weights, residuals, rollout probabilities and advantages are detached.
    """
    assert train_log_probs.shape == rollout_log_probs.shape == advantages.shape
    assert train_head_log_probs.shape[:-1] == train_log_probs.shape
    assert train_head_log_probs.shape == rollout_head_log_probs.shape == head_mask.shape
    assert head_mask.dtype == torch.bool
    assert importance_sampling in ("none", "tis", "mis"), "invalid importance_sampling"
    assert math.isfinite(eps) and eps > 0, "invalid eps"
    if importance_sampling == "tis":
        assert math.isfinite(tis_cap) and tis_cap > 0, "invalid tis_cap"
    elif importance_sampling == "mis":
        assert math.isfinite(mis_max) and mis_max > 0, "invalid mis_max"
        assert math.isfinite(mis_min) and 0 < mis_min <= mis_max, "invalid mis_min"

    advantages = advantages.detach()
    active = advantages != 0
    # Match Miles's treatment of padding and NaNs on zero-advantage rows.
    train_head = train_head_log_probs.masked_fill(
        ~active.unsqueeze(-1) & torch.isnan(train_head_log_probs), -torch.inf
    ).masked_fill(~head_mask, 0.0)
    rollout_head = (
        rollout_head_log_probs.detach()
        .masked_fill(
            ~active.unsqueeze(-1) & torch.isnan(rollout_head_log_probs), -torch.inf
        )
        .masked_fill(~head_mask, 0.0)
    )

    with torch.no_grad():
        p = train_head.exp().masked_fill(~head_mask, 0.0)
        q = rollout_head.exp().masked_fill(~head_mask, 0.0)
        rho = (1 - q.sum(-1)).clamp_min(eps) / (1 - p.sum(-1)).clamp_min(eps) # for tail distribution
        log_ratio = train_log_probs - rollout_log_probs.detach()
        log_ratio = log_ratio.masked_fill(~active & torch.isnan(log_ratio), 0.0)
        if importance_sampling == "none":
            weighted_head, alpha = q, rho
            weight = torch.ones_like(log_ratio)
        else:
            assert not torch.isnan(log_ratio).any(), "NaN importance log-ratio"
            if importance_sampling == "tis":
                weighted_head = torch.minimum(p, tis_cap * q)
                alpha = (tis_cap * rho).clamp_max(1.0)
                weight = log_ratio.clamp(max=math.log(tis_cap)).exp()
            else:
                head_ratio = train_head - rollout_head
                keep_head = (head_ratio >= math.log(mis_min)) & (
                    head_ratio <= math.log(mis_max)
                )
                weighted_head = torch.where(keep_head, p, 0.0)
                alpha = ((rho >= 1 / mis_max) & (rho <= 1 / mis_min)).to(p.dtype)
                keep_sample = (log_ratio >= math.log(mis_min)) & (
                    log_ratio <= math.log(mis_max)
                )
                weight = log_ratio.masked_fill(~keep_sample, -torch.inf).exp()
        residual = weighted_head - alpha.unsqueeze(-1) * p

    correction = (residual * _scored_log_probs(train_head, residual)).sum(-1)
    return -advantages * (
        weight * _scored_log_probs(train_log_probs, weight) - correction
    )


# curr_logits/sampler_logits: [B, G, T, V]
# token_ids: [B, G, T]
def score_centering_loss(
    curr_logits: torch.Tensor,
    sampler_logits: torch.Tensor,
    token_ids: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    top_p: float = 0.95,
    top_k: int = 20,
    *,
    importance_sampling: Literal["none", "tis", "mis"] = "none",
    tis_cap: float = 2.0,
    mis_min: float = 0.5,
    mis_max: float = 5.0,
    temperature: float = 1.0,
    reduction: Literal["response_mean", "token_mean"] = "response_mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Dense-logits convenience wrapper for Miles's sampling-support replay.

    curr_logits and sampler_logits: [..., T, V], e.g. [B, G, T, V].
    token_ids and response_mask: [..., T]; the mask must be boolean.
    advantages: [...], one reward/advantage per response. Miles's paper recipe
    uses rewards - rewards.mean(-1, keepdim=True), without std normalization.
    All tensors must be on the same device, with
    logits aligned to the tokens they predict (apply the causal shift first).

    sampler_logits must come from the policy that generated these tokens,
    after temperature/processors but before top-k/top-p, using this filter
    order. For ALREADY filtered logits or recorded support log probabilities
    (excluded tokens = -inf), pass top_p=1 and top_k=0 to avoid filtering twice.
    Reconstructing filters from logits requires the same cutoff/tie semantics
    as the engine; recorded support probabilities are authoritative in Miles.
    temperature scales only curr_logits: sampler_logits are already adjusted.

    Normalize the trainer over the sampler's support, not the full vocabulary
    or a newly selected trainer top-k. Both policies then go through the same
    estimator as Miles. Use score_centering_token_loss for a partial logged
    head with an unfiltered tail instead of using top_k as a logging limit.

    response_mean weights responses equally. token_mean matches Miles's
    --calculate-per-token-loss recipe (mean over all unmasked tokens).
    """
    assert curr_logits.ndim >= 2 and curr_logits.shape[-1] > 0
    assert curr_logits.is_floating_point()
    assert curr_logits.shape == sampler_logits.shape
    assert curr_logits.shape[:-1] == token_ids.shape == response_mask.shape
    assert curr_logits.shape[:-2] == advantages.shape
    assert token_ids.dtype == torch.long
    assert response_mask.dtype == torch.bool
    assert math.isfinite(temperature) and temperature > 0, "invalid temperature"
    assert reduction in ("response_mean", "token_mean"), "invalid reduction"
    token_counts = response_mask.sum(dim=-1)
    assert (token_counts > 0).all(), "each response needs an unmasked token"

    # Mask before softmax/gather so padding may contain NaNs or token ID -100.
    padding = ~response_mask.unsqueeze(-1)
    dtype = torch.float64 if curr_logits.dtype == torch.float64 else torch.float32
    train_logits = curr_logits.to(dtype).masked_fill(padding, 0.0) / temperature
    sample_logits = sampler_logits.detach().masked_fill(padding, 0.0)
    sampled_ids = token_ids.masked_fill(~response_mask, 0)
    assert torch.isfinite(train_logits).all(), "nonfinite curr_logits"
    assert ((sampled_ids >= 0) & (sampled_ids < curr_logits.shape[-1])).all(), (
        "invalid token_ids"
    )
    advantages = advantages.detach().to(dtype)
    assert torch.isfinite(advantages).all(), "nonfinite advantages"

    head_ids, head_probs = top_k_top_p_distribution(sample_logits, top_p, top_k)
    sampled_head_mask = head_ids == sampled_ids.unsqueeze(-1)
    head_support = head_probs > 0
    in_support = (sampled_head_mask & head_support).any(dim=-1)
    assert (in_support | ~response_mask).all(), "sampled token outside sampler support"

    # Reuse the rollout support. Gradients outside it are exactly zero.
    head_logits = train_logits.gather(-1, head_ids)
    normalizer = head_logits.masked_fill(~head_support, -torch.inf).logsumexp(-1)
    head_log_probs = (head_logits - normalizer.unsqueeze(-1)).masked_fill(
        ~head_support, 0.0
    )
    sampled_log_probs = (
        train_logits.gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1) - normalizer
    )
    sampled_q = head_probs.masked_fill(~sampled_head_mask, 0.0).sum(-1)
    token_advantages = (
        advantages.unsqueeze(-1)
        .expand_as(response_mask)
        .masked_fill(~response_mask, 0.0)
    )
    token_loss = score_centering_token_loss(
        train_log_probs=sampled_log_probs,
        train_head_log_probs=head_log_probs,
        rollout_log_probs=sampled_q.masked_fill(~response_mask, 1.0).log(),
        rollout_head_log_probs=head_probs.masked_fill(~head_support, 1.0).log(),
        head_mask=head_support & response_mask.unsqueeze(-1),
        advantages=token_advantages,
        importance_sampling=importance_sampling,
        tis_cap=tis_cap,
        mis_min=mis_min,
        mis_max=mis_max,
        eps=eps,
    )
    token_loss = token_loss.masked_fill(~response_mask, 0.0)

    if reduction == "token_mean":
        loss = token_loss.sum() / token_counts.sum()
    else:
        loss = (token_loss.sum(dim=-1) / token_counts).mean()
    assert torch.isfinite(loss), "nonfinite score-centering loss"
    return loss

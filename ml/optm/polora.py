"""PoLoRA: Polar LoRA -- a spectral-preconditioned optimizer for LoRA factor pairs.

Adapts each pair (A, B) -- PEFT convention: A is (r, d_in), B is (d_out, r), the
adapter contributing (alpha/r) * B @ A -- with a polar step taken in a
diagonal-Kronecker curvature metric:

    M_A = EMA(grad_A)                          # first moment, per factor
    C_B = B^T P B,  C_A = A Q A^T              # r x r latent curvature
    Z_A = C_B^-1/2 M_A Q^-1/2                  # whiten
    D_A = C_B^-1/2 msign(Z_A) Q^-1/2           # polar step, then unwhiten
    rho = lr / (sigma_max(A) + sigma_max(B))
    A  -= rho * D_A / sigma_max(D_A)           # mirrored for B

msign is steepest descent under the spectral norm, so sandwiching it between two
*inverse* square roots (not one inverse and one forward) conjugates it into
steepest descent under the preconditioned norm ||C_B^1/2 dA Q^1/2||. rho then
bounds the merged update: ||dW|| = ||B dA + dB A|| <= ||B|| ||dA|| + ||dB|| ||A||
= lr, however far the factors have drifted from their init -- which is what makes
lr a distance per step in spectral norm rather than a per-coordinate size.

The diagonals are coupled across the two sides: each accumulates its own gradient
whitened by the *other* side's inverse curvature, stripping out that side's
amplification so Q and P estimate curvature in the merged W space. State per pair
is one momentum plus two vectors (Q of length d_in, P of length d_out) -- about
half of Adam. C_A / C_B are rebuilt from the weights each step, never stored.

Every matrix root is eigh-free and runs on the r x r Gram, so a step costs
O(r^3) per iteration plus one O(r^2 d) matmul -- the long dimension is never squared.

Reference: github.com/nikhilgsh/polora; PolarExpress coefficients from Amsel et
al., "The Polar Express" (arXiv:2505.16932); Gram Newton-Schulz from Dao (2026).
"""
from __future__ import annotations

import contextlib

import torch
from torch.optim import Optimizer

__all__ = ["power_iter_top", "polar_express", "ns_inv_sqrt", "collect_lora_pairs", "Polora"]


def _normalize_or_zero(X, dim):
    """Normalizes over dim. norm == 0 implies X == 0 there, so the clamp yields 0."""
    return X / X.norm(dim=dim, keepdim=True).clamp_min(1e-30)


@contextlib.contextmanager
def _no_tf32(active):
    """TF32's 10-bit mantissa stalls the fp32 Gram iterations; disable while they run."""
    if not active:
        yield
        return
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def power_iter_top(M, symmetric=False, v_init=None, n_iters=8):
    """Largest singular value of M, or largest eigenvalue if M is symmetric PSD.

    Batched over leading dims. The result is floored at the largest row/column L2
    norm -- always a lower bound on the top value -- so a degenerate start vector
    can never over-estimate. The error is therefore one-sided: too few iterations
    under-estimate sigma_max, which makes the caller's rho *over*-estimate. Passing
    the previous call's v back as v_init amortizes that away: the weights move by
    O(lr) per step, so last step's top vector is already nearly converged and 8
    warm iterations beat 25 cold ones.

    Args:
        M: matrix batch (..., m, n).
        symmetric: treat M as symmetric PSD and return its top eigenvalue.
        v_init: previous top vector (..., k) to warm-start from. Entries that are
            zero or nonfinite fall back to the cold start.
        n_iters: power-iteration steps.

    Returns:
        (value, v) -- value has shape M.shape[:-2] (0-dim for a single matrix),
        v is the converged top vector, reusable as the next call's v_init.
    """
    Mf = M.float()
    if symmetric:
        Mf = 0.5 * (Mf + Mf.transpose(-2, -1))
    Mt = Mf.transpose(-2, -1)
    # Iterate in the smaller of the two spaces: the Gram there is cheaper.
    left = symmetric or Mf.shape[-2] <= Mf.shape[-1]
    rows = Mf.square().sum(dim=-1).amax(dim=-1)
    # A symmetric M is iterated on the left, so only its row norms bound the value.
    floor = rows.sqrt() if symmetric else torch.maximum(rows, Mf.square().sum(dim=-2).amax(dim=-1)).sqrt()

    # Cold start M @ 1 (row sums) or M^T @ 1 (column sums); the floor covers a
    # start vector that lands in the null space.
    v_cold = Mf.sum(dim=-1) if left else Mf.sum(dim=-2)
    if v_init is None:
        v = v_cold
    else:
        norm = v_init.norm(dim=-1, keepdim=True)
        v = torch.where((~torch.isfinite(norm)) | (norm == 0), v_cold, v_init.float())

    v = _normalize_or_zero(v, dim=-1).unsqueeze(-1)
    for _ in range(n_iters):
        if symmetric:
            v = Mf @ v
        else:
            # (M M^T)^k v converges to the top left singular vector, and mirrored.
            v = Mf @ (Mt @ v) if left else Mt @ (Mf @ v)
        v = _normalize_or_zero(v, dim=-2)

    if symmetric:
        value = (v * (Mf @ v)).sum(dim=(-2, -1))          # Rayleigh quotient v^T M v
    else:
        value = ((Mt if left else Mf) @ v).norm(dim=(-2, -1))
    return torch.maximum(value, floor), v.squeeze(-1)


# PolarExpress quintic Remez fits (a, b, c) for p(x) = a x + b x^3 + c x^5, from
# Amsel et al. (arXiv:2505.16932) at worst-case conditioning l=1e-3. Unlike Muon's
# single fixed triple, the schedule varies per step: aggressive early to lift tiny
# singular values, then settling to the exact fixed point (1.875, -1.25, 0.375).
_POLAR_RAW = (
    (8.31968561540051, -23.85945031896673, 17.53144504181025),
    (4.123266419055485, -2.980709974760902, 0.5520797360741728),
    (3.9656114721772022, -2.941248486368672, 0.5589295013119786),
    (3.3312009004415994, -2.4980080275937966, 0.5111272954169234),
    (2.320007312889811, -1.6862169729967622, 0.42068027340235137),
    (1.8951443404954809, -1.2722050191923813, 0.377227344813122),
    (1.875006772051659, -1.250007524486259, 0.37500075245392533),
    (1.8750008025096776, -1.2500016050034328, 0.375000802493755),
    (1.8749954784656357, -1.249990956953079, 0.374995478487443),
    (1.8749954775662068, -1.2499909551542292, 0.3749954775880226),
)
_SAFETY = 1.01


def _polar_coeffs(nsteps):
    """First nsteps triples of the composed schedule. The safety margin divides
    every step but the last (keeping iterates inside the convergence basin), and
    steps past the table repeat the asymptotic quintic."""
    take = _POLAR_RAW[:nsteps]
    take = take + (_POLAR_RAW[-1],) * (nsteps - len(take))
    s, s3, s5 = _SAFETY, _SAFETY ** 3, _SAFETY ** 5
    return [(a, b, c) if i == nsteps - 1 else (a / s, b / s3, c / s5)
            for i, (a, b, c) in enumerate(take)]


def _gram_iterate(R, Q, coeffs):
    """Drives R -> I by applying p(R) while accumulating Q = prod p(R_t).

    The polar factor and the inverse square root are the *same* iteration and
    differ only in what R starts as: R = X X^T gives Q @ X -> polar(X), while
    R = S/trace(S) gives Q/sqrt(trace) -> S^-1/2.
    """
    for a, b, c in coeffs:
        M = b * R + c * (R @ R)
        M.diagonal(dim1=-2, dim2=-1).add_(a)          # M = a I + b R + c R^2
        Q = M @ Q
        R = M @ R @ M                                 # congruence keeps R symmetric
    return Q


def polar_express(X, nsteps=8):
    """msign(X) = U V^T from X = U S V^T, without an SVD.

    Runs the Gram form: the iteration touches only the r x r matrix X X^T while
    accumulating Q with Q @ X -> polar(X), so the long dimension is never squared.
    X is prescaled by its Frobenius norm (an upper bound on sigma_max) to enter
    the convergence basin; the polar map is scale-invariant, so this is free.
    An all-zero X maps to zero rather than to NaN.

    Args:
        X: matrix batch (..., m, n); tall inputs are transposed internally.
        nsteps: polynomial iterations.

    Returns:
        float32 tensor shaped like X.
    """
    if X.shape[-1] == 0 or X.shape[-2] == 0:
        return X.float()
    X = X.float()
    tall = X.shape[-2] > X.shape[-1]
    if tall:
        X = X.transpose(-2, -1)                       # keep X X^T the smaller Gram
    Xn = _normalize_or_zero(X, dim=(-2, -1))
    r = Xn.shape[-2]
    eye = torch.eye(r, dtype=Xn.dtype, device=Xn.device).expand(*Xn.shape[:-2], r, r)
    with _no_tf32(Xn.is_cuda):
        Q = _gram_iterate(Xn @ Xn.transpose(-2, -1), eye.clone(), _polar_coeffs(nsteps))
        out = Q @ Xn
    return out.transpose(-2, -1) if tall else out


def ns_inv_sqrt(S, nsteps=8, delta=1e-4, floor=1e-12):
    """(S + delta_eff I)^-1/2 for a symmetric-PSD batch, without an eigendecomposition.

    Damping is *relative* -- delta_eff = delta * lambda_max(S) -- which caps the
    damped condition number at 1 + 1/delta independent of scale. floor keeps the
    result finite when S == 0, which is exactly what happens on step 1 of a
    standard LoRA init (B = 0 makes C_B = B^T P B vanish).

    Args:
        S: symmetric PSD batch (..., r, r); symmetrized internally.
        nsteps: Newton-Schulz iterations.
        delta: relative diagonal damping.
        floor: absolute floor on delta_eff.

    Returns:
        float32 tensor shaped like S.
    """
    S = S.float()
    S = 0.5 * (S + S.transpose(-2, -1))
    r = S.shape[-1]
    eye = torch.eye(r, dtype=S.dtype, device=S.device).expand_as(S)
    batch = (*S.shape[:-2], 1, 1)

    lam_max, _ = power_iter_top(S, symmetric=True)
    S_d = S + (delta * lam_max.reshape(batch)).clamp_min(floor) * eye
    # trace >= lambda_max, so dividing by it puts the spectrum inside (0, 1].
    scale = S_d.diagonal(dim1=-2, dim2=-1).sum(-1).reshape(batch)
    with _no_tf32(S.is_cuda):
        Q = _gram_iterate(S_d / scale, eye.clone(), _polar_coeffs(nsteps))
    return Q / scale.sqrt()                           # (scale * R0)^-1/2


def _damped_rsqrt(x, delta):
    """diag(Q^-1/2) from the raw EMA accumulator x: (x / x_max + delta)^-1/2.

    Dividing by x_max cancels the (1 - curvature_beta) warmup factor exactly, so
    the accumulator needs no bias correction the way Adam's v does, and the
    preconditioner is scale-free.
    """
    return (x / x.amax(dim=-1, keepdim=True) + delta).rsqrt()


def collect_lora_pairs(model):
    """Finds trainable LoRA (A, B) weight tensors on a PEFT-style model.

    Handles both PEFT's ModuleDict-of-adapters layout and a plain lora_A / lora_B
    carrying .weight. Frozen pairs are skipped, so a reference adapter sharing the
    model (e.g. for DPO) is not collected.

    Returns:
        List of (A, B) parameter pairs in module-discovery order.
    """
    pairs = []
    for _, mod in model.named_modules():
        if not (hasattr(mod, "lora_A") and hasattr(mod, "lora_B")):
            continue
        lora_A, lora_B = mod.lora_A, mod.lora_B
        if hasattr(lora_A, "keys"):                   # ModuleDict (standard PEFT)
            pairs += [(lora_A[k].weight, lora_B[k].weight) for k in lora_A if k in lora_B]
        elif hasattr(lora_A, "weight") and hasattr(lora_B, "weight"):
            pairs.append((lora_A.weight, lora_B.weight))
    return [(A, B) for A, B in pairs if A.requires_grad and B.requires_grad]


class Polora(Optimizer):
    """PoLoRA optimizer. Only accepts LoRA (A, B) pairs. See the module docstring
    for the update.

    State is allocated lazily on each parameter's device, so construct after the
    model has been moved. Both factors of a pair must carry a gradient -- there is
    no per-parameter skip, since the update couples them.

    Args:
        model: PEFT-wrapped model. Its trainable LoRA pairs are discovered
            automatically; A is (r, d_in), B is (d_out, r).
        lr: learning rate. Each factor update is rescaled to spectral norm
            rho = lr / (sigma_max(A) + sigma_max(B)), so this is a distance per
            step in spectral norm and is not comparable to an AdamW lr. Because a
            fixed-length step cannot settle inside itself, decay it.
        beta1: momentum coefficient for the per-factor first moments.
        epsilon: floor for the preconditioner init, the curvature damping, and the
            spectral-norm divisions.
        delta: relative damping for the C_A / C_B and Q / P inverse square roots.
        curvature_beta: EMA coefficient for the diagonals Q / P.
        ns_steps: PolarExpress iterations for msign.
        higham_iters: Newton-Schulz iterations for the r x r inverse square roots.
        power_iters: power-iteration steps for every sigma_max. Since sigma_max is
            only ever under-estimated, too few overshoot rho and let ||dW|| exceed
            lr. Each call warm-starts from the previous step's top vector, which
            holds the error under ~0.1% at the reference's 8 iterations; without
            that cache 8 would overshoot by up to 16% and 25 would be needed.
    """

    def __init__(
        self,
        model,
        lr=2e-4,
        beta1=0.9, # momentum, beta1
        epsilon=1e-12, # a numerical floor (divide by 0)
        delta=1e-4, # for damped rsqrt
        curvature_beta=0.99, # beta2
        ns_steps=8,
        higham_iters=8,
        power_iters=8,
    ):
        if lr < 0.0 or epsilon < 0.0 or delta < 0.0:
            raise ValueError(f"Expected lr, epsilon, delta >= 0, got {lr}, {epsilon}, {delta}.")
        if not all(0.0 <= b < 1.0 for b in (beta1, curvature_beta)):
            raise ValueError(f"Expected beta1, curvature_beta in [0, 1), got {beta1}, {curvature_beta}.")
        if min(ns_steps, higham_iters, power_iters) < 1:
            raise ValueError(f"Expected >= 1 iteration, got {ns_steps}, {higham_iters}, {power_iters}.")

        # Pairs are flattened to [A, B, A, B, ...]; step() re-pairs by zipping
        # consecutive params, which keeps tensors out of custom param_groups keys
        # (they would otherwise end up in state_dict).
        flat = []
        for A, B in collect_lora_pairs(model):
            if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[1]:
                raise ValueError(f"Expected a LoRA pair A (r, d_in), B (d_out, r) with matching "
                                 f"r, got A {tuple(A.shape)} and B {tuple(B.shape)}.")
            flat += [A, B]
        if not flat:
            raise ValueError("No trainable LoRA (A, B) pairs found on model.")
        super().__init__(flat, dict(lr=lr, beta1=beta1, epsilon=epsilon, delta=delta,
                                      curvature_beta=curvature_beta, ns_steps=ns_steps,
                                      higham_iters=higham_iters, power_iters=power_iters))

    @torch.no_grad()
    def step(self, closure=None):
        """Applies one PoLoRA update to every (A, B) pair.

        Args:
            closure: standard torch.optim closure, called under grad.

        Returns:
            The closure's loss, or None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, b1 = group["lr"], group["beta1"]
            eps, delta = group["epsilon"], group["delta"]
            # EMA coefficient for the diagonal curvature, Adam's b2 in spirit
            cb = group["curvature_beta"]
            n_pi = group["power_iters"]
            ps = group["params"]

            # params were flattened pairwise at construction, so consecutive
            # entries are (A, B) of the same adapter
            for A, B in zip(ps[::2], ps[1::2]):
                if A.grad is None or B.grad is None:
                    raise ValueError("Polora requires gradients on both factors of every pair.")
                G_A, G_B = A.grad.float(), B.grad.float()
                # .float() is a no-op *view* for fp32 params, so A and B must not be
                # written until Aw/Bw are done with -- the param update goes last.
                Aw, Bw = A.detach().float(), B.detach().float()

                sA, sB = self.state[A], self.state[B]
                if not sA:
                    sA["M"] = torch.zeros_like(Aw)
                    sA["diag"] = torch.full((A.shape[1],), eps, device=A.device)   # Q, d_in, equviant to adam's v
                    sB["M"] = torch.zeros_like(Bw)
                    sB["diag"] = torch.full((B.shape[0],), eps, device=B.device)   # P, d_out, equviant to adam's v
                    # warm-start vectors for the four sigma_max calls, filled below
                    sA["v_w"] = sA["v_d"] = sB["v_w"] = sB["v_d"] = None
                # M = b1 * M + (1 - b1) * G, per-factor gradient EMA
                M_A = sA["M"].mul_(b1).add_(G_A, alpha=1 - b1)
                M_B = sB["M"].mul_(b1).add_(G_B, alpha=1 - b1)
                Q, P = sA["diag"], sB["diag"]

                # create trusted curvature estimate, increase q/p somehow
                q_isqrt, p_isqrt = _damped_rsqrt(Q, delta), _damped_rsqrt(P, delta)
                # the damped, max-normalized diagonals that q_isqrt / p_isqrt invert
                q_dmp, p_dmp = q_isqrt.square().reciprocal(), p_isqrt.square().reciprocal()

                # r x r latent curvature, rebuilt from the weights every step
                C_B = Bw.T @ (p_dmp.unsqueeze(-1) * Bw)
                C_A = (Aw * q_dmp) @ Aw.T
                C_B_isqrt = ns_inv_sqrt(C_B, group["higham_iters"], delta, eps)
                C_A_isqrt = ns_inv_sqrt(C_A, group["higham_iters"], delta, eps)

                # Nesterov look-ahead: the direction one momentum step ahead of M
                M_hat_A = M_A.mul(b1).add(G_A, alpha=1 - b1)
                M_hat_B = M_B.mul(b1).add(G_B, alpha=1 - b1)

                # warm-started from last step's top vectors, so 8 iterations land
                # within ~0.1% instead of the ~7% a cold start leaves
                sig_A, sA["v_w"] = power_iter_top(Aw, v_init=sA["v_w"], n_iters=n_pi)
                sig_B, sB["v_w"] = power_iter_top(Bw, v_init=sB["v_w"], n_iters=n_pi)
                rho = lr / (sig_A + sig_B).clamp_min(eps)

                # whiten -> polar -> unwhiten, both sides by an *inverse* root, which
                # conjugates msign into the preconditioned metric
                Z_A = (C_B_isqrt @ M_hat_A) * q_isqrt
                Z_B = (M_hat_B @ C_A_isqrt) * p_isqrt.unsqueeze(-1)
                D_A = (C_B_isqrt @ polar_express(Z_A, group["ns_steps"])) * q_isqrt
                D_B = (polar_express(Z_B, group["ns_steps"]) @ C_A_isqrt) * p_isqrt.unsqueeze(-1)
                # unwhitening destroys orthogonality, so renormalize to spectral norm rho
                sig_DA, sA["v_d"] = power_iter_top(D_A, v_init=sA["v_d"], n_iters=n_pi)
                sig_DB, sB["v_d"] = power_iter_top(D_B, v_init=sB["v_d"], n_iters=n_pi)
                dA = D_A * -(rho / sig_DA.clamp_min(eps))
                dB = D_B * -(rho / sig_DB.clamp_min(eps))

                # Each diagonal accumulates its own gradient whitened by the *other*
                # side's inverse curvature: G_A = B^T G, so C_B^-1 removes B's
                # amplification and Q ends up measuring curvature in the merged W space.
                C_B_inv, C_A_inv = C_B_isqrt @ C_B_isqrt, C_A_isqrt @ C_A_isqrt
                scale = (1 - cb) / A.shape[0]
                Q.mul_(cb).add_((G_A * (C_B_inv @ G_A)).sum(dim=0), alpha=scale)
                P.mul_(cb).add_((G_B * (G_B @ C_A_inv)).sum(dim=1), alpha=scale)

                A.add_(dA.to(A.dtype))
                B.add_(dB.to(B.dtype))
        return loss


if __name__ == "__main__":
    import math

    # Adapt a frozen MLP whose teacher differs by an exactly rank-r delta per layer,
    # so the reachable optimum is 0. lr is decayed to zero because polora's step has
    # a fixed spectral length -- it discards gradient magnitude, so at constant lr it
    # limit-cycles at an amplitude set by lr instead of settling.
    class LoRALinear(torch.nn.Module):
        """Frozen base weight plus a rank-r adapter, named so collect_lora_pairs finds it."""

        def __init__(self, d_in, d_out, r):
            super().__init__()
            self.base = torch.nn.Linear(d_in, d_out, bias=False)
            self.base.weight.requires_grad_(False)
            self.lora_A = torch.nn.Linear(d_in, r, bias=False)       # (r, d_in)
            self.lora_B = torch.nn.Linear(r, d_out, bias=False)      # (d_out, r)
            torch.nn.init.zeros_(self.lora_B.weight)

        def forward(self, x):
            return self.base(x) + self.lora_B(self.lora_A(x))

    def train(make_opt, lr, steps, decay=True):
        torch.manual_seed(1)
        d, r = 32, 4
        net = torch.nn.Sequential(LoRALinear(d, d, r), torch.nn.Tanh(), LoRALinear(d, d, r))
        teacher = torch.nn.Sequential(torch.nn.Linear(d, d, bias=False), torch.nn.Tanh(),
                                      torch.nn.Linear(d, d, bias=False))
        with torch.no_grad():
            for t, s in ((teacher[0], net[0]), (teacher[2], net[2])):
                t.weight.copy_(s.base.weight + (torch.randn(d, r) @ torch.randn(r, d)) / d)
        X = torch.randn(256, d)
        with torch.no_grad():
            Y = teacher(X)

        opt, first = make_opt(net), None
        for i in range(steps):
            if decay:
                opt.param_groups[0]["lr"] = lr * 0.5 * (1 + math.cos(math.pi * i / steps))
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(X), Y)
            loss.backward()
            opt.step()
            first = loss.item() if first is None else first
        return first, loss.item()

    def polora(lr):
        return lambda net: Polora(model=net, lr=lr)

    def adamw(lr):
        return lambda net: torch.optim.AdamW(
            [p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)

    # each optimizer gets its own lr: polora's is a spectral distance per step,
    # adamw's a per-coordinate size, so the two scales are not comparable
    for steps in (400, 1500):
        first, polora_loss = train(polora(0.1), lr=0.1, steps=steps)
        _, adamw_loss = train(adamw(0.01), lr=0.01, steps=steps)
        print(f"[train]  {steps:>4} steps, lr -> 0:  initial {first:.3e}"
              f"   polora {polora_loss:.3e} ({first / polora_loss:.3g}x)"
              f"   adamw {adamw_loss:.3e} ({first / adamw_loss:.3g}x)")
        assert polora_loss < first / 1000, polora_loss

    # at constant lr the step length never shrinks, so the loss floors instead of
    # converging: the plateau tracks lr, not the gradient.
    for lr in (0.1, 0.03):
        first, floor = train(polora(lr), lr=lr, steps=1500, decay=False)
        print(f"[floor]  constant lr={lr}: initial {first:.3e} -> plateau {floor:.3e}")

    print("all checks passed")

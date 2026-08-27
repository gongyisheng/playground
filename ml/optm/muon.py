"""Muon: MomentUm Orthogonalized by Newton-schulz.

Replaces the momentum matrix M with msign(M) = U V^T (from M = U S V^T),
computed by Newton-Schulz iteration instead of an SVD. 2D+ params only --
pair it with AdamW for biases, norms, embeddings and the output head.
"""

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate msign(G) = U V^T via a quintic Newton-Schulz iteration.

    The coefficients are tuned to maximize the slope at zero rather than to
    converge exactly, so singular values end up roughly in [0.7, 1.3]. That is
    enough for the optimizer and far cheaper than an exact orthogonalization.
    """
    assert G.ndim == 2 # (M, N)
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1) # always run on smaller axis
    if transposed:
        X = X.mT  # keep X X^T the smaller Gram matrix
    X = X / (X.norm() + 1e-7)  # bring singular values into the convergence basin
    for _ in range(steps):
        A = X @ X.mT # (M, M)
        B = b * A + c * A @ A # (M, M)
        X = a * X + B @ X # (M, N)
        # X = aX + b(XX^T)X + c(XX^T)^2X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Muon optimizer. Only accepts params with ndim >= 2.

    Args:
        params: iterable of params or param groups.
        lr: learning rate. Muon updates have ~unit spectral norm, so this is
            not comparable to an AdamW lr.
        momentum: EMA coefficient for the gradient buffer.
        nesterov: use the Nesterov-style lookahead direction.
        ns_steps: Newton-Schulz iterations per step.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"invalid momentum: {momentum}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim < 2:
                    raise ValueError(
                        f"Muon only supports params with ndim >= 2, got {tuple(p.shape)}. "
                        "Route biases/norms/embeddings to AdamW instead."
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            # same as adamw b1, for calculating EMA
            momentum = group["momentum"]
            # nesterov: a bool value (0/1), enable nesterov or not
            # take the direction one momentum-step ahead of buf,
            #   dir = (1 - momentum) * grad + momentum * buf
            #       == (1 - momentum^2) * grad + momentum^2 * buf_prev
            # i.e. a shorter horizon than buf, so the direction lags less
            nesterov = group["nesterov"]
            # Newton-Schulz iterations; more steps -> singular values closer to 1
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # conv kernels and higher-rank params collapse to (out, fan_in)
                # keep the dim = 0
                grad = p.grad.reshape(p.size(0), -1)

                state = self.state[p]
                # gradient EMA: buf = momentum * buf + (1 - momentum) * grad
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]

                # EMA: (1 - momentum) * grad + momentum * buf
                buf.lerp_(grad, 1 - momentum)

                # orthogonalize after the momentum mix -- msign is nonlinear, so
                # the buffer must stay a raw gradient EMA

                # next buf on same grad: (1 - momentum) * grad + momentum * buf
                direction = grad.lerp(buf, momentum) if nesterov else buf
                # smooth its singular values
                update = zeropower_via_newtonschulz5(direction, ns_steps)

                # msign leaves min(m, n) unit singular values, so per-entry RMS
                # is sqrt(min(m, n) / (m * n)) -- shrinks as the matrix gets more
                # rectangular. Rescale so the step size means the same thing
                # across layer shapes: w -= lr * max(1, m/n)^0.5 * msign(dir)
                m, n = update.shape
                # energy is min(m, n)
                # m / n is how far the variance is off
                # when m>n, double m: direction stay n, but entry doubled, rms dropped by 2**0.5
                # goal: keep update RMS / weight RMS steady, no explosion
                scale = max(1.0, m / n) ** 0.5 # fanout/fanin
                p.add_(update.reshape(p.shape).type_as(p), alpha=-lr * scale)

        return loss


if __name__ == "__main__":
    torch.manual_seed(0)

    def msign_sv(cond_lo, cond_hi, steps=5):
        U, _, V = torch.linalg.svd(torch.randn(64, 32), full_matrices=False)
        G = U @ torch.diag(torch.logspace(cond_lo, cond_hi, 32)) @ V
        return torch.linalg.svdvals(zeropower_via_newtonschulz5(G, steps).float())

    sv = msign_sv(-1, 1)
    print(f"[msign] cond 1e2 -> sv range {sv.min():.3f} .. {sv.max():.3f}")
    assert sv.min() > 0.6 and sv.max() < 1.4, sv

    sv = msign_sv(-3, 1)
    print(f"[msign] cond 1e4 -> sv range {sv.min():.3f} .. {sv.max():.3f} (spread {sv.max() / sv.min():.1f}x)")
    assert sv.max() / sv.min() < 100.0, sv  # 1e4:1 spread compressed by >100x

    # 2. smoke train: Muon vs SGD-momentum on a synthetic regression.
    def train(make_opt, steps=200):
        torch.manual_seed(1)
        net = torch.nn.Sequential(
            torch.nn.Linear(32, 64, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 8, bias=False),
        )
        X = torch.randn(512, 32)
        Y = torch.randn(512, 8)
        opt = make_opt(net.parameters())
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(X), Y)
            loss.backward()
            opt.step()
        return loss.item()

    muon_loss = train(lambda ps: Muon(ps, lr=0.02))
    sgd_loss = train(lambda ps: torch.optim.SGD(ps, lr=0.02, momentum=0.95))
    print(f"[train] muon final loss: {muon_loss:.4f}")
    print(f"[train] sgd  final loss: {sgd_loss:.4f}")
    assert muon_loss < 1.0, muon_loss

    # 3. ndim < 2 is rejected.
    try:
        Muon([torch.zeros(4, requires_grad=True)])
    except ValueError as e:
        print(f"[guard] rejected 1D param: {e}")
    else:
        raise AssertionError("expected ValueError for 1D param")

    print("all checks passed")

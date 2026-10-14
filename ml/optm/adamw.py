"""AdamW — Adam with decoupled weight decay, built on torch.optim.Optimizer.

Per parameter, with step count t:

    m_t = b1 m_{t-1} + (1 - b1) g            # first moment (EMA of g)
    v_t = b2 v_{t-1} + (1 - b2) g^2          # second moment (EMA of g^2)
    p  -= lr * wd * p                        # decoupled decay
    p  -= lr * (m_t / (1 - b1^t)) / (sqrt(v_t / (1 - b2^t)) + eps)

The moments start at zero, so early EMAs are biased toward zero; dividing by
1 - beta^t rescales them to unbiased estimates. Adam instead folds decay into g,
where v_t divides it out again and the effective decay ends up coupled to the
gradient scale — decoupling is the whole difference.

Reference: Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (ICLR 2019).
"""
from __future__ import annotations

import torch
from torch.optim import Optimizer

__all__ = ["AdamW"]


class AdamW(Optimizer):
    """AdamW. See the module docstring for the update.

    State is allocated lazily on each parameter's device, so construct after the
    model has been moved. Sparse gradients are not supported. Params with
    grad=None are skipped, which lets a frozen submodule share a group.

    Args:
        params: iterable of parameters or param-group dicts.
        lr: learning rate.
        betas: (beta1, beta2) EMA coefficients for the two moments.
        eps: denominator floor, added after the sqrt.
        weight_decay: decoupled decay coefficient; the shrink is lr * weight_decay.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0.0 or eps < 0.0 or weight_decay < 0.0:
            raise ValueError(f"Expected lr, eps, weight_decay >= 0, got {lr}, {eps}, {weight_decay}.")
        if not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError(f"Expected betas in [0, 1), got {betas}.")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        """Applies one AdamW update to every parameter with a gradient.

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
            lr, (b1, b2) = group["lr"], group["betas"]
            eps, wd = group["eps"], group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients.")

                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]

                # m = b1 * m + (1 - b1) * g, momuntum, ema of g
                # v = b2 * v + (1 - b2) * g * g, variance, ema of g^2
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)

                if wd != 0:
                    p.mul_(1 - lr * wd)
                # Fold bias correction into the step size and the denominator
                # rather than materializing m_hat / v_hat.
                denom = v.sqrt().div_((1 - b2**t) ** 0.5).add_(eps)
                p.addcdiv_(m, denom, value=-lr / (1 - b1**t))
        return loss


if __name__ == "__main__":
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 4))
    reference = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 4))
    reference.load_state_dict(model.state_dict())

    ours = AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
    theirs = torch.optim.AdamW(reference.parameters(), lr=3e-3, weight_decay=0.1)
    x, y = torch.randn(64, 16), torch.randn(64, 4)

    for step in range(50):
        for net, opt in ((model, ours), (reference, theirs)):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(x), y)
            loss.backward()
            opt.step()
        if step % 10 == 0:
            print(f"step {step:3d}  loss {loss.item():.6f}")

    gap = max((a - b).abs().max().item() for a, b in zip(model.parameters(), reference.parameters()))
    print(f"max |ours - torch.optim.AdamW| after 50 steps: {gap:.3e}")
    assert gap < 1e-6, "diverged from the reference implementation"

    resumed = AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
    resumed.load_state_dict(ours.state_dict())
    assert resumed.state[next(model.parameters())]["step"] == 50
    print("state_dict round-trip ok")

"""
OracleClippedSGD — torch.optim.Optimizer implementation.

Identical structure to AlphaRobustSGD but uses the true p and true sigma
as oracle inputs instead of estimating them. This is the theoretical ceiling
that AlphaRobustSGD aims to match.

Used in Phase 2+ for direct Oracle vs AlphaRobust comparison on real models.
"""

import torch
from torch.optim import Optimizer
from typing import List
import numpy as np


class OracleClippedSGD(Optimizer):
    """
    Args:
        params:      Iterable of parameters or param groups.
        lr (float):  Base learning rate.
        p (float):   True moment parameter (oracle). Must be in (1, 2].
        sigma (float): True noise scale (oracle).
        schedule:    'fixed' or 'growing'.
        tau_scale:   C in tau = C * sigma  (fixed schedule).
        tau_min/max: Safety clamps on tau.
        momentum:    SGD momentum.
        weight_decay: L2 regularization.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        p: float = 1.5,
        sigma: float = 1.0,
        schedule: str = 'fixed',
        tau_scale: float = 5.0,
        tau_min: float = 1e-6,
        tau_max: float = 1e6,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        assert 1.0 < p <= 2.0, f"p must be in (1, 2], got {p}"
        assert schedule in ('fixed', 'growing')
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.p           = p
        self.sigma       = sigma
        self.schedule    = schedule
        self.tau_scale   = tau_scale
        self.tau_min     = tau_min
        self.tau_max     = tau_max
        self._step_count = 0

        self._log_tau      : List[float] = []
        self._log_grad_norm: List[float] = []

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        t = self._step_count

        grad_norm = self._global_grad_norm()
        self._log_grad_norm.append(grad_norm)

        tau, lr_eff = self._compute_tau_and_lr(t)
        self._log_tau.append(tau)

        clip_coef = min(1.0, tau / (grad_norm + 1e-8))

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                if group['weight_decay'] != 0:
                    d_p = d_p.add(p, alpha=group['weight_decay'])

                d_p = d_p * clip_coef

                if group['momentum'] != 0:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = d_p.clone()
                    else:
                        state['momentum_buffer'].mul_(group['momentum']).add_(d_p)
                        d_p = state['momentum_buffer']

                p.add_(d_p, alpha=-lr_eff)

        return loss

    @property
    def tau_history(self) -> np.ndarray:
        return np.array(self._log_tau)

    @property
    def grad_norm_history(self) -> np.ndarray:
        return np.array(self._log_grad_norm)

    @property
    def step_count(self) -> int:
        return self._step_count

    def _global_grad_norm(self) -> float:
        total_sq = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    total_sq += p.grad.data.norm(2).item() ** 2
        return float(total_sq ** 0.5)

    def _compute_tau_and_lr(self, t: int):
        lr_base = self.param_groups[0]['lr']

        if self.schedule == 'growing':
            lr_eff = lr_base / (t ** 0.5)
            tau    = self.sigma * (t ** (1.0 / self.p))
        else:
            lr_eff = lr_base
            tau    = self.tau_scale * self.sigma

        tau = float(np.clip(tau, self.tau_min, self.tau_max))
        return tau, lr_eff

    def __repr__(self):
        return (f"OracleClippedSGD(lr={self.param_groups[0]['lr']}, "
                f"p={self.p}, sigma={self.sigma}, schedule={self.schedule})")

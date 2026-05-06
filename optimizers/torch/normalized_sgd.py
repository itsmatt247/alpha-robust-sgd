"""
Normalized SGD (NSGD) — parameter-free convergence under heavy-tailed noise.

Based on:
  Hübler et al. "From Gradient Clipping to Normalization for Heavy Tailed SGD."
  AISTATS 2025. https://arxiv.org/abs/2410.13849

  Liu et al. "Nonconvex Stochastic Optimization under Heavy-Tailed Noises:
  Optimal Convergence without Gradient Clipping." ICLR 2025.

Mechanism:
  Strips gradient magnitude entirely, updates using only the unit direction:
    x_{t+1} = x_t - lr_t * g_t / ||g_t||

  This achieves O(eps^{-2p/(p-1)}) sample complexity without any oracle
  knowledge of p. The price: all gradient magnitude information is discarded,
  including valid signal from steep loss landscape regions.

Role in paper:
  NSGD represents the "competing school" for solving the oracle problem.
  AlphaRobust's thesis: preserving gradient magnitude (up to a principled
  EVT-based threshold) is better than discarding it entirely.
"""

import torch
from torch.optim import Optimizer
from typing import List
import numpy as np


class NormalizedSGD(Optimizer):
    """
    Args:
        params:       Iterable of parameters or param groups.
        lr (float):   Base learning rate.
        schedule:     'fixed' (constant lr) or 'decaying' (lr_t = lr/sqrt(t)).
        momentum:     SGD momentum (applied to normalized gradient).
        weight_decay: L2 regularization.
        eps:          Numerical stability term for norm division.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        schedule: str = 'fixed',
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ):
        assert schedule in ('fixed', 'decaying')
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.schedule    = schedule
        self.eps         = eps
        self._step_count = 0

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

        lr_base = self.param_groups[0]['lr']
        lr_eff  = lr_base / (t ** 0.5) if self.schedule == 'decaying' else lr_base

        # Normalize: scale all gradients so the global norm becomes 1
        norm_coef = 1.0 / (grad_norm + self.eps)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad * norm_coef  # normalized gradient

                if group['weight_decay'] != 0:
                    d_p = d_p.add(p, alpha=group['weight_decay'])

                if group['momentum'] != 0:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = d_p.clone()
                    else:
                        state['momentum_buffer'].mul_(group['momentum']).add_(d_p)
                        d_p = state['momentum_buffer']

                p.add_(d_p, alpha=-lr_eff)

        return loss

    def _global_grad_norm(self) -> float:
        total_sq = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    total_sq += p.grad.data.norm(2).item() ** 2
        return float(total_sq ** 0.5)

    @property
    def grad_norm_history(self) -> np.ndarray:
        return np.array(self._log_grad_norm)

    @property
    def step_count(self) -> int:
        return self._step_count

    def __repr__(self):
        return (f"NormalizedSGD(lr={self.param_groups[0]['lr']}, "
                f"schedule={self.schedule})")

"""
ZClip — Adaptive gradient spike mitigation for LLM pre-training.
Replicates Kumar et al. (2025): https://arxiv.org/abs/2504.02507

Mechanism:
  Maintains an EMA of the mean and variance of gradient norms.
  At each step, computes a Z-score for the current norm.
  If Z-score exceeds threshold z_thresh, clips the gradient.

Statistical note (per Gemini novelty review):
  ZClip assumes gradient norms are locally Gaussian — a misspecification
  for heavy-tailed Transformer gradients. Included as a baseline to show
  that statistically correct EVT-based clipping (AlphaRobust) outperforms
  a Gaussian anomaly detector applied to a power-law phenomenon.
"""

import torch
from torch.optim import Optimizer
from typing import List
import numpy as np


class ZClip(Optimizer):
    """
    Args:
        params:       Iterable of parameters or param groups.
        lr (float):   Learning rate.
        z_thresh:     Z-score threshold above which a gradient is clipped.
        ema_alpha:    EMA decay factor for mean/variance tracking (0 < alpha < 1).
        momentum:     SGD momentum.
        weight_decay: L2 regularization.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        z_thresh: float = 2.5,
        ema_alpha: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.z_thresh   = z_thresh
        self._ema_alpha = ema_alpha
        self._step_count = 0

        # EMA state
        self._ema_mean: float = None
        self._ema_var:  float = None

        self._log_grad_norm: List[float] = []
        self._log_tau:       List[float] = []

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        grad_norm = self._global_grad_norm()
        self._log_grad_norm.append(grad_norm)

        # Update EMA of mean and variance
        if self._ema_mean is None:
            self._ema_mean = grad_norm
            self._ema_var  = 0.0
        else:
            delta = grad_norm - self._ema_mean
            self._ema_mean += self._ema_alpha * delta
            self._ema_var   = (1 - self._ema_alpha) * (
                self._ema_var + self._ema_alpha * delta ** 2
            )

        # Compute Z-score and clipping threshold
        std = float(self._ema_var ** 0.5) + 1e-8
        z   = (grad_norm - self._ema_mean) / std
        if z > self.z_thresh:
            tau = self._ema_mean + self.z_thresh * std
        else:
            tau = float('inf')   # no clip
        self._log_tau.append(min(tau, grad_norm))

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

                p.add_(d_p, alpha=-group['lr'])

        return loss

    @property
    def grad_norm_history(self) -> np.ndarray:
        return np.array(self._log_grad_norm)

    @property
    def tau_history(self) -> np.ndarray:
        return np.array(self._log_tau)

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

    def __repr__(self):
        return (f"ZClip(lr={self.param_groups[0]['lr']}, "
                f"z_thresh={self.z_thresh}, ema_alpha={self.ema_alpha})")

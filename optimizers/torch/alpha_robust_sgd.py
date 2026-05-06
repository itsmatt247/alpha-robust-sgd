"""
AlphaRobustSGD — torch.optim.Optimizer implementation for Phase 2+.

Applies global gradient norm clipping with a threshold derived from a
live Hill estimator on a rolling window of observed gradient norms.
This replaces oracle knowledge of the tail index p with a live estimate p̂.

Design decisions vs the numpy Phase 1 version:
  - Global norm clipping (not per-parameter): standard for Transformers,
    matches torch.nn.utils.clip_grad_norm_ convention.
  - One Hill estimator shared across all param groups (global norm is a
    scalar summary of the full gradient vector).
  - step_count managed internally (torch step() receives no t argument).
  - Supports momentum and weight_decay to be competitive with SGD baselines.
  - schedule='fixed'  : tau = (p̂/2) * tau_scale * sigma_hat (tail_factor; NumPy Phase-1 uses tau_scale*sigma_hat without this factor)
  - schedule='growing': tau_t = sigma_hat * t^{1/p̂}, lr_t = lr_0/sqrt(t)
"""

import torch
from torch.optim import Optimizer
from collections import deque
from typing import Optional, List
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.hill_estimator import HillEstimator


class AlphaRobustSGD(Optimizer):
    """
    Args:
        params:       Iterable of parameters or param groups.
        lr (float):   Base learning rate (decayed as lr/sqrt(t) if schedule='growing').
        window_size:  Rolling window W for the Hill estimator.
        k:            Upper order statistics. Default W // 4.
        epsilon:      p̂ = alpha_hat - epsilon (safety margin below tail index).
        p_min/p_max:  Clamps on p̂.
        schedule:     'fixed' or 'growing'.
        tau_scale:    Multiplier C; fixed schedule uses tau = (p̂/2)*C*sigma_hat (see tail_factor in code).
        tau_0:        Threshold used during burn-in (first W steps).
        tau_min/max:  Absolute safety clamps on tau.
        momentum:     SGD momentum coefficient.
        weight_decay: L2 regularization coefficient.
        deferred_hill: If True, Hill estimator is NOT updated on every step().
                       Instead, call flush_epoch_update() once per PPO epoch
                       (after all minibatches in that epoch complete). Feeds the
                       median of that epoch's minibatch norms to the Hill estimator.
                       Gives 10 updates per rollout (one per epoch) instead of 1,
                       while respecting that within-epoch minibatches are correlated.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        window_size: int = 100,
        k: Optional[int] = None,
        epsilon: float = 0.05,
        p_min: float = 1.01,
        p_max: float = 1.99,
        schedule: str = 'fixed',
        tau_scale: float = 5.0,
        tau_0: float = 10.0,
        tau_min: float = 1e-6,
        tau_max: float = 1e6,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        deferred_hill: bool = False,
    ):
        assert schedule in ('fixed', 'growing'), f"schedule must be 'fixed' or 'growing'"
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Shared across all param groups — tracks global gradient norm
        self._hill        = HillEstimator(window_size=window_size, k=k, epsilon=epsilon)
        self._norm_window = deque(maxlen=window_size)
        self._step_count  = 0

        self.schedule      = schedule
        self.tau_scale     = tau_scale
        self.tau_0         = tau_0
        self.tau_min       = tau_min
        self.tau_max       = tau_max
        self.p_min         = p_min
        self.p_max         = p_max
        self.deferred_hill = deferred_hill

        # For deferred mode: accumulate norms within an epoch
        self._rollout_norms: List[float] = []

        # Diagnostic logs (cleared on reset())
        self._log_p_hat     : List[float] = []
        self._log_alpha_hat : List[float] = []
        self._log_tau       : List[float] = []
        self._log_sigma_hat : List[float] = []
        self._log_grad_norm : List[float] = []

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        t = self._step_count

        # 1. Global gradient norm (pre-clip)
        grad_norm = self._global_grad_norm()
        self._log_grad_norm.append(grad_norm)

        if self.deferred_hill:
            # Accumulate norm; Hill update deferred to flush_hill_update()
            self._rollout_norms.append(grad_norm)
        else:
            # Standard mode: update Hill estimator every step
            self._norm_window.append(grad_norm)
            self._hill.update(grad_norm)

        # 2. Compute threshold and effective LR
        tau, lr_eff = self._compute_tau_and_lr(t)
        self._log_tau.append(tau)

        # 3. Clipping coefficient  min(1, tau / ||g||)
        clip_coef = min(1.0, tau / (grad_norm + 1e-8))

        # 4. Parameter update
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                if group['weight_decay'] != 0:
                    d_p = d_p.add(p, alpha=group['weight_decay'])

                # Apply global clip
                d_p = d_p * clip_coef

                # Momentum
                if group['momentum'] != 0:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = d_p.clone()
                    else:
                        state['momentum_buffer'].mul_(group['momentum']).add_(d_p)
                        d_p = state['momentum_buffer']

                p.add_(d_p, alpha=-lr_eff)

        return loss

    def flush_epoch_update(self):
        """
        Feed the Hill estimator with a single representative gradient norm
        from the current epoch's minibatch norms, then clear the accumulator.

        Call this once per PPO epoch (after all minibatches in that epoch are done).
        This gives 10 Hill updates per rollout (one per epoch) instead of 1,
        while still respecting that minibatches within an epoch share the same
        experience data (correlated). Inter-epoch norms are sufficiently independent
        because the policy has shifted after each gradient epoch.

        Uses the median of the epoch's minibatch norms — less biased than max,
        more representative of the typical gradient scale in this epoch.

        No-op if no norms have been accumulated or if deferred_hill is False.
        """
        if not self.deferred_hill or not self._rollout_norms:
            return

        median_norm = float(np.median(self._rollout_norms))
        self._hill.update(median_norm)
        self._norm_window.append(median_norm)

        self._rollout_norms.clear()

    def flush_hill_update(self):
        """
        Backwards-compatible alias: flushes whatever norms are accumulated.
        Prefer flush_epoch_update() called once per epoch instead.
        """
        self.flush_epoch_update()

    # ------------------------------------------------------------------
    # Logging accessors
    # ------------------------------------------------------------------

    @property
    def p_hat_history(self) -> np.ndarray:
        return np.array(self._log_p_hat)

    @property
    def alpha_hat_history(self) -> np.ndarray:
        return np.array(self._log_alpha_hat)

    @property
    def tau_history(self) -> np.ndarray:
        return np.array(self._log_tau)

    @property
    def sigma_hat_history(self) -> np.ndarray:
        return np.array(self._log_sigma_hat)

    @property
    def grad_norm_history(self) -> np.ndarray:
        return np.array(self._log_grad_norm)

    @property
    def step_count(self) -> int:
        return self._step_count

    def reset_logs(self):
        """Clear diagnostic logs without resetting optimizer state."""
        self._log_p_hat.clear()
        self._log_alpha_hat.clear()
        self._log_tau.clear()
        self._log_sigma_hat.clear()
        self._log_grad_norm.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _global_grad_norm(self) -> float:
        """Compute L2 norm of all gradients across all param groups."""
        total_sq = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    total_sq += p.grad.data.norm(2).item() ** 2
        return float(total_sq ** 0.5)

    def _compute_tau_and_lr(self, t: int):
        """Returns (tau, effective_lr) for step t."""
        lr_base = self.param_groups[0]['lr']

        if self.schedule == 'growing':
            lr_eff = lr_base / (t ** 0.5)
        else:
            lr_eff = lr_base

        p_hat_val  = self._hill.p_hat(p_min=self.p_min, p_max=self.p_max)
        alpha_hat  = self._hill.alpha_hat

        if p_hat_val is None:
            # Burn-in
            self._log_alpha_hat.append(float('nan'))
            self._log_p_hat.append(float('nan'))
            self._log_sigma_hat.append(float('nan'))
            return self.tau_0, lr_eff

        sigma_hat = float(np.median(list(self._norm_window)))

        if self.schedule == 'growing':
            tau = sigma_hat * (t ** (1.0 / p_hat_val))
        else:
            # Scale tau_scale by tail heaviness: heavier tails (lower p̂) → tighter clip.
            # When p̂ ≈ 2 (light tails): factor ≈ 1.0 (permissive, near original behavior).
            # When p̂ ≈ 1 (heavy tails): factor ≈ 0.5 (clip more aggressively).
            # This uses: factor = p̂ / 2, so tau = (p̂/2) * tau_scale * sigma_hat.
            tail_factor = p_hat_val / 2.0
            tau = tail_factor * self.tau_scale * sigma_hat

        tau = float(np.clip(tau, self.tau_min, self.tau_max))

        self._log_alpha_hat.append(alpha_hat)
        self._log_p_hat.append(p_hat_val)
        self._log_sigma_hat.append(sigma_hat)

        return tau, lr_eff

    def __repr__(self):
        return (f"AlphaRobustSGD(lr={self.param_groups[0]['lr']}, "
                f"schedule={self.schedule}, W={self._hill.W}, k={self._hill.k})")

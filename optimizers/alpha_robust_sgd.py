"""
alpha-Robust SGD: scale-adaptive gradient clipping via live Hill estimation.

Core idea: replace oracle knowledge of p with a live Hill estimator on a
rolling window of gradient norms. Everything else mirrors OracleClippedSGD.

Two schedule modes (must match the Oracle you're comparing against):

  schedule='growing':
      tau_t = sigma_hat * t^{1/p_hat}    (sigma_hat = rolling median)
      lr_t  = lr_0 / sqrt(t)

  schedule='fixed':
      tau   = tau_scale * sigma_hat      (constant multiple of estimated scale)
      lr_t  = lr_0

Burn-in (first W steps, Hill estimator not yet ready):
    Uses tau_0 as a fixed safe threshold regardless of schedule.

Phase 2 note: will be ported to torch.optim.Optimizer for neural net training.
"""

import numpy as np
from collections import deque
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.hill_estimator import HillEstimator
from optimizers.baselines import clip_gradient


class AlphaRobustSGD:
    """
    Args:
        lr_0 (float): Base learning rate (decayed if schedule='growing').
        window_size (int): Rolling window size W.
        k (int | None): Upper order statistics for Hill. Default W // 4.
        epsilon (float): p_hat = alpha_hat - epsilon.
        p_min, p_max: Clamps on p_hat.
        schedule (str): 'growing' or 'fixed'.
        tau_scale (float): C in tau = C * sigma_hat (fixed schedule only).
        tau_0 (float): Threshold used during burn-in.
        tau_min, tau_max: Absolute safety clamps on tau.
    """

    def __init__(
        self,
        lr_0: float = 0.01,
        window_size: int = 100,
        k: Optional[int] = None,
        epsilon: float = 0.05,
        p_min: float = 1.01,
        p_max: float = 1.99,
        schedule: str = 'fixed',
        tau_scale: float = 5.0,
        tau_0: float = 10.0,
        tau_min: float = 1e-3,
        tau_max: float = 1e6,
    ):
        assert schedule in ('growing', 'fixed')
        self.lr_0      = lr_0
        self.schedule  = schedule
        self.tau_scale = tau_scale
        self.tau_0     = tau_0
        self.tau_min   = tau_min
        self.tau_max   = tau_max
        self.p_min     = p_min
        self.p_max     = p_max

        self._hill        = HillEstimator(window_size=window_size, k=k, epsilon=epsilon)
        self._norm_window = deque(maxlen=window_size)

        # Logs
        self._log_p_hat     = []
        self._log_alpha_hat = []
        self._log_tau       = []
        self._log_sigma_hat = []
        self._log_grad_norm = []
        self._log_lr        = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, x: np.ndarray, grad: np.ndarray, t: int) -> np.ndarray:
        grad_norm = float(np.linalg.norm(grad))
        self._norm_window.append(grad_norm)
        self._hill.update(grad_norm)
        self._log_grad_norm.append(grad_norm)

        tau, lr_t = self._compute_tau_and_lr(t)
        self._log_tau.append(tau)
        self._log_lr.append(lr_t)

        clipped = clip_gradient(grad, tau)
        return x - lr_t * clipped

    def reset(self):
        self._hill.reset()
        self._norm_window.clear()
        self._log_p_hat.clear()
        self._log_alpha_hat.clear()
        self._log_tau.clear()
        self._log_sigma_hat.clear()
        self._log_grad_norm.clear()
        self._log_lr.clear()

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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_tau_and_lr(self, t: int):
        """
        Returns (tau, lr_t) for step t.

        During burn-in: tau = tau_0, lr = lr_0 (or lr_0/sqrt(t) for growing).
        After burn-in:  compute from schedule using p_hat and sigma_hat.
        """
        p_hat_val = self._hill.p_hat(p_min=self.p_min, p_max=self.p_max)
        alpha_hat = self._hill.alpha_hat

        # LR (independent of burn-in for growing schedule)
        if self.schedule == 'growing':
            lr_t = self.lr_0 / np.sqrt(t)
        else:
            lr_t = self.lr_0

        if p_hat_val is None:
            # Burn-in
            self._log_alpha_hat.append(float("nan"))
            self._log_p_hat.append(float("nan"))
            self._log_sigma_hat.append(float("nan"))
            return self.tau_0, lr_t

        sigma_hat = float(np.median(list(self._norm_window)))

        if self.schedule == 'growing':
            tau = sigma_hat * (t ** (1.0 / p_hat_val))
        else:  # fixed
            tau = self.tau_scale * sigma_hat

        tau = float(np.clip(tau, self.tau_min, self.tau_max))

        self._log_alpha_hat.append(alpha_hat)
        self._log_p_hat.append(p_hat_val)
        self._log_sigma_hat.append(sigma_hat)

        return tau, lr_t

    def __repr__(self):
        return (f"AlphaRobustSGD(lr_0={self.lr_0}, schedule={self.schedule}, "
                f"W={self._hill.W}, k={self._hill.k})")

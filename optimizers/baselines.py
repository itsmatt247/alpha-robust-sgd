"""
Baseline optimizers for Phase 1 (numpy, quadratic bowl).

Interface: every optimizer exposes
    x_new = optimizer.step(x, grad, t)   # t is 1-indexed step count
    optimizer.reset()                     # clear internal state

Phase 2 note: these will be ported to torch.optim.Optimizer subclasses
when we move to neural network training.

Optimizers:
  - VanillaSGD
  - OracleClippedSGD    supports two threshold schedules:
      'growing' : tau_t = sigma * t^{1/p},  lr_t = lr_0 / sqrt(t)
      'fixed'   : tau   = tau_scale * sigma (constant),  lr_t = lr_0
  - AdaGrad             stand-in for AdamW in Phase 1
  - ZClip               EMA-based z-score clipping (Kumar et al. 2025)
                        Assumes Gaussian gradient norms — misspecified for heavy tails.
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def clip_gradient(grad: np.ndarray, tau: float) -> np.ndarray:
    """Norm clipping: Clip(g, tau) = g * min(1, tau / ||g||)."""
    norm = np.linalg.norm(grad)
    if norm <= tau:
        return grad.copy()
    return grad * (tau / norm)


# ---------------------------------------------------------------------------
# Vanilla SGD
# ---------------------------------------------------------------------------

class VanillaSGD:
    """x_{t+1} = x_t - lr * g_t.  No clipping."""

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, x: np.ndarray, grad: np.ndarray, t: int) -> np.ndarray:
        return x - self.lr * grad

    def reset(self):
        pass

    def __repr__(self):
        return f"VanillaSGD(lr={self.lr})"


# ---------------------------------------------------------------------------
# Oracle Clipped SGD  (two schedules)
# ---------------------------------------------------------------------------

class OracleClippedSGD:
    """
    Clipped SGD with oracle knowledge of p and sigma.
    AlphaRobustSGD aims to match this without knowing either.

    schedule='growing':
        tau_t = sigma * t^{1/p}
        lr_t  = lr_0 / sqrt(t)          ← natural theoretical pairing
        (growing threshold + decaying LR so updates shrink overall)

    schedule='fixed':
        tau   = tau_scale * sigma        ← constant, consistent with Zhang et al.
        lr_t  = lr_0                     ← fixed LR

    Args:
        lr_0 (float): Base learning rate (decayed if schedule='growing').
        p (float): True moment parameter (oracle). In (1, 2].
        sigma (float): True noise scale (oracle).
        schedule (str): 'growing' or 'fixed'.
        tau_scale (float): Multiplier C in tau = C * sigma (fixed schedule only).
        tau_min, tau_max: Safety clamps on tau.
    """

    def __init__(
        self,
        lr_0: float = 0.01,
        p: float = 1.5,
        sigma: float = 1.0,
        schedule: str = 'fixed',
        tau_scale: float = 5.0,
        tau_min: float = 1e-3,
        tau_max: float = 1e6,
    ):
        assert 1.0 < p <= 2.0, f"p must be in (1, 2], got {p}"
        assert schedule in ('growing', 'fixed'), f"schedule must be 'growing' or 'fixed'"
        self.lr_0 = lr_0
        self.p = p
        self.sigma = sigma
        self.schedule = schedule
        self.tau_scale = tau_scale
        self.tau_min = tau_min
        self.tau_max = tau_max
        self._taus = []
        self._lrs  = []

    def step(self, x: np.ndarray, grad: np.ndarray, t: int) -> np.ndarray:
        if self.schedule == 'growing':
            lr_t  = self.lr_0 / np.sqrt(t)
            tau_t = self.sigma * (t ** (1.0 / self.p))
        else:  # fixed
            lr_t  = self.lr_0
            tau_t = self.tau_scale * self.sigma

        tau_t = float(np.clip(tau_t, self.tau_min, self.tau_max))
        self._taus.append(tau_t)
        self._lrs.append(lr_t)

        clipped = clip_gradient(grad, tau_t)
        return x - lr_t * clipped

    def reset(self):
        self._taus.clear()
        self._lrs.clear()

    @property
    def tau_history(self) -> np.ndarray:
        return np.array(self._taus)

    @property
    def p_hat_history(self):
        return None

    def __repr__(self):
        return (f"OracleClippedSGD(lr_0={self.lr_0}, p={self.p}, "
                f"sigma={self.sigma}, schedule={self.schedule})")


# ---------------------------------------------------------------------------
# AdaGrad (Phase 1 stand-in for AdamW)
# ---------------------------------------------------------------------------

class AdaGrad:
    """
    Per-coordinate adaptive LR.
    Will be replaced with AdamW in Phase 2.
    """

    def __init__(self, lr: float = 0.1, eps: float = 1e-8):
        self.lr = lr
        self.eps = eps
        self._G = None

    def step(self, x: np.ndarray, grad: np.ndarray, t: int) -> np.ndarray:
        if self._G is None:
            self._G = np.zeros_like(x)
        self._G += grad ** 2
        return x - (self.lr / (np.sqrt(self._G) + self.eps)) * grad

    def reset(self):
        self._G = None

    def __repr__(self):
        return f"AdaGrad(lr={self.lr})"


# ---------------------------------------------------------------------------
# ZClip  (Kumar et al. 2025)
# ---------------------------------------------------------------------------

class ZClip:
    """
    Adaptive gradient clipping via EMA z-score (Kumar et al. 2025).

    Tracks an EMA of the mean and variance of gradient norms. At each step,
    computes z = (||g|| - ema_mean) / sqrt(ema_var). If z > z_thresh, clips
    the gradient to ema_mean + z_thresh * sqrt(ema_var).

    Statistical note: assumes gradient norms are locally Gaussian —
    a misspecification for heavy-tailed (Pareto/alpha-stable) distributions.
    Included as a baseline to contrast with EVT-based clipping (AlphaRobust).

    Args:
        lr (float): Learning rate.
        z_thresh (float): Z-score threshold above which the gradient is clipped.
        ema_alpha (float): EMA decay for mean/variance (0 < ema_alpha < 1).
    """

    def __init__(self, lr: float = 0.01, z_thresh: float = 2.5, ema_alpha: float = 0.01):
        self.lr        = lr
        self.z_thresh  = z_thresh
        self.ema_alpha = ema_alpha
        self._ema_mean: Optional[float] = None
        self._ema_var:  float = 0.0
        self._taus = []

    def step(self, x: np.ndarray, grad: np.ndarray, t: int) -> np.ndarray:
        norm = float(np.linalg.norm(grad))

        if self._ema_mean is None:
            self._ema_mean = norm
            self._ema_var  = 0.0
        else:
            delta          = norm - self._ema_mean
            self._ema_mean += self.ema_alpha * delta
            self._ema_var   = (1 - self.ema_alpha) * (
                self._ema_var + self.ema_alpha * delta ** 2
            )

        std = float(self._ema_var ** 0.5) + 1e-8
        z   = (norm - self._ema_mean) / std

        if z > self.z_thresh:
            tau = float(self._ema_mean + self.z_thresh * std)
            clipped = clip_gradient(grad, tau)
        else:
            tau     = norm
            clipped = grad.copy()

        self._taus.append(tau)
        return x - self.lr * clipped

    def reset(self):
        self._ema_mean = None
        self._ema_var  = 0.0
        self._taus.clear()

    @property
    def tau_history(self) -> np.ndarray:
        return np.array(self._taus)

    @property
    def p_hat_history(self):
        return None

    def __repr__(self):
        return f"ZClip(lr={self.lr}, z_thresh={self.z_thresh}, ema_alpha={self.ema_alpha})"

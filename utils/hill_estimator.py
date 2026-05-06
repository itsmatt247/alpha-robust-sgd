"""
Hill Estimator for online tail index estimation.

Estimates the tail index alpha of a heavy-tailed distribution from the k
largest order statistics in a rolling window of observations.

Relationship to Zhang et al.'s moment parameter p:
    If the tail index is alpha, then E[|X|^p] < inf for p < alpha.
    So p_hat should be used as: p_hat = clip(alpha_hat - epsilon, 1.01, 1.99)
    to stay within the bounded p-th moment regime.

Usage:
    estimator = HillEstimator(window_size=100, k=25)
    for norm in gradient_norm_stream:
        alpha_hat = estimator.update(norm)
"""

import numpy as np
from collections import deque
from typing import Optional, List, Dict


class HillEstimator:
    """
    Rolling-window Hill estimator for the tail index alpha.

    Args:
        window_size (int): Number of most recent observations to keep (W).
        k (int): Number of upper order statistics to use. Must be < window_size.
                 Controls bias-variance tradeoff:
                   - small k → low bias, high variance
                   - large k → high bias (body contaminates tail), low variance
                 Rule of thumb: k ~ W // 4.
        epsilon (float): Safety margin subtracted from alpha_hat to get p_hat.
                         Ensures E[|X|^p] < inf. Default 0.05.
    """

    def __init__(self, window_size: int = 100, k: int = None, epsilon: float = 0.05):
        if k is None:
            k = window_size // 4
        assert 2 <= k < window_size, f"k must satisfy 2 <= k < window_size, got k={k}, W={window_size}"

        self.W = window_size
        self.k = k
        self.epsilon = epsilon

        self._window = deque(maxlen=window_size)
        self._alpha_hat = None  # None until we have enough data

    def update(self, value: float) -> Optional[float]:
        """
        Add a new observation and recompute the Hill estimate.

        Args:
            value: A positive scalar (e.g., gradient norm).

        Returns:
            alpha_hat (float) if window has >= k+1 observations, else None.
        """
        value = float(value)
        if value <= 0:
            value = 1e-12  # guard: zero/negative norms → negligible placeholder
        self._window.append(value)

        if len(self._window) < self.k + 1:
            return None

        self._alpha_hat = self._compute(list(self._window), self.k)
        return self._alpha_hat

    def p_hat(self, p_min: float = 1.01, p_max: float = 1.99) -> Optional[float]:
        """
        Returns the moment parameter estimate p_hat = clip(alpha_hat - epsilon).

        This is what gets plugged into the clipping threshold tau_t = sigma * t^(1/p_hat).

        Returns None during burn-in (fewer than k+1 observations seen).
        """
        if self._alpha_hat is None:
            return None
        raw = self._alpha_hat - self.epsilon
        return float(np.clip(raw, p_min, p_max))

    def reset(self):
        """Clear the window and cached estimate."""
        self._window.clear()
        self._alpha_hat = None

    @property
    def ready(self) -> bool:
        """True once the window has enough data to produce an estimate."""
        return len(self._window) >= self.k + 1

    @property
    def alpha_hat(self) -> Optional[float]:
        return self._alpha_hat

    @staticmethod
    def _compute(data: List[float], k: int) -> float:
        """
        Core Hill estimator.

        Given n observations, uses the k largest to estimate alpha:
            alpha_hat = 1 / ( (1/k) * sum_{i=1}^{k} log(X_{(n-i+1)} / X_{(n-k)}) )

        where X_{(1)} <= ... <= X_{(n)} are the order statistics.

        Args:
            data: List of positive floats (unsorted).
            k: Number of upper order statistics.

        Returns:
            alpha_hat (float), clamped to [0.5, 10.0] for numerical stability.
        """
        arr = np.array(data, dtype=np.float64)
        arr_sorted = np.sort(arr)          # ascending order
        n = len(arr_sorted)

        # threshold: the (n-k)-th order statistic (1-indexed: X_{(n-k)})
        x_threshold = arr_sorted[n - k - 1]

        if x_threshold <= 0:
            return 2.0  # fallback: treat as light-tailed

        # top-k exceedances: log(X_{(n-k+1)}, ..., X_{(n)}) - log(X_{(n-k)})
        top_k = arr_sorted[n - k:]        # k largest values
        log_ratios = np.log(top_k) - np.log(x_threshold)

        mean_log_ratio = np.mean(log_ratios)

        if mean_log_ratio <= 0:
            return 2.0  # degenerate case

        alpha_hat = 1.0 / mean_log_ratio
        return float(np.clip(alpha_hat, 0.5, 10.0))


# ---------------------------------------------------------------------------
# Batch / static API (useful for unit tests and offline analysis)
# ---------------------------------------------------------------------------

def hill_estimate(data: np.ndarray, k: int) -> float:
    """
    Stateless Hill estimate on a fixed array of positive observations.

    Args:
        data: 1-D array of positive floats.
        k: Number of upper order statistics.

    Returns:
        alpha_hat (float).
    """
    return HillEstimator._compute(data.tolist(), k)


def sweep_k(data: np.ndarray, k_values: List[int]) -> Dict[int, float]:
    """
    Compute Hill estimates for multiple k values on the same data.
    Useful for visualizing the Hill plot (alpha_hat vs k) to find stable regions.

    Args:
        data: 1-D array of positive floats.
        k_values: List of k values to sweep.

    Returns:
        Dict mapping k -> alpha_hat.
    """
    return {k: hill_estimate(data, k) for k in k_values}

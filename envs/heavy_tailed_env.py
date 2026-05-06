"""
Synthetic heavy-tailed optimization environment.

Landscape: strongly convex quadratic bowl  f(x) = 1/2 * x^T A x
           with known minimum at x* = 0 and f(x*) = 0.

Noise:     isotropic heavy-tailed noise injected onto the true gradient.
           Direction ~ Uniform(unit sphere in R^d)
           Magnitude ~ Pareto(alpha, scale=sigma)   [tail index = alpha]

           This gives E[||eps||^p] < inf iff p < alpha,
           matching Zhang et al.'s bounded p-th moment assumption.

Usage:
    env = QuadraticEnv(dim=50, alpha=1.5, noise_scale=1.0, condition_number=10.0)
    x = env.init_x()
    for t in range(T):
        g = env.noisy_gradient(x)   # stochastic gradient
        loss = env.loss(x)          # true loss (no noise)
        x = x - lr * g
"""

import numpy as np
from typing import Optional, Tuple


class QuadraticEnv:
    """
    Strongly convex quadratic bowl with heavy-tailed gradient noise.

    Args:
        dim (int): Problem dimension d.
        alpha (float): True tail index of the noise distribution.
                       Moment p exists iff p < alpha. Typical range: (1.0, 2.0).
        noise_scale (float): Scale parameter sigma of the Pareto noise.
                             Controls the "size" of a typical noise draw.
        condition_number (float): Condition number kappa of A.
                                  kappa=1 → isotropic bowl (all eigenvalues equal).
                                  kappa=100 → ill-conditioned.
        init_scale (float): x_0 is drawn uniformly from [-init_scale, init_scale]^d.
        seed (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        dim: int = 50,
        alpha: float = 1.5,
        noise_scale: float = 1.0,
        condition_number: float = 10.0,
        init_scale: float = 5.0,
        seed: int = 42,
    ):
        assert alpha > 1.0, "alpha must be > 1 (need at least finite mean for the noise)"
        assert condition_number >= 1.0, "condition_number must be >= 1"

        self.dim = dim
        self.alpha = alpha
        self.noise_scale = noise_scale
        self.condition_number = condition_number
        self.init_scale = init_scale
        self.rng = np.random.default_rng(seed)

        self._A = self._build_A()
        self._x0 = None  # set by init_x()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_x(self) -> np.ndarray:
        """Sample a random starting point."""
        self._x0 = self.rng.uniform(-self.init_scale, self.init_scale, size=self.dim)
        return self._x0.copy()

    def loss(self, x: np.ndarray) -> float:
        """True (noiseless) loss f(x) = 1/2 * x^T A x."""
        return float(0.5 * x @ self._A @ x)

    def true_gradient(self, x: np.ndarray) -> np.ndarray:
        """True gradient ∇f(x) = Ax."""
        return self._A @ x

    def noisy_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Stochastic gradient = true gradient + heavy-tailed noise.

        Noise construction:
          - direction u ~ Uniform(S^{d-1})
          - magnitude r ~ Pareto(alpha, scale=noise_scale)
            i.e. r = noise_scale * (1 - U)^{-1/alpha},  U ~ Uniform[0,1]
          - sign  s ~ Rademacher (±1 with equal prob) for zero-mean noise
          - eps = s * r * u

        The sign flip ensures the noise is mean-zero (when alpha > 1,
        the Pareto has finite mean, so without flipping the noise would
        add a positive bias to every gradient).
        """
        true_grad = self.true_gradient(x)
        noise = self._sample_noise()
        return true_grad + noise

    def set_alpha(self, new_alpha: float):
        """
        Hot-swap the tail index mid-training.
        Used for stress-testing: inject a sudden shift in noise heaviness.
        """
        assert new_alpha > 1.0
        self.alpha = new_alpha

    def optimal_loss(self) -> float:
        """f(x*) = 0 by construction."""
        return 0.0

    def distance_to_optimum(self, x: np.ndarray) -> float:
        """||x - x*||_2 = ||x||_2  (since x* = 0)."""
        return float(np.linalg.norm(x))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_A(self) -> np.ndarray:
        """
        Construct a random symmetric positive definite matrix A with
        condition number = self.condition_number.

        Eigenvalues are log-uniformly spaced in [1, condition_number]
        so the landscape isn't trivially aligned with any axis.
        The eigenvectors are a random orthogonal matrix (Haar-distributed).
        """
        d = self.dim
        # Random orthogonal matrix via QR decomposition of a Gaussian matrix
        G = self.rng.standard_normal((d, d))
        Q, _ = np.linalg.qr(G)

        # Log-uniform eigenvalues in [1, kappa]
        log_eigs = np.linspace(0, np.log(self.condition_number), d)
        eigenvalues = np.exp(log_eigs)

        A = Q @ np.diag(eigenvalues) @ Q.T
        # Symmetrize to kill any floating-point asymmetry
        return 0.5 * (A + A.T)

    def _sample_noise(self) -> np.ndarray:
        """
        Sample one isotropic heavy-tailed noise vector in R^d.

        Magnitude: Pareto with tail index self.alpha and scale self.noise_scale.
          P(R > r) = (noise_scale / r)^alpha  for r >= noise_scale
          Sampled via inverse CDF: R = noise_scale * U^{-1/alpha}, U ~ Uniform(0,1)

        Direction: uniform on the unit sphere (normalize a standard Gaussian).

        Sign: Rademacher ±1 for zero-mean noise.
        """
        # Magnitude
        u = self.rng.uniform(0.0, 1.0)
        magnitude = self.noise_scale * (u ** (-1.0 / self.alpha))

        # Direction
        direction = self.rng.standard_normal(self.dim)
        direction /= np.linalg.norm(direction)

        # Sign flip for zero mean
        sign = self.rng.choice([-1.0, 1.0])

        return sign * magnitude * direction


# ---------------------------------------------------------------------------
# Convenience: batch gradient norm sampler (for testing Hill estimator offline)
# ---------------------------------------------------------------------------

def sample_gradient_norms(
    env: QuadraticEnv,
    x: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """
    Draw n_samples noisy gradient norms at a fixed x.
    Useful for verifying that the Hill estimator recovers env.alpha
    from gradient norms at a known point.
    """
    norms = np.array([
        np.linalg.norm(env.noisy_gradient(x))
        for _ in range(n_samples)
    ])
    return norms

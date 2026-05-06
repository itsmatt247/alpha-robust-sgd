"""
Unit tests for HillEstimator.

Tests:
  1. Static Pareto samples — does alpha_hat converge to true alpha?
  2. Rolling update API — same result as batch API?
  3. Burn-in period — returns None until k+1 observations are seen.
  4. p_hat clamping — stays within [1.01, 1.99].
  5. Hill plot — alpha_hat is stable across a range of k values.

Run with:
    python -m pytest utils/test_hill_estimator.py -v
or just:
    python utils/test_hill_estimator.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.hill_estimator import HillEstimator, hill_estimate, sweep_k


def sample_pareto(alpha: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample from a Pareto(alpha) distribution with unit scale.
    Tail: P(X > x) = x^{-alpha} for x >= 1.
    Moment p exists iff p < alpha.
    """
    u = rng.uniform(0, 1, size=n)
    return (1 - u) ** (-1.0 / alpha)


# ---------------------------------------------------------------------------
# Test 1: Convergence to true alpha on large static samples
# ---------------------------------------------------------------------------
def test_static_convergence():
    rng = np.random.default_rng(42)
    true_alphas = [1.2, 1.5, 1.8]
    n = 5000
    k = n // 4

    print("Test 1: Static convergence (n=5000, k=n//4)")
    for alpha in true_alphas:
        data = sample_pareto(alpha, n, rng)
        est = hill_estimate(data, k)
        err = abs(est - alpha) / alpha
        status = "PASS" if err < 0.10 else "FAIL"
        print(f"  alpha={alpha:.1f}  alpha_hat={est:.4f}  rel_err={err:.3f}  [{status}]")
        assert err < 0.10, f"Relative error {err:.3f} > 10% for alpha={alpha}"
    print()


# ---------------------------------------------------------------------------
# Test 2: Rolling update API matches batch API
# ---------------------------------------------------------------------------
def test_rolling_matches_batch():
    rng = np.random.default_rng(0)
    alpha = 1.5
    W, k = 200, 50
    data = sample_pareto(alpha, W, rng)

    # Batch
    batch_est = hill_estimate(data, k)

    # Rolling: feed all W points through the update API
    estimator = HillEstimator(window_size=W, k=k)
    rolling_est = None
    for x in data:
        rolling_est = estimator.update(x)

    print("Test 2: Rolling API matches batch API")
    print(f"  batch_est={batch_est:.6f}  rolling_est={rolling_est:.6f}  delta={abs(batch_est - rolling_est):.2e}")
    assert abs(batch_est - rolling_est) < 1e-10, "Rolling and batch estimates differ"
    print("  PASS\n")


# ---------------------------------------------------------------------------
# Test 3: Burn-in — None until k+1 observations
# ---------------------------------------------------------------------------
def test_burnin():
    W, k = 100, 25
    estimator = HillEstimator(window_size=W, k=k)
    rng = np.random.default_rng(7)

    print("Test 3: Burn-in period")
    for i in range(k):  # feed k observations (not enough)
        result = estimator.update(rng.uniform(1, 5))
        assert result is None, f"Expected None at step {i}, got {result}"

    # k+1-th observation should produce a result
    result = estimator.update(rng.uniform(1, 5))
    assert result is not None, "Expected a float after k+1 observations"
    print(f"  None for first {k} steps, estimate={result:.4f} at step {k+1}  PASS\n")


# ---------------------------------------------------------------------------
# Test 4: p_hat stays in [1.01, 1.99]
# ---------------------------------------------------------------------------
def test_p_hat_clamping():
    rng = np.random.default_rng(3)
    print("Test 4: p_hat clamping")

    # Very heavy tail (alpha ~ 1.05) — p_hat should be clamped to >= 1.01
    data_heavy = sample_pareto(1.05, 2000, rng)
    est_heavy = HillEstimator(window_size=2000, k=500)
    for x in data_heavy:
        est_heavy.update(x)
    p = est_heavy.p_hat()
    assert p is not None
    assert 1.01 <= p <= 1.99, f"p_hat={p} out of range"
    print(f"  Heavy tail (alpha=1.05): p_hat={p:.4f}  PASS")

    # Very light tail (normal-ish noise, large alpha) — p_hat should clamp to <= 1.99
    data_light = np.abs(rng.normal(0, 1, 2000)) + 1.0
    est_light = HillEstimator(window_size=2000, k=500)
    for x in data_light:
        est_light.update(x)
    p = est_light.p_hat()
    assert p is not None
    assert 1.01 <= p <= 1.99, f"p_hat={p} out of range"
    print(f"  Light tail (Gaussian):   p_hat={p:.4f}  PASS\n")


# ---------------------------------------------------------------------------
# Test 5: Hill plot — stable region exists across k
# ---------------------------------------------------------------------------
def test_hill_plot():
    rng = np.random.default_rng(99)
    alpha = 1.6
    n = 3000
    data = sample_pareto(alpha, n, rng)
    k_values = list(range(20, 800, 20))
    estimates = sweep_k(data, k_values)

    # Find estimates in the "stable" middle range of k
    mid_k = [k for k in k_values if n // 10 <= k <= n // 3]
    mid_estimates = [estimates[k] for k in mid_k]
    mean_est = np.mean(mid_estimates)
    std_est = np.std(mid_estimates)

    print("Test 5: Hill plot stability")
    print(f"  true alpha={alpha}, mean(alpha_hat) over k in [{mid_k[0]},{mid_k[-1]}] = {mean_est:.4f} ± {std_est:.4f}")
    assert abs(mean_est - alpha) / alpha < 0.15, f"Mean estimate {mean_est:.4f} too far from true {alpha}"
    print("  PASS\n")


if __name__ == "__main__":
    test_static_convergence()
    test_rolling_matches_batch()
    test_burnin()
    test_p_hat_clamping()
    test_hill_plot()
    print("All tests passed.")

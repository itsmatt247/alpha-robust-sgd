"""
Tests for QuadraticEnv.

Tests:
  1. A is symmetric positive definite with the right condition number.
  2. True gradient is exact (Ax).
  3. Noise is zero-mean over many samples (unbiased gradient).
  4. Hill estimator recovers env.alpha from gradient norms (end-to-end check).
  5. set_alpha hot-swap shifts the estimated tail index.
  6. Loss is non-negative and zero only at x=0.

Run:
    python3 envs/test_heavy_tailed_env.py
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.heavy_tailed_env import QuadraticEnv, sample_gradient_norms
from utils.hill_estimator import hill_estimate


def test_A_is_spd():
    env = QuadraticEnv(dim=20, alpha=1.5, condition_number=50.0, seed=0)
    A = env._A
    # Symmetric
    assert np.allclose(A, A.T, atol=1e-10), "A is not symmetric"
    # Positive definite: all eigenvalues > 0
    eigs = np.linalg.eigvalsh(A)
    assert np.all(eigs > 0), f"A has non-positive eigenvalues: {eigs.min():.4e}"
    # Condition number
    kappa = eigs.max() / eigs.min()
    assert abs(kappa - 50.0) / 50.0 < 0.01, f"Condition number {kappa:.2f} != 50"
    print(f"Test 1 PASS — A is SPD, kappa={kappa:.2f}")


def test_true_gradient():
    env = QuadraticEnv(dim=10, alpha=1.5, seed=1)
    x = env.init_x()
    expected = env._A @ x
    actual = env.true_gradient(x)
    assert np.allclose(expected, actual), "True gradient mismatch"
    print("Test 2 PASS — true gradient = Ax")


def test_noise_is_zero_mean():
    """
    Over many samples the average noise should be ~0 (unbiased gradient).
    We fix x=0 so true gradient = 0, and all noisy gradients are pure noise.
    """
    env = QuadraticEnv(dim=30, alpha=1.5, noise_scale=1.0, seed=2)
    x = np.zeros(env.dim)
    n = 20_000
    grads = np.array([env.noisy_gradient(x) for _ in range(n)])
    mean_noise = grads.mean(axis=0)
    max_abs = np.abs(mean_noise).max()
    # With 20k samples the Monte Carlo error should be well below 0.1
    assert max_abs < 0.1, f"Noise not zero-mean: max |mean| = {max_abs:.4f}"
    print(f"Test 3 PASS — noise mean ≈ 0 (max |mean_i| = {max_abs:.4f})")


def test_hill_recovers_alpha():
    """
    Key end-to-end test: Hill estimator should recover env.alpha from gradient norms.

    We fix x at a point where ||true_grad|| is small relative to noise so the
    norms are dominated by the heavy-tailed noise and carry its tail signature.
    """
    print("\nTest 4: Hill estimator recovers env.alpha from gradient norms")
    rng = np.random.default_rng(99)
    true_alphas = [1.2, 1.5, 1.8]
    n_samples = 5000
    k = n_samples // 4

    for alpha in true_alphas:
        env = QuadraticEnv(dim=50, alpha=alpha, noise_scale=5.0,
                           condition_number=1.0, seed=42)
        # x close to 0 so true gradient is negligible vs. large noise
        x = np.zeros(env.dim)
        norms = sample_gradient_norms(env, x, n_samples)
        alpha_hat = hill_estimate(norms, k)
        rel_err = abs(alpha_hat - alpha) / alpha
        status = "PASS" if rel_err < 0.10 else "FAIL"
        print(f"  alpha={alpha:.1f}  alpha_hat={alpha_hat:.4f}  rel_err={rel_err:.3f}  [{status}]")
        assert rel_err < 0.10, f"Hill failed to recover alpha={alpha}: got {alpha_hat:.4f}"
    print()


def test_set_alpha_hot_swap():
    """
    After set_alpha, gradient norms should reflect the new tail index.
    """
    env = QuadraticEnv(dim=50, alpha=1.8, noise_scale=5.0,
                       condition_number=1.0, seed=10)
    x = np.zeros(env.dim)
    n = 3000
    k = n // 4

    # Before swap: should see alpha ~ 1.8
    norms_before = sample_gradient_norms(env, x, n)
    alpha_before = hill_estimate(norms_before, k)

    # Hot-swap to alpha=1.2 (heavier tails)
    env.set_alpha(1.2)
    norms_after = sample_gradient_norms(env, x, n)
    alpha_after = hill_estimate(norms_after, k)

    print("Test 5: set_alpha hot-swap")
    print(f"  Before: alpha_hat={alpha_before:.4f} (true=1.8)")
    print(f"  After:  alpha_hat={alpha_after:.4f}  (true=1.2)")
    assert alpha_before > alpha_after, "Heavier tails after swap should give lower alpha_hat"
    assert abs(alpha_after - 1.2) / 1.2 < 0.15, f"Post-swap estimate too far: {alpha_after:.4f}"
    print("  PASS\n")


def test_loss_properties():
    env = QuadraticEnv(dim=20, alpha=1.5, seed=5)
    x = env.init_x()
    assert env.loss(np.zeros(env.dim)) == 0.0, "Loss at optimum != 0"
    assert env.loss(x) > 0.0, "Loss away from optimum should be positive"
    assert env.optimal_loss() == 0.0
    print("Test 6 PASS — loss properties correct")


if __name__ == "__main__":
    test_A_is_spd()
    test_true_gradient()
    test_noise_is_zero_mean()
    test_hill_recovers_alpha()
    test_set_alpha_hot_swap()
    test_loss_properties()
    print("All environment tests passed.")

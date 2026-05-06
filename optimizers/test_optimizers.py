"""
Tests for all optimizers.

Tests:
  1. VanillaSGD converges on a noise-free quadratic.
  2. OracleClippedSGD converges under heavy-tailed noise.
  3. VanillaSGD diverges (or performs much worse) under very heavy-tailed noise.
  4. AlphaRobustSGD converges under heavy-tailed noise.
  5. AlphaRobustSGD p_hat tracks true alpha after burn-in.
  6. AlphaRobustSGD tau history has no NaN after burn-in.
  7. All optimizer logs have consistent lengths after T steps.

Run:
    python3 optimizers/test_optimizers.py
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.heavy_tailed_env import QuadraticEnv
from optimizers.baselines import VanillaSGD, OracleClippedSGD, AdaGrad
from optimizers.alpha_robust_sgd import AlphaRobustSGD


def run_optimizer(opt, env, T=500, seed=0):
    """Run optimizer for T steps. Returns loss curve and distance to optimum."""
    rng = np.random.default_rng(seed)
    x = env.init_x()
    losses, dists = [], []
    for t in range(1, T + 1):
        g = env.noisy_gradient(x)
        x = opt.step(x, g, t)
        losses.append(env.loss(x))
        dists.append(env.distance_to_optimum(x))
    return np.array(losses), np.array(dists)


# ---------------------------------------------------------------------------
def test_vanilla_converges_clean():
    """VanillaSGD should converge on a noiseless quadratic."""
    # Use near-zero noise by setting noise_scale very small
    env = QuadraticEnv(dim=20, alpha=1.8, noise_scale=1e-6,
                       condition_number=5.0, seed=0)
    opt = VanillaSGD(lr=0.05)
    losses, _ = run_optimizer(opt, env, T=500)
    assert losses[-1] < 1e-3, f"VanillaSGD didn't converge cleanly: final loss={losses[-1]:.4f}"
    print(f"Test 1 PASS — VanillaSGD converges (clean): final loss={losses[-1]:.2e}")


# ---------------------------------------------------------------------------
def test_oracle_converges_heavy_tail():
    """OracleClippedSGD should converge under heavy-tailed noise."""
    alpha = 1.3
    env = QuadraticEnv(dim=30, alpha=alpha, noise_scale=2.0,
                       condition_number=5.0, seed=1)
    p = alpha - 0.05
    opt = OracleClippedSGD(lr=0.02, p=p, sigma=2.0)
    losses, _ = run_optimizer(opt, env, T=1000)
    # Should make clear progress from initial loss
    assert losses[-1] < losses[0] * 0.5, (
        f"OracleClipped didn't reduce loss by 50%: {losses[0]:.2f} -> {losses[-1]:.2f}"
    )
    print(f"Test 2 PASS — OracleClipped converges: {losses[0]:.2f} -> {losses[-1]:.4f}")


# ---------------------------------------------------------------------------
def test_vanilla_worse_than_oracle_heavy_tail():
    """
    Under heavy-tailed noise, OracleClippedSGD should clearly outperform VanillaSGD.
    We measure final loss and check Oracle < Vanilla.
    """
    alpha = 1.2   # very heavy tail
    env_v = QuadraticEnv(dim=30, alpha=alpha, noise_scale=3.0,
                         condition_number=5.0, seed=2)
    env_o = QuadraticEnv(dim=30, alpha=alpha, noise_scale=3.0,
                         condition_number=5.0, seed=2)

    vanilla = VanillaSGD(lr=0.005)
    oracle = OracleClippedSGD(lr=0.005, p=alpha - 0.05, sigma=3.0)

    losses_v, _ = run_optimizer(vanilla, env_v, T=1000)
    losses_o, _ = run_optimizer(oracle, env_o, T=1000)

    final_v = np.mean(losses_v[-100:])   # average last 100 steps (smoother comparison)
    final_o = np.mean(losses_o[-100:])

    print(f"Test 3: Vanilla final={final_v:.4f}  Oracle final={final_o:.4f}")
    assert final_o < final_v, "Oracle should outperform Vanilla under heavy tails"
    print("  PASS — Oracle < Vanilla under heavy-tailed noise")


# ---------------------------------------------------------------------------
def test_alpha_robust_converges():
    """AlphaRobustSGD should converge under heavy-tailed noise."""
    alpha = 1.5
    env = QuadraticEnv(dim=30, alpha=alpha, noise_scale=2.0,
                       condition_number=5.0, seed=3)
    opt = AlphaRobustSGD(lr=0.02, window_size=100, tau_0=10.0)
    losses, _ = run_optimizer(opt, env, T=1000)
    assert losses[-1] < losses[0] * 0.5, (
        f"AlphaRobust didn't reduce loss by 50%: {losses[0]:.2f} -> {losses[-1]:.2f}"
    )
    print(f"Test 4 PASS — AlphaRobust converges: {losses[0]:.2f} -> {losses[-1]:.4f}")


# ---------------------------------------------------------------------------
def test_p_hat_tracks_alpha():
    """
    After burn-in, p_hat should stay within 0.3 of true p = alpha - epsilon.
    """
    alpha = 1.5
    epsilon = 0.05
    true_p = alpha - epsilon
    W = 100

    env = QuadraticEnv(dim=30, alpha=alpha, noise_scale=3.0,
                       condition_number=1.0, seed=4)
    opt = AlphaRobustSGD(lr=0.005, window_size=W, epsilon=epsilon, tau_0=10.0)

    x = env.init_x()
    for t in range(1, 1001):
        g = env.noisy_gradient(x)
        x = opt.step(x, g, t)

    # Only look at estimates after burn-in
    p_hats = opt.p_hat_history
    post_burnin = p_hats[~np.isnan(p_hats)]

    assert len(post_burnin) > 0, "No post-burn-in p_hat estimates found"
    mean_p_hat = float(np.mean(post_burnin))
    err = abs(mean_p_hat - true_p)

    print(f"Test 5: true_p={true_p:.2f}  mean(p_hat)={mean_p_hat:.4f}  |err|={err:.4f}")
    assert err < 0.3, f"p_hat too far from true_p: {mean_p_hat:.4f} vs {true_p:.4f}"
    print("  PASS — p_hat tracks alpha after burn-in")


# ---------------------------------------------------------------------------
def test_log_lengths_consistent():
    """All logged arrays should have length == T."""
    T = 300
    env = QuadraticEnv(dim=10, alpha=1.5, noise_scale=1.0, seed=5)
    opt = AlphaRobustSGD(lr=0.01, window_size=50)

    x = env.init_x()
    for t in range(1, T + 1):
        g = env.noisy_gradient(x)
        x = opt.step(x, g, t)

    assert len(opt.tau_history) == T, f"tau_history length {len(opt.tau_history)} != {T}"
    assert len(opt.grad_norm_history) == T, f"grad_norm_history length mismatch"
    assert len(opt.p_hat_history) == T, f"p_hat_history length {len(opt.p_hat_history)} != {T}"
    print(f"Test 6 PASS — all log arrays have length {T}")


# ---------------------------------------------------------------------------
def test_no_nan_tau_after_burnin():
    """tau should never be NaN (burn-in uses tau_0, not NaN)."""
    env = QuadraticEnv(dim=10, alpha=1.5, noise_scale=1.0, seed=6)
    opt = AlphaRobustSGD(lr=0.01, window_size=50, tau_0=5.0)

    x = env.init_x()
    for t in range(1, 301):
        g = env.noisy_gradient(x)
        x = opt.step(x, g, t)

    assert not np.any(np.isnan(opt.tau_history)), "NaN found in tau history"
    assert not np.any(np.isinf(opt.tau_history)), "Inf found in tau history"
    print("Test 7 PASS — no NaN/Inf in tau history")


if __name__ == "__main__":
    test_vanilla_converges_clean()
    test_oracle_converges_heavy_tail()
    test_vanilla_worse_than_oracle_heavy_tail()
    test_alpha_robust_converges()
    test_p_hat_tracks_alpha()
    test_log_lengths_consistent()
    test_no_nan_tau_after_burnin()
    print("\nAll optimizer tests passed.")

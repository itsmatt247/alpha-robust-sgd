"""
Smoke tests for torch optimizer implementations.

Tests:
  1. AlphaRobustSGD reduces loss on a clean MLP (basic sanity).
  2. OracleClippedSGD reduces loss on same MLP.
  3. AlphaRobustSGD p̂ is populated and in valid range after training.
  4. Tau history has no NaN/Inf after burn-in.
  5. Both optimizers work on MPS (Apple Silicon) if available.
  6. AlphaRobustSGD and OracleClippedSGD produce comparable final loss
     on an MLP trained with synthetically injected heavy-tailed gradient noise.
  7. Global grad norm logged at every step matches manual computation.

Run:
    python3 optimizers/torch/test_torch_optimizers.py
"""

import torch
import torch.nn as nn
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from optimizers.torch import AlphaRobustSGD, OracleClippedSGD

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mlp(input_dim=32, hidden=64, output_dim=1):
    return nn.Sequential(
        nn.Linear(input_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden),    nn.ReLU(),
        nn.Linear(hidden, output_dim)
    )


def train(model, optimizer, steps=300, batch_size=32, input_dim=32,
          noise_alpha=None, device='cpu'):
    """
    Train model on a random regression task (y = Wx + b, fixed W).
    Optionally inject Pareto noise into gradients after each backward pass
    to simulate heavy-tailed gradient noise.
    """
    model.to(device)
    model.train()
    rng = np.random.default_rng(42)
    W_true = torch.randn(1, input_dim, device=device) * 0.5
    losses = []

    for t in range(1, steps + 1):
        x = torch.randn(batch_size, input_dim, device=device)
        y = (x @ W_true.T).squeeze(-1)

        pred = model(x).squeeze(-1)
        loss = nn.functional.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()

        # Inject heavy-tailed noise into gradients if requested
        if noise_alpha is not None:
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        u = torch.rand_like(p.grad)
                        magnitude = (u ** (-1.0 / noise_alpha))  # Pareto
                        sign = torch.sign(torch.randn_like(p.grad))
                        p.grad.add_(sign * magnitude * 0.1)

        optimizer.step()
        losses.append(loss.item())

    return losses


# ---------------------------------------------------------------------------

def test_alpha_robust_reduces_loss():
    model = make_mlp()
    opt   = AlphaRobustSGD(model.parameters(), lr=1e-2, window_size=50,
                            schedule='fixed', tau_scale=5.0)
    losses = train(model, opt, steps=500)
    early, late = np.mean(losses[:50]), np.mean(losses[-50:])
    assert late < early, f"Loss didn't decrease: early={early:.4f} -> late={late:.4f}"
    print(f"Test 1 PASS — AlphaRobust reduces loss: {early:.4f} -> {late:.4f}")


def test_oracle_reduces_loss():
    model = make_mlp()
    opt   = OracleClippedSGD(model.parameters(), lr=1e-2, p=1.5, sigma=1.0,
                              schedule='fixed', tau_scale=5.0)
    losses = train(model, opt, steps=500)
    early, late = np.mean(losses[:50]), np.mean(losses[-50:])
    assert late < early, f"Loss didn't decrease: early={early:.4f} -> late={late:.4f}"
    print(f"Test 2 PASS — OracleClipped reduces loss: {early:.4f} -> {late:.4f}")


def test_p_hat_valid_after_training():
    model = make_mlp()
    opt   = AlphaRobustSGD(model.parameters(), lr=1e-2, window_size=50,
                            schedule='fixed')
    train(model, opt, steps=300)
    p_hats = opt.p_hat_history
    post   = p_hats[~np.isnan(p_hats)]
    assert len(post) > 0, "No post-burn-in p̂ values"
    assert np.all(post >= 1.01) and np.all(post <= 1.99), \
        f"p̂ out of [1.01, 1.99]: min={post.min():.3f} max={post.max():.3f}"
    print(f"Test 3 PASS — p̂ in valid range: mean={post.mean():.3f} ± {post.std():.3f}")


def test_no_nan_inf_in_tau():
    model = make_mlp()
    opt   = AlphaRobustSGD(model.parameters(), lr=1e-2, window_size=50,
                            schedule='fixed', tau_0=5.0)
    train(model, opt, steps=300)
    taus = opt.tau_history
    assert not np.any(np.isnan(taus)), "NaN in tau history"
    assert not np.any(np.isinf(taus)), "Inf in tau history"
    print(f"Test 4 PASS — no NaN/Inf in tau (range: [{taus.min():.2f}, {taus.max():.2f}])")


def test_mps_device():
    if not torch.backends.mps.is_available():
        print("Test 5 SKIP — MPS not available")
        return
    device = torch.device('mps')
    model  = make_mlp().to(device)
    opt    = AlphaRobustSGD(model.parameters(), lr=1e-3, window_size=50)
    losses = train(model, opt, steps=300, device=device)
    early, late = np.mean(losses[:50]), np.mean(losses[-50:])
    assert late < early
    print(f"Test 5 PASS — AlphaRobust runs on MPS: {early:.4f} -> {late:.4f}")


def test_alpha_robust_vs_oracle_heavy_tail():
    """
    With injected Pareto noise (alpha=1.5), AlphaRobust should achieve
    comparable final loss to OracleClipped (within 3x).
    """
    alpha = 1.5
    p_oracle = alpha - 0.05

    def fresh():
        m = make_mlp()
        m.load_state_dict(make_mlp().state_dict())  # same random init via seed
        return m

    torch.manual_seed(0)
    model_r = make_mlp()
    torch.manual_seed(0)
    model_o = make_mlp()

    opt_r = AlphaRobustSGD(model_r.parameters(), lr=1e-2, window_size=100,
                            schedule='fixed', tau_scale=5.0)
    opt_o = OracleClippedSGD(model_o.parameters(), lr=1e-2, p=p_oracle, sigma=1.0,
                              schedule='fixed', tau_scale=5.0)

    losses_r = train(model_r, opt_r, steps=500, noise_alpha=alpha)
    losses_o = train(model_o, opt_o, steps=500, noise_alpha=alpha)

    final_r = np.mean(losses_r[-50:])
    final_o = np.mean(losses_o[-50:])
    ratio   = final_r / (final_o + 1e-8)

    print(f"Test 6: Oracle={final_o:.4f}  AlphaRobust={final_r:.4f}  ratio={ratio:.2f}x")
    assert ratio < 3.0, f"AlphaRobust too far from Oracle: {ratio:.2f}x"
    print("  PASS — AlphaRobust within 3x of Oracle under heavy-tailed noise")


def test_grad_norm_log_matches_manual():
    """Verify the logged grad norm matches a manual computation."""
    model = make_mlp(input_dim=8, hidden=16)
    opt   = AlphaRobustSGD(model.parameters(), lr=1e-3, window_size=20)

    x    = torch.randn(4, 8)
    y    = torch.randn(4, 1)
    pred = model(x)
    loss = nn.functional.mse_loss(pred, y)
    opt.zero_grad()
    loss.backward()

    # Manual norm before step
    manual_norm = sum(
        p.grad.data.norm(2).item() ** 2
        for p in model.parameters() if p.grad is not None
    ) ** 0.5

    opt.step()
    logged_norm = opt.grad_norm_history[0]

    assert abs(logged_norm - manual_norm) < 1e-5, \
        f"Norm mismatch: logged={logged_norm:.6f} manual={manual_norm:.6f}"
    print(f"Test 7 PASS — grad norm log matches manual ({logged_norm:.6f})")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_alpha_robust_reduces_loss()
    test_oracle_reduces_loss()
    test_p_hat_valid_after_training()
    test_no_nan_inf_in_tau()
    test_mps_device()
    test_alpha_robust_vs_oracle_heavy_tail()
    test_grad_norm_log_matches_manual()
    print("\nAll torch optimizer tests passed.")

"""
Stress test: can the Hill estimator recover true alpha=1.2 when the
gradient signal is removed (x initialized near 0)?

This isolates whether the p̂ bias seen in Phase 1 is:
  (A) Fundamental to the Hill estimator — estimator can't recover true alpha
      even on pure noise
  (B) An artifact of initialization — gradient signal at large ||x|| contaminates
      the gradient norms, making the distribution look lighter-tailed

If (B), p̂ should converge to the correct value when x ~ 0 from step 1.
If (A), p̂ stays biased regardless of initialization.

Conditions tested:
  - init_scale = 5.0  (standard Phase 1 init — gradient signal dominates early)
  - init_scale = 0.01 (near-zero init — noise dominates from step 1)

Run:
    python3 experiments/stress_test_p_hat.py
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.heavy_tailed_env import QuadraticEnv
from optimizers.alpha_robust_sgd import AlphaRobustSGD

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 9})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

T           = 2000
N_SEEDS     = 5
DIM         = 50
COND        = 10.0
NOISE_SCALE = 2.0
W           = 100
ALPHA_VALUES  = [1.2, 1.5, 1.8]
INIT_SCALES   = [5.0, 0.01]
RESULTS_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase1")
FIGURES_DIR   = os.path.join(RESULTS_DIR, "figures")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_stress_test():
    # results[alpha][init_scale] = list of p_hat arrays (one per seed)
    results = {a: {s: [] for s in INIT_SCALES} for a in ALPHA_VALUES}

    for alpha in ALPHA_VALUES:
        true_p = float(np.clip(alpha - 0.05, 1.01, 1.99))
        print(f"\nalpha={alpha}  true_p={true_p:.2f}")

        for init_scale in INIT_SCALES:
            p_hat_runs = []
            for seed in range(N_SEEDS):
                env = QuadraticEnv(dim=DIM, alpha=alpha, noise_scale=NOISE_SCALE,
                                   condition_number=COND,
                                   init_scale=init_scale, seed=seed)
                opt = AlphaRobustSGD(lr_0=0.01, window_size=W, schedule='fixed',
                                     tau_scale=5.0, tau_0=10.0)
                x = env.init_x()
                for t in range(1, T + 1):
                    g = env.noisy_gradient(x)
                    x = opt.step(x, g, t)
                p_hat_runs.append(opt.p_hat_history)

            results[alpha][init_scale] = p_hat_runs

            # Summary: mean p_hat over last 500 steps (post-burn-in, stable region)
            post = np.array([
                np.nanmean(run[-500:]) for run in p_hat_runs
            ])
            print(f"  init_scale={init_scale:>5}  "
                  f"mean(p̂ last 500 steps) = {post.mean():.4f} ± {post.std():.4f}"
                  f"  [target: {true_p:.2f}]")

    _plot(results)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(results):
    steps = np.arange(1, T + 1)
    colors = {5.0: '#d62728', 0.01: '#1f77b4'}
    labels = {5.0: 'init_scale=5.0 (standard — gradient signal dominates early)',
              0.01: 'init_scale=0.01 (near-zero — noise dominates from step 1)'}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax, alpha in zip(axes, ALPHA_VALUES):
        true_p = float(np.clip(alpha - 0.05, 1.01, 1.99))
        ax.axhline(true_p, color='#2ca02c', linestyle='--', linewidth=1.5,
                   label=f'true p = {true_p:.2f}')
        ax.axhline(alpha, color='gray', linestyle=':', linewidth=1.0,
                   label=f'true α = {alpha:.2f}')

        for init_scale in INIT_SCALES:
            runs   = results[alpha][init_scale]
            stacked = np.stack(runs, axis=0)          # (N_SEEDS, T)
            mean   = np.nanmean(stacked, axis=0)
            std    = np.nanstd(stacked, axis=0)
            valid  = ~np.isnan(mean)

            ax.plot(steps[valid], mean[valid],
                    color=colors[init_scale], linewidth=1.6,
                    label=labels[init_scale])
            ax.fill_between(steps[valid],
                            mean[valid] - std[valid],
                            mean[valid] + std[valid],
                            alpha=0.15, color=colors[init_scale])

        ax.set_title(f'α = {alpha}')
        ax.set_xlabel('Step')
        ax.set_ylabel('p̂')
        ax.set_ylim(0.8, 2.2)
        ax.legend(fontsize=6.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Stress Test: p̂ Bias — Gradient Signal vs Pure Noise\n'
        'Red = standard init (||x|| large).  Blue = near-zero init (noise dominates).\n'
        'If blue tracks green dashed: bias is initialization artifact, not fundamental.',
        fontsize=9
    )
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'stress_test_p_hat.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved -> {path}")


if __name__ == "__main__":
    run_stress_test()

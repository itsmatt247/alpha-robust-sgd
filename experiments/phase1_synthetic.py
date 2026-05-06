"""
Phase 1: Synthetic sandbox experiment.

Primary comparison: OracleClippedSGD (knows p) vs AlphaRobustSGD (estimates p̂ via Hill).
That gap — or lack thereof — is the central result of the paper.

Two schedule variants are tested side by side:
  'fixed'  : tau = tau_scale * sigma,  fixed LR      (consistent with Zhang et al.)
  'growing': tau_t = sigma * t^{1/p},  lr_t = lr_0/sqrt(t)  (theoretical pairing)

Sweep:
  - alpha_true in {1.2, 1.5, 1.8}
  - window sizes W in {50, 100, 200}  (AlphaRobust only)
  - N_SEEDS independent runs

Outputs -> results/phase1/
  raw_results.csv, summary.csv
  figures/{schedule}_loss_curves.png
  figures/{schedule}_p_hat_tracking.png
  figures/{schedule}_tau_comparison.png

Run:
    python3 experiments/phase1_synthetic.py
"""

import numpy as np
import csv, os, sys, time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.heavy_tailed_env import QuadraticEnv
from optimizers.baselines import VanillaSGD, OracleClippedSGD, AdaGrad, ZClip
from optimizers.alpha_robust_sgd import AlphaRobustSGD

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

T           = 2000
N_SEEDS     = 5
DIM         = 50
COND        = 10.0
NOISE_SCALE = 2.0
TAU_SCALE   = 5.0       # C in tau = C * sigma  (fixed schedule)
TAU_0       = 10.0      # burn-in threshold

# LR tuned per schedule so both are competitive
LR_FIXED    = 0.01
LR_GROWING  = 0.3       # higher base LR because it decays as lr_0/sqrt(t)

ALPHA_VALUES = [1.2, 1.5, 1.8]
WINDOW_SIZES = [50, 100, 200]
SCHEDULES    = ['fixed', 'growing']

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase1")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_single(optimizer, env, T):
    x = env.init_x()
    losses, grad_norms = [], []
    for t in range(1, T + 1):
        g = env.noisy_gradient(x)
        x = optimizer.step(x, g, t)
        losses.append(env.loss(x))
        grad_norms.append(float(np.linalg.norm(g)))

    ph = getattr(optimizer, 'p_hat_history', None)
    p_hats = ph.tolist() if ph is not None and not callable(ph) and len(ph) == T \
             else [float('nan')] * T

    th = getattr(optimizer, 'tau_history', None)
    if th is None or callable(th):
        taus = [float('nan')] * T
    else:
        taus = th.tolist() if len(th) == T else [float('nan')] * T

    return {'loss': losses, 'grad_norm': grad_norms, 'p_hat': p_hats, 'tau': taus}


def build_optimizers(alpha, W, schedule):
    """
    Returns (name, optimizer) pairs for one (alpha, W, schedule) configuration.
    The Oracle/Vanilla/AdaGrad are independent of W.
    """
    p_oracle = float(np.clip(alpha - 0.05, 1.01, 1.99))
    lr = LR_GROWING if schedule == 'growing' else LR_FIXED

    return [
        ("VanillaSGD",
            VanillaSGD(lr=lr)),
        ("OracleClipped",
            OracleClippedSGD(lr_0=lr, p=p_oracle, sigma=NOISE_SCALE,
                             schedule=schedule, tau_scale=TAU_SCALE)),
        ("AdaGrad",
            AdaGrad(lr=lr * 5)),
        ("ZClip",
            ZClip(lr=lr, z_thresh=2.5, ema_alpha=0.01)),
        (f"AlphaRobust_W{W}",
            AlphaRobustSGD(lr_0=lr, window_size=W, schedule=schedule,
                           tau_scale=TAU_SCALE, tau_0=TAU_0)),
    ]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_experiment():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    raw_rows = []
    summary  = {}

    total = len(SCHEDULES) * len(ALPHA_VALUES) * len(WINDOW_SIZES) * N_SEEDS
    count = 0
    t0 = time.time()

    print(f"Phase 1 | T={T} steps | {N_SEEDS} seeds | dim={DIM}")
    print(f"Schedules: {SCHEDULES}")
    print(f"Total configs: {total}  ({total*5} optimizer runs)\n")

    for schedule in SCHEDULES:
        for alpha in ALPHA_VALUES:
            for W in WINDOW_SIZES:
                for seed in range(N_SEEDS):
                    count += 1
                    env  = QuadraticEnv(dim=DIM, alpha=alpha, noise_scale=NOISE_SCALE,
                                        condition_number=COND, seed=seed)
                    opts = build_optimizers(alpha, W, schedule)

                    for opt_name, opt in opts:
                        opt.reset()
                        result = run_single(opt, env, T)

                        for t_idx in range(T):
                            raw_rows.append({
                                'schedule':  schedule,
                                'alpha':     alpha,
                                'W':         W,
                                'seed':      seed,
                                'optimizer': opt_name,
                                'step':      t_idx + 1,
                                'loss':      result['loss'][t_idx],
                                'grad_norm': result['grad_norm'][t_idx],
                                'p_hat':     result['p_hat'][t_idx],
                                'tau':       result['tau'][t_idx],
                            })

                        final_loss = float(np.mean(result['loss'][-100:]))
                        summary.setdefault((schedule, opt_name, alpha, W), []).append(final_loss)

                    elapsed = time.time() - t0
                    print(f"  [{count}/{total}] schedule={schedule} alpha={alpha} W={W} "
                          f"seed={seed}  ({elapsed:.1f}s)")

    print(f"\nDone. Writing outputs to {RESULTS_DIR}/")
    _write_csv(raw_rows)
    _write_summary(summary)
    print("Generating figures...")
    data = _load_grouped(raw_rows)
    for schedule in SCHEDULES:
        _plot_loss_curves(data, schedule)
        _plot_p_hat_tracking(data, schedule)
        _plot_tau_comparison(data, schedule)
    print(f"All outputs written to {RESULTS_DIR}/")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def _write_csv(rows):
    path = os.path.join(RESULTS_DIR, "raw_results.csv")
    fields = ['schedule','alpha','W','seed','optimizer','step','loss','grad_norm','p_hat','tau']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  raw_results.csv  ({len(rows)} rows)")


def _write_summary(summary):
    path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['schedule','optimizer','alpha','W',
                         'mean_final_loss','std_final_loss','min_final_loss','max_final_loss'])
        for (schedule, opt_name, alpha, W), losses in sorted(summary.items()):
            writer.writerow([schedule, opt_name, alpha, W,
                             f"{np.mean(losses):.6f}", f"{np.std(losses):.6f}",
                             f"{np.min(losses):.6f}", f"{np.max(losses):.6f}"])
    print(f"  summary.csv")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _load_grouped(rows):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: {'loss':[],'grad_norm':[],'p_hat':[],'tau':[]})
    ))))
    for r in rows:
        d = data[r['schedule']][r['optimizer']][r['alpha']][r['W']][r['seed']]
        d['loss'].append(r['loss'])
        d['grad_norm'].append(r['grad_norm'])
        d['p_hat'].append(r['p_hat'])
        d['tau'].append(r['tau'])
    return data


def _mean_std(data_by_seed, key):
    arrays = [np.array(v[key]) for v in data_by_seed.values()]
    stacked = np.stack(arrays, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


def _smooth(arr, k=20):
    return np.convolve(arr, np.ones(k)/k, mode='valid')


def _plot_loss_curves(data, schedule):
    import matplotlib.pyplot as plt
    import matplotlib; matplotlib.rcParams.update({'font.size': 9})

    W_show = 100
    steps  = np.arange(1, T + 1)

    # Colour the two key players clearly; dim the others
    colors = {
        'VanillaSGD':              '#d62728',
        'OracleClipped':           '#2ca02c',
        'AdaGrad':                 '#aaaaaa',
        'ZClip':                   '#ff7f0e',
        f'AlphaRobust_W{W_show}':  '#1f77b4',
    }
    lws = {
        'OracleClipped':           2.2,
        f'AlphaRobust_W{W_show}':  2.2,
        'ZClip':                   1.6,
        'VanillaSGD':              1.4,
        'AdaGrad':                 1.0,
    }
    labels = {
        'VanillaSGD':              'Vanilla SGD',
        'OracleClipped':           'Oracle Clipped (knows p)',
        'AdaGrad':                 'AdaGrad',
        'ZClip':                   'ZClip (Gaussian assumption)',
        f'AlphaRobust_W{W_show}':  f'α-Robust SGD W={W_show} (estimates p̂)',
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, alpha in zip(axes, ALPHA_VALUES):
        for opt_name, color in colors.items():
            sched_data = data.get(schedule, {})
            opt_data   = sched_data.get(opt_name, {})
            alpha_data = opt_data.get(alpha, {})
            w_data     = alpha_data.get(W_show, {})
            if not w_data:
                continue
            mean, std = _mean_std(w_data, 'loss')
            ms = _smooth(mean)
            ax.semilogy(steps[:len(ms)], ms,
                        label=labels[opt_name], color=color,
                        linewidth=lws.get(opt_name, 1.4))

        ax.set_title(f'α = {alpha}  [{schedule} schedule]')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss (log scale)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Phase 1 Loss Curves — {schedule} schedule\n'
                 f'Primary: Oracle (green) vs α-Robust (blue)', fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, f'{schedule}_loss_curves.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {schedule}_loss_curves.png")


def _plot_p_hat_tracking(data, schedule):
    import matplotlib.pyplot as plt
    import matplotlib; matplotlib.rcParams.update({'font.size': 9})

    steps = np.arange(1, T + 1)
    w_colors = {50: '#aec7e8', 100: '#1f77b4', 200: '#08306b'}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, alpha in zip(axes, ALPHA_VALUES):
        true_p = float(np.clip(alpha - 0.05, 1.01, 1.99))
        ax.axhline(true_p, color='#2ca02c', linestyle='--', linewidth=1.5,
                   label=f'true p = {true_p:.2f}  (oracle)')
        ax.axhline(alpha,  color='gray',    linestyle=':',  linewidth=1.0,
                   label=f'true α = {alpha:.2f}')

        for W in WINDOW_SIZES:
            opt_name = f'AlphaRobust_W{W}'
            w_data   = data.get(schedule,{}).get(opt_name,{}).get(alpha,{}).get(W,{})
            if not w_data:
                continue
            mean, std = _mean_std(w_data, 'p_hat')
            valid = ~np.isnan(mean)
            ax.plot(steps[valid], mean[valid], color=w_colors[W],
                    linewidth=1.4, label=f'W={W}')
            ax.fill_between(steps[valid],
                            mean[valid] - std[valid],
                            mean[valid] + std[valid],
                            alpha=0.15, color=w_colors[W])

        ax.set_title(f'α = {alpha}')
        ax.set_xlabel('Step')
        ax.set_ylabel('p̂')
        ax.set_ylim(0.8, 2.2)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Phase 1: p̂ Tracking — {schedule} schedule\n'
                 f'Green dashed = target (oracle p).  Blue = α-Robust estimate.', fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, f'{schedule}_p_hat_tracking.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {schedule}_p_hat_tracking.png")


def _plot_tau_comparison(data, schedule):
    import matplotlib.pyplot as plt
    import matplotlib; matplotlib.rcParams.update({'font.size': 9})

    steps  = np.arange(1, T + 1)
    W_show = 100

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, alpha in zip(axes, ALPHA_VALUES):
        # Oracle tau
        o_data = data.get(schedule,{}).get('OracleClipped',{}).get(alpha,{}).get(W_show,{})
        if o_data:
            mean_o, _ = _mean_std(o_data, 'tau')
            valid = ~np.isnan(mean_o)
            ax.semilogy(steps[valid], mean_o[valid], color='#2ca02c',
                        linestyle='--', linewidth=1.8, label='Oracle τ_t')

        # AlphaRobust tau
        r_data = data.get(schedule,{}).get(f'AlphaRobust_W{W_show}',{}).get(alpha,{}).get(W_show,{})
        if r_data:
            mean_r, std_r = _mean_std(r_data, 'tau')
            valid = ~np.isnan(mean_r)
            ax.semilogy(steps[valid], mean_r[valid], color='#1f77b4',
                        linewidth=1.8, label=f'α-Robust τ_t (W={W_show})')
            ax.fill_between(steps[valid],
                            np.maximum(mean_r[valid] - std_r[valid], 1e-3),
                            mean_r[valid] + std_r[valid],
                            alpha=0.15, color='#1f77b4')

        ax.set_title(f'α = {alpha}')
        ax.set_xlabel('Step')
        ax.set_ylabel('τ_t (log scale)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Phase 1: Clipping Threshold τ_t — {schedule} schedule\n'
                 f'How close does α-Robust (blue) track Oracle (green)?', fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, f'{schedule}_tau_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {schedule}_tau_comparison.png")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_experiment()

"""
Phase 2: GPT-2 small fine-tuning with injected heavy-tailed gradient noise.

Framing:
  We study the heavy-tailed gradient regime (bounded p-th moment, Zhang et al. 2020)
  on a real Transformer loss landscape. Pareto noise is injected into GPT-2 gradients
  after each backward pass to put the training in the exact regime the theory studies.
  This is the standard experimental design for this class of paper.

Primary comparison:
  - VanillaSGD          : diverges / lags under heavy-tailed noise
  - AdamW               : current standard, implicit heavy-tail resilience
  - PseudoOracleClipped : ClippedSGD with p set from AlphaRobust pilot estimate
  - AlphaRobustSGD      : our method — live p̂ estimation, no oracle needed

Key design choices:
  - SGD-based optimizers use --lr (default 1e-2); AdamW uses --lr_adamw (1e-4)
    because AdamW's adaptive 1/sqrt(v) scaling makes its effective step ~100x larger
  - noise_alpha controls tail heaviness: 1.2 (very heavy) / 1.5 / 1.8 (lighter)
  - noise_scale must exceed clean gradient norms (~3 for GPT-2) so the Hill
    estimator can detect the Pareto tail; use noise_scale=5.0 for cluster runs

Usage:
  # Smoke test (pipeline check, ~1 min):
  python3 experiments/phase2_gpt2.py --smoke

  # Local run with noise:
  python3 experiments/phase2_gpt2.py --pretrained --noise_alpha 1.5

  # Cluster sweep (3 separate jobs):
  python3 experiments/phase2_gpt2.py --pretrained --noise_alpha 1.2 --steps 5000 ...
  python3 experiments/phase2_gpt2.py --pretrained --noise_alpha 1.5 --steps 5000 ...
  python3 experiments/phase2_gpt2.py --pretrained --noise_alpha 1.8 --steps 5000 ...
"""

import argparse
import os, sys, csv, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimizers.torch import AlphaRobustSGD, OracleClippedSGD, ZClip, NormalizedSGD

os.environ["TOKENIZERS_PARALLELISM"] = "false"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase2")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke',        action='store_true',
                   help='Fast smoke test: tiny data, few steps')
    p.add_argument('--pretrained',   action='store_true', default=False,
                   help='Load GPT-2 small pretrained weights (required for real results)')
    p.add_argument('--steps',        type=int,   default=500)
    p.add_argument('--pilot_steps',  type=int,   default=100,
                   help='Steps for AlphaRobust pilot to estimate pseudo-oracle p')
    p.add_argument('--batch_size',   type=int,   default=4)
    p.add_argument('--seq_len',      type=int,   default=128)
    p.add_argument('--lr',           type=float, default=1e-2,
                   help='LR for SGD-based optimizers (VanillaSGD, Clipped, AlphaRobust)')
    p.add_argument('--lr_adamw',     type=float, default=1e-4,
                   help='LR for AdamW (adaptive optimizer, needs separate tuning)')
    p.add_argument('--window_size',  type=int,   default=100)
    p.add_argument('--tau_scale',    type=float, default=5.0)
    p.add_argument('--n_seeds',      type=int,   default=3)
    p.add_argument('--noise_alpha',  type=float, default=1.5,
                   help='Pareto tail index for injected gradient noise (1.2/1.5/1.8)')
    p.add_argument('--noise_scale',  type=float, default=1.0,
                   help='Scale of injected isotropic Pareto noise (sets Pareto minimum)')
    p.add_argument('--device',       type=str,   default='auto')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------

def inject_pareto_noise(model, noise_alpha: float, noise_scale: float):
    """
    Inject isotropic Pareto(noise_alpha) noise into gradients.

    Matches Phase 1 noise model exactly:
      - direction: uniform on the unit sphere in R^d (via normalized Gaussian)
      - magnitude: Pareto(noise_alpha) * noise_scale
      - sign: Rademacher (±1) for zero-mean noise

    This ensures the global gradient noise norm has tail index = noise_alpha,
    which is what the Hill estimator is designed to recover.

    Per-element Pareto would create ||noise|| ~ d^{1/alpha} which is
    catastrophically large for 117M params.
    """
    with torch.no_grad():
        # 1. Sample Pareto magnitude (scalar)
        u = max(torch.rand(1).item(), 1e-8)
        magnitude = noise_scale * (u ** (-1.0 / noise_alpha))
        sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0

        # 2. Random direction: sample Gaussian for each param, then normalize
        direction_parts = []
        total_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                d = torch.randn_like(param.grad)
                direction_parts.append((param, d))
                total_sq += d.norm(2).item() ** 2

        dir_norm = total_sq ** 0.5
        if dir_norm < 1e-12:
            return

        # 3. Add noise: sign * magnitude * (direction / ||direction||)
        scale = sign * magnitude / dir_norm
        for param, d in direction_parts:
            param.grad.add_(d, alpha=scale)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    def __init__(self, token_ids: np.ndarray, seq_len: int):
        n_chunks = len(token_ids) // (seq_len + 1)
        token_ids = token_ids[: n_chunks * (seq_len + 1)]
        chunks = token_ids.reshape(n_chunks, seq_len + 1)
        self.inputs  = torch.tensor(chunks[:, :-1], dtype=torch.long)
        self.targets = torch.tensor(chunks[:, 1:],  dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def load_data(seq_len: int, max_tokens: int = None):
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset

    print("Loading WikiText-2...")
    ds  = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    text = "\n\n".join(ds["text"])
    ids  = tok.encode(text)
    if max_tokens:
        ids = ids[:max_tokens]
    return np.array(ids, dtype=np.int32), tok.vocab_size


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(device: torch.device, smoke: bool, pretrained: bool = False):
    from transformers import GPT2Config, GPT2LMHeadModel

    if smoke:
        cfg = GPT2Config(vocab_size=50257, n_positions=256,
                         n_embd=128, n_layer=2, n_head=2)
        model = GPT2LMHeadModel(cfg)
        print("Using tiny GPT-2 (2L/128d) [random init, smoke only]")
    elif pretrained:
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        print("Using GPT-2 small (117M) [pretrained weights]")
    else:
        cfg = GPT2Config.from_pretrained("gpt2")
        model = GPT2LMHeadModel(cfg)
        print("Using GPT-2 small (117M) [random init — use --pretrained for real results]")

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f}M")
    return model


# ---------------------------------------------------------------------------
# Pilot: estimate pseudo-oracle p̂
# ---------------------------------------------------------------------------

def estimate_p_pilot(model_state, loader_iter, args, device):
    """
    Run pilot_steps of AlphaRobust (with noise injection) on a fresh model copy.
    Returns (p_pseudo, sigma_pseudo):
      - p_pseudo: mean(p̂) over last 50 pilot steps
      - sigma_pseudo: median gradient norm from pilot (for Oracle's sigma)
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    if args.smoke:
        cfg = GPT2Config(vocab_size=50257, n_positions=256,
                         n_embd=128, n_layer=2, n_head=2)
        pilot_model = GPT2LMHeadModel(cfg).to(device)
    elif args.pretrained:
        pilot_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    else:
        cfg = GPT2Config.from_pretrained("gpt2")
        pilot_model = GPT2LMHeadModel(cfg).to(device)

    pilot_model.load_state_dict(model_state)
    pilot_opt = AlphaRobustSGD(
        pilot_model.parameters(), lr=args.lr,
        window_size=args.window_size, schedule='fixed',
        tau_scale=args.tau_scale,
    )

    pilot_model.train()
    for _ in range(args.pilot_steps):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        pilot_opt.zero_grad()
        out = pilot_model(x, labels=y)
        out.loss.backward()
        inject_pareto_noise(pilot_model, args.noise_alpha, args.noise_scale)
        pilot_opt.step()

    p_hats   = pilot_opt.p_hat_history
    valid    = p_hats[~np.isnan(p_hats)]
    p_pseudo = float(np.mean(valid[-50:])) if len(valid) >= 50 else float(np.mean(valid))
    p_pseudo = float(np.clip(p_pseudo, 1.01, 1.99))

    # Sigma from pilot: median gradient norm (matches AlphaRobust's internal sigma_hat)
    grad_norms = pilot_opt.grad_norm_history
    sigma_pseudo = float(np.median(grad_norms)) if len(grad_norms) > 0 else 1.0

    print(f"  Pilot p̂ estimate: {p_pseudo:.4f}  (pseudo-oracle p)")
    print(f"  Pilot σ̂ estimate: {sigma_pseudo:.4f}  (pseudo-oracle sigma)")
    return p_pseudo, sigma_pseudo


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_one(model, optimizer, loader, steps, device, opt_name, args):
    model.train()
    loader_iter = iter(loader)
    losses, grad_norms, p_hats, taus = [], [], [], []

    for t in range(1, steps + 1):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x, labels=y)
        loss = out.loss
        loss.backward()

        # Inject heavy-tailed noise into gradients
        inject_pareto_noise(model, args.noise_alpha, args.noise_scale)

        optimizer.step()

        losses.append(loss.item())

        gn = getattr(optimizer, 'grad_norm_history', None)
        grad_norms.append(float(gn[-1]) if gn is not None and len(gn) > 0 else float('nan'))

        ph = getattr(optimizer, 'p_hat_history', None)
        p_hats.append(float(ph[-1]) if ph is not None and len(ph) > 0 else float('nan'))

        th = getattr(optimizer, 'tau_history', None)
        taus.append(float(th[-1]) if th is not None and len(th) > 0 else float('nan'))

        if t % 50 == 0:
            ppl = np.exp(np.mean(losses[-50:]))
            print(f"    [{opt_name}] step {t}/{steps}  "
                  f"loss={losses[-1]:.4f}  ppl={ppl:.2f}")

    return {'loss': losses, 'grad_norm': grad_norms,
            'p_hat': p_hats, 'tau': taus}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    if args.smoke:
        args.steps       = 100
        args.pilot_steps = 30
        args.batch_size  = 2
        args.seq_len     = 64
        args.n_seeds     = 1
        args.pretrained  = False
        print("=== SMOKE TEST MODE (pipeline check only, results not meaningful) ===")

    print(f"Noise: Pareto(alpha={args.noise_alpha}), scale={args.noise_scale}")
    print(f"LR: SGD={args.lr}, AdamW={args.lr_adamw}")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Data
    max_tokens = 50_000 if args.smoke else None
    token_ids, vocab_size = load_data(args.seq_len, max_tokens)
    dataset = TokenDataset(token_ids, args.seq_len)
    print(f"Dataset: {len(dataset)} sequences of length {args.seq_len}")

    all_rows = []
    summary  = {}

    for seed in range(args.n_seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed+1}/{args.n_seeds}")
        print(f"{'='*60}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, drop_last=True,
                            generator=torch.Generator().manual_seed(seed))

        model      = load_model(device, args.smoke, args.pretrained)
        init_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Pilot: estimate pseudo-oracle p
        print(f"\nRunning pilot ({args.pilot_steps} steps) to estimate pseudo-oracle p...")
        pilot_loader = DataLoader(dataset, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True,
                                  generator=torch.Generator().manual_seed(seed + 1000))
        p_pseudo, sigma_pseudo = estimate_p_pilot(init_state, iter(pilot_loader),
                                                     args, device)

        # Optimizer lineup
        # SGD-based methods share --lr; AdamW uses --lr_adamw (different scale)
        optimizers = [
            ("VanillaSGD",
             lambda m: torch.optim.SGD(m.parameters(), lr=args.lr)),
            ("NormalizedSGD",
             lambda m: NormalizedSGD(m.parameters(), lr=args.lr,
                                     schedule='fixed')),
            ("ZClip",
             lambda m: ZClip(m.parameters(), lr=args.lr,
                             z_thresh=2.5, ema_alpha=0.01)),
            ("AdamW",
             lambda m: torch.optim.AdamW(m.parameters(), lr=args.lr_adamw,
                                         weight_decay=0.01)),
            ("PseudoOracleClipped",
             lambda m: OracleClippedSGD(m.parameters(), lr=args.lr,
                                        p=p_pseudo, sigma=sigma_pseudo,
                                        schedule='fixed', tau_scale=args.tau_scale)),
            ("AlphaRobustSGD",
             lambda m: AlphaRobustSGD(m.parameters(), lr=args.lr,
                                      window_size=args.window_size,
                                      schedule='fixed', tau_scale=args.tau_scale)),
        ]

        for opt_name, opt_fn in optimizers:
            print(f"\n--- {opt_name} ---")
            m = load_model(device, args.smoke, args.pretrained)
            m.load_state_dict(init_state)
            opt = opt_fn(m)

            t0      = time.time()
            result  = run_one(m, opt, loader, args.steps, device, opt_name, args)
            elapsed = time.time() - t0

            final_loss = float(np.mean(result['loss'][-50:]))
            final_ppl  = float(np.exp(final_loss))
            summary.setdefault(opt_name, []).append(final_loss)

            print(f"  Done in {elapsed:.1f}s | "
                  f"final_loss={final_loss:.4f} | final_ppl={final_ppl:.2f}")

            for t_idx in range(args.steps):
                all_rows.append({
                    'seed':        seed,
                    'optimizer':   opt_name,
                    'step':        t_idx + 1,
                    'loss':        result['loss'][t_idx],
                    'perplexity':  np.exp(result['loss'][t_idx]),
                    'grad_norm':   result['grad_norm'][t_idx],
                    'p_hat':       result['p_hat'][t_idx],
                    'tau':         result['tau'][t_idx],
                    'p_pseudo':    p_pseudo,
                    'noise_alpha': args.noise_alpha,
                })

    _write_csv(all_rows, args)
    _write_summary(summary, args)
    _plot(all_rows, args)
    print(f"\nAll outputs -> {RESULTS_DIR}/")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _tag(args):
    if args.smoke:
        return "smoke"
    tag = f"alpha{args.noise_alpha}"
    if args.window_size != 100:
        tag += f"_W{args.window_size}"
    if args.tau_scale != 5.0:
        tag += f"_tau{args.tau_scale}"
    return tag


def _write_csv(rows, args):
    tag  = _tag(args)
    path = os.path.join(RESULTS_DIR, f"raw_results_{tag}.csv")
    fields = ['seed','optimizer','step','loss','perplexity',
              'grad_norm','p_hat','tau','p_pseudo','noise_alpha']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  raw_results_{tag}.csv  ({len(rows)} rows)")


def _write_summary(summary, args):
    tag  = _tag(args)
    path = os.path.join(RESULTS_DIR, f"summary_{tag}.csv")
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['optimizer','mean_final_loss','std_final_loss','mean_final_ppl'])
        for opt_name, losses in sorted(summary.items()):
            w.writerow([opt_name,
                        f"{np.mean(losses):.4f}",
                        f"{np.std(losses):.4f}",
                        f"{np.exp(np.mean(losses)):.2f}"])
    print(f"  summary_{tag}.csv")


def _plot(rows, args):
    import matplotlib.pyplot as plt
    import matplotlib; matplotlib.rcParams.update({'font.size': 9})

    tag = _tag(args)

    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[r['optimizer']]['loss'].append((r['step'], r['loss']))
        data[r['optimizer']]['p_hat'].append((r['step'], r['p_hat']))

    colors = {
        'VanillaSGD':           '#d62728',
        'NormalizedSGD':        '#9467bd',
        'ZClip':                '#8c564b',
        'AdamW':                '#ff7f0e',
        'PseudoOracleClipped':  '#2ca02c',
        'AlphaRobustSGD':       '#1f77b4',
    }
    lws = {'PseudoOracleClipped': 2.2, 'AlphaRobustSGD': 2.2,
           'VanillaSGD': 1.4, 'AdamW': 1.4, 'NormalizedSGD': 1.4, 'ZClip': 1.4}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for opt_name, color in colors.items():
        if opt_name not in data:
            continue
        pts     = sorted(data[opt_name]['loss'])
        steps_  = [p[0] for p in pts]
        losses  = [p[1] for p in pts]
        k       = max(1, len(losses) // 50)
        smoothed = np.convolve(losses, np.ones(k)/k, mode='valid')
        ax1.semilogy(steps_[:len(smoothed)], smoothed,
                     label=opt_name, color=color,
                     linewidth=lws.get(opt_name, 1.4))

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title(f'Phase 2: GPT-2  |  Pareto noise α={args.noise_alpha}')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    if 'AlphaRobustSGD' in data:
        pts    = sorted(data['AlphaRobustSGD']['p_hat'])
        steps_ = [p[0] for p in pts]
        p_hats = [p[1] for p in pts]
        valid  = [(s, v) for s, v in zip(steps_, p_hats) if not np.isnan(v)]
        if valid:
            vs, vp = zip(*valid)
            ax2.plot(vs, vp, color='#1f77b4', linewidth=1.6, label='p̂ (AlphaRobust)')
            p_pseudo = rows[0]['p_pseudo']
            ax2.axhline(p_pseudo, color='#2ca02c', linestyle='--', linewidth=1.4,
                        label=f'pseudo-oracle p = {p_pseudo:.3f}')
            ax2.axhline(args.noise_alpha - 0.05, color='#d62728', linestyle=':',
                        linewidth=1.2, label=f'true p ≈ {args.noise_alpha - 0.05:.2f}')

    ax2.set_xlabel('Step')
    ax2.set_ylabel('p̂')
    ax2.set_ylim(0.8, 2.2)
    ax2.set_title('Live p̂ Estimate (AlphaRobust)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Phase 2: GPT-2 + Pareto(α={args.noise_alpha}) noise  [{tag}]',
                 fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, f'phase2_{tag}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  figures/phase2_{tag}.png")


if __name__ == "__main__":
    main()

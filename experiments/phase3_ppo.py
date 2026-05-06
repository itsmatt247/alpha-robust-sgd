"""
Phase 3: PPO on MuJoCo continuous control with pluggable optimizers.

Framing:
  Policy gradient methods in RL exhibit naturally heavy-tailed gradient noise
  (Garg et al., ICML 2021). Unlike Phase 2, NO artificial noise injection is
  needed — the heavy tails arise organically from the RL objective.

  This makes Phase 3 the cleanest test of AlphaRobust: does the Hill estimator
  detect real heavy-tailed gradients (not injected ones) and does adaptive
  clipping improve training?

Primary comparison:
  - VanillaSGD          : no clipping, expected to be unstable
  - NormalizedSGD       : discards gradient magnitude (Hübler 2025, Liu 2025)
  - ZClip               : EMA-based Z-score clipping (Kumar et al. 2025)
  - Adam                : standard RL optimizer
  - PseudoOracleClipped : ClippedSGD with p̂ from pilot
  - AlphaRobustSGD      : our method — live p̂ estimation

Environments:
  - HalfCheetah-v4   : locomotion, 17-dim obs, 6-dim act
  - Hopper-v4        : locomotion, 11-dim obs, 3-dim act
  - Walker2d-v4      : locomotion, 17-dim obs, 6-dim act

Usage:
  # Smoke test:
  python3 experiments/phase3_ppo.py --smoke

  # Single environment:
  python3 experiments/phase3_ppo.py --env HalfCheetah-v4 --total_timesteps 1000000

  # Cluster sweep (one job per env):
  python3 experiments/phase3_ppo.py --env HalfCheetah-v4 --total_timesteps 1000000
  python3 experiments/phase3_ppo.py --env Hopper-v4 --total_timesteps 1000000
  python3 experiments/phase3_ppo.py --env Walker2d-v4 --total_timesteps 1000000
"""

import argparse
import os, sys, csv, time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimizers.torch import AlphaRobustSGD, OracleClippedSGD, ZClip, NormalizedSGD

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "phase3")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true',
                   help='Fast smoke test: tiny network, few steps')
    p.add_argument('--env', type=str, default='HalfCheetah-v4',
                   choices=['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4', 'Humanoid-v4'])
    p.add_argument('--total_timesteps', type=int, default=1_000_000)
    p.add_argument('--n_seeds', type=int, default=5)
    p.add_argument('--lr', type=float, default=3e-4,
                   help='LR for SGD-based optimizers')
    p.add_argument('--lr_adam', type=float, default=3e-4,
                   help='LR for Adam')
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--gae_lambda', type=float, default=0.95)
    p.add_argument('--clip_eps', type=float, default=0.2,
                   help='PPO clipping epsilon (policy loss)')
    p.add_argument('--n_steps', type=int, default=2048,
                   help='Steps per rollout')
    p.add_argument('--n_epochs', type=int, default=10,
                   help='PPO epochs per rollout')
    p.add_argument('--batch_size', type=int, default=64,
                   help='Minibatch size for PPO updates')
    p.add_argument('--ent_coef', type=float, default=0.0)
    p.add_argument('--vf_coef', type=float, default=0.5)
    p.add_argument('--max_grad_norm', type=float, default=0.5,
                   help='Max grad norm for vanilla PPO (not used by AlphaRobust)')
    p.add_argument('--window_size', type=int, default=100)
    p.add_argument('--tau_scale', type=float, default=5.0)
    p.add_argument('--pilot_updates', type=int, default=50,
                   help='PPO updates for pilot to estimate pseudo-oracle p')
    p.add_argument('--no_adv_norm', action='store_true',
                   help='Disable advantage normalization — exposes raw heavy-tailed '
                        'gradient signal. Required for AlphaRobust to operate in its '
                        'intended regime (Garg et al. 2021).')
    p.add_argument('--device', type=str, default='auto')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Actor-Critic Network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Separate actor and critic MLPs with continuous action space."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        mean = self.actor_mean(obs)
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(obs).squeeze(-1)
        return mean, std, value

    def get_action(self, obs):
        mean, std, value = self(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate(self, obs, actions):
        mean, std, value = self(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values + [last_value])

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    def get_tensors(self, advantages, returns, device, normalize_adv=True):
        obs = torch.tensor(np.array(self.obs), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device)
        adv = torch.tensor(advantages, dtype=torch.float32, device=device)
        ret = torch.tensor(returns, dtype=torch.float32, device=device)
        if normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return obs, actions, old_log_probs, adv, ret

    def clear(self):
        self.__init__()


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(model, optimizer, buffer, last_value, args, device,
               use_builtin_clip=False):
    """
    Run PPO epochs over the collected rollout.

    Args:
        use_builtin_clip: If True, apply torch.nn.utils.clip_grad_norm_
            (for VanillaSGD and Adam). AlphaRobust/Oracle/ZClip/NSGD handle
            clipping internally, so this is False for them.

    Returns:
        dict with pg_loss, vf_loss, entropy, grad_norms
    """
    advantages, returns = buffer.compute_gae(last_value, args.gamma, args.gae_lambda)
    obs, actions, old_log_probs, adv, ret = buffer.get_tensors(
        advantages, returns, device, normalize_adv=not args.no_adv_norm
    )

    n_samples = len(obs)
    grad_norms_epoch = []

    for _ in range(args.n_epochs):
        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, args.batch_size):
            end = start + args.batch_size
            mb_idx = indices[start:end]

            mb_obs = obs[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_adv = adv[mb_idx]
            mb_ret = ret[mb_idx]

            new_log_probs, entropy, values = model.evaluate(mb_obs, mb_actions)

            # Policy loss (PPO clip)
            ratio = (new_log_probs - mb_old_log_probs).exp()
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * mb_adv
            pg_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            vf_loss = 0.5 * (values - mb_ret).pow(2).mean()

            # Entropy bonus
            entropy_loss = -entropy.mean()

            loss = pg_loss + args.vf_coef * vf_loss + args.ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()

            # Track gradient norm before any clipping
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms_epoch.append(total_norm)

            if use_builtin_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()

        # Flush deferred Hill update once per epoch (not per minibatch).
        # Minibatches within an epoch share the same rollout data (correlated),
        # so we feed one median norm per epoch — giving 10 updates per rollout
        # instead of 1, with sufficient inter-epoch independence.
        if hasattr(optimizer, 'flush_epoch_update'):
            optimizer.flush_epoch_update()

    return {
        'grad_norms': grad_norms_epoch,
    }


# ---------------------------------------------------------------------------
# Collect rollout
# ---------------------------------------------------------------------------

def collect_rollout(env, model, n_steps, device):
    buffer = RolloutBuffer()
    obs, _ = env.reset()
    episode_rewards = []
    current_ep_reward = 0.0

    for _ in range(n_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = model.get_action(obs_t)

        action_np = action.squeeze(0).cpu().numpy()
        log_prob_np = log_prob.item()
        value_np = value.item()

        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        buffer.add(obs, action_np, log_prob_np, reward, float(done), value_np)
        current_ep_reward += reward

        if done:
            episode_rewards.append(current_ep_reward)
            current_ep_reward = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

    # Last value for GAE
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _, _, last_value = model(obs_t)
    last_value = last_value.item()

    return buffer, last_value, episode_rewards


# ---------------------------------------------------------------------------
# Pilot: estimate pseudo-oracle p̂ for this environment
# ---------------------------------------------------------------------------

def estimate_p_pilot(env_name, model_state, args, device):
    """
    Run a short pilot with AlphaRobust to estimate p̂ from real RL gradients.
    Returns (p_pseudo, sigma_pseudo).
    """
    import gymnasium as gym
    pilot_env = gym.make(env_name)

    obs_dim = pilot_env.observation_space.shape[0]
    act_dim = pilot_env.action_space.shape[0]
    hidden = 32 if args.smoke else 64

    pilot_model = ActorCritic(obs_dim, act_dim, hidden).to(device)
    pilot_model.load_state_dict(model_state)

    pilot_opt = AlphaRobustSGD(
        pilot_model.parameters(), lr=args.lr,
        window_size=args.window_size, schedule='fixed',
        tau_scale=args.tau_scale,
        deferred_hill=True,
    )

    # Collect rollouts and do PPO updates
    for _ in range(args.pilot_updates):
        buffer, last_value, _ = collect_rollout(
            pilot_env, pilot_model, args.n_steps, device
        )
        ppo_update(pilot_model, pilot_opt, buffer, last_value, args, device)
        buffer.clear()

    pilot_env.close()

    p_hats = pilot_opt.p_hat_history
    valid = p_hats[~np.isnan(p_hats)]
    p_pseudo = float(np.mean(valid[-50:])) if len(valid) >= 50 else float(np.mean(valid)) if len(valid) > 0 else 1.5
    p_pseudo = float(np.clip(p_pseudo, 1.01, 1.99))

    grad_norms = pilot_opt.grad_norm_history
    sigma_pseudo = float(np.median(grad_norms)) if len(grad_norms) > 0 else 1.0

    print(f"  Pilot p̂ estimate: {p_pseudo:.4f}  (pseudo-oracle p)")
    print(f"  Pilot σ̂ estimate: {sigma_pseudo:.4f}  (pseudo-oracle sigma)")
    return p_pseudo, sigma_pseudo


# ---------------------------------------------------------------------------
# Main training loop for one optimizer
# ---------------------------------------------------------------------------

def run_one(env_name, model_state, opt_name, opt_fn, args, device):
    """Train PPO with a specific optimizer. Returns metrics dict."""
    import gymnasium as gym
    env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    hidden = 32 if args.smoke else 64

    model = ActorCritic(obs_dim, act_dim, hidden).to(device)
    model.load_state_dict(model_state)
    optimizer = opt_fn(model)

    # Determine if we need manual grad clipping (for vanilla optimizers)
    use_builtin_clip = opt_name in ('VanillaSGD', 'Adam')

    total_steps = 0
    n_updates = args.total_timesteps // args.n_steps
    all_episode_rewards = []
    all_grad_norms = []
    reward_per_update = []

    for update_i in range(1, n_updates + 1):
        buffer, last_value, ep_rewards = collect_rollout(
            env, model, args.n_steps, device
        )
        all_episode_rewards.extend(ep_rewards)
        total_steps += args.n_steps

        update_info = ppo_update(
            model, optimizer, buffer, last_value, args, device,
            use_builtin_clip=use_builtin_clip,
        )
        all_grad_norms.extend(update_info['grad_norms'])

        # Mean reward over episodes in this rollout
        mean_reward = np.mean(ep_rewards) if ep_rewards else float('nan')
        reward_per_update.append(mean_reward)

        if update_i % 10 == 0:
            recent = all_episode_rewards[-20:] if len(all_episode_rewards) >= 20 else all_episode_rewards
            avg_r = np.mean(recent) if recent else float('nan')
            print(f"    [{opt_name}] update {update_i}/{n_updates}  "
                  f"avg_reward={avg_r:.1f}  episodes={len(all_episode_rewards)}")

        buffer.clear()

    env.close()

    # Collect p̂ and tau histories if available
    p_hats = getattr(optimizer, 'p_hat_history', np.array([]))
    tau_hist = getattr(optimizer, 'tau_history', np.array([]))

    return {
        'episode_rewards': all_episode_rewards,
        'reward_per_update': reward_per_update,
        'grad_norms': all_grad_norms,
        'p_hats': p_hats if len(p_hats) > 0 else None,
        'taus': tau_hist if len(tau_hist) > 0 else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    if args.smoke:
        args.total_timesteps = 20_000
        args.n_steps = 256
        args.n_epochs = 4
        args.batch_size = 32
        args.n_seeds = 2
        args.pilot_updates = 5
        args.env = 'HalfCheetah-v4'
        print("=== SMOKE TEST MODE ===")

    import gymnasium as gym

    print(f"Environment: {args.env}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"LR: SGD={args.lr}, Adam={args.lr_adam}")
    print(f"Seeds: {args.n_seeds}")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Get env dims for model creation
    tmp_env = gym.make(args.env)
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.shape[0]
    tmp_env.close()
    hidden = 32 if args.smoke else 64
    print(f"Obs dim: {obs_dim}, Act dim: {act_dim}")

    all_rows = []
    summary = {}

    for seed in range(args.n_seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed+1}/{args.n_seeds}")
        print(f"{'='*60}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create initial model (shared starting point)
        model = ActorCritic(obs_dim, act_dim, hidden).to(device)
        init_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Pilot: estimate pseudo-oracle p
        print(f"\nRunning pilot ({args.pilot_updates} updates) to estimate pseudo-oracle p...")
        p_pseudo, sigma_pseudo = estimate_p_pilot(
            args.env, init_state, args, device
        )

        # Optimizer lineup
        optimizers = [
            ("VanillaSGD",
             lambda m: torch.optim.SGD(m.parameters(), lr=args.lr)),
            ("NormalizedSGD",
             lambda m: NormalizedSGD(m.parameters(), lr=args.lr,
                                     schedule='fixed')),
            ("ZClip",
             lambda m: ZClip(m.parameters(), lr=args.lr,
                             z_thresh=2.5, ema_alpha=0.01)),
            ("Adam",
             lambda m: torch.optim.Adam(m.parameters(), lr=args.lr_adam)),
            ("PseudoOracleClipped",
             lambda m: OracleClippedSGD(m.parameters(), lr=args.lr,
                                        p=p_pseudo, sigma=sigma_pseudo,
                                        schedule='fixed', tau_scale=args.tau_scale)),
            ("AlphaRobustSGD",
             lambda m: AlphaRobustSGD(m.parameters(), lr=args.lr,
                                      window_size=args.window_size,
                                      schedule='fixed', tau_scale=args.tau_scale,
                                      deferred_hill=True)),
        ]

        for opt_name, opt_fn in optimizers:
            print(f"\n--- {opt_name} ---")
            t0 = time.time()
            result = run_one(args.env, init_state, opt_name, opt_fn, args, device)
            elapsed = time.time() - t0

            ep_rewards = result['episode_rewards']
            final_reward = float(np.mean(ep_rewards[-20:])) if len(ep_rewards) >= 20 else float(np.mean(ep_rewards)) if ep_rewards else float('nan')
            summary.setdefault(opt_name, []).append(final_reward)

            print(f"  Done in {elapsed:.1f}s | "
                  f"final_reward={final_reward:.2f} | "
                  f"episodes={len(ep_rewards)}")

            # Log per-update rows
            for u_idx, r in enumerate(result['reward_per_update']):
                row = {
                    'seed': seed,
                    'optimizer': opt_name,
                    'update': u_idx + 1,
                    'mean_episode_reward': r,
                    'p_pseudo': p_pseudo,
                    'env': args.env,
                }

                # Add grad norm (mean per update)
                n_per_update = args.n_epochs * (args.n_steps // args.batch_size)
                gn_start = u_idx * n_per_update
                gn_end = min(gn_start + n_per_update, len(result['grad_norms']))
                if gn_start < len(result['grad_norms']):
                    row['mean_grad_norm'] = float(np.mean(result['grad_norms'][gn_start:gn_end]))
                else:
                    row['mean_grad_norm'] = float('nan')

                # Add p_hat if available
                if result['p_hats'] is not None and gn_start < len(result['p_hats']):
                    row['p_hat'] = float(result['p_hats'][min(gn_end - 1, len(result['p_hats']) - 1)])
                else:
                    row['p_hat'] = float('nan')

                all_rows.append(row)

    _write_csv(all_rows, args)
    _write_summary(summary, args)
    _plot(all_rows, summary, args)
    print(f"\nAll outputs -> {RESULTS_DIR}/")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _tag(args):
    if args.smoke:
        return "smoke"
    env_short = args.env.replace('-v4', '').lower()
    return env_short


def _write_csv(rows, args):
    tag = _tag(args)
    path = os.path.join(RESULTS_DIR, f"raw_results_{tag}.csv")
    fields = ['seed', 'optimizer', 'update', 'mean_episode_reward',
              'mean_grad_norm', 'p_hat', 'p_pseudo', 'env']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  raw_results_{tag}.csv  ({len(rows)} rows)")


def _write_summary(summary, args):
    tag = _tag(args)
    path = os.path.join(RESULTS_DIR, f"summary_{tag}.csv")
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['optimizer', 'mean_final_reward', 'std_final_reward'])
        for opt_name, rewards in sorted(summary.items()):
            w.writerow([opt_name,
                        f"{np.mean(rewards):.2f}",
                        f"{np.std(rewards):.2f}"])
    print(f"  summary_{tag}.csv")


def _plot(rows, summary, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.size': 9})

    tag = _tag(args)

    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[r['optimizer']]['reward'].append((r['update'], r['mean_episode_reward']))
        data[r['optimizer']]['p_hat'].append((r['update'], r['p_hat']))

    colors = {
        'VanillaSGD':           '#d62728',
        'NormalizedSGD':        '#9467bd',
        'ZClip':                '#8c564b',
        'Adam':                 '#ff7f0e',
        'PseudoOracleClipped':  '#2ca02c',
        'AlphaRobustSGD':       '#1f77b4',
    }
    lws = {'PseudoOracleClipped': 2.2, 'AlphaRobustSGD': 2.2,
           'VanillaSGD': 1.4, 'Adam': 1.4, 'NormalizedSGD': 1.4, 'ZClip': 1.4}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for opt_name, color in colors.items():
        if opt_name not in data:
            continue
        pts = sorted(data[opt_name]['reward'])
        updates = [p[0] for p in pts]
        rewards = [p[1] for p in pts]
        # Smooth
        k = max(1, len(rewards) // 50)
        smoothed = np.convolve(rewards, np.ones(k)/k, mode='valid')
        ax1.plot(updates[:len(smoothed)], smoothed,
                 label=opt_name, color=color,
                 linewidth=lws.get(opt_name, 1.4))

    ax1.set_xlabel('PPO Update')
    ax1.set_ylabel('Mean Episode Reward')
    ax1.set_title(f'Phase 3: PPO on {args.env}')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # p̂ tracking
    if 'AlphaRobustSGD' in data:
        pts = sorted(data['AlphaRobustSGD']['p_hat'])
        valid = [(s, v) for s, v in pts if not np.isnan(v)]
        if valid:
            vs, vp = zip(*valid)
            ax2.plot(vs, vp, color='#1f77b4', linewidth=1.6, label='p̂ (AlphaRobust)')
            if rows:
                p_pseudo = rows[0]['p_pseudo']
                ax2.axhline(p_pseudo, color='#2ca02c', linestyle='--', linewidth=1.4,
                            label=f'pseudo-oracle p = {p_pseudo:.3f}')

    ax2.set_xlabel('PPO Update')
    ax2.set_ylabel('p̂')
    ax2.set_ylim(0.8, 2.2)
    ax2.set_title('Live p̂ Estimate (AlphaRobust)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Phase 3: PPO on {args.env}', fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, f'phase3_{tag}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  figures/phase3_{tag}.png")


if __name__ == "__main__":
    main()

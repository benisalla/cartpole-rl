"""PPO with the clipped surrogate objective and GAE.

Collect a rollout (N envs x n steps) with the current policy, freeze its log-probabilities
log pi_old(a|s), then reuse the batch for several epochs of mini-batch updates on

  ratio  r = pi(a|s) / pi_old(a|s) = exp(log pi - log pi_old)
  L_clip = -mean( min( r * A,  clip(r, 1 - eps, 1 + eps) * A ) )
  L      = L_clip + c_v * mean((V(s) - R)^2) - c_ent * mean(H)

Why the clip: reusing a batch many times is what makes PPO sample efficient, but after a few
gradient steps the policy is no longer the one that collected the data and the advantage
estimates stop being trustworthy. The min(...) removes any incentive to push r outside
[1 - eps, 1 + eps]: a cheap stand-in for TRPO's KL trust region. Advantages are normalized
per mini-batch, and the learning rate decays linearly to 0 over the run.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from cartpole_rl.agents import to_tensor
from cartpole_rl.algorithms.rollout import RolloutCollector, gae
from cartpole_rl.env import VecEnv


def ppo_loss(model, obs, actions, old_logp, adv, ret, clip: float, value_coef: float, entropy_coef: float):
    logits, v = model(obs)
    dist = Categorical(logits=logits)
    logp = dist.log_prob(actions)
    ratio = torch.exp(logp - old_logp)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    surrogate = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
    pg_loss = -surrogate.mean()
    v_loss = (v - ret).pow(2).mean()
    entropy = dist.entropy().mean()
    clip_frac = ((ratio - 1).abs() > clip).float().mean()
    return pg_loss + value_coef * v_loss - entropy_coef * entropy, {
        "policy_loss": pg_loss.item(), "value_loss": v_loss.item(), "entropy": entropy.item(),
        "clip_fraction": clip_frac.item(),
    }


def learn(cfg, agent, rec):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    venv = VecEnv(cfg.n_envs, cfg.obs, seed=cfg.seed, frame_height=cfg.frame_height, frame_width=cfg.frame_width)
    model = agent.model
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    collector = RolloutCollector(venv, model, rec, cfg.gamma)
    while not rec.done:
        for group in opt.param_groups:  # linear learning-rate decay
            group["lr"] = cfg.lr * max(0.0, 1 - rec.steps / cfg.total_steps)
        b = collector.collect(cfg.n_steps)
        adv, ret = gae(b["rewards"], b["values"], b["dones"], b["last_value"], cfg.gamma, cfg.gae_lambda)
        obs = to_tensor(b["obs"].reshape(-1, *b["obs"].shape[2:]))
        actions = torch.as_tensor(b["actions"].reshape(-1))
        old_logp = torch.as_tensor(b["logp"].reshape(-1))
        adv_t, ret_t = torch.as_tensor(adv.reshape(-1)), torch.as_tensor(ret.reshape(-1))
        n = len(actions)
        mb = min(cfg.minibatch_size, n)
        for _ in range(cfg.ppo_epochs):
            perm = rng.permutation(n)
            for start in range(0, n, mb):
                i = torch.as_tensor(perm[start:start + mb])
                loss, info = ppo_loss(model, obs[i], actions[i], old_logp[i], adv_t[i], ret_t[i],
                                      cfg.clip, cfg.value_coef, cfg.entropy_coef)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
        rec.log(**info)
    venv.close()

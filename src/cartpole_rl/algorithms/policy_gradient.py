"""REINFORCE (Monte Carlo policy gradient), with and without a learned baseline.

Play one full episode with the stochastic policy pi_theta(a|s), then compute the
reward-to-go G_t = sum_{k>=t} gamma^{k-t} r_k for every step and ascend

  grad J = E[ sum_t grad log pi(a_t|s_t) * A_t ]

  reinforce           A_t = normalize(G_t)                 (mean 0, std 1 within the episode)
  reinforce-baseline  A_t = normalize(G_t - V_phi(s_t))     and V_phi is fit to G_t by MSE

Loss actually minimised:  L_pi = -mean_t(log pi(a_t|s_t) * A_t) - c_ent * mean_t H[pi(.|s_t)]

Why subtract a baseline: any b(s) leaves the gradient unbiased (E[grad log pi * b(s)] = 0) but a
good one, the expected return from s, shrinks the variance: the update now asks "was this
action better than usual from here?" instead of "was the episode long?".

Exploration comes from sampling the policy itself; no epsilon is needed (the original notebook
mixed in an epsilon-random action, which makes the logged log-probabilities wrong for the
action taken; see legacy/README.md).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from cartpole_rl.agents import to_tensor
from cartpole_rl.env import make_env


def rewards_to_go(rewards, gamma: float) -> np.ndarray:
    """G_t = r_t + gamma * G_{t+1}, computed backwards in one pass."""
    G, out = 0.0, np.zeros(len(rewards), dtype=np.float32)
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        out[t] = G
    return out


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + 1e-8) if len(x) > 1 else x - x.mean()


def learn(cfg, agent, rec):
    env = make_env(cfg.obs, seed=cfg.seed, frame_height=cfg.frame_height, frame_width=cfg.frame_width)
    model = agent.model
    baseline = cfg.method == "reinforce-baseline"
    opt = torch.optim.Adam(model.actor_parameters(), lr=cfg.lr)
    opt_v = torch.optim.Adam(model.critic_parameters(), lr=cfg.critic_lr) if baseline else None
    torch.manual_seed(cfg.seed)

    ep = 0
    while not rec.done:
        obs, _ = env.reset(seed=cfg.seed + ep)
        ep += 1
        observations, actions, rewards, done = [], [], [], False
        while not done:
            with torch.no_grad():
                a = int(Categorical(logits=model.logits(to_tensor(obs)[None])).sample())
            observations.append(obs)
            actions.append(a)
            obs, r, term, trunc, _ = env.step(a)
            rewards.append(r)
            done = term or trunc
            rec.step()

        # one update per episode, on the whole trajectory at once
        x = to_tensor(np.stack(observations))
        acts = torch.as_tensor(actions)
        G = torch.as_tensor(rewards_to_go(rewards, cfg.gamma))
        dist = Categorical(logits=model.logits(x))
        logp = dist.log_prob(acts)
        if baseline:
            v = model.value(x)
            adv = normalize(G - v.detach())
            v_loss = ((v - G) ** 2).mean()
            opt_v.zero_grad()
            v_loss.backward()
            opt_v.step()
        else:
            adv = normalize(G)
        loss = -(logp * adv).mean() - cfg.entropy_coef * dist.entropy().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        rec.episode(sum(rewards))
        rec.log(policy_loss=loss.item())
    env.close()

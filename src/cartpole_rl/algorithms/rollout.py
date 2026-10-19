"""Shared by A2C and PPO: collect a fixed-length rollout from a vector of envs, then turn it
into advantages with GAE.

GAE (generalized advantage estimation), for each env, backwards in time:

  delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t)        (one-step TD error)
  A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
  R_t     = A_t + V(s_t)                                            (target for the critic)

lambda = 0 gives the one-step TD advantage (low variance, biased by V); lambda = 1 gives the
Monte Carlo return minus V (unbiased, high variance); values in between trade the two.

Truncation (the 500-step time limit) is not failure: in that case we add gamma * V(final_obs)
to the reward and then treat the step as done, so the value is bootstrapped, not zeroed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from cartpole_rl.agents import to_tensor


def gae(rewards, values, dones, last_value, gamma: float, lam: float):
    """rewards, values, dones: (T, N) arrays; last_value: (N,). Returns advantages, returns."""
    T = rewards.shape[0]
    adv = np.zeros_like(rewards)
    next_adv = np.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nonterminal * next_value - values[t]
        next_adv = delta + gamma * lam * nonterminal * next_adv
        adv[t] = next_adv
        next_value = values[t]
    return adv, adv + values


class RolloutCollector:
    def __init__(self, venv, model, rec, gamma: float):
        self.venv, self.model, self.rec, self.gamma = venv, model, rec, gamma
        self.obs = venv.reset()

    @torch.no_grad()
    def collect(self, n_steps: int) -> dict:
        n = self.venv.n
        shape = self.obs.shape[1:]
        buf = {
            "obs": np.zeros((n_steps, n, *shape), np.float32), "actions": np.zeros((n_steps, n), np.int64),
            "logp": np.zeros((n_steps, n), np.float32), "values": np.zeros((n_steps, n), np.float32),
            "rewards": np.zeros((n_steps, n), np.float32), "dones": np.zeros((n_steps, n), np.float32),
        }
        for t in range(n_steps):
            logits, v = self.model(to_tensor(self.obs))
            dist = Categorical(logits=logits)
            a = dist.sample()
            nxt, r, term, trunc, final_obs, finished = self.venv.step(a.numpy())
            if trunc.any():  # bootstrap the time-limit cut from V(final observation)
                v_final = self.model.value(to_tensor(final_obs[trunc]))
                r = r.copy()
                r[trunc] += self.gamma * v_final.numpy() * (~term[trunc])
            buf["obs"][t], buf["actions"][t] = self.obs, a.numpy()
            buf["logp"][t], buf["values"][t] = dist.log_prob(a).numpy(), v.numpy()
            buf["rewards"][t], buf["dones"][t] = r, (term | trunc)
            self.obs = nxt
            self.rec.step(n)
            for ret, _length in finished:
                self.rec.episode(ret)
        _, last_v = self.model(to_tensor(self.obs))
        buf["last_value"] = last_v.numpy()
        return buf

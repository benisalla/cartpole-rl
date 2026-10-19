"""Tabular control on the discretized state: Q-learning, SARSA, Monte Carlo control.

The state (4 floats) is cut into bins (``env.Discretizer``) so that Q(s, a) is a plain
numpy array of shape (n_states, 2). Actions are epsilon-greedy: with probability eps a random
action, else argmax_a Q(s, a); eps decays linearly from ``eps_start`` to ``eps_end``.

Update rules (alpha = step size, gamma = discount):

  Q-learning (off-policy TD):   Q(s,a) += alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
  SARSA      (on-policy TD):    Q(s,a) += alpha * [r + gamma * Q(s',a')      - Q(s,a)],  a' ~ eps-greedy
  MC control (no bootstrap):    after the episode, for every (s_t, a_t):
                                Q(s_t,a_t) += alpha * [G_t - Q(s_t,a_t)],  G_t = sum_k gamma^k r_{t+k}

Q-learning learns about the greedy policy while behaving eps-greedily; SARSA learns about the
eps-greedy policy it actually follows (so it is more careful near failure); MC waits for the
true return instead of trusting its own estimate of the next state (no bias, more variance).

At a termination (pole fell) the future value is 0. At a truncation (500 steps) the episode
was cut by the time limit, not by failure, so TD methods still bootstrap from s'.
"""

from __future__ import annotations

import numpy as np

from cartpole_rl.algorithms.schedules import linear
from cartpole_rl.env import make_env


def epsilon_greedy(q_row: np.ndarray, eps: float, rng: np.random.Generator) -> int:
    if rng.random() < eps:
        return int(rng.integers(len(q_row)))
    # break ties at random so an all-zero row does not always pick action 0
    best = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(best))


def learn(cfg, agent, rec):
    env = make_env("state", seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    Q, disc, g, alpha = agent.q, agent.disc, cfg.gamma, cfg.alpha
    eps = lambda: linear(cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps, rec.steps)  # noqa: E731

    ep = 0
    while not rec.done:
        obs, _ = env.reset(seed=cfg.seed + ep)
        ep += 1
        s = disc(obs)
        a = epsilon_greedy(Q[s], eps(), rng)
        trajectory, total, done = [], 0.0, False
        while not done:
            obs, r, term, trunc, _ = env.step(a)
            s2 = disc(obs)
            a2 = epsilon_greedy(Q[s2], eps(), rng)
            total += r
            done = term or trunc
            if cfg.method == "q-learning":
                target = r + (0.0 if term else g * Q[s2].max())
                Q[s, a] += alpha * (target - Q[s, a])
            elif cfg.method == "sarsa":
                target = r + (0.0 if term else g * Q[s2, a2])
                Q[s, a] += alpha * (target - Q[s, a])
            else:  # mc-control: store, update at the end of the episode
                trajectory.append((s, a, r))
            s, a = s2, a2
            rec.step()
            if rec.done:
                break
        if cfg.method == "mc-control":
            G = 0.0 if not trunc else Q[s].max()  # bootstrap a time-limit cut, like the TD methods
            for s_t, a_t, r_t in reversed(trajectory):
                G = r_t + g * G
                Q[s_t, a_t] += alpha * (G - Q[s_t, a_t])
        rec.episode(total)
        rec.log(epsilon=eps())
    env.close()

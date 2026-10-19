"""DQN and its variants (one loop, flags switch the ideas on):

  dqn          Q-network + replay buffer + target network
  double-dqn   --double: pick a' with the online net, evaluate it with the target net
  dueling-dqn  --dueling (+ double): Q = V(s) + A(s,a) - mean A   (see networks.QNet)
  --prioritized  prioritized replay;  --n-step N  n-step returns  (see buffers.py)

Loss for a batch of transitions (s, a, R, s', done, discount = gamma^n):

  y = R + discount * (1 - done) * Q_target(s', a*)
      a* = argmax_a Q_target(s', a)        (dqn: max over the target net, tends to overestimate)
      a* = argmax_a Q_online(s', a)        (double: decouple selecting from evaluating)
  L = mean_i  w_i * Huber(Q_online(s_i, a_i) - y_i)

Why a target network: y depends on the same weights we are changing; a slowly moving copy
(Polyak averaging, theta_target <- tau * theta + (1 - tau) * theta_target) keeps the target
from running away. Why Huber: it is quadratic near zero and linear for large errors, so one
surprising transition cannot produce a huge gradient.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from cartpole_rl.agents import to_tensor
from cartpole_rl.algorithms.schedules import linear
from cartpole_rl.buffers import NStepCollector, PrioritizedReplay, ReplayBuffer
from cartpole_rl.env import make_env


def dqn_loss(online, target, batch, gamma_unused=None, double: bool = False):
    """Returns (loss, td_errors). Kept as a function so tests can check it on a fixed batch."""
    obs, nxt = to_tensor(batch["obs"]), to_tensor(batch["next_obs"])
    actions = torch.as_tensor(batch["actions"])
    q = online(obs).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        q_next_target = target(nxt)
        if double:
            a_star = online(nxt).argmax(1, keepdim=True)
            q_next = q_next_target.gather(1, a_star).squeeze(1)
        else:
            q_next = q_next_target.max(1).values
        y = (torch.as_tensor(batch["rewards"])
             + torch.as_tensor(batch["discounts"]) * (1 - torch.as_tensor(batch["dones"])) * q_next)
    td = q - y
    loss = (torch.as_tensor(batch["weights"]) * F.smooth_l1_loss(q, y, reduction="none")).mean()
    return loss, td.detach().numpy()


def learn(cfg, agent, rec):
    env = make_env(cfg.obs, seed=cfg.seed, frame_height=cfg.frame_height, frame_width=cfg.frame_width)
    online = agent.model
    target = copy.deepcopy(online).requires_grad_(False)
    opt = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    obs_shape = env.observation_space.shape
    store = np.float16 if cfg.obs != "state" else np.float32  # pixels: halve the replay memory
    Buffer = PrioritizedReplay if cfg.prioritized else ReplayBuffer
    kw = {"alpha": cfg.per_alpha} if cfg.prioritized else {}
    buffer = Buffer(cfg.buffer_size, obs_shape, seed=cfg.seed, dtype=store, **kw)
    collector = NStepCollector(buffer, cfg.n_step, cfg.gamma)
    rng = np.random.default_rng(cfg.seed)

    ep = 0
    while not rec.done:
        obs, _ = env.reset(seed=cfg.seed + ep)
        ep += 1
        total, done = 0.0, False
        while not done and not rec.done:
            eps = linear(cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps, rec.steps)
            a = int(rng.integers(2)) if rng.random() < eps else agent.act(obs)
            nxt, r, term, trunc, _ = env.step(a)
            collector.add(obs, a, r, nxt, term, trunc)
            obs, total, done = nxt, total + r, term or trunc
            rec.step()

            if rec.steps >= cfg.learning_starts and rec.steps % cfg.train_every == 0:
                beta = linear(cfg.per_beta, 1.0, cfg.total_steps, rec.steps)
                batch = buffer.sample(cfg.batch_size, beta)
                loss, td = dqn_loss(online, target, batch, double=cfg.double)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                opt.step()
                buffer.update_priorities(batch["idx"], td)
                with torch.no_grad():  # Polyak update of the target network
                    for p_t, p in zip(target.parameters(), online.parameters()):
                        p_t.mul_(1 - cfg.tau).add_(cfg.tau * p)
                if rec.steps % 500 == 0:
                    rec.log(loss=loss.item(), epsilon=eps)
        rec.episode(total)
    env.close()

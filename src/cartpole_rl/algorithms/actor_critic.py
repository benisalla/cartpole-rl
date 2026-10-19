"""Actor-critic: a policy (actor) improved with advantages estimated by a value net (critic).

One-step actor-critic (``actor-critic``), updated after *every* step of a single env:

  delta = r + gamma * (1 - terminated) * V(s') - V(s)      the TD error is the advantage estimate
  L_critic = delta^2                                        V moves toward the bootstrapped target
  L_actor  = -log pi(a|s) * delta.detach() - c_ent * H      push up actions that beat expectation

A2C (``a2c``, synchronous advantage actor-critic): the same idea, but with N envs stepped in
parallel for n steps; advantages from GAE (lambda = 1 here: n-step returns), one gradient step
on the whole (n x N) batch with one combined loss:

  L = -mean(log pi(a|s) * A) + c_v * mean((V(s) - R)^2) - c_ent * mean(H[pi(.|s)])

Why the vector of envs: the N trajectories are at different moments of different episodes,
so the batch is far less correlated than n steps of one env; it plays the role that the
replay buffer plays for DQN, without storing anything.
"""

from __future__ import annotations

import torch
from torch.distributions import Categorical

from cartpole_rl.agents import to_tensor
from cartpole_rl.algorithms.rollout import RolloutCollector, gae
from cartpole_rl.env import VecEnv, make_env


def learn_one_step(cfg, agent, rec):
    env = make_env(cfg.obs, seed=cfg.seed, frame_height=cfg.frame_height, frame_width=cfg.frame_width)
    model = agent.model
    opt_pi = torch.optim.Adam(model.actor_parameters(), lr=cfg.lr)
    opt_v = torch.optim.Adam(model.critic_parameters(), lr=cfg.critic_lr)
    torch.manual_seed(cfg.seed)

    ep = 0
    while not rec.done:
        obs, _ = env.reset(seed=cfg.seed + ep)
        ep += 1
        total, done = 0.0, False
        while not done and not rec.done:
            x = to_tensor(obs)[None]
            logits, v = model(x)
            dist = Categorical(logits=logits)
            a = dist.sample()
            nxt, r, term, trunc, _ = env.step(int(a))
            with torch.no_grad():
                v_next = 0.0 if term else model.value(to_tensor(nxt)[None])
            delta = r + cfg.gamma * v_next - v
            critic_loss = delta.pow(2).mean()
            actor_loss = -(dist.log_prob(a) * delta.detach()).mean() - cfg.entropy_coef * dist.entropy().mean()
            opt_pi.zero_grad()
            opt_v.zero_grad()
            (actor_loss + critic_loss).backward()  # the two parameter sets are disjoint
            opt_pi.step()
            opt_v.step()
            obs, total, done = nxt, total + r, term or trunc
            rec.step()
        rec.episode(total)
    env.close()


def learn_a2c(cfg, agent, rec):
    torch.manual_seed(cfg.seed)
    venv = VecEnv(cfg.n_envs, cfg.obs, seed=cfg.seed, frame_height=cfg.frame_height, frame_width=cfg.frame_width)
    model = agent.model
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    collector = RolloutCollector(venv, model, rec, cfg.gamma)
    while not rec.done:
        b = collector.collect(cfg.n_steps)
        adv, ret = gae(b["rewards"], b["values"], b["dones"], b["last_value"], cfg.gamma, cfg.gae_lambda)
        obs = to_tensor(b["obs"].reshape(-1, *b["obs"].shape[2:]))
        logits, v = model(obs)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(torch.as_tensor(b["actions"].reshape(-1)))
        adv_t, ret_t = torch.as_tensor(adv.reshape(-1)), torch.as_tensor(ret.reshape(-1))
        pg_loss = -(logp * adv_t).mean()
        v_loss = (v - ret_t).pow(2).mean()
        entropy = dist.entropy().mean()
        loss = pg_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        opt.step()
        rec.log(policy_loss=pg_loss.item(), value_loss=v_loss.item(), entropy=entropy.item())
    venv.close()

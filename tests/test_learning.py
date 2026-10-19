"""Do the methods actually learn? Small budgets, a fixed seed, and a bar well above random
play (a random policy scores about 22 on CartPole)."""

import json

import pytest

from cartpole_rl import default_config, load_agent, train
from cartpole_rl.evaluate import evaluate_agent

RANDOM = 22.0


def quick(method, steps, tmp_out, **kw):
    cfg = default_config(method, total_steps=steps, eval_every=steps // 2, eval_episodes=5,
                         final_eval_episodes=5, output_dir=tmp_out, **kw)
    return train(cfg, verbose=False)


def test_q_learning_learns(tmp_out):
    r = quick("q-learning", 40_000, tmp_out)
    assert r.final_eval["mean"] > 5 * RANDOM


def test_ppo_learns(tmp_out):
    r = quick("ppo", 12_000, tmp_out)
    assert r.final_eval["mean"] > 5 * RANDOM


def test_reinforce_baseline_learns(tmp_out):
    r = quick("reinforce-baseline", 12_000, tmp_out)
    assert r.final_eval["mean"] > 3 * RANDOM


def test_dqn_overfits_one_batch():
    """The Q network can fit a fixed batch of targets (the supervised core of DQN)."""
    import copy

    import numpy as np
    import torch

    from cartpole_rl.algorithms.dqn import dqn_loss
    from cartpole_rl.networks import QNet

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    online = QNet((4,), 2, hidden=64)
    target = copy.deepcopy(online)
    batch = {"obs": rng.normal(size=(32, 4)).astype(np.float32),
             "next_obs": rng.normal(size=(32, 4)).astype(np.float32),
             "actions": rng.integers(0, 2, 32), "rewards": rng.normal(size=32).astype(np.float32),
             "dones": np.ones(32, np.float32), "discounts": np.full(32, 0.99, np.float32),
             "weights": np.ones(32, np.float32)}  # done=1: targets are the rewards, fixed
    opt = torch.optim.Adam(online.parameters(), 1e-2)
    for _ in range(300):
        loss, _ = dqn_loss(online, target, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 1e-3


@pytest.mark.parametrize("method", ["sarsa", "mc-control", "dqn", "double-dqn", "dueling-dqn", "reinforce",
                                    "actor-critic", "a2c"])
def test_every_method_runs_and_checkpoints(method, tmp_out):
    r = quick(method, 1_500, tmp_out, learning_starts=200)
    assert r.steps >= 1_500 and r.episodes > 0 and r.history["evals"]
    agent, cfg = load_agent(r.checkpoint)
    assert cfg.method == method
    again = evaluate_agent(agent, n_episodes=5, seed=100_000)
    assert again["mean"] == pytest.approx(r.final_eval["mean"])  # same weights, same seeds, same score
    with open(tmp_out + "/history.json") as f:
        assert json.load(f)["summary"]["method"] == method


def test_dqn_flags_prioritized_and_nstep(tmp_out):
    r = quick("dueling-dqn", 1_500, tmp_out, learning_starts=200, prioritized=True, n_step=3)
    assert r.config.prioritized and r.config.n_step == 3 and r.steps >= 1_500


@pytest.mark.parametrize("method,obs", [("ppo", "frame-diff"), ("reinforce", "frame"), ("dqn", "frame-diff")])
def test_pixel_runs(method, obs, tmp_out):
    cfg = default_config(method, obs=obs, total_steps=300, eval_every=300, eval_episodes=1, final_eval_episodes=1,
                         n_envs=2, n_steps=32, minibatch_size=32, ppo_epochs=1, learning_starts=100,
                         buffer_size=500, hidden=32, output_dir=tmp_out)
    r = train(cfg, verbose=False)
    assert r.steps >= 300 and r.final_eval["episodes"] == 1

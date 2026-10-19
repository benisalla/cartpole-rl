"""Exact checks of the pieces each update rule is built from."""

import numpy as np
import torch

from cartpole_rl.algorithms.dqn import dqn_loss
from cartpole_rl.algorithms.policy_gradient import rewards_to_go
from cartpole_rl.algorithms.ppo import ppo_loss
from cartpole_rl.algorithms.rollout import gae
from cartpole_rl.algorithms.schedules import linear
from cartpole_rl.buffers import NStepCollector, PrioritizedReplay, ReplayBuffer
from cartpole_rl.networks import ActorCritic, QNet


def test_rewards_to_go():
    np.testing.assert_allclose(rewards_to_go([1, 1, 1], 0.5), [1.75, 1.5, 1.0])


def test_gae_lambda_one_is_monte_carlo_minus_value():
    r = np.array([[1.0], [1.0], [1.0]], np.float32)
    v = np.array([[0.5], [0.2], [0.1]], np.float32)
    d = np.zeros_like(r)
    adv, ret = gae(r, v, d, np.array([2.0], np.float32), gamma=0.9, lam=1.0)
    G0 = 1 + 0.9 + 0.81 + 0.729 * 2.0
    np.testing.assert_allclose(ret[0, 0], G0, rtol=1e-6)
    np.testing.assert_allclose(adv[0, 0], G0 - 0.5, rtol=1e-6)


def test_gae_lambda_zero_is_td_error_and_done_cuts():
    r = np.array([[1.0], [1.0]], np.float32)
    v = np.array([[0.5], [0.3]], np.float32)
    d = np.array([[0.0], [1.0]], np.float32)
    adv, _ = gae(r, v, d, np.array([9.0], np.float32), gamma=0.9, lam=0.0)
    np.testing.assert_allclose(adv[:, 0], [1 + 0.9 * 0.3 - 0.5, 1 - 0.3], rtol=1e-6)  # 9.0 never used


def test_linear_schedule():
    assert linear(1.0, 0.0, 10, 0) == 1.0
    assert abs(linear(1.0, 0.0, 10, 5) - 0.5) < 1e-9
    assert linear(1.0, 0.0, 10, 50) == 0.0


def test_nstep_collector_returns_and_flush():
    buf = ReplayBuffer(10, (1,))
    col = NStepCollector(buf, n=3, gamma=0.5)
    for t in range(4):
        col.add([t], 0, 1.0, [t + 1], terminated=(t == 3), truncated=False)
    # 4 transitions in: t=0 and t=1 emitted as full 3-step, t=2 and t=3 flushed at termination
    assert len(buf) == 4
    np.testing.assert_allclose(buf.rewards[:4], [1.75, 1.75, 1.5, 1.0])
    np.testing.assert_allclose(buf.discounts[:4], [0.125, 0.125, 0.25, 0.5])
    np.testing.assert_array_equal(buf.dones[:4], [0, 1, 1, 1])


def test_prioritized_replay_prefers_large_errors():
    buf = PrioritizedReplay(100, (1,), alpha=1.0, seed=0)
    for i in range(100):
        buf.add([i], 0, 0.0, [i], False, 0.99)
    td = np.ones(100)
    td[7] = 50.0  # P(7) = 50 / 149, about a third, versus 1% under uniform sampling
    buf.update_priorities(np.arange(100), td)
    batch = buf.sample(512, beta=1.0)
    hot = batch["idx"] == 7
    assert 0.2 < hot.mean() < 0.5
    # importance weights undo the bias: the over-sampled transition gets the smallest weight
    assert batch["weights"].max() == 1.0
    np.testing.assert_allclose(batch["weights"][hot], 1 / 50, rtol=1e-4)


def test_double_dqn_target_uses_online_argmax():
    online, target = QNet((4,), 2), QNet((4,), 2)
    for net, w in ((online, [[0.0], [1.0]]), (target, [[5.0], [-1.0]])):  # constant outputs per action
        for p in net.parameters():
            torch.nn.init.zeros_(p)
        net.q.bias.data = torch.tensor(w).squeeze(1)
    batch = {"obs": np.zeros((1, 4), np.float32), "next_obs": np.zeros((1, 4), np.float32),
             "actions": np.array([0]), "rewards": np.array([1.0], np.float32),
             "dones": np.array([0.0], np.float32), "discounts": np.array([0.5], np.float32),
             "weights": np.ones(1, np.float32)}
    _, td_plain = dqn_loss(online, target, batch, double=False)   # y = 1 + 0.5 * max(5, -1) = 3.5
    _, td_double = dqn_loss(online, target, batch, double=True)   # online picks a=1 -> y = 1 + 0.5 * -1
    np.testing.assert_allclose(td_plain, [0.0 - 3.5])
    np.testing.assert_allclose(td_double, [0.0 - 0.5])


def test_dueling_head_mean_advantage_is_zero_offset():
    q = QNet((4,), 2, dueling=True)
    x = torch.randn(5, 4)
    h = q.body(x)
    np.testing.assert_allclose(q(x).mean(1).detach(), q.value(h).squeeze(1).detach(), atol=1e-6)


def test_ppo_clip_kills_gradient_outside_trust_region():
    torch.manual_seed(0)
    model = ActorCritic((4,), 2)
    obs = torch.randn(8, 4)
    actions = torch.zeros(8, dtype=torch.long)
    with torch.no_grad():
        logp_now = torch.distributions.Categorical(logits=model.logits(obs)).log_prob(actions)
    adv = torch.linspace(-1, 1, 8)  # signs survive the per-batch normalisation inside ppo_loss
    # good actions (A > 0) already made e times more likely, bad ones (A < 0) e times less:
    # both are past the clip range, so the objective is flat in both cases
    old = logp_now - torch.sign(adv) * 1.0
    loss, info = ppo_loss(model, obs, actions, old, adv, torch.zeros(8), clip=0.2, value_coef=0.0,
                          entropy_coef=0.0)
    loss.backward()
    assert info["clip_fraction"] == 1.0
    assert model.pi.weight.grad.abs().max() == 0  # clipped: no incentive to move further

# Testing

```bash
OMP_NUM_THREADS=1 pytest -q          # about 40 s on one CPU thread (1.5 min on a busy machine)
```

The tests are in `tests/` and run on the CPU with tiny budgets. They check three kinds of
things.

**The pieces of each update rule, against numbers worked out by hand**
(`test_algorithms_units.py`):

| test | what is checked |
|---|---|
| `test_rewards_to_go` | `G_t = r_t + gamma G_{t+1}` on a 3-step example |
| `test_gae_lambda_one_is_monte_carlo_minus_value` | GAE with lambda = 1 equals the bootstrapped Monte Carlo return minus V |
| `test_gae_lambda_zero_is_td_error_and_done_cuts` | lambda = 0 gives the TD error, and `done` cuts the bootstrap |
| `test_nstep_collector_returns_and_flush` | n-step returns, discounts `gamma^k`, tail flush at the end of an episode |
| `test_prioritized_replay_prefers_large_errors` | sampling frequency follows the priorities and importance weights undo it |
| `test_double_dqn_target_uses_online_argmax` | plain vs double DQN targets on two hand-built networks |
| `test_dueling_head_mean_advantage_is_zero_offset` | `mean_a Q(s,a) = V(s)` for the dueling head |
| `test_ppo_clip_kills_gradient_outside_trust_region` | no gradient when the ratio is past the clip in the direction of the advantage |
| `test_linear_schedule` | epsilon / learning-rate schedule |

**The environment and the networks** (`test_env.py`, `test_networks_agents.py`): seeding,
pixel shapes and value ranges, the difference channel is zero on reset and non-zero after
moving, the discretizer covers its range, the vector env auto-resets and reports episodes,
CNN / residual CNN output shapes, the actor and critic parameter sets are disjoint, every
method builds an agent that acts.

**Learning and plumbing** (`test_learning.py`, `test_cli_and_report.py`):

* Q-learning, PPO and REINFORCE + baseline reach several times the random-play return
  (about 22) within 12 000 to 40 000 steps,
* the Q network fits a fixed batch of targets to a loss below 1e-3 (the supervised core of DQN),
* every other method runs, saves a checkpoint, reloads it and scores exactly the same on the
  same evaluation seeds,
* the DQN prioritized + n-step flags, and pixel runs for PPO, REINFORCE and DQN, run end to end,
* `record_grid` (the side-by-side before / after GIF) tiles the panels and captions as expected,
* `cartpole-train`, `cartpole-eval`, `cartpole-play` (GIF) and `cartpole-compare` (table and
  figures) work from their `main` functions.

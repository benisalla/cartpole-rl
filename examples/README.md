# Examples

Short scripts that use the `cartpole_rl` package from Python, in the order one would use it.
Each runs on a CPU (one thread) with small defaults, from the repository root, after
`pip install -e ".[dev]"`. Outputs go to `examples/output/` (git-ignored).

| script | what it shows | time |
|---|---|---|
| [01_tabular_q_learning.py](01_tabular_q_learning.py) | the state discretizer, `default_config` + `train`, and the learned greedy policy drawn as a map over pole angle x angular velocity | ~15 s |
| [02_dqn_variants.py](02_dqn_variants.py) | DQN, then `double=True`, then `dueling=True`, same budget, return curves in one figure | ~3 min |
| [03_policy_gradient.py](03_policy_gradient.py) | reward-to-go, REINFORCE vs REINFORCE + learned baseline | ~1 min |
| [04_ppo_and_record.py](04_ppo_and_record.py) | PPO, checkpoint round trip (`load_agent`, `evaluate_agent`) and a GIF of the agent | ~1 min |
| [05_compare_families.py](05_compare_families.py) | one method per family through `cartpole-compare`: table + curves | ~2 min |
| [06_pixel_observations.py](06_pixel_observations.py) | what the CNN sees (`frame`, `frame-diff`) and a short PPO run from pixels | ~3 min |

```bash
python examples/01_tabular_q_learning.py
python examples/02_dqn_variants.py
python examples/03_policy_gradient.py
python examples/04_ppo_and_record.py
python examples/05_compare_families.py
python examples/06_pixel_observations.py
```

Every single method is also one command with working defaults, e.g.
`cartpole-train --method sarsa` or `cartpole-train --method a2c` (see the README table).

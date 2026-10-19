# Training

Code: [`src/cartpole_rl/train.py`](../src/cartpole_rl/train.py),
[`src/cartpole_rl/config.py`](../src/cartpole_rl/config.py)

## One command per method

Every method has defaults that solve CartPole-v1 on one CPU thread, so the flag `--method`
is all you need:

```bash
cartpole-train --method q-learning            # ~15 s
cartpole-train --method sarsa
cartpole-train --method mc-control
cartpole-train --method dqn                   # a few minutes
cartpole-train --method double-dqn
cartpole-train --method dueling-dqn
cartpole-train --method dueling-dqn --prioritized --n-step 3
cartpole-train --method reinforce
cartpole-train --method reinforce-baseline
cartpole-train --method actor-critic
cartpole-train --method a2c
cartpole-train --method ppo --record ppo.gif  # ~30 s, then a GIF of the agent
cartpole-play checkpoints/ppo/best.pt --vs-random --out before_after.gif  # random vs trained
cartpole-train --method ppo --obs frame-diff  # pixels: slow (rendering), short budget
cartpole-train --method ppo --obs frame-diff --net rescnn
```

Each run prints one line per evaluation:

```
[ppo/state] step   15000  ep   183  train(last20)  264.0  eval  500.0 +-   0.0    22.0s  confirm(100) 498.1
```

`train(last20)` is the mean of the last 20 training episodes (with exploration),
`eval` the 10-episode greedy evaluation, `confirm(100)` appears when the eval reached 475 and
is checked on 100 more episodes (see [evaluation.md](evaluation.md)). Output goes to
`checkpoints/<method>/` (`best.pt`, `history.json`).

## From Python

```python
from cartpole_rl import default_config, train, load_agent

cfg = default_config("double-dqn", total_steps=50_000, prioritized=True)
result = train(cfg)                      # TrainResult: agent, history, final_eval, solved_at, ...
agent, cfg = load_agent(result.checkpoint)
```

## Default hyper-parameters

All in `METHOD_DEFAULTS` in `config.py`; any keyword to `default_config` or CLI flag
overrides them. Common to all: gamma 0.99, hidden 128, Adam, one CPU thread, evaluation every
5 000 steps, stop when solved.

| method | budget (env steps) | key settings |
|---|---|---|
| `q-learning`, `sarsa` | 400 k | alpha 0.1, epsilon 1 -> 0.02 over 150 k steps, bins (1, 6, 12, 12) |
| `mc-control` | 400 k | alpha 0.05, same epsilon and bins |
| `dqn` | 80 k | lr 1e-3, batch 64, buffer 50 k, warm-up 1 k, train every 2 steps, tau 0.005, epsilon 1 -> 0.02 over 15 k |
| `double-dqn` | 80 k | + `double=True` |
| `dueling-dqn` | 80 k | + `double=True, dueling=True` |
| `reinforce` | 300 k | lr 2e-3, one episode per update, normalized returns |
| `reinforce-baseline` | 300 k | actor lr 2e-3, critic lr 5e-3 |
| `actor-critic` | 100 k | actor lr 5e-4, critic lr 2e-3, one update per step |
| `a2c` | 300 k | 8 envs x 8 steps, lr 1e-3, GAE lambda 1, entropy 0.01, grad clip 0.5 |
| `ppo` | 150 k | 8 envs x 32 steps, gamma 0.98, lambda 0.8, 20 epochs, minibatch 256, clip 0.2, lr 1e-3 -> 0 |
| any, `--obs frame / frame-diff` | 50 k | lr 3e-4; ppo: 8 x 64 steps, 4 epochs, minibatch 128; dqn: buffer 10 k, train every 4 |

The budgets are upper limits; with `stop_when_solved` (default) most runs stop much earlier.

**Why these values.** The tabular settings were picked so that every cell of the
864-state table is visited many times before epsilon is small. DQN's warm-up fills the
buffer with random play first so the first batches are not 64 copies of the same few states.
PPO's (gamma 0.98, lambda 0.8, 20 epochs, 8 x 32 steps) are the usual CartPole values: many
small rollouts reused heavily. The pixel settings are chosen for wall time, not for solving:
four PPO epochs instead of twenty because every CNN forward/backward is ~100x the cost of the
MLP.

## Threads, speed and memory

`--threads` (default 1) calls `torch.set_num_threads`; with 1 thread the small MLPs are as
fast as with 8, and several runs can share a machine. A full state-observation comparison of
all 11 methods took about 10 minutes of wall time in the README runs (each stops when
solved); pixel runs take about 5 minutes per 50 k steps, mostly in rendering. Memory stays under 500 MB (the pixel DQN replay buffer
stores frames as float16 for that reason).

## Comparing methods

```bash
cartpole-compare --methods all --output-dir runs/compare             # every state method
cartpole-compare --methods dueling-dqn --tag per-nstep3 --set prioritized=true --set n_step=3
cartpole-compare --methods ppo --obs frame-diff --net rescnn --output-dir runs/pixels
cartpole-compare --plot-only runs/compare                            # redraw table + figures
```

`--scale 0.5` halves every default budget, `--steps N` gives everyone the same budget,
`--set key=value` overrides any config field.

## On Kaggle / Colab

Everything runs on the CPU: the networks are tiny and the time goes into stepping (and, for
pixels, rendering) the environment, which a GPU does not speed up. What the notebook
(`notebooks/train_colab_kaggle.ipynb`) adds is room: the full comparison with several seeds,
and longer pixel runs than the ones in the README, using the free notebook CPUs while your
own machine stays free.

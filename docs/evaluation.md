# Evaluation: one protocol for every method

Code: [`src/cartpole_rl/evaluate.py`](../src/cartpole_rl/evaluate.py),
[`src/cartpole_rl/metrics.py`](../src/cartpole_rl/metrics.py)

Comparing methods is only fair if they are measured the same way. Every method, tabular or
deep, value or policy based, goes through the same three things.

## 1. Greedy evaluation episodes

`evaluate_agent(agent, n_episodes, seed)` plays `n_episodes` episodes with the **greedy**
action (argmax of Q, or argmax of the policy logits): no epsilon, no sampling. Exploration is
a training device; what we want to know is how good the learned behaviour is. Episode `i`
starts from `env.reset(seed=seed + i)`, so every method faces the *same* start states.

Reported numbers: mean return, std, min, max, and the success rate (fraction of episodes
with return >= 475).

| seed range | used for |
|---|---|
| `10_000 + i` | periodic evaluation during training (`eval_episodes`, default 10) |
| `50_000 + i` | confirmation of a candidate "solved" (`final_eval_episodes`, default 100) |
| `100_000 + i` | the final evaluation after training (100 episodes; 20 for pixel runs) |

## 2. When is a method "solved"?

Every `eval_every` environment steps (5 000 for state runs) the recorder runs the 10-episode
greedy evaluation. A mean >= 475 there is only a *candidate*: ten episodes are a small sample
and an almost-solving policy can pass by luck. The candidate is then **confirmed with 100
more episodes** on other start states, which is gymnasium's own definition (average of 475
over 100 consecutive episodes). The first confirmed point gives the three comparison numbers:

* **steps to solve**: environment steps used for training until then (the main measure of
  sample efficiency; evaluation episodes are not counted),
* **episodes to solve**: training episodes finished until then,
* **time to solve**: wall-clock seconds, on one CPU thread.

With `stop_when_solved=True` (the default) training stops there. `--no-stop` keeps going,
which is how you can watch methods that unlearn (DQN often does).

## 3. Which weights are kept

The recorder keeps a copy of the weights at the best periodic evaluation, and once a
confirmed solve happens it keeps those. At the end of training they are restored and saved as
`best.pt`. This is the usual cure for policies that get worse after solving, and the same
rule applies to every method, so it does not favour anyone.

## Curves

`history.json` in each run folder holds every training episode return with the step at which
it ended, every evaluation point, and per-update scalars (losses, entropy, epsilon, PPO clip
fraction). The plots use the **environment step** on the x axis, never "updates" or
"episodes", because that is the one unit all methods share: an A2C update uses 64 steps, a
REINFORCE update a whole episode, a DQN update one step. Training returns are smoothed with a
trailing 20-episode mean; evaluation points are drawn as dots.

Training returns are noisy and biased low (they include exploration), so they are only used
for the curves. All table numbers come from greedy evaluation.

## The CLI

```bash
cartpole-eval checkpoints/ppo/best.pt --episodes 100      # JSON with mean / std / success rate
cartpole-play checkpoints/ppo/best.pt --out ppo.gif       # one greedy episode as a GIF (or .mp4)
cartpole-compare --plot-only runs/compare                 # rebuild table + figures from saved runs
```

# The environment: CartPole-v1 as an MDP

Code: [`src/cartpole_rl/env.py`](../src/cartpole_rl/env.py)

## The task in plain words

A pole stands on a cart that can roll left and right on a track. Every 0.02 s you push the
cart left or right with a fixed force. You get **+1 for every step the pole stays up**. The
episode ends when the pole leans more than 12 degrees, when the cart leaves the track, or
after 500 steps. So the return of an episode is simply how long it lasted (max 500), and
gymnasium calls the task **solved at an average return of 475**.

## The MDP

| piece | CartPole-v1 |
|---|---|
| state `s` | `[x, x_dot, theta, theta_dot]`: cart position, cart velocity, pole angle, pole angular velocity |
| actions `a` | `0` push left, `1` push right |
| transition `P(s' \| s, a)` | deterministic physics (Euler integration); only the start state is random (each variable uniform in +-0.05) |
| reward `r` | `+1` per step, including the last one |
| termination | `\|theta\| > 0.2095` rad (12 deg) or `\|x\| > 2.4` |
| truncation | 500 steps (the `TimeLimit` wrapper) |
| discount `gamma` | not part of the env; we use 0.99 (0.98 for PPO) |

**Return.** `G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ...`. With reward 1 per step,
`G_t = (1 - gamma^k) / (1 - gamma)` for `k` remaining steps, so it saturates at
`1 / (1 - gamma) = 100` for gamma = 0.99: the agent effectively looks about 100 steps (2 s)
ahead, which is plenty to see the pole falling.

**Why "terminated" and "truncated" are different.** gymnasium returns both flags. *Terminated*
means the pole fell: there is no future, the value of the next state is 0. *Truncated* means
the 500-step clock ran out: the pole was fine, and if the episode had continued the agent would
have kept collecting reward. Treating a truncation like a failure teaches the agent that
states seen around step 500 are bad, which they are not. Every method here therefore
bootstraps (uses `V(s')` or `max Q(s', .)`) after a truncation and uses 0 only after a
termination. See `tabular.py`, `dqn.py` (`dones` = terminated only), `rollout.py`.

## Observation modes (`--obs`)

| mode | shape | what the agent sees | used by |
|---|---|---|---|
| `state` | `(4,)` | the four physics numbers | every method (default) |
| `frame` | `(1, 40, 80)` | one rendered frame: rows 150-330 of the 400 x 600 render (where cart and pole are), grey, resized, inverted so objects are bright | neural methods |
| `frame-diff` | `(2, 40, 80)` | the frame above **and** `frame(t) - frame(t-1)` | neural methods |

Why `frame` alone is hard: one image shows where the pole is, not where it is going. Two
states with the same angle but opposite angular velocity look identical, yet need opposite
actions. The problem is *partially observable*. The difference image puts the motion back
(it is non-zero only where something moved, with a sign that says which way), which is the
idea of the original notebook ("CNN on img(t) - img(t-1)"). Keeping the current frame as a
first channel tells the network *where* things are, the second *how they move*.

Rendering needs pygame. It works without a display: `env.py` sets
`SDL_VIDEODRIVER=dummy` before gymnasium imports pygame. Rendering costs about 5 ms per step
on a CPU, which is why the pixel runs are short.

## Discretizing the state for tables (`Discretizer`)

The tabular methods need a finite set of states. Each variable is clipped to a range and cut
into equal bins:

| variable | range | default bins |
|---|---|---|
| `x` | +-2.4 | 1 (ignored) |
| `x_dot` | +-3.0 | 6 |
| `theta` | +-0.21 | 12 |
| `theta_dot` | +-3.5 | 12 |

That gives `1 x 6 x 12 x 12 = 864` states, and the table is `864 x 2`. Ignoring `x` sounds
wrong (the cart can leave the track) but in practice the angle variables decide almost
everything, fewer states means each one is visited more often, and the tables still reach 500
steps. Change it with `default_config("q-learning", bins=(3, 6, 12, 12))`.

## The vector env (`VecEnv`)

A2C and PPO step `N` copies of the env in lockstep and reset finished ones automatically.
It is written out (instead of `gymnasium.vector`) so the one subtle detail is visible: when an
episode ends, `step` returns the observation that came *before* the reset (`final_obs`), so a
truncated episode can still be bootstrapped from `V(final_obs)`.

## Seeds

`train` seeds `random`, `numpy` and `torch`; each training episode `k` is reset with
`seed + k`; vector env `i` with `seed + i`. Evaluation uses its own seeds (see
[evaluation.md](evaluation.md)), so training never sees the evaluation start states.

"""Generate notebooks/train_colab_kaggle.ipynb from the package.

    python notebooks/build_notebook.py

The notebook installs the package from GitHub and only *imports* from it, so it never
drifts from src/. Markdown cells explain each method step by step between the code cells.
"""

from pathlib import Path

import nbformat as nbf

REPO = "https://github.com/benisalla/cartpole-rl"
cells = []


def md(text: str):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(f"""
# CartPole solved every way

This notebook trains every family of reinforcement-learning method on the same problem,
CartPole-v1, with the [`cartpole-rl`]({REPO}) package, and explains each one step by step:

1. the problem as an MDP, returns, value vs policy methods
2. tabular control: Q-learning, SARSA, Monte Carlo control (exploration with epsilon-greedy)
3. deep value methods: DQN, double DQN, dueling DQN, prioritized replay, n-step returns
4. policy gradients: REINFORCE with and without a baseline
5. actor-critic: one-step actor-critic, A2C, GAE
6. PPO and its clipped objective
7. the full comparison (several seeds), then policies that learn from pixels

Everything runs on the CPU (the networks are tiny; time goes into stepping the environment),
so a free Kaggle or Colab CPU session is enough. Running the whole notebook takes about
1.5-2 hours, mostly the multi-seed comparison and the pixel runs; each section can be run
on its own after the setup cells.
""")

md("""
## Setup

The repository is private: on Kaggle/Colab either make it public, or add a GitHub token
(`https://<token>@github.com/...`), or upload the repository folder and `pip install -e .` it.
""")
code(f"""
import os, sys
os.environ["SDL_VIDEODRIVER"] = "dummy"      # render CartPole frames without a display
!pip install -q "cartpole-rl @ git+{REPO}"
""")
code("""
import numpy as np, torch, matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display

from cartpole_rl import default_config, train, load_agent, evaluate_agent
from cartpole_rl.env import make_env, Discretizer
from cartpole_rl.viz import plot_runs
torch.set_num_threads(1)
print(torch.__version__)
""")

md("""
## 1. The problem

A pole is hinged on a cart. Each step (0.02 s) we push the cart left (action 0) or right
(action 1) and receive +1. The episode **terminates** if the pole leans past 12 degrees or the
cart leaves the track, and is **truncated** after 500 steps. The return of an episode is its
length; the task counts as solved at an average of 475.

As an MDP: the state is `s = [x, x_dot, theta, theta_dot]`, the transitions are deterministic
physics (only the start state is random), the reward is always 1. What makes it a control
problem is that the pole is unstable: doing nothing, or pushing at random, lets it fall in
about 20 steps.
""")
code("""
env = make_env("state", seed=0)
obs, _ = env.reset(seed=0)
print("state:", obs, "| actions:", env.action_space)
returns = []
for ep in range(100):                          # a random policy
    obs, _ = env.reset(seed=ep); done, total = False, 0
    while not done:
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        total += r; done = term or trunc
    returns.append(total)
print(f"random policy: {np.mean(returns):.1f} +- {np.std(returns):.1f}  (solved = 475)")
""")

md("""
### Returns and discounting

The agent maximises the discounted return `G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ...`.
With a reward of 1 per step, `G_t` grows with the number of remaining steps and saturates at
`1 / (1 - gamma)` (100 for gamma = 0.99): the agent effectively plans about 100 steps ahead.
`rewards_to_go` computes all `G_t` of an episode in one backward pass.

**Terminated vs truncated.** After a termination the future is worth 0. After a truncation
(500 steps) the pole was still up: every method here keeps bootstrapping from the next state
in that case, otherwise it would learn that the states seen around step 500 are bad.
""")
code("""
from cartpole_rl.algorithms.policy_gradient import rewards_to_go
G = rewards_to_go([1.0] * 500, 0.99)
plt.figure(figsize=(6, 3)); plt.plot(G); plt.xlabel("t"); plt.ylabel("G_t"); plt.title("reward-to-go, 500-step episode")
plt.show()
""")

md("""
### Value methods vs policy methods

* **Value methods** learn `Q(s, a)`, the expected return after action `a` in state `s`, and act
  with `argmax_a Q(s, a)`. They can learn from old data (off-policy, so replay buffers work) but
  need explicit exploration.
* **Policy methods** learn `pi(a|s)` directly and make good actions more likely. They explore by
  sampling, but the gradient needs fresh data from the current policy (on-policy).
* **Actor-critic** methods learn both: the critic `V(s)` only serves to judge the actor's actions
  with less noise.

Every run below uses the same evaluation: greedy actions, fixed start states, 10 episodes every
5 000 steps, and a run counts as *solved* when a 100-episode greedy evaluation averages 475.
""")

md("""
## 2. Tabular control

A table needs finitely many states, so the 4 numbers are cut into bins: 1 x 6 x 12 x 12 = 864
states (cart position ignored, most bins on the pole angle and angular velocity).

* **Q-learning**: `Q[s,a] += alpha (r + gamma max_a' Q[s',a'] - Q[s,a])`: the target assumes the
  greedy action next, whatever the agent actually does (off-policy).
* **SARSA**: `Q[s,a] += alpha (r + gamma Q[s',a'] - Q[s,a])` with the `a'` actually taken: it
  learns the value of the exploring policy, so it is more careful near failure.
* **Monte Carlo control**: wait for the end of the episode, then `Q[s,a] += alpha (G_t - Q[s,a])`:
  no bootstrapping, unbiased, noisier.

**Exploration**: epsilon-greedy, epsilon going linearly from 1 to 0.02. Early on the agent
tries everything so every table cell gets visited; later it mostly exploits what it learnt.
""")
code("""
d = Discretizer((1, 6, 12, 12))
print("rows in the table:", d.n_states, "| state [0, 0, 0.05, 0.5] -> row", d([0, 0, 0.05, 0.5]))
tab = {}
for m in ["q-learning", "sarsa", "mc-control"]:
    r = train(default_config(m, output_dir=f"runs/nb/{m}"), verbose=False)
    tab[m] = r.history
    print(f"{m:12s} solved at {r.solved_at} | final {r.final_eval['mean']:.1f}")
plot_runs(tab, "tabular.png", "Tabular methods"); display(Image("tabular.png"))
""")

md("""
## 3. Deep value methods: DQN and friends

Replace the table with a network `Q_theta(s, .)` (4 -> 128 -> 128 -> 2). Each step: act
epsilon-greedily, store `(s, a, r, s', terminated)` in a **replay buffer**, sample a random
batch of 64, and minimise the Huber loss to the target

`y = r + gamma (1 - terminated) Q_target(s', a*)`.

* **Replay buffer**: consecutive states are nearly identical; random batches from the last 50 k
  transitions are closer to independent samples, and each transition is reused many times.
* **Target network** `Q_target`: a slowly updated copy (`theta_t <- 0.995 theta_t + 0.005 theta`).
  Without it the target moves with every gradient step and the network chases its own tail.
* **Double DQN**: `a* = argmax Q_online(s')`, evaluated by `Q_target`. Taking the max of noisy
  estimates overestimates; letting a second network evaluate the choice mostly cancels it.
* **Dueling**: `Q = V(s) + A(s, a) - mean A`. In most CartPole states both actions are worth about
  the same; the V stream learns from every transition.
* **Prioritized replay** (`prioritized=True`): sample transitions with large TD error more
  often, and correct the bias with importance weights.
* **n-step returns** (`n_step=3`): the target sums 3 real rewards before bootstrapping, so
  information travels 3 steps per update.
""")
code("""
dq = {}
for name, kw in [("dqn", {}), ("double-dqn", {}), ("dueling-dqn", {}),
                 ("dueling + PER + 3-step", {"prioritized": True, "n_step": 3})]:
    method = "dueling-dqn" if name.startswith("dueling +") else name
    r = train(default_config(method, output_dir=f"runs/nb/{name.replace(' ', '')}", **kw), verbose=False)
    dq[name] = r.history
    print(f"{name:24s} solved at {r.solved_at} | final {r.final_eval['mean']:.1f}")
plot_runs(dq, "dqn.png", "DQN family"); display(Image("dqn.png"))
""")

md("""
## 4. Policy gradients: REINFORCE

The policy is a network that outputs two logits; actions are *sampled*. After a whole episode:

`loss = - mean_t log pi(a_t|s_t) * A_t`

where `A_t` is the normalized reward-to-go (REINFORCE) or the normalized `G_t - V(s_t)`
(REINFORCE with baseline, `V` fit to `G_t`). The gradient of `log pi(a|s)` points to "make
this action more likely here"; weighting it by how good the outcome was is the whole idea.

Why a baseline: subtracting `V(s)` leaves the expected gradient unchanged but removes the part
of the return that would have come anyway from that state; the update then reacts only to what
the action changed, which cuts the variance.
""")
code("""
pg = {}
for m in ["reinforce", "reinforce-baseline"]:
    r = train(default_config(m, output_dir=f"runs/nb/{m}"), verbose=False)
    pg[m] = r.history
    print(f"{m:20s} solved at {r.solved_at} | final {r.final_eval['mean']:.1f}")
plot_runs(pg, "pg.png", "Policy gradient"); display(Image("pg.png"))
""")

md("""
## 5. Actor-critic and A2C

**One-step actor-critic** updates after every single step using the TD error as the advantage:
`delta = r + gamma V(s') - V(s)`; the critic minimises `delta^2`, the actor
`- log pi(a|s) * delta`. No waiting for the end of the episode, less variance than REINFORCE,
but each update rests on one correlated sample and on a critic that may still be wrong.

**A2C** runs 8 environments in parallel for 8 steps, so each update sees 64 samples from 8
different episodes (decorrelated without a replay buffer), and uses one combined loss:
`- mean(log pi * A) + 0.5 mean((V - R)^2) - 0.01 entropy`. The entropy bonus keeps the policy
from becoming deterministic too early.

### GAE

Advantages for A2C and PPO come from generalized advantage estimation:
`A_t = delta_t + gamma lambda A_{t+1}`. `lambda = 0` gives the one-step TD error (low variance,
biased by V), `lambda = 1` the Monte Carlo return minus V (unbiased, noisy). The cell below
shows both on a small rollout.
""")
code("""
from cartpole_rl.algorithms.rollout import gae
r = np.ones((6, 1), np.float32); v = np.linspace(5, 3, 6, dtype=np.float32)[:, None]; d = np.zeros_like(r)
for lam in (0.0, 0.8, 1.0):
    adv, _ = gae(r, v, d, np.array([3.0], np.float32), gamma=0.98, lam=lam)
    print(f"lambda={lam}: advantages {adv[:, 0].round(2)}")
""")
code("""
ac = {}
for m in ["actor-critic", "a2c"]:
    r = train(default_config(m, output_dir=f"runs/nb/{m}"), verbose=False)
    ac[m] = r.history
    print(f"{m:14s} solved at {r.solved_at} | final {r.final_eval['mean']:.1f}")
""")

md("""
## 6. PPO

PPO collects 8 x 32 steps, then reuses them for 20 epochs. Reuse is what makes it sample
efficient, but after a few gradient steps the policy differs from the one that collected the
data. The clipped objective

`L = - mean( min(ratio * A, clip(ratio, 1 - 0.2, 1 + 0.2) * A) )`, `ratio = pi(a|s) / pi_old(a|s)`

removes any gain from moving an action's probability more than 20% away from where it was:
for a good action (`A > 0`) the objective is flat once the ratio passes 1.2, for a bad one once
it falls below 0.8. The plot shows the objective of one sample as a function of the ratio.
""")
code("""
ratio = np.linspace(0.5, 1.5, 200)
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
for ax, A in zip(axes, (1.0, -1.0)):
    ax.plot(ratio, np.minimum(ratio * A, np.clip(ratio, 0.8, 1.2) * A))
    ax.axvline(1, color="gray", lw=0.5); ax.set_title(f"advantage {A:+.0f}"); ax.set_xlabel("ratio")
plt.show()
r = train(default_config("ppo", output_dir="runs/nb/ppo"), verbose=False)
ac["ppo"] = r.history
print("ppo solved at", r.solved_at, "| final", r.final_eval["mean"])
plot_runs(ac, "ac.png", "Actor-critic family"); display(Image("ac.png"))
""")

md("""
## 7. The full comparison, three seeds

`cartpole-compare` trains every method with its default budget, the same evaluation, and
writes a markdown table and the curves. One seed can be lucky; three show the spread.
""")
code("""
for seed in (0, 1, 2):
    !cartpole-compare --methods all --seed {seed} --output-dir runs/compare-seed{seed}
for seed in (0, 1, 2):
    display(Markdown(f"### seed {seed}\\n" + open(f"runs/compare-seed{seed}/results.md").read()))
display(Image("runs/compare-seed0/curves.png"))
""")

md("""
## 8. Learning from pixels

The original notebooks fed rendered images to a CNN. Here the frame is cropped around the cart,
turned grey and resized to 40 x 80. A single frame (`--obs frame`) shows where the pole is but
not where it is going, so the problem is partially observable; `--obs frame-diff` adds
`frame(t) - frame(t-1)` as a second channel, which shows the motion. Rendering every step is
slow, so these runs are longer than the README ones but still far from the millions of steps
pixel-based agents usually get.
""")
code("""
env = make_env("frame-diff", seed=0); obs, _ = env.reset(seed=0)
for _ in range(6): obs, *_ = env.step(1)
fig, axes = plt.subplots(1, 3, figsize=(12, 2.5))
axes[0].imshow(env.render()); axes[1].imshow(obs[0], cmap="gray"); axes[2].imshow(obs[1], cmap="bwr", vmin=-1, vmax=1)
for ax, t in zip(axes, ["render", "frame", "frame(t) - frame(t-1)"]): ax.set_title(t); ax.axis("off")
plt.show()
""")
code("""
px = {}
for obs_mode, net in [("frame", "cnn"), ("frame-diff", "cnn"), ("frame-diff", "rescnn")]:
    cfg = default_config("ppo", obs=obs_mode, net=net, total_steps=200_000, output_dir=f"runs/nb/ppo-{obs_mode}-{net}")
    r = train(cfg, verbose=False)
    px[f"ppo [{obs_mode}, {net}]"] = r.history
    print(obs_mode, net, "final", r.final_eval["mean"], "solved at", r.solved_at)
plot_runs(px, "pixels.png", "PPO from pixels", window=10); display(Image("pixels.png"))
""")

md("""
## 9. Watch an agent
""")
code("""
from cartpole_rl.play import record
agent, cfg = load_agent("runs/nb/ppo/best.pt")
print(record(agent, "ppo.gif"))
display(Image("ppo.gif"))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
out = Path(__file__).with_name("train_colab_kaggle.ipynb")
nbf.write(nb, out)
print("wrote", out, f"({len(cells)} cells)")

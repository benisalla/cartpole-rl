"""01 - Tabular Q-learning on a discretized CartPole (about 15 seconds).

What it shows:
  * how the 4 continuous state variables become one table index (``Discretizer``),
  * ``default_config`` + ``train``: the two calls every method goes through,
  * what the learned table looks like: the greedy action as a function of pole angle and
    angular velocity (saved to examples/output/q_policy.png).

    python examples/01_tabular_q_learning.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cartpole_rl import default_config, train
from cartpole_rl.env import Discretizer

OUT = Path("examples/output")
OUT.mkdir(parents=True, exist_ok=True)

d = Discretizer((1, 6, 12, 12))
print(f"bins (x, x_dot, theta, theta_dot) = {tuple(int(b) for b in d.bins)} -> {d.n_states} table rows")
print("state [0, 0, 0.05, 0.5] -> row", d([0, 0, 0.05, 0.5]))

cfg = default_config("q-learning", output_dir=str(OUT / "q-learning"))
result = train(cfg)
print("final greedy evaluation:", {k: v for k, v in result.final_eval.items() if k != "returns"})

# The greedy policy, sliced at x_dot = 0 (middle bin): push left (0) or right (1)?
q = result.agent.q.reshape(*d.bins, 2)
policy = q[0, 3].argmax(-1)  # [theta bin, theta_dot bin]
visited = np.abs(q[0, 3]).sum(-1) > 0
fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
ax.imshow(np.where(visited, policy, np.nan), origin="lower", cmap="coolwarm", extent=(-3.5, 3.5, -0.21, 0.21),
          aspect="auto")
ax.set_xlabel("pole angular velocity (rad/s)")
ax.set_ylabel("pole angle (rad)")
ax.set_title("greedy action: blue = push left, red = push right")
fig.savefig(OUT / "q_policy.png", dpi=120)
print("wrote", OUT / "q_policy.png")

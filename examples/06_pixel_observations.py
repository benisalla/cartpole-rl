"""06 - Learning from pixels: what the CNN sees, and a short PPO run on frame differences
(about 3 minutes; rendering every step is what makes pixels slow).

Saves examples/output/pixel_obs.png: the raw render, the 40 x 80 grey crop (``frame``), and
the two channels of ``frame-diff`` (current frame, current - previous). Then trains PPO with
the CNN policy for a few thousand steps: enough to see the return rise above random play
(about 22), not enough to solve. Longer pixel runs belong on a GPU (see the notebook).

    python examples/06_pixel_observations.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cartpole_rl import default_config, train
from cartpole_rl.env import make_env

OUT = Path("examples/output")
OUT.mkdir(parents=True, exist_ok=True)

env = make_env("frame-diff", seed=0)
obs, _ = env.reset(seed=0)
for _ in range(6):
    obs, *_ = env.step(1)
raw = env.render()
env.close()
fig, axes = plt.subplots(1, 3, figsize=(12, 2.8), constrained_layout=True)
axes[0].imshow(raw)
axes[0].set_title("render (400 x 600 x 3)")
axes[1].imshow(obs[0], cmap="gray")
axes[1].set_title("frame: crop, grey, 40 x 80")
axes[2].imshow(obs[1], cmap="bwr", vmin=-1, vmax=1)
axes[2].set_title("frame-diff 2nd channel: frame(t) - frame(t-1)")
for ax in axes:
    ax.axis("off")
fig.savefig(OUT / "pixel_obs.png", dpi=110)
print("wrote", OUT / "pixel_obs.png", "| observation shape", obs.shape)

cfg = default_config("ppo", obs="frame-diff", total_steps=8_000, eval_every=4_000, eval_episodes=3,
                     final_eval_episodes=5, output_dir=str(OUT / "ppo-pixels"))
r = train(cfg)
print(f"CNN ({r.n_params:,} params) greedy return after {r.steps} steps: {r.final_eval['mean']:.1f}")

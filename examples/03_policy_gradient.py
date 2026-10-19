"""03 - REINFORCE with and without a learned baseline (about 1 minute).

Same seed, same budget, the only difference is the advantage: normalized G_t versus
normalized G_t - V(s_t). Also prints the reward-to-go of a short episode so the quantity the
gradient is weighted by is visible.

    python examples/03_policy_gradient.py
"""

from pathlib import Path

from cartpole_rl import default_config, train
from cartpole_rl.algorithms.policy_gradient import rewards_to_go
from cartpole_rl.viz import plot_runs

OUT = Path("examples/output")
print("reward-to-go of a 5-step episode, gamma=0.99:", rewards_to_go([1, 1, 1, 1, 1], 0.99).round(3))

histories = {}
for method in ("reinforce", "reinforce-baseline"):
    r = train(default_config(method, total_steps=25_000, output_dir=str(OUT / method)))
    histories[method] = r.history
    print(f"{method:20s} greedy return {r.final_eval['mean']:.1f}  solved at: {r.solved_at}")

print("wrote", plot_runs(histories, str(OUT / "policy_gradient.png"), "REINFORCE vs REINFORCE + baseline"))

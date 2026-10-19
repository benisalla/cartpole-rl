"""02 - DQN and its variants, one flag at a time (2-4 minutes on one CPU thread).

Trains plain DQN, then the same budget with ``double=True`` and ``dueling=True``, and prints
the greedy return of each. Swap in ``prioritized=True`` or ``n_step=3`` to try the other two
ideas. A shorter budget than the README runs, so not every variant solves here: that
difference between runs is itself worth looking at.

    python examples/02_dqn_variants.py
"""

from pathlib import Path

from cartpole_rl import default_config, train
from cartpole_rl.viz import plot_runs

OUT = Path("examples/output")
STEPS = 30_000

histories = {}
for name, flags in [("dqn", {}), ("double", {"double": True}), ("double + dueling", {"double": True, "dueling": True})]:
    cfg = default_config("dqn", total_steps=STEPS, output_dir=str(OUT / f"dqn-{len(histories)}"), **flags)
    r = train(cfg)
    histories[name] = r.history
    print(f"{name:18s} greedy return {r.final_eval['mean']:.1f} +- {r.final_eval['std']:.1f}")

print("wrote", plot_runs(histories, str(OUT / "dqn_variants.png"), "DQN variants"))

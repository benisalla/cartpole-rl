"""05 - One method per family, same evaluation, one table and one figure (3-5 minutes).

Q-learning (tabular), double DQN (value-based, deep), REINFORCE + baseline (policy gradient)
and A2C (actor-critic), each with a reduced budget. This is ``cartpole-compare`` from
Python; the README results use the full default budgets.

    python examples/05_compare_families.py
"""

from pathlib import Path

from cartpole_rl.compare import main as compare

OUT = Path("examples/output/compare")
compare(["--methods", "q-learning,double-dqn,reinforce-baseline,a2c", "--scale", "0.4", "--output-dir", str(OUT)])
print("table:", OUT / "results.md", "| curves:", OUT / "curves.png")

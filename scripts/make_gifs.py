"""Regenerate the README's before / after GIFs from saved checkpoints.

    python scripts/make_gifs.py                       # checkpoints from runs/compare (cartpole-compare)
    python scripts/make_gifs.py --runs checkpoints    # or from cartpole-train output folders

Writes
  assets/before_after.gif    random agent (left) vs trained PPO (right), same start state
  assets/families_grid.gif   one trained agent per family + the random agent, same start state

A missing checkpoint is trained on the spot with the method's defaults (seconds to a few
minutes). The start state is the first evaluation seed on which every trained agent reaches
500 steps, so the grid shows what each family learnt rather than an unlucky episode; the
seed is printed.
"""

import argparse
from pathlib import Path

from cartpole_rl import default_config, load_agent, train
from cartpole_rl.env import make_env
from cartpole_rl.evaluate import FINAL_EVAL_SEED, run_episode
from cartpole_rl.play import RandomAgent, record_grid

FAMILIES = [  # (label, method)
    ("Q-learning (table)", "q-learning"),
    ("double DQN", "double-dqn"),
    ("REINFORCE+baseline", "reinforce-baseline"),
    ("A2C (actor-critic)", "a2c"),
    ("PPO", "ppo"),
]


def get_agent(runs: Path, method: str):
    ckpt = runs / method / "best.pt"
    if not ckpt.exists():
        print(f"no checkpoint for {method}: training it with the defaults")
        train(default_config(method, output_dir=str(runs / method)))
    return load_agent(str(ckpt))[0]


def pick_seed(agents, tries: int = 50) -> int:
    env = make_env("state")
    for seed in range(FINAL_EVAL_SEED, FINAL_EVAL_SEED + tries):
        if all(run_episode(a, env, seed=seed)[1] == 500 for a in agents):
            env.close()
            return seed
    env.close()
    return FINAL_EVAL_SEED


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", default="runs/compare", help="folder with <method>/best.pt")
    p.add_argument("--out", default="assets")
    a = p.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    agents = {method: get_agent(runs, method) for _, method in FAMILIES}
    seed = pick_seed(list(agents.values()))
    print("start-state seed:", seed)

    lengths = record_grid([("random agent", RandomAgent(seed), "state"), ("trained PPO", agents["ppo"], "state")],
                          str(out / "before_after.gif"), seed=seed, cols=2, every=4, scale=0.5)
    print("before_after.gif:", lengths)

    entries = [("random", RandomAgent(seed), "state")]
    entries += [(label, agents[method], "state") for label, method in FAMILIES]
    lengths = record_grid(entries, str(out / "families_grid.gif"), seed=seed, cols=3, every=5, scale=0.4,
                          font_size=12)
    print("families_grid.gif:", lengths)
    for f in ("before_after.gif", "families_grid.gif"):
        print(f, f"{(out / f).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

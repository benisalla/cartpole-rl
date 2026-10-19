"""The one evaluation every method goes through.

Greedy actions (no exploration, no sampling), fixed episode seeds so that every method is
tested on the same start states, and three numbers: mean return, std, and success rate
(fraction of episodes with return >= 475). "Solved" means mean return >= 475, gymnasium's
threshold for CartPole-v1.

CLI::

    cartpole-eval checkpoints/ppo/best.pt --episodes 100
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from cartpole_rl.config import SOLVED_RETURN
from cartpole_rl.env import make_env

EVAL_SEED = 10_000        # periodic evaluation during training
CONFIRM_SEED = 50_000     # confirmation of a candidate "solved" during training
FINAL_EVAL_SEED = 100_000  # final evaluation: start states never used before


def run_episode(agent, env, seed: int | None = None, greedy: bool = True,
                frames: list | None = None) -> tuple[float, int]:
    obs, _ = env.reset(seed=seed)
    total, length, done = 0.0, 0, False
    while not done:
        if frames is not None:
            frames.append(env.render())
        obs, r, term, trunc, _ = env.step(agent.act(obs, greedy=greedy))
        total += r
        length += 1
        done = term or trunc
    return total, length


def evaluate_agent(agent, obs: str = "state", n_episodes: int = 10, seed: int = EVAL_SEED, env=None,
                   frame_height: int = 40, frame_width: int = 80) -> dict:
    """Mean / std / success rate of the greedy policy over ``n_episodes`` fixed-seed episodes."""
    own = env is None
    if own:
        env = make_env(obs, frame_height=frame_height, frame_width=frame_width)
    returns = [run_episode(agent, env, seed=seed + i)[0] for i in range(n_episodes)]
    if own:
        env.close()
    r = np.asarray(returns)
    return {
        "mean": float(r.mean()),
        "std": float(r.std()),
        "min": float(r.min()),
        "max": float(r.max()),
        "success_rate": float((r >= SOLVED_RETURN).mean()),
        "solved": bool(r.mean() >= SOLVED_RETURN),
        "episodes": n_episodes,
        "returns": r.tolist(),
    }


def main(argv=None):
    from cartpole_rl.train import load_agent

    p = argparse.ArgumentParser(description="Evaluate a trained CartPole checkpoint (greedy policy).")
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=FINAL_EVAL_SEED)
    a = p.parse_args(argv)
    agent, cfg = load_agent(a.checkpoint)
    res = evaluate_agent(agent, cfg.obs, a.episodes, a.seed, frame_height=cfg.frame_height, frame_width=cfg.frame_width)
    res.pop("returns")
    print(json.dumps({"method": cfg.method, "obs": cfg.obs, **res}, indent=2))


if __name__ == "__main__":
    main()

"""Train one method on CartPole-v1:  ``cartpole-train --method ppo``.

Every method goes through the same steps:

  1. ``default_config(method)``  known-good hyper-parameters (any CLI flag overrides them)
  2. ``build_agent(cfg)``        Q table, Q network or policy network
  3. ``LEARNERS[method]``        the algorithm's own loop, reporting to a shared ``Recorder``
  4. final greedy evaluation on 100 unseen start states, checkpoint + history on disk
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cartpole_rl.agents import build_agent
from cartpole_rl.config import METHODS, NETS, OBS_MODES, TABULAR, TrainConfig, default_config
from cartpole_rl.env import make_env
from cartpole_rl.evaluate import FINAL_EVAL_SEED, evaluate_agent
from cartpole_rl.metrics import Recorder


@dataclass
class TrainResult:
    agent: object
    config: TrainConfig
    history: dict
    final_eval: dict
    solved_at: dict | None
    wall_time: float
    steps: int
    episodes: int
    n_params: int
    checkpoint: str | None = None

    def summary(self) -> dict:
        fe = {k: v for k, v in self.final_eval.items() if k != "returns"}
        return {
            "method": self.config.method, "obs": self.config.obs, "net": self.config.resolved_net()
            if self.config.method not in TABULAR else "table",
            "seed": self.config.seed, "params": self.n_params, "steps": self.steps, "episodes": self.episodes,
            "wall_time": round(self.wall_time, 1), "solved_at": self.solved_at, "final_eval": fe,
        }


def set_seed(seed: int, threads: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if threads == 1:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")


def count_params(agent) -> int:
    if hasattr(agent, "model"):
        return sum(p.numel() for p in agent.model.parameters())
    return int(agent.q.size)


def obs_shape_of(cfg: TrainConfig):
    env = make_env(cfg.obs, frame_height=cfg.frame_height, frame_width=cfg.frame_width)
    shape = env.observation_space.shape
    env.close()
    return shape


def train(cfg: TrainConfig, verbose: bool = True, save: bool = True) -> TrainResult:
    from cartpole_rl.algorithms import LEARNERS

    set_seed(cfg.seed, cfg.threads)
    agent = build_agent(cfg, obs_shape_of(cfg))
    rec = Recorder(cfg, agent, verbose=verbose)
    LEARNERS[cfg.method](cfg, agent, rec)
    wall = rec.finish()  # restores the best evaluated weights
    final = evaluate_agent(agent, cfg.obs, cfg.final_eval_episodes, seed=FINAL_EVAL_SEED,
                           frame_height=cfg.frame_height, frame_width=cfg.frame_width)
    result = TrainResult(agent, cfg, rec.history(), final, rec.solved_at, wall, rec.steps, rec.episodes,
                         count_params(agent))
    if verbose:
        sa = rec.solved_at
        s = f"solved at step {sa['step']} (episode {sa['episode']}, {sa['time']:.0f}s)" if sa else "not solved"
        print(f"[{cfg.method}/{cfg.obs}] done: {rec.steps} steps, {rec.episodes} episodes, {wall:.0f}s, {s}; "
              f"final eval {final['mean']:.1f} +- {final['std']:.1f} over {final['episodes']} episodes")
    if save:
        result.checkpoint = save_run(result)
    return result


def save_run(result: TrainResult) -> str:
    out = Path(result.config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "best.pt"
    torch.save({"config": result.config.to_dict(), "state": result.agent.state_dict()}, path)
    with open(out / "history.json", "w") as f:
        json.dump({"summary": result.summary(), "history": result.history,
                   "config": result.config.to_dict()}, f)
    return str(path)


def load_agent(path: str):
    """Rebuild a trained agent from ``best.pt``. Returns (agent, config)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = TrainConfig.from_dict(ckpt["config"])
    agent = build_agent(cfg, obs_shape_of(cfg))
    agent.load_state_dict(ckpt["state"])
    if hasattr(agent, "model"):
        agent.model.eval()
    return agent, cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train one RL method on CartPole-v1.")
    p.add_argument("--method", choices=METHODS, default="ppo")
    p.add_argument("--obs", choices=OBS_MODES, default="state")
    p.add_argument("--net", choices=("auto",) + NETS, default=None)
    p.add_argument("--steps", type=int, default=None, help="env-step budget (default: per method)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--final-eval-episodes", type=int, default=None)
    p.add_argument("--no-stop", action="store_true", help="keep training after solving")
    p.add_argument("--double", action="store_true", default=None, help="dqn: double Q-learning target")
    p.add_argument("--dueling", action="store_true", default=None, help="dqn: dueling head")
    p.add_argument("--prioritized", action="store_true", default=None, help="dqn: prioritized replay")
    p.add_argument("--n-step", type=int, default=None, help="dqn: n-step returns")
    p.add_argument("--n-envs", type=int, default=None, help="a2c / ppo: parallel envs")
    p.add_argument("--entropy-coef", type=float, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--record", default=None, help="write a GIF/MP4 of the trained agent to this path")
    return p


def config_from_args(a) -> TrainConfig:
    overrides = dict(net=a.net, total_steps=a.steps, seed=a.seed, threads=a.threads, lr=a.lr, gamma=a.gamma,
                     eval_every=a.eval_every, eval_episodes=a.eval_episodes,
                     final_eval_episodes=a.final_eval_episodes, double=a.double, dueling=a.dueling,
                     prioritized=a.prioritized, n_step=a.n_step, n_envs=a.n_envs, entropy_coef=a.entropy_coef,
                     output_dir=a.output_dir or f"checkpoints/{a.method}" + ("" if a.obs == "state" else f"-{a.obs}"))
    if a.no_stop:
        overrides["stop_when_solved"] = False
    return default_config(a.method, a.obs, **overrides)


def main(argv=None):
    a = build_parser().parse_args(argv)
    cfg = config_from_args(a)
    result = train(cfg)
    print(f"checkpoint: {result.checkpoint}")
    if a.record:
        from cartpole_rl.play import record

        print(record(result.agent, a.record, obs=cfg.obs, frame_height=cfg.frame_height,
                     frame_width=cfg.frame_width))


if __name__ == "__main__":
    main()

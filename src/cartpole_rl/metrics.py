"""Bookkeeping shared by every algorithm: the ``Recorder``.

Each training loop calls ``rec.step()`` after every environment step and ``rec.episode(...)``
when an episode ends, and loops ``while not rec.done``. The recorder
  * counts env steps and episodes (the common x-axis of every curve),
  * runs the greedy evaluation every ``eval_every`` steps,
  * remembers when the agent first solved the task (steps, episodes, seconds). A short eval
    (``eval_episodes``, 10 by default) that reaches 475 is only a candidate: it is confirmed by
    ``final_eval_episodes`` (100) more episodes on other start states, which is gymnasium's
    definition of solved (average 475 over 100 consecutive episodes),
  * keeps the best weights seen by evaluation (DQN in particular can unlearn a solved policy;
    keeping the best checkpoint is the usual cure, and the same rule applies to everyone),
  * stops the run when the budget is spent or, with ``stop_when_solved``, when solved.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from cartpole_rl.config import SOLVED_RETURN, TrainConfig
from cartpole_rl.env import make_env
from cartpole_rl.evaluate import CONFIRM_SEED, evaluate_agent


def smooth(values, window: int = 20) -> np.ndarray:
    """Trailing moving average (the first points average over what exists so far)."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return v
    c = np.cumsum(np.insert(v, 0, 0.0))
    idx = np.arange(1, len(v) + 1)
    lo = np.maximum(0, idx - window)
    return (c[idx] - c[lo]) / (idx - lo)


class Recorder:
    def __init__(self, cfg: TrainConfig, agent, verbose: bool = True):
        self.cfg, self.agent, self.verbose = cfg, agent, verbose
        self.eval_env = make_env(cfg.obs, frame_height=cfg.frame_height, frame_width=cfg.frame_width)
        self.steps = 0
        self.episodes = 0
        self.episode_returns: list[float] = []
        self.episode_end_steps: list[int] = []
        self.evals: list[dict] = []
        self.solved_at: dict | None = None
        self.best_mean = -np.inf
        self.best_state = None
        self.extra: dict[str, list] = {}  # per-update scalars (losses, epsilon, ...)
        self._next_eval = cfg.eval_every
        self.t0 = time.perf_counter()

    # ---- called by the algorithms -------------------------------------------------------
    def step(self, n: int = 1):
        self.steps += n
        while self.steps >= self._next_eval:
            self._next_eval += self.cfg.eval_every
            self.evaluate()

    def episode(self, ret: float, length: int | None = None):
        self.episodes += 1
        self.episode_returns.append(float(ret))
        self.episode_end_steps.append(self.steps)

    def log(self, **scalars):
        for k, v in scalars.items():
            self.extra.setdefault(k, []).append(float(v))

    @property
    def done(self) -> bool:
        if self.steps >= self.cfg.total_steps:
            return True
        return bool(self.cfg.stop_when_solved and self.solved_at is not None)

    # ---- evaluation ---------------------------------------------------------------------
    def evaluate(self):
        res = evaluate_agent(self.agent, n_episodes=self.cfg.eval_episodes, env=self.eval_env)
        elapsed = time.perf_counter() - self.t0
        point = {"step": self.steps, "episode": self.episodes, "time": elapsed,
                 "mean": res["mean"], "std": res["std"]}
        self.evals.append(point)
        confirmed = None
        if self.solved_at is None and res["mean"] >= SOLVED_RETURN:
            conf = evaluate_agent(self.agent, n_episodes=self.cfg.final_eval_episodes, seed=CONFIRM_SEED,
                                  env=self.eval_env)
            confirmed = conf["mean"]
            point["confirm_mean"] = confirmed
            if confirmed >= SOLVED_RETURN:
                self.solved_at = {"step": self.steps, "episode": self.episodes, "time": elapsed}
                self.best_mean = np.inf  # a confirmed solution is the checkpoint we keep
                self.best_state = copy.deepcopy(self.agent.state_dict())
        if self.solved_at is None and res["mean"] > self.best_mean:
            self.best_mean = res["mean"]
            self.best_state = copy.deepcopy(self.agent.state_dict())
        if self.verbose:
            recent = np.mean(self.episode_returns[-20:]) if self.episode_returns else float("nan")
            print(f"[{self.cfg.method}/{self.cfg.obs}] step {self.steps:>7d}  ep {self.episodes:>5d}  "
                  f"train(last20) {recent:6.1f}  eval {res['mean']:6.1f} +- {res['std']:5.1f}  "
                  f"{elapsed:6.1f}s" + (f"  confirm({self.cfg.final_eval_episodes}) {confirmed:6.1f}"
                                         if confirmed is not None else ""), flush=True)

    def finish(self):
        """Make sure there is at least one evaluation, restore the best weights."""
        if not self.evals or self.evals[-1]["step"] != self.steps:
            self.evaluate()
        if self.best_state is not None:
            self.agent.load_state_dict(self.best_state)
        self.eval_env.close()
        return time.perf_counter() - self.t0

    def history(self) -> dict:
        return {
            "episode_returns": self.episode_returns,
            "episode_end_steps": self.episode_end_steps,
            "evals": self.evals,
            "extra": self.extra,
        }

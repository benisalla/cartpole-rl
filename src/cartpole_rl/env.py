"""The environment side: CartPole-v1 with three observation modes, a state discretizer for
the tabular methods, and a small synchronous vector env for A2C / PPO.

CartPole-v1 (gymnasium): a pole hinged on a cart. Each step you push the cart left (0) or
right (1) and get reward +1. The episode *terminates* when the pole leans more than 12 degrees
or the cart leaves the track (|x| > 2.4), and is *truncated* at 500 steps. So the return of an
episode is just its length, the best possible return is 500, and gymnasium calls the task
solved at an average return of 475.

Observation modes:
  * ``state``      the 4 numbers [x, x_dot, theta, theta_dot] (what every method uses by default)
  * ``frame``      one rendered frame, cropped around the cart, grey, downscaled to 1 x H x W.
                   A single image has no velocity in it, so this mode is partially observable:
                   that is the point of comparing it with the next one.
  * ``frame-diff`` 2 x H x W: the current frame and (current - previous) frame. The difference
                   image shows where things moved, i.e. the velocities, as in the original
                   notebook's "img(t) - img(t-1)" idea.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from PIL import Image

ENV_ID = "CartPole-v1"

# rgb_array rendering goes through pygame; make it work on machines without a display.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def preprocess_frame(rgb: np.ndarray, height: int = 40, width: int = 80) -> np.ndarray:
    """(400, 600, 3) uint8 render -> (height, width) float32 in [0, 1], objects bright.

    Rows 150..330 hold the pole and the cart (the rest is empty sky and track), so we crop
    them first: less input, same information.
    """
    crop = rgb[150:330]
    grey = Image.fromarray(crop).convert("L").resize((width, height), Image.BILINEAR)
    return 1.0 - np.asarray(grey, dtype=np.float32) / 255.0


class PixelObservation(gym.ObservationWrapper):
    """Replace the 4-number state with rendered pixels (``frame`` or ``frame-diff``)."""

    def __init__(self, env: gym.Env, mode: str = "frame", height: int = 40, width: int = 80):
        super().__init__(env)
        assert mode in ("frame", "frame-diff")
        self.mode, self.height, self.width = mode, height, width
        channels = 1 if mode == "frame" else 2
        self.observation_space = gym.spaces.Box(0.0, 1.0, (channels, height, width), np.float32)
        self._prev: np.ndarray | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev = None
        return self.observation(obs), info

    def observation(self, _state):
        frame = preprocess_frame(self.env.render(), self.height, self.width)
        if self.mode == "frame":
            return frame[None]
        prev = frame if self._prev is None else self._prev
        self._prev = frame
        return np.stack([frame, frame - prev]).astype(np.float32)


def make_env(obs: str = "state", seed: int | None = None, frame_height: int = 40, frame_width: int = 80,
             render: bool = False) -> gym.Env:
    """CartPole-v1 with the requested observation mode. ``render=True`` keeps rgb_array frames
    available for recording a GIF even in ``state`` mode."""
    need_pixels = obs != "state" or render
    env = gym.make(ENV_ID, render_mode="rgb_array" if need_pixels else None)
    if obs != "state":
        env = PixelObservation(env, obs, frame_height, frame_width)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


class Discretizer:
    """Map the continuous state to one integer, so a Q *table* can be used.

    Each of the 4 variables is clipped to a range and cut into ``bins[i]`` equal intervals.
    One bin means "ignore this variable". The defaults ignore the cart position (the pole
    rarely survives long enough for it to matter early on) and spend most bins on the pole
    angle and angular velocity, which decide whether the pole falls.
    """

    LOW = np.array([-2.4, -3.0, -0.21, -3.5])
    HIGH = -LOW

    def __init__(self, bins=(1, 6, 12, 12)):
        self.bins = np.asarray(bins, dtype=np.int64)
        self.n_states = int(np.prod(self.bins))
        # mixed-radix multipliers: index = sum(b_i * prod(bins after i))
        self._mult = np.concatenate([np.cumprod(self.bins[::-1])[::-1][1:], [1]])

    def __call__(self, state) -> int:
        s = np.clip(np.asarray(state, dtype=np.float64), self.LOW, self.HIGH)
        frac = (s - self.LOW) / (self.HIGH - self.LOW)
        idx = np.minimum((frac * self.bins).astype(np.int64), self.bins - 1)
        return int((idx * self._mult).sum())


class VecEnv:
    """N copies of the env stepped in lockstep, with auto-reset.

    Written out (instead of ``gymnasium.vector``) so the bootstrapping detail is visible:
    when an episode ends we return the *final* observation separately, because a truncated
    episode (500 steps reached) did not fail, and its value target must still bootstrap from
    V(final_obs). Only a real termination (pole fell) cuts the return to zero.
    """

    def __init__(self, n: int, obs: str = "state", seed: int = 0, **kw):
        self.envs = [make_env(obs, **kw) for _ in range(n)]
        self.n = n
        self.seed = seed
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self._ret = np.zeros(n)
        self._len = np.zeros(n, dtype=np.int64)

    def reset(self) -> np.ndarray:
        obs = [e.reset(seed=self.seed + i)[0] for i, e in enumerate(self.envs)]
        self._ret[:] = 0
        self._len[:] = 0
        return np.stack(obs)

    def step(self, actions):
        next_obs, final_obs, rewards, terms, truncs, finished = [], [], [], [], [], []
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            o, r, term, trunc, _ = env.step(int(a))
            self._ret[i] += r
            self._len[i] += 1
            final_obs.append(o)
            if term or trunc:
                finished.append((float(self._ret[i]), int(self._len[i])))
                self._ret[i], self._len[i] = 0, 0
                o, _ = env.reset()
            next_obs.append(o)
            rewards.append(r)
            terms.append(term)
            truncs.append(trunc)
        return (np.stack(next_obs), np.asarray(rewards, np.float32), np.asarray(terms),
                np.asarray(truncs), np.stack(final_obs), finished)

    def close(self):
        for e in self.envs:
            e.close()

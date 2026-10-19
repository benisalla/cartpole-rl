"""Watch a trained agent: record one greedy episode to a GIF (Pillow) or MP4 (imageio).

    cartpole-play checkpoints/ppo/best.pt --out ppo.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from cartpole_rl.env import ENV_ID, PixelObservation
from cartpole_rl.evaluate import FINAL_EVAL_SEED, run_episode


def record(agent, path: str, obs: str = "state", seed: int = FINAL_EVAL_SEED, every: int = 2, scale: float = 0.5,
           fps: int = 25, frame_height: int = 40, frame_width: int = 80) -> str:
    """Play one greedy episode and save every ``every``-th frame. Returns a one-line summary."""
    import gymnasium as gym

    env = gym.make(ENV_ID, render_mode="rgb_array")
    if obs != "state":
        env = PixelObservation(env, obs, frame_height, frame_width)
    frames: list = []
    ret, length = run_episode(agent, env, seed=seed, frames=frames)
    env.close()
    frames = frames[::every]
    if scale != 1.0:
        h, w = frames[0].shape[:2]
        size = (int(w * scale), int(h * scale))
        frames = [np.asarray(Image.fromarray(f).resize(size, Image.BILINEAR)) for f in frames]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".gif":
        imgs = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=32) for f in frames]
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
    else:
        import imageio.v2 as imageio  # optional dependency: pip install "cartpole-rl[video]"

        imageio.mimsave(out, frames, fps=fps)
    return f"recorded {out}: return {ret:.0f}, {length} steps"


def main(argv=None):
    from cartpole_rl.train import load_agent

    p = argparse.ArgumentParser(description="Record a trained CartPole agent to a GIF or MP4.")
    p.add_argument("checkpoint")
    p.add_argument("--out", default="examples/output/agent.gif")
    p.add_argument("--seed", type=int, default=FINAL_EVAL_SEED)
    p.add_argument("--every", type=int, default=2, help="keep one frame out of N")
    p.add_argument("--fps", type=int, default=25)
    a = p.parse_args(argv)
    agent, cfg = load_agent(a.checkpoint)
    print(record(agent, a.out, cfg.obs, a.seed, a.every, fps=a.fps, frame_height=cfg.frame_height,
                 frame_width=cfg.frame_width))


if __name__ == "__main__":
    main()

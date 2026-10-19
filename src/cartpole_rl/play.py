"""Watch a trained agent: record one greedy episode to a GIF (Pillow) or MP4 (imageio).

    cartpole-play checkpoints/ppo/best.pt --out ppo.gif
    cartpole-play checkpoints/ppo/best.pt --vs-random --out before_after.gif   # side by side

``record_grid`` plays several agents on the *same* start state and tiles their frames, each
panel with a caption (name, step count, and whether the pole fell); it is what
``scripts/make_gifs.py`` uses for the README's before / after GIFs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


class RandomAgent:
    """The "before training" baseline: pushes left or right at random."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, greedy: bool = True) -> int:
        return int(self.rng.integers(2))


def _play_frames(agent, obs_mode: str, seed: int, frame_height: int = 40, frame_width: int = 80):
    import gymnasium as gym

    env = gym.make(ENV_ID, render_mode="rgb_array")
    if obs_mode != "state":
        env = PixelObservation(env, obs_mode, frame_height, frame_width)
    frames: list = []
    ret, length = run_episode(agent, env, seed=seed, frames=frames)
    frames.append(env.render())  # the state after the last action (pole fallen, or step 500)
    env.close()
    return frames, int(length)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def record_grid(entries, path: str, seed: int = FINAL_EVAL_SEED, cols: int = 2, every: int = 4,
                scale: float = 0.4, fps: int = 25, hold: int = 30, font_size: int = 14) -> list[tuple[str, int]]:
    """Play every agent on the same start state and save one tiled GIF.

    ``entries``: list of (label, agent, obs_mode). Each panel shows its own episode; an agent
    whose pole falls early stays on its last frame with "fell at step N" in the caption, so
    the before / after difference is visible at a glance. Returns [(label, episode length)].
    """
    runs = [(label, *_play_frames(agent, obs, seed)) for label, agent, obs in entries]
    longest = max(length for _, _, length in runs)
    h, w = runs[0][1][0].shape[:2]
    pw, ph, cap = int(w * scale), int(h * scale), 24
    rows = (len(runs) + cols - 1) // cols
    font = _font(font_size)
    images = []
    ticks = list(range(0, longest + 1, every))
    if ticks[-1] != longest:
        ticks.append(longest)
    for t in ticks + [longest] * hold:  # hold the final picture for a moment
        canvas = Image.new("RGB", (cols * pw, rows * (ph + cap)), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (label, frames, length) in enumerate(runs):
            k = min(t, length)
            x, y = (i % cols) * pw, (i // cols) * (ph + cap)
            canvas.paste(Image.fromarray(frames[k]).resize((pw, ph), Image.BILINEAR), (x, y + cap))
            if t >= length and length < 500:
                status, color = f"fell at step {length}", (200, 40, 40)
            elif t >= length:
                status, color = "500 / 500 steps", (20, 120, 60)
            else:
                status, color = f"step {k}", (60, 60, 60)
            draw.text((x + 6, y + 4), f"{label}: {status}", fill=color, font=font)
            if i % cols:
                draw.line([(x, y), (x, y + ph + cap)], fill=(200, 200, 200))
        images.append(canvas.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return [(label, length) for label, _, length in runs]


def main(argv=None):
    from cartpole_rl.train import load_agent

    p = argparse.ArgumentParser(description="Record a trained CartPole agent to a GIF or MP4.")
    p.add_argument("checkpoint")
    p.add_argument("--out", default="examples/output/agent.gif")
    p.add_argument("--seed", type=int, default=FINAL_EVAL_SEED)
    p.add_argument("--every", type=int, default=2, help="keep one frame out of N")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--vs-random", action="store_true",
                   help="side-by-side GIF: a random agent (left) and this agent (right), same start state")
    a = p.parse_args(argv)
    agent, cfg = load_agent(a.checkpoint)
    if a.vs_random:
        entries = [("random agent", RandomAgent(a.seed), "state"), (f"trained {cfg.method}", agent, cfg.obs)]
        lengths = record_grid(entries, a.out, seed=a.seed, cols=2, every=a.every, fps=a.fps)
        print(f"recorded {a.out}: " + ", ".join(f"{lab} {n} steps" for lab, n in lengths))
        return
    print(record(agent, a.out, cfg.obs, a.seed, a.every, fps=a.fps, frame_height=cfg.frame_height,
                 frame_width=cfg.frame_width))


if __name__ == "__main__":
    main()

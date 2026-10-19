"""Plots: return curves (small multiples, one panel per family) and a steps-to-solve bar chart.

Every panel shares the same axes (environment steps on x, return on y, the 475 "solved"
line) so curves can be compared across panels too. Within a panel there are at most three
series; colours follow the series slot, never the rank.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cartpole_rl.config import SOLVED_RETURN  # noqa: E402
from cartpole_rl.metrics import smooth  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

FAMILIES = [
    ("Tabular (discretized state)", ["q-learning", "sarsa", "mc-control"]),
    ("Value-based, deep", ["dqn", "double-dqn", "dueling-dqn"]),
    ("Policy gradient", ["reinforce", "reinforce-baseline"]),
    ("Actor-critic", ["actor-critic", "a2c", "ppo"]),
]


def _style(ax, title, full_scale: bool = True):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    if full_scale:
        ax.axhline(SOLVED_RETURN, color=MUTED, lw=1, ls="--")
        ax.set_ylim(0, 520)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_panel(ax, runs: dict, title: str, window: int = 20, show_eval: bool = True, full_scale: bool = True):
    """``runs``: label -> history dict (``episode_returns``, ``episode_end_steps``, ``evals``)."""
    _style(ax, title, full_scale)
    for color, (label, h) in zip(SERIES, runs.items()):
        x = np.asarray(h["episode_end_steps"])
        y = smooth(h["episode_returns"], window)
        ax.plot(x, y, color=color, lw=2, label=label)
        if show_eval and h.get("evals"):
            ex = [e["step"] for e in h["evals"]]
            ey = [e["mean"] for e in h["evals"]]
            ax.plot(ex, ey, color=color, lw=0, marker="o", ms=4, alpha=0.55)
    if not full_scale:
        ax.set_ylim(0, None)  # after plotting, so the top follows the data
    if runs:
        ax.legend(frameon=False, fontsize=9, loc="best", labelcolor=INK)


def plot_families(histories: dict, path: str, window: int = 20):
    """2 x 2 grid of the four families; ``histories``: method -> history dict."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    for ax, (title, methods) in zip(axes.flat, FAMILIES):
        plot_panel(ax, {m: histories[m] for m in methods if m in histories}, title, window)
    for ax in axes[1]:
        ax.set_xlabel("environment steps", color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"episode return (smoothed, {window} ep.)", color=MUTED)
    fig.suptitle("CartPole-v1: training return (lines) and greedy evaluation (dots); dashed = 475",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_runs(histories: dict, path: str, title: str = "CartPole-v1", window: int = 20, full_scale: bool = True):
    """One panel with up to four runs (used for the pixel runs and the examples).
    ``full_scale=False`` zooms the y axis on the data instead of 0..520 (for runs far from solving)."""
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    plot_panel(ax, histories, title, window, full_scale=full_scale)
    ax.set_xlabel("environment steps", color=MUTED)
    ax.set_ylabel(f"episode return (smoothed, {window} ep.)", color=MUTED)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_steps_to_solve(rows: list[dict], path: str):
    """Horizontal bars: env steps until the confirmed solve (hatched grey = not solved)."""
    rows = sorted(rows, key=lambda r: (r["solved_step"] is None, r["solved_step"] or r["steps"]))
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(rows) + 1.2), constrained_layout=True)
    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        solved = r["solved_step"] is not None
        val = r["solved_step"] if solved else r["steps"]
        ax.barh(i, val, height=0.6, color=SERIES[0] if solved else "#c3c2b7",
                hatch=None if solved else "//", edgecolor="white")
        ax.text(val, i, f"  {val / 1000:.0f}k" + ("" if solved else " (not solved)"), va="center",
                fontsize=9, color=INK)
    ax.set_yticks(y, [r["label"] for r in rows], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("environment steps to solve (100-episode greedy mean >= 475)", color=MUTED)
    ax.grid(True, axis="x", color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(0, max((r["solved_step"] or r["steps"]) for r in rows) * 1.25)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path

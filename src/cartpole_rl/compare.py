"""Run several methods with the same evaluation and write a comparison:

    cartpole-compare --methods all                  # every state-observation method
    cartpole-compare --methods ppo,dqn --scale 0.5  # a subset, half the default budgets
    cartpole-compare --plot-only runs/compare       # redraw table + figures from saved runs

Outputs in ``--output-dir``: one folder per run (best.pt, history.json), ``results.json``,
``results.md`` (the markdown table used in the README), ``curves.png`` and
``steps_to_solve.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cartpole_rl.config import METHODS, OBS_MODES, default_config
from cartpole_rl.viz import plot_families, plot_runs, plot_steps_to_solve


def run_label(summary: dict) -> str:
    label = summary["method"]
    if summary.get("tag"):
        label += f" ({summary['tag']})"
    if summary["obs"] != "state":
        label += f" [{summary['obs']}, {summary['net']}]"
    return label


def table_rows(summaries: list[dict]) -> list[dict]:
    rows = []
    for s in summaries:
        sa, fe = s["solved_at"], s["final_eval"]
        rows.append({
            "label": run_label(s), "params": s["params"], "steps": s["steps"], "episodes": s["episodes"],
            "solved_step": sa["step"] if sa else None, "solved_episode": sa["episode"] if sa else None,
            "solved_time": sa["time"] if sa else None, "wall_time": s["wall_time"],
            "eval_mean": fe["mean"], "eval_std": fe["std"], "success": fe["success_rate"],
            "eval_episodes": fe["episodes"],
        })
    return rows


def markdown_table(rows: list[dict]) -> str:
    head = ("| method | params | solved | steps to solve | episodes to solve | time to solve | "
            "final greedy return | episodes >= 475 | steps run | wall time |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for r in rows:
        if r["solved_step"] is not None:
            solve = f"yes | {r['solved_step']:,} | {r['solved_episode']:,} | {r['solved_time']:.0f} s"
        else:
            solve = "no | - | - | -"
        n = "" if r["eval_episodes"] == 100 else f" ({r['eval_episodes']} ep.)"
        lines.append(f"| `{r['label']}` | {r['params']:,} | {solve} | {r['eval_mean']:.1f} +- {r['eval_std']:.1f}{n} | "
                     f"{100 * r['success']:.0f}% | {r['steps']:,} | {r['wall_time']:.0f} s |")
    return head + "\n".join(lines) + "\n"


def load_runs(out_dir: Path) -> dict[str, dict]:
    runs = {}
    for f in sorted(out_dir.glob("*/history.json")):
        runs[f.parent.name] = json.loads(f.read_text())
    return runs


def write_report(out_dir: Path) -> str:
    runs = load_runs(out_dir)
    order = {m: i for i, m in enumerate(METHODS)}
    obs_order = {o: i for i, o in enumerate(OBS_MODES)}
    summaries = sorted((r["summary"] for r in runs.values()),
                       key=lambda s: (obs_order[s["obs"]], order[s["method"]], s.get("net", ""), s.get("tag", "")))
    rows = table_rows(summaries)
    md = markdown_table(rows)
    (out_dir / "results.md").write_text(md)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    state = {r["summary"]["method"]: r["history"] for r in runs.values()
             if r["summary"]["obs"] == "state" and not r["summary"].get("tag")}
    if state:
        plot_families(state, str(out_dir / "curves.png"))
    pixel = {run_label(r["summary"]): r["history"] for r in runs.values() if r["summary"]["obs"] != "state"}
    if pixel:
        plot_runs(pixel, str(out_dir / "pixel_curves.png"), "Pixel observations (short runs; random play ~22)",
                  window=50, full_scale=any(r["solved_step"] for r in rows))
    if any(r["solved_step"] for r in rows):
        plot_steps_to_solve(rows, str(out_dir / "steps_to_solve.png"))
    return md


def main(argv=None):
    from cartpole_rl.train import train

    p = argparse.ArgumentParser(description="Train several methods on CartPole-v1 and compare them.")
    p.add_argument("--methods", default="all", help="comma-separated list, or 'all'")
    p.add_argument("--obs", choices=OBS_MODES, default="state")
    p.add_argument("--net", default=None, help="mlp | cnn | rescnn (default: auto)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scale", type=float, default=1.0, help="multiply every default step budget")
    p.add_argument("--steps", type=int, default=None, help="same step budget for every method")
    p.add_argument("--tag", default="", help="suffix for run folders, e.g. 'per-nstep3'")
    p.add_argument("--set", action="append", default=[], help="extra config override key=value (repeatable)")
    p.add_argument("--output-dir", default="runs/compare")
    p.add_argument("--plot-only", default=None, help="only rebuild the report from this folder")
    a = p.parse_args(argv)
    if a.plot_only:
        print(write_report(Path(a.plot_only)))
        return
    out = Path(a.output_dir)
    methods = [m for m in METHODS if a.obs == "state" or m not in ("q-learning", "sarsa", "mc-control")] \
        if a.methods == "all" else a.methods.split(",")
    extra = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        extra[k.replace("-", "_")] = json.loads(v) if v[:1] in "0123456789-[{tf" else v
    for m in methods:
        base = default_config(m, a.obs, net=a.net, seed=a.seed, **extra)
        steps = a.steps or int(base.total_steps * a.scale)
        name = m + ("" if a.obs == "state" else f"-{a.obs}-{base.resolved_net()}") + (f"-{a.tag}" if a.tag else "")
        cfg = default_config(m, a.obs, net=a.net, seed=a.seed, total_steps=steps, output_dir=str(out / name),
                             **extra)
        result = train(cfg)
        if a.tag:  # remember the variant in the saved summary
            hist = json.loads((out / name / "history.json").read_text())
            hist["summary"]["tag"] = a.tag
            (out / name / "history.json").write_text(json.dumps(hist))
        print(f"== {name}: {result.summary()['final_eval']['mean']:.1f}")
    print(write_report(out))


if __name__ == "__main__":
    main()

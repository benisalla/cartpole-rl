"""Rebuild the comparison reports and copy them (plus a GIF of the PPO agent) into assets/.

    python scripts/export_results.py            # after cartpole-compare runs into runs/compare, runs/pixels
"""

import json
import shutil
from pathlib import Path

from cartpole_rl.compare import write_report
from cartpole_rl.play import record
from cartpole_rl.train import load_agent

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)

for src, prefix in ((Path("runs/compare"), ""), (Path("runs/pixels"), "pixels_")):
    if not src.exists():
        continue
    write_report(src)
    for name in ("results.md", "results.json", "curves.png", "pixel_curves.png", "steps_to_solve.png"):
        if (src / name).exists():
            shutil.copy(src / name, ASSETS / f"{prefix}{name}")
    # small per-run summaries (no raw curves) so the numbers stay in the repository
    summaries = {d.name: json.loads((d / "history.json").read_text())["summary"]
                 for d in sorted(src.iterdir()) if (d / "history.json").exists()}
    (ASSETS / f"{prefix}summaries.json").write_text(json.dumps(summaries, indent=2))

ckpt = Path("runs/compare/ppo/best.pt")
if ckpt.exists():
    agent, cfg = load_agent(str(ckpt))
    print(record(agent, str(ASSETS / "ppo_cartpole.gif"), every=3, scale=0.4))
print("assets:", sorted(p.name for p in ASSETS.iterdir()))

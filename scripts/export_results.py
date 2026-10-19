"""Rebuild the comparison reports and copy them into assets/ (the GIFs come from make_gifs.py).

    python scripts/export_results.py            # after cartpole-compare runs into runs/compare, runs/pixels
"""

import json
import shutil
from pathlib import Path

from cartpole_rl.compare import write_report

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

print("GIFs: run scripts/make_gifs.py")
print("assets:", sorted(p.name for p in ASSETS.iterdir()))

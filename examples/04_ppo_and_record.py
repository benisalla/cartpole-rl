"""04 - PPO on 8 parallel envs, then a GIF of the trained agent (about 1 minute).

Uses the whole round trip: train -> checkpoint on disk -> ``load_agent`` -> evaluate ->
``record``. The GIF is written to examples/output/ppo.gif.

    python examples/04_ppo_and_record.py
"""

from pathlib import Path

from cartpole_rl import default_config, evaluate_agent, load_agent, train
from cartpole_rl.play import record

OUT = Path("examples/output")
result = train(default_config("ppo", total_steps=40_000, output_dir=str(OUT / "ppo")))
print("checkpoint:", result.checkpoint)

agent, cfg = load_agent(result.checkpoint)
res = evaluate_agent(agent, n_episodes=20, seed=123)
print(f"reloaded agent: {res['mean']:.1f} +- {res['std']:.1f}, {100 * res['success_rate']:.0f}% of episodes >= 475")
print(record(agent, str(OUT / "ppo.gif")))

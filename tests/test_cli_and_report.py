from pathlib import Path

from cartpole_rl.compare import main as compare_main
from cartpole_rl.evaluate import main as eval_main
from cartpole_rl.metrics import smooth
from cartpole_rl.play import main as play_main
from cartpole_rl.train import main as train_main


def test_smooth():
    s = smooth([0, 10, 20, 30], window=2)
    assert list(s) == [0, 5, 15, 25]


def test_cli_train_eval_play(tmp_path, capsys):
    out = tmp_path / "ql"
    train_main(["--method", "q-learning", "--steps", "3000", "--eval-every", "1500", "--eval-episodes", "2",
                "--final-eval-episodes", "2", "--output-dir", str(out)])
    ckpt = out / "best.pt"
    assert ckpt.exists() and (out / "history.json").exists()
    eval_main([str(ckpt), "--episodes", "3"])
    assert '"method": "q-learning"' in capsys.readouterr().out
    gif = tmp_path / "agent.gif"
    play_main([str(ckpt), "--out", str(gif), "--every", "4"])
    assert gif.exists() and gif.stat().st_size > 1000


def test_compare_writes_table_and_figures(tmp_path):
    out = tmp_path / "cmp"
    compare_main(["--methods", "q-learning,sarsa", "--steps", "2000", "--set", "eval_every=1000",
                  "--set", "eval_episodes=2", "--set", "final_eval_episodes=2", "--output-dir", str(out)])
    md = (out / "results.md").read_text()
    assert "`q-learning`" in md and "`sarsa`" in md
    for f in ("curves.png", "results.json"):
        assert Path(out / f).exists()
    # the steps-to-solve chart is only drawn when at least one run solved
    assert Path(out / "steps_to_solve.png").exists() == ("| yes |" in md)

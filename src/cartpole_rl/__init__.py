"""cartpole_rl: every family of RL method on CartPole-v1, one code base, one evaluation.

Typical use from Python::

    from cartpole_rl import TrainConfig, train
    result = train(TrainConfig(method="ppo", total_steps=60_000))
    print(result.final_eval)
"""

from cartpole_rl.config import METHODS, TrainConfig, default_config
from cartpole_rl.evaluate import evaluate_agent
from cartpole_rl.train import TrainResult, load_agent, train

__all__ = ["METHODS", "TrainConfig", "default_config", "train", "TrainResult", "load_agent", "evaluate_agent"]
__version__ = "0.1.0"

"""One dataclass holds every knob; ``default_config(method)`` fills in values that are known
to work on CartPole-v1 so that ``cartpole-train --method X`` works with no other flag.

Why one flat config instead of one class per algorithm: the point of the repo is to compare
methods, so it helps to see in one place which knobs each method actually reads (the
``METHOD_DEFAULTS`` table below is that list).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

# The methods, grouped by family. The dqn variants are one algorithm with flags.
TABULAR = ("q-learning", "sarsa", "mc-control")
VALUE_BASED = ("dqn", "double-dqn", "dueling-dqn")
POLICY_GRADIENT = ("reinforce", "reinforce-baseline")
ACTOR_CRITIC = ("actor-critic", "a2c", "ppo")
METHODS = TABULAR + VALUE_BASED + POLICY_GRADIENT + ACTOR_CRITIC

OBS_MODES = ("state", "frame", "frame-diff")
NETS = ("mlp", "cnn", "rescnn")

SOLVED_RETURN = 475.0  # gymnasium's reward_threshold for CartPole-v1


@dataclass
class TrainConfig:
    method: str = "ppo"
    obs: str = "state"            # state | frame | frame-diff
    net: str = "auto"             # mlp | cnn | rescnn (auto: mlp for state, cnn for pixels)
    seed: int = 0
    threads: int = 1
    total_steps: int = 100_000    # environment-step budget (all methods count the same way)
    gamma: float = 0.99
    lr: float = 1e-3
    hidden: int = 128

    # evaluation (shared by every method)
    eval_every: int = 5_000       # env steps between greedy evaluations
    eval_episodes: int = 10
    final_eval_episodes: int = 100
    stop_when_solved: bool = True  # stop once a greedy eval reaches SOLVED_RETURN

    # tabular
    bins: tuple[int, ...] = (1, 6, 12, 12)  # bins for (x, x_dot, theta, theta_dot)
    alpha: float = 0.1
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_decay_steps: int = 20_000

    # dqn family
    buffer_size: int = 50_000
    batch_size: int = 64
    learning_starts: int = 1_000
    train_every: int = 1
    tau: float = 0.005            # soft target update
    double: bool = False
    dueling: bool = False
    prioritized: bool = False
    n_step: int = 1
    per_alpha: float = 0.6
    per_beta: float = 0.4

    # policy gradient / actor-critic
    critic_lr: float = 5e-3
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    n_envs: int = 8
    n_steps: int = 32             # rollout length per env (a2c, ppo)
    gae_lambda: float = 0.95
    clip: float = 0.2
    ppo_epochs: int = 10
    minibatch_size: int = 256
    max_grad_norm: float = 0.5

    # pixels
    frame_height: int = 40
    frame_width: int = 80

    output_dir: str = "checkpoints/run"
    extra: dict = field(default_factory=dict)

    def resolved_net(self) -> str:
        if self.net != "auto":
            return self.net
        return "mlp" if self.obs == "state" else "cnn"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bins"] = list(self.bins)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        names = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in names}
        if "bins" in d:
            d["bins"] = tuple(d["bins"])
        return cls(**d)


# Values that solve CartPole-v1 on a CPU in a few minutes. Only the knobs a method reads.
METHOD_DEFAULTS: dict[str, dict] = {
    "q-learning": dict(total_steps=400_000, alpha=0.1, gamma=0.99, eps_decay_steps=150_000),
    "sarsa": dict(total_steps=400_000, alpha=0.1, gamma=0.99, eps_decay_steps=150_000),
    "mc-control": dict(total_steps=400_000, alpha=0.05, gamma=0.99, eps_decay_steps=150_000),
    "dqn": dict(total_steps=80_000, lr=1e-3, eps_decay_steps=15_000, train_every=2),
    "double-dqn": dict(total_steps=80_000, lr=1e-3, eps_decay_steps=15_000, train_every=2, double=True),
    "dueling-dqn": dict(total_steps=80_000, lr=1e-3, eps_decay_steps=15_000, train_every=2, double=True,
                        dueling=True),
    "reinforce": dict(total_steps=300_000, lr=2e-3),
    "reinforce-baseline": dict(total_steps=300_000, lr=2e-3, critic_lr=5e-3),
    "actor-critic": dict(total_steps=100_000, lr=5e-4, critic_lr=2e-3),
    "a2c": dict(total_steps=300_000, lr=1e-3, n_envs=8, n_steps=8, gae_lambda=1.0,
                entropy_coef=0.01),
    "ppo": dict(total_steps=150_000, lr=1e-3, n_envs=8, n_steps=32, gamma=0.98, gae_lambda=0.8,
                ppo_epochs=20, minibatch_size=256, clip=0.2),
}

# Pixel runs are slow (rendering + CNN): short budgets, smaller learning rate.
PIXEL_DEFAULTS = dict(total_steps=50_000, eval_every=10_000, eval_episodes=5, final_eval_episodes=20,
                      lr=3e-4, ppo_epochs=4, n_steps=64, minibatch_size=128,   # ppo / a2c
                      buffer_size=10_000, train_every=4)                          # dqn


def default_config(method: str, obs: str = "state", **overrides) -> TrainConfig:
    """A ready-to-train config for ``method``; keyword overrides win over the defaults."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {', '.join(METHODS)}")
    if obs not in OBS_MODES:
        raise ValueError(f"unknown obs mode {obs!r}; choose from {', '.join(OBS_MODES)}")
    if obs != "state" and method in TABULAR:
        raise ValueError("tabular methods need a discretized state: use --obs state")
    values = dict(METHOD_DEFAULTS[method])
    if obs != "state":
        values.update(PIXEL_DEFAULTS)
    values.update({k: v for k, v in overrides.items() if v is not None})
    return TrainConfig(method=method, obs=obs, **values)

"""What a trained agent is: something with ``act(obs) -> action``.

Three kinds, one interface, so evaluation, recording and checkpoints never care which
algorithm produced the agent:
  * ``TabularAgent``  a Q table indexed by the discretized state
  * ``QAgent``        a Q network; act = argmax_a Q(s, a)
  * ``PolicyAgent``   a policy network; act = argmax (greedy) or a sample from pi(.|s)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from cartpole_rl.config import POLICY_GRADIENT, TABULAR, VALUE_BASED, TrainConfig
from cartpole_rl.env import Discretizer
from cartpole_rl.networks import ActorCritic, QNet

# Argument validation in torch.distributions costs more than the tiny networks themselves when
# acting one step at a time; our inputs are always valid logits.
torch.distributions.Distribution.set_default_validate_args(False)


def to_tensor(obs) -> torch.Tensor:
    return torch.as_tensor(np.asarray(obs), dtype=torch.float32)


class TabularAgent:
    def __init__(self, bins, n_actions: int = 2):
        self.disc = Discretizer(bins)
        self.q = np.zeros((self.disc.n_states, n_actions))
        self.n_actions = n_actions

    def act(self, obs, greedy: bool = True) -> int:
        return int(np.argmax(self.q[self.disc(obs)]))

    def state_dict(self):
        return {"q": torch.from_numpy(self.q.copy())}

    def load_state_dict(self, d):
        self.q = d["q"].numpy().copy()


class QAgent:
    def __init__(self, model: QNet):
        self.model = model

    @torch.no_grad()
    def act(self, obs, greedy: bool = True) -> int:
        return int(self.model(to_tensor(obs)[None]).argmax(1).item())

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, d):
        self.model.load_state_dict(d)


class PolicyAgent:
    def __init__(self, model: ActorCritic):
        self.model = model

    @torch.no_grad()
    def act(self, obs, greedy: bool = True) -> int:
        logits = self.model.logits(to_tensor(obs)[None])
        if greedy:
            return int(logits.argmax(1).item())
        return int(Categorical(logits=logits).sample().item())

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, d):
        self.model.load_state_dict(d)


def build_agent(cfg: TrainConfig, obs_shape, n_actions: int = 2):
    """The untrained agent for ``cfg.method`` (also used to rebuild one from a checkpoint)."""
    if cfg.method in TABULAR:
        return TabularAgent(cfg.bins, n_actions)
    net = cfg.resolved_net()
    if cfg.method in VALUE_BASED:
        return QAgent(QNet(obs_shape, n_actions, net, cfg.hidden, dueling=cfg.dueling))
    critic = cfg.method != "reinforce"
    assert cfg.method in POLICY_GRADIENT or critic
    return PolicyAgent(ActorCritic(obs_shape, n_actions, net, cfg.hidden, critic=critic))

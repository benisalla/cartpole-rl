import numpy as np
import pytest
import torch

from cartpole_rl.agents import build_agent
from cartpole_rl.config import METHODS, default_config
from cartpole_rl.networks import ActorCritic, CNNBody, QNet


@pytest.mark.parametrize("net", ["cnn", "rescnn"])
@pytest.mark.parametrize("channels", [1, 2])
def test_cnn_shapes(net, channels):
    q = QNet((channels, 40, 80), 2, net, 64)
    ac = ActorCritic((channels, 40, 80), 2, net, 64)
    x = torch.rand(3, channels, 40, 80)
    assert q(x).shape == (3, 2)
    logits, v = ac(x)
    assert logits.shape == (3, 2) and v.shape == (3,)
    assert ac.shared  # pixels: one CNN for actor and critic


def test_residual_block_has_skip():
    body = CNNBody((2, 40, 80), residual=True)
    assert any(type(m).__name__ == "ResidualBlock" for m in body.modules())


def test_actor_critic_parameter_split_is_disjoint():
    ac = ActorCritic((4,), 2)
    a = {id(p) for p in ac.actor_parameters()}
    c = {id(p) for p in ac.critic_parameters()}
    assert a and c and not (a & c)
    assert len(a) + len(c) == len(list(ac.parameters()))


@pytest.mark.parametrize("method", METHODS)
def test_every_method_builds_an_agent_that_acts(method):
    agent = build_agent(default_config(method), (4,))
    a = agent.act(np.zeros(4, np.float32))
    assert a in (0, 1)


def test_tabular_rejects_pixels():
    with pytest.raises(ValueError):
        default_config("q-learning", obs="frame")

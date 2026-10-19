"""Neural networks: one *body* (state -> features) and small *heads* on top.

  body  mlp     4 -> hidden -> hidden                       (state observations)
        cnn     C x 40 x 80 -> 3 convs -> hidden              (pixel observations)
        rescnn  same, with a residual block after the first conv (the notebook's ResNet idea)
  heads QNet         features -> Q(s, a) for each action  (optionally dueling: V(s) + A(s, a))
        ActorCritic  features -> policy logits  (+ V(s) if critic=True)

Heads output *logits*, never probabilities: ``Categorical(logits=...)`` computes log-softmax
in a numerically safe way, while ``log(softmax(x))`` can hit log(0).
"""

from __future__ import annotations

import torch
from torch import nn


class MLPBody(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.out_dim = hidden

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """y = relu(x + conv(relu(conv(x)))): the skip lets gradients bypass the two convs."""

    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return torch.relu(x + self.c2(torch.relu(self.c1(x))))


class CNNBody(nn.Module):
    """(B, C, 40, 80) -> conv 5x5/2 -> (16, 18, 38) -> conv 5x5/2 -> (32, 7, 17)
    -> conv 3x3 -> (32, 5, 15) -> flatten 2400 -> Linear(hidden)."""

    def __init__(self, in_shape, hidden: int = 128, residual: bool = False):
        super().__init__()
        c = in_shape[0]
        layers = [nn.Conv2d(c, 16, 5, stride=2), nn.ReLU()]
        if residual:
            layers.append(ResidualBlock(16))
        layers += [nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(), nn.Conv2d(32, 32, 3), nn.ReLU(), nn.Flatten()]
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, *in_shape)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n_flat, hidden), nn.ReLU())
        self.out_dim = hidden

    def forward(self, x):
        return self.fc(self.conv(x))


def make_body(obs_shape, net: str = "mlp", hidden: int = 128) -> nn.Module:
    if net == "mlp":
        return MLPBody(int(obs_shape[0]), hidden)
    if net in ("cnn", "rescnn"):
        if len(obs_shape) != 3:
            raise ValueError("cnn bodies need pixel observations (--obs frame or frame-diff)")
        return CNNBody(tuple(obs_shape), hidden, residual=net == "rescnn")
    raise ValueError(f"unknown net {net!r}")


class QNet(nn.Module):
    """Q(s, .) for every action. Dueling: Q = V(s) + A(s, a) - mean_a A(s, a).

    Why dueling helps: in many CartPole states both actions are about equally good; the
    V stream learns "how good is this state" from every transition, whatever the action.
    Subtracting the mean advantage makes the V / A split unique.
    """

    def __init__(self, obs_shape, n_actions: int, net: str = "mlp", hidden: int = 128, dueling: bool = False):
        super().__init__()
        self.body = make_body(obs_shape, net, hidden)
        self.dueling = dueling
        if dueling:
            self.value = nn.Linear(self.body.out_dim, 1)
            self.adv = nn.Linear(self.body.out_dim, n_actions)
        else:
            self.q = nn.Linear(self.body.out_dim, n_actions)

    def forward(self, x):
        h = self.body(x)
        if self.dueling:
            a = self.adv(h)
            return self.value(h) + a - a.mean(dim=1, keepdim=True)
        return self.q(h)


class ActorCritic(nn.Module):
    """Policy logits and (optionally) a state value V(s).

    For state inputs the actor and critic get separate MLP bodies (their gradients do not
    fight); for pixels they share one CNN body, which halves the cost of the convolutions.
    """

    def __init__(self, obs_shape, n_actions: int, net: str = "mlp", hidden: int = 128, critic: bool = True):
        super().__init__()
        self.has_critic = critic
        self.actor_body = make_body(obs_shape, net, hidden)
        self.shared = net != "mlp"
        self.pi = nn.Linear(self.actor_body.out_dim, n_actions)
        if critic:
            self.critic_body = self.actor_body if self.shared else make_body(obs_shape, net, hidden)
            self.v = nn.Linear(self.critic_body.out_dim, 1)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)  # start close to a uniform policy
        nn.init.zeros_(self.pi.bias)

    def logits(self, x):
        return self.pi(self.actor_body(x))

    def value(self, x):
        return self.v(self.critic_body(x)).squeeze(-1)

    def forward(self, x):
        h = self.actor_body(x)
        logits = self.pi(h)
        if not self.has_critic:
            return logits, None
        hv = h if self.shared else self.critic_body(x)
        return logits, self.v(hv).squeeze(-1)

    def actor_parameters(self):
        return list(self.actor_body.parameters()) + list(self.pi.parameters())

    def critic_parameters(self):
        if not self.has_critic:
            return []
        own = [] if self.shared else list(self.critic_body.parameters())
        return own + list(self.v.parameters())

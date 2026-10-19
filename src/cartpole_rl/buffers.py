"""Experience replay for the DQN family.

Why replay: consecutive CartPole states are almost identical; training on them in order is
like doing SGD on a sorted data set. Sampling random mini-batches from the last N transitions
breaks the correlation and lets every transition be reused many times.

  ReplayBuffer       uniform sampling, a ring buffer of numpy arrays.
  PrioritizedReplay  sample transition i with p_i ~ (|delta_i| + eps)^alpha (big TD error =
                     more to learn), and correct the bias with importance weights
                     w_i = (N * P(i))^-beta / max_j w_j, beta annealed to 1.
  NStepCollector     turns 1-step transitions into n-step ones:
                     R = r_t + gamma r_{t+1} + ... + gamma^{n-1} r_{t+n-1}, bootstrap from s_{t+n}
                     with discount gamma^n. Rewards travel n steps back per update instead of 1.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape, seed: int = 0, dtype=np.float32):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)       # 1 only for a real termination
        self.discounts = np.zeros(capacity, dtype=np.float32)   # gamma^k (k = n for n-step)
        self.pos, self.size = 0, 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add(self, obs, action, reward, next_obs, done, discount):
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.dones[i], self.discounts[i] = next_obs, float(done), discount
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return i

    def _batch(self, idx):
        return {
            "obs": self.obs[idx].astype(np.float32), "actions": self.actions[idx],
            "rewards": self.rewards[idx], "next_obs": self.next_obs[idx].astype(np.float32),
            "dones": self.dones[idx], "discounts": self.discounts[idx],
            "weights": np.ones(len(idx), dtype=np.float32), "idx": idx,
        }

    def sample(self, batch_size: int, beta: float = 0.4):
        return self._batch(self.rng.integers(0, self.size, size=batch_size))

    def update_priorities(self, idx, td_errors):
        pass  # uniform replay ignores priorities


class PrioritizedReplay(ReplayBuffer):
    def __init__(self, capacity, obs_shape, alpha: float = 0.6, seed: int = 0, dtype=np.float32):
        super().__init__(capacity, obs_shape, seed, dtype)
        self.alpha = alpha
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.max_priority = 1.0

    def add(self, *args):
        i = super().add(*args)
        self.priorities[i] = self.max_priority  # new transitions are seen at least once
        return i

    def sample(self, batch_size, beta=0.4):
        p = self.priorities[: self.size] ** self.alpha
        p /= p.sum()
        idx = self.rng.choice(self.size, size=batch_size, p=p)
        batch = self._batch(idx)
        w = (self.size * p[idx]) ** (-beta)
        batch["weights"] = (w / w.max()).astype(np.float32)
        return batch

    def update_priorities(self, idx, td_errors):
        prio = np.abs(td_errors) + 1e-6
        self.priorities[idx] = prio
        self.max_priority = max(self.max_priority, float(prio.max()))


class NStepCollector:
    """Feed 1-step transitions in, get n-step transitions out (into ``buffer``)."""

    def __init__(self, buffer: ReplayBuffer, n: int, gamma: float):
        self.buffer, self.n, self.gamma = buffer, n, gamma
        self.queue: deque = deque()

    def _emit(self, next_obs, done):
        R, disc = 0.0, 1.0
        for (_, _, r) in self.queue:
            R += disc * r
            disc *= self.gamma
        obs0, a0, _ = self.queue[0]
        self.buffer.add(obs0, a0, R, next_obs, done, disc)
        self.queue.popleft()

    def add(self, obs, action, reward, next_obs, terminated: bool, truncated: bool):
        self.queue.append((obs, action, reward))
        if len(self.queue) == self.n:
            self._emit(next_obs, terminated)
        if terminated or truncated:  # flush the shorter tails at the end of the episode
            while self.queue:
                self._emit(next_obs, terminated)

"""Traffic-signal controllers.

Baselines (no learning):
    FixedTimeAgent  - classic pre-timed signal, switches every `green_time` seconds.
    ActuatedAgent   - after a minimum green, serves the direction with the longer
                      queue, like a sensor-actuated signal.
Reinforcement-learning agents:
    QLearningAgent  - tabular Q-learning on a discretised state.
    DQNAgent        - Deep Q-Network; `double=True` gives Double DQN and
                      `dueling=True` adds a dueling value/advantage head.

Every agent has the same interface, so training, evaluation and the demo
treat them all alike:
    act(obs, greedy=False) -> action
    update(obs, action, reward, next_obs, done)   (does nothing for baselines)
    save(path) / load(path)                       (does nothing for baselines)
    epsilon                                       (current exploration rate)
"""
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from traffic_env import QUEUE_NORM, TIME_NORM


def linear_decay(start, end, duration, step):
    """Anneal linearly from `start` to `end` over `duration` steps, then hold `end`."""
    if duration <= 0:
        return end
    return start + min(step / duration, 1.0) * (end - start)


def current_phase(obs):
    return int(np.argmax(obs[4:6]))


def seconds_in_phase(obs):
    return int(round(float(obs[6]) * TIME_NORM))


# ------------------------------------------------------------------ baselines
class BaselineAgent:
    """Hand-written controller: acts from the observation, never learns."""

    learns = False
    epsilon = 0.0

    def update(self, obs, action, reward, next_obs, done):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


class FixedTimeAgent(BaselineAgent):
    """Pre-timed signal: switch phase every `green_time` seconds, ignoring traffic."""

    def __init__(self, green_time=30):
        if green_time >= TIME_NORM:
            raise ValueError("green_time must be below TIME_NORM (the state clips there)")
        self.green_time = green_time

    def act(self, obs, greedy=True):
        phase = current_phase(obs)
        return 1 - phase if seconds_in_phase(obs) >= self.green_time else phase


class ActuatedAgent(BaselineAgent):
    """Queue-actuated signal: after `min_green` seconds, give green to the longer queue."""

    def __init__(self, min_green=10):
        self.min_green = min_green

    def act(self, obs, greedy=True):
        phase = current_phase(obs)
        if seconds_in_phase(obs) < self.min_green:
            return phase
        north_south, east_west = obs[0] + obs[1], obs[2] + obs[3]
        if north_south == east_west:
            return phase
        return 0 if north_south > east_west else 1


# ----------------------------------------------------------------- Q-learning
class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration.

    A table needs a small, discrete state, so the observation is compressed:
        North-South queue (N + S) -> 6 bins (0 | 1-3 | 4-7 | 8-13 | 14-21 | 22+ vehicles)
        East-West queue (E + W)   -> 6 bins (same edges)
        current phase             -> 2 values
        time in phase             -> 4 bins (<10 s | 10-19 s | 20-39 s | 40+ s)
    giving 6 * 6 * 2 * 4 = 288 states x 2 actions. (Binning each lane separately
    would give thousands of states, too many to visit in 150 episodes.)
    """

    learns = True
    QUEUE_EDGES = np.array([1, 4, 8, 14, 22])
    TIME_EDGES = np.array([10, 20, 40])

    def __init__(self, n_actions=2, total_steps=1, seed=0, alpha=0.1, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay_frac=0.6):
        self.n_actions = n_actions
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_steps = int(eps_decay_frac * total_steps)
        self.rng = np.random.default_rng(seed)
        n_queue_bins, n_time_bins = len(self.QUEUE_EDGES) + 1, len(self.TIME_EDGES) + 1
        self.q_table = np.zeros((n_queue_bins, n_queue_bins, 2, n_time_bins, n_actions))
        self.steps = 0

    @property
    def epsilon(self):
        return linear_decay(self.eps_start, self.eps_end, self.eps_decay_steps, self.steps)

    def discretise(self, obs):
        queues = np.rint(obs[:4] * QUEUE_NORM)
        north_south, east_west = np.digitize([queues[0] + queues[1], queues[2] + queues[3]],
                                             self.QUEUE_EDGES)
        time_bin = int(np.digitize(seconds_in_phase(obs), self.TIME_EDGES))
        return (int(north_south), int(east_west), current_phase(obs), time_bin)

    def act(self, obs, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q_values = self.q_table[self.discretise(obs)]
        best = np.flatnonzero(q_values == q_values.max())
        return int(self.rng.choice(best))  # break ties randomly

    def update(self, obs, action, reward, next_obs, done):
        state, next_state = self.discretise(obs), self.discretise(next_obs)
        target = reward if done else reward + self.gamma * self.q_table[next_state].max()
        index = state + (action,)
        self.q_table[index] += self.alpha * (target - self.q_table[index])
        self.steps += 1

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)


# ------------------------------------------------------------------------ DQN
class QNetwork(nn.Module):
    """MLP that maps a state to one Q-value per action.

    With `dueling=True` the last layer is split into a state value V(s) and
    advantages A(s, a), recombined as Q = V + A - mean(A).
    """

    def __init__(self, obs_dim, n_actions, hidden=128, dueling=False):
        super().__init__()
        self.dueling = dueling
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        if dueling:
            self.value = nn.Linear(hidden, 1)
            self.advantage = nn.Linear(hidden, n_actions)
        else:
            self.head = nn.Linear(hidden, n_actions)

    def forward(self, x):
        features = self.body(x)
        if not self.dueling:
            return self.head(features)
        advantage = self.advantage(features)
        return self.value(features) + advantage - advantage.mean(dim=1, keepdim=True)


class ReplayBuffer:
    """Fixed-size circular buffer of transitions, sampled uniformly."""

    def __init__(self, capacity, obs_dim, rng):
        self.capacity, self.rng = capacity, rng
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = self.pos = 0

    def add(self, obs, action, reward, next_obs, done):
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.dones[i] = next_obs, float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (torch.from_numpy(self.obs[idx]),
                torch.from_numpy(self.actions[idx]).unsqueeze(1),
                torch.from_numpy(self.rewards[idx]).unsqueeze(1),
                torch.from_numpy(self.next_obs[idx]),
                torch.from_numpy(self.dones[idx]).unsqueeze(1))


class DQNAgent:
    """Deep Q-Network (Mnih et al., 2015) with optional Double and Dueling extensions.

    double=True  -> Double DQN target (van Hasselt et al., 2016): the online network
                    picks the next action, the target network evaluates it, which
                    reduces the over-estimation of Q-values.
    dueling=True -> Dueling architecture (Wang et al., 2016).
    """

    learns = True

    def __init__(self, obs_dim=7, n_actions=2, total_steps=1, seed=0, double=False,
                 dueling=False, hidden=128, lr=5e-4, gamma=0.95, batch_size=64,
                 buffer_size=50_000, learning_starts=1_000, train_freq=1,
                 target_update=1_000, eps_start=1.0, eps_end=0.05,
                 eps_decay_frac=0.5, grad_clip=10.0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions
        self.double = double
        self.gamma, self.batch_size, self.grad_clip = gamma, batch_size, grad_clip
        self.learning_starts, self.train_freq = learning_starts, train_freq
        self.target_update = target_update
        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_steps = int(eps_decay_frac * total_steps)

        self.q_net = QNetwork(obs_dim, n_actions, hidden, dueling)
        self.target_net = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, obs_dim, self.rng)
        self.steps = 0

    @property
    def epsilon(self):
        return linear_decay(self.eps_start, self.eps_end, self.eps_decay_steps, self.steps)

    def act(self, obs, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            q_values = self.q_net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(q_values.argmax(dim=1).item())

    def update(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)
        self.steps += 1
        if self.steps >= self.learning_starts and self.steps % self.train_freq == 0:
            self._learn()
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def _learn(self):
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        q_taken = self.q_net(obs).gather(1, actions)
        with torch.no_grad():
            if self.double:
                next_actions = self.q_net(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_obs).gather(1, next_actions)
            else:
                next_q = self.target_net(next_obs).max(dim=1, keepdim=True).values
            target = rewards + self.gamma * (1.0 - dones) * next_q
        loss = F.smooth_l1_loss(q_taken, target)  # Huber loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path, map_location="cpu")
        self.q_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)


# -------------------------------------------------------------------- factory
def make_agent(name, obs_dim=7, n_actions=2, total_steps=1, seed=0):
    """Build an agent by name using the hyperparameters in config.py."""
    if name == "fixed_time":
        return FixedTimeAgent(**config.FIXED_TIME)
    if name == "actuated":
        return ActuatedAgent(**config.ACTUATED)
    if name == "qlearning":
        return QLearningAgent(n_actions, total_steps, seed, **config.QLEARNING)
    if name in config.DQN_VARIANTS:
        return DQNAgent(obs_dim, n_actions, total_steps, seed,
                        **config.DQN_VARIANTS[name], **config.DQN)
    raise ValueError(f"unknown agent {name!r}; choose one of {config.ALL_AGENTS}")


def model_path(name, seed):
    extension = ".npy" if name == "qlearning" else ".pt"
    return os.path.join(config.MODELS_DIR, f"{name}_seed{seed}{extension}")


def load_trained_agent(name, seed=0):
    """Return a ready-to-run agent; learning agents load their saved weights."""
    agent = make_agent(name, seed=seed)
    if agent.learns:
        path = model_path(name, seed)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found - train it first: python train.py --agents {name} --seeds {seed}")
        agent.load(path)
    return agent

"""Single four-way intersection simulator, exposed as a Gymnasium environment.

Traffic model
-------------
* Four one-lane approaches: North, South, East, West.
* Two signal phases: 0 = North-South green, 1 = East-West green.
* Vehicles arrive as a Poisson process whose rate can change over time
  (the demand *scenario*).
* A green lane discharges ``saturation_flow`` vehicles per second
  (0.5 veh/s = 1800 veh/h, a typical urban value).
* Changing phase costs ``yellow_time`` seconds during which no lane moves.
* Safety rule (as in real controllers): once a phase has lasted ``max_green``
  seconds it is switched even if the agent asks to extend it, so no approach
  can be starved.
* Time advances in 1-second ticks; the agent acts every ``decision_interval`` ticks.

MDP seen by the agent
---------------------
State  : 7 numbers in [0, 1]
         queue length of N, S, E, W divided by QUEUE_NORM (clipped),
         one-hot current phase (2 values),
         seconds spent in the current phase divided by TIME_NORM (clipped).
Action : 0 = North-South green, 1 = East-West green.
         Choosing the phase that is already green extends it.
Reward : -(vehicle-seconds spent waiting during the interval) / REWARD_SCALE.
         The episode return is proportional to the total delay of all drivers,
         so maximising return is the same as minimising waiting time.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

LANES = ("N", "S", "E", "W")
PHASE_LANES = {0: (0, 1), 1: (2, 3)}  # phase -> lanes that get green
QUEUE_NORM = 40.0     # queue length that maps to 1.0 in the state
TIME_NORM = 60.0      # time-in-phase (s) that maps to 1.0 in the state
REWARD_SCALE = 100.0  # keeps per-step rewards roughly in [-5, 0]
SCENARIOS = ("uniform", "asymmetric", "rush_hour", "random")


def make_demand(scenario, rng, episode_length):
    """Return a function t -> arrival rate (veh/s) for lanes N, S, E, W."""
    T = float(episode_length)

    if scenario == "uniform":
        rates = np.full(4, 0.10)
        return lambda t: rates

    if scenario == "asymmetric":
        # A main road (North-South) crossing a side street.
        rates = np.array([0.20, 0.20, 0.06, 0.06])
        return lambda t: rates

    if scenario == "rush_hour":
        # North-South demand peaks a third of the way in, East-West two thirds in.
        def rush_hour(t):
            ns = 0.04 + 0.18 * np.exp(-(((t - T / 3) / (T / 8)) ** 2))
            ew = 0.04 + 0.18 * np.exp(-(((t - 2 * T / 3) / (T / 8)) ** 2))
            return np.array([ns, ns, ew, ew])
        return rush_hour

    if scenario == "random":
        # Training demand: a new random pattern every episode, so the agent
        # has to learn a general policy instead of memorising one timetable.
        base = rng.uniform(0.03, 0.18, size=2)      # mean rate of NS and EW lanes
        lane_skew = rng.uniform(0.85, 1.15, size=4)  # N and S (E and W) differ a little
        amplitude = rng.uniform(0.0, 0.5)            # how much demand swings over time
        shift = rng.uniform(0.0, 2 * np.pi)

        def random_demand(t):
            wave = amplitude * np.sin(2 * np.pi * t / T + shift)
            ns, ew = base[0] * (1 + wave), base[1] * (1 - wave)
            return np.array([ns, ns, ew, ew]) * lane_skew
        return random_demand

    raise ValueError(f"unknown scenario {scenario!r}; choose one of {SCENARIOS}")


class TrafficIntersectionEnv(gym.Env):
    """Four-way signalised intersection (see module docstring for the MDP)."""

    metadata = {"render_modes": []}

    def __init__(self, scenario="random", episode_length=1800, decision_interval=5,
                 yellow_time=3, saturation_flow=0.5, max_green=60):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; choose one of {SCENARIOS}")
        if yellow_time >= decision_interval:
            raise ValueError("yellow_time must be shorter than decision_interval")
        self.scenario = scenario
        self.episode_length = episode_length
        self.decision_interval = decision_interval
        self.yellow_time = yellow_time
        self.saturation_flow = saturation_flow
        self.max_green = max_green

        self.observation_space = spaces.Box(0.0, 1.0, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.queues = np.zeros(len(LANES), dtype=np.int64)

    # ------------------------------------------------------------ Gymnasium API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.demand = make_demand(self.scenario, self.np_random, self.episode_length)
        self.queues = np.zeros(len(LANES), dtype=np.int64)
        self.departed_per_lane = np.zeros(len(LANES), dtype=np.int64)
        self._credit = np.zeros(len(LANES))  # fractional discharge capacity per lane
        self.t = 0
        self.phase = 0
        self.time_in_phase = 0
        self.yellow_left = 0
        self.arrived = 0
        self.departed = 0
        self.switches = 0
        self.forced_switches = 0  # switches imposed by the max-green rule
        self.delay = 0  # vehicle-seconds waited so far, by every vehicle
        self.last_rates = self.demand(0)
        return self.get_obs(), self.metrics()

    def step(self, action):
        self.apply_action(action)
        delay_before = self.delay
        for _ in range(self.decision_interval):
            self.tick()
            if self.t >= self.episode_length:
                break
        reward = -(self.delay - delay_before) / REWARD_SCALE
        truncated = self.t >= self.episode_length  # time limit, not a terminal state
        return self.get_obs(), reward, False, truncated, self.metrics()

    # ------------------------------------------------------- simulation pieces
    # step() is built from these so the demo can animate one second at a time.
    def apply_action(self, action):
        """Set the requested phase; switching starts a yellow period."""
        action = int(action)
        if action not in PHASE_LANES:
            raise ValueError(f"invalid action {action}")
        if action == self.phase and self.time_in_phase >= self.max_green:
            action = 1 - self.phase  # max-green safety rule overrides the controller
            self.forced_switches += 1
        if action != self.phase:
            self.phase = action
            self.time_in_phase = 0
            self.yellow_left = self.yellow_time
            self.switches += 1

    def tick(self):
        """Advance the simulation by one second."""
        rates = self.demand(self.t)
        self.last_rates = rates
        arrivals = self.np_random.poisson(rates)
        self.queues += arrivals
        self.arrived += int(arrivals.sum())

        if self.yellow_left > 0:
            self.yellow_left -= 1
            green = ()
        else:
            green = PHASE_LANES[self.phase]

        for lane in range(len(LANES)):
            if lane not in green:
                self._credit[lane] = 0.0  # start-up loss: a red lane builds no capacity
                continue
            self._credit[lane] += self.saturation_flow
            leaving = min(int(self._credit[lane] + 1e-9), int(self.queues[lane]))
            self.queues[lane] -= leaving
            self.departed_per_lane[lane] += leaving
            self.departed += leaving
            self._credit[lane] -= leaving
            if self.queues[lane] == 0:
                # Unused capacity cannot be banked while the lane is empty.
                self._credit[lane] = min(self._credit[lane], 1.0)

        # Every vehicle still queued has waited one more second.
        self.delay += int(self.queues.sum())
        self.time_in_phase += 1
        self.t += 1

    def get_obs(self):
        queues = np.clip(self.queues / QUEUE_NORM, 0.0, 1.0)
        phase = (1.0, 0.0) if self.phase == 0 else (0.0, 1.0)
        in_phase = min(self.time_in_phase / TIME_NORM, 1.0)
        return np.array([*queues, *phase, in_phase], dtype=np.float32)

    def metrics(self):
        """Performance so far. avg_wait counts vehicles still queued with their wait so far."""
        return {
            "time": self.t,
            "avg_wait": self.delay / max(self.arrived, 1),  # s of delay per vehicle
            "avg_queue": self.delay / max(self.t, 1),       # vehicles waiting, time-averaged
            "throughput": self.departed,
            "arrived": self.arrived,
            "waiting": int(self.queues.sum()),
            "switches": self.switches,
            "forced_switches": self.forced_switches,
        }

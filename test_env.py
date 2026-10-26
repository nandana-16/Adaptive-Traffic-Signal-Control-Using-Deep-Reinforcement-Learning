"""Sanity checks for the simulator and baselines.

    python test_env.py        (or: python -m pytest test_env.py)
"""
import numpy as np
from gymnasium.utils.env_checker import check_env

import config
from agents import FixedTimeAgent
from traffic_env import REWARD_SCALE, TrafficIntersectionEnv


def make_env(scenario="random"):
    return TrafficIntersectionEnv(scenario=scenario, **config.ENV_CONFIG)


def test_gymnasium_api():
    check_env(make_env(), skip_render_check=True)


def test_same_seed_gives_same_arrivals_whatever_the_actions():
    # Fair comparison: every controller must face exactly the same vehicles.
    arrivals = []
    for policy in (lambda t: 0, lambda t: 1, lambda t: t % 2):
        env = make_env()
        env.reset(seed=123)
        for t in range(50):
            env.step(policy(t))
        arrivals.append(env.arrived)
    assert arrivals[0] == arrivals[1] == arrivals[2]


def test_vehicles_are_conserved():
    env = make_env("rush_hour")
    env.reset(seed=1)
    rng = np.random.default_rng(0)
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(int(rng.integers(2)))
        assert info["arrived"] == info["throughput"] + info["waiting"]
        done = terminated or truncated


def test_return_equals_total_delay():
    env = make_env("uniform")
    env.reset(seed=2)
    total, done = 0.0, False
    while not done:
        _, reward, terminated, truncated, info = env.step(0)
        total += reward
        done = terminated or truncated
    assert np.isclose(total, -env.delay / REWARD_SCALE)
    assert np.isclose(info["avg_wait"], env.delay / env.arrived)


def test_nobody_moves_during_yellow():
    env = make_env("uniform")
    env.reset(seed=3)
    for _ in range(20):
        env.step(0)
    env.apply_action(1)  # switch -> yellow
    for _ in range(env.yellow_time):
        before = env.departed
        env.tick()
        assert env.departed == before


def test_green_lane_discharges_at_saturation_flow():
    env = make_env("uniform")
    env.reset(seed=4)
    env.queues[:] = 100  # long queues so the lanes never empty
    env.demand = lambda t: np.zeros(4)  # no new arrivals
    for _ in range(20):
        env.tick()
    # 20 s of green on two lanes at 0.5 veh/s each = 20 vehicles.
    assert env.departed == 20
    assert env.queues[0] == env.queues[1] == 90 and env.queues[2] == env.queues[3] == 100


def test_fixed_time_switches_on_schedule():
    env = make_env("uniform")
    agent = FixedTimeAgent(green_time=30)
    obs, _ = env.reset(seed=5)
    for _ in range(360):
        obs, *_ = env.step(agent.act(obs))
    # 1800 s with a 30 s phase length -> a switch every 30 s.
    assert env.switches == 1800 // 30 - 1
    assert env.forced_switches == 0


def test_max_green_prevents_starvation():
    # A controller that always asks to extend the current phase is overridden.
    env = make_env("uniform")
    env.reset(seed=6)
    for _ in range(360):
        env.step(env.phase)
    assert env.switches == env.forced_switches == 1800 // env.max_green - 1


if __name__ == "__main__":
    tests = [value for name, value in dict(globals()).items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"All {len(tests)} tests passed.")

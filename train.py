"""Train the traffic-signal controllers.

    python train.py                                # every agent, every seed in config.py
    python train.py --agents dqn double_dqn        # a subset of agents
    python train.py --seeds 0 --episodes 20        # quick smoke test

The two baselines do not learn, but they are run through exactly the same
training episodes so the learning curves have a reference on identical traffic.

Outputs
    results/models/<agent>_seed<k>.pt / .npy   trained weights / Q-table
    results/logs/<agent>_seed<k>.csv           one row per training episode
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

import config
from agents import make_agent, model_path
from traffic_env import TrafficIntersectionEnv

LOG_FIELDS = ["episode", "return", "avg_wait", "avg_queue", "throughput", "switches",
              "forced_switches", "epsilon"]


def train(agent_name, seed, episodes):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = TrafficIntersectionEnv(scenario=config.TRAIN_SCENARIO, **config.ENV_CONFIG)
    steps_per_episode = -(-env.episode_length // env.decision_interval)  # ceiling division
    agent = make_agent(agent_name, env.observation_space.shape[0], env.action_space.n,
                       total_steps=episodes * steps_per_episode, seed=seed)

    rows, start = [], time.time()
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed * 100_000 + episode)
        episode_return, done = 0.0, False
        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # Only `terminated` stops bootstrapping: hitting the time limit is not a
            # real end state, so the target still includes the value of next_obs.
            agent.update(obs, action, reward, next_obs, terminated)
            obs = next_obs
            episode_return += reward
            done = terminated or truncated

        rows.append({
            "episode": episode + 1,
            "return": round(episode_return, 4),
            "avg_wait": round(info["avg_wait"], 4),
            "avg_queue": round(info["avg_queue"], 4),
            "throughput": info["throughput"],
            "switches": info["switches"],
            "forced_switches": info["forced_switches"],
            "epsilon": round(agent.epsilon, 4),
        })
        if (episode + 1) % 10 == 0 or episode + 1 == episodes:
            recent_wait = np.mean([row["avg_wait"] for row in rows[-10:]])
            print(f"  {agent_name:<13} seed {seed}  episode {episode + 1:>4}/{episodes}"
                  f"  avg wait (last 10) {recent_wait:7.1f} s  epsilon {agent.epsilon:.2f}"
                  f"  [{time.time() - start:5.0f} s]", flush=True)

    if agent.learns:
        agent.save(model_path(agent_name, seed))
    log_path = os.path.join(config.LOGS_DIR, f"{agent_name}_seed{seed}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", nargs="+", default=config.ALL_AGENTS, choices=config.ALL_AGENTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    parser.add_argument("--episodes", type=int, default=config.TRAIN_EPISODES)
    args = parser.parse_args()

    torch.set_num_threads(1)  # tiny networks train faster on a single thread
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    total_start = time.time()
    for name in args.agents:
        print(f"Training {config.AGENT_LABELS[name]} ({len(args.seeds)} seed(s), "
              f"{args.episodes} episodes each)")
        for seed in args.seeds:
            train(name, seed, args.episodes)
    print(f"Finished in {(time.time() - total_start) / 60:.1f} min. Next: python evaluate.py")


if __name__ == "__main__":
    main()

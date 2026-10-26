"""Evaluate every controller on held-out traffic scenarios and draw the report figures.

    python evaluate.py                  # all agents, all trained seeds, all scenarios
    python evaluate.py --episodes 2     # faster

Learning agents act greedily (no exploration). Each evaluation episode uses a
fixed seed, so every controller faces exactly the same arriving vehicles.

Outputs
    results/eval_episodes.csv            one row per (agent, seed, scenario, episode)
    results/summary.csv, summary.md      mean +- std tables for the report
    results/traces/<scenario>.csv        queue lengths over one episode, per agent
    results/plots/*.png                  figures (drawn by plots.py)
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

import config
import plots
from agents import load_trained_agent
from traffic_env import TrafficIntersectionEnv

METRICS = ["avg_wait", "avg_queue", "throughput", "switches", "forced_switches", "return"]
TRACE_FIELDS = ["agent", "time", "queue_ns", "queue_ew", "demand_ns", "demand_ew", "phase"]


def run_episode(env, agent, seed, record=False):
    """Play one greedy episode; optionally record the queues after every decision."""
    obs, info = env.reset(seed=seed)
    trace, episode_return, done = [], 0.0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(agent.act(obs, greedy=True))
        episode_return += reward
        done = terminated or truncated
        if record:
            rates = env.last_rates
            trace.append({
                "time": env.t,
                "queue_ns": int(env.queues[0] + env.queues[1]),
                "queue_ew": int(env.queues[2] + env.queues[3]),
                "demand_ns": round(60 * (rates[0] + rates[1]), 3),  # veh/min, both lanes
                "demand_ew": round(60 * (rates[2] + rates[3]), 3),
                "phase": env.phase,
            })
    result = {key: info[key] for key in METRICS if key != "return"}
    result["return"] = episode_return
    return result, trace


def summarise(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["agent"])].append(row)
    summary = []
    for (scenario, agent), items in groups.items():
        entry = {"scenario": scenario, "agent": agent, "runs": len(items)}
        for metric in METRICS:
            values = np.array([item[metric] for item in items], dtype=float)
            entry[f"{metric}_mean"] = round(float(values.mean()), 3)
            entry[f"{metric}_std"] = round(float(values.std()), 3)
        summary.append(entry)
    fixed_wait = {e["scenario"]: e["avg_wait_mean"] for e in summary if e["agent"] == "fixed_time"}
    for entry in summary:
        base = fixed_wait.get(entry["scenario"])
        entry["wait_reduction_vs_fixed_pct"] = (
            round(100 * (1 - entry["avg_wait_mean"] / base), 1) if base else "")
    return summary


def write_markdown(summary, scenarios, path):
    by_key = {(e["scenario"], e["agent"]): e for e in summary}
    agents = [a for a in config.ALL_AGENTS if any((s, a) in by_key for s in scenarios)]
    tables = [
        ("Average waiting time per vehicle (s) - lower is better", "avg_wait", 1),
        ("Average number of waiting vehicles - lower is better", "avg_queue", 1),
        ("Vehicles served in 30 minutes - higher is better", "throughput", 0),
        ("Phase switches per episode", "switches", 0),
        ("Switches forced by the 60 s max-green rule (included above)", "forced_switches", 1),
    ]
    header = "| Controller | " + " | ".join(config.SCENARIO_LABELS[s] for s in scenarios) + " |"
    rule = "|---|" + "---|" * len(scenarios)
    lines = ["# Evaluation results", "",
             "Mean +- standard deviation over evaluation episodes "
             "(and over training seeds for learning agents).", ""]
    for title, metric, digits in tables:
        lines += [f"## {title}", "", header, rule]
        for agent in agents:
            cells = []
            for scenario in scenarios:
                e = by_key.get((scenario, agent))
                cells.append(f"{e[metric + '_mean']:.{digits}f} +- {e[metric + '_std']:.{digits}f}"
                             if e else "-")
            lines.append(f"| {config.AGENT_LABELS[agent]} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Reduction in average waiting time vs. fixed-time signal (%)", "", header, rule]
    for agent in agents:
        cells = []
        for scenario in scenarios:
            e = by_key.get((scenario, agent))
            value = e["wait_reduction_vs_fixed_pct"] if e else ""
            cells.append(f"{value:.1f}" if value != "" else "-")
        lines.append(f"| {config.AGENT_LABELS[agent]} | " + " | ".join(cells) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", nargs="+", default=config.ALL_AGENTS, choices=config.ALL_AGENTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    parser.add_argument("--scenarios", nargs="+", default=config.EVAL_SCENARIOS,
                        choices=config.EVAL_SCENARIOS + ["random"])
    parser.add_argument("--episodes", type=int, default=config.EVAL_EPISODES)
    args = parser.parse_args()
    os.makedirs(config.TRACES_DIR, exist_ok=True)

    rows = []
    for scenario in args.scenarios:
        env = TrafficIntersectionEnv(scenario=scenario, **config.ENV_CONFIG)
        traces = []
        for name in args.agents:
            # Baselines are deterministic given the traffic, so one "seed" is enough.
            seeds = args.seeds if name in config.LEARNING_AGENTS else [args.seeds[0]]
            for seed in seeds:
                try:
                    agent = load_trained_agent(name, seed)
                except FileNotFoundError as error:
                    print(f"  skipping: {error}")
                    continue
                for k in range(args.episodes):
                    record = k == 0 and seed == args.seeds[0]
                    result, trace = run_episode(env, agent, config.EVAL_SEED_OFFSET + k, record)
                    rows.append({"agent": name,
                                 "seed": seed if name in config.LEARNING_AGENTS else "-",
                                 "scenario": scenario, "episode": k, **result})
                    traces += [{"agent": name, **step} for step in trace]
            done = [r["avg_wait"] for r in rows if r["agent"] == name and r["scenario"] == scenario]
            if done:
                print(f"{config.SCENARIO_LABELS[scenario]:<11} {config.AGENT_LABELS[name]:<13}"
                      f" avg wait {np.mean(done):7.1f} s  (n={len(done)})")
        write_csv(os.path.join(config.TRACES_DIR, f"{scenario}.csv"), traces, TRACE_FIELDS)

    if not rows:
        print("Nothing evaluated - run python train.py first.")
        return
    write_csv(os.path.join(config.RESULTS_DIR, "eval_episodes.csv"), rows,
              ["agent", "seed", "scenario", "episode"] + METRICS)
    summary = summarise(rows)
    write_csv(os.path.join(config.RESULTS_DIR, "summary.csv"), summary, list(summary[0].keys()))
    write_markdown(summary, args.scenarios, os.path.join(config.RESULTS_DIR, "summary.md"))
    print(f"\nTables written to {os.path.join(config.RESULTS_DIR, 'summary.md')}")
    plots.make_all()


if __name__ == "__main__":
    main()

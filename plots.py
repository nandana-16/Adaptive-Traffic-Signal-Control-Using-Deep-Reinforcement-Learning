"""Report figures. Run `python plots.py` to redraw them from the saved results.

    training_curves.png   reward and waiting time per training episode
    eval_avg_wait.png     average wait per controller in each test scenario
    rush_hour_queues.png  demand and queue length over the rush-hour episode
"""
import csv
import glob
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # draw to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter

import config

# Colour-blind-checked categorical palette, assigned in this fixed order so that
# neighbouring bars and lines stay distinguishable. A controller keeps its colour
# in every figure.
AGENT_COLORS = {
    "fixed_time": "#2a78d6",    # blue
    "actuated": "#eb6834",      # orange
    "qlearning": "#1baf7a",     # aqua
    "dqn": "#eda100",           # yellow
    "double_dqn": "#e87ba4",    # magenta
    "dueling_ddqn": "#008300",  # green
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SMOOTHING_EPISODES = 10


def apply_style():
    # Figures are drawn at their printed size (about 6.5 in wide), so these point
    # sizes are what appears on the page of the report.
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlesize": 8.5,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "lines.linewidth": 1.3,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def moving_average(values, window):
    """Trailing moving average that keeps the series length (shorter window at the start)."""
    values = np.asarray(values, dtype=float)
    sums = np.cumsum(np.insert(values, 0, 0.0))
    end = np.arange(1, len(values) + 1)
    start = np.maximum(end - window, 0)
    return (sums[end] - sums[start]) / (end - start)


def save(fig, name):
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    path = os.path.join(config.PLOTS_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")
    return path


# ------------------------------------------------------------ training curves
def load_training_logs():
    logs = defaultdict(list)  # agent -> one dict of arrays per seed
    for path in sorted(glob.glob(os.path.join(config.LOGS_DIR, "*_seed*.csv"))):
        name = os.path.basename(path).rsplit("_seed", 1)[0]
        rows = read_csv(path)
        if name in AGENT_COLORS and rows:
            logs[name].append({key: np.array([float(r[key]) for r in rows])
                               for key in ("episode", "return", "avg_wait")})
    return logs


def plot_training_curves():
    logs = load_training_logs()
    if not logs:
        print("  no training logs found - skipping training curves")
        return None
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))
    panels = [
        ("return", "Episode return (higher is better)", "(a) Reward per training episode"),
        ("avg_wait", "Average wait per vehicle (s)", "(b) Waiting time during training"),
    ]
    for ax, (key, ylabel, title) in zip(axes, panels):
        for name in config.ALL_AGENTS:
            if name not in logs:
                continue
            length = min(len(run["episode"]) for run in logs[name])
            curves = np.array([moving_average(run[key][:length], SMOOTHING_EPISODES)
                               for run in logs[name]])
            mean, std = curves.mean(axis=0), curves.std(axis=0)
            episodes = logs[name][0]["episode"][:length]
            color = AGENT_COLORS[name]
            ax.plot(episodes, mean, color=color, label=config.AGENT_LABELS[name])
            if len(curves) > 1:
                ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.12, lw=0)
        ax.set_title(title)
        ax.set_xlabel("Training episode")
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=1)
    axes[1].set_yscale("log")  # early random policies wait many times longer
    axes[1].set_yticks([5, 10, 20, 40, 80])
    axes[1].yaxis.set_major_formatter(ScalarFormatter())
    axes[1].yaxis.set_minor_formatter(NullFormatter())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0),
               handlelength=1.5, columnspacing=1.2)
    fig.text(0.99, -0.01, f"Lines: mean over seeds, {SMOOTHING_EPISODES}-episode moving average; "
             "bands: +-1 std across seeds.", ha="right", va="top", fontsize=6.5, color=MUTED)
    return save(fig, "training_curves.png")


# -------------------------------------------------------------- evaluation bars
def plot_eval_bars():
    path = os.path.join(config.RESULTS_DIR, "summary.csv")
    if not os.path.exists(path):
        print("  no summary.csv found - skipping evaluation bars")
        return None
    summary = read_csv(path)
    scenarios = [s for s in config.EVAL_SCENARIOS + ["random"]
                 if any(row["scenario"] == s for row in summary)]
    # One panel per scenario with its own y-scale (small multiples), so a scenario
    # with long waits does not flatten the others.
    fig, axes = plt.subplots(1, len(scenarios), figsize=(2.2 * len(scenarios), 2.5), squeeze=False)
    for i, (ax, scenario) in enumerate(zip(axes[0], scenarios)):
        entries = {row["agent"]: row for row in summary if row["scenario"] == scenario}
        agents = [a for a in config.ALL_AGENTS if a in entries]
        means = np.array([float(entries[a]["avg_wait_mean"]) for a in agents])
        stds = np.array([float(entries[a]["avg_wait_std"]) for a in agents])
        x = np.arange(len(agents))
        ax.bar(x, means, width=0.55, color=[AGENT_COLORS[a] for a in agents],
               yerr=stds, error_kw={"ecolor": INK_SECONDARY, "elinewidth": 0.8, "capsize": 2})
        # Label selectively: the fixed-time reference and the best controller.
        labelled = {int(np.argmin(means))}
        if "fixed_time" in agents:
            labelled.add(agents.index("fixed_time"))
        for j in sorted(labelled):
            ax.annotate(f"{means[j]:.1f} s", (x[j], means[j] + stds[j]), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7,
                        color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([config.AGENT_LABELS[a] for a in agents], rotation=40, ha="right")
        ax.set_title(config.SCENARIO_LABELS[scenario])
        ax.set_ylim(0, (means + stds).max() * 1.18)
        if i == 0:
            ax.set_ylabel("Average wait per vehicle (s)")
    fig.tight_layout()
    return save(fig, "eval_avg_wait.png")


# ----------------------------------------------------------- queue timeline
def best_learning_agent(scenario):
    path = os.path.join(config.RESULTS_DIR, "summary.csv")
    if not os.path.exists(path):
        return None
    rows = [r for r in read_csv(path)
            if r["scenario"] == scenario and r["agent"] in config.DQN_VARIANTS]
    if not rows:
        return None
    return min(rows, key=lambda r: float(r["avg_wait_mean"]))["agent"]


def plot_timeline(scenario="rush_hour"):
    path = os.path.join(config.TRACES_DIR, f"{scenario}.csv")
    if not os.path.exists(path):
        print(f"  no trace for {scenario} - skipping timeline")
        return None
    by_agent = defaultdict(list)
    for row in read_csv(path):
        by_agent[row["agent"]].append(row)
    wanted = ["fixed_time", "actuated", "qlearning", best_learning_agent(scenario)]
    agents = [a for a in config.ALL_AGENTS if a in wanted and a in by_agent]
    if not agents:
        return None

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.5, 2.9), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 2]})
    reference = by_agent[agents[0]]
    minutes = np.array([float(r["time"]) for r in reference]) / 60
    peak_demand = 0.0
    for key, label, color in (("demand_ns", "North-South", INK), ("demand_ew", "East-West", MUTED)):
        demand = np.array([float(r[key]) for r in reference])
        top.plot(minutes, demand, color=color)
        peak = int(np.argmax(demand))
        peak_demand = max(peak_demand, demand[peak])
        top.annotate(label, (minutes[peak], demand[peak]), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=7,
                     color=INK_SECONDARY)
    top.set_ylim(0, peak_demand * 1.35)  # headroom for the labels
    top.set_ylabel("Arrivals\n(veh/min)")
    top.set_title(f"{config.SCENARIO_LABELS[scenario]} scenario: traffic demand (top) "
                  "and vehicles waiting (bottom)")

    window = max(1, 60 // config.ENV_CONFIG["decision_interval"])  # 1-minute average
    highest = 0.0
    for name in agents:
        rows = by_agent[name]
        queue = moving_average([float(r["queue_ns"]) + float(r["queue_ew"]) for r in rows], window)
        t = np.array([float(r["time"]) for r in rows]) / 60
        bottom.plot(t, queue, color=AGENT_COLORS[name], label=config.AGENT_LABELS[name])
        highest = max(highest, queue.max())
    bottom.set_ylabel("Vehicles waiting\n(1-min average)")
    bottom.set_xlabel("Time (min)")
    bottom.set_ylim(0, highest * 1.3)  # headroom so the legend never covers a line
    bottom.set_xlim(0, minutes[-1])
    bottom.legend(loc="upper left", ncol=len(agents))
    fig.tight_layout()
    return save(fig, f"{scenario}_queues.png")


def make_all():
    apply_style()
    print("Drawing figures:")
    plot_training_curves()
    plot_eval_bars()
    plot_timeline("rush_hour")


if __name__ == "__main__":
    make_all()

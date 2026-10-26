"""Draw the system-architecture and workflow diagrams for the report.

    python diagrams.py   ->  results/plots/architecture.png, results/plots/workflow.png

Figures are drawn at their printed size (6.3 in wide) so text stays legible in the report.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import config
from plots import AXIS, INK, INK_SECONDARY, SURFACE, apply_style, save

BOX_FILL = "#f0efec"
ACCENT_FILL = "#cde2fb"  # highlights the learning component
INCH_PER_UNIT = 0.525     # canvas units -> inches (a 12-unit canvas prints 6.3 in wide)
TITLE_PT, TEXT_PT, LABEL_PT, CODE_PT = 8.5, 7.5, 7.5, 6.5


def box(ax, x, y, w, h, title, lines, fill=BOX_FILL, subtitle=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=fill, edgecolor=AXIS, linewidth=0.8))
    ax.text(x + 0.18, y + h - 0.2, title, ha="left", va="top", fontsize=TITLE_PT,
            fontweight="semibold", color=INK)
    if subtitle:
        ax.text(x + w - 0.18, y + h - 0.24, subtitle, ha="right", va="top", fontsize=CODE_PT,
                color=INK_SECONDARY, family="monospace")
    for i, line in enumerate(lines):
        ax.text(x + 0.18, y + h - 0.66 - i * 0.36, line, ha="left", va="top",
                fontsize=TEXT_PT, color=INK_SECONDARY)


def arrow(ax, start, end, label=None, label_pos=None, rad=0.0, ha="center"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                 color=INK_SECONDARY, linewidth=1.0,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        x, y = label_pos or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.12)
        ax.text(x, y, label, ha=ha, va="bottom", fontsize=LABEL_PT, color=INK,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0})


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width * INCH_PER_UNIT, height * INCH_PER_UNIT))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def draw_architecture():
    fig, ax = canvas(12, 6.3)
    box(ax, 0.1, 3.5, 3.9, 2.7, "Traffic simulator", [
        "4-way junction, 2 phases",
        "Poisson vehicle arrivals",
        "0.5 veh/s per green lane",
        "3 s yellow, 60 s max green",
        "Decision every 5 s",
    ], subtitle="traffic_env.py")
    box(ax, 8.0, 3.5, 3.9, 2.7, "Online Q-network", [
        "7 inputs, 128-128 ReLU",
        "Q(s, a) for the 2 phases",
        "Dueling: Q = V + A - mean(A)",
        "ε-greedy, ε: 1.0 → 0.05",
    ], fill=ACCENT_FILL, subtitle="agents.py")
    box(ax, 0.1, 0.1, 3.4, 1.75, "Replay buffer", [
        "Last 50,000 transitions",
        "(s, a, r, s') sampled at random",
    ])
    box(ax, 4.3, 0.1, 3.4, 1.75, "TD target and loss", [
        "y = r + γ·Q'(s', a*)",
        "a* = argmax Q(s', ·)  (Double)",
        "Huber loss, Adam optimiser",
    ])
    box(ax, 8.5, 0.1, 3.4, 1.75, "Target network Q'", [
        "Frozen copy of the Q-network",
        "Gives Q'(s', a*) for the target",
    ])

    arrow(ax, (4.0, 5.3), (8.0, 5.3), "state s(t), reward r(t)")
    arrow(ax, (8.0, 4.25), (4.0, 4.25), "action a(t): phase for the next 5 s")
    arrow(ax, (1.8, 3.5), (1.8, 1.85), "store (s, a, r, s')", label_pos=(1.92, 2.5), ha="left")
    arrow(ax, (3.5, 0.95), (4.3, 0.95), "batch\nof 64", label_pos=(3.9, 1.05))
    arrow(ax, (10.2, 3.5), (10.2, 1.85), "copy every\n1,000 steps", label_pos=(10.32, 2.3), ha="left")
    arrow(ax, (8.5, 0.95), (7.7, 0.95))
    arrow(ax, (6.0, 1.85), (8.0, 3.62), "gradient step", label_pos=(6.35, 2.75), rad=-0.15)
    return save(fig, "architecture.png")


def draw_workflow():
    fig, ax = canvas(12, 2.3)
    steps = [
        ("1  Model", ["Intersection MDP", "6 controllers"]),
        ("2  Train", ["Random demand", "150 episodes x 3 seeds"]),
        ("3  Evaluate", ["3 unseen scenarios", "5 episodes, greedy policy"]),
        ("4  Report", ["Tables, figures", "live pygame demo"]),
    ]
    width, gap = 2.55, 0.5
    for i, (title, lines) in enumerate(steps):
        x = 0.1 + i * (width + gap)
        box(ax, x, 0.3, width, 1.7, title, lines, fill=ACCENT_FILL if i == 1 else BOX_FILL)
        if i < len(steps) - 1:
            arrow(ax, (x + width, 1.15), (x + width + gap, 1.15))
    return save(fig, "workflow.png")


if __name__ == "__main__":
    apply_style()
    print("Drawing diagrams:")
    draw_architecture()
    draw_workflow()
    print(f"Done - see {config.PLOTS_DIR}")

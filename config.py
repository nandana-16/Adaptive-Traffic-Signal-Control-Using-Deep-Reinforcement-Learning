"""Central configuration for the traffic-signal RL project.

Every hyperparameter reported in the paper lives here, so experiments are
reproducible and every number can be traced back to one place.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
TRACES_DIR = os.path.join(RESULTS_DIR, "traces")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
SCREENSHOTS_DIR = os.path.join(RESULTS_DIR, "screenshots")

# ----------------------------------------------------------------- environment
ENV_CONFIG = {
    "episode_length": 1800,  # simulated seconds per episode (30 minutes)
    "decision_interval": 5,  # agent chooses a phase every 5 s -> 360 decisions/episode
    "yellow_time": 3,        # seconds lost (nobody moves) whenever the phase changes
    "saturation_flow": 0.5,  # vehicles/second leaving one green lane (= 1800 veh/h)
    "max_green": 60,         # safety rule: a phase is forced to change after 60 s
}

TRAIN_SCENARIO = "random"  # fresh random demand pattern every training episode
EVAL_SCENARIOS = ["uniform", "asymmetric", "rush_hour"]  # fixed, unseen test patterns

# ------------------------------------------------------------------ experiment
SEEDS = [0, 1, 2]           # independent training runs per learning agent
TRAIN_EPISODES = 150
EVAL_EPISODES = 5           # evaluation episodes per scenario (per seed)
EVAL_SEED_OFFSET = 10_000   # evaluation traffic never overlaps training traffic

# ---------------------------------------------------------------------- agents
FIXED_TIME = {"green_time": 30}  # seconds per phase (incl. yellow) -> 60 s cycle
ACTUATED = {"min_green": 10}     # minimum seconds before the actuated signal may switch

QLEARNING = {
    "alpha": 0.1,
    "gamma": 0.95,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_frac": 0.6,  # fraction of training over which epsilon is annealed
}

DQN = {
    "hidden": 128,
    "lr": 5e-4,
    "gamma": 0.95,
    "batch_size": 64,
    "buffer_size": 50_000,
    "learning_starts": 1_000,
    "train_freq": 1,
    "target_update": 1_000,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_frac": 0.5,
    "grad_clip": 10.0,
}

# The three DQN-family agents differ only in these two switches.
DQN_VARIANTS = {
    "dqn": {"double": False, "dueling": False},
    "double_dqn": {"double": True, "dueling": False},
    "dueling_ddqn": {"double": True, "dueling": True},
}

BASELINE_AGENTS = ["fixed_time", "actuated"]
LEARNING_AGENTS = ["qlearning", "dqn", "double_dqn", "dueling_ddqn"]
ALL_AGENTS = BASELINE_AGENTS + LEARNING_AGENTS

AGENT_LABELS = {
    "fixed_time": "Fixed-time",
    "actuated": "Actuated",
    "qlearning": "Q-learning",
    "dqn": "DQN",
    "double_dqn": "Double DQN",
    "dueling_ddqn": "Dueling DDQN",
}

SCENARIO_LABELS = {
    "uniform": "Uniform",
    "asymmetric": "Asymmetric",
    "rush_hour": "Rush hour",
    "random": "Random",
}

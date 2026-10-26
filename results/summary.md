# Evaluation results

Mean +- standard deviation over evaluation episodes (and over training seeds for learning agents).

## Average waiting time per vehicle (s) - lower is better

| Controller | Uniform | Asymmetric | Rush hour |
|---|---|---|---|
| Fixed-time | 12.4 +- 0.4 | 29.0 +- 3.5 | 18.9 +- 3.1 |
| Actuated | 7.8 +- 0.4 | 8.7 +- 0.7 | 6.7 +- 0.4 |
| Q-learning | 9.7 +- 1.2 | 11.9 +- 2.3 | 10.8 +- 4.3 |
| DQN | 7.6 +- 0.3 | 7.8 +- 0.6 | 6.5 +- 0.7 |
| Double DQN | 7.5 +- 0.3 | 7.5 +- 0.4 | 6.5 +- 0.6 |
| Dueling DDQN | 7.4 +- 0.2 | 7.4 +- 0.4 | 6.2 +- 0.4 |

## Average number of waiting vehicles - lower is better

| Controller | Uniform | Asymmetric | Rush hour |
|---|---|---|---|
| Fixed-time | 5.0 +- 0.3 | 15.3 +- 2.0 | 6.2 +- 1.3 |
| Actuated | 3.1 +- 0.2 | 4.6 +- 0.4 | 2.2 +- 0.1 |
| Q-learning | 3.9 +- 0.5 | 6.3 +- 1.2 | 3.5 +- 1.4 |
| DQN | 3.0 +- 0.2 | 4.1 +- 0.4 | 2.1 +- 0.3 |
| Double DQN | 3.0 +- 0.2 | 3.9 +- 0.3 | 2.1 +- 0.2 |
| Dueling DDQN | 3.0 +- 0.1 | 3.9 +- 0.3 | 2.0 +- 0.2 |

## Vehicles served in 30 minutes - higher is better

| Controller | Uniform | Asymmetric | Rush hour |
|---|---|---|---|
| Fixed-time | 713 +- 29 | 932 +- 24 | 582 +- 28 |
| Actuated | 717 +- 30 | 943 +- 26 | 585 +- 29 |
| Q-learning | 716 +- 30 | 942 +- 26 | 584 +- 29 |
| DQN | 717 +- 30 | 942 +- 26 | 585 +- 29 |
| Double DQN | 717 +- 30 | 943 +- 26 | 585 +- 29 |
| Dueling DDQN | 717 +- 30 | 943 +- 26 | 585 +- 29 |

## Phase switches per episode

| Controller | Uniform | Asymmetric | Rush hour |
|---|---|---|---|
| Fixed-time | 59 +- 0 | 59 +- 0 | 59 +- 0 |
| Actuated | 154 +- 4 | 127 +- 4 | 117 +- 6 |
| Q-learning | 171 +- 30 | 141 +- 15 | 151 +- 34 |
| DQN | 155 +- 14 | 120 +- 16 | 130 +- 23 |
| Double DQN | 148 +- 13 | 118 +- 10 | 120 +- 9 |
| Dueling DDQN | 170 +- 18 | 121 +- 16 | 177 +- 44 |

## Switches forced by the 60 s max-green rule (included above)

| Controller | Uniform | Asymmetric | Rush hour |
|---|---|---|---|
| Fixed-time | 0.0 +- 0.0 | 0.0 +- 0.0 | 0.0 +- 0.0 |
| Actuated | 0.0 +- 0.0 | 0.2 +- 0.4 | 0.2 +- 0.4 |
| Q-learning | 0.2 +- 0.4 | 1.4 +- 2.2 | 2.3 +- 2.2 |
| DQN | 0.0 +- 0.0 | 0.1 +- 0.3 | 0.3 +- 0.6 |
| Double DQN | 0.0 +- 0.0 | 0.3 +- 0.8 | 1.1 +- 1.5 |
| Dueling DDQN | 0.0 +- 0.0 | 0.8 +- 1.3 | 2.1 +- 2.4 |

## Reduction in average waiting time vs. fixed-time signal (%)

| Controller | Uniform | Asymmetric | Rush hour |
|---|---|---|---|
| Fixed-time | 0.0 | 0.0 | 0.0 |
| Actuated | 37.2 | 69.9 | 64.5 |
| Q-learning | 21.9 | 59.0 | 42.8 |
| DQN | 38.6 | 73.2 | 65.5 |
| Double DQN | 39.3 | 74.3 | 65.5 |
| Dueling DDQN | 40.2 | 74.4 | 67.3 |

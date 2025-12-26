# Autonomous Highway Navigation via Deep Q-Learning

This project explores autonomous highway driving using Deep Q-Learning (DQN) in a stochastic multi-lane traffic environment. The focus is on understanding how reward design and algorithmic choices influence learned driving behavior, rather than maximizing reward alone.

The project is intentionally **work-in-progress** and structured to support systematic experimentation and ablation studies.

---

## Objective

Train a reinforcement learning agent that can:

- Avoid collisions (safety)
- Maintain high but controlled speed (efficiency)
- Minimize unnecessary lane changes (driving convention)

The agent must balance these competing objectives under stochastic traffic conditions.

---

## Environment

- **Simulator:** `highway-env` (`highway-fast-v0`)
- **Framework:** Gymnasium
- **Traffic Model:** IDM-based surrounding vehicles
- **Lanes:** 3
- **Episode Duration:** 30 seconds

### Observation Space

Kinematic observations of nearby vehicles (relative position and velocity).

### Action Space

Discrete meta-actions:

- Lane left / lane right
- Accelerate / decelerate
- Maintain speed

---

## Algorithm & Stack

- **Algorithm:** Deep Q-Network (DQN)
- **Framework:** Stable Baselines3 (PyTorch)
- **Policy Network:** MLP with two hidden layers (256 units each)
- **Replay Buffer:** 15,000 transitions
- **Optimizer:** Adam
- **Exploration:** ε-greedy decay
- **Target Network:** Periodic updates

---

## Reward Design (Current)

The reward function is designed to discourage exploitation and promote stable behavior.

```python
HIGH_SPEED_LANE_CONFIG = {
    "collision_reward": -2.0,
    "high_speed_reward": 0.1,
    "lane_change_reward": -0.05,
    "right_lane_reward": 0.0,
    "reward_speed_range": [25, 30],
    "normalize_reward": False
}
```

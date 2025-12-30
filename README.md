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

### Experiments

The project follows a controlled experimental progression to evaluate learning stability and behavior emergence in autonomous highway navigation:

1. **Baseline DQN Training**  
   A Double DQN agent was trained to verify basic learning capability in the highway environment.

2. **Reward Shaping Ablations**  
   The right-lane reward was removed to isolate survival and speed incentives. Speed reward scaling and lane-change penalties were introduced to discourage unnecessary weaving while preserving efficiency.

3. **Reward Normalization Study**  
   Training runs with and without reward normalization demonstrated that normalization is critical for stable DQN learning in this environment.

4. **Optimization Stability Analysis**  
   The DQN learning rate was reduced to mitigate late-stage instability. While smoother convergence was observed, significant policy degradation persisted, indicating algorithmic rather than hyperparameter-level limitations.

5. **Algorithm Comparison (PPO)**  
   PPO is introduced as an on-policy baseline to evaluate whether replay-free learning improves stability and long-horizon performance under identical task settings.

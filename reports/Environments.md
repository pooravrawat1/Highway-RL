## Environment Overview

- Environment: highway-fast-v0
- Simulator: highway-env (Gymnasium)
- Task: Multi-lane autonomous highway navigation with stochastic traffic

## Observation Space

- Type: Kinematic state representation
- Format: V × F matrix (vehicles × features)
- Features include normalized longitudinal position, lateral position (lane),
  longitudinal velocity, and lateral velocity for the ego vehicle and nearby vehicles.

## Action Space

- Type: Discrete meta-actions (5)
- Actions: Lane Left, Idle, Lane Right, Faster, Slower
- Actions represent high-level driving intentions rather than low-level control.

## Environment Dynamics

- Lanes: 3
- Vehicles: 20 (IDM-based stochastic drivers)
- Policy frequency: 1 Hz
- Simulation frequency: 5 Hz
- Episode duration: 30 seconds

## Reward Function

The reward at each timestep is a weighted combination of:

- Collision penalty (-1, terminal)
- Speed reward encouraging velocities in the 20–30 m/s range
- Right-lane reward incentivizing conventional driving behavior

Rewards are normalized to improve learning stability.

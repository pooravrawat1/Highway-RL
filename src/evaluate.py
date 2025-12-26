import gymnasium as gym
from stable_baselines3 import DQN
from highway_env import register_highway_envs
from config import HIGH_SPEED_LANE_CONFIG

register_highway_envs()

env = gym.make("highway-fast-v0", render_mode="human")
env.unwrapped.configure(HIGH_SPEED_LANE_CONFIG)

model = DQN.load("agents/dqn_high_speed_3")

obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
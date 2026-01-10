import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from config import HIGH_SPEED_LANE_CONFIG

env = gym.make("highway-fast-v0", render_mode="human")
env.unwrapped.configure(HIGH_SPEED_LANE_CONFIG)

model = PPO.load("agents/ppo_smart_lane_v1")

obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
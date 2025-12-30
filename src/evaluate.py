import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from config import OBS_3_VEHICLES_CONFIG

env = gym.make("highway-fast-v0", render_mode="human")
env.unwrapped.configure(OBS_3_VEHICLES_CONFIG)

model = DQN.load("agents/dqn_obs_3_vehicles")

obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
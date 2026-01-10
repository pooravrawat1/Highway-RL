import os
import gymnasium as gym
import highway_env
from config import HIGH_SPEED_LANE_CONFIG

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


ENV_ID = "highway-fast-v0"
SEED = 42
TOTAL_TIMESTEPS = 50000

LOG_DIR = "logs/ppo_smart_lane_v1/"
MODEL_DIR = "agents/"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def make_env():
    env = gym.make(ENV_ID)
    env.unwrapped.configure(HIGH_SPEED_LANE_CONFIG)
    env = Monitor(env, LOG_DIR)
    env.reset(seed=SEED)
    return env

set_random_seed(SEED)
env = make_env()

model = PPO(
    policy="MlpPolicy",
    env=env,

    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,

    clip_range=0.2,
    ent_coef=0.0,

    tensorboard_log=LOG_DIR,
    verbose=1,
    seed=SEED,
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    tb_log_name="ppo_smart_lane_v1"
)

model_path = os.path.join(MODEL_DIR, "ppo_smart_lane_v1")
model.save(model_path)
print(f"Model saved to {model_path}")

print("Running evaluation rollout...")

eval_env = gym.make(ENV_ID, render_mode="human")
eval_env.unwrapped.configure(HIGH_SPEED_LANE_CONFIG)

obs, _ = eval_env.reset(seed=SEED)

for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)

    if terminated or truncated:
        obs, _ = eval_env.reset()

eval_env.close()
print("Eval done.")

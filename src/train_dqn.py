import os
import gymnasium as gym
import highway_env
from config import HIGH_SPEED_LANE_CONFIG

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

ENV_ID = "highway-fast-v0"
SEED = 42

TOTAL_TIMESTEPS = 50000

LOG_DIR = "logs/high_speed_lane/"
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

model = DQN(
    policy="MlpPolicy",
    env=env,

    learning_rate=1e-3,
    batch_size=64,
    gamma=0.99,

    buffer_size=15_000,
    learning_starts=1_000,

    tau=1.0,                     
    target_update_interval=500,

    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,

    train_freq=1,
    gradient_steps=1,

    tensorboard_log="logs/high_speed_lane",
    verbose=1,

    seed=SEED
)


model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="dqn_high_speed")

model_path = os.path.join(MODEL_DIR, "dqn_high_speed")
model.save(model_path)
print(f"Model saved to {model_path}")

print("Quick evaluation rollout...")

eval_env = gym.make(ENV_ID, render_mode="human")
obs, _ = eval_env.reset(seed=SEED)

for step in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)

    if terminated or truncated:
        obs, _ = eval_env.reset()

eval_env.close()
print("Eval done")

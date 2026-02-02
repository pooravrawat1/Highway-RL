import gymnasium as gym
import highway_env
import os
import sys
import glob
from stable_baselines3 import PPO, DQN
from config import HIGH_SPEED_LANE_CONFIG

env = gym.make("highway-fast-v0", render_mode="human")
env.unwrapped.configure(HIGH_SPEED_LANE_CONFIG)

# Check if a specific model was provided as argument
if len(sys.argv) > 1:
    model_name = sys.argv[1]
    model_path = f"agents/{model_name}"
    
    # Check if the file exists (with or without .zip)
    if not os.path.exists(f"{model_path}.zip") and not os.path.exists(model_path):
        print(f"Error: Model '{model_name}' not found in agents/ folder")
        print("\nAvailable models:")
        for f in sorted(glob.glob("agents/*.zip")):
            print(f"  - {os.path.splitext(os.path.basename(f))[0]}")
        sys.exit(1)
    
    print(f"Loading specified model: {model_name}")
else:
    # Find the most recent model file
    model_files = glob.glob("agents/*.zip")
    if not model_files:
        raise FileNotFoundError("No model files found in agents/ folder")
    
    latest_model = max(model_files, key=os.path.getmtime)
    model_name = os.path.splitext(os.path.basename(latest_model))[0]
    model_path = latest_model.replace(".zip", "")
    
    print(f"Loading latest model: {model_name}")

# Load the appropriate algorithm
if model_name.startswith("ppo"):
    model = PPO.load(model_path if 'model_path' in locals() else f"agents/{model_name}")
elif model_name.startswith("dqn"):
    model = DQN.load(model_path if 'model_path' in locals() else f"agents/{model_name}")
else:
    raise ValueError(f"Unknown model type: {model_name}")

obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
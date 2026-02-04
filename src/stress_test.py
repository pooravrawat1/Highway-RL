import gymnasium as gym
import highway_env
import numpy as np
import os
import glob
import sys
from stable_baselines3 import PPO, DQN
from config import HIGH_SPEED_LANE_CONFIG

def load_best_model(agents_dir="agents/"):
    # Find all zip files
    model_files = glob.glob(os.path.join(agents_dir, "*.zip"))
    if not model_files:
        print("No models found!")
        sys.exit(1)
        
    # Get the most recently modified file
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"Loading Best Model: {latest_model}")
    
    # Try loading as PPO, then DQN
    try:
        return PPO.load(latest_model)
    except:
        return DQN.load(latest_model)

def run_test(model, config, scenario_name, episodes=10):
    print(f"\n=============================================")
    print(f"TESTING SCENARIO: {scenario_name}")
    print(f"Traffic Density: {config['vehicles_count']} vehicles")
    print(f"=============================================")
    
    # Create environment
    env = gym.make("highway-fast-v0", render_mode=None)
    env.unwrapped.configure(config)
    
    success_count = 0
    total_rewards = []
    
    for i in range(episodes):
        # Use different seeds for each episode to test generalization
        seed = 1000 + i
        obs, _ = env.reset(seed=seed)
        
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        while not (done or truncated):
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
        total_rewards.append(episode_reward)
        
        # Check if we crashed or timed out successfully
        # In highway-env, 'crashed' is usually in info, or we check if we survived long enough
        is_crashed = info.get("crashed", False)
        
        if not is_crashed:
            success_count += 1
            status = "SUCCESS"
        else:
            status = "CRASH  "
            
        print(f"Episode {i+1}: {status} | Reward: {episode_reward:.1f} | Steps: {steps}")

    avg_reward = np.mean(total_rewards)
    success_rate = (success_count / episodes) * 100
    
    print(f"\n>>> RESULT {scenario_name}:")
    print(f"    Avg Reward: {avg_reward:.2f}")
    print(f"    Success Rate: {success_rate:.1f}%")
    return success_rate

if __name__ == "__main__":
    model = load_best_model()
    
    # 1. Standard Condition (Control Group)
    base_config = HIGH_SPEED_LANE_CONFIG.copy()
    run_test(model, base_config, "Standard Traffic")

    # 2. High Density (Stress Test)
    # Increase vehicles from 30 to 60 to simulate heavy traffic
    dense_config = HIGH_SPEED_LANE_CONFIG.copy()
    dense_config["vehicles_count"] = 60 
    run_test(model, dense_config, "HEAVY Traffic (2x Density)")
    
    # 3. Low density (Speed Test)
    # Fewer cars, should have near 100% success and high speed
    light_config = HIGH_SPEED_LANE_CONFIG.copy()
    light_config["vehicles_count"] = 10
    run_test(model, light_config, "Light Traffic (Speed Run)")

import os
import numpy as np
from stable_baselines3 import PPO
from custom_env import CustomMetaDriveEnv

def evaluate_model(agent_name, model_path, stage, env_map, traffic_density, episodes=100):
    if not os.path.exists(model_path + ".zip"):
        print(f"Skipping {agent_name} Stage {stage}: Could not find {model_path}.zip")
        return

    print(f"Testing {agent_name} Stage {stage} ({episodes} episodes)...")

    # Set up the environment parameters for the specific stage
    env_config = dict(
        use_render=False,  # Run headless for maximum speed
        traffic_density=traffic_density,
        map=env_map,
        random_traffic=(stage == 3),
        target_speed=30.0 if stage == 1 else 15.0
    )
    
    env = CustomMetaDriveEnv(env_config)
    model = PPO.load(model_path, env=env)

    successes = 0
    ep_lengths = []
    ep_rewards = []

    for ep in range(episodes):
        obs = env.reset()
        # Handle Gym API differences (some return tuple, some return array)
        if isinstance(obs, tuple):
            obs = obs[0]
            
        done = False
        total_reward = 0.0
        length = 0

        while not done:
            # deterministic=True forces the AI to take the BEST action, stopping random exploration
            action, _ = model.predict(obs, deterministic=True)
            
            step_result = env.step(action)
            # Handle Gym 0.21 vs 0.26 API
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

            total_reward += reward
            length += 1

            if done:
                # MetaDrive triggers 'arrive_dest' if the car finishes without crashing
                if info.get("arrive_dest", False):
                    successes += 1
                ep_lengths.append(length)
                ep_rewards.append(total_reward)

    env.close()

    # Calculate metrics
    success_rate = (successes / episodes) * 100
    mean_length = np.mean(ep_lengths)
    mean_reward = np.mean(ep_rewards)

    print(f"--- {agent_name} (Stage {stage}) ---")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Mean Length:  {mean_length:.1f}")
    print(f"Mean Reward:  {mean_reward:.2f}")
    print("-" * 35 + "\n")

if __name__ == "__main__":
    print("Starting Automated Table Generation for DEFAULT REWARD models...\n")

    # 1. Stage 1 (Straight Road)
    evaluate_model("Baseline (Default)", "../models/default_reward/ppo_baseline_stage1", 1, "S", 0.0)
    evaluate_model("Hybrid (Default)", "../models/default_reward/ppo_hybrid_stage1", 1, "S", 0.0)

    # 2. Stage 2 (Curve)
    evaluate_model("Baseline (Default)", "../models/default_reward/ppo_baseline_stage2", 2, "C", 0.0)
    evaluate_model("Hybrid (Default)", "../models/default_reward/ppo_hybrid_stage2", 2, "C", 0.0)

    # 3. Stage 3 (Traffic)
    evaluate_model("Baseline (Default)", "../models/default_reward/ppo_baseline_stage3", 3, "SCSCCSS", 0.05)
    evaluate_model("Hybrid (Default)", "../models/default_reward/ppo_hybrid_stage3", 3, "SCSCCSS", 0.05)
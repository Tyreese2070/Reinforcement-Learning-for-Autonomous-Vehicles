import os
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from custom_env import CustomMetaDriveEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Run Stage 2 or Stage 3 Curriculum Learning")
    parser.add_argument("--agent", type=str, choices=["baseline", "hybrid"], required=True, 
                        help="Which agent to train (baseline or hybrid)")
    parser.add_argument("--stage", type=int, choices=[2, 3], required=True, 
                        help="Curriculum stage: 2 (Curve) or 3 (Traffic)")
    return parser.parse_args()

def get_curriculum_config(agent_type, stage):
    """Set filepath and env based on given stage"""
    
    log_dir = f"../logs/{agent_type}"

    if stage == 2:
        env_map = "C"
        traffic_density = 0.0
        random_traffic = False
        
        # Stage 2 loads Stage 1 models
        if agent_type == "baseline":
            load_model_path = "../models/ppo_baseline"
            save_model_path = "../models/ppo_baseline_stage2"
        else:
            load_model_path = "../models/ppo_il_pretrain"
            save_model_path = "../models/ppo_hybrid_stage2"
            
    elif stage == 3:
        env_map = "SCSCCSS"
        traffic_density = 0.05
        random_traffic = True
        
        # Stage 3 loads Stage 2 models
        if agent_type == "baseline":
            load_model_path = "../models/ppo_baseline_stage2"
            save_model_path = "../models/ppo_baseline_stage3"
        else:
            load_model_path = "../models/ppo_hybrid_stage2"
            save_model_path = "../models/ppo_hybrid_stage3"

    return env_map, traffic_density, random_traffic, load_model_path, save_model_path, log_dir

def make_env(env_map, traffic_density, random_traffic, log_dir, rank, seed=42):
    def _init():
        env_config = dict(
            use_render=False,
            traffic_density=traffic_density,
            map=env_map,
            start_seed=seed + rank,
            random_traffic=random_traffic,
            target_speed=30.0,
        )
        env = CustomMetaDriveEnv(env_config)
        env = Monitor(env, log_dir)
        return env
    return _init

def train():
    args = parse_args()
    
    env_map, traffic_density, random_traffic, load_model_path, save_model_path, log_dir = get_curriculum_config(args.agent, args.stage)

    print(f"\n--- Training {args.agent.upper()} Agent | Stage {args.stage} ---")
    print(f"Map: {env_map} | Traffic: {traffic_density}")
    print(f"Loading from: {load_model_path}.zip")

    cpu_cores = os.cpu_count()
    env = SubprocVecEnv([make_env(env_map, traffic_density, random_traffic, log_dir, i) for i in range(cpu_cores)])

    try:
        model = PPO.load(load_model_path, env=env, tensorboard_log=log_dir)
    except FileNotFoundError:
        print(f"\nERROR: Could not find {load_model_path}.zip!")
        print(f"Make sure you have completely trained Stage {args.stage - 1} for this agent first.")
        exit()

    print(f"Training for 500,000 steps...")
    #model.learn(total_timesteps=500000, reset_num_timesteps=True) true uses a new graph for tensorboard
    model.learn(total_timesteps=500000, reset_num_timesteps=False, tb_log_name=f"stage{args.stage}")

    os.makedirs("../models", exist_ok=True)
    model.save(save_model_path)
    print("Model successfully saved to:", save_model_path)
    env.close()

if __name__ == "__main__":
    train()
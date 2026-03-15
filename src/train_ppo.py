import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from custom_env import CustomMetaDriveEnv

#TRAINING_STEPS = 100000
TRAINING_STEPS = 500000
MODEL_LOCATION = "../models/ppo_custom_reward"
LOG_DIR = "../logs/ppo_custom_reward"

def make_env(rank, seed=42):

    def _init():
        env_config = dict(
            use_render=False,
            traffic_density=0.0, # for the more challenging tasks, increase this
            map="S",
            start_seed=seed + rank,
            random_traffic=False,
            target_speed=30.0,
        )
        env = CustomMetaDriveEnv(env_config)
        return env
    return _init

def train_baselines():
    cpu_cores = os.cpu_count()
    env = SubprocVecEnv([make_env(i) for i in range(cpu_cores)])

    model = PPO("MlpPolicy", 
                env, 
                verbose=1, 
                tensorboard_log=LOG_DIR, 
                learning_rate=0.0003, 
                n_steps=1024,
                batch_size=256,
                ent_coef=0.01,)
    print("Training PPO model")
    model.learn(total_timesteps=TRAINING_STEPS)
    print("Trained Model Saved to:", MODEL_LOCATION)
    model.save(MODEL_LOCATION)

    if not os.path.exists("../models"):
        os.makedirs("../models")
    model.save(MODEL_LOCATION)
    print("Model saved to:", MODEL_LOCATION)
    env.close()

if __name__ == "__main__":
    train_baselines()
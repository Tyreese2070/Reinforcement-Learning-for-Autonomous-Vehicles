import os
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

TRAINING_STEPS = 100000
MODEL_LOCATION = "../models/ppo_baseline"
LOG_DIR = "../logs/ppo_baseline"

env_config = dict(
    use_render=False,
    traffic_density=0.1,
    map="S",
    start_seed=42,
    random_traffic=False,
)

def train_baselines():
    env = MetaDriveEnv(env_config)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0003, n_steps=2048)
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
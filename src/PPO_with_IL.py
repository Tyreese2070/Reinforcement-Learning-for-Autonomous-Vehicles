import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from custom_env import CustomMetaDriveEnv

#TRAINING_STEPS = 100000
TRAINING_STEPS = 500000
MODEL_LOCATION = "../models/ppo_il_pretrain"
LOG_DIR = "../logs/ppo_il_pretrain"

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
        env = Monitor(env, LOG_DIR)
        return env
    return _init

def train_baselines():
    cpu_cores = os.cpu_count()
    env = SubprocVecEnv([make_env(i) for i in range(cpu_cores)])

    # Match sb3 model to behavioural cloning model
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])
    )

    model = PPO("MlpPolicy", 
                env,
                policy_kwargs=policy_kwargs, 
                verbose=1, 
                tensorboard_log=LOG_DIR, 
                learning_rate=0.0003, 
                n_steps=1024,
                batch_size=256,
                ent_coef=0.01,)
    
    # load pretrained bc model weights into ppo model
    print("Loading BC weights into PPO model") 
    bc_weights = torch.load("../models/bc_model.pth", map_location="cpu", weights_only=True)

    with torch.no_grad():
        # Layer 1: 259 -> 256
        model.policy.mlp_extractor.policy_net[0].weight.copy_(bc_weights['network.0.weight'])
        model.policy.mlp_extractor.policy_net[0].bias.copy_(bc_weights['network.0.bias'])
        
        # Layer 2: 256 -> 128
        model.policy.mlp_extractor.policy_net[2].weight.copy_(bc_weights['network.2.weight'])
        model.policy.mlp_extractor.policy_net[2].bias.copy_(bc_weights['network.2.bias'])
        
        # Layer 3: 128 -> 64
        model.policy.mlp_extractor.policy_net[4].weight.copy_(bc_weights['network.4.weight'])
        model.policy.mlp_extractor.policy_net[4].bias.copy_(bc_weights['network.4.bias'])

        model.policy.action_net.weight.copy_(bc_weights['network.6.weight'])
        model.policy.action_net.bias.copy_(bc_weights['network.6.bias'])

    print("BC weights loaded")

    # Train the model
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
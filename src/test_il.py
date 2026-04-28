import torch
import torch.nn as nn
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv

class BehaviouralCloningModel(nn.Module):
    def __init__(self, input_dim=259, output_dim=2):
        super(BehaviouralCloningModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh() 
        )

    def forward(self, x):
        return self.network(x)

def test_autonomous_driving():
    """
    Test trained bc model
    """

    device = torch.device("cpu")
    model = BehaviouralCloningModel().to(device)
    
    print("Loading AI weights...")
    model.load_state_dict(torch.load("../models/bc_model.pth", map_location=device))
    
    model.eval() # disable training
    print("Brain loaded successfully!")

    config = {
        "map": "S",
        "use_render": True,
        "manual_control": False
    }
    env = MetaDriveEnv(config)

    try:
        obs, info = env.reset()
        print("\nStarting autonomous driving test. Watch the simulator window!")
        
        for i in range(1500):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_tensor = model(obs_tensor)
            
            action = action_tensor.squeeze(0).cpu().numpy()

            #if abs(action[0]) < 0.05:
            #    action[0] = 0.0 # Deadzone for steering test
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Episode finished at frame {i}. Resetting...")
                obs, info = env.reset()
                
    finally:
        env.close()

if __name__ == "__main__":
    test_autonomous_driving()
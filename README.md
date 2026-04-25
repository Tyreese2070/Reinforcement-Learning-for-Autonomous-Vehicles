# Setup
```
git clone https://github.com/Tyreese2070/Reinforcement-Learning-for-Autonomous-Vehicles.git

cd Reinforcement-Learning-for-Autonomous-Vehicles

python -m venv venv
.\venv\Scripts\activate

pip install -r requirements.txt

git clone https://github.com/metadriverse/metadrive.git
cd metadrive
pip install -e .
```

# Directory Structure

/data: Contains the human demonstrations intended for imitation learning

/logs: tensorboard logs for assessing the performance of models

/models: folder containing all saved models

/src scripts for training or testing models, recording driving data, and the custom environment

# Usage
custom_env.py: Modify the MetaDrive environment to change features such as the reward function

log_data.py: Run with a joystick controller connected, type a name for the csv file, press X (Xbox) Square (Dualshock / Dualsense) to save driving data

PPO_with_IL.py: For training the hybrid il + ppo model for stage 1.
test_il.py: Testing the pure il model

test_ppo.py: testing any PPO model

train_cl.py: CLI for training stage 2 and stage 3 models for either hybrid or pure PPO models

train_il.py: 

train_ppo.py: For training the baseline pure PPO model on stage 1.
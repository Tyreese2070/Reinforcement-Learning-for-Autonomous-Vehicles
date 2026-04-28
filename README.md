# Overview
This project creates a hybrid algorithm, combining PPO, behavioural cloning, and curriculum learning to improve sample efficiency found in standard reinforcement learning and to overcome the cold start from random exploration.

Created and tested using Python 3.11.9


# Repository Structure

```text

├── data/                   # Contains human driving demonstrations (.csv)

├── logs/                   # TensorBoard event files for continuous training monitoring

├── models/                 # Saved PyTorch (.pth) and SB3 (.zip) policy weights

│   ├── default/            # Contains the same models but using MetaDrive's default reward functions 

├── src/                    # Main source code directory

│   ├── custom_env.py       # Custom MetaDrive wrapper with dense reward function

│   ├── log_data.py         # PyGame script to record DualSense controller data

│   ├── train_il.py         # PyTorch Behavioural Cloning training script

│   ├── train_ppo.py        # Baseline PPO training script

│   ├── PPO_with_IL.py      # Architecture to inject IL weights into the PPO network

│   ├── train_cl.py         # Curriculum Learning stage progression logic

│   ├── test_ppo.py         # Visual rendering/evaluation of trained models

├── README.md

└── requirements.txt        # Python dependencies
```

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
MetaDrive Documentation: https://metadrive-simulator.readthedocs.io/en/latest/
